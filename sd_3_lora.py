"""
sd_3_lora.py
============================
LoRA fine-tuning of Stable Diffusion 3 (Medium) on MatCLIP microstructure dataset.
Uses HuggingFace diffusers + peft for efficient training.

Why this differs from the SD1.5 script
---------------------------------------
SD3 is a Multimodal Diffusion Transformer (MMDiT), not a UNet. It conditions on
THREE text encoders instead of one:
    - CLIP-L   (77 token limit, like SD1.5)
    - CLIP-G   (OpenCLIP bigG, 77 token limit)
    - T5-XXL   (this is the one that gives you longer sequence length)

The 77-token CLIP limit is architecturally fixed and cannot be extended. The T5
branch, however, supports much longer captions — SD3 was trained with T5 sequences
up to 256 tokens, and the encoder itself has no hard cap, so pushing beyond that
is supported by the code (just increasingly out-of-distribution the further you go).
That's the knob to use for "more sequence length": MAX_T5_SEQ_LEN below.

Also swapped:
    UNet2DConditionModel      -> SD3Transformer2DModel
    DDPMScheduler              -> FlowMatchEulerDiscreteScheduler (SD3's flow-matching sched.)
    CLIPTextModel (x1)         -> CLIPTextModelWithProjection (x2) + T5EncoderModel
    CLIPTokenizer (x1)         -> CLIPTokenizer (x2) + T5TokenizerFast

Usage:
    python sd_3_lora.py

SLURM:
    sbatch train_lora_sd3.slurm

Outputs:
    checkpoints_diffusion_sd3_lora/          — LoRA weights, saved best + periodic epochs
    checkpoints_diffusion_sd3_lora/final/    — final LoRA adapter, ready for inference
"""

import os
import math
import random
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    T5TokenizerFast,
    T5EncoderModel,
)
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
)
from peft import LoraConfig, get_peft_model, PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SD3_PATH        = "/home/cdacapp01/wk-ganesh/multi-modal/models/stable-diffusion-3-medium"
BASE_DIR        = "/home/cdacapp01/wk-ganesh/multi-modal"
CSV_PATH        = os.path.join(BASE_DIR, "matclip_phase1_cluster_caption_filtered.csv")
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints_diffusion_sd3_lora")

# Training data
split           = {"train"}
MIN_IMAGE_SIZE  = 0

# Sequence length — this is the actual "more sequence length" knob.
# CLIP-L / CLIP-G are hard-capped at 77 tokens (architectural, not adjustable).
# T5-XXL is the long-context branch: SD3 was trained with up to 256 T5 tokens.
# Bumping this higher (e.g. 384/512) still runs, but goes beyond what SD3 saw
# in training, so quality on the "extra" tail of very long captions may degrade
# unless you fine-tune specifically with those longer captions (which this
# script does, since LoRA training here backprops through the T5 branch's
# projection/cross-attn usage in the transformer).
MAX_T5_SEQ_LEN  = 256            # bump to 384 / 512 for longer captions
CLIP_SEQ_LEN    = 77             # fixed, do not change

# LoRA config
LORA_RANK       = 16             # higher = more capacity, more VRAM
LORA_ALPHA      = 32             # scaling factor (usually 2x rank)
LORA_DROPOUT    = 0.1

# Training hyperparams
IMAGE_SIZE      = 1024           # SD3 native resolution (also supports 512)
BATCH_SIZE      = 8              # SD3 transformer is heavier than SD1.5 UNet — start conservative
GRAD_ACCUM      = 4              # effective batch = 8 × 4 = 32
EPOCHS          = 100
LR              = 1e-4           # standard for LoRA
LR_WARMUP_STEPS = 200
MAX_GRAD_NORM   = 1.0
MIXED_PRECISION = True           # bfloat16 for A100

# Resume / seed
RESUME_FROM     = None           # path to a saved adapter dir to resume from
SEED            = 42

os.makedirs(CKPT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class MicrostructureDataset(Dataset):
    """
    Same filtering logic as the SD1.5 version, but tokenizes each caption with
    THREE tokenizers (CLIP-L, CLIP-G, T5) since SD3 conditions on all three.
    """
    def __init__(self, csv_path, tokenizer_clip_l, tokenizer_clip_g, tokenizer_t5,
                 image_size=1024, split=None, min_size=512,
                 clip_seq_len=77, t5_seq_len=256):
        df = pd.read_csv(csv_path)

        if split:
            df = df[df["split"].isin(split)].copy()

        df = df[df["generated_caption"].notna()].copy()
        df = df[df["generated_caption"].str.strip() != ""].copy()

        logger.info(f"Checking image sizes for {len(df)} rows...")
        valid_paths = []
        for _, row in df.iterrows():
            p = row["final_image_path"]
            if os.path.exists(p):
                try:
                    with Image.open(p) as img:
                        w, h = img.size
                        if min(w, h) >= min_size:
                            valid_paths.append(row.name)
                except Exception:
                    pass
        df = df.loc[valid_paths]
        logger.info(f"After size filter (>={min_size}px): {len(df)} images")

        self.df               = df.reset_index(drop=True)
        self.tok_clip_l       = tokenizer_clip_l
        self.tok_clip_g       = tokenizer_clip_g
        self.tok_t5           = tokenizer_t5
        self.image_size       = image_size
        self.clip_seq_len     = clip_seq_len
        self.t5_seq_len       = t5_seq_len

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # -> [-1, 1]
        ])

    def __len__(self):
        return len(self.df)

    def _tok(self, tokenizer, caption, max_len):
        return tokenizer(
            caption,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["final_image_path"]).convert("RGB")
        img = self.transform(img)

        caption = str(row["generated_caption"]).strip()

        return {
            "pixel_values":   img,
            "input_ids_l":    self._tok(self.tok_clip_l, caption, self.clip_seq_len),
            "input_ids_g":    self._tok(self.tok_clip_g, caption, self.clip_seq_len),
            "input_ids_t5":   self._tok(self.tok_t5, caption, self.t5_seq_len),
            "caption":        caption,
        }


train_losses = []
val_losses = []


# ─────────────────────────────────────────────
# TEXT ENCODING (mirrors StableDiffusion3Pipeline._get_clip_prompt_embeds /
# _get_t5_prompt_embeds, but keeping it explicit here so it's easy to see
# exactly how the long T5 sequence gets combined with the two CLIP branches)
# ─────────────────────────────────────────────
def encode_prompts(input_ids_l, input_ids_g, input_ids_t5,
                    text_encoder_l, text_encoder_g, text_encoder_t5,
                    joint_attention_dim=4096):
    # CLIP-L: pooled + penultimate hidden states
    out_l = text_encoder_l(input_ids_l, output_hidden_states=True)
    pooled_l = out_l[0]
    hidden_l = out_l.hidden_states[-2]

    # CLIP-G: pooled + penultimate hidden states
    out_g = text_encoder_g(input_ids_g, output_hidden_states=True)
    pooled_g = out_g[0]
    hidden_g = out_g.hidden_states[-2]

    # Pooled projections: concat CLIP-L + CLIP-G pooled vectors
    pooled = torch.cat([pooled_l, pooled_g], dim=-1)

    # CLIP hidden states concat on channel dim, then pad channel dim up to T5's width
    clip_hidden = torch.cat([hidden_l, hidden_g], dim=-1)
    pad = clip_hidden.new_zeros(
        clip_hidden.shape[0], clip_hidden.shape[1],
        joint_attention_dim - clip_hidden.shape[-1]
    )
    clip_hidden = torch.cat([clip_hidden, pad], dim=-1)

    # T5: full sequence (this is the branch with the extended MAX_T5_SEQ_LEN)
    t5_hidden = text_encoder_t5(input_ids_t5)[0]

    # Sequence dim concat: [CLIP tokens (77) ++ T5 tokens (MAX_T5_SEQ_LEN)]
    encoder_hidden_states = torch.cat([clip_hidden, t5_hidden], dim=1)

    return encoder_hidden_states, pooled


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if MIXED_PRECISION else torch.float32

    logger.info("=" * 60)
    logger.info("  MatCLIP — SD3 LoRA Diffusion Training")
    logger.info(f"  Device      : {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU         : {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info(f"  LoRA        : rank={LORA_RANK}, alpha={LORA_ALPHA}")
    logger.info(f"  Batch       : {BATCH_SIZE} x {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} effective")
    logger.info(f"  T5 seq len  : {MAX_T5_SEQ_LEN} (extended context, CLIP branches fixed at {CLIP_SEQ_LEN})")
    logger.info("=" * 60)

    # ── Load SD3 components ──────────────────
    logger.info("Loading SD3 components...")
    tokenizer_l  = CLIPTokenizer.from_pretrained(SD3_PATH, subfolder="tokenizer")
    tokenizer_g  = CLIPTokenizer.from_pretrained(SD3_PATH, subfolder="tokenizer_2")
    tokenizer_t5 = T5TokenizerFast.from_pretrained(SD3_PATH, subfolder="tokenizer_3")

    text_encoder_l  = CLIPTextModelWithProjection.from_pretrained(
        SD3_PATH, subfolder="text_encoder").to(device, dtype=dtype)
    text_encoder_g  = CLIPTextModelWithProjection.from_pretrained(
        SD3_PATH, subfolder="text_encoder_2").to(device, dtype=dtype)
    text_encoder_t5 = T5EncoderModel.from_pretrained(
        SD3_PATH, subfolder="text_encoder_3").to(device, dtype=dtype)

    vae = AutoencoderKL.from_pretrained(SD3_PATH, subfolder="vae").to(device, dtype=dtype)

    transformer = SD3Transformer2DModel.from_pretrained(
        SD3_PATH, subfolder="transformer").to(device, dtype=dtype)

    noise_sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        SD3_PATH, subfolder="scheduler")

    joint_attention_dim = transformer.config.joint_attention_dim  # typically 4096

    # Freeze everything except the transformer LoRA
    vae.requires_grad_(False)
    text_encoder_l.requires_grad_(False)
    text_encoder_g.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)
    transformer.requires_grad_(False)

    # ── Apply LoRA to the transformer ────────
    logger.info(f"Applying LoRA (rank={LORA_RANK}) to SD3 transformer attention layers...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
        ],
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # ── Dataset & Dataloader ─────────────────
    logger.info("Building dataset...")
    common_kwargs = dict(
        csv_path=CSV_PATH,
        tokenizer_clip_l=tokenizer_l,
        tokenizer_clip_g=tokenizer_g,
        tokenizer_t5=tokenizer_t5,
        image_size=IMAGE_SIZE,
        min_size=MIN_IMAGE_SIZE,
        clip_seq_len=CLIP_SEQ_LEN,
        t5_seq_len=MAX_T5_SEQ_LEN,
    )
    train_dataset = MicrostructureDataset(split={"train"}, **common_kwargs)
    val_dataset   = MicrostructureDataset(split={"val"}, **common_kwargs)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Optimizer & Scheduler (LR schedule, not diffusion scheduler) ────
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=LR, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8,
    )

    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = LR_WARMUP_STEPS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────
    global_step = 0
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        logger.info(f"Resuming from checkpoint: {RESUME_FROM}")
        weights = load_safetensors(f"{RESUME_FROM}/adapter_model.safetensors")
        set_peft_model_state_dict(transformer, weights)
        logger.info("Checkpoint weights loaded successfully")

        import re as _re
        _match = _re.search(r"step_(\d+)", RESUME_FROM)
        if _match:
            global_step = int(_match.group(1))
            logger.info(f"Resuming global_step from {global_step}")

    logger.info(f"Starting training: {EPOCHS} epochs, {total_steps} total steps")

    best_loss    = float("inf")
    running_loss = 0.0

    transformer.train()

    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    logger.info(f"Starting from epoch {start_epoch+1}/{EPOCHS}")

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            ids_l  = batch["input_ids_l"].to(device)
            ids_g  = batch["input_ids_g"].to(device)
            ids_t5 = batch["input_ids_t5"].to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

            # Flow-matching noise: sample timestep in [0,1], linear interpolate
            bsz = latents.shape[0]
            u = torch.rand(bsz, device=device)
            timesteps = (u * noise_sched.config.num_train_timesteps).long()
            sigmas = noise_sched.sigmas.to(device=device, dtype=dtype)[timesteps].view(-1, 1, 1, 1)

            noise = torch.randn_like(latents)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            # flow-matching target is (noise - latents), i.e. velocity
            target = noise - latents

            # Text encoding (long-context T5 branch included here)
            with torch.no_grad():
                encoder_hidden_states, pooled = encode_prompts(
                    ids_l, ids_g, ids_t5,
                    text_encoder_l, text_encoder_g, text_encoder_t5,
                    joint_attention_dim=joint_attention_dim,
                )

            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled,
            ).sample

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss = loss / GRAD_ACCUM
            loss.backward()

            running_loss += loss.item() * GRAD_ACCUM
            epoch_loss   += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = running_loss / 50
                    lr_now   = lr_scheduler.get_last_lr()[0]
                    vram     = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    logger.info(
                        f"  Epoch {epoch+1:>3}/{EPOCHS} | "
                        f"Step {global_step:>5}/{total_steps} | "
                        f"Loss {avg_loss:.4f} | "
                        f"LR {lr_now:.2e} | "
                        f"VRAM {vram:.1f}GB"
                    )
                    running_loss = 0.0

        avg_epoch = epoch_loss / (step + 1)
        transformer.eval()

        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device, dtype=dtype)
                ids_l  = batch["input_ids_l"].to(device)
                ids_g  = batch["input_ids_g"].to(device)
                ids_t5 = batch["input_ids_t5"].to(device)

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                torch.manual_seed(SEED)
                bsz = latents.shape[0]
                u = torch.rand(bsz, device=device)
                timesteps = (u * noise_sched.config.num_train_timesteps).long()
                sigmas = noise_sched.sigmas.to(device=device, dtype=dtype)[timesteps].view(-1, 1, 1, 1)

                noise = torch.randn_like(latents)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                target = noise - latents

                encoder_hidden_states, pooled = encode_prompts(
                    ids_l, ids_g, ids_t5,
                    text_encoder_l, text_encoder_g, text_encoder_t5,
                    joint_attention_dim=joint_attention_dim,
                )

                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled,
                ).sample

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_path = os.path.join(CKPT_DIR, "best")
            transformer.save_pretrained(best_path)
            logger.info(f"New best model saved (Validation Loss = {best_loss:.4f})")

        train_losses.append(avg_epoch)
        val_losses.append(avg_val_loss)

        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Train Loss: {avg_epoch:.4f} "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        if (epoch + 1) % 25 == 0:
            epoch_path = os.path.join(CKPT_DIR, f"epoch_{epoch+1}")
            transformer.save_pretrained(epoch_path)
            logger.info(f"Epoch checkpoint saved to {epoch_path}")

        transformer.train()
        logger.info(f"Epoch {epoch+1}/{EPOCHS} complete — avg loss: {avg_epoch:.4f}")

    # ── Save final LoRA weights ──────────────
    final_path = os.path.join(CKPT_DIR, "final")
    transformer.save_pretrained(final_path)
    logger.info(f"\nTraining complete. Final LoRA saved -> {final_path}")
    logger.info(f"   Best loss achieved: {best_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker="o", label="Training Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, marker="s", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (SD3 LoRA)")
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(CKPT_DIR, "loss_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Loss curve saved to {plot_path}")

    # ── Quick inference test ─────────────────
    logger.info("\nRunning quick inference test...")
    try:
        transformer_base = SD3Transformer2DModel.from_pretrained(
            SD3_PATH, subfolder="transformer", torch_dtype=torch.bfloat16,
        ).to(device)
        transformer_lora = PeftModel.from_pretrained(transformer_base, final_path)
        transformer_merged = transformer_lora.merge_and_unload()
        transformer_merged = transformer_merged.to(device, dtype=torch.bfloat16)

        pipe = StableDiffusion3Pipeline.from_pretrained(
            SD3_PATH,
            transformer=transformer_merged,
            torch_dtype=torch.bfloat16,
        ).to(device)

        test_prompts = [
            "SEM micrograph showing cellular microstructure with hexagonal cell walls, bright boundaries, dark interiors. Process: LPBF, 285W, 960mm/s.",
            "SEM micrograph showing columnar grains aligned parallel to build direction, high aspect ratio. Process: LPBF, 300W.",
            "SEM micrograph showing dendritic microstructure with primary arms and secondary branches. Process: DED.",
        ]

        out_dir = os.path.join(CKPT_DIR, "test_images")
        os.makedirs(out_dir, exist_ok=True)

        for i, prompt in enumerate(test_prompts):
            image = pipe(
                prompt,
                num_inference_steps=28,
                guidance_scale=7.0,
                max_sequence_length=MAX_T5_SEQ_LEN,   # <- also set at inference time
            ).images[0]
            out_path = os.path.join(out_dir, f"test_{i+1}.png")
            image.save(out_path)
            logger.info(f"  Test image saved -> {out_path}")

        logger.info("Inference test complete.")
    except Exception as e:
        logger.warning(f"Inference test failed: {e} — training weights are still saved.")


if __name__ == "__main__":
    main()