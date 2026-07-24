"""
train_flux_lora.py
========================
LoRA fine-tuning of FLUX.1-dev on MatCLIP microstructure dataset.
Uses HuggingFace diffusers + peft for efficient training.

FLUX is architecturally different from SD 1.5 in several important ways that
this script accounts for:
  - Two text encoders: CLIP (pooled embedding) + T5-XXL (sequence embedding)
  - VAE has 16 latent channels and uses a shift_factor + scaling_factor
  - Denoiser is a transformer (FluxTransformer2DModel / MMDiT), not a UNet,
    and expects "packed" latent patches + positional id tensors (img_ids/txt_ids)
  - Flow-matching training objective (velocity target) instead of epsilon
    prediction with DDPM noise scheduling
  - FLUX.1-dev is guidance-distilled, so a guidance scale tensor must be
    passed into the transformer forward pass even during training

Usage:
    python flux_lora.py

SLURM:
    sbatch train_lora.slurm

Outputs:
    checkpoints_flux_lora_filtered_flux/          — LoRA weights saved every epoch / best
    checkpoints_flux_lora_filtered_flux/final/    — final LoRA adapter ready for inference

Requirements:
    diffusers >= 0.30 (needs FluxTransformer2DModel, FluxPipeline, FlowMatchEulerDiscreteScheduler)
    transformers (needs T5EncoderModel / T5TokenizerFast)
    peft, safetensors, accelerate
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
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from peft import LoraConfig, get_peft_model, PeftModel, set_peft_model_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FLUX_PATH       = "/home/cdacapp01/wk-ganesh/multi-modal/models/FLUX.1-dev"
BASE_DIR        = "/home/cdacapp01/wk-ganesh/multi-modal"
CSV_PATH        = os.path.join(BASE_DIR, "matclip_phase1_cluster_caption_filtered.csv")
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints_flux_lora_filtered_flux")

# Training data
MIN_IMAGE_SIZE  = 0

# LoRA config
LORA_RANK       = 16            # higher = more capacity, more VRAM
LORA_ALPHA      = 32            # scaling factor (usually 2x rank)
LORA_DROPOUT    = 0.1
# FLUX attention / MLP module names (different from SD's UNet naming)
LORA_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",              # self-attention (image stream)
    "add_q_proj", "add_k_proj", "add_v_proj",        # joint attention (text stream)
    "to_add_out",
    "proj_mlp", "proj_out",                          # single-stream MLP blocks
]

# Training hyperparams
IMAGE_SIZE      = 512           # can go up to 1024 for FLUX, but VRAM-hungry
# FLUX.1-dev is a ~12B parameter transformer — this is NOT SD 1.5.
# BATCH_SIZE=32 will not fit on a single A100/H100 at 512x512. Start small
# and raise it only after confirming it fits your GPU memory.
BATCH_SIZE      = 1
GRAD_ACCUM      = 16            # effective batch = 1 × 16 = 16
EPOCHS          = 1
LR              = 1e-4          # standard for LoRA
LR_WARMUP_STEPS = 200
MAX_GRAD_NORM   = 1.0
MIXED_PRECISION = True          # bfloat16
GRADIENT_CHECKPOINTING = True   # strongly recommended for FLUX, saves a lot of VRAM

# Flow-matching timestep sampling (logit-normal, matches diffusers' official
# FLUX LoRA training example)
WEIGHTING_SCHEME = "logit_normal"
LOGIT_MEAN       = 0.0
LOGIT_STD        = 1.0

# Text encoding
MAX_T5_SEQ_LEN   = 256           # 512 is the FLUX max; 256 is commonly used for finetuning and is cheaper
GUIDANCE_SCALE   = 3.5           # FLUX.1-dev is guidance-distilled; fixed value used at train time

RESUME_FROM      = None  # resume checkpoint (path to a saved LoRA adapter dir)
SEED             = 42

os.makedirs(CKPT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class MicrostructureDataset(Dataset):
    """
    Returns raw pixel values + raw caption text. Unlike the SD1.5 version,
    tokenization is NOT done here — FLUX needs two different tokenizers
    (CLIP + T5) applied together at encode time, so captions are tokenized
    in the training loop via encode_prompt().
    """
    def __init__(self, csv_path, image_size=512, split=None, min_size=512):
        df = pd.read_csv(csv_path)

        # Filter split
        if split:
            df = df[df["split"].isin(split)].copy()

        # Filter by image existence
        df = df[df["generated_caption"].notna()].copy()
        df = df[df["generated_caption"].str.strip() != ""].copy()

        # Filter by image size
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
                except:
                    pass
        df = df.loc[valid_paths]
        logger.info(f"After size filter (≥{min_size}px): {len(df)} images")

        self.df = df.reset_index(drop=True)
        self.image_size = image_size

        # Image transforms — resize shortest side, center crop, normalize
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["final_image_path"]).convert("RGB")
        img = self.transform(img)

        caption = str(row["generated_caption"]).strip()

        return {
            "pixel_values": img,
            "caption": caption,
        }


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    return {"pixel_values": pixel_values, "captions": captions}


# ─────────────────────────────────────────────
# TEXT ENCODING (CLIP pooled + T5 sequence)
# ─────────────────────────────────────────────
def encode_prompt(captions, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two,
                   device, max_sequence_length=256):
    # CLIP → pooled embedding (used for global conditioning)
    clip_inputs = tokenizer_one(
        captions, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).to(device)
    clip_outputs = text_encoder_one(clip_inputs.input_ids, output_hidden_states=False)
    pooled_prompt_embeds = clip_outputs.pooler_output  # (B, 768)

    # T5 → sequence embedding (used as encoder_hidden_states / cross-attention context)
    t5_inputs = tokenizer_two(
        captions, padding="max_length", max_length=max_sequence_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    t5_outputs = text_encoder_two(t5_inputs.input_ids)
    prompt_embeds = t5_outputs[0]  # (B, seq_len, 4096)

    dtype = prompt_embeds.dtype
    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


# ─────────────────────────────────────────────
# LATENT PACKING (FLUX-specific — packs 2x2 patches into tokens)
# ─────────────────────────────────────────────
def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def prepare_latent_image_ids(height, width, device, dtype):
    """height, width here are already the *packed* (halved) spatial dims."""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


# ─────────────────────────────────────────────
# FLOW-MATCHING TIMESTEP SAMPLING / SIGMA HELPERS
# ─────────────────────────────────────────────
def compute_density_for_timestep_sampling(batch_size, logit_mean=0.0, logit_std=1.0):
    u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,))
    u = torch.nn.functional.sigmoid(u)
    return u


def get_sigmas(noise_scheduler, timesteps, n_dim, dtype, device):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


train_losses = []
val_losses = []

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
    logger.info("  MatCLIP — LoRA FLUX.1-dev Training")
    logger.info(f"  Device : {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU    : {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info(f"  LoRA   : rank={LORA_RANK}, alpha={LORA_ALPHA}")
    logger.info(f"  Batch  : {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} effective")
    logger.info("=" * 60)

    # ── Load FLUX components ─────────────────
    logger.info("Loading FLUX.1-dev components...")
    tokenizer_one    = CLIPTokenizer.from_pretrained(FLUX_PATH, subfolder="tokenizer")
    tokenizer_two    = T5TokenizerFast.from_pretrained(FLUX_PATH, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(FLUX_PATH, subfolder="text_encoder").to(device, dtype=dtype)
    text_encoder_two = T5EncoderModel.from_pretrained(FLUX_PATH, subfolder="text_encoder_2").to(device, dtype=dtype)
    vae              = AutoencoderKL.from_pretrained(FLUX_PATH, subfolder="vae").to(device, dtype=dtype)
    transformer      = FluxTransformer2DModel.from_pretrained(FLUX_PATH, subfolder="transformer").to(device, dtype=dtype)
    noise_sched      = FlowMatchEulerDiscreteScheduler.from_pretrained(FLUX_PATH, subfolder="scheduler")

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # Freeze VAE and text encoders — only train transformer LoRA
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    transformer.requires_grad_(False)

    if GRADIENT_CHECKPOINTING:
        transformer.enable_gradient_checkpointing()

    # ── Apply LoRA to transformer ────────────
    logger.info(f"Applying LoRA (rank={LORA_RANK}) to FLUX transformer blocks...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # ── Dataset & Dataloader ─────────────────
    logger.info("Building dataset...")
    train_dataset = MicrostructureDataset(
        csv_path=CSV_PATH,
        image_size=IMAGE_SIZE,
        split={"train"},
        min_size=MIN_IMAGE_SIZE,
    )

    val_dataset = MicrostructureDataset(
        csv_path=CSV_PATH,
        image_size=IMAGE_SIZE,
        split={"val"},
        min_size=MIN_IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── Optimizer & Scheduler ────────────────
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = LR_WARMUP_STEPS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        logger.info(f"Resuming from checkpoint: {RESUME_FROM}")
        from safetensors.torch import load_file as _load_safetensors
        weights = _load_safetensors(f"{RESUME_FROM}/adapter_model.safetensors")
        set_peft_model_state_dict(transformer, weights)
        logger.info("Checkpoint weights loaded successfully")

    logger.info(f"Starting training: {EPOCHS} epochs, {total_steps} total steps")

    global_step = 0
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        import re as _re
        _match = _re.search(r'step_(\d+)', RESUME_FROM)
        if _match:
            global_step = int(_match.group(1))
            logger.info(f"Resuming global_step from {global_step}")

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
            captions     = batch["captions"]
            bsz          = pixel_values.shape[0]

            # ── Encode images to latent space ──
            with torch.no_grad():
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype)

            num_channels_latents = model_input.shape[1]
            latent_h = model_input.shape[2]
            latent_w = model_input.shape[3]

            # ── Text embeddings (CLIP pooled + T5 sequence) ──
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    captions, tokenizer_one, tokenizer_two,
                    text_encoder_one, text_encoder_two,
                    device, max_sequence_length=MAX_T5_SEQ_LEN,
                )

            # ── Pack latents into FLUX's patch-token format ──
            packed_model_input = pack_latents(model_input, bsz, num_channels_latents, latent_h, latent_w)
            noise = torch.randn_like(model_input)
            packed_noise = pack_latents(noise, bsz, num_channels_latents, latent_h, latent_w)

            latent_image_ids = prepare_latent_image_ids(latent_h // 2, latent_w // 2, device, dtype)

            # ── Sample timesteps (logit-normal, flow matching) ──
            u = compute_density_for_timestep_sampling(bsz, logit_mean=LOGIT_MEAN, logit_std=LOGIT_STD)
            indices = (u * noise_sched.config.num_train_timesteps).long()
            timesteps = noise_sched.timesteps[indices].to(device=device)

            sigmas = get_sigmas(noise_sched, timesteps, n_dim=packed_model_input.ndim, dtype=dtype, device=device)
            noisy_model_input = (1.0 - sigmas) * packed_model_input + sigmas * packed_noise

            # ── Guidance embedding (FLUX.1-dev is guidance-distilled) ──
            if transformer.config.guidance_embeds:
                guidance = torch.full((bsz,), GUIDANCE_SCALE, device=device, dtype=dtype)
            else:
                guidance = None

            # ── Predict velocity ──
            model_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            # ── Flow-matching loss: target is (noise - clean latent) ──
            target = packed_noise - packed_model_input
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss = loss / GRAD_ACCUM
            loss.backward()

            running_loss += loss.item() * GRAD_ACCUM
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = running_loss / 50
                    lr_now   = scheduler.get_last_lr()[0]
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
                captions     = batch["captions"]
                bsz          = pixel_values.shape[0]

                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype)

                num_channels_latents = model_input.shape[1]
                latent_h = model_input.shape[2]
                latent_w = model_input.shape[3]

                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    captions, tokenizer_one, tokenizer_two,
                    text_encoder_one, text_encoder_two,
                    device, max_sequence_length=MAX_T5_SEQ_LEN,
                )

                packed_model_input = pack_latents(model_input, bsz, num_channels_latents, latent_h, latent_w)

                torch.manual_seed(SEED)
                noise = torch.randn_like(model_input)
                packed_noise = pack_latents(noise, bsz, num_channels_latents, latent_h, latent_w)

                latent_image_ids = prepare_latent_image_ids(latent_h // 2, latent_w // 2, device, dtype)

                u = compute_density_for_timestep_sampling(bsz, logit_mean=LOGIT_MEAN, logit_std=LOGIT_STD)
                indices = (u * noise_sched.config.num_train_timesteps).long()
                timesteps = noise_sched.timesteps[indices].to(device=device)

                sigmas = get_sigmas(noise_sched, timesteps, n_dim=packed_model_input.ndim, dtype=dtype, device=device)
                noisy_model_input = (1.0 - sigmas) * packed_model_input + sigmas * packed_noise

                if transformer.config.guidance_embeds:
                    guidance = torch.full((bsz,), GUIDANCE_SCALE, device=device, dtype=dtype)
                else:
                    guidance = None

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                target = packed_noise - packed_model_input
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

    # ── Save Final LoRA weights ──────────────
    final_path = os.path.join(CKPT_DIR, "final")
    transformer.save_pretrained(final_path)
    logger.info(f"\n Training complete. Final LoRA saved → {final_path}")
    logger.info(f"   Best loss achieved: {best_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker="o", label="Training Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, marker="s", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(CKPT_DIR, "loss_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Loss curve saved to {plot_path}")

    # ── Quick inference test ─────────────────
    logger.info("\nRunning quick inference test...")
    try:
        transformer_base = FluxTransformer2DModel.from_pretrained(
            FLUX_PATH, subfolder="transformer", torch_dtype=torch.bfloat16,
        ).to(device)
        transformer_lora = PeftModel.from_pretrained(transformer_base, final_path)
        transformer_merged = transformer_lora.merge_and_unload()
        transformer_merged = transformer_merged.to(device, dtype=torch.bfloat16)

        pipe = FluxPipeline.from_pretrained(
            FLUX_PATH,
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
                num_inference_steps=30,
                guidance_scale=GUIDANCE_SCALE,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            ).images[0]
            out_path = os.path.join(out_dir, f"test_{i+1}.png")
            image.save(out_path)
            logger.info(f"  Test image saved → {out_path}")

        logger.info("✅ Inference test complete.")
    except Exception as e:
        logger.warning(f"Inference test failed: {e} — training weights are still saved.")


if __name__ == "__main__":
    main()