"""
train_flux_lora_multigpu.py
========================
Multi-GPU LoRA fine-tuning of FLUX.1-dev on MatCLIP microstructure dataset.
Uses HuggingFace diffusers + peft + accelerate (DistributedDataParallel).

This is a data-parallel setup: every GPU holds a full copy of the frozen
FLUX components (VAE, CLIP, T5, transformer backbone). Only the LoRA
adapter weights (small) have gradients synced across GPUs each step, so
communication overhead is low even though the frozen model is large.

accelerate's `Accelerator()` auto-detects the distributed environment from
the standard torch.distributed env vars (RANK, LOCAL_RANK, WORLD_SIZE,
MASTER_ADDR, MASTER_PORT), so this script does NOT require `accelerate
launch` or `accelerate config` — launching with torchrun is sufficient.

Launch with torchrun (single node, multi-GPU):
    torchrun --standalone --nproc_per_node=<NUM_GPUS> train_flux_lora_multigpu.py

Launch with torchrun (multi-node — run this same command on every node):
    torchrun \\
        --nnodes=<NUM_NODES> \\
        --nproc_per_node=<GPUS_PER_NODE> \\
        --rdzv_backend=c10d \\
        --rdzv_endpoint=<MASTER_NODE_HOST>:<PORT> \\
        train_flux_lora_multigpu.py

SLURM (example, adjust to your cluster):
    srun torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE train_flux_lora_multigpu.py

(accelerate launch also still works identically, if you ever prefer it:
    accelerate launch --multi_gpu --num_processes=4 train_flux_lora_multigpu.py
)

Outputs:
    checkpoints_flux_lora_filtered_flux/          — LoRA weights saved every epoch / best
    checkpoints_flux_lora_filtered_flux/final/    — final LoRA adapter ready for inference

Requirements:
    diffusers >= 0.30, transformers, peft, safetensors, accelerate
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

from accelerate import Accelerator
from accelerate.utils import set_seed

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
LORA_RANK       = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.1
LORA_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj",
    "to_add_out",
    "proj_mlp", "proj_out",
]

# Training hyperparams
IMAGE_SIZE      = 512
# This is PER-GPU batch size. Effective global batch = BATCH_SIZE * GRAD_ACCUM * num_gpus.
BATCH_SIZE      = 1
GRAD_ACCUM      = 16
EPOCHS          = 50
LR              = 1e-4
LR_WARMUP_STEPS = 200
MAX_GRAD_NORM   = 1.0
MIXED_PRECISION = "bf16"        # passed directly to Accelerator: "no" | "fp16" | "bf16"
GRADIENT_CHECKPOINTING = True

WEIGHTING_SCHEME = "logit_normal"
LOGIT_MEAN       = 0.0
LOGIT_STD        = 1.0

MAX_T5_SEQ_LEN   = 256
GUIDANCE_SCALE   = 3.5

RESUME_FROM      = None
SEED             = 42

os.makedirs(CKPT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class MicrostructureDataset(Dataset):
    def __init__(self, csv_path, image_size=512, split=None, min_size=512):
        df = pd.read_csv(csv_path)

        if split:
            df = df[df["split"].isin(split)].copy()

        df = df[df["generated_caption"].notna()].copy()
        df = df[df["generated_caption"].str.strip() != ""].copy()

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

        self.df = df.reset_index(drop=True)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["final_image_path"]).convert("RGB")
        img = self.transform(img)
        caption = str(row["generated_caption"]).strip()
        return {"pixel_values": img, "caption": caption}


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    return {"pixel_values": pixel_values, "captions": captions}


# ─────────────────────────────────────────────
# TEXT ENCODING
# ─────────────────────────────────────────────
def encode_prompt(captions, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two,
                   device, max_sequence_length=256):
    clip_inputs = tokenizer_one(
        captions, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).to(device)
    clip_outputs = text_encoder_one(clip_inputs.input_ids, output_hidden_states=False)
    pooled_prompt_embeds = clip_outputs.pooler_output

    t5_inputs = tokenizer_two(
        captions, padding="max_length", max_length=max_sequence_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    t5_outputs = text_encoder_two(t5_inputs.input_ids)
    prompt_embeds = t5_outputs[0]

    dtype = prompt_embeds.dtype
    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


# ─────────────────────────────────────────────
# LATENT PACKING
# ─────────────────────────────────────────────
def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def prepare_latent_image_ids(height, width, device, dtype):
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


def flux_forward_and_loss(transformer, vae, noise_sched, tokenizer_one, tokenizer_two,
                           text_encoder_one, text_encoder_two, batch, device, dtype):
    """One forward pass + flow-matching loss computation. Shared by train/val loops."""
    pixel_values = batch["pixel_values"].to(device, dtype=dtype)
    captions     = batch["captions"]
    bsz          = pixel_values.shape[0]

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
        model_input = model_input.to(dtype)

    num_channels_latents = model_input.shape[1]
    latent_h = model_input.shape[2]
    latent_w = model_input.shape[3]

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            captions, tokenizer_one, tokenizer_two,
            text_encoder_one, text_encoder_two,
            device, max_sequence_length=MAX_T5_SEQ_LEN,
        )

    packed_model_input = pack_latents(model_input, bsz, num_channels_latents, latent_h, latent_w)
    noise = torch.randn_like(model_input)
    packed_noise = pack_latents(noise, bsz, num_channels_latents, latent_h, latent_w)

    latent_image_ids = prepare_latent_image_ids(latent_h // 2, latent_w // 2, device, dtype)

    u = compute_density_for_timestep_sampling(bsz, logit_mean=LOGIT_MEAN, logit_std=LOGIT_STD)
    indices = (u * noise_sched.config.num_train_timesteps).long()
    timesteps = noise_sched.timesteps[indices].to(device=device)

    sigmas = get_sigmas(noise_sched, timesteps, n_dim=packed_model_input.ndim, dtype=dtype, device=device)
    noisy_model_input = (1.0 - sigmas) * packed_model_input + sigmas * packed_noise

    base_transformer = transformer.module if hasattr(transformer, "module") else transformer
    if base_transformer.config.guidance_embeds:
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
    return loss


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def main():
    # Pin this process to its assigned GPU before any CUDA calls happen.
    # torchrun sets LOCAL_RANK per process; doing this explicitly (rather than
    # relying on default device ordering) avoids cases where multiple ranks
    # on the same node end up contending for GPU 0.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM,
        mixed_precision=MIXED_PRECISION,
    )
    set_seed(SEED)

    device = accelerator.device
    dtype  = torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float32

    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("  MatCLIP — Multi-GPU LoRA FLUX.1-dev Training")
        logger.info(f"  Num processes (GPUs): {accelerator.num_processes}")
        logger.info(f"  Mixed precision      : {MIXED_PRECISION}")
        logger.info(f"  LoRA   : rank={LORA_RANK}, alpha={LORA_ALPHA}")
        logger.info(
            f"  Batch  : {BATCH_SIZE} × {GRAD_ACCUM} accum × {accelerator.num_processes} GPUs "
            f"= {BATCH_SIZE*GRAD_ACCUM*accelerator.num_processes} effective"
        )
        logger.info("=" * 60)

    # ── Load FLUX components ─────────────────
    if accelerator.is_main_process:
        logger.info("Loading FLUX.1-dev components...")
    tokenizer_one    = CLIPTokenizer.from_pretrained(FLUX_PATH, subfolder="tokenizer")
    tokenizer_two    = T5TokenizerFast.from_pretrained(FLUX_PATH, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(FLUX_PATH, subfolder="text_encoder").to(device, dtype=dtype)
    text_encoder_two = T5EncoderModel.from_pretrained(FLUX_PATH, subfolder="text_encoder_2").to(device, dtype=dtype)
    vae              = AutoencoderKL.from_pretrained(FLUX_PATH, subfolder="vae").to(device, dtype=dtype)
    transformer      = FluxTransformer2DModel.from_pretrained(FLUX_PATH, subfolder="transformer").to(device, dtype=dtype)
    noise_sched      = FlowMatchEulerDiscreteScheduler.from_pretrained(FLUX_PATH, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    transformer.requires_grad_(False)

    if GRADIENT_CHECKPOINTING:
        transformer.enable_gradient_checkpointing()

    if accelerator.is_main_process:
        logger.info(f"Applying LoRA (rank={LORA_RANK}) to FLUX transformer blocks...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    if accelerator.is_main_process:
        transformer.print_trainable_parameters()

    # ── Dataset & Dataloader ─────────────────
    train_dataset = MicrostructureDataset(
        csv_path=CSV_PATH, image_size=IMAGE_SIZE, split={"train"}, min_size=MIN_IMAGE_SIZE,
    )
    val_dataset = MicrostructureDataset(
        csv_path=CSV_PATH, image_size=IMAGE_SIZE, split={"val"}, min_size=MIN_IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        transformer.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8,
    )

    # total_steps is computed per-process after accelerate shards the dataloader;
    # since drop_last=True and accelerate splits evenly, this stays consistent across processes.
    steps_per_epoch_estimate = (len(train_dataset) // accelerator.num_processes) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps  = max(1, steps_per_epoch_estimate) * EPOCHS
    warmup_steps = LR_WARMUP_STEPS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        if accelerator.is_main_process:
            logger.info(f"Resuming from checkpoint: {RESUME_FROM}")
        from safetensors.torch import load_file as _load_safetensors
        weights = _load_safetensors(f"{RESUME_FROM}/adapter_model.safetensors")
        set_peft_model_state_dict(transformer, weights)

    # ── Prepare for distributed training ─────
    transformer, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_loader, val_loader, lr_scheduler
    )

    if accelerator.is_main_process:
        logger.info(f"Starting training: {EPOCHS} epochs, ~{total_steps} total optimizer steps")

    global_step = 0
    best_loss    = float("inf")
    running_loss = 0.0
    running_count = 0

    transformer.train()

    for epoch in range(EPOCHS):
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(transformer):
                loss = flux_forward_and_loss(
                    transformer, vae, noise_sched, tokenizer_one, tokenizer_two,
                    text_encoder_one, text_encoder_two, batch, device, dtype,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), MAX_GRAD_NORM)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Gather loss across all GPUs for accurate logging
            gathered_loss = accelerator.gather(loss.detach().repeat(BATCH_SIZE)).mean()
            running_loss += gathered_loss.item()
            running_count += 1
            epoch_loss_sum += gathered_loss.item()
            epoch_loss_count += 1

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 50 == 0:
                    avg_loss = running_loss / max(1, running_count)
                    lr_now   = lr_scheduler.get_last_lr()[0]
                    vram     = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    logger.info(
                        f"  Epoch {epoch+1:>3}/{EPOCHS} | "
                        f"Step {global_step:>5}/{total_steps} | "
                        f"Loss {avg_loss:.4f} | "
                        f"LR {lr_now:.2e} | "
                        f"VRAM/GPU {vram:.1f}GB"
                    )
                    running_loss = 0.0
                    running_count = 0

        avg_epoch = epoch_loss_sum / max(1, epoch_loss_count)

        # ── Validation ────────────────────────
        transformer.eval()
        val_loss_sum = 0.0
        val_loss_count = 0

        with torch.no_grad():
            for batch in val_loader:
                torch.manual_seed(SEED)
                loss = flux_forward_and_loss(
                    transformer, vae, noise_sched, tokenizer_one, tokenizer_two,
                    text_encoder_one, text_encoder_two, batch, device, dtype,
                )
                gathered_loss = accelerator.gather(loss.detach().repeat(BATCH_SIZE)).mean()
                val_loss_sum += gathered_loss.item()
                val_loss_count += 1

        avg_val_loss = val_loss_sum / max(1, val_loss_count)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_path = os.path.join(CKPT_DIR, "best")
                unwrapped = accelerator.unwrap_model(transformer)
                unwrapped.save_pretrained(best_path)
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
                unwrapped = accelerator.unwrap_model(transformer)
                unwrapped.save_pretrained(epoch_path)
                logger.info(f"Epoch checkpoint saved to {epoch_path}")

        transformer.train()
        accelerator.wait_for_everyone()

    # ── Save Final LoRA weights (main process only) ──
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(CKPT_DIR, "final")
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(final_path)
        logger.info(f"\n Training complete. Final LoRA saved → {final_path}")
        logger.info(f"   Best loss achieved: {best_loss:.4f}")

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Training Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, marker="s", label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.grid(True)
        plt.legend()
        plot_path = os.path.join(CKPT_DIR, "loss_curve.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Loss curve saved to {plot_path}")

        # ── Quick inference test (main process only, single GPU) ──
        logger.info("\nRunning quick inference test...")
        try:
            transformer_base = FluxTransformer2DModel.from_pretrained(
                FLUX_PATH, subfolder="transformer", torch_dtype=torch.bfloat16,
            ).to(device)
            transformer_lora = PeftModel.from_pretrained(transformer_base, final_path)
            transformer_merged = transformer_lora.merge_and_unload()
            transformer_merged = transformer_merged.to(device, dtype=torch.bfloat16)

            pipe = FluxPipeline.from_pretrained(
                FLUX_PATH, transformer=transformer_merged, torch_dtype=torch.bfloat16,
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
                    prompt, num_inference_steps=30, guidance_scale=GUIDANCE_SCALE,
                    height=IMAGE_SIZE, width=IMAGE_SIZE,
                ).images[0]
                out_path = os.path.join(out_dir, f"test_{i+1}.png")
                image.save(out_path)
                logger.info(f"  Test image saved → {out_path}")

            logger.info("✅ Inference test complete.")
        except Exception as e:
            logger.warning(f"Inference test failed: {e} — training weights are still saved.")


if __name__ == "__main__":
    main()
