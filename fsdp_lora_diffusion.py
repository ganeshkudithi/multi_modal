

import os
import math
import random
import logging
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SD_PATH         = "/home/cdacapp01/wk-ganesh/multi-modal/models/stable-diffusion-v1-5"
BASE_DIR        = "/home/cdacapp01/wk-ganesh/multi-modal"
CSV_PATH        = os.path.join(BASE_DIR, "matclip_phase1_cluster.csv")
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints_diffusion_lora")

# Training data
TARGET_TIERS    = {"A"}
MIN_IMAGE_SIZE  = 0

# LoRA config
LORA_RANK       = 16            # higher = more capacity, more VRAM
LORA_ALPHA      = 32            # scaling factor (usually 2x rank)
LORA_DROPOUT    = 0.1

# Training hyperparams
IMAGE_SIZE      = 512           # SD 1.5 native resolution
BATCH_SIZE      = 64           # A100 80GB can handle 4 at 512x512
GRAD_ACCUM      = 4             # effective batch = 4 × 4 = 16
EPOCHS          = 1
LR              = 1e-4          # standard for LoRA
LR_WARMUP_STEPS = 200
MAX_GRAD_NORM   = 1.0
MIXED_PRECISION = True          # bfloat16 for A100

# Saving
SAVE_EVERY_STEPS = 500
RESUME_FROM      = "checkpoints_diffusion_lora/step_85000"  # resume checkpoint
SEED            = 42

os.makedirs(CKPT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class MicrostructureDataset(Dataset):
    def __init__(self, csv_path, tokenizer, image_size=512,
                 target_tiers=None, min_size=512):
        df = pd.read_csv(csv_path)

        # Filter tiers
        if target_tiers:
            df = df[df["quality_tier"].isin(target_tiers)].copy()

        # Filter by image existence
        df = df[df["final_image_path"].notna()].copy()
        df = df[df["diffusion_caption"].notna()].copy()
        df = df[df["diffusion_caption"].str.strip() != ""].copy()

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

        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_size = image_size

        # Image transforms — resize shortest side to 512, random crop, normalize
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

        # Load image
        img = Image.open(row["final_image_path"]).convert("RGB")
        img = self.transform(img)

        # Build caption: diffusion_caption + key params if available
        caption = self._build_caption(row)

        # Tokenize
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": img,
            "input_ids": tokens.input_ids.squeeze(0),
            "caption": caption,
        }

    def _build_caption(self, row):
        """
        Combine diffusion_caption (visual) + key process params.
        Format: '<visual description>. Process: LPBF, 285W, 960mm/s.'
        """
        base = str(row["diffusion_caption"]).strip()

        # Append process params if available
        extras = []
        process = row.get("param_process_type", "")
        if pd.notna(process) and str(process).strip() not in ("", "nan"):
            extras.append(str(process))

        laser = row.get("param_laser_power_W")
        if pd.notna(laser):
            extras.append(f"{laser:.0f}W")

        speed = row.get("param_scan_speed_mm_s")
        if pd.notna(speed):
            extras.append(f"{speed:.0f}mm/s")

        material = row.get("param_material_grade")
        if pd.notna(material) and str(material).strip().lower() not in ("nan", "none", ""):
            extras.append(str(material))

        if extras:
            base = base + " Process: " + ", ".join(extras) + "."

        return base


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype  = torch.bfloat16 if MIXED_PRECISION else torch.float32

    logger.info("=" * 60)
    logger.info("  MatCLIP — LoRA Diffusion Training")
    logger.info(f"  Device : {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU    : {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info(f"  LoRA   : rank={LORA_RANK}, alpha={LORA_ALPHA}")
    logger.info(f"  Batch  : {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} effective")
    logger.info("=" * 60)

    # ── Load SD components ───────────────────
    logger.info("Loading SD 1.5 components...")
    tokenizer    = CLIPTokenizer.from_pretrained(SD_PATH, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
    SD_PATH,
    subfolder="text_encoder"
).to(device, dtype=dtype)

    vae = AutoencoderKL.from_pretrained(
    SD_PATH,
    subfolder="vae"
).to(device, dtype=dtype)

    unet = UNet2DConditionModel.from_pretrained(
    SD_PATH,
    subfolder="unet"
).to(device, dtype=dtype)
    noise_sched  = DDPMScheduler.from_pretrained(SD_PATH, subfolder="scheduler")

    # Freeze VAE and text encoder — only train UNet LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # ── Apply LoRA to UNet ───────────────────
    logger.info(f"Applying LoRA (rank={LORA_RANK}) to UNet attention layers...")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_k_proj",
            "add_v_proj",
        ],
        bias="none",
    )

    # Convert UNet into a PEFT model
    unet = get_peft_model(unet, lora_config)

    # Print trainable parameters
    unet.print_trainable_parameters()

    

    # --------------------------------------------------
    # Mixed Precision Policy
    # --------------------------------------------------
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # --------------------------------------------------
    # Wrap with FSDP
    # --------------------------------------------------
    unet = FSDP(
        unet,
        use_orig_params=True,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        device_id=local_rank,
    )

    logger.info("UNet wrapped with FSDP successfully.")

    # ── Dataset & Dataloader ─────────────────
    logger.info("Building dataset...")
    dataset = MicrostructureDataset(
        csv_path=CSV_PATH,
        tokenizer=tokenizer,
        image_size=IMAGE_SIZE,
        target_tiers=TARGET_TIERS,
        min_size=MIN_IMAGE_SIZE,
    )
    logger.info(f"Dataset size: {len(dataset)} images")

    sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)

    dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=8,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)

    # ── Optimizer & Scheduler ────────────────
    optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, unet.parameters()),
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-8,
)

    total_steps    = (len(dataloader) // GRAD_ACCUM) * EPOCHS
    warmup_steps   = LR_WARMUP_STEPS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training Loop ────────────────────────
    # Resume from checkpoint if specified
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        
        logger.info(f"Resuming from checkpoint: {RESUME_FROM}")
        # Load saved LoRA weights into already-initialized LoRA model
        from peft import set_peft_model_state_dict
        import torch as _torch
        from safetensors.torch import load_file as _load_safetensors
        weights = _load_safetensors(f"{RESUME_FROM}/adapter_model.safetensors")
        set_peft_model_state_dict(unet, weights)
        logger.info("Checkpoint weights loaded successfully")

    logger.info(f"Starting training: {EPOCHS} epochs, {total_steps} total steps")
    logger.info(f"Saving checkpoints every {SAVE_EVERY_STEPS} steps → {CKPT_DIR}")

    # Set global_step from checkpoint name if resuming
    global_step = 0
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        import re as _re
        _match = _re.search(r'step_(\d+)', RESUME_FROM)
        if _match:
            global_step = int(_match.group(1))
            logger.info(f"Resuming global_step from {global_step}")
    best_loss    = float("inf")
    running_loss = 0.0

    unet.train()

    steps_per_epoch = len(dataloader) // GRAD_ACCUM
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    logger.info(f"Starting from epoch {start_epoch+1}/{EPOCHS}")
    for epoch in range(start_epoch, EPOCHS):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            input_ids    = batch["input_ids"].to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise and timesteps
            noise      = torch.randn_like(latents)
            timesteps  = torch.randint(
                0, noise_sched.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()

            # Add noise to latents
            noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict noise
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # Loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = loss / GRAD_ACCUM
            loss.backward()

            loss_detached = loss.detach()

            dist.all_reduce(loss_detached)
            loss_detached /= world_size

            running_loss += loss_detached.item()
            epoch_loss += loss_detached.item()
            # epoch_loss   += loss.item()

            # Gradient accumulation step
            if (step + 1) % GRAD_ACCUM == 0:
                unet.clip_grad_norm_(MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
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

                # Save checkpoint
                if global_step % SAVE_EVERY_STEPS == 0:
                    ckpt_path = os.path.join(CKPT_DIR, f"step_{global_step}")
                    if rank == 0:
                        unet.state_dict()
                        unet.module.save_pretrained(ckpt_path)
                    logger.info(f"  [SAVED] checkpoint → {ckpt_path}")

                    # Track best loss
                    avg_epoch_loss = epoch_loss / (step + 1)
                    if avg_epoch_loss < best_loss:
                        best_loss = avg_epoch_loss
                        best_path = os.path.join(CKPT_DIR, "best")
                        if rank == 0:
                            unet.module.save_pretrained(best_path)
                        logger.info(f"  [BEST]  loss={best_loss:.4f} → {best_path}")

        avg_epoch = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} complete — avg loss: {avg_epoch:.4f}")

    # ── Save Final LoRA weights ──────────────
    final_path = os.path.join(CKPT_DIR, "final")
    if rank == 0:
        unet.module.save_pretrained(final_path)
    logger.info(f"\n✅ Training complete. Final LoRA saved → {final_path}")
    logger.info(f"   Best loss achieved: {best_loss:.4f}")

    # ── Quick inference test ─────────────────
    logger.info("\nRunning quick inference test...")
    try:
        from peft import PeftModel
        unet_base = UNet2DConditionModel.from_pretrained(SD_PATH, subfolder="unet")
        unet_lora = PeftModel.from_pretrained(unet_base, final_path)
        unet_merged = unet_lora.merge_and_unload()

        pipe = StableDiffusionPipeline.from_pretrained(
            SD_PATH,
            unet=unet_merged,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
        ).to(device)

        test_prompts = [
            "SEM micrograph showing cellular microstructure with hexagonal cell walls, bright boundaries, dark interiors. Process: LPBF, 285W, 960mm/s.",
            "SEM micrograph showing columnar grains aligned parallel to build direction, high aspect ratio. Process: LPBF, 300W.",
            "SEM micrograph showing dendritic microstructure with primary arms and secondary branches. Process: DED.",
        ]

        out_dir = os.path.join(CKPT_DIR, "test_images")
        os.makedirs(out_dir, exist_ok=True)

        for i, prompt in enumerate(test_prompts):
            image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
            out_path = os.path.join(out_dir, f"test_{i+1}.png")
            image.save(out_path)
            logger.info(f"  Test image saved → {out_path}")

        logger.info("✅ Inference test complete.")
    except Exception as e:
        logger.warning(f"Inference test failed: {e} — training weights are still saved.")
    if dist.is_initialized():
        dist.destroy_process_group()    


if __name__ == "__main__":
    main()



