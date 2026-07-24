"""
infer_flux_lora.py
========================
Inference script for the FLUX.1-dev LoRA adapter trained by
train_flux_lora_multigpu.py (also works with the checkpoints from
sd_3_lora.py if you point --model_path / --sd3 at the right base model).

Two loading modes:
    1. Adapter mode (default) — loads the base FLUX pipeline + applies the
       LoRA adapter at inference time via PEFT. Fast to switch adapters,
       supports --lora_scale to blend adapter strength.
    2. Merge mode (--merge)   — merges LoRA weights into the transformer
       once, then runs the merged model. Slightly faster per-image if you're
       generating many images from the same checkpoint, since there's no
       LoRA computation overhead during the forward pass.

Usage examples:
    # single prompt
    python infer_flux_lora.py \\
        --prompt "SEM micrograph showing dendritic microstructure with primary arms and secondary branches. Process: DED." \\
        --lora_path /home/cdacapp01/wk-ganesh/multi-modal/checkpoints_flux_lora_filtered_flux/final

    # batch of prompts from a text file (one prompt per line)
    python infer_flux_lora.py \\
        --prompts_file prompts.txt \\
        --lora_path .../checkpoints_flux_lora_filtered_flux/best \\
        --num_images 4 --merge

    # compare base model vs LoRA on the same prompt
    python infer_flux_lora.py --prompt "..." --lora_path .../final --compare_base
"""

import os
import argparse
import logging

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Matches the paths used in train_flux_lora_multigpu.py
DEFAULT_FLUX_PATH = "/home/cdacapp01/wk-ganesh/multi-modal/models/FLUX.1-dev"
DEFAULT_BASE_DIR  = "/home/cdacapp01/wk-ganesh/multi-modal"
DEFAULT_CKPT_DIR  = os.path.join(DEFAULT_BASE_DIR, "checkpoints_flux_lora_filtered_flux")
DEFAULT_LORA_PATH = os.path.join(DEFAULT_CKPT_DIR, "final")   # falls back to "best" if "final" doesn't exist yet

DEFAULT_PROMPT = (
    "SEM micrograph showing cellular microstructure with hexagonal cell walls, "
    "bright boundaries, dark interiors. Process: LPBF, 285W, 960mm/s."
)


def parse_args():
    p = argparse.ArgumentParser(description="Run inference with a FLUX.1-dev LoRA adapter")

    # Model paths
    p.add_argument("--model_path", type=str, default=DEFAULT_FLUX_PATH,
                    help="Path to base FLUX.1-dev diffusers checkpoint")
    p.add_argument("--lora_path", type=str, default=DEFAULT_LORA_PATH,
                    help="Path to trained LoRA adapter dir (default: .../checkpoints_flux_lora_filtered_flux/final)")

    # Prompts
    p.add_argument("--prompt", type=str, default=None,
                    help=f"Single prompt to generate (default if nothing else given: \"{DEFAULT_PROMPT[:50]}...\")")
    p.add_argument("--prompts_file", type=str, default=None,
                    help="Text file with one prompt per line (generates all of them)")

    # Generation params
    p.add_argument("--num_images", type=int, default=1,
                    help="Number of images per prompt")
    p.add_argument("--steps", type=int, default=30, help="Inference steps")
    p.add_argument("--guidance_scale", type=float, default=3.5,
                    help="Guidance scale (matches GUIDANCE_SCALE used at train time by default)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--max_sequence_length", type=int, default=256,
                    help="T5 sequence length — match MAX_T5_SEQ_LEN from training")
    p.add_argument("--seed", type=int, default=42)

    # Adapter behavior
    p.add_argument("--merge", action="store_true",
                    help="Merge LoRA into the transformer before inference (see docstring)")
    p.add_argument("--lora_scale", type=float, default=1.0,
                    help="LoRA blend strength when NOT merging (0.0 = base model, 1.0 = full adapter)")
    p.add_argument("--compare_base", action="store_true",
                    help="Also generate the same prompt(s) with the base model (no LoRA) for comparison")

    # Output
    p.add_argument("--output_dir", type=str, default="./inference_outputs",
                    help="Where to save generated images")

    return p.parse_args()


def load_prompts(args):
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts.extend([line.strip() for line in f if line.strip()])

    if not prompts:
        # No prompt given via CLI — ask interactively instead of silently using a default.
        user_input = input(
            f'Enter a prompt (or press Enter to use the default: "{DEFAULT_PROMPT[:60]}..."): '
        ).strip()
        prompts.append(user_input if user_input else DEFAULT_PROMPT)

    return prompts


def resolve_lora_path(args):
    """If the default 'final' checkpoint doesn't exist yet (e.g. mid-training run),
    fall back to 'best', and error clearly if neither exists."""
    path = args.lora_path
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_model.safetensors")):
        return path

    if path == DEFAULT_LORA_PATH:
        fallback = os.path.join(DEFAULT_CKPT_DIR, "best")
        if os.path.isdir(fallback) and os.path.exists(os.path.join(fallback, "adapter_model.safetensors")):
            logger.info(f"'{path}' not found — falling back to '{fallback}'")
            return fallback

    raise FileNotFoundError(
        f"No adapter_model.safetensors found under '{path}'. "
        f"Pass --lora_path pointing at a saved checkpoint dir "
        f"(e.g. {DEFAULT_CKPT_DIR}/final or {DEFAULT_CKPT_DIR}/best)."
    )


def build_pipeline_merged(args, device, dtype):
    """Merge-and-unload mode: load base transformer, merge LoRA weights in, load into pipeline."""
    logger.info("Loading base FLUX transformer...")
    transformer_base = FluxTransformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=dtype,
    ).to(device)

    logger.info(f"Loading LoRA adapter from {args.lora_path} and merging...")
    transformer_lora = PeftModel.from_pretrained(transformer_base, args.lora_path)
    transformer_merged = transformer_lora.merge_and_unload()
    transformer_merged = transformer_merged.to(device, dtype=dtype)

    logger.info("Building FluxPipeline with merged transformer...")
    pipe = FluxPipeline.from_pretrained(
        args.model_path, transformer=transformer_merged, torch_dtype=dtype,
    ).to(device)
    return pipe


def build_pipeline_adapter(args, device, dtype):
    """Adapter mode: load base pipeline, attach LoRA weights without merging.

    NOTE: we deliberately do NOT use pipe.load_lora_weights() here. That method
    tries to auto-detect which community LoRA naming convention a checkpoint
    uses, and can misidentify a plain PEFT-saved adapter (the format produced
    by get_peft_model(...).save_pretrained() in the training script) as some
    other convention, then crash trying to convert keys that don't exist.
    Loading directly through PeftModel sidesteps that detection entirely and
    is the same mechanism the training script's own inference test uses.
    """
    logger.info("Loading base FluxPipeline...")
    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=dtype).to(device)

    logger.info(f"Loading LoRA adapter from {args.lora_path} via PEFT...")
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_path)
    pipe.transformer = pipe.transformer.to(device, dtype=dtype)

    if args.lora_scale != 1.0:
        logger.info(f"Scaling LoRA contribution by {args.lora_scale}")
        # Each PEFT LoRA layer exposes a `scaling` dict keyed by adapter name
        # (scaling = lora_alpha / r by default); multiply it to blend strength.
        for module in pipe.transformer.modules():
            if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                for adapter_name in module.scaling:
                    module.scaling[adapter_name] = module.scaling[adapter_name] * args.lora_scale

    return pipe


def build_base_pipeline(args, device, dtype):
    """Plain base model, no LoRA — used for --compare_base."""
    logger.info("Loading base FluxPipeline (no LoRA, for comparison)...")
    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    return pipe


def generate(pipe, prompt, args, generator, out_dir, tag):
    for i in range(args.num_images):
        gen = torch.Generator(device=generator.device).manual_seed(args.seed + i)
        image = pipe(
            prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            max_sequence_length=args.max_sequence_length,
            generator=gen,
        ).images[0]

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt[:60])
        out_path = os.path.join(out_dir, f"{tag}_{safe_name}_{i+1}.png")
        image.save(out_path)
        logger.info(f"  Saved -> {out_path}")


def main():
    args = parse_args()
    args.lora_path = resolve_lora_path(args)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if device.type == "cpu":
        logger.warning(
            "No CUDA GPU visible — FLUX.1-dev is a ~12B parameter model and will be "
            "extremely slow or will run out of memory on CPU. If you're on a SLURM "
            "cluster, this usually means you're on a login node rather than a GPU "
            "compute node. Run this via `srun --gres=gpu:1 --pty python infer_flux_lora.py` "
            "or submit it as a batch job, the same way you launch training."
        )

    logger.info("=" * 60)
    logger.info("  FLUX.1-dev LoRA Inference")
    logger.info(f"  Device       : {device}")
    logger.info(f"  Base model   : {args.model_path}")
    logger.info(f"  LoRA adapter : {args.lora_path}")
    logger.info(f"  Mode         : {'merge' if args.merge else 'adapter (lora_scale=' + str(args.lora_scale) + ')'}")
    logger.info("=" * 60)

    prompts = load_prompts(args)
    logger.info(f"Loaded {len(prompts)} prompt(s)")

    generator = torch.Generator(device=device)

    if args.merge:
        pipe = build_pipeline_merged(args, device, dtype)
    else:
        pipe = build_pipeline_adapter(args, device, dtype)

    logger.info("Running generation...")
    for prompt in prompts:
        logger.info(f'Prompt: "{prompt}"')
        generate(pipe, prompt, args, generator, args.output_dir, tag="lora")

    if args.compare_base:
        # free the LoRA pipeline's transformer before loading a second full copy
        del pipe
        torch.cuda.empty_cache()

        base_pipe = build_base_pipeline(args, device, dtype)
        for prompt in prompts:
            logger.info(f'[base] Prompt: "{prompt}"')
            generate(base_pipe, prompt, args, generator, args.output_dir, tag="base")

    logger.info(f"\nDone. Images saved under {args.output_dir}")


if __name__ == "__main__":
    main()