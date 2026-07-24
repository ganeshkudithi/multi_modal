import os
import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from peft import PeftModel

# ── Paths ──────────────────────────────────────────────
# Base SD3 model — same path used as SD3_PATH in train_diffusion_sd3_lora.py
SD3_PATH = "/home/cdacapp01/wk-ganesh/multi-modal/models/stable-diffusion-3-medium"

# The LoRA adapter checkpoint (NOT the base model). Point this at whichever
# checkpoint you want: .../final, .../best, or .../epoch_25 — all are valid,
# they're just adapter weights saved at different points in training.
LORA_PATH = "/home/cdacapp01/wk-ganesh/multi-modal/checkpoints_diffusion_sd3_lora/epoch_25"

OUTPUT_DIR = "inference_generated_images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

if DEVICE == "cpu":
    print("WARNING: no CUDA GPU visible — SD3 will be very slow / may OOM on CPU. "
          "Run this on a GPU compute node, not a login node.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load base transformer, then merge LoRA in ───────────
print("Loading base SD3 transformer...")
transformer = SD3Transformer2DModel.from_pretrained(
    SD3_PATH,
    subfolder="transformer",
    torch_dtype=DTYPE,
)

print(f"Loading trained LoRA from {LORA_PATH} ...")
transformer = PeftModel.from_pretrained(transformer, LORA_PATH)

print("Merging LoRA into transformer...")
transformer = transformer.merge_and_unload()
transformer = transformer.to(DEVICE, dtype=DTYPE)

# ── Build full SD3 pipeline (loads its own tokenizers / 3 text encoders / vae) ──
print("Loading SD3 pipeline (this pulls in tokenizer x3, text_encoder x3, vae)...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    SD3_PATH,
    transformer=transformer,
    torch_dtype=DTYPE,
)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=False)

# ── Prompts ──────────────────────────────────────────────
prompts = [
    "LPBF metal with equiaxed microstructure. Laser power 200 W. Heat input 13.3 J/mm. 100 µm scale bar.",
]

# ── Generate ──────────────────────────────────────────────
generator = torch.Generator(device=DEVICE).manual_seed(54)

for idx, prompt in enumerate(prompts):
    print(f"\nGenerating image {idx+1}")

    image = pipe(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.0,
        generator=generator,
        height=512,
        width=512,
        max_sequence_length=256,   # match MAX_T5_SEQ_LEN used in training
    ).images[0]

    save_path = os.path.join(OUTPUT_DIR, f"sample_image_{idx+1}.png")
    image.save(save_path)
    print(f"Saved: {save_path}")

print("\nInference complete.")
