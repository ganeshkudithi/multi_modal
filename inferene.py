import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel


SD_PATH         = "/home/cdacapp01/wk-ganesh/multi-modal/checkpoints_diffusion_sd3_lora/epoch_25"

LORA_PATH = "/home/cdacapp01/wk-ganesh/multi-modal/checkpoints_diffusion_lora_filtered/final"

OUTPUT_DIR = "inference_generated_images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

os.makedirs(OUTPUT_DIR, exist_ok=True)



print("Loading base Stable Diffusion...")

unet = UNet2DConditionModel.from_pretrained(
    SD_PATH,
    subfolder="unet",
    torch_dtype=DTYPE,
)



print("Loading trained LoRA...")

unet = PeftModel.from_pretrained(
    unet,
    LORA_PATH,
)

print("Merging LoRA...")

unet = unet.merge_and_unload()

unet = unet.to(DEVICE, dtype=DTYPE)



pipe = StableDiffusionPipeline.from_pretrained(
    SD_PATH,
    unet=unet,
    torch_dtype=DTYPE,
    safety_checker=None,
)

pipe = pipe.to(DEVICE)

# Optional (PyTorch 2.x)
pipe.set_progress_bar_config(disable=False)



prompts = [

    "LPBF metal with  equiaxed microstructure. Laser power 200 W. Heat input 13.3 J/mm. 100 µm scale bar.",
]

# ============================================================
# Generate Images
# ============================================================

generator = torch.Generator(device=DEVICE).manual_seed(54)

for idx, prompt in enumerate(prompts):

    print(f"\nGenerating image {idx+1}")

    image = pipe(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
        height=512,
        width=512,
    ).images[0]

    save_path = os.path.join(OUTPUT_DIR, f"sample_image_100_bottom.png")

    image.save(save_path)

    print(f"Saved: {save_path}")

print("\nInference complete.")
