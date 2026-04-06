from diffusers import AutoencoderKL
import torch

class VAEWrapper:
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        ).cuda()

    def encode(self, image):
        image = image.unsqueeze(0).cuda()
        latent = self.vae.encode(image).latent_dist.sample()
        return latent

    def decode(self, latent):
        image = self.vae.decode(latent).sample
        return image
