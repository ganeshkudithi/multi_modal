import torch
from torch.utils.data import DataLoader
from dataset import MicroDataset
from models.model import TextToImageModel
from utils import VAEWrapper
import torch.nn as nn
import config

# Load components
dataset = MicroDataset("data/dataset.json")
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

model = TextToImageModel().cuda()
vae = VAEWrapper()

optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
criterion = nn.MSELoss()

# Training loop
for epoch in range(config.EPOCHS):
    for text, image in dataloader:

        image = image.cuda()

        # Step 1: Encode real image → latent
        with torch.no_grad():
            true_latent = vae.encode(image)

        # Step 2: Predict latent from text
        pred_latent = model(text)

        # Reshape
        pred_latent = pred_latent.view_as(true_latent)

        # Step 3: Compute loss
        loss = criterion(pred_latent, true_latent)

        # Step 4: Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")
