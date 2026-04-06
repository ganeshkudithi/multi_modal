import torch.nn as nn

class OutputProjector(nn.Module):
    def __init__(self, llm_dim=4096, latent_dim=4*64*64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(llm_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)
