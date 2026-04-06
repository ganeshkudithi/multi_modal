import torch.nn as nn
from models.projector import OutputProjector
from models.llm import LLMWrapper

class TextToImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = LLMWrapper()
        self.projector = OutputProjector()

    def forward(self, text):
        llm_features = self.llm.encode(text)
        latent = self.projector(llm_features)
        return latent
