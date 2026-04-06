import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MicroDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item["text_input"]
        image = Image.open(item["image"]).convert("RGB")
        image = self.transform(image)

        return text, image
