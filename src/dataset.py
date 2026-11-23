import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import config

TAMANHO = 224
MEDIA = (0.485, 0.456, 0.406)
DESVIO = (0.229, 0.224, 0.225)


def transformacoes(treino):
    normalizar = [T.ToTensor(), T.Normalize(MEDIA, DESVIO)]
    if treino:
        return T.Compose([
            T.Resize((TAMANHO, TAMANHO)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            *normalizar,
        ])
    return T.Compose([T.Resize((TAMANHO, TAMANHO)), *normalizar])


class CrackDataset(Dataset):
    def __init__(self, split):
        df = pd.read_csv(config.SPLITS_CSV)
        df = df[df["split"] == split]
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transformacoes(treino=(split == "train"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(config.DATA_RAW / self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]
