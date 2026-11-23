import pandas as pd
from pathlib import Path
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

class PastaDataset(Dataset):
    """Imagens em <raiz>/negative e <raiz>/positive (maiúsculas ou minúsculas), para testes externos."""

    def __init__(self, raiz, redimensionar=True):
        self.paths, self.labels = [], []
        for pasta in sorted(Path(raiz).iterdir()):
            nome = pasta.name.lower()
            if pasta.is_dir() and nome in ("negative", "positive"):
                rotulo = 1 if nome == "positive" else 0
                for p in sorted(pasta.iterdir()):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        self.paths.append(p)
                        self.labels.append(rotulo)
        if not self.paths:
            raise FileNotFoundError(f"Não encontrei pastas positive/negative com imagens em {raiz}")
        if redimensionar:
            self.transform = transformacoes(treino=False)
        else:
            self.transform = T.Compose([T.ToTensor(), T.Normalize(MEDIA, DESVIO)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]
