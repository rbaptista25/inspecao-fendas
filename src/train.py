import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from . import config
from .dataset import CrackDataset
from .models import SmallCNN

EPOCAS = 8
LOTE = 64
LR = 1e-3
LIMITE = None  # ex.: 640 para um teste rápido


def uma_epoca(modelo, loader, criterio, otimizador=None):
    """Percorre o loader uma vez. Treina se houver otimizador; senão só mede."""
    treino = otimizador is not None
    modelo.train(treino)
    perda_total, certas, total = 0.0, 0, 0
    with torch.set_grad_enabled(treino):
        for x, y in tqdm(loader, leave=False):
            saida = modelo(x)
            perda = criterio(saida, y)
            if treino:
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
            perda_total += perda.item() * x.size(0)
            certas += (saida.argmax(1) == y).sum().item()
            total += x.size(0)
    return perda_total / total, certas / total


def carregar(split, baralhar):
    ds = CrackDataset(split)
    if LIMITE:
        ds = Subset(ds, range(LIMITE))
    return DataLoader(ds, batch_size=LOTE, shuffle=baralhar, num_workers=2)


def main():
    torch.manual_seed(config.SEED)
    loader_treino = carregar("train", baralhar=True)
    loader_val = carregar("val", baralhar=False)

    modelo = SmallCNN()
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    melhor, historico = 0.0, []

    for epoca in range(1, EPOCAS + 1):
        perda_t, acc_t = uma_epoca(modelo, loader_treino, criterio, otimizador)
        perda_v, acc_v = uma_epoca(modelo, loader_val, criterio)
        historico.append({"epoca": epoca, "perda_treino": perda_t, "acc_treino": acc_t,
                          "perda_val": perda_v, "acc_val": acc_v})
        print(f"Época {epoca}/{EPOCAS}  treino: perda {perda_t:.4f} exat. {acc_t:.4f}"
              f"  |  validação: perda {perda_v:.4f} exat. {acc_v:.4f}")
        if acc_v > melhor:
            melhor = acc_v
            torch.save(modelo.state_dict(), config.MODELS_DIR / "cnn.pt")

    pd.DataFrame(historico).to_csv(config.RESULTS_DIR / "cnn_historico.csv", index=False)
    print("Melhor exatidão na validação:", round(melhor, 4))


if __name__ == "__main__":
    main()
