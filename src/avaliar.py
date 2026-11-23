import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from . import config
from .dataset import CrackDataset, PastaDataset
from .models import SmallCNN


def main():
    # Sem argumentos avalia no teste; com uma pasta, avalia nessa pasta (teste externo)
    if len(sys.argv) > 1:
        ds, nome = PastaDataset(sys.argv[1]), "externo"
    else:
        ds, nome = CrackDataset("test"), "teste"
    print(f"A avaliar '{nome}' com {len(ds)} imagens")

    modelo = SmallCNN()
    modelo.load_state_dict(torch.load(config.MODELS_DIR / "cnn.pt"))
    modelo.eval()

    loader = DataLoader(ds, batch_size=64, num_workers=2)
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs.append(torch.softmax(modelo(x), dim=1)[:, 1].numpy())
    prob = np.concatenate(probs)
    pred = (prob >= 0.5).astype(int)
    y = np.array(ds.labels)

    acc = accuracy_score(y, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1)
    print(f"{nome}: exatidão {acc:.4f}  precisão {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    print(confusion_matrix(y, pred))

    erros = pd.DataFrame({"path": ds.paths, "label": y, "pred": pred, "prob_fenda": prob})
    erros = erros[erros["label"] != erros["pred"]]
    erros.to_csv(config.RESULTS_DIR / f"cnn_{nome}_erros.csv", index=False)
    print("Erros:", len(erros))


if __name__ == "__main__":
    main()
