import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from . import config


def main():
    # python -m src.erros           -> erros do teste original
    # python -m src.erros externo   -> amostra de erros do teste externo
    nome = sys.argv[1] if len(sys.argv) > 1 else "teste"
    erros = pd.read_csv(config.RESULTS_DIR / f"cnn_{nome}_erros.csv")
    erros = erros.sample(min(12, len(erros)), random_state=config.SEED)

    colunas = 4
    linhas = (len(erros) + colunas - 1) // colunas
    fig, eixos = plt.subplots(linhas, colunas, figsize=(3 * colunas, 3.4 * linhas), squeeze=False)
    for ax in eixos.flat:
        ax.axis("off")
    for ax, (_, e) in zip(eixos.flat, erros.iterrows()):
        caminho = config.DATA_RAW / e["path"] if nome == "teste" else e["path"]
        ax.imshow(Image.open(caminho))
        real = config.CLASSES[e["label"]]
        previsto = config.CLASSES[e["pred"]]
        ax.set_title(f"real: {real}\nprevisto: {previsto} ({e['prob_fenda']:.2f})", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"cnn_{nome}_erros.png", dpi=130)
    print(f"Guardado em results/cnn_{nome}_erros.png")


if __name__ == "__main__":
    main()
