import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from . import config


def main():
    erros = pd.read_csv(config.RESULTS_DIR / "cnn_teste_erros.csv")
    colunas = 4
    linhas = (len(erros) + colunas - 1) // colunas
    fig, eixos = plt.subplots(linhas, colunas, figsize=(3 * colunas, 3.4 * linhas), squeeze=False)
    for ax in eixos.flat:
        ax.axis("off")
    for ax, (_, e) in zip(eixos.flat, erros.iterrows()):
        ax.imshow(Image.open(config.DATA_RAW / e["path"]))
        real = config.CLASSES[e["label"]]
        previsto = config.CLASSES[e["pred"]]
        ax.set_title(f"real: {real}\nprevisto: {previsto} ({e['prob_fenda']:.2f})", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "cnn_teste_erros.png", dpi=130)
    print("Guardado em results/cnn_teste_erros.png")


if __name__ == "__main__":
    main()
