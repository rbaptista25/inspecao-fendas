import matplotlib
matplotlib.use("Agg")  # desenhar para ficheiro, sem abrir janelas
import matplotlib.pyplot as plt
from PIL import Image

from . import config


def main():
    # 1. Contar as imagens de cada classe
    files = {}
    for cls in config.CLASSES:
        files[cls] = sorted((config.DATA_RAW / cls).glob("*.jpg"))
        print(f"{cls}: {len(files[cls])} imagens")

    # 2. Tamanho e modo de cor da primeira imagem de cada classe
    for cls in config.CLASSES:
        im = Image.open(files[cls][0])
        print(cls, im.size, im.mode)

    # 3. Grelha com 8 exemplos de cada classe, espaçados pela lista
    n = 8
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for row, cls in enumerate(config.CLASSES):
        for col in range(n):
            axes[row][col].imshow(Image.open(files[cls][col * 500]))
            axes[row][col].axis("off")
        axes[row][0].set_title(cls, loc="left")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(config.RESULTS_DIR / "exploracao_amostras.png", dpi=120)
    print("Grelha guardada em results/exploracao_amostras.png")


if __name__ == "__main__":
    main()
