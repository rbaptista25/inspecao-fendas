import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config


def main():
    h = pd.read_csv(config.RESULTS_DIR / "cnn_historico.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))

    a1.plot(h["epoca"], h["perda_treino"], label="treino")
    a1.plot(h["epoca"], h["perda_val"], label="validação")
    a1.set_xlabel("época")
    a1.set_ylabel("perda")
    a1.legend()

    a2.plot(h["epoca"], h["acc_treino"], label="treino")
    a2.plot(h["epoca"], h["acc_val"], label="validação")
    a2.set_xlabel("época")
    a2.set_ylabel("exatidão")
    a2.legend()

    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "cnn_curvas.png", dpi=150)
    print("Guardado em results/cnn_curvas.png")


if __name__ == "__main__":
    main()
