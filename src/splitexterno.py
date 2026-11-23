from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

RAIZ = Path("data/externo/Concrete & Pavement Crack Dataset")


def main():
    # 1. Lista de todas as imagens externas com o respetivo rótulo
    rows = []
    for pasta in sorted(RAIZ.iterdir()):
        nome = pasta.name.lower()
        if pasta.is_dir() and nome in ("negative", "positive"):
            rotulo = 1 if nome == "positive" else 0
            for p in sorted(pasta.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rows.append((str(p), rotulo))
    df = pd.DataFrame(rows, columns=["path", "label"])

    # 2. Metade para seleção, metade para o resultado final (estratificado, semente fixa)
    selecao, final = train_test_split(
        df, test_size=0.5, stratify=df["label"], random_state=config.SEED
    )
    out = pd.concat([selecao.assign(split="selecao"), final.assign(split="final")])
    out.to_csv(config.ROOT / "data" / "externo_splits.csv", index=False)
    print(out.groupby(["split", "label"]).size().unstack())


if __name__ == "__main__":
    main()
