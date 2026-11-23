import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def main():
    # 1. Tabela com todas as imagens e o respetivo rótulo
    rows = []
    for label, cls in enumerate(config.CLASSES):
        for path in sorted((config.DATA_RAW / cls).glob("*.jpg")):
            rows.append((f"{cls}/{path.name}", label))
    df = pd.DataFrame(rows, columns=["path", "label"])

    # 2. Tirar 15 % para teste
    train_val, test = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=config.SEED
    )
    # 3. Do que sobra, tirar a validação: 15 % do total = 0.15 / 0.85 do restante
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, stratify=train_val["label"], random_state=config.SEED
    )

    # 4. Juntar numa só tabela com a coluna 'split' e guardar
    out = pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    )
    out.to_csv(config.SPLITS_CSV, index=False)
    print(out.groupby(["split", "label"]).size().unstack())


if __name__ == "__main__":
    main()
