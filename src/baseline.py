import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from . import config

LIMITE = None  # põe por exemplo 1000 para um teste rápido


def carregar(split):
    """Devolve (características HOG, rótulos) de um conjunto: 'train', 'val' ou 'test'."""
    df = pd.read_csv(config.SPLITS_CSV)
    df = df[df["split"] == split]
    if LIMITE:
        df = df.sample(LIMITE, random_state=config.SEED)
    X = []
    for p in tqdm(df["path"], desc=split):
        img = Image.open(config.DATA_RAW / p).convert("L").resize((64, 64))
        X.append(hog(np.asarray(img), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
    return np.array(X), df["label"].to_numpy()


def main():
    X_train, y_train = carregar("train")
    X_val, y_val = carregar("val")
    print("Forma das características de treino:", X_train.shape)

    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_val)
    acc = accuracy_score(y_val, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, pred, average="binary", pos_label=1)
    print(f"Validação: exatidão {acc:.4f}  precisão {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    print(confusion_matrix(y_val, pred))
    # Avaliação final no teste (a baseline já está fechada)
    X_test, y_test = carregar("test")
    pred_t = modelo.predict(X_test)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred_t, average="binary", pos_label=1)
    print(f"Teste: exatidão {accuracy_score(y_test, pred_t):.4f}  precisão {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
    print(confusion_matrix(y_test, pred_t))

if __name__ == "__main__":
    main()
