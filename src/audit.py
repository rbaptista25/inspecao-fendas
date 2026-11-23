import hashlib
import pandas as pd
from collections import Counter

from PIL import Image

from . import config


def main():
    problems = []      # imagens que não abrem
    sizes = Counter()  # (tamanho, modo de cor) -> quantas imagens
    hashes = {}        # impressão digital -> lista de ficheiros com esse conteúdo

    for cls in config.CLASSES:
        for path in sorted((config.DATA_RAW / cls).glob("*.jpg")):
            try:
                with Image.open(path) as im:
                    im.verify()               # verifica se o ficheiro está íntegro
                with Image.open(path) as im:  # reabrir, porque o verify() invalida a imagem
                    sizes[(im.size, im.mode)] += 1
            except Exception as e:
                problems.append((str(path), str(e)))
                continue
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            hashes.setdefault(digest, []).append(f"{cls}/{path.name}")

    print("Imagens que não abrem:", len(problems))
    for p in problems[:10]:
        print("  ", p)
    print("Tamanhos e modos de cor:", dict(sizes))

    duplicados = [files for files in hashes.values() if len(files) > 1]
    print("Grupos de ficheiros idênticos:", len(duplicados))
    conflitos = [f for f in duplicados if len({x.split("/")[0] for x in f}) > 1]
    print("Idênticos com etiquetas diferentes:", len(conflitos))
    for f in conflitos[:10]:
        print("  ", f)
    extra = sum(len(f) - 1 for f in duplicados)
    print("Cópias redundantes:", extra)
    por_classe = Counter(f[0].split("/")[0] for f in duplicados)
    print("Grupos por classe:", dict(por_classe))
    print("Maior grupo:", max(len(f) for f in duplicados), "ficheiros")
    for f in sorted(duplicados, key=len, reverse=True)[:3]:
        print("  ", f)

    # Quantos grupos têm cópias em conjuntos diferentes (fuga direta)?
    df = pd.read_csv(config.SPLITS_CSV).set_index("path")
    atravessam = sum(1 for f in duplicados if df.loc[f, "split"].nunique() > 1)
    print("Grupos com cópias em conjuntos diferentes:", atravessam)


if __name__ == "__main__":
    main()

