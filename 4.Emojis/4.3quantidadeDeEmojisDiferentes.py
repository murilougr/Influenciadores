import os, glob
import pandas as pd
import emoji

PASTA = "/content/drive/MyDrive/Brasil"
arquivos = sorted(glob.glob(os.path.join(PASTA, "*.csv")))[:20]

if not arquivos:
    raise FileNotFoundError(f"Nenhum CSV encontrado em: {PASTA}")

def tem_emoji(texto: str) -> bool:
    if pd.isna(texto):
        return False
    return len(emoji.emoji_list(str(texto))) > 0

resultados = []
total_legendas = 0
total_legendas_com_emoji = 0

for i, arq in enumerate(arquivos, start=1):
    try:
        df = pd.read_csv(arq, sep=None, engine="python")

        # Checar/normalizar colunas
        cols = {c.strip(): c for c in df.columns}
        if "TEXTO" not in cols or "CATEGORIA" not in cols:
            raise ValueError(f"Colunas esperadas não encontradas. Achei: {list(df.columns)}")

        if cols["TEXTO"] != "TEXTO" or cols["CATEGORIA"] != "CATEGORIA":
            df = df.rename(columns={cols["TEXTO"]: "TEXTO", cols["CATEGORIA"]: "CATEGORIA"})

        cat = df["CATEGORIA"].astype(str).str.strip().str.lower()
        mask_legenda = (cat == "legenda")

        n_legendas = int(mask_legenda.sum())
        n_legendas_com_emoji = int(df.loc[mask_legenda, "TEXTO"].apply(tem_emoji).sum())

        perc = (n_legendas_com_emoji / n_legendas * 100) if n_legendas > 0 else 0.0

        resultados.append({
            "seq": i,
            "arquivo": os.path.basename(arq),
            "legenda_com_emoji": n_legendas_com_emoji,
            "%_legendas_com_emoji": round(perc, 2),
        })

        total_legendas += n_legendas
        total_legendas_com_emoji += n_legendas_com_emoji

    except Exception as e:
        resultados.append({
            "seq": i,
            "arquivo": os.path.basename(arq),
            "legenda_com_emoji": None,
            "%_legendas_com_emoji": None,
            "erro": str(e),
        })

from collections import Counter

contador_emojis = Counter()
exemplos = {}

for arq in arquivos:
    try:
        df = pd.read_csv(arq, sep=None, engine="python")

        # Normalizar colunas
        cols = {c.strip(): c for c in df.columns}
        df = df.rename(columns={
            cols.get("TEXTO", "TEXTO"): "TEXTO",
            cols.get("CATEGORIA", "CATEGORIA"): "CATEGORIA"
        })

        # Filtrar legendas
        cat = df["CATEGORIA"].astype(str).str.strip().str.lower()
        df_legendas = df[cat == "legenda"]

        for texto in df_legendas["TEXTO"].dropna():
            lista = emoji.emoji_list(str(texto))

            for item in lista:
                emj = item["emoji"]
                contador_emojis[emj] += 1

                # Guardar um exemplo (primeiro que aparecer)
                if emj not in exemplos:
                    exemplos[emj] = texto

    except Exception as e:
        print(f"Erro em {arq}: {e}")

# Top 10 emojis
top10 = contador_emojis.most_common(10)

# Criar tabela final
resultado_final = pd.DataFrame([
    {
        "emoji": emj,
        "quantidade": qtd,
        "exemplo": exemplos.get(emj, "")[:120]  # corta texto longo
    }
    for emj, qtd in top10
])

resultado_final