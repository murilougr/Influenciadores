import os, glob
import pandas as pd
import emoji

# Pasta e 20 CSVs
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

tabela = pd.DataFrame(resultados)

# Linha TOTAL (considera o total de legendas de todos os arquivos)
perc_total = (total_legendas_com_emoji / total_legendas * 100) if total_legendas > 0 else 0.0
linha_total = pd.DataFrame([{
    "seq": "TOTAL",
    "arquivo": f"{len(arquivos)} arquivos",
    "legenda_com_emoji": total_legendas_com_emoji,
    "%_legendas_com_emoji": round(perc_total, 2),
}])

tabela_final = pd.concat([tabela, linha_total], ignore_index=True)
tabela_final