#!pip -q install emoji

import os
import glob
import pandas as pd
import emoji

PASTA = "/content/drive/MyDrive/Brasil"
arquivos = sorted(glob.glob(os.path.join(PASTA, "*.csv")))[:20]

if not arquivos:
    raise FileNotFoundError(f"Nenhum CSV encontrado em: {PASTA}")

def tem_emoji(texto: str) -> bool:
    """Retorna True se o texto tiver pelo menos um emoji."""
    if pd.isna(texto):
        return False
    return len(emoji.emoji_list(str(texto))) > 0

def contar_por_categoria(df: pd.DataFrame, categoria: str) -> int:
    """Conta linhas onde CATEGORIA == categoria e TEXTO contém ao menos um emoji."""
    # Normaliza para evitar problemas de maiúsculas/minúsculas e espaços
    cat = df["CATEGORIA"].astype(str).str.strip().str.lower()
    mask_cat = (cat == categoria.lower())

    # TEXTO com emoji
    texto = df["TEXTO"]
    mask_emoji = texto.apply(tem_emoji)

    return int((mask_cat & mask_emoji).sum())

resultados = []

for i, arq in enumerate(arquivos, start=1):
    try:
        # Tente inferir separador; se você souber que é sempre "," ou ";", pode fixar
        df = pd.read_csv(arq, sep=None, engine="python")

        # Checar colunas obrigatórias
        cols = {c.strip(): c for c in df.columns}
        if "TEXTO" not in cols or "CATEGORIA" not in cols:
            raise ValueError(f"Colunas esperadas não encontradas. Achei: {list(df.columns)}")

        # Se tiver variações de nomes/espacos, alinha nomes exatamente:
        if cols["TEXTO"] != "TEXTO" or cols["CATEGORIA"] != "CATEGORIA":
            df = df.rename(columns={cols["TEXTO"]: "TEXTO", cols["CATEGORIA"]: "CATEGORIA"})

        resultados.append({
            "seq": i,
            "arquivo": os.path.basename(arq),
            "legenda_com_emoji": contar_por_categoria(df, "legenda"),
            "transcricao_com_emoji": contar_por_categoria(df, "transcricao"),
        })

    except Exception as e:
        # Se algum arquivo der problema, registra e segue (opcional)
        resultados.append({
            "seq": i,
            "arquivo": os.path.basename(arq),
            "legenda_com_emoji": None,
            "transcricao_com_emoji": None,
            "erro": str(e),
        })

tabela_final = pd.DataFrame(resultados)
tabela_final