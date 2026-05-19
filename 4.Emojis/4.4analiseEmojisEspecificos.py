import os, glob
import pandas as pd
import emoji
from collections import Counter, defaultdict

PASTA = "/content/drive/MyDrive/Brasil"
arquivos = sorted(glob.glob(os.path.join(PASTA, "*.csv")))[:20]

# 🔎 Lista de emojis que você quer analisar
emojis_interesse = [
"🔥","🌈","😭","🤡","🚨","😢","😡","❌","😓","💔","🤢","🤤","💸","▶️",
"😏","💣","😲","👩‍❤️‍👨","✝️","🚫","🆘","☹️","💑","😨","💊","👎",
"🙁","😪","👩‍❤️‍💋‍👨","🤬","😠","👺","🫦","✖️","🌶","👹",
"👨‍👨‍👧‍👧","😕","👄","👫🏻","🍑","☹","👩🏻‍❤️‍💋‍👨🏻","🐒","⏪","🔽"
]

# Estruturas para armazenar dados
contador = Counter()
legendas_por_emoji = defaultdict(list)

def extrair_emojis(texto):
    if pd.isna(texto):
        return []
    return [d["emoji"] for d in emoji.emoji_list(str(texto))]

for arq in arquivos:
    df = pd.read_csv(arq, sep=None, engine="python")

    cols = {c.strip(): c for c in df.columns}
    df = df.rename(columns={
        cols.get("TEXTO","TEXTO"): "TEXTO",
        cols.get("CATEGORIA","CATEGORIA"): "CATEGORIA"
    })

    mask_legenda = df["CATEGORIA"].astype(str).str.strip().str.lower() == "legenda"

    for texto in df.loc[mask_legenda, "TEXTO"].dropna():
        emojis_encontrados = extrair_emojis(texto)

        for e in emojis_encontrados:
            if e in emojis_interesse:
                contador[e] += 1
                legendas_por_emoji[e].append(texto)

# Criar tabela final
dados = []

for e in emojis_interesse:
    dados.append({
        "emoji": e,
        "quantidade": contador[e],
        "legendas": legendas_por_emoji[e]  # lista completa de legendas
    })

df_resultado = pd.DataFrame(dados).sort_values(by="quantidade", ascending=False).reset_index(drop=True)

df_resultado