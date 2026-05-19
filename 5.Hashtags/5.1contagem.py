import os
import pandas as pd
import unicodedata

pasta = "/content/drive/MyDrive/Brasil"
arquivos = [f for f in os.listdir(pasta) if f.endswith(".csv")]

def normalizar(txt):
    txt = "" if pd.isna(txt) else str(txt)
    txt = txt.strip().lower()
    # remove acentos (ex.: "legenda"/"legêndá" etc.)
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("utf-8")
    return txt

resultados = []
seq = 1

total_legendas_geral = 0
total_legendas_hash_geral = 0

for nome_arquivo in arquivos:
    caminho = os.path.join(pasta, nome_arquivo)

    try:
        df = pd.read_csv(caminho)

        if "TEXTO" not in df.columns or "CATEGORIA" not in df.columns:
            print(f"[AVISO] Colunas necessárias não encontradas em {nome_arquivo}")
            continue

        # normaliza categoria e texto
        cat = df["CATEGORIA"].apply(normalizar)
        texto = df["TEXTO"].astype(str)

        # foca só em legenda
        legenda_df = df[cat == "legenda"].copy()
        total_legendas = len(legenda_df)

        # dentro de legenda, quantas têm #
        if total_legendas > 0:
            total_legendas_hash = legenda_df["TEXTO"].astype(str).str.contains("#", na=False).sum()
            pct_legendas_hash = (total_legendas_hash / total_legendas) * 100
        else:
            total_legendas_hash = 0
            pct_legendas_hash = 0.0

        resultados.append([seq, nome_arquivo, total_legendas, pct_legendas_hash])

        total_legendas_geral += total_legendas
        total_legendas_hash_geral += int(total_legendas_hash)

        seq += 1

    except Exception as e:
        print(f"Erro no arquivo {nome_arquivo}: {e}")

# linha TOTAL
pct_total = (total_legendas_hash_geral / total_legendas_geral) * 100 if total_legendas_geral > 0 else 0.0
resultados.append(["", "TOTAL", total_legendas_geral, pct_total])

tabela_final = pd.DataFrame(
    resultados,
    columns=["seq", "arquivo", "qtd_legendas", "%_legendas_com_#"]
)

# opcional: arredondar a % pra 2 casas
tabela_final["%_legendas_com_#"] = pd.to_numeric(tabela_final["%_legendas_com_#"], errors="coerce").round(2)

tabela_final