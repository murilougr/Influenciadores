#!pip install fasttext-langdetect

#!pip install numpy==1.26.4 --force-reinstall

#!wget -q https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

#!pip install langdetect

#!pip install emoji

import os
import pandas as pd
import fasttext
import re
import emoji
from google.colab import files  # para baixar o CSV no Colab

# Carregar modelo FastText
modelo = fasttext.load_model("lid.176.bin")

PASTA = "/content/unificado"
COLUNA = "TEXTO"

# ------------------------------------------
# FUNÇÃO PARA REMOVER EMOJIS
# ------------------------------------------
def remover_emojis(texto):
    return emoji.replace_emoji(texto, replace=' ')  # substitui emoji por espaço

# ------------------------------------------
# LIMPEZA LEVE
# ------------------------------------------
url_re = re.compile(r"https?://\S+|www\.\S+")
mention_re = re.compile(r"@\w+")
hashtag_re = re.compile(r"#\w+")

def limpar_texto(t):
    if not isinstance(t, str):
        return ""

    t = t.lower()

    # remover emojis com biblioteca emoji
    t = remover_emojis(t)

    # remoções leves
    t = url_re.sub(" ", t)
    t = mention_re.sub(" ", t)
    t = hashtag_re.sub(" ", t)

    # remover símbolos/pontuação
    t = re.sub(r"[^\w\s]", " ", t)

    # normalizar espaços
    t = re.sub(r"\s+", " ", t).strip()

    return t

# ------------------------------------------
# PROCESSAR CSVs
# ------------------------------------------
resultados = []
outros_registros = []  # registros de outros idiomas para arquivo separado

arquivos = [f for f in os.listdir(PASTA) if f.endswith(".csv")]

for arq in arquivos:
    caminho = os.path.join(PASTA, arq)

    try:
        df = pd.read_csv(caminho)
        if COLUNA not in df.columns:
            print(f"[AVISO] coluna {COLUNA} não encontrada em {arq}. Pulando…")
            continue

        textos = df[COLUNA].astype(str).tolist()

        textos_limpados = []
        idiomas = []

        for txt in textos:
            t_limpo = limpar_texto(txt)

            textos_limpados.append(t_limpo)

            try:
                label, prob = modelo.predict(t_limpo)
                idioma = label[0].replace("__label__", "")
            except:
                idioma = "erro"

            idiomas.append(idioma)

            # guardar outros idiomas
            if idioma not in ["pt", "en"]:
                outros_registros.append([
                    arq,
                    txt,
                    t_limpo,
                    idioma
                ])

        total_linhas = len(textos_limpados)

        pt_linhas = idiomas.count("pt")
        en_linhas = idiomas.count("en")
        outros_linhas = total_linhas - (pt_linhas + en_linhas)

        pt_percentual = (pt_linhas / total_linhas * 100) if total_linhas > 0 else 0
        en_percentual = (en_linhas / total_linhas * 100) if total_linhas > 0 else 0
        outros_percentual = (outros_linhas / total_linhas * 100) if total_linhas > 0 else 0

        resultados.append([
            arq,
            total_linhas,
            pt_linhas,
            round(pt_percentual, 2),
            en_linhas,
            round(en_percentual, 2),
            outros_linhas,
            round(outros_percentual, 2),
        ])

        # -----------------------------------------------------
        # 🚀 NOVA SEÇÃO: REMOVER OUTROS IDIOMAS E RECRIAR NUMERO
        # -----------------------------------------------------
        df["idioma"] = idiomas
        df["texto_limpo"] = textos_limpados

        # manter só pt e en
        df_filtrado = df[df["idioma"].isin(["pt", "en"])].copy()

        # recriar NUMERO se existir, ou criar caso não tenha
        df_filtrado["NUMERO"] = range(1, len(df_filtrado) + 1)

        # salvar CSV limpo
        caminho_limpo = os.path.join("/content", arq.replace(".csv", "_limpo.csv"))
        df_filtrado.to_csv(caminho_limpo, index=False, encoding="utf-8-sig")

        print(f"[OK] CSV limpo gerado: {caminho_limpo}")

    except Exception as e:
        print(f"Erro no arquivo {arq}: {e}")


# ------------------------------------------
# TABELA FINAL
# ------------------------------------------
tabela = pd.DataFrame(
    resultados,
    columns=[
        "arquivo",
        "total_linhas_pos_limpeza",
        "pt_linhas",
        "pt_percentual",
        "en_linhas",
        "en_percentual",
        "outros_idiomas_linhas",
        "outros_idiomas_percentual"
    ]
)

linha_total = {
    "arquivo": "TOTAL_GERAL",
    "total_linhas_pos_limpeza": tabela["total_linhas_pos_limpeza"].sum(),
    "pt_linhas": tabela["pt_linhas"].sum(),
    "pt_percentual": None,
    "en_linhas": tabela["en_linhas"].sum(),
    "en_percentual": None,
    "outros_idiomas_linhas": tabela["outros_idiomas_linhas"].sum(),
    "outros_idiomas_percentual": None
}

tabela = pd.concat([tabela, pd.DataFrame([linha_total])], ignore_index=True)
tabela.index = range(1, len(tabela) + 1)

# ------------------------------------------
# CSV DE OUTROS IDIOMAS
# ------------------------------------------
if outros_registros:
    tabela_outros = pd.DataFrame(
        outros_registros,
        columns=[
            "arquivo",
            "texto_original",
            "texto_limpo",
            "idioma_detectado"
        ]
    )
    tabela_outros.index = range(1, len(tabela_outros) + 1)

    csv_path = "/content/outros_idiomas_textos.csv"
    tabela_outros.to_csv(csv_path, index=False, encoding="utf-8")
    files.download(csv_path)
else:
    print("Nenhum texto classificado como outro idioma (diferente de pt/en).")

tabela