import re
import pandas as pd
import os
from glob import glob
from pathlib import Path

# ========== REGEX PARA TOKENIZAR ==========
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", flags=re.UNICODE)

# ========== FUNÇÃO LIWC COM PORCENTAGEM ==========
def liwc_process_macro_pct(text, categories, dictionary, macro_ids):
    text = str(text).lower()
    tokens = TOKEN_RE.findall(text)
    total = len(tokens)

    # Inicializa contagens absolutas
    counts = { categories[mid]: 0 for mid in macro_ids }

    # Conta tokens por categoria macro
    for token in tokens:
        for pattern, cat_ids in dictionary.items():

            # --- radical (*)
            if "*" in pattern:
                root = pattern.replace("*", "")
                if token.startswith(root):
                    for c in cat_ids:
                        if c in macro_ids:
                            counts[categories[c]] += 1

            # --- palavra exata
            else:
                if token == pattern:
                    for c in cat_ids:
                        if c in macro_ids:
                            counts[categories[c]] += 1

    # ======== Transformar contagem em % ========
    results = {}
    for cat in counts:
        if total > 0:
            results[cat] = (counts[cat] * 100) / total
        else:
            results[cat] = 0.0

    return results


# ============================================================
# ========= APLICAR LIWC EM CADA CSV INDIVIDUALMENTE =========
# ============================================================

# Caminho da pasta CSV no servidor
pasta_csvs = Path.home() / "work" / "Untitled Folder" / "Teste" / "csv"

col_text = "TEXTO"

# Lista todos os CSVs
arquivos = glob(str(pasta_csvs / "*.csv"))

print(f"📂 {len(arquivos)} arquivos encontrados!")

resultados = []  # cada item = uma linha no CSV final

for arq in arquivos:
    nome_arquivo = os.path.basename(arq)

    try:
        df = pd.read_csv(arq)

        if col_text not in df.columns:
            print(f"⚠️ Coluna TEXTO ausente em {nome_arquivo}")
            continue

        # --- juntar o texto do arquivo inteiro ---
        texto_unico = " ".join(df[col_text].astype(str).tolist())

        # --- aplicar LIWC ---
        resultado = liwc_process_macro_pct(texto_unico, categories, dictionary, macro_ids_28)

        # adicionar nome do arquivo
        resultado_final = {"arquivo": nome_arquivo}
        resultado_final.update(resultado)

        resultados.append(resultado_final)

        print(f"✔️ Processado: {nome_arquivo}")

    except Exception as e:
        print(f"❌ Erro ao processar {nome_arquivo}: {e}")


# ======= Criar DF final =======
df_final = pd.DataFrame(resultados)

# ======= Salvar =======
out_path = pasta_csvs / "LIWC_por_arquivo_pct.csv"
df_final.to_csv(out_path, index=False)

print("\n✅ LIWC aplicado a CADA arquivo individualmente!")
print("📌 Arquivo salvo em:", out_path)

df_final