import os
import pandas as pd
from google.colab import files

pasta = "/content"
coluna_texto = "transcricao"

# Listar todos os CSVs no /content
arquivos = [f for f in os.listdir(pasta) if f.endswith(".csv")]

print(f"{len(arquivos)} CSVs encontrados.\n")

resultados = []     # para relatório final
arquivos_limpos = []

for nome_csv in arquivos:
    caminho = os.path.join(pasta, nome_csv)

    try:
        df = pd.read_csv(caminho)

        if coluna_texto not in df.columns:
            print(f"⚠️  Coluna {coluna_texto} não existe em {nome_csv}. Pulando.\n")
            continue

        # Contar erros
        erros_mask = df[coluna_texto].astype(str).str.startswith("ERRO: Failed to load audio")
        n_erros = erros_mask.sum()

        # Remover erros
        df_limpo = df[~erros_mask]

        # Nome novo
        nome_limpo = nome_csv.replace(".csv", "_limpo.csv")
        caminho_limpo = os.path.join(pasta, nome_limpo)

        # Salvar
        df_limpo.to_csv(caminho_limpo, index=False)
        arquivos_limpos.append(caminho_limpo)

        resultados.append({
            "csv": nome_csv,
            "linhas_originais": len(df),
            "erros_encontrados": n_erros,
            "linhas_finais": len(df_limpo)
        })

        print(f"✅ {nome_csv}: removidos {n_erros} erros → gerado {nome_limpo}")

    except Exception as e:
        print(f"❌ Erro ao processar {nome_csv}: {e}\n")


print("\n📥 Iniciando downloads...\n")
for arq in arquivos_limpos:
    files.download(arq)

# Mostrar relatório final como tabela
print("\n📊 RELATÓRIO FINAL\n")
relatorio_df = pd.DataFrame(resultados)
display(relatorio_df)
