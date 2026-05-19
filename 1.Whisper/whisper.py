# === Instalação das dependências ===
#!pip install -q git+https://github.com/openai/whisper.git
#!apt-get update -qq && apt-get install -y ffmpeg

import os
import csv
import whisper

# === Caminho base ===
base_root = "<caminho>"

# === Carrega o modelo Whisper ===
# (tiny, base, small, medium, large, turbo)
model = whisper.load_model("turbo")

# === Função auxiliar para processar um país ===
def processar_pais(pais_path):
    print(f"\n🌎 Processando país: {os.path.basename(pais_path)}")

    if not os.path.exists(pais_path):
        print(f"⚠️  Caminho não encontrado: {pais_path}")
        return

    pessoas = [p for p in os.listdir(pais_path) if os.path.isdir(os.path.join(pais_path, p))]
    print(f"🔍 {len(pessoas)} pastas encontradas em {pais_path}")

    for pessoa in pessoas:
        pasta_videos = os.path.join(pais_path, pessoa, "Videos")
        if not os.path.exists(pasta_videos):
            print(f"⚠️  Pasta 'Videos' não encontrada em {pessoa}, pulando...")
            continue

        arquivos = [
            os.path.join(pasta_videos, f)
            for f in os.listdir(pasta_videos)
            if f.lower().endswith(".mp4")
        ]

        if not arquivos:
            print(f"⚠️  Nenhum vídeo encontrado em {pessoa}")
            continue

        print(f"\n🎬 {pessoa}: {len(arquivos)} vídeos encontrados.")

        # Nome do CSV de saída
        csv_saida = os.path.join(pais_path, f"{pessoa}_transcricoes.csv")

        # Transcreve cada vídeo
        with open(csv_saida, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["arquivo", "transcricao"])  # cabeçalho

            for caminho in arquivos:
                nome_arquivo = os.path.basename(caminho)
                print(f"🎧 Transcrevendo: {nome_arquivo} ...")

                try:
                    resultado = model.transcribe(caminho)
                    texto = resultado["text"].strip()
                    writer.writerow([nome_arquivo, texto])
                except Exception as e:
                    print(f"❌ Erro em {nome_arquivo}: {e}")
                    writer.writerow([nome_arquivo, f"ERRO: {e}"])

        print(f"✅ Transcrições salvas em: {csv_saida}")

# === Lista de países a processar ===
paises = ["Brasil", "EUA"]

for pais in paises:
    pais_path = os.path.join(base_root, pais)
    processar_pais(pais_path)

print("\n🎉 Transcrição concluída para Brasil e EUA!")