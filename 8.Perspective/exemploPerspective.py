import pandas as pd
from googleapiclient import discovery
import json
import ssl
import time
import os
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---
ssl._create_default_https_context = ssl._create_unverified_context
API_KEY = 'AIzaSyDqrIbzz4Sxm4_tOPQisutx6NxQileUtGw'
ARQUIVO_ENTRADA = '../data/steam_reviews_gold.zip'
ARQUIVO_SAIDA = '../data/steam_reviews_perspective_results.csv'

client = discovery.build(
    "commentanalyzer", "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)

# --- 2. FUNÇÃO DE REQUISIÇÃO ---
def analisar_perspective(texto, idioma):
    request_body = {
        'comment': { 'text': texto },
        'languages': [idioma],
        'requestedAttributes': {
            'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {},
            'INSULT': {}, 'PROFANITY': {}, 'THREAT': {}
        },
        'doNotStore': True # Ética de dados: não armazena nos servidores do Google
    }
    try:
        response = client.comments().analyze(body=request_body).execute()
        resultados = {}
        for atributo, dados in response['attributeScores'].items():
            resultados[f'perspective_{atributo.lower()}'] = float(dados['summaryScore']['value'])
        return resultados
    except Exception as e:
        return {"perspective_erro": str(e)}

# --- 3. MOTOR DE PROCESSAMENTO ---
def processar_base_gold():
    print(f"Iniciando processamento da base: {ARQUIVO_ENTRADA}")
    df = pd.read_csv(ARQUIVO_ENTRADA)

    # --- LIMPEZA DE COLUNAS SOLICITADA ---
    cols_para_remover = ['language', 'detected_language', 'confidence', 'voted_up', 'votes_up', 'votes_funny']
    df = df.drop(columns=[c for c in cols_para_remover if c in df.columns])

    col_idioma = 'primary_lang'
    col_texto = 'review'

    # --- DEFINIÇÃO DA ORDEM DAS COLUNAS ---
    # Pegamos as colunas base e inserimos as novas na posição correta
    colunas_base = df.columns.tolist()
    idx_review = colunas_base.index(col_texto)

    # Novas colunas que virão da API
    novas_cols = [
        'perspective_toxicity', 'perspective_severe_toxicity',
        'perspective_identity_attack', 'perspective_insult',
        'perspective_profanity', 'perspective_threat', 'perspective_erro'
    ]

    # Ordem final: colunas antes da review + review + perspective_toxicity + resto
    ordem_final = colunas_base[:idx_review + 1] + ['perspective_toxicity'] + \
                  colunas_base[idx_review + 1:] + novas_cols[1:]

    # Lógica de Checkpoint
    if os.path.exists(ARQUIVO_SAIDA):
        df_progresso = pd.read_csv(ARQUIVO_SAIDA)
        processados = len(df_progresso)
        print(f"Retomando do índice: {processados}")
        df_para_processar = df.iloc[processados:]
    else:
        df_para_processar = df
        # Cria o arquivo com o cabeçalho na ordem correta
        pd.DataFrame(columns=ordem_final).to_csv(ARQUIVO_SAIDA, index=False)

    print(f"Reviews restantes: {len(df_para_processar)}")

    # Loop de processamento
    for index, row in tqdm(df_para_processar.iterrows(), total=len(df_para_processar)):
        texto = row[col_texto]
        idioma = 'pt' if 'pt' in str(row[col_idioma]).lower() else 'en'

        if pd.isna(texto) or str(texto).strip() == "":
            res_api = {c: None for c in novas_cols}
            res_api['perspective_erro'] = "texto_vazio"
        else:
            res_api = analisar_perspective(texto, idioma)
            # Garante que todas as colunas novas existam no dicionário
            for c in novas_cols:
                if c not in res_api: res_api[c] = None

        # Consolidação com reordenamento imediato
        linha_dict = {**row.to_dict(), **res_api}
        df_row = pd.DataFrame([linha_dict])[ordem_final]

        # Salva linha por linha
        df_row.to_csv(ARQUIVO_SAIDA, mode='a', header=False, index=False, float_format='%.8f')

        # Taxa de transferência controlada (1 QPS)
        time.sleep(0.6)

    print(f"\n✅ Processamento concluído com sucesso!")

processar_base_gold()