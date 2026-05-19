import os
import warnings
import numpy as np
import pandas as pd

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURAÇÃO
# =========================================================

csv_files = [
    # csvs
]

RANDOM_STATE = 42
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# =========================================================
# 2. FUNÇÃO PARA LER CSV
# =========================================================
def read_csv_flexible(file_path):
    seps = [",", ";", "\t"]
    last_error = None

    for sep in seps:
        try:
            df = pd.read_csv(file_path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_error = e

    try:
        return pd.read_csv(file_path)
    except Exception as e:
        last_error = e
        raise last_error

# =========================================================
# 3. LER TEXTOS
# =========================================================
all_texts = []
metadata = []

for file_name in csv_files:
    file_path = file_name

    print("\nLendo:", file_path)

    try:
        df = pd.read_csv(file_path)

        df = df[df["CATEGORIA"].isin(["transcricao", "legenda"])]

        if "TEXTO" not in df.columns:
            print(f"Coluna 'TEXTO' não encontrada em: {file_name}")
            print(f"Colunas disponíveis: {list(df.columns)}")
            continue

        df = df.dropna(subset=["TEXTO"]).copy()
        df["TEXTO"] = df["TEXTO"].astype(str).str.strip()
        df = df[df["TEXTO"] != ""]
        df = df.drop_duplicates(subset=["TEXTO"])

        textos = df["TEXTO"].tolist()

        for idx, text in enumerate(textos):
            all_texts.append(text)
            metadata.append({
                "arquivo_origem": file_name,
                "id_local_arquivo": idx,
                "texto": text
            })

        print(f"{file_name}: {len(textos)} textos carregados.")

    except Exception as e:
        print(f"Erro ao ler {file_name}: {e}")

print(f"\nTotal de textos carregados: {len(all_texts)}")

if len(all_texts) == 0:
    raise ValueError("Nenhum texto foi carregado.")

metadata_df = pd.DataFrame(metadata)

# =========================================================
# 4. GERAR EMBEDDINGS UMA VEZ
# =========================================================
print("\n" + "=" * 80)
print("GERANDO EMBEDDINGS...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embeddings = embedding_model.encode(
    all_texts,
    show_progress_bar=True,
    batch_size=32
)

print(f"Embeddings gerados: {embeddings.shape}")

# =========================================================
# 5. CONFIGURAÇÕES TOP 3
# =========================================================
configs = [
    {
        "config_name": "top1_nn20_nc15_mcs100_ms10",
        "n_neighbors": 10,
        "n_components": 10,
        "min_cluster_size": 50,
        "min_samples": 5
    },
    {
        "config_name": "top2_nn20_nc15_mcs150_ms5",
        "n_neighbors": 10,
        "n_components": 15,
        "min_cluster_size": 50,
        "min_samples": 5
    },
    {
        "config_name": "top3_nn20_nc15_mcs150_ms10",
        "n_neighbors": 10,
        "n_components": 10,
        "min_cluster_size": 50,
        "min_samples": 10
    }
]

# =========================================================
# 6. FUNÇÕES AUXILIARES
# =========================================================
def evaluate_topics(topics):
    total_docs = len(topics)
    num_outliers = sum(1 for t in topics if t == -1)
    outlier_pct = (num_outliers / total_docs) * 100 if total_docs > 0 else np.nan

    valid_topics = [t for t in topics if t != -1]
    unique_topics = sorted(set(valid_topics))
    num_topics = len(unique_topics)

    if num_topics == 0:
        return {
            "num_topics": 0,
            "num_outliers": num_outliers,
            "outlier_pct": outlier_pct,
            "mean_topic_size": 0,
            "median_topic_size": 0,
            "largest_topic_size": 0
        }

    topic_sizes = pd.Series(valid_topics).value_counts()

    return {
        "num_topics": num_topics,
        "num_outliers": num_outliers,
        "outlier_pct": outlier_pct,
        "mean_topic_size": float(topic_sizes.mean()),
        "median_topic_size": float(topic_sizes.median()),
        "largest_topic_size": int(topic_sizes.max())
    }

def save_topic_words(topic_model, valid_topics, output_txt_path):
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("PALAVRAS REPRESENTATIVAS DOS TÓPICOS\n")
        f.write("=" * 80 + "\n\n")
        for topic_id in sorted(valid_topics):
            f.write(f"TÓPICO {topic_id}\n")
            f.write("-" * 40 + "\n")
            words = topic_model.get_topic(topic_id)
            if words is None:
                f.write("Sem palavras disponíveis.\n\n")
                continue
            for term, score in words:
                f.write(f"{term}\t{score}\n")
            f.write("\n")

def save_model_summary(config, metrics, output_txt_path):
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("RESUMO DO MODELO\n")
        f.write("=" * 80 + "\n\n")

        f.write("PARÂMETROS\n")
        f.write("-" * 40 + "\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

        f.write("\nMÉTRICAS\n")
        f.write("-" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

# =========================================================
# 7. PASTA DE SAÍDA
# =========================================================
base_output_dir = "eua_transc_leg_top_3"
os.makedirs(base_output_dir, exist_ok=True)

# salva textos usados
metadata_df.to_csv(
    os.path.join(base_output_dir, "textos_usados.csv"),
    index=False,
    encoding="utf-8-sig"
)

# salva embeddings
np.save(os.path.join(base_output_dir, "embeddings.npy"), embeddings)

summary_rows = []

# =========================================================
# 8. RODAR AS 3 CONFIGURAÇÕES
# =========================================================
for cfg in configs:
    print("\n" + "=" * 100)
    print(f"RODANDO {cfg['config_name']}")

    config_output_dir = os.path.join(base_output_dir, cfg["config_name"])
    os.makedirs(config_output_dir, exist_ok=True)

    try:
        umap_model = UMAP(
            n_neighbors=cfg["n_neighbors"],
            n_components=cfg["n_components"],
            min_dist=0.0,
            metric="cosine",
            random_state=RANDOM_STATE
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=cfg["min_cluster_size"],
            min_samples=cfg["min_samples"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )

        topic_model = BERTopic(
            language="multilingual",
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=False,
            verbose=True
        )

        topics, probs = topic_model.fit_transform(all_texts, embeddings)

        metrics = evaluate_topics(topics)
        valid_topics = sorted(set([t for t in topics if t != -1]))

        print("\n===== RESUMO =====")
        print(f"num_topics: {metrics['num_topics']}")
        print(f"num_outliers: {metrics['num_outliers']}")
        print(f"outlier_pct: {metrics['outlier_pct']:.2f}")
        print(f"mean_topic_size: {metrics['mean_topic_size']:.2f}")
        print(f"median_topic_size: {metrics['median_topic_size']:.2f}")
        print(f"largest_topic_size: {metrics['largest_topic_size']}")

        # topic info
        topic_info = topic_model.get_topic_info()

        # documentos + tópicos
        results_df = metadata_df.copy()
        results_df["topic"] = topics

        topic_name_map = {}
        if "Topic" in topic_info.columns and "Name" in topic_info.columns:
            topic_name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
            results_df["topic_name"] = results_df["topic"].map(topic_name_map)

        # caminhos
        topic_info_path = os.path.join(config_output_dir, "topic_info.csv")
        docs_topics_path = os.path.join(config_output_dir, "documentos_topicos.csv")
        topic_words_path = os.path.join(config_output_dir, "palavras_topicos.txt")
        summary_txt_path = os.path.join(config_output_dir, "resumo_modelo.txt")
        model_path = os.path.join(config_output_dir, "bertopic_model")

        # salvar
        topic_info.to_csv(topic_info_path, index=False, encoding="utf-8-sig")
        results_df.to_csv(docs_topics_path, index=False, encoding="utf-8-sig")
        save_topic_words(topic_model, valid_topics, topic_words_path)
        save_model_summary(cfg, metrics, summary_txt_path)
        topic_model.save(model_path)

        summary_rows.append({
            "config_name": cfg["config_name"],
            "n_neighbors": cfg["n_neighbors"],
            "n_components": cfg["n_components"],
            "min_cluster_size": cfg["min_cluster_size"],
            "min_samples": cfg["min_samples"],
            "num_topics": metrics["num_topics"],
            "num_outliers": metrics["num_outliers"],
            "outlier_pct": metrics["outlier_pct"],
            "mean_topic_size": metrics["mean_topic_size"],
            "median_topic_size": metrics["median_topic_size"],
            "largest_topic_size": metrics["largest_topic_size"]
        })

        print(f"Arquivos salvos em: {config_output_dir}")

    except Exception as e:
        print(f"Erro na configuração {cfg['config_name']}: {e}")
        summary_rows.append({
            "config_name": cfg["config_name"],
            "n_neighbors": cfg["n_neighbors"],
            "n_components": cfg["n_components"],
            "min_cluster_size": cfg["min_cluster_size"],
            "min_samples": cfg["min_samples"],
            "num_topics": np.nan,
            "num_outliers": np.nan,
            "outlier_pct": np.nan,
            "mean_topic_size": np.nan,
            "median_topic_size": np.nan,
            "largest_topic_size": np.nan,
            "error": str(e)
        })

# =========================================================
# 9. RESUMO FINAL COMPARATIVO
# =========================================================
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(base_output_dir, "resumo_comparativo_top3.csv")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("PROCESSAMENTO FINALIZADO")
print(f"Resumo comparativo salvo em: {summary_path}")
print(summary_df)