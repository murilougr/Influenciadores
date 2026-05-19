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
# CONFIGURAÇÃO
# =========================================================
csv_files = [
    #csvs
]

RANDOM_STATE = 42
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# =========================================================
# FUNÇÃO PARA LER CSV
# =========================================================
def read_csv_flexible(file_path):

    seps = [",", ";", "\t"]

    for sep in seps:
        try:
            df = pd.read_csv(file_path, sep=sep)
            if df.shape[1] > 1:
                return df
        except:
            pass

    return pd.read_csv(file_path)

# =========================================================
# LER TRANSCRIÇÕES
# =========================================================
texts = []

for file_name in csv_files:

    file_path = file_name

    print("\nLendo:", file_path)

    df = pd.read_csv(file_path)

    df = df[df["CATEGORIA"].str.lower() == "transcricao"]

    df = df.dropna(subset=["TEXTO"])

    df["TEXTO"] = df["TEXTO"].astype(str).str.strip()

    df = df[df["TEXTO"] != ""]

    texts.extend(df["TEXTO"].tolist())

print("Total de transcrições:", len(texts))

# =========================================================
# EMBEDDINGS
# =========================================================
print("\nGerando embeddings...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)

print("Embeddings shape:", embeddings.shape)

# =========================================================
# GRID SEARCH
# =========================================================

param_grid = {

    "n_neighbors": [10,20,30],
    "n_components": [10,15],
    "min_cluster_size": [80,100,150],
    "min_samples": [5,10]

}

results = []

total_docs = len(texts)

for nn in param_grid["n_neighbors"]:
    for nc in param_grid["n_components"]:
        for mcs in param_grid["min_cluster_size"]:
            for ms in param_grid["min_samples"]:

                print("\n===================================")
                print(nn, nc, mcs, ms)

                try:

                    umap_model = UMAP(
                        n_neighbors=nn,
                        n_components=nc,
                        min_dist=0.0,
                        metric="cosine",
                        random_state=RANDOM_STATE
                    )

                    hdbscan_model = HDBSCAN(
                        min_cluster_size=mcs,
                        min_samples=ms,
                        metric="euclidean",
                        cluster_selection_method="eom",
                        prediction_data=True
                    )

                    topic_model = BERTopic(
                        language="multilingual",
                        umap_model=umap_model,
                        hdbscan_model=hdbscan_model,
                        calculate_probabilities=False,
                        verbose=False
                    )

                    topics, _ = topic_model.fit_transform(texts, embeddings)

                    num_outliers = sum(1 for t in topics if t == -1)

                    outlier_pct = (num_outliers/total_docs)*100

                    valid_topics = [t for t in topics if t != -1]

                    num_topics = len(set(valid_topics))

                    topic_sizes = pd.Series(valid_topics).value_counts()

                    mean_size = topic_sizes.mean()

                    median_size = topic_sizes.median()

                    largest_topic = topic_sizes.max()

                    results.append({

                        "n_neighbors": nn,
                        "n_components": nc,
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                        "num_topics": num_topics,
                        "num_outliers": num_outliers,
                        "outlier_pct": outlier_pct,
                        "mean_topic_size": mean_size,
                        "median_topic_size": median_size,
                        "largest_topic_size": largest_topic

                    })

                except Exception as e:

                    print("Erro:", e)

# =========================================================
# SALVAR RESULTADOS
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "grid_search_transcricoes.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nGrid search finalizado")
print(results_df.sort_values("outlier_pct").head(10))