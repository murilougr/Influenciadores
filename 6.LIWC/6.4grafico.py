import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ========================================================
# FUNÇÕES
# ========================================================

def fdr_correction(p_values):
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]

    adjusted = ranked * n / (np.arange(1, n + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    p_fdr = np.empty(n)
    p_fdr[order] = adjusted
    return p_fdr

def median_ci_bootstrap(series, n_boot=5000, ci=95, random_state=42):
    s = pd.Series(series).dropna().astype(float).values
    n = len(s)

    if n == 0:
        return np.nan, 0.0
    if n == 1:
        return np.median(s), 0.0

    rng = np.random.default_rng(random_state)
    boot_medians = np.array([
        np.median(rng.choice(s, size=n, replace=True))
        for _ in range(n_boot)
    ])

    med = np.median(s)
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_medians, alpha)
    upper = np.percentile(boot_medians, 100 - alpha)

    err = max(med - lower, upper - med)
    return med, err

def processar(df_br, df_us, categorias):
    resultados = []

    for cat in categorias:
        br_vals = df_br[cat].dropna().astype(float)
        us_vals = df_us[cat].dropna().astype(float)

        if len(br_vals) == 0 or len(us_vals) == 0:
            continue

        stat, p = mannwhitneyu(br_vals, us_vals, alternative="two-sided")

        med_br, err_br = median_ci_bootstrap(br_vals)
        med_us, err_us = median_ci_bootstrap(us_vals)

        resultados.append({
            "categoria": cat,
            "mediana_BR": med_br,
            "mediana_EUA": med_us,
            "erro_BR": err_br,
            "erro_EUA": err_us,
            "p_value": p
        })

    df = pd.DataFrame(resultados)
    df["p_fdr"] = fdr_correction(df["p_value"])

    return df[df["p_fdr"] < 0.05].copy()

# ========================================================
# DADOS
# ========================================================

categorias = [
    "posemo", "sad",
    "friend", "family",
    "body", "health", "ingest",
    "power", "reward", "affiliation", "achiev",
    "leisure", "relig",
    "swear", "nonflu", "filler"
]

# ========================================================
# TRADUÇÃO
# ========================================================

traducao = {
    "posemo": "E. positiva\n(P. Afetivos)",
    "sad": "Tristeza\n(P. Afetivos)",
    "social": "Social\n(P. Sociais)",
    "friend": "Amigos\n(P. Sociais)",
    "family": "Família\n(P. Sociais)",
    "body": "Corpo\n(P. Biológicos)",
    "health": "Saúde\n(P. Biológicos)",
    "ingest": "Alimentação\n(P. Biológicos)",
    "power": "Poder\n(Motivações)",
    "reward": "Recompensa\n(Motivações)",
    "affiliation": "Afiliação\n(Motivações)",
    "achiev": "Realização\n(Motivações)",
    "leisure": "Lazer\n(P. Pessoais)",
    "relig": "Religião\n(P. Pessoais)",
    "swear": "Palavrões\n(L. Informal)",
    "nonflu": "Hesitação\n(L. Informal)",
    "filler": "Preenchimento\n(L. Informal)",
}

# legenda
df_br_L = pd.read_csv("LIWC_por_arquivo_pct_legenda_BRASIL.csv")
df_us_L = pd.read_csv("LIWC_por_arquivo_pct_legenda_EUA.csv")

# transcrição
df_br_T = pd.read_csv("LIWC_por_arquivo_pct_transcricao_BRASIL.csv")
df_us_T = pd.read_csv("LIWC_por_arquivo_pct_transcricao_EUA.csv")

# ========================================================
# PROCESSAMENTO
# ========================================================

df_L = processar(df_br_L, df_us_L, categorias)
df_T = processar(df_br_T, df_us_T, categorias)

df_L["tipo"] = "L"
df_T["tipo"] = "T"

df_all = pd.concat([df_L, df_T], ignore_index=True)

# labels traduzidos
df_all["label"] = df_all["categoria"].map(traducao)

# cor do label conforme país com maior mediana
df_all["cor_label"] = np.where(
    df_all["mediana_BR"] > df_all["mediana_EUA"],
    "#0F4D0F",   # verde = Brasil maior
    "#b81414"    # vermelho = EUA maior
)

# ========================================================
# GRÁFICO
# ========================================================

x = np.arange(len(df_all))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 6))

ax.bar(
    x - width/2, df_all["mediana_BR"], width,
    yerr=df_all["erro_BR"], capsize=4,
    color="#0F4D0F", label="Brasil"
)

ax.bar(
    x + width/2, df_all["mediana_EUA"], width,
    yerr=df_all["erro_EUA"], capsize=4,
    color="#b81414", label="EUA"
)

# ========================================================
# EIXO X
# ========================================================

ax.set_xticks(x)
ax.set_xticklabels(
    df_all["label"],
    rotation=90,
    ha="center",
    fontsize=14,
    fontweight="bold"
)

# aplicar cor em cada categoria do eixo x
for ticklabel, cor in zip(ax.get_xticklabels(), df_all["cor_label"]):
    ticklabel.set_color(cor)

# ========================================================
# LINHA DIVISÓRIA ENTRE L E T
# ========================================================

n_L = len(df_L)
ax.axvline(n_L - 0.5, color="black", linestyle="--", linewidth=1)

# ========================================================
# TÍTULOS ACIMA DAS METADES
# ========================================================

ymax = max(
    (df_all["mediana_BR"] + df_all["erro_BR"]).max(),
    (df_all["mediana_EUA"] + df_all["erro_EUA"]).max()
)

if n_L > 0:
    centro_L = (0 + (n_L - 1)) / 2
    ax.text(
        centro_L, ymax + 0.45, "Legendas",
        ha="center", va="bottom",
        fontsize=16, fontweight="bold"
    )

n_T = len(df_T)
if n_T > 0:
    inicio_T = n_L
    fim_T = len(df_all) - 1
    centro_T = (inicio_T + fim_T) / 2
    ax.text(
        centro_T, ymax + 0.45, "Transcrições",
        ha="center", va="bottom",
        fontsize=16, fontweight="bold"
    )

# aumentar limite superior para caber os títulos
ax.set_ylim(top=ax.get_ylim()[1] + 0.8)

# ========================================================
# FINAL
# ========================================================

ax.set_ylabel("Frequência (%)", fontsize=14, fontweight="bold")
ax.tick_params(axis="y", labelsize=14)
ax.legend(fontsize=14)
ax.margins(x=0.01)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("comparacao_L_T_mediana_sem_p.png", dpi=300)
plt.show()