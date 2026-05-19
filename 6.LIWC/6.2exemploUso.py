import re
import pandas as pd

# ============================================================
# 🔹 REGEX TOKENIZER
# ============================================================
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", flags=re.UNICODE)


# ============================================================
# 🔹 FUNÇÃO PARA ANALISAR UMA FRASE USANDO O DICIONÁRIO LIWC
# ============================================================
def liwc_detect_categories(text, categories, dictionary, categorias_desejadas):
    text = str(text).lower()
    tokens = TOKEN_RE.findall(text)

    categorias_encontradas = set()   # evita duplicatas

    # percorre cada token da frase
    for token in tokens:
        for pattern, cat_ids in dictionary.items():

            # ----- radical (*)
            if "*" in pattern:
                root = pattern.replace("*", "")
                if token.startswith(root):
                    for cid in cat_ids:
                        cat_name = categories.get(cid)
                        if cat_name in categorias_desejadas:
                            categorias_encontradas.add(cat_name)

            # ----- palavra exata
            else:
                if token == pattern:
                    for cid in cat_ids:
                        cat_name = categories.get(cid)
                        if cat_name in categorias_desejadas:
                            categorias_encontradas.add(cat_name)

    return list(categorias_encontradas)


# ============================================================
# 🔹 LISTA DE 10 FRASES (MODIFIQUE AQUI)
# ============================================================
frases = [
    "Sapatos / vegetação / mulher / mangas / banco / coração / brinquedo / olhos / praia / feliz",
    "Sapatos / cavalo / sofá / brinquedo / macacão / olhos / flores / banho / feliz / praia",
    "Três / máscaras / todos / mulheres / canto / medo / homens / camisetas / fantasias / corporais",
    "Sentadas / gráfico / ambos / água / macacão / abraçando / brinquedo / vegetação / bebê / banco",
    "Homem / praia / mulher / ambos / banho / inflável / canto / lazer / tubarão / vegetação",
    "Mulher / bebê / sentadas / macacão / selfie / abraçando / flores / cama / aniversário / carro",
    "Praia / todos / banho / palmeiras / inflável / mulheres / adultos / homem / verão / diversão",
    "Homem / bebê / beisebol / cowboy / ombros / médicos / cama / macacão / praia / televisão",
    "Natal / árvore / dourados / luzes / laços / urso / mulher / sofá / boneco / feliz",
    "Ele / gato / galáctico / pôster / casual / youtube / brasil / praia / beisebol / vegetação",
    "Balões / flores / youtube / cupcakes / 13 / feliz / toalha / milhões / dourado / princesa",
    "Luzes / microfone / aplaudindo / público / teatro / artista / concerto / vivo / trajes / celulares",
    "Meninas / sentadas / felizes / detalhes / tênis / tiara / florais / coração / banner / mechas",
    "Família / mulher / homem / centro / felizes / familiar / menino / balões / membros / camisetas",
    "Árvore / luzes / família / dourados / chapéus / homem / festivas / mulher / menino / ambos",
    "Mcdonald / pizza / cozinha / homem / bebidas / hambúrgueres / seção / chocolate / exagerada / subway",
    "Homem / barba / casal / tênis / gráfico / canto / selfie / moletom / iluminação / sofá",
    "Bebê / dormindo / superfície / cama / flores / urso / sorridente / sofá / macia / banho"
]


# ============================================================
# 🔹 LISTA DE CATEGORIAS QUE VOCÊ QUER ANALISAR
#     (use exatamente os nomes que aparecem no dicionário)
# ============================================================
categorias_desejadas = [
    "relativ", "motion", "space", "time",
    "social", "family", "friend", "female", "male",
    "drives", "affiliation", "achieve", "power", "reward", "risk",
    "affect", "posemo", "negemo", "anx", "anger", "sad",
    "percept", "see", "hear", "feel"
]


# ============================================================
# 🔹 PROCESSA CADA FRASE E COLETA CATEGORIAS
# ============================================================
resultados = []

for frase in frases:
    cats = liwc_detect_categories(frase, categories, dictionary, categorias_desejadas)
    resultados.append({
        "frase": frase,
        "categorias_detectadas": ", ".join(cats) if cats else ""
    })


# ============================================================
# 🔹 CRIA DATAFRAME FINAL
# ============================================================
df_resultado = pd.DataFrame(resultados)

print(df_resultado)
df_resultado
