import pandas as pd
import re

# Lê o CSV
df = pd.read_csv("arquivo.csv")

# Função para substituir qualquer palavra que começa com @ por apenas "@"
def substituir_mencoes(texto):
    if pd.isna(texto):
        return texto
    
    # Substitui strings que começam com @ até encontrar espaço
    return re.sub(r'@\S+', '@', str(texto))

# Aplica na coluna TEXTO
df["TEXTO"] = df["TEXTO"].apply(substituir_mencoes)

# Salva o resultado
df.to_csv("arquivo_tratado.csv", index=False)
