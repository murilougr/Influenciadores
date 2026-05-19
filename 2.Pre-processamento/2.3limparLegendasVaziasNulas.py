# Retirar nula/vazias
import pandas as pd

df_filtrado = df_filtrado[df_filtrado['text'].notna()]
df_filtrado = df_filtrado[df_filtrado['text'].astype(str).str.strip() != ""]

