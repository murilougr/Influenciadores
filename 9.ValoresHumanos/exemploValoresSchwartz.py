#!pip install transformers torch

from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="devnote5676/schwartz-values-classifier",
    tokenizer="devnote5676/schwartz-values-classifier"
)

text = "Já estou preparando minha casa pro natal. Vocês também vão montar uma árvore? 🎄 #reels #love #natal2023 #fy"

inp1 = "<security> [SEP] " + text
print("security:", clf(inp1))

inp2 = "<power> [SEP] " + text
print("power:", clf(inp2))

inp3 = "<achievement> [SEP] " + text
print("achievement:", clf(inp3))

inp4 = "<hedonism> [SEP] " + text
print("hedonism:", clf(inp4))

inp5 = "<stimulation> [SEP] " + text
print("stimulation:", clf(inp5))

inp6 = "<self-direction> [SEP] " + text
print("self-direction:", clf(inp6))

inp7 = "<universalism> [SEP] " + text
print("universalism:", clf(inp7))

inp8 = "<benevolence> [SEP] " + text
print("benevolence:", clf(inp8))

inp9 = "<conformity> [SEP] " + text
print("conformity:", clf(inp9))

inp10 = "<tradition> [SEP] " + text
print("tradition:", clf(inp10))
