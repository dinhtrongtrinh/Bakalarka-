import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Skript je v embedding_models, data čteme ze sousední složky alza_scraper.
# Cesta funguje nezávisle na tom, z jaké složky skript spustíme.
project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "alza_scraper" / "splited_datatsets" / "train.jsonl"

# Pro první pokus načteme pouze první dvě neprázdné položky trénovací sady.
records = []
with dataset_path.open(encoding="utf-8") as file:
    for line in file:
        if line.strip() and json.loads(line)["intent_id"] == "order_delay":
            records.append(json.loads(line))
            if len(records) == 2:
                break

if len(records) != 2:
    raise ValueError("Ukázka potřebuje alespoň dvě věty v trénovacím datasetu.")

sentences = [record["text"] for record in records]
for index, record in enumerate(records, start=1):
    print(f"Věta {index} ({record['intent_id']}): {record['text']}")

# Při prvním použití se stáhnou váhy, při dalším se použije místní cache.
# CPU stačí pro dvě věty; model pouze používáme, jeho váhy netrénujeme.
model = SentenceTransformer("google/embeddinggemma-300m", device="cpu")

# Modelu předáme instrukci pro klasifikaci. Pro obě věty je stejná.
# encode vrátí NumPy matici: jeden řádek pro každou větu.
# normalize_embeddings=True upraví L2 délku každého vektoru na 1.
embeddings = model.encode(
    sentences,
    prompt="task: classification | query: ",
    normalize_embeddings=True,
    convert_to_numpy=True,
)

print("Tvar výstupu (počet vět, rozměr embeddingu):", embeddings.shape)
print("Prvních 8 hodnot prvního embeddingu:", embeddings[0, :8])
print("L2 délky vektorů (přibližně 1):", np.linalg.norm(embeddings, axis=1))

# U vektorů s jednotkovou délkou je skalární součin kosinovou podobností.
# @ násobí matice, .T prohodí řádky a sloupce: vznikne matice 2 × 2.
similarities = embeddings @ embeddings.T
print("Matice kosinových podobností:")
print(similarities)
print(f"Podobnost první a druhé věty: {similarities[0, 1]:.4f}")
