import json
from collections import defaultdict
from pathlib import Path
import random

curr_dir = Path(__file__).resolve().parent

# Připravíme společnou výstupní složku pro train i validation.
# open(..., "w") vytvoří soubor, ale chybějící složku samo nevytvoří.
output_dir = curr_dir / "datasets_triplets"
output_dir.mkdir(parents=True, exist_ok=True)

#priprava train datasetu 
train_datasets_dir = curr_dir.parent / "split_datasets" / "random" / "train.jsonl"

#nacteni datasetu
with open(train_datasets_dir, "r", encoding="utf-8") as file:
    train_datasets = [json.loads(line) for line in file if line.strip()]

records_by_intent = defaultdict(list)
for record in train_datasets:
    records_by_intent[record["intent_id"]].append(record)

#kruhove parovani s Triple Loss
triplets = []
for intent_name, records in records_by_intent.items():

    if intent_name == "oos":
        continue

    negative_candidates = []
    for dataset in train_datasets:
        if dataset["intent_id"] != intent_name:
            negative_candidates.append(dataset["text"])

    if len(records) < 2:
        continue
    random.seed(42)
    for index, record in enumerate(records):
        negative = random.choice(negative_candidates)
        anchor = record["text"]

        next_index = index + 1

        # Za posledním záznamem už další není, proto se vrátíme na první.
        if next_index == len(records):
            next_index = 0

        positive = records[next_index]["text"]

        pair_sentence = {
            "anchor": anchor,
            "positive": positive,
            "negative": negative
        }

        triplets.append(pair_sentence)

# Cesta k výstupnímu souboru.
output_path_train = output_dir / "train_triplets.jsonl"

# Každou trojici uložíme jako jeden JSON objekt na samostatný řádek.
with open(output_path_train, "w", encoding="utf-8") as file:
    for triplet in triplets:
        json_text = json.dumps(triplet, ensure_ascii=False)
        file.write(json_text + "\n")

print("Počet uložených trojic:", len(triplets))
print("Soubor:", output_path_train)






#priprava val datasetu 
val_datasets_dir = curr_dir.parent / "split_datasets" / "random" / "validation.jsonl"

#nacteni datasetu
with open(val_datasets_dir, "r", encoding="utf-8") as file:
    val_datasets = [json.loads(line) for line in file if line.strip()]

records_by_intent = defaultdict(list)
for record in val_datasets:
    records_by_intent[record["intent_id"]].append(record)

#kruhove parovani s Triple Loss
triplets = []
for intent_name, records in records_by_intent.items():

    if intent_name == "oos":
        continue

    negative_candidates = []
    for dataset in val_datasets:
        if dataset["intent_id"] != intent_name:
            negative_candidates.append(dataset["text"])

    if len(records) < 2:
        continue
    random.seed(42)
    for index, record in enumerate(records):
        negative = random.choice(negative_candidates)
        anchor = record["text"]

        next_index = index + 1

        # Za posledním záznamem už další není, proto se vrátíme na první.
        if next_index == len(records):
            next_index = 0

        positive = records[next_index]["text"]

        pair_sentence = {
            "anchor": anchor,
            "positive": positive,
            "negative": negative
        }

        triplets.append(pair_sentence)

# Cesta k výstupnímu souboru.
output_path_val = output_dir / "val_triplets.jsonl"

# Každou trojici uložíme jako jeden JSON objekt na samostatný řádek.
with open(output_path_val, "w", encoding="utf-8") as file:
    for triplet in triplets:
        json_text = json.dumps(triplet, ensure_ascii=False)
        file.write(json_text + "\n")

print("Počet uložených trojic:", len(triplets))
print("Soubor:", output_path_val)
