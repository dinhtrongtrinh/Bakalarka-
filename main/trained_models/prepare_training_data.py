import json
import random
import hashlib
from pathlib import Path


# 1. Použijeme už hotový random split. Žádné nové rozdělování dat.
current_dir = Path(__file__).resolve().parent
datasets_dir = current_dir.parent / "split_datasets" / "random"
output_dir = current_dir / "data" / "random"
output_dir.mkdir(parents=True, exist_ok=True)

# Zkontrolujeme, že mezi splity nejsou stejné texty. Čtení testu zde slouží
# pouze ke kontrole rozdělení; test se nepoužije pro sestavení trénovacích trojic.
texts_by_split = {}
source_summary = {}
for split in ["train", "validation", "test"]:
    input_path = datasets_dir / f"{split}.jsonl"
    texts_by_split[split] = set()
    number_of_records = 0
    with input_path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                texts_by_split[split].add(record["text"].strip().casefold())
                number_of_records += 1
    source_summary[split] = {
        "path": str(input_path),
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "number_of_records": number_of_records,
    }

for first_split, second_split in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    overlap = texts_by_split[first_split].intersection(texts_by_split[second_split])
    if overlap:
        raise ValueError(f"Stejné texty ve splitech {first_split} a {second_split}: {len(overlap)}")

with (output_dir / "source_summary.json").open("w", encoding="utf-8") as file:
    json.dump(source_summary, file, ensure_ascii=False, indent=2)

# Train a validation zpracujeme odděleně. Test vůbec nečteme.
for split in ["train", "validation"]:
    # Vlastní generátor zajistí stejný výběr trojic při opakovaném spuštění.
    rng = random.Random(42)
    records = []
    with (datasets_dir / f"{split}.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    # 2. Slovník: intent_id -> seznam jeho vět.
    sentences_by_intent = {}
    for record in records:
        intent = record["intent_id"]
        if intent not in sentences_by_intent:
            sentences_by_intent[intent] = []
        sentences_by_intent[intent].append(record["text"])

    # 3. Pro každou in-scope větu sestavíme jednu trojici.
    triplets = []
    for record in records:
        intent = record["intent_id"]
        anchor = record["text"]

        # OOS není jeden význam. Jeho věty tedy nespojujeme jako pozitivní pár.
        # OOS ale zůstává dostupné mezi negativními příklady níže.
        if intent == "oos":
            continue

        # Positive musí mít stejný intent, ale jiný text než anchor.
        positives = []
        for text in sentences_by_intent[intent]:
            if text != anchor:
                positives.append(text)
        negatives = []
        for item in records:
            if item["intent_id"] != intent:
                negatives.append(item["text"])
        if not positives or not negatives:
            raise ValueError(f"Nelze sestavit trojici pro {split}: {intent}")

        triplets.append({
            "anchor": anchor,
            "positive": rng.choice(positives),
            "negative": rng.choice(negatives),
        })

    # 4. Trainer dostane jen tři textové sloupce v tomto pořadí.
    # Prompty přidá až trénovací skript, aby se neuložily do původních textů.
    rng.shuffle(triplets)
    output_path = output_dir / f"{split}_triplets.jsonl"
    with output_path.open("w", encoding="utf-8") as file:
        for triplet in triplets:
            file.write(json.dumps(triplet, ensure_ascii=False) + "\n")

    print(f"{split}: {len(records)} vět -> {len(triplets)} trojic, uloženo do {output_path}")
