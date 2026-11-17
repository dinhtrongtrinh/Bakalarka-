import json
from collections import defaultdict
from pathlib import Path
from random import Random

from sklearn.model_selection import train_test_split


curr_dir = Path(__file__).resolve().parent
datasets_dir = curr_dir.parent / "datasets" / "raw"
output_dir = curr_dir / "by_generator"

# OpenAI označuje dataset generovaný přes OpenAI API (v zadání ChatGPT).
dataset_paths = {
    "openai": datasets_dir / "dataset_openai_gpt-4o-mini.jsonl",
    "gemini": datasets_dir / "dataset_gemini_gemini-3.8-flash.jsonl",
    "claude": datasets_dir / "dataset_claude_claude-haiku-5-5.jsonl",
}

# Každý generátor držíme zvlášť. Všechny původní položky záznamů zachováme.
datasets_by_generator = {}
for generator, dataset_path in dataset_paths.items():
    with dataset_path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if not records:
        raise ValueError(f"Dataset generátoru {generator} je prázdný.")
    datasets_by_generator[generator] = records

with (datasets_dir / "dataset_OOS.jsonl").open(encoding="utf-8") as file:
    oos_records = [json.loads(line) for line in file if line.strip()]

if len(oos_records) < 3:
    raise ValueError("OOS dataset musí obsahovat alespoň 3 záznamy.")
if any(record["intent_id"] != "oos" for record in oos_records):
    raise ValueError("V OOS datasetu musí mít všechny záznamy intent_id='oos'.")

# OOS rozdělíme jednou na třetiny a stejné části použijeme ve všech variantách.
# Při 50 větách vznikne 16 train, 17 validation a 17 test příkladů.
oos_train, oos_remaining = train_test_split(
    oos_records, test_size=2 / 3, shuffle=True, random_state=42
)
oos_validation, oos_test = train_test_split(
    oos_remaining, test_size=0.50, shuffle=True, random_state=42
)

# Postupně vynecháme OpenAI, Gemini a Claude z trénovacích dat.
for evaluation_generator, evaluation_records in datasets_by_generator.items():
    train_generators = [
        generator for generator in datasets_by_generator
        if generator != evaluation_generator
    ]

    train = []
    validation = []
    test = []
    for generator in train_generators:
        train.extend(datasets_by_generator[generator])

    # Vynechaný generátor poskytne validaci a test. Každý intent dělíme zvlášť,
    # aby byly jeho příklady zastoupené v obou sadách.
    records_by_intent = defaultdict(list)
    for record in evaluation_records:
        records_by_intent[record["intent_id"]].append(record)

    train_intents = {record["intent_id"] for record in train}
    if train_intents != set(records_by_intent):
        raise ValueError(f"Generátoři nemají stejné intenty ve variantě {evaluation_generator}.")

    for intent_id, records in sorted(records_by_intent.items()):
        if len(records) < 2:
            raise ValueError(f"{evaluation_generator}: intent {intent_id} má méně než 2 věty.")
        intent_validation, intent_test = train_test_split(
            records, test_size=0.50, shuffle=True, random_state=42
        )
        validation.extend(intent_validation)
        test.extend(intent_test)

    train.extend(oos_train)
    validation.extend(oos_validation)
    test.extend(oos_test)
    splits = {"train": train, "validation": validation, "test": test}

    # Stejná věta nesmí být současně v train a validation/test ani v obou
    # vyhodnocovacích sadách. Případnou duplicitu neopravujeme potichu.
    texts_by_split = {
        name: {record["text"].strip().casefold() for record in records}
        for name, records in splits.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if texts_by_split[left] & texts_by_split[right]:
            raise ValueError(f"eval_{evaluation_generator}: stejné texty v {left} a {right}.")

    experiment_dir = output_dir / f"eval_{evaluation_generator}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"\neval_{evaluation_generator}: train = {' + '.join(train_generators)}")

    # Promíchání mění pouze pořadí vět uvnitř sad. Režim w přepisuje výstupy
    # této varianty, takže opakované spuštění nepřidává duplicitní záznamy.
    random = Random(42)
    for split_name, records in splits.items():
        random.shuffle(records)
        output_path = experiment_dir / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        oos_count = sum(record["intent_id"] == "oos" for record in records)
        print(f"{split_name}: {len(records)} vět, z toho {oos_count} OOS -> {output_path}")

    # Popis varianty uložíme vedle dat pro pozdější kontrolu experimentu.
    summary = {
        "train_generators": train_generators,
        "evaluation_generator": evaluation_generator,
        "random_state": 42,
        "validation_fraction_of_evaluation_generator": 0.5,
        "counts": {name: len(records) for name, records in splits.items()},
        "oos_counts": {"train": len(oos_train), "validation": len(oos_validation), "test": len(oos_test)},
    }
    with (experiment_dir / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
