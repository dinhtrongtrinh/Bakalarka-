from datasets import load_dataset
import json
from pathlib import Path


dataset = load_dataset(
    "fewshot-goes-multilingual/cs_squad-3.0",
    split="train",
)
candidates = dataset.shuffle(seed=42).select(range(50))

# output_path - absulutni cesta (presna dana adresa)
output_path = Path(__file__).resolve().parent / "datasets" / "dataset_OOS.jsonl"
#vytvori se nova slozka
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    for row in candidates:
        record = {
            "text": row["question"],
            "intent_id": "oos",
            "main_intent": "out_of_scope",
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
print(f"Uloženo {len(candidates)} záznamů do {output_path}")