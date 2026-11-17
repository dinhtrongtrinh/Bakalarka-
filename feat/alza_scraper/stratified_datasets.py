import json
from collections import defaultdict
from pathlib import Path
from random import Random
from sklearn.model_selection import train_test_split

curr_dir = Path(__file__).resolve().parent

# Cesty odvozujeme od skriptu, aby fungovaly bez ohledu na místo spuštění.
datasets_dir = curr_dir / "datasets"

# Složku pro výsledky vytvoříme, pokud ještě neexistuje.
splited_datasets_dir = curr_dir / "splited_datatsets"
splited_datasets_dir.mkdir(parents=True, exist_ok=True)

# Vstup tvoří tři Alza datasety a dataset dotazů mimo doménu (OOS).
claude_datasets_dir = datasets_dir / "dataset_claude_claude-haiku-5-5.jsonl"
openai_datasets_dir = datasets_dir / "dataset_openai_gpt-4o-mini.jsonl"
gemini_datasets_dir = datasets_dir / "dataset_gemini_gemini-3.8-flash.jsonl"
oos_datasets_dir = datasets_dir / "dataset_OOS.jsonl"

# Každý neprázdný řádek JSONL převedeme na slovník. Zachováme všechna pole.
with open(claude_datasets_dir, "r", encoding="utf-8") as file:
    claude_datasets = [json.loads(line) for line in file if line.strip()]

with open(openai_datasets_dir, "r", encoding="utf-8") as file:
    openai_datasets = [json.loads(line) for line in file if line.strip()]

with open(gemini_datasets_dir, "r", encoding="utf-8") as file:
    gemini_datasets = [json.loads(line) for line in file if line.strip()]

with open(oos_datasets_dir, "r", encoding="utf-8") as file:
    oos_datasets = [json.loads(line) for line in file if line.strip()]

# Operátor + spojí seznamy záznamů do jednoho společného datasetu.
all_datasets = claude_datasets + openai_datasets + gemini_datasets + oos_datasets

# Seskupíme záznamy podle konkrétního intentu. defaultdict(list) vytvoří
# pro každý nový intent prázdný seznam. OOS má vlastní skupinu "oos".
records_by_intent = defaultdict(list)
for record in all_datasets:
    records_by_intent[record["intent_id"]].append(record)

train = []
validation = []
test = []

for intent_id, records in sorted(records_by_intent.items()):
    # Pro dvě postupná rozdělení potřebujeme alespoň čtyři záznamy.
    if len(records) < 4:
        raise ValueError(f"Intent {intent_id!r} má méně než 4 záznamy.")

    # Každý intent rozdělíme samostatně: 70 % train a 30 % zbytek.
    # random_state=42 umožní zopakovat stejný náhodný výběr.
    intent_train, remaining = train_test_split(
        records,
        test_size=0.30,
        shuffle=True,
        random_state=42,
    )

    # Polovina ze zbývajících 30 % odpovídá 15 % původní skupiny.
    # Při lichém počtu záznamů dostane test o jeden více než validace.
    intent_validation, intent_test = train_test_split(
        remaining,
        test_size=0.50,
        shuffle=True,
        random_state=42,
    )

    # extend přidá všechny záznamy dané části do společného seznamu.
    train.extend(intent_train)
    validation.extend(intent_validation)
    test.extend(intent_test)

    print(
        f"{intent_id}: celkem {len(records)}, "
        f"train {len(intent_train)}, validation {len(intent_validation)}, "
        f"test {len(intent_test)}"
    )

# Promícháme pořadí uvnitř hotových částí, aby věty nebyly řazené podle
# intentu. Příslušnost k train/validation/test už tím nezměníme.
random = Random(42)
for records in (train, validation, test):
    random.shuffle(records)

# Každou část uložíme zvlášť. items() vrací název části a její záznamy.
splits = {"train": train, "validation": validation, "test": test}

for split_name, records in splits.items():
    output_path = splited_datasets_dir / f"{split_name}.jsonl"

    # Režim "w" přepíše předchozí výsledek při opakovaném spuštění.
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            # Jeden JSON objekt na řádek; ensure_ascii=False zachová diakritiku.
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{split_name}: {len(records)} záznamů -> {output_path}")

print(f"Celkem: {len(all_datasets)} záznamů")
