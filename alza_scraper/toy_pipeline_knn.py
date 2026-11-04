# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers>=5.1,<6", "transformers>=4.56,<6", "scikit-learn>=1.5,<2"]
# ///
"""Toy baseline českých intentů: normalizované embeddingy -> cosine 3-NN.

Spuštění:
    uv run toy_pipeline_knn.py
    uv run toy_pipeline_knn.py /cesta/dataset.jsonl
    uv run toy_pipeline_knn.py --offline

Instalace bez uv:
    pip install 'sentence-transformers>=5.1,<6' 'transformers>=4.56,<6' 'scikit-learn>=1.5,<2'
    python toy_pipeline_knn.py /cesta/dataset.jsonl

EmbeddingGemma může vyžadovat přijetí licence na Hugging Face a HF_TOKEN.
Při nedostupnosti modelu se zkusí veřejný multilingual MiniLM. Offline režim
vyžaduje alespoň jeden z modelů již v lokální cache. Skript nemění dataset.
"""
import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys
import textwrap
import unicodedata

LOG = logging.getLogger("toy_knn")
PRIMARY_MODEL = "google/embeddinggemma-300m"
FALLBACK_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_FILENAME = "dataset_openai_gpt-4o-mini.jsonl"


def resolve_dataset(path=None):
    if path is not None:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate
        raise ValueError(f"Dataset nebyl nalezen: {candidate}. Zadejte platnou cestu k JSONL.")
    root = Path(__file__).resolve().parent
    candidates = [Path.cwd() / DEFAULT_FILENAME, Path.cwd() / "dataset_openai.jsonl",
                  Path.cwd() / "datasets" / DEFAULT_FILENAME,
                  root / "datasets" / DEFAULT_FILENAME, root / DEFAULT_FILENAME]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"Soubor {DEFAULT_FILENAME} ani dataset_openai.jsonl nebyl nalezen "
                     "v aktuální složce ani ve složce datasets. "
                     "Spusťte: python toy_pipeline_knn.py /cesta/k/datasetu.jsonl")


def load_sample(path, intent_count=5, train_per_intent=8, val_per_intent=2):
    """První intenty v pořadí souboru; pro každý první potřebný počet zpráv."""
    required = train_per_intent + val_per_intent
    groups, seen = {}, {}
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Neplatný JSON na řádku {line_number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Řádek {line_number} musí obsahovat JSON objekt.")
            text, label = row.get("text"), row.get("intent_id")
            if not isinstance(text, str) or not text.strip() or not isinstance(label, str) or not label.strip():
                raise ValueError(f"Řádek {line_number}: chybí neprázdný text nebo intent_id.")
            label, text = label.strip(), " ".join(text.split())
            if label not in groups:
                if len(groups) >= intent_count:
                    continue
                groups[label] = []
            # Přesná kopie zprávy nesmí skončit zároveň v trénování a validaci.
            key = unicodedata.normalize("NFKC", text).casefold()
            if key in seen:
                if seen[key] != label:
                    raise ValueError(f"Řádek {line_number}: stejná zpráva má dva různé intenty.")
                continue
            seen[key] = label
            if len(groups[label]) < required:
                groups[label].append(text)
    if len(groups) < intent_count:
        raise ValueError(f"Dataset obsahuje jen {len(groups)} intentů, požadováno {intent_count}.")
    insufficient = {label: len(texts) for label, texts in groups.items() if len(texts) < required}
    if insufficient:
        raise ValueError(f"Každý vybraný intent potřebuje {required} unikátních zpráv. "
                         f"Nedostatečné počty: {insufficient}")
    texts = [text for values in groups.values() for text in values]
    labels = [label for label, values in groups.items() for _ in values]
    return texts, labels


def split_sample(texts, labels, val_per_intent=2, seed=42):
    from sklearn.model_selection import train_test_split
    # U vyváženého vzorku 10 zpráv na třídu takto získáme přesně 8/2.
    # Stratifikace zachová zastoupení tříd; náhodné promíchání rozptýlí styly.
    return train_test_split(texts, labels, test_size=len(set(labels)) * val_per_intent,
                            stratify=labels, random_state=seed)


def embed_texts(train_texts, val_texts, offline=False, device="cpu", model_factory=None):
    import numpy as np
    if model_factory is None:
        from sentence_transformers import SentenceTransformer
        model_factory = SentenceTransformer
    failures = []
    for name in (PRIMARY_MODEL, FALLBACK_MODEL):
        for local_only in ((True,) if offline else (True, False)):
            model = None
            try:
                LOG.info("Načítám %s (%s).", name, "lokální cache" if local_only else "stažení / cache")
                model = model_factory(name, device=device, local_files_only=local_only)
                # Obě sady jsou zákaznické zprávy, proto používáme stejný prompt.
                # Prompt neobsahuje label; nepředáváme modelu správné odpovědi.
                prompt = "task: classification | query: " if name == PRIMARY_MODEL else ""
                options = dict(batch_size=16, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True, prompt=prompt)
                # L2 normalizace zajistí jednotkovou délku. Cosine potom porovnává
                # směr reprezentací; délka vektoru neovlivňuje podobnost zpráv.
                train = model.encode(train_texts, **options)
                val = model.encode(val_texts, **options)
                for vectors, expected in ((train, len(train_texts)), (val, len(val_texts))):
                    if (vectors.ndim != 2 or len(vectors) != expected or not np.isfinite(vectors).all()
                            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)):
                        raise ValueError("Model nevrátil platné normalizované embeddingy.")
                LOG.info("Použitý model: %s; rozměr embeddingu: %d.", name, train.shape[1])
                return train, val, name
            except Exception as error:
                # Selhání váhy/modelu či sítě dovoluje přechod na záložní model.
                # KeyboardInterrupt se nechytá; Ctrl+C normálně přeruší běh.
                failures.append(f"{name} ({'cache' if local_only else 'online'}): {type(error).__name__}")
                LOG.warning("Model %s není použitelný (%s). Zkouším další možnost.", name, type(error).__name__)
            finally:
                del model
    raise ValueError("Nepodařilo se načíst a použít ani jeden embeddingový model. "
                     "EmbeddingGemma může vyžadovat přijetí licence a HF_TOKEN; "
                     "MiniLM musí být v cache nebo dostupný ke stažení. " + "; ".join(failures))


def classify(train_vectors, train_labels, val_vectors):
    from sklearn.neighbors import KNeighborsClassifier
    # k=3 je jednoduchá baseline: hlasování není závislé jen na jednom vzoru.
    # Lichý počet omezuje remízy u dvou tříd; u více tříd jsou remízy stále možné.
    classifier = KNeighborsClassifier(n_neighbors=3, metric="cosine", algorithm="brute", weights="uniform")
    classifier.fit(train_vectors, train_labels)
    predicted = classifier.predict(val_vectors)
    probabilities = classifier.predict_proba(val_vectors)
    columns = {label: index for index, label in enumerate(classifier.classes_)}
    confidence = [float(probabilities[index, columns[label]]) for index, label in enumerate(predicted)]
    return predicted, confidence


def print_results(texts, true_labels, predicted, confidence):
    from sklearn.metrics import accuracy_score, classification_report
    print(f"\nCelková přesnost: {accuracy_score(true_labels, predicted):.2%}\n")
    print(classification_report(true_labels, predicted, labels=sorted(set(true_labels)),
                                digits=3, zero_division=0))
    print("Confidence = podíl hlasů mezi 3 sousedy (33,3 / 66,7 / 100 %), nikoli kalibrovaná pravděpodobnost.\n")
    widths = [58, max(14, max(map(len, true_labels))), max(17, max(map(len, predicted))), 12, 8]
    def table_row(values):
        return " | ".join(str(value).ljust(width) for value, width in zip(values, widths))
    print(table_row(["Původní text", "Skutečný intent", "Predikovaný intent", "Confidence", "Status"]))
    print("-+-".join("-" * width for width in widths))
    for text, actual, prediction, score in zip(texts, true_labels, predicted, confidence):
        lines = textwrap.wrap(text, width=widths[0]) or [""]
        print(table_row([lines[0], actual, prediction, f"{score:.1%}", "SPRAVNE" if actual == prediction else "CHYBA"]))
        for line in lines[1:]:
            print(table_row([line, "", "", "", ""]))
        print("-+-".join("-" * width for width in widths))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", help="Cesta k JSONL; výchozí dataset_openai_gpt-4o-mini.jsonl")
    parser.add_argument("--intents", type=int, default=5)
    parser.add_argument("--train-per-intent", type=int, default=8)
    parser.add_argument("--val-per-intent", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="Pouze embeddingové modely z lokální cache")
    parser.add_argument("--device", default="cpu", help="cpu (výchozí), cuda nebo mps")
    args = parser.parse_args(argv)
    if args.intents < 2 or args.train_per_intent < 3 or args.val_per_intent < 1:
        parser.error("Použijte alespoň 2 intenty, 3 trénovací a 1 validační zprávu na intent.")
    try:
        path = resolve_dataset(args.dataset)
        texts, labels = load_sample(path, args.intents, args.train_per_intent, args.val_per_intent)
        train_texts, val_texts, train_labels, val_labels = split_sample(texts, labels, args.val_per_intent, args.seed)
        LOG.info("Dataset: %s", path)
        LOG.info("Train (%d): %s", len(train_texts), dict(Counter(train_labels)))
        LOG.info("Validace (%d): %s", len(val_texts), dict(Counter(val_labels)))
        train_vectors, val_vectors, model_name = embed_texts(train_texts, val_texts, args.offline, args.device)
        predicted, confidence = classify(train_vectors, train_labels, val_vectors)
        print(f"\nEmbeddingový model: {model_name}")
        print_results(val_texts, val_labels, predicted, confidence)
        print("Toy výsledek na syntetických datech; malá validace není nezávislý reálný benchmark.")
        return 0
    except ImportError as error:
        LOG.error("Chybí závislost (%s). Použijte uv run toy_pipeline_knn.py nebo instalaci uvedenou v záhlaví.", error.name)
    except (OSError, ValueError, RuntimeError) as error:
        LOG.error("%s", error)
    except KeyboardInterrupt:
        LOG.warning("Běh přerušen uživatelem.")
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
