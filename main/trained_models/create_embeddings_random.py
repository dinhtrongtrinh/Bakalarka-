import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# 1. Nastavení přečteme z tréninku, aby se nelišil prompt ani délka vstupu.
current_dir = Path(__file__).resolve().parent
experiment_dir = current_dir / "outputs" / "embeddinggemma_random_lora"
datasets_dir = current_dir.parent / "split_datasets" / "random"
with (experiment_dir / "model_info.json").open(encoding="utf-8") as file:
    model_info = json.load(file)
prompt = model_info["prompt"]
max_seq_length = model_info["max_seq_length"]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Změna datasetu po tréninku by zneplatnila porovnání.
for split in ["train", "validation", "test"]:
    actual_hash = hashlib.sha256((datasets_dir / f"{split}.jsonl").read_bytes()).hexdigest()
    if actual_hash != model_info["source_splits"][split]["sha256"]:
        raise ValueError(f"Split {split} se od tréninku změnil.")

# 2. Načteme původní věty, správné intenty i OOS. Trojice už zde nepotřebujeme.
records_train = []
with (datasets_dir / "train.jsonl").open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records_train.append(json.loads(line))

records_validation = []
with (datasets_dir / "validation.jsonl").open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records_validation.append(json.loads(line))

records_test = []
with (datasets_dir / "test.jsonl").open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records_test.append(json.loads(line))

sentences_train = []
labels_train = []
for record in records_train:
    sentences_train.append(record["text"])
    labels_train.append(record["intent_id"])

sentences_validation = []
labels_validation = []
for record in records_validation:
    sentences_validation.append(record["text"])
    labels_validation.append(record["intent_id"])

sentences_test = []
labels_test = []
for record in records_test:
    sentences_test.append(record["text"])
    labels_test.append(record["intent_id"])

# 3. ==================== PŮVODNÍ EMBEDDINGGEMMA ====================
# Obě varianty počítáme znovu se stejnou délkou, promptem a přesností.
# Staré NPZ z původních experimentů proto nepřepisujeme ani nemícháme.
output_dir = experiment_dir / "embeddings" / "baseline"
output_dir.mkdir(parents=True, exist_ok=True)
model = SentenceTransformer(
    model_info["base_model"],
    revision=model_info["base_revision"],
    device=device,
)
model.max_seq_length = max_seq_length
model.float()

embeddings_train = model.encode(
    sentences_train,
    prompt=prompt,
    batch_size=8,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)
embeddings_validation = model.encode(
    sentences_validation,
    prompt=prompt,
    batch_size=8,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)
embeddings_test = model.encode(
    sentences_test,
    prompt=prompt,
    batch_size=8,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

# Jeden řádek = jedna věta. Pořadí textů, štítků a vektorů musí souhlasit.
for vectors, sentences in [
    (embeddings_train, sentences_train),
    (embeddings_validation, sentences_validation),
    (embeddings_test, sentences_test),
]:
    if vectors.shape != (len(sentences), 768):
        raise ValueError("Nesprávný rozměr embeddingů.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddingy obsahují NaN nebo nekonečno.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        raise ValueError("Embeddingy nejsou normalizované.")

np.savez_compressed(
    output_dir / "train.npz",
    embeddings=embeddings_train,
    labels=np.array(labels_train),
    texts=np.array(sentences_train),
    model_name=model_info["base_model"],
    prompt=prompt,
    max_seq_length=max_seq_length,
)
np.savez_compressed(
    output_dir / "validation.npz",
    embeddings=embeddings_validation,
    labels=np.array(labels_validation),
    texts=np.array(sentences_validation),
    model_name=model_info["base_model"],
    prompt=prompt,
    max_seq_length=max_seq_length,
)
np.savez_compressed(
    output_dir / "test.npz",
    embeddings=embeddings_test,
    labels=np.array(labels_test),
    texts=np.array(sentences_test),
    model_name=model_info["base_model"],
    prompt=prompt,
    max_seq_length=max_seq_length,
)
print("Embeddingy uloženy do:", output_dir)

# Před načtením dalšího modelu nebo ukončením uvolníme jeho GPU paměť.
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 4. ==================== EMBEDDINGGEMMA PO FINE-TUNINGU ====================
# Obě varianty počítáme znovu se stejnou délkou, promptem a přesností.
# Staré NPZ z původních experimentů proto nepřepisujeme ani nemícháme.
output_dir = experiment_dir / "embeddings" / "finetuned"
output_dir.mkdir(parents=True, exist_ok=True)
model_path = experiment_dir / "merged_model"
if not (model_path / "modules.json").is_file():
    raise FileNotFoundError("Chybí sloučený SentenceTransformer model. Dokonči nejdřív trénink.")
model = SentenceTransformer(str(model_path), device=device)
model.max_seq_length = max_seq_length
model.float()

embeddings_train = model.encode(
    sentences_train,
    prompt=prompt,
    batch_size=8,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)
embeddings_validation = model.encode(
    sentences_validation,
    prompt=prompt,
    batch_size=8,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)
embeddings_test = model.encode(
    sentences_test,
    prompt=prompt,
    batch_size=8,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

# Jeden řádek = jedna věta. Pořadí textů, štítků a vektorů musí souhlasit.
for vectors, sentences in [
    (embeddings_train, sentences_train),
    (embeddings_validation, sentences_validation),
    (embeddings_test, sentences_test),
]:
    if vectors.shape != (len(sentences), 768):
        raise ValueError("Nesprávný rozměr embeddingů.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddingy obsahují NaN nebo nekonečno.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        raise ValueError("Embeddingy nejsou normalizované.")

np.savez_compressed(
    output_dir / "train.npz",
    embeddings=embeddings_train,
    labels=np.array(labels_train),
    texts=np.array(sentences_train),
    model_name=str(model_path),
    prompt=prompt,
    max_seq_length=max_seq_length,
)
np.savez_compressed(
    output_dir / "validation.npz",
    embeddings=embeddings_validation,
    labels=np.array(labels_validation),
    texts=np.array(sentences_validation),
    model_name=str(model_path),
    prompt=prompt,
    max_seq_length=max_seq_length,
)
np.savez_compressed(
    output_dir / "test.npz",
    embeddings=embeddings_test,
    labels=np.array(labels_test),
    texts=np.array(sentences_test),
    model_name=str(model_path),
    prompt=prompt,
    max_seq_length=max_seq_length,
)
print("Embeddingy uloženy do:", output_dir)

# Před načtením dalšího modelu nebo ukončením uvolníme jeho GPU paměť.
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

