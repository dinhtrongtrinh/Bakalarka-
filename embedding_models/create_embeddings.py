import json
from pathlib import Path
import sys
import torch

import numpy as np
from sentence_transformers import SentenceTransformer

project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "embedding_models" / "outputs"

output_path_train = output_dir / "train_embeddinggemma.npz"
output_path_val = output_dir / "validation_embeddinggemma.npz"
output_path_test = output_dir / "test_embeddinggemma.npz"

if output_path_train.exists() and output_path_val.exists() and output_path_test.exists():
    print("Věty už máš embedovaný.")
    sys.exit(0)

dataset_path = project_dir / "alza_scraper" / "splited_datatsets"

dataset_path_train = dataset_path / "train.jsonl"
dataset_path_val = dataset_path / "validation.jsonl"
dataset_path_test = dataset_path / "test.jsonl"

records_train = []
records_val = []
records_test = []
with dataset_path_train.open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records_train.append(json.loads(line))

with dataset_path_val.open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records_val.append(json.loads(line))

with dataset_path_test.open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records_test.append(json.loads(line))

labels_train = [record["intent_id"] for record in records_train]
labels_val = [record["intent_id"] for record in records_val]
labels_test = [record["intent_id"] for record in records_test]

sentences_train = [record["text"] for record in records_train]
sentences_val = [record["text"] for record in records_val]
sentences_test = [record["text"] for record in records_test]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("google/embeddinggemma-300m", device=device)

embeddings_train = model.encode(
    sentences_train,
    prompt="task: classification | query: ",
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=16,
    show_progress_bar=True,
)
embeddings_val = model.encode(
    sentences_val,
    prompt="task: classification | query: ",
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=16,
    show_progress_bar=True,
)
embeddings_test = model.encode(
    sentences_test,
    prompt="task: classification | query: ",
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=16,
    show_progress_bar=True,
)
assert len(embeddings_train) == len(labels_train) == len(sentences_train)
assert len(embeddings_val) == len(labels_val) == len(sentences_val)
assert len(embeddings_test) == len(labels_test) == len(sentences_test)
output_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    output_path_train,
    embeddings=embeddings_train,
    labels=np.array(labels_train),
    texts=np.array(sentences_train),
)

np.savez_compressed(
    output_path_val,
    embeddings=embeddings_val,
    labels=np.array(labels_val),
    texts=np.array(sentences_val),
)

np.savez_compressed(
    output_path_test,
    embeddings=embeddings_test,
    labels=np.array(labels_test),
    texts=np.array(sentences_test),
)
