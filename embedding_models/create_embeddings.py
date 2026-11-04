import json
from pathlib import Path
import torch

import numpy as np
from sentence_transformers import SentenceTransformer

project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "alza_scraper" / "splited_datatsets"

dataset_path_train = dataset_path / "train.jsonl"
dataset_path_val = dataset_path / "validation.jsonl"
dataset_path_test = dataset_path / "test.jsonl"

records = []
with dataset_path_train.open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records.append(json.loads(line))

with dataset_path_val.open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records.append(json.loads(line))

with dataset_path_test.open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records.append(json.loads(line))

sentences = [record["text"] for record in records]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("google/embeddinggemma-300m", device=device)

embeddings = model.encode(
    sentences,
    prompt="task: classification | query: ",
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=16,
    show_progress_bar=True,
)
labels = [record["intent_id"] for record in records]
print("Tvar embeddingů:", embeddings.shape)
print("Počet štítků:", len(labels))

assert len(embeddings) == len(labels)
