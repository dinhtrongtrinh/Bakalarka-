from pathlib import Path
import argparse
import json

import joblib
import torch
from sentence_transformers import SentenceTransformer


# 1. Jedna nová česká věta jako argument při spuštění skriptu.
parser = argparse.ArgumentParser(description="Predikce intentu natrénovanou pipeline.")
parser.add_argument("text", help="Věta zákazníka v uvozovkách.")
arguments = parser.parse_args()

current_dir = Path(__file__).resolve().parent
experiment_dir = current_dir / "outputs" / "embeddinggemma_random_lora"
with (experiment_dir / "model_info.json").open(encoding="utf-8") as file:
    model_info = json.load(file)
with (experiment_dir / "comparison" / "selected_classifiers.json").open(encoding="utf-8") as file:
    selected_classifiers = json.load(file)

# 2. Načteme embedding model po fine-tuningu a klasifikátor vybraný na validaci.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(str(experiment_dir / "merged_model"), device=device)
model.max_seq_length = model_info["max_seq_length"]
model.float()
classifier_name = selected_classifiers["finetuned"]["classifier"]
classifier_path = experiment_dir / "classifiers" / "finetuned" / classifier_name / "best_classifier.joblib"
classifier = joblib.load(classifier_path)

# 3. Stejný prompt a normalizace jako při tvorbě trénovacích embeddingů.
embedding = model.encode(
    [arguments.text],
    prompt=model_info["prompt"],
    normalize_embeddings=True,
    convert_to_numpy=True,
)
prediction = classifier.predict(embedding)
print("Text:", arguments.text)
print("Použitý klasifikátor:", classifier_name)
print("Předpovězený intent:", prediction[0])
