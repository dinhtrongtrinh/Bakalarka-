import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


project_dir = Path(__file__).resolve().parent.parent
datasets_dir = project_dir / "alza_scraper" / "datasets_by_generator"
output_dir = project_dir / "embedding_models" / "outputs_by_generator"
model_cache_dir = project_dir / "embedding_models" / ".model_cache"

# Oba modely tvoří 768rozměrné vektory, ale každý má vlastní vstupní instrukci.
# ModernBERT zde znamená variantu dotrénovanou pro embeddingy, ne základní MLM.
embedding_models = {
    "embeddinggemma": {
        "model_name": "google/embeddinggemma-300m",
        "prompt": "task: classification | query: ",
    },
    "modernbert": {
        "model_name": "nomic-ai/modernbert-embed-base",
        "prompt": "classification: ",
    },
}
evaluation_generators = ["openai", "gemini", "claude"]

# Načteme všech devět sad a vytvoříme seznam unikátních textů. Stejná věta se
# ve třech experimentech opakuje, ale pro každý model ji převedeme jen jednou.
datasets = {}
sentences = []
text_to_index = {}
for evaluation_generator in evaluation_generators:
    for split_name in ("train", "validation", "test"):
        dataset_path = datasets_dir / f"eval_{evaluation_generator}" / f"{split_name}.jsonl"
        with dataset_path.open(encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]
        if not records:
            raise ValueError(f"Dataset {dataset_path} je prázdný.")
        datasets[(evaluation_generator, split_name)] = records
        for record in records:
            text = record["text"]
            if text not in text_to_index:
                text_to_index[text] = len(sentences)
                sentences.append(text)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    # Omezíme počet CPU vláken, aby malé dávky nezpomalovala jejich režie.
    torch.set_num_threads(4)
print(f"Unikátních vět: {len(sentences)}, zařízení: {device}", flush=True)

for model_alias, settings in embedding_models.items():
    model_name = settings["model_name"]
    prompt = settings["prompt"]
    model_output_dir = output_dir / model_alias
    model_output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_output_dir / "sentence_embeddings.npz"

    embeddings = np.zeros((len(sentences), 768), dtype=np.float32)
    available = np.zeros(len(sentences), dtype=bool)
    reused_files = []
    model_revision = "unknown"

    # Nejprve zkusíme vlastní cache. U Gemmy navíc využijeme již vypočtené
    # původní embeddingy se stejným modelem, klasifikační instrukcí a normalizací.
    # Staré soubory nemají uloženou revizi modelu; tuto skutečnost zachováme.
    cache_paths = [cache_path]
    legacy_paths = []
    if model_alias == "embeddinggemma":
        legacy_paths = sorted(
            (project_dir / "embedding_models" / "outputs_by_AImodels").glob("*.npz")
        )
        cache_paths.extend(legacy_paths)

    for saved_path in cache_paths:
        if not saved_path.exists():
            continue
        with np.load(saved_path, allow_pickle=False) as data:
            if saved_path not in legacy_paths:
                if data["model_name"].item() != model_name or data["prompt"].item() != prompt:
                    continue
            saved_embeddings = data["embeddings"]
            saved_texts = data["texts"]
            if saved_embeddings.shape != (len(saved_texts), 768):
                raise ValueError(f"Chybný tvar uložených embeddingů: {saved_path}")
            if not np.isfinite(saved_embeddings).all() or not np.allclose(
                np.linalg.norm(saved_embeddings, axis=1), 1, atol=1e-5
            ):
                raise ValueError(f"Neplatné nebo nenormalizované embeddingy: {saved_path}")
            reused_count = 0
            for text, vector in zip(saved_texts, saved_embeddings):
                index = text_to_index.get(str(text))
                if index is not None and not available[index]:
                    embeddings[index] = vector
                    available[index] = True
                    reused_count += 1
            if reused_count:
                if "reused_files" in data.files:
                    reused_files.extend(str(path) for path in data["reused_files"])
                else:
                    reused_files.append(str(saved_path.relative_to(project_dir)))
                model_revision = (
                    data["model_revision"].item()
                    if "model_revision" in data.files else "legacy_unrecorded"
                )
                print(f"{model_alias}: použito {reused_count} vektorů z {saved_path.name}", flush=True)

    missing_indices = np.flatnonzero(~available)
    if len(missing_indices):
        print(f"{model_alias}: počítám {len(missing_indices)} chybějících vět", flush=True)
        model_options = {}
        if model_alias == "modernbert":
            # Použijeme standardní pozornost PyTorch bez dodatečného FlashAttention
            # a bez automatické kompilace. Funguje na CPU i CUDA.
            model_options = {
                "model_kwargs": {"attn_implementation": "sdpa"},
                "config_kwargs": {"reference_compile": False},
            }
        model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(model_cache_dir),
            **model_options,
        )
        computed = model.encode(
            [sentences[index] for index in missing_indices],
            prompt=prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=True,
        )
        if computed.shape != (len(missing_indices), 768):
            raise ValueError(f"Neočekávaný tvar embeddingů: {computed.shape}")
        embeddings[missing_indices] = computed
        available[missing_indices] = True
        revision = getattr(model[0].auto_model.config, "_commit_hash", None)
        if not revision:
            revision_path = model_cache_dir / ("models--" + model_name.replace("/", "--")) / "refs" / "main"
            if revision_path.exists():
                revision = revision_path.read_text(encoding="utf-8").strip()
        if revision:
            model_revision = revision if not reused_files else f"mixed_cache_and_{revision}"
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not available.all() or not np.isfinite(embeddings).all():
        raise ValueError(f"{model_alias}: některé embeddingy chybí nebo nejsou platné.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5):
        raise ValueError(f"{model_alias}: vektory nemají jednotkovou L2 délku.")

    # Cache obsahuje pouze texty a jejich vektory. Štítky načítáme ze správné
    # sady až při exportu; při tvorbě embeddingů se z nich model nic neučí.
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        texts=np.array(sentences),
        model_name=np.array(model_name),
        prompt=np.array(prompt),
        model_revision=np.array(model_revision),
        reused_files=np.array(reused_files, dtype=str),
    )

    for (evaluation_generator, split_name), records in datasets.items():
        experiment_output_dir = model_output_dir / f"eval_{evaluation_generator}"
        experiment_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = experiment_output_dir / f"{split_name}.npz"
        texts = [record["text"] for record in records]
        labels = [record["intent_id"] for record in records]
        indices = [text_to_index[text] for text in texts]
        np.savez_compressed(
            output_path,
            embeddings=embeddings[indices],
            labels=np.array(labels),
            texts=np.array(texts),
            model_name=np.array(model_name),
            prompt=np.array(prompt),
            model_revision=np.array(model_revision),
        )
        print(f"Uloženo: {output_path} | tvar: {embeddings[indices].shape}", flush=True)
