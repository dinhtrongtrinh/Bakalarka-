# Knihovny: JSONL soubory, cesty, práce s vektory a embeddingové modely.
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# parent.parent je složka main. Cesty fungují nezávisle na místě spuštění.
project_dir = Path(__file__).resolve().parent.parent
model_cache_dir = project_dir / "embedding_models" / ".model_cache"

# Pro oba modely použijeme dostupnou NVIDIA GPU, jinak CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Použité zařízení:", device)

# Vstupy a výstupy pro rozdělení podle generátoru dat.
datasets_dir = project_dir / "split_datasets" / "by_generator"
base_output_dir = project_dir / "embedding_models" / "outputs" / "by_generator"

# Jediná vnější smyčka postupně zpracuje tři experimenty.
# Uvnitř jsou modely napsané zvlášť: nejprve Gemma, potom ModernBERT.
for evaluation_generator in ("openai", "gemini", "claude"):
    dataset_path = datasets_dir / f"eval_{evaluation_generator}"
    print()
    print("Experiment:", evaluation_generator)

    # Cesty ke třem částem aktuálního rozdělení.
    dataset_path_train = dataset_path / "train.jsonl"
    dataset_path_val = dataset_path / "validation.jsonl"
    dataset_path_test = dataset_path / "test.jsonl"

    # Načtení train: každý neprázdný řádek JSONL převedeme na slovník.
    records_train = []
    with dataset_path_train.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records_train.append(json.loads(line))

    # Stejným způsobem načteme validation.
    records_val = []
    with dataset_path_val.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records_val.append(json.loads(line))

    # Stejným způsobem načteme test.
    records_test = []
    with dataset_path_test.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records_test.append(json.loads(line))

    # Prázdné sady by nešlo smysluplně převést, proto je zkontrolujeme.
    if not records_train or not records_val or not records_test:
        raise ValueError("Train, validation i test musí obsahovat alespoň jednu větu.")

    # Správné intenty uložíme odděleně. Embeddingový model je nedostane jako vstup.
    labels_train = [record["intent_id"] for record in records_train]
    labels_val = [record["intent_id"] for record in records_val]
    labels_test = [record["intent_id"] for record in records_test]

    # Ze stejných záznamů ve stejném pořadí získáme texty. OOS nijak nefiltrujeme.
    sentences_train = [record["text"] for record in records_train]
    sentences_val = [record["text"] for record in records_val]
    sentences_test = [record["text"] for record in records_test]

    print("Počty vět:", len(sentences_train), len(sentences_val), len(sentences_test))


    # ==================== EMBEDDINGGEMMA ====================
    # EmbeddingGemma používá vlastní instrukci pro klasifikaci textu.
    model_name = "google/embeddinggemma-300m"
    prompt = "task: classification | query: "

    # Výstupy tohoto modelu mají vlastní složku, aby se modely nepřepisovaly.
    output_dir = base_output_dir / "embeddinggemma" / f"eval_{evaluation_generator}"
    output_path_train = output_dir / "train.npz"
    output_path_val = output_dir / "validation.npz"
    output_path_test = output_dir / "test.npz"

    # Pokud soubory existují, ověříme jejich texty, štítky a rozměry.
    # Jen existence souboru nestačí: po změně vstupních dat by mohl být zastaralý.
    saved_datasets = (
        (output_path_train, sentences_train, labels_train),
        (output_path_val, sentences_val, labels_val),
        (output_path_test, sentences_test, labels_test),
    )
    outputs_match = all(path.exists() for path, _, _ in saved_datasets)
    if outputs_match:
        for path, sentences, labels in saved_datasets:
            with np.load(path, allow_pickle=False) as data:
                if not (
                    np.array_equal(data["texts"], np.array(sentences))
                    and np.array_equal(data["labels"], np.array(labels))
                    and data["embeddings"].shape == (len(sentences), 768)
                    and ("model_name" not in data.files or data["model_name"].item() == model_name)
                    and ("prompt" not in data.files or data["prompt"].item() == prompt)
                ):
                    outputs_match = False
                    break

    # Přeskočíme jen tento model. Program potom pokračuje dalším blokem.
    if outputs_match:
        print("EMBEDDINGGEMMA: hotové vektory odpovídají vstupům, přeskakuji výpočet.")
    else:
        # Načtení modelu. Při prvním použití se může stáhnout; jeho váhy netrénujeme.
        model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(model_cache_dir),
        )

        # Train převedeme na NumPy matici: jedna věta = jeden řádek.
        # normalize_embeddings nastaví L2 délku na 1; batch_size je velikost dávky.
        embeddings_train = model.encode(
            sentences_train,
            prompt=prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=True,
        )

        # Validation převedeme stejným modelem a se stejnou instrukcí.
        embeddings_val = model.encode(
            sentences_val,
            prompt=prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=True,
        )

        # Test převedeme stejně. Žádné učení na jeho štítcích zde neprobíhá.
        embeddings_test = model.encode(
            sentences_test,
            prompt=prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=True,
        )

        # Každá sada musí mít jeden 768rozměrný vektor pro každý svůj štítek.
        assert embeddings_train.shape == (len(labels_train), 768)
        assert embeddings_val.shape == (len(labels_val), 768)
        assert embeddings_test.shape == (len(labels_test), 768)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Uložení train: vektory, štítky a texty zůstávají ve stejném pořadí.
        np.savez_compressed(
            output_path_train,
            embeddings=embeddings_train,
            labels=np.array(labels_train),
            texts=np.array(sentences_train),
            model_name=np.array(model_name),
            prompt=np.array(prompt),
        )

        # Stejné uložení validation do vlastního souboru.
        np.savez_compressed(
            output_path_val,
            embeddings=embeddings_val,
            labels=np.array(labels_val),
            texts=np.array(sentences_val),
            model_name=np.array(model_name),
            prompt=np.array(prompt),
        )

        # Stejné uložení test do vlastního souboru.
        np.savez_compressed(
            output_path_test,
            embeddings=embeddings_test,
            labels=np.array(labels_test),
            texts=np.array(sentences_test),
            model_name=np.array(model_name),
            prompt=np.array(prompt),
        )
        print("Uloženo:", output_dir)

        # Uvolnění modelu, aby další model zbytečně nesdílel paměť GPU s předchozím.
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


    # ==================== MODERNBERT ====================
    # Tohle je embeddingová varianta ModernBERT. Její klasifikační prefix
    # je "classification: ", takže nestačí změnit pouze název modelu.
    model_name = "nomic-ai/modernbert-embed-base"
    prompt = "classification: "

    # Výstupy tohoto modelu mají vlastní složku, aby se modely nepřepisovaly.
    output_dir = base_output_dir / "modernbert" / f"eval_{evaluation_generator}"
    output_path_train = output_dir / "train.npz"
    output_path_val = output_dir / "validation.npz"
    output_path_test = output_dir / "test.npz"

    # Pokud soubory existují, ověříme jejich texty, štítky a rozměry.
    # Jen existence souboru nestačí: po změně vstupních dat by mohl být zastaralý.
    saved_datasets = (
        (output_path_train, sentences_train, labels_train),
        (output_path_val, sentences_val, labels_val),
        (output_path_test, sentences_test, labels_test),
    )
    outputs_match = all(path.exists() for path, _, _ in saved_datasets)
    if outputs_match:
        for path, sentences, labels in saved_datasets:
            with np.load(path, allow_pickle=False) as data:
                if not (
                    np.array_equal(data["texts"], np.array(sentences))
                    and np.array_equal(data["labels"], np.array(labels))
                    and data["embeddings"].shape == (len(sentences), 768)
                    and ("model_name" not in data.files or data["model_name"].item() == model_name)
                    and ("prompt" not in data.files or data["prompt"].item() == prompt)
                ):
                    outputs_match = False
                    break

    # Přeskočíme jen tento model. Program potom pokračuje dalším blokem.
    if outputs_match:
        print("MODERNBERT: hotové vektory odpovídají vstupům, přeskakuji výpočet.")
    else:
        # ModernBERT Embed vytváří větné vektory. Použijeme běžnou pozornost
        # PyTorch (sdpa) a vypneme automatickou kompilaci, stejně jako v archivu.
        model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(model_cache_dir),
            model_kwargs={"attn_implementation": "sdpa"},
            config_kwargs={"reference_compile": False},
        )

        # Train převedeme na NumPy matici: jedna věta = jeden řádek.
        # normalize_embeddings nastaví L2 délku na 1; batch_size je velikost dávky.
        embeddings_train = model.encode(
            sentences_train,
            prompt=prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=True,
        )

        # Validation převedeme stejným modelem a se stejnou instrukcí.
        embeddings_val = model.encode(
            sentences_val,
            prompt=prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=True,
        )

        # Test převedeme stejně. Žádné učení na jeho štítcích zde neprobíhá.
        embeddings_test = model.encode(
            sentences_test,
            prompt=prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=True,
        )

        # Každá sada musí mít jeden 768rozměrný vektor pro každý svůj štítek.
        assert embeddings_train.shape == (len(labels_train), 768)
        assert embeddings_val.shape == (len(labels_val), 768)
        assert embeddings_test.shape == (len(labels_test), 768)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Uložení train: vektory, štítky a texty zůstávají ve stejném pořadí.
        np.savez_compressed(
            output_path_train,
            embeddings=embeddings_train,
            labels=np.array(labels_train),
            texts=np.array(sentences_train),
            model_name=np.array(model_name),
            prompt=np.array(prompt),
        )

        # Stejné uložení validation do vlastního souboru.
        np.savez_compressed(
            output_path_val,
            embeddings=embeddings_val,
            labels=np.array(labels_val),
            texts=np.array(sentences_val),
            model_name=np.array(model_name),
            prompt=np.array(prompt),
        )

        # Stejné uložení test do vlastního souboru.
        np.savez_compressed(
            output_path_test,
            embeddings=embeddings_test,
            labels=np.array(labels_test),
            texts=np.array(sentences_test),
            model_name=np.array(model_name),
            prompt=np.array(prompt),
        )
        print("Uloženo:", output_dir)

        # Uvolnění modelu, aby další model zbytečně nesdílel paměť GPU s předchozím.
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
