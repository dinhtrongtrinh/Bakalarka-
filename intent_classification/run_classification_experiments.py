import argparse
import csv
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import platform
import traceback
import warnings

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
    log_loss, matthews_corrcoef,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_info, threadpool_limits


project_dir = Path(__file__).resolve().parent.parent


def write_json(path, data):
    # Dočasný soubor brání tomu, aby přerušení zanechalo napůl zapsaný JSON.
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    temporary_path.replace(path)


def write_csv(path, records):
    if not records:
        return
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(path)


def write_predictions(path, predictions):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(file, **predictions)
    temporary_path.replace(path)


def parameter_grid(classifier):
    if classifier == "logistic_regression":
        return [
            {"C": float(c), "solver": "lbfgs", "max_iter": 3000, "l1_ratio": 0.0}
            for c in np.geomspace(0.001, 1000, 20)
        ]
    if classifier == "knn":
        return [
            {"n_neighbors": k, "weights": weight, "metric": "cosine", "algorithm": "brute"}
            for k in range(1, 21)
            for weight in ("uniform", "distance")
        ]
    return [
        {
            "hidden_layer_sizes": [first, second], "activation": "relu",
            "solver": "adam", "alpha": 0.0001, "max_iter": 1000,
            "random_state": 42, "early_stopping": False,
        }
        for first in (2 ** power for power in range(1, 11))
        for second in (2 ** power for power in range(1, 11))
    ]


def create_model(classifier, parameters):
    if classifier == "logistic_regression":
        return LogisticRegression(**parameters)
    if classifier == "knn":
        return KNeighborsClassifier(**parameters)
    parameters = dict(parameters)
    parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
    return MLPClassifier(**parameters)


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        dataset = {name: data[name].copy() for name in ("embeddings", "labels", "texts")}
    x, y, texts = dataset["embeddings"], dataset["labels"], dataset["texts"]
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(texts) or not len(y):
        raise ValueError(f"Nesouhlasí rozměry dat: {path}")
    if not np.isfinite(x).all():
        raise ValueError(f"Neplatné hodnoty embeddingů: {path}")
    return dataset


def evaluate(model, dataset, labels):
    x, truth = dataset["embeddings"], dataset["labels"]
    started = perf_counter()
    predicted = model.predict(x)
    predict_seconds = perf_counter() - started
    started = perf_counter()
    probabilities = model.predict_proba(x)
    probability_seconds = perf_counter() - started
    if not np.array_equal(model.classes_, labels):
        raise ValueError("Pořadí tříd modelu neodpovídá pořadí uložených statistik.")

    report = classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=labels)
    per_class = []
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        fn = int(matrix[index].sum()) - tp
        fp = int(matrix[:, index].sum()) - tp
        tn = len(truth) - tp - fn - fp
        row = report[str(label)]
        per_class.append({
            "intent_id": str(label), "support": int(row["support"]),
            "predicted_count": tp + fp, "correct_count": tp,
            "precision": row["precision"], "recall": row["recall"],
            "f1": row["f1-score"], "class_accuracy": row["recall"],
            "one_vs_rest_accuracy": (tp + tn) / len(truth),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    metrics = {
        "samples": len(truth), "correct": int(np.count_nonzero(truth == predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(report["macro avg"]["recall"]),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "log_loss": float(log_loss(truth, probabilities, labels=labels)),
        "matthews_correlation": float(matthews_corrcoef(truth, predicted)),
        "cohen_kappa": float(cohen_kappa_score(truth, predicted, labels=labels)),
        "predict_seconds": predict_seconds, "predict_proba_seconds": probability_seconds,
        "predict_ms_per_sentence": 1000 * predict_seconds / len(truth),
        "classification_report": report, "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
    predictions = {
        "texts": dataset["texts"], "true_labels": truth,
        "predicted_labels": predicted, "probabilities": probabilities,
        "classes": labels,
    }
    return metrics, predictions


def fit_model(classifier, parameters, dataset):
    model = create_model(classifier, parameters)
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(dataset["embeddings"], dataset["labels"])
    return model, perf_counter() - started, [
        {"category": warning.category.__name__, "message": str(warning.message)}
        for warning in caught
    ]


def search_row(trial):
    row = {
        "trial_id": trial["trial_id"],
        "parameters": json.dumps(trial["parameters"], ensure_ascii=False),
        "fit_seconds": trial["fit_seconds"], "convergence_warning": trial["convergence_warning"],
        "n_iter": json.dumps(trial["n_iter"]), "loss": trial["loss"],
    }
    for split in ("train", "validation"):
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "log_loss"):
            row[f"{split}_{metric}"] = trial[split][metric]
    return row


def save_best_model(path, model, trial_id):
    temporary_path = path.with_suffix(".tmp")
    joblib.dump({"model": model, "trial_id": trial_id}, temporary_path, compress=3)
    temporary_path.replace(path)


def run_experiment(job):
    # Jedna úloha = jeden embedding, vynechaný generátor a typ klasifikátoru.
    # Každý proces používá jedno BLAS vlákno, aby se paralelní běhy nepřetěžovaly.
    with threadpool_limits(limits=1):
        return run_experiment_single_thread(job)


def run_experiment_single_thread(job):
    experiment_started = perf_counter()
    folder = Path(job["output_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != job:
            raise ValueError(f"Výsledky v {folder} patří jiným vstupům nebo nastavení. Zvol jinou výstupní složku.")
    else:
        write_json(manifest_path, job)
    completed_path = folder / "result.json"
    if completed_path.exists():
        return json.loads(completed_path.read_text(encoding="utf-8"))

    train = load_dataset(Path(job["inputs"]["train"]["path"]))
    validation = load_dataset(Path(job["inputs"]["validation"]["path"]))
    labels = np.unique(train["labels"])
    if set(validation["labels"]) != set(labels):
        raise ValueError("Train a validation neobsahují stejné třídy.")
    train_texts = {str(text).strip().casefold() for text in train["texts"]}
    if train_texts & {str(text).strip().casefold() for text in validation["texts"]}:
        raise ValueError("Stejné texty v train a validation.")

    trials_dir = folder / "trials"
    trials_dir.mkdir(exist_ok=True)
    trials = []
    best_f1 = -1.0
    model_path = folder / "best_model.joblib"
    for number, parameters in enumerate(job["parameter_grid"], start=1):
        trial_id = f"trial_{number:03d}"
        trial_path = trials_dir / f"{trial_id}.json"
        predictions_path = trials_dir / f"{trial_id}_predictions.npz"
        if trial_path.exists() and predictions_path.exists():
            trial = json.loads(trial_path.read_text(encoding="utf-8"))
            if trial["parameters"] != parameters:
                raise ValueError(f"Nesouhlasí parametry uloženého pokusu {trial_id}.")
            model = None
        else:
            model, fit_seconds, caught = fit_model(job["classifier"], parameters, train)
            train_metrics, train_predictions = evaluate(model, train, labels)
            validation_metrics, validation_predictions = evaluate(model, validation, labels)
            n_iter = np.asarray(getattr(model, "n_iter_", [])).tolist()
            loss = getattr(model, "loss_", None)
            trial = {
                "trial_id": trial_id, "parameters": parameters,
                "fit_seconds": fit_seconds, "n_iter": n_iter,
                "loss": float(loss) if loss is not None else None,
                "loss_curve": [float(value) for value in getattr(model, "loss_curve_", [])],
                "warnings": caught,
                "convergence_warning": any(w["category"] == ConvergenceWarning.__name__ for w in caught),
                "train": train_metrics, "validation": validation_metrics,
            }
            write_predictions(predictions_path, {
                **{f"train_{key}": value for key, value in train_predictions.items()},
                **{f"validation_{key}": value for key, value in validation_predictions.items()},
            })
            write_json(trial_path, trial)
        trials.append(trial)
        if trial["validation"]["macro_f1"] > best_f1:
            best_f1 = trial["validation"]["macro_f1"]
            if model is not None:
                save_best_model(model_path, model, trial_id)
        write_csv(folder / "search_results.csv", [search_row(t) for t in trials])
        write_json(folder / "progress.json", {
            "completed_trials": len(trials), "total_trials": len(job["parameter_grid"]),
            "best_validation_macro_f1": best_f1, "last_trial": trial_id,
        })
        if number % 10 == 0 or number == len(job["parameter_grid"]):
            print(f"{job['embedding']}/{job['experiment']}/{job['classifier']}: {number}/{len(job['parameter_grid'])}, nejlepší val F1={best_f1:.4f}", flush=True)

    # Vítěze vybíráme výhradně podle validace. Při shodě první pokus v pořadí.
    best = max(trials, key=lambda trial: trial["validation"]["macro_f1"])
    saved = joblib.load(model_path) if model_path.exists() else None
    reconstructed = saved is None or saved["trial_id"] != best["trial_id"]
    reconstruction_warnings = []
    reconstruction_seconds = 0.0
    if reconstructed:
        model, reconstruction_seconds, reconstruction_warnings = fit_model(job["classifier"], best["parameters"], train)
        save_best_model(model_path, model, best["trial_id"])
    else:
        model = saved["model"]

    # Test načítáme a hodnotíme až po uzavření hledání hyperparametrů.
    test = load_dataset(Path(job["inputs"]["test"]["path"]))
    test_texts = {str(text).strip().casefold() for text in test["texts"]}
    if set(test["labels"]) != set(labels) or train_texts & test_texts:
        raise ValueError("Nesouhlasí třídy testu nebo se překrývá s train.")
    if test_texts & {str(text).strip().casefold() for text in validation["texts"]}:
        raise ValueError("Stejné texty v validation a test.")
    test_metrics, test_predictions = evaluate(model, test, labels)
    write_predictions(folder / "test_predictions.npz", test_predictions)
    prediction_rows = [
        {
            "text": str(text), "true_intent": str(truth), "predicted_intent": str(predicted),
            "correct": bool(truth == predicted), "max_probability": float(probability.max()),
        }
        for text, truth, predicted, probability in zip(
            test_predictions["texts"], test_predictions["true_labels"],
            test_predictions["predicted_labels"], test_predictions["probabilities"]
        )
    ]
    write_csv(folder / "test_predictions.csv", prediction_rows)
    result = {
        "embedding": job["embedding"], "experiment": job["experiment"],
        "classifier": job["classifier"], "best_trial": best["trial_id"],
        "best_parameters": best["parameters"], "selection_metric": "validation_macro_f1",
        "tie_break": "first_trial", "final_training_split": "train",
        "labels": labels.tolist(), "trials": len(trials),
        "estimator_parameters": model.get_params(),
        "model_file_bytes": model_path.stat().st_size,
        "experiment_wall_seconds": perf_counter() - experiment_started,
        "search_fit_seconds": sum(trial["fit_seconds"] for trial in trials),
        "convergence_warning_trials": sum(trial["convergence_warning"] for trial in trials),
        "best_model_reconstructed": reconstructed,
        "reconstruction_fit_seconds": reconstruction_seconds,
        "reconstruction_warnings": reconstruction_warnings,
        "train": best["train"], "validation": best["validation"], "test": test_metrics,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    for split in ("train", "validation", "test"):
        write_csv(folder / f"{split}_per_class.csv", result[split]["per_class"])
        matrix_rows = [
            {"true_intent": label, **{predicted: result[split]["confusion_matrix"][i][j] for j, predicted in enumerate(result["labels"])}}
            for i, label in enumerate(result["labels"])
        ]
        write_csv(folder / f"{split}_confusion_matrix.csv", matrix_rows)
    write_json(completed_path, result)
    return result


def summary_row(result):
    row = {key: result[key] for key in ("embedding", "experiment", "classifier", "best_trial", "trials")}
    row["best_parameters"] = json.dumps(result["best_parameters"], ensure_ascii=False)
    row["search_fit_seconds"] = result["search_fit_seconds"]
    row["experiment_wall_seconds"] = result["experiment_wall_seconds"]
    row["model_file_bytes"] = result["model_file_bytes"]
    row["convergence_warning_trials"] = result["convergence_warning_trials"]
    for split in ("train", "validation", "test"):
        for metric in ("samples", "correct", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1", "log_loss", "matthews_correlation", "cohen_kappa", "predict_seconds", "predict_ms_per_sentence"):
            row[f"{split}_{metric}"] = result[split][metric]
    return row


def main():
    run_started = perf_counter()
    parser = argparse.ArgumentParser(description="Hledání klasifikátorů nad oběma embeddingy a třemi generátory.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=project_dir / "intent_classification" / "results_by_generator")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers musí být alespoň 1")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    jobs = []
    # Nejprve krátké LR/k-NN experimenty, potom šest delších MLP experimentů.
    for classifier in ("logistic_regression", "knn", "mlp"):
        for embedding in ("embeddinggemma", "modernbert"):
            for experiment in ("eval_openai", "eval_gemini", "eval_claude"):
                inputs = {}
                for split in ("train", "validation", "test"):
                    path = project_dir / "embedding_models" / "outputs_by_generator" / embedding / experiment / f"{split}.npz"
                    inputs[split] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                jobs.append({
                    "embedding": embedding, "experiment": experiment, "classifier": classifier,
                    "output_dir": str(output_dir / embedding / experiment / classifier),
                    "inputs": inputs, "parameter_grid": parameter_grid(classifier),
                    "versions": versions, "source_sha256": source_hash, "blas_threads": 1,
                })
    write_json(output_dir / "run_settings.json", {
        "jobs": len(jobs), "trials": sum(len(job["parameter_grid"]) for job in jobs),
        "selection_metric": "validation_macro_f1", "versions": versions,
        "workers": args.workers, "cpu_count": os.cpu_count(),
        "metric_definitions": {
            "class_accuracy": "TP / (TP + FN), stejné jako recall skutečné třídy",
            "one_vs_rest_accuracy": "(TP + TN) / N",
            "confusion_matrix": "řádky = skutečné třídy, sloupce = předpovězené třídy; pořadí v labels",
            "max_probability": "výstup modelu, nikoli záruka kalibrované jistoty",
            "test_protocol": "test jen pro vítěze validačního hledání; bez přeučení na validation",
        },
        "threadpools_before_limit": threadpool_info(),
    })
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(run_experiment, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception:
                failure = {"job": job["output_dir"], "error": traceback.format_exc()}
                failures.append(failure)
                write_json(output_dir / "failures.json", failures)
                print(f"CHYBA: {job['output_dir']}\n{failure['error']}", flush=True)
                continue
            results.append(result)
            results.sort(key=lambda row: (row["embedding"], row["experiment"], row["classifier"]))
            write_csv(output_dir / "best_models.csv", [summary_row(row) for row in results])
            write_json(output_dir / "best_models.json", results)
            print(f"HOTOVO {len(results)}/18: {result['embedding']}/{result['experiment']}/{result['classifier']}; val F1={result['validation']['macro_f1']:.4f}, test F1={result['test']['macro_f1']:.4f}", flush=True)
    all_search = []
    for job in jobs:
        path = Path(job["output_dir"]) / "search_results.csv"
        if path.exists():
            with path.open(encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file):
                    all_search.append({"embedding": job["embedding"], "experiment": job["experiment"], "classifier": job["classifier"], **row})
    write_csv(output_dir / "all_search_results.csv", all_search)
    write_json(output_dir / "run_status.json", {
        "completed_experiments": len(results), "failed_experiments": len(failures),
        "completed_trials": len(all_search), "expected_trials": sum(len(job["parameter_grid"]) for job in jobs),
        "wall_seconds": perf_counter() - run_started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    if failures:
        raise SystemExit(f"Nedokončených experimentů: {len(failures)}. Opravený běh lze znovu spustit.")
    print(f"Všech 18 experimentů dokončeno. Statistiky: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
