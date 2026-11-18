import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. Načteme statistiky všech šesti klasifikátorů, nikoli jejich konzolové výpisy.
current_dir = Path(__file__).resolve().parent
experiment_dir = current_dir / "outputs" / "embeddinggemma_random_lora"
output_dir = experiment_dir / "comparison"
output_dir.mkdir(parents=True, exist_ok=True)
classifier_names = ["train_knn", "train_logistic_regression", "train_mlp"]
all_results = []
results_by_variant = {"baseline": {}, "finetuned": {}}

for variant in ["baseline", "finetuned"]:
    for classifier_name in classifier_names:
        metrics_path = experiment_dir / "classifiers" / variant / classifier_name / "metrics.json"
        with metrics_path.open(encoding="utf-8") as file:
            metrics = json.load(file)
        all_results.append(metrics)
        results_by_variant[variant][classifier_name] = metrics

# 2. Před porovnáním ověříme, že oba modely pracovaly se stejnými větami.
for split in ["train", "validation", "test"]:
    with np.load(experiment_dir / "embeddings" / "baseline" / f"{split}.npz") as data:
        baseline_texts = data["texts"]
        baseline_labels = data["labels"]
    with np.load(experiment_dir / "embeddings" / "finetuned" / f"{split}.npz") as data:
        finetuned_texts = data["texts"]
        finetuned_labels = data["labels"]
    if not np.array_equal(baseline_texts, finetuned_texts):
        raise ValueError(f"Varianty mají jiné věty nebo pořadí ve splitu {split}.")
    if not np.array_equal(baseline_labels, finetuned_labels):
        raise ValueError(f"Varianty mají jiné štítky ve splitu {split}.")

# 3. Tabulka obsahuje všechny modely. Testové skóre slouží k vyhodnocení,
# nikoli k dodatečnému výběru vrstev, C, k nebo checkpointu embedding modelu.
columns = [
    "variant", "classifier", "best_parameters",
    "validation_accuracy", "validation_macro_f1",
    "test_correct", "test_total", "test_accuracy",
    "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_weighted_f1",
]
with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    for result in all_results:
        writer.writerow([
            result["model_variant"],
            result["classifier"],
            json.dumps(result["best_parameters"], ensure_ascii=False),
            result["validation_accuracy"],
            result["validation_macro_f1"],
            result["test_correct"],
            result["test_total"],
            result["test_accuracy"],
            result["test_report"]["macro avg"]["precision"],
            result["test_report"]["macro avg"]["recall"],
            result["test_macro_f1"],
            result["test_report"]["weighted avg"]["f1-score"],
        ])

# 4. Změnu vyjádříme v procentních bodech: 0.80 -> 0.83 je +3 p. b.
improvements = []
for classifier_name in classifier_names:
    baseline = results_by_variant["baseline"][classifier_name]
    finetuned = results_by_variant["finetuned"][classifier_name]
    improvement = {
        "classifier": classifier_name,
        "validation_macro_f1_change_pp": 100 * (finetuned["validation_macro_f1"] - baseline["validation_macro_f1"]),
        "test_macro_f1_change_pp": 100 * (finetuned["test_macro_f1"] - baseline["test_macro_f1"]),
        "test_accuracy_change_pp": 100 * (finetuned["test_accuracy"] - baseline["test_accuracy"]),
    }
    improvements.append(improvement)
    print(classifier_name)
    print("  Původní test macro F1:", baseline["test_macro_f1"])
    print("  Fine-tuned test macro F1:", finetuned["test_macro_f1"])
    print("  Změna v procentních bodech:", improvement["test_macro_f1_change_pp"])

# 5. Pro praktické používání vybereme klasifikátor podle VALIDACE.
# Při shodě ponecháme první v pořadí k-NN, logistická regrese, MLP.
selected_classifiers = {}
for variant in ["baseline", "finetuned"]:
    best_result = None
    for classifier_name in classifier_names:
        result = results_by_variant[variant][classifier_name]
        if best_result is None or result["validation_macro_f1"] > best_result["validation_macro_f1"]:
            best_result = result
    selected_classifiers[variant] = {
        "classifier": best_result["classifier"],
        "selection_metric": "validation_macro_f1",
        "validation_macro_f1": best_result["validation_macro_f1"],
        "test_macro_f1": best_result["test_macro_f1"],
    }

with (output_dir / "selected_classifiers.json").open("w", encoding="utf-8") as file:
    json.dump(selected_classifiers, file, ensure_ascii=False, indent=2)
with (output_dir / "comparison.json").open("w", encoding="utf-8") as file:
    json.dump({"results": all_results, "changes": improvements}, file, ensure_ascii=False, indent=2)

# 6. Graf ukazuje testové accuracy a macro F1 obou variant vedle sebe.
baseline_accuracies = []
finetuned_accuracies = []
baseline_f1_scores = []
finetuned_f1_scores = []
for classifier_name in classifier_names:
    baseline_accuracies.append(results_by_variant["baseline"][classifier_name]["test_accuracy"])
    finetuned_accuracies.append(results_by_variant["finetuned"][classifier_name]["test_accuracy"])
    baseline_f1_scores.append(results_by_variant["baseline"][classifier_name]["test_macro_f1"])
    finetuned_f1_scores.append(results_by_variant["finetuned"][classifier_name]["test_macro_f1"])

positions = np.arange(3)
figure, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].bar(positions - 0.2, baseline_accuracies, width=0.4, label="Původní model")
axes[0].bar(positions + 0.2, finetuned_accuracies, width=0.4, label="Fine-tuned model")
axes[0].set_title("Test accuracy")
axes[1].bar(positions - 0.2, baseline_f1_scores, width=0.4, label="Původní model")
axes[1].bar(positions + 0.2, finetuned_f1_scores, width=0.4, label="Fine-tuned model")
axes[1].set_title("Test macro F1")
for axis in axes:
    axis.set_xticks(positions, labels=["k-NN", "Logistická regrese", "MLP"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Hodnota metriky")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
figure.tight_layout()
figure.savefig(output_dir / "comparison.png", dpi=160)
plt.close(figure)
print("Porovnání uloženo do:", output_dir)
