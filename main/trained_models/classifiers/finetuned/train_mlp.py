from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

import csv
import json
import joblib
import numpy as np
import matplotlib

# Agg ukládá grafy do souborů, neotevírá žádná okna.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. Cesty k embeddingům a výsledkům této varianty modelu.
# parents[2] odkazuje ze složky classifiers/finetuned do trained_models.
current_dir = Path(__file__).resolve().parents[2]
experiment_dir = current_dir / "outputs" / "embeddinggemma_random_lora"
model_variant = "finetuned"
embeddings_dir = experiment_dir / "embeddings" / model_variant
results_dir = experiment_dir / "classifiers" / model_variant / Path(__file__).stem
results_dir.mkdir(parents=True, exist_ok=True)

# 2. Trénujeme na train a hledáme parametry na validation.
# Načtení trénovacích vektorů a jejich správných intentů.
with np.load(
    embeddings_dir / "train.npz",
    allow_pickle=False,
) as data:
    X_train = data["embeddings"]
    y_train = data["labels"]

# Načtení validačních vektorů pro výběr hyperparametru.
with np.load(
    embeddings_dir / "validation.npz",
    allow_pickle=False,
) as data:
    X_val = data["embeddings"]
    y_val = data["labels"]


print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Počet tříd:", len(np.unique(y_train)))

# Mocniny dvojky: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024.
# Deset velikostí pro každou vrstvu znamená 10 × 10 = 100 modelů.
list_layer_sizes = [2 ** power for power in range(1, 11)]
list_best_mlp = []
best_f1 = -1.0
best_y_pred = None
best_model = None
total_combinations = len(list_layer_sizes) ** 2

# Vnější smyčka volí první skrytou vrstvu, vnitřní druhou.
for first_layer_size in list_layer_sizes:
    for second_layer_size in list_layer_sizes:
        combination = len(list_best_mlp) + 1
        print(
            f"[{combination}/{total_combinations}] "
            f"Trénuji vrstvy ({first_layer_size}, {second_layer_size})...",
            flush=True,
        )

        # Každou kombinaci učíme od začátku na stejném train datasetu.
        # Ostatní parametry ponecháváme stejné pro férové porovnání.
        mlp = MLPClassifier(
            hidden_layer_sizes=(first_layer_size, second_layer_size),
            max_iter=1000,
            random_state=42,
        )
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)

        # Tuple obsahuje: první vrstvu, druhou vrstvu, accuracy a macro F1.
        list_best_mlp.append(
            (first_layer_size, second_layer_size, accuracy, f1_macro)
        )
        print(
            f"accuracy={accuracy:.4f}, macro F1={f1_macro:.4f}, "
            f"počet epoch={mlp.n_iter_}",
            flush=True,
        )

        # Uchováme předpovědi nejlepší kombinace pro závěrečný report.
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_y_pred = y_pred
            best_model = mlp

# Macro F1 je čtvrtá položka (index 3). Při shodě vyhraje první
# vyzkoušená kombinace v pořadí smyček.
best_first_layer, best_second_layer, best_accuracy, best_f1_macro = max(
    list_best_mlp, key=lambda result: result[3]
)
print(
    "\nNejlepší vrstvy podle validačního macro F1: "
    f"({best_first_layer}, {best_second_layer})"
)
print(f"Validation accuracy: {best_accuracy:.4f}")
print(f"Validation macro F1: {best_f1_macro:.4f}")
# Stejný report vypíšeme do terminálu a níže uložíme do souboru.
report = classification_report(y_val, best_y_pred, digits=3, zero_division=0)
print(report)

# Uložíme všechny kombinace i jejich metriky, aby šly později porovnat.
with (results_dir / "hyperparameters.csv").open("w", encoding="utf-8", newline="") as output:
    writer = csv.writer(output)
    writer.writerow(["first_layer", "second_layer", "accuracy", "macro_f1"])
    writer.writerows(list_best_mlp)

# Report patří validační sadě; test se při hledání parametrů nepoužívá.
with (results_dir / "best_result.txt").open("w", encoding="utf-8") as output:
    output.write(f"Embeddingy: {embeddings_dir}\n")
    output.write("Výběr podle: validation macro F1\n")
    output.write(f"Nejlepší vrstvy: ({best_first_layer}, {best_second_layer})\n")
    output.write(f"Validation accuracy: {best_accuracy:.6f}\n")
    output.write(f"Validation macro F1: {best_f1_macro:.6f}\n\n")
    output.write(report)

# Pořadí výsledků odpovídá dvěma trénovacím smyčkám.
# Řádky mapy = první vrstva, sloupce = druhá vrstva.
accuracy_grid = np.array([result[2] for result in list_best_mlp]).reshape(
    len(list_layer_sizes), len(list_layer_sizes)
)
f1_grid = np.array([result[3] for result in list_best_mlp]).reshape(
    len(list_layer_sizes), len(list_layer_sizes)
)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, grid, title in zip(
    axes, [accuracy_grid, f1_grid], ["Validation accuracy", "Validation macro F1"]
):
    heatmap = ax.imshow(grid, origin="lower", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(list_layer_sizes)), labels=list_layer_sizes, rotation=45)
    ax.set_yticks(range(len(list_layer_sizes)), labels=list_layer_sizes)
    ax.set_xlabel("Velikost druhé skryté vrstvy")
    ax.set_ylabel("Velikost první skryté vrstvy")
    ax.set_title(title)
    fig.colorbar(heatmap, ax=ax)

# Hvězdička označuje kombinaci vybranou podle macro F1.
axes[1].scatter(
    list_layer_sizes.index(best_second_layer),
    list_layer_sizes.index(best_first_layer),
    marker="*", s=220, color="red",
    label=f"Nejlepší vrstvy: ({best_first_layer}, {best_second_layer})",
)
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.suptitle("MLP: výběr velikostí obou vrstev na validation")

# Graf pouze uložíme. Při dalším běhu se soubory tohoto modelu přepíšou.
fig.tight_layout()
fig.savefig(results_dir / "hyperparameters.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("Výsledky uloženy do:", results_dir)

# 3. Až po výběru parametrů načteme test. Už podle něj nic nevybíráme.
# Nejlepší natrénovaný model jsme si odložili během smyčky, netrénujeme jej znovu.
with np.load(embeddings_dir / "test.npz", allow_pickle=False) as data:
    X_test = data["embeddings"]
    y_test = data["labels"]
    test_sentences = data["texts"]

y_test_pred = best_model.predict(X_test)
all_labels = np.unique(y_train)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_macro_f1 = f1_score(y_test, y_test_pred, labels=all_labels, average="macro", zero_division=0)
test_report = classification_report(
    y_test, y_test_pred, labels=all_labels, digits=4, zero_division=0,
)
test_report_dictionary = classification_report(
    y_test, y_test_pred, labels=all_labels, output_dict=True, zero_division=0,
)
validation_report_dictionary = classification_report(
    y_val, best_y_pred, labels=all_labels, output_dict=True, zero_division=0,
)

# 4. Uložíme přesný model, jeho testové predikce a statistiky po třídách.
joblib.dump(best_model, results_dir / "best_classifier.joblib")
with (results_dir / "test_report.txt").open("w", encoding="utf-8") as output:
    output.write("Test — parametry byly vybrány pouze na validation.\n\n")
    output.write(test_report)

with (results_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as output:
    for text, actual_intent, predicted_intent in zip(test_sentences, y_test, y_test_pred):
        prediction = {
            "text": str(text),
            "actual_intent": str(actual_intent),
            "predicted_intent": str(predicted_intent),
            "correct": bool(actual_intent == predicted_intent),
        }
        output.write(json.dumps(prediction, ensure_ascii=False) + "\n")

# JSON usnadní závěrečné porovnání bez čtení metrik z textového reportu.
metrics = {
    "model_variant": model_variant,
    "classifier": Path(__file__).stem,
    "best_parameters": {"hidden_layer_sizes": [int(best_first_layer), int(best_second_layer)], "max_iter": 1000, "random_state": 42},
    "selection_metric": "validation_macro_f1",
    "validation_accuracy": float(best_accuracy),
    "validation_macro_f1": float(best_f1_macro),
    "validation_report": validation_report_dictionary,
    "test_accuracy": float(test_accuracy),
    "test_macro_f1": float(test_macro_f1),
    "test_report": test_report_dictionary,
    "test_correct": int(np.sum(y_test == y_test_pred)),
    "test_total": len(y_test),
}
with (results_dir / "metrics.json").open("w", encoding="utf-8") as output:
    json.dump(metrics, output, ensure_ascii=False, indent=2)

# Řádky matice = skutečné intenty, sloupce = předpovězené intenty.
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_test_pred, labels=all_labels)
with (results_dir / "test_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as output:
    writer = csv.writer(output)
    writer.writerow(["actual / predicted"] + all_labels.tolist())
    for intent, row in zip(all_labels, matrix):
        writer.writerow([str(intent)] + row.tolist())

print("Test accuracy:", test_accuracy)
print("Test macro F1:", test_macro_f1)
print(test_report)
