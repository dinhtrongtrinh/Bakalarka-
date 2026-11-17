from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

import numpy as np


project_dir = Path(__file__).resolve().parent.parent
embeddings_dir = project_dir / "embedding_models" / "outputs"

#loading train vectors
with np.load(
    embeddings_dir / "train_embeddinggemma.npz",
    allow_pickle=False,
) as data:
    X_train = data["embeddings"]
    y_train = data["labels"]

#loading val vectors
with np.load(
    embeddings_dir / "validation_embeddinggemma.npz",
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
print(classification_report(y_val, best_y_pred, digits=3, zero_division=0))
