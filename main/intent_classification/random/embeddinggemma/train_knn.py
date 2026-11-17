from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

import csv
import numpy as np
import matplotlib

# Agg ukládá grafy do souborů, neotevírá žádná okna.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Od umístění tohoto skriptu odvodíme cestu do složky main.
project_dir = Path(__file__).resolve().parents[3]
# Každý experiment čte své vlastní embeddingy EmbeddingGemma.
embeddings_dir = project_dir / "embedding_models" / "outputs" / "random" / "embeddinggemma"

# Výsledky každého klasifikátoru mají vlastní složku vedle skriptu.
results_dir = Path(__file__).resolve().parent / "results" / Path(__file__).stem
results_dir.mkdir(parents=True, exist_ok=True)

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

# Vyzkoušíme počty sousedů od 1 do 20 včetně.
list_k = range(1, 21)
list_best_k = []
best_f1 = -1.0
best_y_pred = None

for k in list_k:
    # Pro každé k vytvoříme nový klasifikátor. Referenční příklady bere
    # pouze z train; úspěšnost měříme na validačních větách.
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)

    # Uložíme přímo k i jeho metriky, nikoli jen pozici v seznamu.
    list_best_k.append((k, accuracy, f1_macro))
    print(f"k={k}: accuracy={accuracy:.4f}, macro F1={f1_macro:.4f}")

    # Uchováme předpovědi nejlepšího modelu pro závěrečný report.
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_y_pred = y_pred

# Stejně jako u logistické regrese vybíráme podle třetí položky: macro F1.
# Při shodě vyhraje první vyzkoušené k, tedy menší počet sousedů.
best_k, best_accuracy, best_f1_macro = max(list_best_k, key=lambda result: result[2])
print(f"\nNejlepší k podle validačního macro F1: {best_k}")
print(f"Validation accuracy: {best_accuracy:.4f}")
print(f"Validation macro F1: {best_f1_macro:.4f}")
# Stejný report vypíšeme do terminálu a níže uložíme do souboru.
report = classification_report(y_val, best_y_pred, digits=3, zero_division=0)
print(report)

# Uložíme všechny kombinace i jejich metriky, aby šly později porovnat.
with (results_dir / "hyperparameters.csv").open("w", encoding="utf-8", newline="") as output:
    writer = csv.writer(output)
    writer.writerow(["k", "accuracy", "macro_f1"])
    writer.writerows(list_best_k)

# Report patří validační sadě; test se při hledání parametrů nepoužívá.
with (results_dir / "best_result.txt").open("w", encoding="utf-8") as output:
    output.write(f"Embeddingy: {embeddings_dir}\n")
    output.write("Výběr podle: validation macro F1\n")
    output.write(f"Nejlepší k: {best_k}\n")
    output.write(f"Validation accuracy: {best_accuracy:.6f}\n")
    output.write(f"Validation macro F1: {best_f1_macro:.6f}\n\n")
    output.write(report)

# Obě metriky vykreslíme pro každé vyzkoušené k.
k_values = [result[0] for result in list_best_k]
accuracies = [result[1] for result in list_best_k]
f1_values = [result[2] for result in list_best_k]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_values, accuracies, marker="o", label="Validation accuracy")
ax.plot(k_values, f1_values, marker="o", label="Validation macro F1")
ax.scatter(best_k, best_f1_macro, marker="*", s=220, color="red", zorder=5,
           label=f"Nejlepší k = {best_k}")
ax.set_xlabel("Počet sousedů k")
ax.set_xticks(k_values)
ax.set_ylabel("Hodnota metriky")
ax.set_ylim(0, 1.05)
ax.set_title("k-NN: výběr hyperparametru na validation")
ax.grid(alpha=0.3)
ax.legend()

# Graf pouze uložíme. Při dalším běhu se soubory tohoto modelu přepíšou.
fig.tight_layout()
fig.savefig(results_dir / "hyperparameters.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("Výsledky uloženy do:", results_dir)
