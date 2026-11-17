from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
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
print(classification_report(y_val, best_y_pred, digits=3, zero_division=0))
