from pathlib import Path
from sklearn.linear_model import LogisticRegression
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


# Dvacet hodnot C v geometrické řadě od 0.001 do 1000.
list_c = np.geomspace(0.001, 1000, 20)
list_best_c = []
best_f1 = -1.0
best_y_pred = None

for c in list_c:
    # Pro každé C trénujeme nový model pouze na trénovacích datech.
    model = LogisticRegression(max_iter=3000, C=c, solver="lbfgs")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)

    # Ukládáme C společně s oběma metrikami, aby bylo jasné, k čemu patří.
    list_best_c.append((c, accuracy, f1_macro))
    print(f"C={c:.6g}: accuracy={accuracy:.4f}, macro F1={f1_macro:.4f}")

    # Uchováme předpovědi nejlepšího modelu pro závěrečný report.
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_y_pred = y_pred

# Třetí položka tuple (index 2) je macro F1, podle kterého vybíráme.
# Při shodě vyhraje první vyzkoušené C, tedy menší hodnota.
best_c, best_accuracy, best_f1_macro = max(list_best_c, key=lambda result: result[2])
print(f"\nNejlepší C podle validačního macro F1: {best_c:.6g}")
print(f"Validation accuracy: {best_accuracy:.4f}")
print(f"Validation macro F1: {best_f1_macro:.4f}")
print(classification_report(y_val, best_y_pred, digits=3, zero_division=0))


