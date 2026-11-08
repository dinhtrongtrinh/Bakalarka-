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


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)


f1_macro = f1_score(y_val, y_pred, average='macro')
class_report = classification_report(y_val, y_pred)
print(f"Validation Accuracy (k=5): {accuracy_score(y_val, y_pred):.2f}")
print(f1_macro)
print(class_report)
