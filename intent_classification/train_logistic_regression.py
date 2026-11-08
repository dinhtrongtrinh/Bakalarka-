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

model = LogisticRegression(max_iter=3000,C=1,solver='lbfgs')
model.fit(X_train,y_train)

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
class_report = classification_report(y_val, y_pred)
f1_macro = f1_score(y_val, y_pred, average='macro')
print(f"accuracy: {round(accuracy,2)}")
print("Macro-average F1 score:", f1_macro)
print(class_report)


