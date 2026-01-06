import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler   # ← DÒNG CÒN THIẾU

from src.dataset import build_dataset


# ===== cấu hình =====
FS = 125
PATH_REST = "data/raw_ppg/rest.csv"
PATH_ACTIVE = "data/raw_ppg/active.csv"

# ======================
# BUILD DATASET
# ======================
X_rest, y_rest = build_dataset(PATH_REST, FS, label=0)
X_active, y_active = build_dataset(PATH_ACTIVE, FS, label=1)

X = np.vstack([X_rest, X_active])
y = np.hstack([y_rest, y_active])

print("Dataset shape:", X.shape)
print("Labels:", np.unique(y))

# ======================
# TRAIN / TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# đánh giá model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Weights:", model.coef_)
print("Bias:", model.intercept_)



