import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# ── 1. Load Data ─────────────────────────────────────────────────────────────
df = pd.read_csv("loan_data_new.csv")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# ── 2. Clean Column Names ─────────────────────────────────────────────────────
df.columns = df.columns.str.strip()

# ── 3. Handle Missing Values ──────────────────────────────────────────────────
df.dropna(inplace=True)
df = df[(df['Age'] >= 18) & (df['Age'] <= 70)]

# ── 4. Encode Categorical Columns ─────────────────────────────────────────────
categorical_cols = ["Gender", "Education", "Home Onwership", "Loan Intent", "Previous Loan"]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── 5. Features & Target ───────────────────────────────────────────────────────
feature_cols = [
    "Age", "Gender", "Education", "Person Income", "Employee Experience",
    "Home Onwership", "Loan Amount", "Loan Intent", "Loan interest Rate",
    "Loan percentage", "Credit History", "Credit Score", "Previous Loan"
]

X = df[feature_cols]
y = df["Loan Status"]

print("\nClass distribution:\n", y.value_counts())

# ── 6. Train / Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 7. Train Decision Tree ─────────────────────────────────────────────────────
model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
model.fit(X_train, y_train)

# ── 8. Evaluate ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ── 9. Feature Importance ──────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nFeature Importances:\n", importance_df.to_string(index=False))

# ── 10. Save Model & Encoders ──────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/decision_tree_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("model/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("\n✅ Model saved to model/decision_tree_model.pkl")
print("✅ Encoders saved to model/label_encoders.pkl")