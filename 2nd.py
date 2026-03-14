import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("application_train.csv")

# Drop ID column if exists
if "SK_ID_CURR" in df.columns:
    df.drop("SK_ID_CURR", axis=1, inplace=True)

# =========================
# 2. Handle Missing Values
# =========================

# Separate features and target first
X = df.drop("TARGET", axis=1)
y = df["TARGET"]

# Numeric columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Categorical columns
cat_cols = X.select_dtypes(include=['object', 'string']).columns

# Impute numeric columns (median)
num_imputer = SimpleImputer(strategy='median')
X[num_cols] = num_imputer.fit_transform(X[num_cols])

# Fill categorical missing values
X[cat_cols] = X[cat_cols].fillna("Unknown")

# =========================
# 3. Encode Categorical Data
# =========================
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# =========================
# 4. Feature Scaling (Important for LR)
# =========================
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# =========================
# 5. Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. Model Training
# =========================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression(max_iter=3000)

rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# =========================
# 7. Predictions
# =========================
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# =========================
# 8. Evaluation
# =========================
rf_acc = accuracy_score(y_test, rf_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print("\n===== MODEL EVALUATION =====")
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# =========================
# 9. Save Everything (Deployment Ready)
# =========================
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(lr_model, open("lr_model.pkl", "wb"))
pickle.dump(label_encoders, open("encoders.pkl", "wb"))
pickle.dump(num_imputer, open("imputer.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("feature_columns.pkl", "wb"))

print("\nTraining Successful ✅")
