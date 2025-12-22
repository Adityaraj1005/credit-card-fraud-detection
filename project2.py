# Import required libraries
import numpy as np
import pandas as pd

# Load dataset
dataset = pd.read_csv('creditcard.csv.zip')

# Separate features (X) and target (y)
X = dataset.iloc[:, :-1].values  # All columns except the last (features)
y = dataset.iloc[:, -1].values   # Last column (Fraud label: 0 or 1)

# Check for missing values
dataset.isnull().sum()  # Returns count of nulls per column

# Split data into training and testing sets
# stratify=y ensures class distribution is same in train and test
# random_state=0 ensures reproducibility 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Scale 'Amount' column (last column) for numerical stability
X_train[:, -1:] = sc.fit_transform(X_train[:, -1:])
X_test[:, -1:] = sc.transform(X_test[:, -1:])

# Scale 'Time' column (first column)
X_train[:, 0:1] = sc.fit_transform(X_train[:, 0:1])
X_test[:, 0:1] = sc.transform(X_test[:, 0:1])

# ------------------------------
# Logistic Regression (baseline)
# ------------------------------
from sklearn.linear_model import LogisticRegression

# Initialize classifier (can add class_weight='balanced' if desired)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)  # Train model

# Evaluate Logistic Regression
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)  # Predict test set

cm = confusion_matrix(y_test, y_pred)  # Confusion matrix
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import precision_score, recall_score, f1_score
print("Precision:", precision_score(y_test, y_pred))  # TP / (TP + FP)
print("Recall:", recall_score(y_test, y_pred))        # TP / (TP + FN)
print("F1-score:", f1_score(y_test, y_pred))          # Harmonic mean of precision & recall

# ------------------------------
# Random Forest Classifier
# ------------------------------
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest
# n_estimators=100 → 100 trees
# criterion='entropy' → splits based on information gain
# random_state=0 → reproducible results
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)  # Train model

# Evaluate Random Forest
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# ------------------------------
# XGBoost Classifier
# ------------------------------
from xgboost import XGBClassifier

# Initialize XGBoost
# random_state=0 → reproducible results
classifier = XGBClassifier(random_state=0)
classifier.fit(X_train, y_train)  # Train model

# Evaluate XGBoost
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
