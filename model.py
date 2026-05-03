import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and target
X = data.drop("disease", axis=1)
y = data["disease"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# =========================
# 🔥 USER INPUT SECTION
# =========================

print("\n--- Enter Symptoms (0 = No, 1 = Yes) ---")

fever = int(input("Fever (0/1): "))
cough = int(input("Cough (0/1): "))
fatigue = int(input("Fatigue (0/1): "))
headache = int(input("Headache (0/1): "))

# Convert to array
user_data = np.array([[fever, cough, fatigue, headache]])

# Predict
prediction = model.predict(user_data)

print("\n✅ Predicted Disease:", prediction[0])