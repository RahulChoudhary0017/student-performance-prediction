import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample dataset (tu CSV bhi use kar sakta hai)
data = {
    "attendance": [85, 95, 60, 40, 70],
    "study_hours": [5, 8, 2, 1, 4],
    "assignment": [78, 92, 50, 30, 65],
    "midterm": [80, 90, 48, 35, 70],
    "previous_marks": [75, 88, 55, 40, 68],
    "performance": ["Good", "Excellent", "Average", "Poor", "Good"]
}

df = pd.DataFrame(data)

X = df.drop("performance", axis=1)
y = df["performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
import os
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/student_model.pkl")

print("Model saved successfully!")