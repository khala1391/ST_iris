import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import os

# Define the path relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and target names to model.pkl
with open(MODEL_PATH, "wb") as f:
    pickle.dump((model, iris.target_names), f)

print(f"✅ Model saved to: {MODEL_PATH}")
