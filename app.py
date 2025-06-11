import streamlit as st
import pickle
import os
from pathlib import Path

# Define the path relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"

# Title
st.title("🌸 Iris Flower Predictor (using Pickled Model)")
st.write("Enter the measurements of the iris flower:")

# Check if model file exists before loading
if not MODEL_PATH.exists():
    st.error(f"❌ Model file not found at: `{MODEL_PATH}`.\n\nMake sure the file exists.")
    st.stop()

# Load the pickled model
with open(MODEL_PATH, "rb") as f:
    model, target_names = pickle.load(f)

# Input sliders
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    predicted_species = target_names[prediction]
    st.success(f"🌼 The predicted species is: **{predicted_species}**")
