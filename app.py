import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


import joblib

# Load model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

st.title("🌤️ Next-Day Temperature Predictor (Tehran)")
st.write("Enter the temperatures of the last 30 days (°C) to predict tomorrow's temperature.")

# User input for 30 days
temp_input = []
for i in range(30):
    t = st.number_input(f"Day {i+1} temperature", min_value=-10.0, max_value=50.0, step=0.1, key=i)
    temp_input.append(t)

if st.button("Predict"):
    input_scaled = scaler.transform(np.array(temp_input).reshape(-1, 1)).flatten()
    prediction_input = np.array(input_scaled).reshape(1, 30, 1)

    y_pred_scaled = model.predict(prediction_input)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    st.success(f"🌡️ Predicted temperature for tomorrow: **{y_pred[0][0]:.2f} °C**")
