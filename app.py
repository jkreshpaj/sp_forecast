import streamlit as st
from datetime import datetime
import pandas as pd
from model_utils import load_and_train_model, predict_power

st.set_page_config(page_title="Solar Power Forecast", page_icon="🔆")

# === Cache the model so it doesn't reload on every interaction ===
@st.cache_resource
def get_model():
    return load_and_train_model()

model = get_model()

st.title("🔆 Power Forecast")
st.write("Enter environmental and solar data to predict solar power output.")

# === Input fields ===
with st.form("input_form"):
    pressure = st.number_input("Atmospheric Pressure (hPa)", 950.0, 1050.0, 1016.3)
    temperature = st.number_input("Ambient Temperature (°C)", -20.0, 60.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
    ghi = st.number_input("Global Horizontal Irradiance (W/m²)", 0.0, 1200.0, 500.0)
    wind_direction = st.number_input("Wind Direction (°)", 0.0, 360.0, 100.0)
    wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 1.2)
    gti = st.number_input("Global Tilted Irradiance (W/m²)", 0.0, 1500.0, 520.0)
    module_temp = st.number_input("Module Temperature (°C)", -20.0, 100.0, 30.0)
    timestamp = st.time_input("Time of Measurement")

    submit = st.form_submit_button("⚡ Predict Power Output")

# === Predict only when the form is submitted ===
if submit:
    input_data = {
        "pressure": pressure,
        "temperature": temperature,
        "humidity": humidity,
        "ghi": ghi,
        "wind_direction": wind_direction,
        "wind_speed": wind_speed,
        "gti": gti,
        "module_temp": module_temp,
        "timestamp": pd.Timestamp.combine(pd.Timestamp.today(), timestamp)
    }

    hybrid_power, physics_estimate, residual_est = predict_power(model, input_data)

    st.success(f"🔋 Forecasted Power Output: **{hybrid_power:.2f} MW**")
    st.info(f"🧮 Physics Estimate: {physics_estimate:.2f} MW")
    st.info(f"🎯 ML Residual Correction: {residual_est:.2f} MW")
