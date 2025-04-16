import streamlit as st
import json
import pandas as pd
import numpy as np
from pandas import json_normalize
from sklearn.ensemble import RandomForestRegressor

# Configuration
SYS_CAPACITY = 62.2  # MW
REF_TEMP = 25  # °C
LOSS_COEFF = -0.0026  # per °C

# Cache model training to only run once
@st.cache_resource
def train_model():
    with open("solar_data.json", "r") as f:
        raw_data = json.load(f)
    
    df = json_normalize(raw_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["is_daylight"] = (df["ghi"] > 0).astype(int)
    
    X = df.drop(["timestamp", "power"], axis=1)
    y = df["power"]
    
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def calculate_math_predictions(X, system_capacity=SYS_CAPACITY, 
                              temp_coeff=LOSS_COEFF, ref_temp=REF_TEMP):
    reference_GTI = 1000
    gti = X["gti"]
    module_temp = X["module_temp"]
    
    temp_loss = temp_coeff * (module_temp - ref_temp)
    power = (gti / reference_GTI) * system_capacity * (1 + temp_loss)
    return np.clip(power, 0, system_capacity)

# Streamlit UI
st.set_page_config(page_title="Solar Power Predictor", layout="wide")
st.title("☀️ Solar Power Prediction Dashboard")

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pressure = st.number_input("Pressure (hPa)", value=1020.0)
        temperature = st.number_input("Temperature (°C)", value=15.0)
        humidity = st.number_input("Humidity (%)", value=50.0)
        ghi = st.number_input("GHI (W/m²)", value=100.0)
        
    with col2:
        wind_dir = st.number_input("Wind Direction (°)", value=180.0)
        wind_speed = st.number_input("Wind Speed (m/s)", value=2.0)
        gti = st.number_input("GTI (W/m²)", value=100.0)
        module_temp = st.number_input("Module Temp (°C)", value=20.0)
    
    hour = st.slider("Hour of Day", 0, 23, 12)
    month = st.slider("Month", 1, 12, 6)
    is_daylight = st.checkbox("Daylight", value=True)

# Load model
model = train_model()

# Prediction button
if st.button("Predict Power Output"):
    # Create input dataframe
    input_data = pd.DataFrame([{
        "pressure": pressure,
        "temperature": temperature,
        "humidity": humidity,
        "ghi": ghi,
        "wind_direction": wind_dir,
        "wind_speed": wind_speed,
        "gti": gti,
        "module_temp": module_temp,
        "hour": hour,
        "month": month,
        "is_daylight": int(is_daylight)
    }])

    # Generate predictions
    rf_pred = model.predict(input_data)[0]
    math_pred = calculate_math_predictions(input_data).values[0]
    hybrid_pred = (rf_pred + math_pred) / 2

    # Display results
    st.subheader("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="**Random Forest Prediction**", 
                 value=f"{rf_pred:.2f} MW",
                 delta=f"{(rf_pred/SYS_CAPACITY*100):.1f}% of capacity")
    
    with col2:
        st.metric(label="**Physics-Based Prediction**", 
                 value=f"{math_pred:.2f} MW",
                 delta=f"{(math_pred/SYS_CAPACITY*100):.1f}% of capacity")
    
    with col3:
        st.metric(label="**Hybrid Prediction**", 
                 value=f"{hybrid_pred:.2f} MW",
                 delta=f"{(hybrid_pred/SYS_CAPACITY*100):.1f}% of capacity")
    
    # Add visual spacer
    st.markdown("---")
    
    # Show input summary
    st.subheader("Input Summary")
    st.dataframe(input_data.style.format("{:.2f}"), use_container_width=True)

else:
    st.info("Adjust input parameters in the sidebar and click 'Predict Power Output'")

# Add system info in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown(f"""
    **System Configuration**
    - Max Capacity: {SYS_CAPACITY} MW
    - Temp Coefficient: {LOSS_COEFF*100:.1f}%/°C
    - Reference Temp: {REF_TEMP}°C
    """)
