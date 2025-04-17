import streamlit as st
import json
import pandas as pd
import numpy as np
from pandas import json_normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Configuration
SYS_CAPACITY = 62.2  # MW
REF_TEMP = 25  # °C
LOSS_COEFF = -0.004  # per °C
SYSTEM_LOSSES = 0.12  # 12% system losses

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
    
    # Add wind cooling effect feature
    df["wind_cooling"] = df["wind_speed"] * (df["module_temp"] - df["temperature"])
    
    X = df.drop(["timestamp", "power"], axis=1)
    y = df["power"]
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_

def enhanced_physics_model(X, system_capacity=SYS_CAPACITY, 
                          temp_coeff=LOSS_COEFF, ref_temp=REF_TEMP):
    gti = X["gti"]
    module_temp = X["module_temp"]
    wind_cooling = X["wind_cooling"]
    
    # Effective temperature with wind cooling
    effective_temp = module_temp - (wind_cooling * 0.1)  # Empirical coefficient
    
    # Temperature derating
    temp_loss = temp_coeff * (effective_temp - ref_temp)
    
    # Power calculation with system losses
    power = (gti / 1000) * system_capacity * (1 + temp_loss) * (1 - SYSTEM_LOSSES)
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
    # Calculate wind cooling effect
    wind_cooling = wind_speed * (module_temp - temperature)
    
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
        "is_daylight": int(is_daylight),
        "wind_cooling": wind_cooling
    }])

    # Generate predictions
    rf_pred = model.predict(input_data)[0]
    physics_pred = enhanced_physics_model(input_data.iloc[0])
    hybrid_pred = (rf_pred * 0.7) + (physics_pred * 0.3)  # Weighted hybrid

    # Display results
    st.subheader("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="**Random Forest Prediction**", 
                 value=f"{rf_pred:.2f} MW",
                 delta=f"{(rf_pred/SYS_CAPACITY*100):.1f}% of capacity")
    
    with col2:
        st.metric(label="**Physics-Based Prediction**", 
                 value=f"{physics_pred:.2f} MW",
                 delta=f"{(physics_pred/SYS_CAPACITY*100):.1f}% of capacity")
    
    with col3:
        st.metric(label="**Hybrid Prediction**", 
                 value=f"{hybrid_pred:.2f} MW",
                 delta=f"{(hybrid_pred/SYS_CAPACITY*100):.1f}% of capacity")
    
    # Add visual spacer
    st.markdown("---")
    
    # Show input summary
    st.subheader("Input Summary")
    input_data_display = input_data.copy()
    input_data_display["wind_cooling_effect"] = wind_cooling * 0.1  # Show actual cooling effect
    st.dataframe(input_data_display.style.format("{:.2f}"), use_container_width=True)

else:
    st.info("Adjust input parameters in the sidebar and click 'Predict Power Output'")

# Add system info in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown(f"""
    **System Configuration**
    - Max Capacity: {SYS_CAPACITY} MW
    - Temp Coefficient: {LOSS_COEFF*100:.1f}%/°C
    - System Losses: {SYSTEM_LOSSES*100:.0f}%
    - Reference Temp: {REF_TEMP}°C
    - Wind Cooling Factor: 0.1
    """)
