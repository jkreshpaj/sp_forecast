import streamlit as st
import requests
import math
import json
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from tinydb import TinyDB, Query
from pvlib.solarposition import get_solarposition
from pandas import json_normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

METEOBLU_API_KEY="mIgWDuMXxfFsa7p1"
LATITUDE=40.7948342
LONGITUDE=19.4022414
TILT=20  # Tilt angle of your panel in degrees
AZIMUTH=0
MODULE_TEMP=25  # °C
SYS_CAPACITY = 62.2  # MW
REF_TEMP = 25  # °C
LOSS_COEFF = -0.004  # per °C
SYSTEM_LOSSES = 0.12  # 12% system losses
TIMEZONE = pytz.timezone("Europe/Tirane")

db = TinyDB('db.json')

st.set_page_config(page_title="Solar Power Forecast", layout="wide")

def search_today_forecast():
    Forecast = Query()
    today_date = datetime.today().date().strftime('%Y-%m-%d')
    result = db.search(Forecast.timestamp.matches(f'^{today_date}'))
    
    if len(result):
        return result
    else:
        return None

def get_solar_angles(lat, lon, time):
    now = to_datetime(time)
    solpos = get_solarposition(now, lat, lon)
    zenith = float(solpos['zenith'])
    azimuth = float(solpos['azimuth'])
    return zenith, azimuth

def haurwitz_model(zenith_angle_deg):
    zenith_rad = math.radians(zenith_angle_deg)
    cos_theta = math.cos(zenith_rad)
    if cos_theta <= 0:
        return 0
    ghi_clear = 1098 * cos_theta * math.exp(-0.059 / cos_theta)
    return ghi_clear

def apply_cloud_cover_correction(ghi_clear, cloud_cover_percent):
    cloud_fraction = cloud_cover_percent / 100
    ghi_cloud = ghi_clear * (1 - 0.75 * (cloud_fraction ** 3.4))
    return ghi_cloud

def estimate_gti(ghi, zenith_deg, tilt_deg, solar_azimuth, panel_azimuth):
    zenith_rad = math.radians(zenith_deg)
    tilt_rad = math.radians(tilt_deg)

    cos_theta_t = math.cos(zenith_rad) * math.cos(tilt_rad) + \
                  math.sin(zenith_rad) * math.sin(tilt_rad) * \
                  math.cos(math.radians(solar_azimuth - panel_azimuth))
    cos_theta_t = max(0, cos_theta_t)
    cos_theta_z = math.cos(zenith_rad)

    if cos_theta_z == 0:
        return 0

    Rb = cos_theta_t / cos_theta_z
    gti = ghi * ((1 - Rb) * (1 + math.cos(tilt_rad)) / 2 + Rb)
    return gti

def get_weather_data(lat, lon):
    url = "https://my.meteoblue.com/packages/basic-1h_basic-day"
    cloud_url = "https://my.meteoblue.com/packages/clouds-1h_clouds-day"
    params = {
        "lat": lat,
        "lon": lon,
        "apikey": METEOBLU_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "data_1h" not in data:
        return []

    cloud_response = requests.get(cloud_url, params=params)
    cloud_data = cloud_response.json()

    time_list = data["data_1h"].get("time", [])
    windspeed_list = data["data_1h"].get("windspeed", [])
    winddirection_list = data["data_1h"].get("winddirection", [])
    temperature_list = data["data_1h"].get("temperature", [])
    humidity_list = data["data_1h"].get("relativehumidity", [])
    pressure_list = data["data_1h"].get("sealevelpressure", [])
    isdaylight_list = data["data_1h"].get("isdaylight", [])
    cloud_cover_list = cloud_data["data_1h"].get("totalcloudcover", [])
    
    formatted = [
            {
                "timestamp": time,
                "temperature": temperature,
                "humidity": humidity,
                "pressure": pressure,
                "wind_speed": windspeed,
                "wind_direction": winddirection,
                "cloud_cover": cloud_cover,
                "module_temp": MODULE_TEMP,
                "is_daylight": isdaylight
            }
            for time, temperature, humidity, pressure, windspeed, winddirection, cloud_cover, isdaylight in zip(
                time_list,
                temperature_list,
                humidity_list,
                pressure_list,
                windspeed_list,
                winddirection_list,
                cloud_cover_list,
                isdaylight_list
            )
    ]


    filtered = [
        entry for entry in formatted
        if datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M").date() == datetime.today().date()
    ];

    return filtered;

def to_datetime(timestamp):
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M")

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

def enhanced_physics_model(X, system_capacity=SYS_CAPACITY, temp_coeff=LOSS_COEFF, ref_temp=REF_TEMP):
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

model = train_model()

def make_predictions():
    previous_data = search_today_forecast()
    if previous_data:
        return previous_data

    data = get_weather_data(LATITUDE, LONGITUDE)
    for entry in data:
        zenith, solar_azimuth = get_solar_angles(LATITUDE, LONGITUDE, entry["timestamp"])
        ghi_clear = haurwitz_model(zenith)
        ghi = apply_cloud_cover_correction(ghi_clear, entry["cloud_cover"])
        gti = estimate_gti(ghi, zenith, TILT, solar_azimuth, AZIMUTH)

        entry["ghi"] = ghi
        entry["gti"] = gti
        entry["wind_cooling"] = entry["wind_speed"] * (entry["module_temp"] - entry["temperature"])

        input_data = pd.DataFrame([{
            "pressure": entry["pressure"],
            "temperature": entry["temperature"],
            "humidity": entry["humidity"],
            "ghi": entry["ghi"],
            "wind_direction": entry["wind_direction"],
            "wind_speed": entry["wind_speed"],
            "gti": entry["gti"],
            "module_temp": entry["module_temp"],
            "hour": to_datetime(entry["timestamp"]).hour,
            "month": to_datetime(entry["timestamp"]).month,
            "is_daylight": entry["is_daylight"],
            "wind_cooling": entry["wind_cooling"]
        }])

        rf_pred = model.predict(input_data)[0]
        physics_pred = enhanced_physics_model(input_data.iloc[0])
        hybrid_pred = (rf_pred * 0.3) + (physics_pred * 0.7)    

        entry["power_model"] = rf_pred
        entry["power_physics"] = physics_pred
        entry["power_hybrid"] = hybrid_pred

    db.insert_multiple(data)

    return data

forecasts = make_predictions()
current_hour = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:00')
current_forecast = next((item for item in forecasts if item["timestamp"] == current_hour), None)

col1, col2, col3 = st.columns(3)

with col1:
    if current_forecast:
        st.markdown(f"""
        |  **Information** | |
        | ------ | ------ |
        | **📍 Location** | {LATITUDE}, {LONGITUDE} |
        | **🕒 Time** | {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')} (Europe/Tirana) |
        | **💧 Humidity** | {current_forecast['humidity']:.1f}% |
        | **🧭 Wind Direction** | {current_forecast['wind_direction']:.1f}° |
        | **🔆 GHI (Adjusted)** | {current_forecast['ghi']:.2f} W/m² |
        | **📐 GTI (Tilted)** | {current_forecast['gti']:.2f} W/m² |
        """)

with col2:
    if current_forecast:
        st.metric(label="**Random Forest Prediction**", 
                 value=f"{current_forecast['power_model']:.2f} MW")

        st.metric(label="**Physics-Based Prediction**", 
                 value=f"{current_forecast['power_physics']:.2f} MW")

        st.metric(label="**Hybrid Prediction**", 
                 value=f"{current_forecast['power_hybrid']:.2f} MW",
                 delta=f"{(current_forecast['power_hybrid']/SYS_CAPACITY*100):.1f}% of capacity")

with col3:
    st.markdown(f"""
    |  **System Configuration** |  |
    | ------ | ------ |
    | **Max Capacity** | {SYS_CAPACITY} MW |
    | **Temp Coefficient** | {LOSS_COEFF*100:.1f}%/°C |
    | **System Losses** | {SYSTEM_LOSSES*100:.0f}% |
    | **Reference Temp** | {REF_TEMP}°C |
    | **Wind Cooling Factor** | 0.1 |
    | **Panel Tilt** | {TILT}° |
    """)

st.title("Forecast")

aspd = pd.DataFrame(forecasts).set_index("timestamp")
aspd["power_model"] = aspd["power_model"].shift(1)
aspd["power_physics"] = aspd["power_physics"].shift(1)
aspd["power_hybrid"] = aspd["power_hybrid"].shift(1)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌡️ Temperature")
    st.line_chart(aspd[["temperature"]], use_container_width=True)

with col2:
    st.subheader("☁️  Cloud Cover")
    st.line_chart(aspd[["cloud_cover"]], use_container_width=True)

with col3:
    st.subheader("🌬️ Wind Speed")
    st.line_chart(aspd[["cloud_cover"]], use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔆 GHI")
    st.line_chart(aspd[["ghi"]], use_container_width=True)

with col2:
    st.subheader("📐 GTI")
    st.line_chart(aspd[["gti"]], use_container_width=True)

st.subheader("Power forecast")
st.line_chart(aspd[["power_model", "power_physics", "power_hybrid"]], use_container_width=True)
