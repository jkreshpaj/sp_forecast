import json
import pandas as pd
import numpy as np
import requests
from pandas import json_normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from pvlib import solarposition, clearsky, atmosphere
from datetime import datetime, timedelta

# Configuration
LATITUDE = 40.7948342
LONGITUDE = 19.4022414
SYS_CAPACITY = 62.2  # MW
REF_TEMP = 25  # °C
LOSS_COEFF = -0.004  # per °C
SYSTEM_LOSSES = 0.12  # 12% system losses
API_URL = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}"

def train_model():
    with open("solar_data.json", "r") as f:
        raw_data = json.load(f)
    
    df = json_normalize(raw_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["is_daylight"] = (df["ghi"] > 0).astype(int)
    df["wind_cooling"] = df["wind_speed"] * (df["module_temp"] - df["temperature"])
    
    X = df.drop(["timestamp", "power"], axis=1)
    y = df["power"]
    
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
    
    effective_temp = module_temp - (wind_cooling * 0.1)
    temp_loss = temp_coeff * (effective_temp - ref_temp)
    power = (gti / 1000) * system_capacity * (1 + temp_loss) * (1 - SYSTEM_LOSSES)
    return np.clip(power, 0, system_capacity)

def get_weather_forecast():
    try:
        params = {
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,"
                      "wind_speed_10m,wind_direction_10m,cloud_cover",
            "forecast_days": 1
        }
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(e)
        return None

def process_forecast_data(json_data):
    if not json_data:
        return pd.DataFrame()
    
    try:
        hourly = json_data["hourly"]
        df = pd.DataFrame({
            "timestamp": hourly["time"],
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "pressure": np.array(hourly["pressure_msl"]) / 1000,  # Convert Pa to kPa
            "wind_speed": hourly["wind_speed_10m"],
            "wind_direction": hourly["wind_direction_10m"],
            "cloud_cover": hourly["cloud_cover"]
        })
        
        # Calculate solar parameters
        times = pd.to_datetime(df['timestamp'], utc=True)
        
        # Get solar position
        solpos = solarposition.get_solarposition(
            times,
            LATITUDE,
            LONGITUDE
        )
        
        # Calculate airmass components
        airmass = atmosphere.get_relative_airmass(solpos['apparent_zenith'])
        pressure_pa = df['pressure'] * 1000  # Convert kPa back to Pa
        airmass_absolute = atmosphere.get_absolute_airmass(airmass, pressure=pressure_pa)
        
        # Use Ineichen model with pre-calculated parameters
        cs = clearsky.ineichen(
            apparent_zenith=solpos['apparent_zenith'],
            airmass_absolute=airmass_absolute,
            dni_extra=1364,
            linke_turbidity=3
        )
        
        # Apply cloud adjustment
        cloud_adj = (100 - df['cloud_cover']) / 100
        df["ghi"] = cs['ghi'] * cloud_adj
        df["gti"] = df["ghi"] * 1.1
        
        # Add derived features
        df["hour"] = times.dt.hour
        df["month"] = times.dt.month
        df["is_daylight"] = (df["ghi"] > 10).astype(int)
        df["module_temp"] = df["temperature"] + (df["ghi"] * 0.02)
        df["wind_cooling"] = df["wind_speed"] * (df["module_temp"] - df["temperature"])
        
        # Select final columns matching training data
        training_features = [
            "pressure", "temperature", "humidity", "wind_speed",
            "wind_direction", "gti", "module_temp", "hour",
            "month", "is_daylight", "wind_cooling"
        ]
        
        return df[training_features + ["timestamp"]]
    
    except Exception as e:
        print(e)
        return pd.DataFrame()

# model = train_model()
weather_data = get_weather_forecast()
print('got weather data', weather_data)
forecast_df = process_forecast_data(weather_data)
print(forecast_df.to_json(orient='records', date_format='iso'))
