import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Constants
AREA = 3.1                    # Panel area in square meters
ETA_SYSTEM = 0.13             # Realistic system efficiency
GAMMA = 0.004                 # Temperature loss coefficient
DELTA = 0.1                   # Wind cooling coefficient
T_STC = 25.0                  # Standard temperature (°C)

def physics_power(row):
    """Estimate physical power output in MW based on irradiance and temperature"""
    temp_correction = 1 - GAMMA * (row['temperature'] - T_STC - DELTA * row['wind_speed'])
    corrected_efficiency = ETA_SYSTEM * temp_correction
    power_mw = row['gti'] * AREA * corrected_efficiency / 1_000_000  # Convert W to MW
    return max(power_mw, 0)

def load_and_train_model(json_path='solar_data.json'):
    with open(json_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute

    df['physics_power'] = df.apply(physics_power, axis=1)
    df['residual'] = df['power'] - df['physics_power']

    features = ['pressure', 'temperature', 'humidity', 'wind_direction', 'wind_speed', 'gti', 'hour', 'minute']
    X = df[features]
    y = df['residual']

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model

def predict_power(model, input_data):
    df = pd.DataFrame([input_data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute

    physics_est = physics_power(input_data)
    residual_est = model.predict(df[['pressure', 'temperature', 'humidity', 'wind_direction', 'wind_speed', 'gti', 'hour', 'minute']])[0]

    return physics_est + residual_est, physics_est, residual_est

