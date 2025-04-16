import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Constants
A = 3.1
eta_stc = 0.18
gamma = 0.004
delta = 0.1
T_stc = 25.0

def physics_power(row):
    temp_correction = 1 - gamma * (row['temperature'] - T_stc - delta * row['wind_speed'])
    return A * eta_stc * temp_correction * row['ghi']

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
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute

    physics_est = physics_power(input_data)
    residual_est = model.predict(df[['pressure', 'temperature', 'humidity', 'wind_direction', 'wind_speed', 'gti', 'hour', 'minute']])[0]

    return physics_est + residual_est, physics_est, residual_est

