import requests
import math
from datetime import datetime, timedelta
import pytz
from pvlib.solarposition import get_solarposition

# === CONFIG ===
LAT = 40.7948342      # Your location
LON = 19.4022414
TILT = 20        # Tilt angle of your panel in degrees
AZIMUTH = 0      # 0 = South-facing in Northern Hemisphere
TIMEZONE = "Europe/Tirane"

# Round ISO timestamp to nearest hour
def round_time_to_hour(dt_str):
    dt = datetime.fromisoformat(dt_str)
    if dt.minute >= 30:
        dt += timedelta(hours=1)
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.isoformat(timespec='minutes')

# Get weather data from Open-Meteo API
def get_weather_data(lat, lon, timezone):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "cloud_cover,pressure_msl,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "current_weather": True,
        "timezone": timezone
    }
    response = requests.get(url, params=params)
    data = response.json()

    current_time = round_time_to_hour(data['current_weather']['time'])

    try:
        idx = data['hourly']['time'].index(current_time)
    except ValueError:
        idx = -1  # fallback to latest available

    # cloud_cover = data['hourly']['cloud_cover'][idx]
    cloud_cover = 100
    pressure = data['hourly']['pressure_msl'][idx]
    humidity = data['hourly']['relative_humidity_2m'][idx]
    wind_speed = data['hourly']['wind_speed_10m'][idx]
    wind_direction = data['hourly']['wind_direction_10m'][idx]

    return data['current_weather'], cloud_cover, pressure, humidity, wind_speed, wind_direction

# Get solar angles from pvlib
def get_solar_angles(lat, lon, timezone):
    now = datetime.now(pytz.timezone(timezone))
    solpos = get_solarposition(now, lat, lon)
    zenith = float(solpos['zenith'])
    azimuth = float(solpos['azimuth'])
    return zenith, azimuth, now

# Clear-sky GHI using Haurwitz model
def haurwitz_model(zenith_angle_deg):
    zenith_rad = math.radians(zenith_angle_deg)
    cos_theta = math.cos(zenith_rad)
    if cos_theta <= 0:
        return 0
    ghi_clear = 1098 * cos_theta * math.exp(-0.059 / cos_theta)
    return ghi_clear

# Cloud cover correction
def apply_cloud_cover_correction(ghi_clear, cloud_cover_percent):
    cloud_fraction = cloud_cover_percent / 100
    ghi_cloud = ghi_clear * (1 - 0.75 * (cloud_fraction ** 3.4))
    return ghi_cloud

# Estimate Global Tilted Irradiance (GTI)
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

# Main logic
def main():
    weather, cloud_cover, pressure, humidity, wind_speed, wind_direction = get_weather_data(LAT, LON, TIMEZONE)
    zenith, solar_azimuth, time_now = get_solar_angles(LAT, LON, TIMEZONE)

    ghi_clear = haurwitz_model(zenith)
    ghi = apply_cloud_cover_correction(ghi_clear, cloud_cover)
    gti = estimate_gti(ghi, zenith, TILT, solar_azimuth, AZIMUTH)

    print(f"\n📍 Location: {LAT}, {LON}")
    print(f"🕒 Time: {time_now.strftime('%Y-%m-%d %H:%M:%S')} ({TIMEZONE})")
    print(f"☁️ Cloud Cover: {cloud_cover:.1f}%")
    print(f"🌡️ Pressure: {pressure:.1f} hPa")
    print(f"💧 Humidity: {humidity:.1f}%")
    print(f"🌬️ Wind Speed: {wind_speed:.1f} m/s")
    print(f"🧭 Wind Direction: {wind_direction:.1f}°")
    print(f"🌞 Solar Zenith: {zenith:.2f}°, Azimuth: {solar_azimuth:.2f}°")
    print(f"🔆 GHI (Clear Sky): {ghi_clear:.2f} W/m²")
    print(f"🔆 GHI (Adjusted): {ghi:.2f} W/m²")
    print(f"📐 GTI (Tilted): {gti:.2f} W/m²")

if __name__ == "__main__":
    main()
