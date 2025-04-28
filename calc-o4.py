import requests
import pandas as pd
import pvlib

# ── CONFIG ──────────────────────────────────────────────────────────────────────
LAT, LON = 40.7948342, 19.4022414
APIKEY = '6poS7GKcuWLzClbE'
TZ = 'Europe/Tirane'
TILT = 20            # degrees
AZIMUTH = 180        # south
SYS_CAPACITY = 62.2  # MW
REF_TEMP = 25        # °C
LOSS_COEFF = -0.0026 # per °C
SYSTEM_LOSSES = 0.08 # 8% system losses
# ── END CONFIG ──────────────────────────────────────────────────────────────────

# fetch both endpoints
url_basic  = f'https://my.meteoblue.com/packages/basic-1h_basic-day?lat={LAT}&lon={LON}&apikey={APIKEY}'
url_clouds = f'https://my.meteoblue.com/packages/clouds-1h_clouds-day?lat={LAT}&lon={LON}&apikey={APIKEY}'
jbasic, jclouds = requests.get(url_basic).json(), requests.get(url_clouds).json()

# build and merge DataFrames
df = pd.merge(
    pd.DataFrame(jbasic['data_1h']),
    pd.DataFrame(jclouds['data_1h']),
    on='time'
)

# timestamp → index with timezone
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M')
df = df.set_index('time').tz_localize('UTC').tz_convert(TZ)

# rename and add features
df = df.rename(columns={
    'relativehumidity': 'humidity',
    'windspeed':        'wind_speed',
    'winddirection':    'wind_direction',
    'isdaylight':       'is_daylight',
    'sealevelpressure': 'pressure',
})
df['hour'], df['month'] = df.index.hour, df.index.month

# clear‐sky irradiance
location = pvlib.location.Location(LAT, LON, tz=TZ)
cs = location.get_clearsky(df.index, model='ineichen')
df['ghi_clearsky'], df['dni_clearsky'], df['dhi_clearsky'] = cs['ghi'], cs['dni'], cs['dhi']

# adjust by cloud cover
cf = df['totalcloudcover']/100
df['ghi'], df['dni'], df['dhi'] = (
    df['ghi_clearsky']*(1-cf),
    df['dni_clearsky']*(1-cf),
    df['dhi_clearsky']*(1-cf),
)

# solar position & tilted irradiance
solpos = location.get_solarposition(df.index)
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=TILT, surface_azimuth=AZIMUTH,
    solar_zenith=solpos['zenith'], solar_azimuth=solpos['azimuth'],
    dni=df['dni'], ghi=df['ghi'], dhi=df['dhi']
)
df['gti'] = poa['poa_global']

# ── NEW: MODULE CELL TEMPERATURE ─────────────────────────────────────────────────
# use SAPM open-rack glass-back parameters
sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

df['module_temperature'] = pvlib.temperature.sapm_cell(
    poa_global = df['gti'],
    temp_air   = df['temperature'],
    wind_speed = df['wind_speed'],
    **sapm_params
)
# ── END NEW ────────────────────────────────────────────────────────────────────

# physics‐based power using cell temperature
df['power_physics'] = (
    SYS_CAPACITY
    * (df['gti'] / 1000)
    * (1 + LOSS_COEFF * (df['module_temperature'] - REF_TEMP))
    * (1 - SYSTEM_LOSSES)
).clip(lower=0)

# zero‐out at night
df.loc[df['is_daylight']==0, 'power_physics'] = 0

df.to_csv('output.csv')
