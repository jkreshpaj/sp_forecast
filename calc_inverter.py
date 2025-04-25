import requests
import pandas as pd
import pvlib
import numpy as np
from pvlib.inverter import sandia

# ── CONFIG ──────────────────────────────────────────────────────────────────────
LAT, LON = 40.7948342, 19.4022414
APIKEY = '6poS7GKcuWLzClbE'
TZ = 'Europe/Tirane'
TILT = 20            # degrees
AZIMUTH = 180        # south
SYS_CAPACITY = 62.2  # MW
REF_TEMP = 25        # °C
LOSS_COEFF = -0.0026 # per °C
# ── INVERTER SANDIA PARAMETERS ──────────────────────────────────────────────────
INV_PARAMS = {
    'Paco':  300000.0,  # AC rated power [W]
    'Pdco':  300000.0,  # DC power for rated AC at Vdco [W]
    'Vdc0':  1080.0,    # DC voltage at Pac0 operating point [V]
    'Vdco':  1080.0,    # DC voltage at Pac0 operating point [V]
    'Pso':     10.0,    # DC power to start inversion [W]
    'C0':      0.005,   # curvature coefficient [1/W]
    'C1':     -0.00002, # voltage sensitivity [1/V]
    'C2':      0.0,     # Pso vs voltage [1/V]
    'C3':      0.0,     # C0 vs voltage [1/V]
    'Pnt':     0.0      # night tare [W]
}
# ── END CONFIG ──────────────────────────────────────────────────────────────────

# 1) FETCH METEOBLUE FORECASTS
url_basic  = f'https://my.meteoblue.com/packages/basic-1h_basic-day?lat={LAT}&lon={LON}&apikey={APIKEY}'
url_clouds = f'https://my.meteoblue.com/packages/clouds-1h_clouds-day?lat={LAT}&lon={LON}&apikey={APIKEY}'
jbasic, jclouds = requests.get(url_basic).json(), requests.get(url_clouds).json()

# 2) BUILD FORECAST DATAFRAME
df = pd.merge(
    pd.DataFrame(jbasic['data_1h']),
    pd.DataFrame(jclouds['data_1h']),
    on='time'
)
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M')
df = df.set_index('time').tz_localize('UTC').tz_convert(TZ)
df = df.rename(columns={
    'relativehumidity': 'humidity',
    'windspeed':        'wind_speed',
    'winddirection':    'wind_direction',
    'isdaylight':       'is_daylight',
    'sealevelpressure': 'pressure',
})
df['hour'], df['month'] = df.index.hour, df.index.month

# 3) CLEAR-SKY & CLOUD-ADJUSTED IRRADIANCE
location = pvlib.location.Location(LAT, LON, tz=TZ)
cs = location.get_clearsky(df.index, model='ineichen')
df['ghi_cs'], df['dni_cs'], df['dhi_cs'] = cs['ghi'], cs['dni'], cs['dhi']
cf = df['totalcloudcover'] / 100.0
df['ghi'] = df['ghi_cs'] * (1 - cf)
df['dni'] = df['dni_cs'] * (1 - cf)
df['dhi'] = df['dhi_cs'] * (1 - cf)

# 4) TILTED IRRADIANCE & MODULE TEMPERATURE
solpos = location.get_solarposition(df.index)
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=TILT, surface_azimuth=AZIMUTH,
    solar_zenith=solpos['zenith'], solar_azimuth=solpos['azimuth'],
    dni=df['dni'], ghi=df['ghi'], dhi=df['dhi']
)
df['gti'] = poa['poa_global']
sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
df['module_temperature'] = pvlib.temperature.sapm_cell(
    poa_global=df['gti'],
    temp_air=df['temperature'],
    wind_speed=df['wind_speed'],
    **sapm_params
)

# 5) DC POWER OUTPUT (pre-inverter)
df['power_dc_w'] = (
    SYS_CAPACITY * 1e6
    * (df['gti'] / 1000.0)
    * (1 + LOSS_COEFF * (df['module_temperature'] - REF_TEMP))
).clip(lower=0)

# 6) AC POWER VIA SANDIA INVERTER MODEL
# assume constant DC voltage at Vdc0
v_dc = np.full(len(df), INV_PARAMS['Vdc0'])
df['power_ac_w'] = sandia(
    v_dc, df['power_dc_w'], INV_PARAMS
).clip(lower=0)

# convert to MW & zero-out at night
df['power_ac_mw'] = df['power_ac_w'] / 1e6
df.loc[df['is_daylight'] == 0, 'power_ac_mw'] = 0

# 7) SAVE
df.to_csv('power_with_sandia_inverter.csv')
print("Wrote: power_with_sandia_inverter.csv")
