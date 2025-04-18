from datetime import datetime
from pvlib.solarposition import get_solarposition

LATITUDE=40.7948342
LONGITUDE=19.4022414


now = datetime.now()
print(now)

solpos = get_solarposition(now, LATITUDE, LONGITUDE)
zenith = float(solpos['zenith'])
azimuth = float(solpos['azimuth'])

print(zenith, azimuth)
