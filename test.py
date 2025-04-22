import pandas as pd
from datetime import datetime
from pvlib.solarposition import get_solarposition
from tinydb import TinyDB, Query

LATITUDE=40.7948342
LONGITUDE=19.4022414

db = TinyDB('db.json')
Forecast = Query()
today_date = datetime.today().date().strftime('%Y-%m-%d 16:00')
result = db.search(Forecast.timestamp.matches(f'^{today_date}'))

print(result)


# now = datetime.now()
# print(now)

# solpos = get_solarposition(now, LATITUDE, LONGITUDE)
# zenith = float(solpos['zenith'])
# azimuth = float(solpos['azimuth'])

# print(zenith, azimuth)
