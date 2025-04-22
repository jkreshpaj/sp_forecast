import pvlib
import pandas as pd
import numpy as np

def calculate_ghi_gti(
    latitude,
    longitude,
    time,
    cloud_cover,
    tilt,
    surface_azimuth,
    albedo=0.2,
    altitude=None,
    temperature=None
):
    """
    Calculate Global Horizontal Irradiance (GHI) and Global Tilted Irradiance (GTI) for a given location and time.

    Parameters:
    - latitude (float): Latitude in degrees (positive north, negative south).
    - longitude (float): Longitude in degrees (positive east, negative west).
    - time (str or pd.Timestamp): Time of calculation (e.g., '2025-05-18 16:00:00').
    - cloud_cover (float): Cloud cover percentage (0 to 100).
    - tilt (float): Surface tilt angle in degrees (0 = horizontal, 90 = vertical).
    - surface_azimuth (float): Surface azimuth in degrees (0 = north, 180 = south).
    - albedo (float, optional): Ground albedo (default 0.2).
    - altitude (float, optional): Elevation in meters (default None).
    - temperature (float, optional): Temperature in Celsius (default None).

    Returns:
    - ghi (float): Global Horizontal Irradiance in W/m².
    - gti (float): Global Tilted Irradiance in W/m².
    """
    # Convert time to pandas Timestamp and wrap in a pandas Index
    time = pd.to_datetime(time)
    times = pd.DatetimeIndex([time])

    # Create location object
    location = pvlib.location.Location(latitude, longitude, altitude=altitude)

    # Get solar position (zenith and azimuth angles)
    solpos = location.get_solarposition(times)
    zenith = solpos['apparent_zenith'].values[0]
    azimuth = solpos['azimuth'].values[0]

    # Handle nighttime case (sun below horizon)
    if zenith >= 90:
        return 0.0, 0.0

    # Get Linke turbidity (monthly value) using times instead of time
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude).iloc[0]

    # Calculate clear-sky irradiance
    clearsky = pvlib.clearsky.ineichen(times, location, linke_turbidity=linke_turbidity)
    ghi_clear = clearsky['ghi'].values[0]
    dni_clear = clearsky['dni'].values[0]
    dhi_clear = clearsky['dhi'].values[0]

    # Adjust GHI for cloud cover using a cloud modification factor (CMF)
    cloud_fraction = cloud_cover / 100.0
    cmf = 1 - 0.75 * cloud_fraction**1.5  # Empirical formula
    ghi_cloudy = max(ghi_clear * cmf, 0)  # Ensure non-negative

    # Calculate extraterrestrial radiation and clearness index (Kt)
    extra = pvlib.irradiance.get_extra_radiation(time)
    ghi_extraterrestrial = extra * np.cos(np.radians(zenith))
    kt = ghi_cloudy / ghi_extraterrestrial if ghi_extraterrestrial > 0 else 0

    # Estimate diffuse fraction using the Erbs model
    if kt <= 0.22:
        diffuse_fraction = 1 - 0.09 * kt
    elif 0.22 < kt <= 0.8:
        diffuse_fraction = (
            0.9511 - 0.1604 * kt + 4.388 * kt**2 -
            16.638 * kt**3 + 12.336 * kt**4
        )
    else:
        diffuse_fraction = 0.165

    # Calculate cloudy DHI and DNI
    dhi_cloudy = diffuse_fraction * ghi_cloudy
    dni_cloudy = (ghi_cloudy - dhi_cloudy) / np.cos(np.radians(zenith)) if np.cos(np.radians(zenith)) > 0 else 0

    # Ensure non-negative values
    dhi_cloudy = max(dhi_cloudy, 0)
    dni_cloudy = max(dni_cloudy, 0)

    # Calculate GTI (total irradiance on tilted surface)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=zenith,
        solar_azimuth=azimuth,
        dni=dni_cloudy,
        ghi=ghi_cloudy,
        dhi=dhi_cloudy,
        albedo=albedo,
        model='isotropic'  # Option: 'perez' for higher accuracy
    )
    gti = max(poa['poa_global'].values[0], 0)  # Ensure non-negative

    return ghi_cloudy, gti

# Example usage
if __name__ == "__main__":
    # Sample inputs: Boulder, CO, May 18, 2025, 4:00 PM, 50% clouds
    lat = 40.0
    lon = -105.0
    time = '2025-05-18 16:00:00'
    cloud_cover = 50  # 50% cloud cover
    tilt = 30  # 30-degree tilt
    surface_azimuth = 180  # Facing south
    albedo = 0.2  # Typical ground albedo
    altitude = 1655  # Boulder altitude in meters
    temperature = 25  # Temperature in Celsius (not used in this model)

    ghi, gti = calculate_ghi_gti(lat, lon, time, cloud_cover, tilt, surface_azimuth, albedo, altitude, temperature)
    print(f"GHI: {ghi:.2f} W/m²")
    print(f"GTI: {gti:.2f} W/m²")
