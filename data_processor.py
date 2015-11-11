import numpy as np
from netCDF4 import Dataset, num2date

def MTP_obs_extract(scan_number):
    '''
    Read the MTP observation and return geographical 
    variables and TBs for a single scan
    '''
    f = Dataset('./data/NCAR_MTP_data.nc')

    time = num2date(f.variables['time'][scan_number], units = f.variables['time'].units)
    lon = f.variables['lon'][scan_number]
    lat = f.variables['lat'][scan_number]
    hgt = f.variables['hgt'][scan_number]
    mtp_data = f.variables['mtp_data'][:,:,scan_number]

    return time, lon, lat, hgt, mtp_data

def merra_climatology_and_covariance(time, lon, lat):
    '''
    Subset the merra data for time, lon, and lat, then 
    return interpolated climatological profile and covariance
    ''' 
    f = Dataset('./data/merra_temperature_and_moisture_June01-June06_1979-2014.nc')
    lats= f.variables['lat'][:]
    lons= f.variables['lon'][:]
    hgts= f.variables['height'][:]

    day_index = time.day-1

    hours = np.arange(9)*3
    hour_index = np.min(np.where(abs(hours - time.hour).min() == abs(hours - time.hour))[0])
    if hour_index == 8:
        day_index = day_index+1 
        hour_index = 0
 
    y_index, x_index = np.where((lats - lat)**2+(lons - lon)**2 == np.min((lats - lat)**2+(lons - lon)**2))
    if len(y_index) >=2:
        y_index = y_index[0]
        x_index = x_index[0]
    

    T_data0 = f.variables['T'][:, day_index, hour_index, :, y_index, x_index]
    Q_data0 = f.variables['q'][:, day_index, hour_index, :, y_index, x_index]
 
    return np.squeeze(T_data0), np.squeeze(Q_data0), hgts



