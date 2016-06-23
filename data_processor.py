import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset, num2date
from datetime import datetime

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
    out_temp = f.variables['outside_temperature'][scan_number]
    mtp_data = f.variables['mtp_data'][:,:,scan_number]

    return time, lon, lat, hgt, out_temp, mtp_data

def extract_merra_profiles(filename, time, lon, lat):
    '''
    Subset the merra data for time, lon, and lat, then 
    return interpolated climatological profile and covariance
    ''' 
    f = Dataset(filename)
    merra_lats= f.variables['lat'][:]
    merra_lons= f.variables['lon'][:]
    merra_hgts= f.variables['height'][:]
    merra_time0 = f.variables['time']
    merra_time = num2date(merra_time0[:], units=merra_time0.units)[-1,:,:]
    year_end = merra_time[0,0].year 
    new_date = datetime(year_end, time.month, time.day, time.hour, time.minute, time.second)
    time_diff = abs(merra_time - new_date) 
    day_index, hour_index = np.where(time_diff == np.min(time_diff))

    y_index, x_index = np.where((merra_lats - lat)**2+(merra_lons - lon)**2 == np.min((merra_lats - lat)**2+(merra_lons - lon)**2))
    if len(y_index) >=2:
        y_index = y_index[0]
        x_index = x_index[0]
    
    T_data0 = f.variables['T'][:, day_index, hour_index, :, y_index, x_index]
    Q_data0 = f.variables['q'][:, day_index, hour_index, :, y_index, x_index]
    RH_data0 = f.variables['RH'][:, day_index, hour_index, :, y_index, x_index]
 
    return ma.squeeze(T_data0), ma.squeeze(Q_data0), ma.squeeze(RH_data0),merra_hgts



