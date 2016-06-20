import utils
import data_processor as dp
from microwave_absorption_coeff import gas_absr
from Jacobian import calculate_K, calculate_F
from plots import plot_temperature_profile_from_pickle_file, plot_map_track

import numpy as np
import numpy.ma as ma
import math

import sys
from netCDF4 import Dataset, num2date

import matplotlib.pyplot as plt

f = Dataset('./data/NCAR_MTP_data.nc')
lon = f.variables['lon'][:]
lat = f.variables['lat'][:]
hgt = f.variables['hgt'][:]
out_temp = f.variables['outside_temperature'][:]
mtp_data = f.variables['mtp_data'][:]
time = f.variables['time'][:]

time = num2date(time, units = f.variables['time'].units)
ncase = 3660
merra_temp = ma.zeros([2,ncase]) -9999.
merra2_temp = ma.zeros([2,ncase]) -9999.
for icase in np.arange(ncase):
    print icase
    if hgt[icase] > 7.:
        temp_merra, q_merra, heights_merra = dp.merra_climatology_and_covariance('./data/merra_temperature_and_moisture_June01-June06_1979-2014.nc',
                                                                  time[icase], lon[icase], lat[icase])
        temp_obs, heights = utils.interpolate_temperature(heights_merra, temp_merra, hgt[icase])
        merra_temp[0,icase] = ma.mean(temp_obs[:,9]+temp_obs[:,10])*0.5
        merra_temp[1,icase] = (temp_obs[-1,9]+temp_obs[-1,10])*0.5

        temp_merra, q_merra, heights_merra = dp.merra_climatology_and_covariance('./data/merra2_temperature_and_moisture_June01-June06_1979-2014.nc',
                                                                  time[icase], lon[icase], lat[icase])
        temp_obs, heights = utils.interpolate_temperature(heights_merra, temp_merra, hgt[icase])
        merra2_temp[0,icase] = ma.mean(temp_obs[:,9]+temp_obs[:,10])*0.5
        merra2_temp[1,icase] = (temp_obs[-1,9]+temp_obs[-1,10])*0.5

merra_temp = ma.masked_equal(merra_temp, -9999.)
merra2_temp = ma.masked_equal(merra2_temp, -9999.)

fig = plt.figure()
ax = fig.add_subplot(311)
x = np.arange(ncase)+1
ax.plot(x,merra_temp[0,:], label='MERRA Clim.')
ax.plot(x,merra_temp[1,:], label='MERRA 2014')
ax.plot(x,merra2_temp[0,:], label='MERRA2 Clim.')
ax.plot(x,merra2_temp[1,:], label='MERRA2 2014')
for iangle in np.arange(3):
    ax.plot(x, mtp_data[iangle,5,:], label = 'MTP freq %d' %(iangle+1))
ax.plot(x, out_temp[:], label = 'Outside temperature' , lw=2)
#ax.set_xlabel('Scan number')
ax.set_ylabel('Temperature [K]')
ax.set_xlim([0, 3660])
ax.set_title('Comparison of the temperature between MTP and MERRA')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=6, frameon=False, fontsize='small')

ax = fig.add_subplot(312)
x = np.arange(ncase)+1
ax.set_xlim([0, 3660])
ax.plot(x,merra_temp[1,:]-out_temp, label='MERRA 2014 - Outside')
ax.plot(x,merra2_temp[1,:]-out_temp, label='MERRA2 2014 - Outside')
ax.plot(x,merra_temp[0,:]-out_temp, label='MERRA clim - Outside')
ax.plot(x,merra2_temp[0,:]-out_temp, label='MERRA2 clim - Outside')
ax.plot([0,3660],[0,0], 'k')
ax.set_title('Temperature difference')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4, frameon=False, fontsize='small')

ax = fig.add_subplot(313)
ax.plot(x, lat, 'k',lw=2)
ax.set_xlabel('Scan number')
ax.set_xlim([0, 3660])
ax.set_ylabel('latitude')
ax.set_title('Latitudes')

fig.subplots_adjust(hspace=0.8)

fig.savefig('temperature_data_from_MTP_zero_angle_and_MERRA', dpi=600, bbox_inches='tight')

