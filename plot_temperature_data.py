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
for icase in np.arange(ncase):
    print icase
    if hgt[icase] > 7.:
        temp_merra, q_merra, heights_merra = dp.merra_climatology_and_covariance(time[icase], lon[icase], lat[icase])

        temp_obs, heights = utils.interpolate_temperature(heights_merra, temp_merra, hgt[icase])
        merra_temp[0,icase] = ma.mean(temp_obs[:,9]+temp_obs[:,10])*0.5
        merra_temp[1,icase] = (temp_obs[-1,9]+temp_obs[-1,10])*0.5

merra_temp = ma.masked_equal(merra_temp, -9999.)

fig = plt.figure()
ax = fig.add_subplot(211)
x = np.arange(ncase)+1
ax.plot(x,merra_temp[0,:], label='MERRA Clim.', lw=2)
ax.plot(x,merra_temp[1,:], label='MERRA in 2014', lw=2)
for iangle in np.arange(3):
    ax.plot(x, mtp_data[iangle,5,:], label = 'MTP freq %d' %(iangle+1))
ax.set_xlabel('Scan number')
ax.set_ylabel('Temperature [K]')
ax.set_xlim([0, 3660])
ax.set_title('Comparison of the temperature between MTP and MERRA')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), fancybox=True, shadow=True, ncol=5, frameon=False, fontsize='small')

ax = fig.add_subplot(212)
ax.plot(x, lat, 'k',lw=2)
ax.set_xlabel('Scan number')
ax.set_xlim([0, 3660])
ax.set_ylabel('latitude')
ax.set_title('Latitudes')

fig.subplots_adjust(hspace=0.6)

fig.savefig('temperature_data_from_MTP_zero_angle_and_MERRA', dpi=600, bbox_inches='tight')

