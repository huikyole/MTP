import utils
import data_processor as dp
from microwave_absorption_coeff import gas_absr
from Jacobian import calculate_K, calculate_F
from plots import plot_temperature_profile_from_pickle_file, plot_map_track

import numpy as np
import numpy.ma as ma
import math

import sys

import pickle 
import yaml

def retrieve_MTP_temp(case, broadening, output_pickle='', broadening_altitude=False):
    '''
    case - case number
    broadening - array of broadening in GHz
    output_pickle: output pickle file containing 
    1) altitude, 2) climatological temperature, 3) observed temperature in 2014, and 4) retrieved temperature
    '''
    H = 8   # scale height 
    mtp_freq = [56.363, 57.612, 58.363]
    angles = [60,44.4,30,17.5,8.6,0,-8.6,-20.5,-36.9,-58.2]
    angles = np.array(angles[::-1])
    angles_rad = math.pi/180.*angles

    time, lon, lat, hgt, out_temp, mtp_data = dp.MTP_obs_extract(case)  # hgt: flight altitude

    temp_merra, q_merra, heights_merra = dp.merra_climatology_and_covariance(time, lon, lat)

    temp_obs, heights = utils.interpolate_temperature(heights_merra, temp_merra, hgt)
    q_obs, heights = utils.interpolate_humidity(heights_merra, q_merra, hgt)
    pressure = 1000.*np.exp(-heights/H)

    if broadening_altitude:
        broadening = [broadening.max()*ii/pressure[0] for ii in pressure]

    cov_obs = np.matrix(np.cov(temp_obs.T, ddof=1))
    cov_mtp = np.matrix(np.diag(np.repeat(1,30)))
    clim_obs = ma.mean(temp_obs, axis=0)
    clim_q = ma.mean(q_obs, axis=0)

    nfreq =3
    nangle = 10
    nz = 20

    # A: forward model for clim_obs, K: linearized Jacobian
    A,K = calculate_K(mtp_freq, angles_rad, heights, hgt, temp_obs, clim_obs, clim_q, bandwidth=broadening)

    K = np.matrix(K)

    G = cov_obs*K.transpose()*np.linalg.inv(K*cov_obs*K.transpose() + cov_mtp)

	# mtp data => column matrix
    y = np.zeros(30)
    for ii in np.arange(30):
    	ifreq = ii % 3
    	iangle = int(ii/3)
    	y[ii] = mtp_data[ifreq, iangle]

	# MAP solution
    T_est=np.matrix(clim_obs).transpose()+G*(np.matrix(y).transpose()-A*(np.matrix(clim_obs).transpose()))


	# Newton method

	# Initial guess (T = clim_obs)
    Ti = T_est
    Ta = np.matrix(clim_obs).transpose()


    for ii in np.arange(3):
		#F = calculate_F(mtp_freq, angles_rad, np.array(Ti).ravel(), heights)
    	F,K = calculate_K(mtp_freq, angles_rad, heights, hgt, temp_obs, np.array(Ti).ravel(), clim_q, bandwidth=broadening)  # A: forward model for clim_obs, K: linearized Jacobian
    	T_ii=Ta+ G*(np.matrix(y).transpose()-F*Ti+K*(Ti -Ta))
    	Ti = T_ii
  
    output = np.zeros([4,nz])
    output[0, :] = heights[:]
    output[1, :] = clim_obs
    output[2, :] = temp_obs[-1,:]
    output[3, :] = np.array(Ti[:]).ravel()

    if output_pickle !='':
        pickle.dump(output, open(output_pickle, 'wb'))
    
    return output


# Parse configuration
config_file = yaml.load(open(str(sys.argv[1])))

case_info = config_file['case']
ncase = case_info['case_end']-case_info['case_start']+1

nz = 20  # number of retrieved levels
array = ma.zeros([ncase, 4, nz]) - 999.
start = case_info['case_start']
longitudes = np.zeros(ncase)-999.
latitudes = np.zeros(ncase)-999.

# Retrieve a vertical temperature profile when flight altitude is higher than 7 km
for icase in np.arange(start, start+ncase):
    print 'case_number:',icase
    time, lon, lat, hgt, out_temp, mtp_data = dp.MTP_obs_extract(icase)  # hgt: flight altitude
    if hgt >= 7.:
        array[icase-start,:] = retrieve_MTP_temp(icase,np.repeat(0., 20))
        longitudes[icase-start] = lon
        latitudes[icase-start] = lat

# Save the retrieved temperature in a pickle file
array = ma.masked_less(array,0)
pickle.dump(array, open(config_file['output_file'],'wb'))

# Plotting figures
figure_info = config_file['figure']

if figure_info['draw_figure']:
    plot_map_track(longitudes, latitudes, figure_info['map_filename'] )
    plot_temperature_profile_from_pickle_file(config_file['output_file'], figure_info['profile_filename'])     
