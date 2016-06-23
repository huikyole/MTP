import utils
import data_processor as dp
from microwave_absorption_coeff import gas_absr
from Jacobian import calculate_K, calculate_F
from Jacobian_RH import calculate_K_RH
#from plots import plot_temperature_profile_from_pickle_file, plot_map_track

import numpy as np
import numpy.ma as ma
import math

import sys

import pickle 
from pickle import load
import yaml

from time import strptime
from datetime import datetime

def RH_to_q(RH, T):
    pws = np.exp(77.3450+0.0057*T-7235/T)/T**(8.2)   # saturation pressure of water vapor
    return RH*pws*2.2/T

def q_to_RH(q, T):
    pws = np.exp(77.3450+0.0057*T-7235/T)/T**(8.2)   # saturation pressure of water vapor
    return q*T/2.2/pws

def retrieve_MTHP_temp_from_ground(mtp_data, mtp_freq, mtp_angles, temp_obs, clim_q, heights, altitude, output_pickle='output.pickle'):
    '''
    temp_obs[nT, nZ]
    mtp_data
    clim_q 
    '''

    nfreq, nangle = mtp_data.shape
    cov_obs = np.matrix(np.cov(temp_obs.T, ddof=1))
    cov_mtp = np.matrix(np.diag(np.repeat(1,nfreq*nangle)))
    clim_obs = ma.mean(temp_obs, axis=0)

    nz = temp_obs.shape[1]

    # A: forward model for clim_obs, K: linearized Jacobian
    A,K = calculate_K(mtp_freq, mtp_angles, heights, altitude, temp_obs, clim_obs, clim_q)

    K = np.matrix(K)

    G = cov_obs*K.transpose()*np.linalg.inv(K*cov_obs*K.transpose() + cov_mtp)

	# mtp data => column matrix
    y = np.zeros(nfreq*nangle)
    for ii in np.arange(nfreq*nangle):
    	ifreq = ii % nfreq
    	iangle = int(ii/nfreq)
    	y[ii] = mtp_data[ifreq, iangle]
	# MAP solution
    T_est=np.matrix(clim_obs).transpose()+G*(np.matrix(y).transpose()-K*(np.matrix(clim_obs).transpose()))


	# Newton method

	# Initial guess (T = clim_obs)
    Ti = T_est
    Ta = np.matrix(clim_obs).transpose()

    print Ti
    for ii in np.arange(3):
		#F = calculate_F(mtp_freq, mtp_angles, np.array(Ti).ravel(), heights)
    	F,K = calculate_K(mtp_freq, mtp_angles, heights, altitude, temp_obs, np.array(Ti).ravel(), clim_q)  # A: forward model for clim_obs, K: linearized Jacobian
    	T_ii=Ta+ G*(np.matrix(y).transpose()-F*Ti+K*(Ti -Ta))
    	Ti = T_ii
        print Ti
  
    output = np.zeros([4,nz])
    output[0, :] = heights[:]
    output[1, :] = clim_obs
    output[2, :] = temp_obs[-1,:]
    output[3, :] = np.array(Ti[:]).ravel()

    if output_pickle !='':
        pickle.dump(output, open(output_pickle, 'wb'))
    
    return output

def retrieve_MTHP_RH_from_ground(mtp_data, mtp_freq, mtp_angles, rh_obs, retrieved_temp, heights, altitude, output_pickle='output_RH.pickle'):
    '''
    temp_obs[nT, nZ]
    mtp_data
    clim_q
    '''

    nfreq, nangle = mtp_data.shape
    cov_obs = np.matrix(np.cov(rh_obs.T, ddof=1))
    cov_mtp = np.matrix(np.diag(np.repeat(1,nfreq*nangle)))
    clim_obs = ma.mean(rh_obs, axis=0)

    nz = rh_obs.shape[1]

    # A: forward model for clim_obs, K: linearized Jacobian
    A,K = calculate_K_RH(mtp_freq, mtp_angles, heights, altitude, rh_obs, clim_obs, retrieved_temp)

    K = np.matrix(K)

    G = cov_obs*K.transpose()*np.linalg.inv(K*cov_obs*K.transpose() + cov_mtp)

    # mtp data => column matrix
    y = np.zeros(nfreq*nangle)
    for ii in np.arange(nfreq*nangle):
        ifreq = ii % nfreq
        iangle = int(ii/nfreq)
        y[ii] = mtp_data[ifreq, iangle]
    # MAP solution
    RH_est=np.matrix(clim_obs).transpose()+G*(np.matrix(y).transpose()-A*np.matrix(retrieved_temp).T-K*(np.matrix(clim_obs).transpose()))
    print clim_obs


    # Newton method

    # Initial guess (T = clim_obs)
    RHi = RH_est
    RHa = np.matrix(clim_obs).transpose()

    print RHi
    for ii in np.arange(3):
        F,K = calculate_K_RH(mtp_freq, mtp_angles, heights, altitude, rh_obs, np.array(RHi).ravel(), retrieved_temp)  # A: forward model for clim_obs, K: linearized Jacobian
        RH_ii=RHa+ G*(np.matrix(y).transpose()-F*np.matrix(retrieved_temp).T+K*(RHi -RHa))
        RHi = RH_ii
        print RHi

    output = np.zeros([4,nz])
    output[0, :] = heights[:]
    output[1, :] = clim_obs
    output[2, :] = rh_obs[-1,:]
    output[3, :] = np.array(RHi[:]).ravel()

    if output_pickle !='':
        pickle.dump(output, open(output_pickle, 'wb'))

    return output

config_file = yaml.load(open(str(sys.argv[1])))

# Hardware information
mthp_freq_temp = np.array([53, 54.2, 54.4, 54.6, 55.78, 56.36, 57.62, 58.2, 58.36]) # GHz
nfreq_temp = len(mthp_freq_temp)    # 9 channels
mthp_freq_rh = np.array([178.84, 179, 179.58, 180.84, 181.42, 182.6, 182.8, 183]) # GHz
nfreq_rh = len(mthp_freq_rh)    # 8 channels
nangle=7
mthp_angles = np.arange(nangle)*10+30
angles_rad = math.pi/180.*mthp_angles

obs_info = config_file['observation_info']
lon = obs_info['longitude']
lat = obs_info['latitude']
alt = obs_info['altitude']
time = datetime(*(strptime(obs_info['time'], '%m-%d-%Y_%H:%M:%S')[0:6]))

# MERRA2 reanalysis data
temp_merra, q_merra, rh_merra, heights_merra = dp.extract_merra_profiles('./data/merra2_temperature_and_moisture_June21_July10_1979-2015_over_CA.nc',time, lon, lat)


# Algorithm test
# Generate the pseudo mtp_data using the forward model
temp_obs, heights = utils.interpolate_linear(heights_merra, temp_merra, alt)
q_obs, heights = utils.interpolate_log_linear(heights_merra, q_merra, alt)
clim_obs = ma.mean(temp_obs, axis=0)
clim_q = ma.mean(q_obs, axis=0)
# A: forward model for clim_obs, K: linearized Jacobian
'''
A,K = calculate_K(mthp_freq_temp, angles_rad, heights, alt, temp_obs, clim_obs, clim_q)
test_data = np.array(A*(np.matrix(temp_obs[-1,:]).transpose())).ravel()
MTP_data_test = np.zeros([nfreq_temp, nangle])
for iangle in np.arange(nangle):
    MTP_data_test[:,iangle] = test_data[nfreq_temp*iangle:nfreq_temp*iangle+nfreq_temp] 

output_test = retrieve_MTHP_temp_from_ground(MTP_data_test, mthp_freq_temp, angles_rad, temp_obs, clim_q, heights, alt)
'''
output_test = load(open('output.pickle'))

nz2=10
heights2= heights[0:nz2]
retrieved_temp = output_test[3,0:nz2]
nyear, nz = q_obs.shape
rh_obs = np.zeros([nyear, nz2])
clim_obs2 = np.zeros(nz2)
for iz in np.arange(nz2):
    for iyear in np.arange(nyear):
        rh_obs[iyear, iz] = q_to_RH(q_obs[iyear, iz], temp_obs[iyear, iz])
clim_obs2=ma.mean(rh_obs,axis=0)
A2,K2 = calculate_K_RH(mthp_freq_rh, angles_rad, heights2, alt, rh_obs, clim_obs2, retrieved_temp)
test_data2 = np.array(A2*np.matrix(retrieved_temp).T + K2*np.matrix(rh_obs[-1,:]-clim_obs2).T).ravel()
MTP_data_test2= np.zeros([nfreq_rh, nangle])
for iangle in np.arange(nangle):
    MTP_data_test2[:,iangle] = test_data2[nfreq_rh*iangle:nfreq_rh*iangle+nfreq_rh]

output_test2= retrieve_MTHP_RH_from_ground(MTP_data_test2, mthp_freq_rh, angles_rad, rh_obs, retrieved_temp, heights2, alt)

'''
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
'''
