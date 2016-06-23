import numpy as np
import numpy.ma as ma
from scipy import interpolate

def retrieval_height_from_ground(ground_altitude):
    '''
    Returns retrieval levels: 20
    T1 - T5: 0.1 km
    T6 - T10: 0.2 km 
    T11 - T15: 0.4 km 
    T16 - T20: 0.8 km 
    '''
    nz = 20
    heights = np.zeros(nz)
    delta_h = np.repeat([0.1, 0.2, 0.4, 0.8],5)  
    heights[0] = delta_h[0]*0.5+ground_altitude 
    for iz in np.arange(nz-1)+1:
        heights[iz] = heights[iz-1] + delta_h[iz]

    return heights

def interpolate_linear(merra_heights, merra_temp, observation_level):
    '''
    Interpolate merra temperature or RH data into the retrieval height
    '''

    nt, nz = merra_temp.shape
    heights = retrieval_height_from_ground(observation_level)
    nz2 = heights.size

    temp_interp = ma.zeros([nt, nz2])

    for it in np.arange(nt): 
        merra_temp2 = merra_temp[it,:]
        merra_heights2= merra_heights[merra_temp2.mask == False]
        func = interpolate.interp1d(merra_heights2, merra_temp2[merra_temp2.mask == False], fill_value='extrapolate') 
        temp_interp[it,:] = func(heights)

    return temp_interp, heights
       
def interpolate_log_linear(merra_heights, merra_q, observation_level):
    '''
    Interpolate exponentially varying variables such as water vapor density data into the retrieval height
    '''

    nt, nz = merra_q.shape
    heights = retrieval_height_from_ground(observation_level)
    nz2 = heights.size

    q_interp = ma.zeros([nt, nz2])

    for it in np.arange(nt):
        merra_q2 = merra_q[it,:]
        merra_heights2= merra_heights[merra_q2.mask == False]
        func = interpolate.interp1d(merra_heights2, np.log(merra_q2[merra_q2.mask == False]), fill_value='extrapolate')
        q_interp[it,:] = np.exp(func(heights))
    return q_interp, heights
