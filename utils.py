import numpy as np
import numpy.ma as ma
from scipy import interpolate

def retrieval_height(flight_level):
    '''
    Returns retrieval levels (up to 20)
    T1, T2, T19, T20: 1.6 km
    T3, T4, T17, T18: 0.8 km
    T5, T6, T15, T16: 0.4 km
    T7-T14: 0.2 km
    '''
    heights = np.zeros(20)
    delta_h = np.array([0.1, 0.3, 0.5, 0.7, 1.1, 1.5, 2.3, 3.1, 4.7, 6.3])  
    heights[10:] = flight_level+delta_h
    heights[:10] = flight_level-delta_h[::-1]

    return heights

def interpolate_temperature(merra_heights, merra_temp, flight_level):
    '''
    Interpolate merra temperature data into the retrieval height
    '''

    nt, nz = merra_temp.shape
    heights = retrieval_height(flight_level)
    nz2 = heights.size

    temp_interp = ma.zeros([nt, nz2])
   
    for it in np.arange(nt): 
        func = interpolate.interp1d(merra_heights, merra_temp[it,:]) 
        temp_interp[it,:] = func(heights)

    return temp_interp, heights
       
def interpolate_humidity(merra_heights, merra_q, flight_level):
    '''
    Interpolate merra water vapor density data into the retrieval height
    '''

    nt, nz = merra_q.shape
    heights = retrieval_height(flight_level)
    nz2 = heights.size

    q_interp = ma.zeros([nt, nz2])

    for it in np.arange(nt):
        func = interpolate.interp1d(merra_heights, np.log(merra_q[it,:]))
        q_interp[it,:] = np.exp(func(heights))

    return q_interp, heights
