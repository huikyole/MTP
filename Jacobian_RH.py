import numpy as np
from microwave_absorption_coeff import gas_absr
from scipy import stats

def RH_to_q(RH, T):
    pws = np.exp(77.3450+0.0057*T-7235/T)/T**(8.2)   # saturation pressure of water vapor 
    return RH*pws*2.2/T

def calculate_K_RH(mtp_freq, angles_rad, heights, flight_lev, rh_obs, clim_obs, clim_temp, bandwidth=np.repeat(0, 21)):
    '''
    Calculate Jacobian matrix (K) of the forward matrix(A)
    K_ij = dy_i/dx_j at x = clim_obs
    '''
    H = 8. 
    pressure = 1000.*np.exp(-heights/H)

    num_of_band = 11

    nfreq = len(mtp_freq)
    nangle = len(angles_rad)
    N = nfreq*nangle                    
    M = len(heights)

    nyear = rh_obs.shape[0]

    K = np.zeros([N, M])

    absr_coeff_clim = np.zeros([nfreq, M])
    absr_coeff = np.zeros([nfreq, M])

    angles = np.repeat(angles_rad, nfreq)

    q = np.zeros([nyear, M])
    for iz in np.arange(M):
        for iyear in np.arange(nyear):
            q[iyear, iz] = RH_to_q(rh_obs[iyear, iz], clim_temp[iz]) 

    for iz in np.arange(M):
        q_clim = RH_to_q(clim_obs[iz], clim_temp[iz])
        for ifreq in np.arange(nfreq):
            if bandwidth[iz] > 0:
                for freq in np.linspace(-bandwidth[iz], bandwidth[iz] ,num_of_band) + mtp_freq[ifreq]:
                    absr_coeff_clim[ifreq, iz] = absr_coeff_clim[ifreq, iz] + gas_absr(freq, clim_temp[iz], pressure[iz], q_clim)/num_of_band
            # broadening should be considered here
            else:
                absr_coeff_clim[ifreq, iz] = gas_absr(mtp_freq[ifreq], clim_temp[iz], pressure[iz], q_clim)   
    A = calculate_forward_matrix(absr_coeff_clim, heights, angles)
    y0 = np.array(np.matrix(A)*np.matrix(clim_temp).transpose()).ravel()

    for iz in np.arange(M):    # for each x_j
        temp_array = np.zeros(M)
        #temp_array[:] = clim_obs
        y = np.zeros([nyear, N])
        for iyear in np.arange(nyear):
            absr_coeff = np.zeros(absr_coeff_clim.shape)        # Initialize
            absr_coeff[:] = absr_coeff_clim[:]
            absr_coeff[:,iz] = 0.
            for ifreq in np.arange(nfreq):
                if bandwidth[iz] > 0: 
                    for freq in np.linspace(-bandwidth[iz], bandwidth[iz] ,num_of_band) + mtp_freq[ifreq]:
                        absr_coeff[ifreq, iz] = absr_coeff[ifreq, iz]+gas_absr(freq, clim_temp[iz], pressure[iz], q[iyear, iz])/num_of_band   # absr_coeff[:,iz] is different
                else: 
                    absr_coeff[ifreq, iz] = gas_absr(mtp_freq[ifreq], clim_temp[iz], pressure[iz], q[iyear, iz])
            temp_A = calculate_forward_matrix(absr_coeff, heights, angles) # temporary forward model matrix
            y[iyear,:] = np.array(np.matrix(temp_A)*np.matrix(clim_temp).transpose()).ravel()

        for iobs in np.arange(N):
            #slope, intercept, _, _, _ = stats.linregress(temp_obs[:,iz], y[:,iobs])     # non-zero intercept
            x = rh_obs[:,iz] - clim_obs[iz]
            x = x[:,np.newaxis] 
            slope, _, _, _ = np.linalg.lstsq(x, y[:,iobs]-y0[iobs])
            K[iobs, iz] = slope 

        #for ii in np.arange(N):
        #    K[ii,:] = K[ii,:]/np.sum(K[ii,:]) 
             
    return A, K
    
def calculate_F(mtp_freq, angles_rad, temperature, heights):
    '''
    Calculate forward model using retrieved temperature
    '''
    H = 8. 
    pressure = 1000.*np.exp(-heights/H)

    nfreq = len(mtp_freq)
    nangle = len(angles_rad)
    N = nfreq*nangle                    
    angles = np.repeat(angles_rad, nfreq)

    M = len(temperature)
    absr_coeff = np.zeros([nfreq, M])
    for iz in np.arange(M):
        for ifreq in np.arange(nfreq):
            absr_coeff[ifreq, iz] = gas_absr(mtp_freq[ifreq], temperature[iz], pressure[iz], clim_q[iz])
    return np.matrix(calculate_forward_matrix(absr_coeff, heights, angles))


def calculate_forward_matrix(absr_coeff, heights, angles):
    '''
    Calculate a Forward model
    angles: repeated array of size N
    '''
  
    nfreq = absr_coeff.shape[0]
    N = angles.size                    
    M = heights.size 
    thickness = np.zeros(M)
    thickness[0] = heights[1]-heights[0]
    for iz in np.arange(M-1)+1:
        thickness[iz] = heights[iz] - heights[iz-1]

    A = np.zeros([N, M])
    # Angles above
    for ii in np.arange(N):
        ifreq = ii % nfreq
        theta = angles[ii]
        transmittance = np.zeros(M)
        absorptivity = np.zeros(M)
        for jj in np.arange(M):
            transmittance[jj] = np.exp(-absr_coeff[ifreq, jj]*thickness[jj]/abs(np.sin(theta)))
            absorptivity[jj] = 1-transmittance[jj]
        for jj in np.arange(M-1)+1:
            A[ii,jj] = absorptivity[jj]*np.prod(transmittance[0:jj])
        A[ii,0] = absorptivity[0]
 
    # Normalization
    for ii in np.arange(N):
        A[ii,:] =  A[ii,:]/np.sum(A[ii,:])

    return A

