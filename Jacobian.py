import numpy as np
from microwave_absorption_coeff import gas_absr
from scipy import stats

def calculate_K(mtp_freq, angles_rad, heights, flight_lev, temp_obs, clim_obs, clim_q, bandwidth=np.repeat(0, 21)):
    '''
    Calculate Jacobian matrix (K) of the forward matrix(A)
    K_ij = dy_i/dx_j at x = clim_obs
    '''
    num_of_band = 11

    N = 30
    M = 20

    nfreq = 3
    nz = M
    nyear = 36

    H = 8. # scale height

    pressure = 1000.*np.exp(-heights/H)

    K = np.zeros([N, M])

    absr_coeff_clim = np.zeros([nfreq, M])
    absr_coeff = np.zeros([nfreq, M])

    angles = np.zeros(N)
    for iangle, angle in enumerate(angles_rad):
        angles[3*iangle:3*iangle+3] = np.repeat(angle, 3)

    for iz in np.arange(M):
        for ifreq in np.arange(nfreq):
            if bandwidth[iz] > 0:
                for freq in np.linspace(-bandwidth[iz], bandwidth[iz] ,num_of_band) + mtp_freq[ifreq]:
                    absr_coeff_clim[ifreq, iz] = absr_coeff_clim[ifreq, iz] + gas_absr(freq, clim_obs[iz], pressure[iz], clim_q[iz])/num_of_band
            # broadening should be considered here
            else:
                absr_coeff_clim[ifreq, iz] = gas_absr(mtp_freq[ifreq], clim_obs[iz], pressure[iz], clim_q[iz])   

    A = calculate_forward_matrix(absr_coeff_clim, heights, angles)
    y0 = np.array(np.matrix(A)*np.matrix(clim_obs).transpose()).ravel()

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
                        absr_coeff[ifreq, iz] = absr_coeff[ifreq, iz]+gas_absr(freq, temp_obs[iyear,iz], pressure[iz], clim_q[iz])/num_of_band   # absr_coeff[:,iz] is different
                else: 
                    absr_coeff[ifreq, iz] = gas_absr(mtp_freq[ifreq], temp_obs[iyear,iz], pressure[iz], clim_q[iz])
            temp_array[:] = clim_obs[:]
            temp_array[iz] = temp_obs[iyear,iz]
            temp_A = calculate_forward_matrix(absr_coeff, heights, angles) # temporary forward model matrix
            y[iyear,:] = np.array(np.matrix(temp_A)*np.matrix(temp_array).transpose()).ravel()

        for iobs in np.arange(N):
            #slope, intercept, _, _, _ = stats.linregress(temp_obs[:,iz], y[:,iobs])     # non-zero intercept
            x = temp_obs[:,iz]-clim_obs[iz]
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

    angles = np.zeros(30)
    for iangle, angle in enumerate(angles_rad):
        angles[3*iangle:3*iangle+3] = np.repeat(angle, 3)

    M = len(temperature)
    nfreq = len(mtp_freq)
    absr_coeff = np.zeros([nfreq, M])
    for iz in np.arange(M):
        for ifreq in np.arange(nfreq):
            absr_coeff[ifreq, iz] = gas_absr(mtp_freq[ifreq], temperature[iz], pressure[iz], clim_q[iz])
    return np.matrix(calculate_forward_matrix(absr_coeff, heights, angles))


def calculate_forward_matrix(absr_coeff, heights, angles):
    '''
    Calculate a Forward model
    '''

    thickness = [1.6, 1.6, 0.8, 0.8, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4,0.8, 0.8, 1.6, 1.6]

    N = angles.size   # 30
    M = heights.size  # 20

    A = np.zeros([N, M])

    # Angles below
    for ii in np.arange(12):
        ifreq = ii % 3
        theta = angles[ii]
        transmittance = np.zeros(M)
        absorptivity = np.zeros(M)
        for jj in np.arange(M):
            transmittance[jj] = np.exp(-absr_coeff[ifreq, jj]*thickness[jj]/abs(np.sin(theta)))
            absorptivity[jj] = 1-transmittance[jj]
        for jj in np.arange(9):
            A[ii,jj] = absorptivity[jj]*np.prod(transmittance[jj+1:10])
        A[ii,9] = absorptivity[9]

    # Zero Angles
    for ii in np.arange(3)+12:
        A[ii,9] = 1.
        A[ii,10] = 1. 

    # Angles above
    for ii in np.arange(15)+15:
        ifreq = ii % 3
        theta = angles[ii]
        transmittance = np.zeros(M)
        absorptivity = np.zeros(M)
        for jj in np.arange(M):
            transmittance[jj] = np.exp(-absr_coeff[ifreq, jj]*thickness[jj]/abs(np.sin(theta)))
            absorptivity[jj] = 1-transmittance[jj]
        for jj in np.arange(9)+11:
            A[ii,jj] = absorptivity[jj]*np.prod(transmittance[10:jj])
        A[ii,10] = absorptivity[10]
 
    # Normalization
    for ii in np.arange(N):
        A[ii,:] =  A[ii,:]/np.sum(A[ii,:])

    return A

