import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pickle import load

def plot_temperature_profile_from_pickle_file(pickle_file, figure_file):
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    data = load(open(pickle_file))[0,:]
    ax.plot(data[1,:], data[0,:],'r',linewidth=2, label='a priori (climatology)')
    ax.plot(data[2,:], data[0,:], 'b-o',linewidth=2, label='observed (2014)')
    ax.plot(data[3,:], data[0,:], 'g-o',linewidth=2, label='retrieved (2014)')

    ax.legend(loc='upper center', bbox_to_anchor=(1.2, 0.5))
    ax.set_xlim([190,280])
    ax.set_xlabel('T [K]')
    ax.set_xticks(np.arange(5)*20+190)
    ax.set_ylabel('Altitude [km]')
    plt.show()
    fig.savefig(figure_file,dpi=600,bbox_inches='tight') 

def plot_map_track(lons, lats, figure_file):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    m = Basemap(ax = ax, llcrnrlon=100.,llcrnrlat=-60.,urcrnrlon=215,urcrnrlat=30)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color = 'coral')
    m.drawmapboundary()

    npoints = len(lons)
    for ipoint in np.arange(npoints):
        x,y = m(lons[ipoint], lats[ipoint])
        m.plot(x,y,'bo', markersize = 16)
    plt.show()
    fig.savefig(figure_file,dpi=600,bbox_inches='tight') 
    
