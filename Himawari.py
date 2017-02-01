#-*-coding:utf-8-*-

#!/usr/bin/env python

# This is the code to read the Himawari sample dataset

from mpl_toolkits.basemap import Basemap
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

dir_H08 = '/data2/hexg/Downscaling/Data/Himawari/NC_H08_20150519_0110_JP02'
dir_fig = '/data2/hexg/Downscaling/Figures'

band_num = 16 

#band = Dataset('%s/NC_H08_20150519_0110_B%02d_JP02.nc' % (dir_H08, band_num)).variables['albedo'][:]
band = Dataset('%s/NC_H08_20150519_0110_B%02d_JP02.nc' % (dir_H08, band_num)).variables['tbb'][:]

plt.figure()
M = Basemap(resolution='h' ,llcrnrlat=21.5, llcrnrlon=119, urcrnrlat=48.5, urcrnrlon=152)
#M.imshow(band[::-1], vmin=0, vmax=1)
M.imshow(band[::-1], vmin=200, vmax=310)
M.drawcoastlines(color='r')
M.colorbar()

plt.savefig('%s/Himawari_sample_band%02d.png' % (dir_fig, band_num))
plt.show()

