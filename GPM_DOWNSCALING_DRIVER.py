#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import dataPre_Library_GPM as DL
from datetime import datetime
from mpl_toolkits.basemap import Basemap

### Basic informatior
region_info = {}
region_info['name'] = 'Japan'
region_info['minlat'] = 21.5 
region_info['minlon'] = 119.0
region_info['maxlat'] = 48.5
region_info['maxlon'] = 152.0
region_info['nlat_fine'] = 1351         # 2km 
region_info['nlon_fine'] = 1651         # 2km
region_info['res_fine'] = 0.2
#region_info['res_coarse'] = 0.25
#region_info['maxlat'] = region_info['minlat'] + region_info['res_fine']*(region_info['nlat_fine']-1)
#region_info['maxlon'] = region_info['minlon'] + region_info['res_fine']*(region_info['nlon_fine']-1)

date_info = {}
date_info['stime'] = datetime(2014, 5, 1)
date_info['ftime'] = datetime(2014, 5, 2)
#date_info['ntime'] = 1000              # should from the GPyM module

data_info = {}
data_info['prec_scale'] = 'linear'    # 'linear' or 'log'
#data_info['path_RF'] = '/home/wind/hexg/Research/Data/NLDAS2'

RF_config = {}
RF_config['rand_row_num'] = 25
RF_config['rand_col_num'] = 25
RF_config['ntree'] = 30
RF_config['njob'] = 6

features_name = {}
#features_name['static'] = ['slope', 'aspect', 'gtopomean', 'gtopostd', 'texture', 'vegeType']
features_name['GPM_GMI'] = ['cloudWaterPath', 'surfacePrecipitation']
#features_name['GPM_GMI'] = ['cloudWaterPath', 'convectPrecipFraction', 'iceWaterPath', 'liquidPrecipFraction', 'mixedWaterPath', 'mostLikelyPrecipitation', 'precip1stTertial', 'precip2ndTertial', 'probabilityOfPrecip', 'rainWaterPath', 'snowCoverIndex', 'surfacePrecipitation', 'surfaceSkinTempIndex', 'surfaceTypeIndex', 'temp2mIndex', 'totalColumnWaterVapor', 'totalColumnWaterVaporIndex']

### Use Random Forests model for precipitation downscaling
RFDS = DL.RandomForestsDownScaling(region_info, date_info, data_info, RF_config, features_name)
#cov_GPM = RFDS.get_GPM()
#prec_GPM_KuPR = RFDS.prepare_prec_fine(prec_scale='linear')
#train_df, test_df = RFDS.prepare_training_testing_data()
RFDS.fit_RF()
RFDS.predict_RF_mean()

