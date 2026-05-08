import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pandas as pd
import csv
import re
import os
from calendar import monthrange
import math
from GO_helpers import is_leap_year, buildSortableString, saveASCFile, getDataFrame 
from functools import partial

#import matplotlib.pyplot as plt

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# created by GO, last edited by GO 2024-05-23

# purpose: prep files for ECOSPACE - 3 day averages (and ECOSIM 1D TS) 
#          input: daily mean NEMO outputs
#          output: depth integrated and single levels for key vars output as ASC (daily) for daily NEMO model

#Data in:
# 1/ NEMO outputs in 2D or 3D (var mldkz5 / turbocline depth) as daily means in annual NC files, DIRECT from NEMO outputs "xxx_1d_grid_T_2D_1980.nc"
# 2/ Salinity - can be either vertical averages over chosen levels or from a specific depth level, DIRECT from NEMO outputs "xxx_1d_grid_T_selectedlevs_1980.nc"
# 3/ Light as daily means from climate re-analysis (e.g. ERA5) with near-surface downward radiative flux (var msdwswrf Wm-2) interpolated to NEMO grid as annual NC files# 
# file name e.g. "ERA5_NEMOgrid_light_daily_1980.nc".

# Log:
# - GO 2024-05-21 edits to write out daily files for non-depth integrated salinity (e.g., just 4 m)
# - GO 2024-05-22 note that the depth averaging is not right - needs to be weighted by level width (code exists somewhere to fix), not major issue for shallows
# - GO 2024-05-23 Ecospace and Ecosim have max of 500 yr simulations - must therefore change for 2015-2018 validation using PP to 3-day model
#                 Leap years divide by 3 well - 122 'years' per year, others are 121 full time steps and the 0.66 x 3 days get dropped
# - GO 2024-05-24 Ecosim needs monthly forcing TS in order for enviro functions be set up
#                 Can be more efficient in the main day loop - eliminate redundant code, loop over vars
#                 Made this much more efficient by selecting desired depths during each month loop instead of grabbing all depths
#                 Fixed the redundant loops and repeated code by adding dictionary
# - GO 2024-05-27 Due to Ecospace and Ecosim limit of max 500 years, I treated 3-day outputs as monthly. 
#                 Meaning one model 'year' is actually 36 days. 
#                 Does not divide into 365 easily.
#                 Modified code to lump the remaining 5-6 days in December into average of final time step so the output is compatible with EwE spatial-temp framework
# - GO 2024-08-31 Ran after the Graham problems, used module load StdEnv/2020, module load scipy-stack/2023b, module load netcdf
# - GO 2024-09-27 Depth avg'ing still not right!
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# //////////////////////////////////// FUNCTIONS //////////////////////////////////////
# now in GO_helpers.py -2023-04-05


# originally CDO was used, it is much faster but I don't have time to implement here
# ################# Basic Params #################
startyear = 1980
endyear = 1989
startMonth = 1
endMonth = 12
NEMO_run = "216" #NEMO run code
out_code = "var" 
timestep = "3day"

#multiplier is 100 to convert from m/s (NEMO) to cm/s (EwE) - GO 2022-02
#then for tricking Ecospace to run daily divide by 365 * 3 for 3-day
# 'Advection vectors Xvel(,) are in cm/sec convert to km/year, same units as the mvel()/(Dispersal Rate)
# '[km/year] / [cell length]
# AdScale = 315.36 / Me.EcoSpaceData.CellLength
# AdScale = 315.36 / Me.EcoSpaceData.CellLength
advec_mult = 100 / 365 * 3 # see AdScale in cEcospace
upwel_mult = 1000 / 365 * 3 # see AdScale in cEcospace

#lump final days (121 or 121,122) into 120th time step?
lump_final = True

dctVariables = {"salt1_PSU":{
                    "varname": "vosaline",
                    "2Dor3D": "3D",
                    "depthlev1":0,
                    "depthlev2":10,
                    "depthlev_avg":True, # if false, uses just depth_lev2 vals
                    "multiplier":1, #for units conversion
                    "sigdigfmt":'%0.1f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_T_y{}m{}.nc" ,
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                 "salt2_PSU":{
                    "varname": "vosaline",
                    "2Dor3D": "3D",
                    "depthlev1":0,
                    "depthlev2":4,
                    "depthlev_avg":False,
                    "multiplier":1, #for units conversion
                    "sigdigfmt":'%0.1f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_T_y{}m{}.nc" ,
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                 "salt3_PSU":{
                    "varname": "vosaline",
                    "2Dor3D": "3D",
                    "depthlev1":30,
                    "depthlev2":40,
                    "depthlev_avg":True,
                    "multiplier":1, #for units conversion
                    "sigdigfmt":'%0.1f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_T_y{}m{}.nc" ,
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                  "temp1_C":{
                    "varname": "votemper",
                    "2Dor3D": "3D",
                    "depthlev1":0,
                    "depthlev2":10,
                    "depthlev_avg":True,
                    "multiplier":1, #for units conversion
                    "sigdigfmt":'%0.1f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_T_y{}m{}.nc" ,
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                  "temp2_C":{
                    "varname": "votemper",
                    "2Dor3D": "3D",
                    "depthlev1":0,
                    "depthlev2":4,
                    "depthlev_avg":False,
                    "multiplier":1, #for units conversion
                    "sigdigfmt":'%0.1f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_T_y{}m{}.nc" ,
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },                    
                  "temp3_C":{
                    "varname": "votemper",
                    "2Dor3D": "3D",
                    "depthlev1":30,
                    "depthlev2":40,
                    "depthlev_avg":True,
                    "multiplier":1, #for units conversion
                    "sigdigfmt":'%0.1f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_T_y{}m{}.nc" ,
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                  "mixing_m":{
                    "varname": "mldkz5",
                    "2Dor3D": "2D",
                    "depthlev1":0,
                    "depthlev2":0,
                    "depthlev_avg":False,
                    "multiplier":1, #for units conversion
                    "sigdigfmt":'%0.1f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_T_2D_y{}m{}.nc",
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                    "mixed_m":{
                    "varname": "mldr10_1",
                    "2Dor3D": "2D",
                    "depthlev1":0,
                    "depthlev2":0,
                    "depthlev_avg":False,
                    "multiplier":1, #for units conversion
                    "sigdigfmt":'%0.1f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_T_2D_y{}m{}.nc",
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                  "advecX1_cmsec":{
                    "varname": "vozocrtx",
                    "2Dor3D": "3D",
                    "depthlev1":0,
                    "depthlev2":10,
                    "depthlev_avg":True,
                    "multiplier":advec_mult, #for units conversion
                    "sigdigfmt":'%0.3f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_U_y{}m{}.nc",
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                "advecY1_cmsec":{
                    "varname": "vomecrty",
                    "2Dor3D": "3D",
                    "depthlev1":0,
                    "depthlev2":10,
                    "depthlev_avg":True,
                    "multiplier":advec_mult, #for units conversion
                    "sigdigfmt":'%0.3f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_V_y{}m{}.nc",
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                  "advecX2_cmsec":{
                    "varname": "vozocrtx",
                    "2Dor3D": "3D",
                    "depthlev1":0,
                    "depthlev2":40,
                    "depthlev_avg":True,
                    "multiplier":advec_mult, #for units conversion
                    "sigdigfmt":'%0.3f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_U_y{}m{}.nc",
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                "advecY2_cmsec":{
                    "varname": "vomecrty",
                    "2Dor3D": "3D",
                    "depthlev1":0,
                    "depthlev2":40,
                    "depthlev_avg":True,
                    "multiplier":advec_mult, #for units conversion
                    "sigdigfmt":'%0.3f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_V_y{}m{}.nc",
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                "upwelZ1_mmsec":{
                    "varname": "vovecrtz",
                    "2Dor3D": "3D",
                    "depthlev1":4,
                    "depthlev2":12,
                    "depthlev_avg":True,
                    "multiplier":upwel_mult, #for units conversion
                    "sigdigfmt":'%0.5f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_W_y{}m{}.nc",
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    },
                "upwelZ2_mmsec":{
                    "varname": "vovecrtz",
                    "2Dor3D": "3D",
                    "depthlev1":4,
                    "depthlev2":40,
                    "depthlev_avg":True,
                    "multiplier":upwel_mult, #for units conversion
                    "sigdigfmt":'%0.5f',
                    "dir_in":"/project/6006412/mdunphy/nemo_results/SalishSea1500/",
                    "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                    "filename":"SalishSea1500-RUN{}_1d_grid_W_y{}m{}.nc",
                    "dir_out":"/project/6006412/goldford/ECOSPACE/",
                    "ecospace_subdir_out":"DATA/SS1500-RUN{}/ECOSPACE_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecospace": True,
                    "ecosim_subdir_out":"DATA/SS1500-RUN{}/ECOSIM_in_{}_vars".format(NEMO_run, timestep),
                    "save_ecosim": True
                    }
                }


# avg_advec = True
# if avg_advec == True: 
  # out_code_advecX = out_code + "{}_{}mAvg".format("advecX", advec_depthlev)
  # out_code_advecY = out_code + "{}_{}mAvg".format("advecY", advec_depthlev)
# else:
  # out_code_advecX = out_code + "{}_{}m".format("advecX", advec_depthlev)
  # out_code_advecY = out_code + "{}_{}m".format("advecY", advec_depthlev)


# ################# ASC Params #################
ASCheader = "ncols        93\nnrows        151 \nxllcorner    -123.80739\nyllcorner    48.29471 \ncellsize     0.013497347 \nNODATA_value  0.0"


# ################# PATHS #################
plume_dir = "/project/6006412/goldford/ECOSPACE/"
plume_subdir = "DATA/"
ecospacegrid_dir = "/project/6006412/goldford/ECOSPACE/"
ecospacegrid_subdir = "DATA/"
#greigDir = "/project/6006412/goldford/ECOSPACE/"
#michaelDir = "/project/6006412/mdunphy/nemo_results/SalishSea1500/"
#lightSubDir = "DATA/light_monthly/"
lightSubDir = "DATA/light_daily/"

#depthSubDir = "DATA/"
# old
#mixingSubDir = "DATA/SS1500-RUN{}/NEMO_annual_NC".format(NEMO_run)
#salinitySubDir = "DATA/SS1500-RUN{}/NEMO_annual_NC".format(NEMO_run)
#outputsSubDir ="DATA/SS1500-RUN{}/ECOSPACE_in_{}".format(NEMO_run, PAR_code)

# new (for daily)
#mixingSubDir = "SalishSea1500-RUN{}/CDF".format(NEMO_run)
#salinitySubDir = "SalishSea1500-RUN{}/CDF".format(NEMO_run)
#temperatureSubDir = "SalishSea1500-RUN{}/CDF".format(NEMO_run)
#advecSubDir = "SalishSea1500-RUN{}/CDF".format(NEMO_run)
#outputsSubDir ="DATA/SS1500-RUN{}/ECOSPACE_in_3daily_vars".format(NEMO_run)

# OLD
#mixingFileName = 'SalishSea1500-RUN{}_MonthlyMean_grid_T_2D_y{}.nc'
#lightFileName = "RDRS21_NEMOgrid_light_monthly_{}.nc"  
#salinityFileName ='SalishSea1500-RUN{}_MonthlyMean_grid_T_y{}.nc' 
#salinityFileName ='SalishSea1500-RUN{}_MonthlyMean_grid_T_singledepth10m_{}.nc' #depth=9.5m
#salinityFileName ='SalishSea1500-RUN{}_MonthlyMean_grid_T_avgtop10m_y{}.nc'

# NEW - for daily
#mixingFileName = 'SalishSea1500-RUN{}_1d_grid_T_2D_y{}m{}.nc'
lightFileName = "RDRS21_NEMOgrid_light_daily_{}.nc"
#salinityFileName ='SalishSea1500-RUN{}_1d_grid_T_y{}m{}.nc'
#temperatureFileName ='SalishSea1500-RUN{}_1d_grid_T_y{}m{}.nc' 
#advectXFileName ='SalishSea1500-RUN{}_1d_grid_U_y{}m{}.nc' 
#advectYFileName ='SalishSea1500-RUN{}_1d_grid_V_y{}m{}.nc' 


# ################# VARS #################
varNameLight = "solar_rad"
#varNameSalinity = 'vosaline'
#varNameTemp = 'votemper'
#varNamesMixing = 'mldkz5' #turbocline
#varNameXadvect = 'vozocrtx' #(U)
#varNameYadvect = 'vomecrty' #(V) grid
#multiplier is 100 to convert from m/s (NEMO) to cm/s (EwE) - GO 2022-02
#then for tricking Ecospace to run daily divide by 365 * 3 for 3-day
# 'Advection vectors Xvel(,) are in cm/sec convert to km/year, same units as the mvel()/(Dispersal Rate)
# '[km/year] / [cell length]
# AdScale = 315.36 / Me.EcoSpaceData.CellLength
#advec_mult = 100 / 365 * 3 # see AdScale in cEcospace
#lstVars = ['mldkz5']; 

saveTemplate = "{}_{}_{}.asc" # new: ewename, year, dayofyear

#Number of depth strata to use
#Depth indexes are zero based
#So 10 will be the 9 index
#nDepthIndexes = 10 #= 9.001736   10.003407
#nDepthIndexes = 23 #= 31.101034   39.11886
#nDepthIndexes = 26 #= 86.96747   109.73707 


# ################# MAP CLIP #################
#array offsets for clipping EwE grid out of NEMO grid
#Calculated by Greig in "Basemap Converter (Py3).ipynb"
upperleft_row_ewe = 253
upperleft_col_ewe = 39
bottomleft_row_ewe = 102
bottomleft_col_ewe = 39


# ################# PLUME MASK #################
#Create the plume mask
PlumeRegionFilename = os.path.join(plume_dir, plume_subdir, "Fraser_Plume_Region.asc")
dfPlumeRegion = getDataFrame(PlumeRegionFilename,"-9999.00000000")
dfPlumeMask = dfPlumeRegion == 1 #Plume is region one


# ################# LAND MASK #################
ecospacegrid_f = os.path.join(ecospacegrid_dir, ecospacegrid_subdir, "ecospacedepthgrid.asc")
df_dep = getDataFrame(ecospacegrid_f,"-9999.00000000")
dfLandMask = df_dep == 0 #mask is where elev =0


# ################# MAIN LOOP #################


for var in dctVariables:
    var_ecosim_all = []
    print(var)
    varname = dctVariables[var]["varname"]
    is2Dor3D = dctVariables[var]["2Dor3D"]
    depthlev1 = dctVariables[var]["depthlev1"]
    depthlev2 = dctVariables[var]["depthlev2"]
    depthlev_avg = dctVariables[var]["depthlev_avg"]
    dir_in = dctVariables[var]["dir_in"]
    subdir_in = dctVariables[var]["subdir"]
    filename_in = dctVariables[var]["filename"]
    var_mult = dctVariables[var]["multiplier"]
    if is2Dor3D == "3D":
        if depthlev_avg:
            out_code_var = out_code + "{}_{}-{}mAvg".format(var, depthlev1, depthlev2)
        else:
            out_code_var = out_code + "{}_{}-{}m".format(var, depthlev1, depthlev2)
    else:
        out_code_var = out_code + "{}".format(var)
    dir_out = dctVariables[var]["dir_out"]
    ecospace_subdir_out = dctVariables[var]["ecospace_subdir_out"]
    save_ecospace = dctVariables[var]["save_ecospace"]
    ecosim_subdir_out = dctVariables[var]["ecosim_subdir_out"]
    save_ecosim = dctVariables[var]["save_ecosim"]
    sigdigfmt = dctVariables[var]["sigdigfmt"]
  
    print(varname)
    i = 0
    for iyear in range(startyear, endyear+1):

        print(iyear)

        var_12mo = []

        # read and create stack of all months of data
        for imon in range(startMonth,13):
            print(imon)

            varfullpath = os.path.join(dir_in, subdir_in, filename_in.format(NEMO_run, iyear, buildSortableString(imon,2))) 
            ds_var = nc.Dataset(varfullpath)
            vars_dat = ds_var.variables[varname]
            
            if is2Dor3D == "3D":
                if depthlev_avg == True:
                    # var names are 'day' but really are 3day
                    VarMonth1 = vars_dat[:,depthlev1:depthlev2,:,:]
		    # not correct - should be accounting for variable bin widths here
                    VarMonth = np.ma.average(VarMonth1, axis = 1) # avg over depths - will reduce to (12,299,132)
                    # replace the dim to match if just one level used
                    #SalinityMonth = np.expand_dims(SalinityMonth2, axis=1)    
                else:
                    VarMonth = vars_dat[:,depthlev2,:,:]
            elif is2Dor3D == "2D":
                VarMonth = vars_dat[:,:,:]
            else:
                print("Check 2D or 3D flag in dict.")
                break

            if imon == startMonth:
                var_12mo = VarMonth
            else:
                var_12mo = np.concatenate((var_12mo, VarMonth), axis=0)

            if (imon == endMonth) and (iyear == endyear):
                break

            # why?
            if imon == 12:
                break
                
      # ############### LOOP 3DAY BLOCKS #################
        leapTF = is_leap_year(iyear)
        num_days = 365 
        if leapTF:
            num_days = 366
                   
        num_days = num_days // 3 # 3 day 'years'     
        if not leapTF:
            num_days += 1 # this helps catch day 364 and 365
        
        j = 0 # for file naming - the hacked 3day year
        for iday in range(1,num_days+1):
            
            #default block of 3 days    
            day_strt = (iday-1)+(j*2)
            day_end = day_strt+2
            middle_day = day_strt+2
            
            # exceptions for last blocks of days of year -->
            # lump final 121th and 122th with 120th block?
            if (lump_final) and (iday == 120):

                if not leapTF:
                    #day_end = day_strt+5
                    middle_day = day_strt+3
                else:
                    #day_end = day_strt+6
                    middle_day = day_strt+4
                varday1 = var_12mo[day_strt:,:,:]
                   
            else:            
                # catch if last block of days (364,365) is only 2 long
                if not leapTF:
                    if iday == num_days+1:
                        day_end = day_strt+1
                        middle_day = day_strt+1                     
                varday1 = var_12mo[day_strt:day_end,:,:]
                
            varday = np.ma.average(varday1, axis = 0)
            varday = varday * var_mult
            ar_inf = np.where(np.isinf(varday)) # unused?

 
            # save to ECOSPACE forcing file
            if save_ecospace:
                savepath1 = os.path.join(dir_out, ecospace_subdir_out)
                if (os.path.isdir(savepath1) != True):
                    os.mkdir(savepath1)
                savepath = os.path.join(dir_out, ecospace_subdir_out, out_code_var) 
                if (os.path.isdir(savepath) != True):
                    os.mkdir(savepath)
                savefn = saveTemplate.format(out_code_var,iyear,buildSortableString(middle_day,2))
                savefullpath = os.path.join(savepath, savefn)
                print("Saving file " + savefullpath)                
                saveASCFile(savefullpath, varday, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe, sigdigfmt, ASCheader, dfPlumeMask, dfLandMask)
            
            if save_ecosim:
                # ############# ECOSIM CSV Forcing files #############
                # ecosim forcing grid requires monthly TS for long-term forcing
                # so here we avg across map and then expand, inserting 
                
                var_ecosim = varday.flatten()
                var_ecosim = var_ecosim * var_mult
                var_ecosim = var_ecosim[var_ecosim != 0] # drop zeros for averaging
                var_ecosim = np.mean(var_ecosim)
                # Extract the rounding decimal places from format string
                match = re.search(r'\.(\d+)f', sigdigfmt)
                if match:
                    precision = int(match.group(1))
                    var_ecosim = np.round(var_ecosim,precision)
                else:
                    print("Issue with finding sig digits to round for ecosim out.")
                var_ecosim_all.append([iyear,middle_day,var_ecosim])
            
            if lump_final and (iday == 120):
                break
              
            j+=1
        i+=1    
    if save_ecosim:                    
        
        # add index corresponding to hacked 3day year
        var_ecosim_all_idx = [[index + 1] + sublist for index, sublist in enumerate(var_ecosim_all)]
        # expand the array since ecosim wants values per month not year in enviro forcing grid
        var_ecosim_all_expnd = [sublist for sublist in var_ecosim_all_idx for _ in range(12)]
        # index by time step (fake 3day 'months' = 6hr)
        var_ecosim_all_expnd_idx = [[i + 1] + row for i, row in enumerate(var_ecosim_all_expnd)]
        
        column_names = ['threeday_yrmo','threeday_yr','year', 'dayofyear', out_code_var]

        savepath1 = os.path.join(dir_out, ecosim_subdir_out)
        if (os.path.isdir(savepath1) != True):
            os.mkdir(savepath1)
        savepath = os.path.join(dir_out, ecosim_subdir_out, out_code_var)
        if (os.path.isdir(savepath) != True):
            os.mkdir(savepath)
        savefn = out_code_var + "_" + timestep + "_" + str(startyear) + "-" + str(endyear) + ".csv"
        savefullpath = os.path.join(savepath, savefn)
        print("Saving Ecosim file " + savefullpath)

        with open(savefullpath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            writer.writerows(var_ecosim_all_expnd_idx)


# dctVariables = {"salinity":{"varname": "vosaline",
                            # "depthlev":10, 
                            # "depthlev_avg":True, 
                            # "subdir":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                            # "filename":"SalishSea1500-RUN{}/CDF".format(NEMO_run),
                            # "multiplier":1}
                            # }








# for iyear in range(startYear, endyear+1):

    # print(iyear)
    # #YearSubDirTemplate = str(iyear) # unused -GO

    # #LIGHT data (hrly data in annual, or monthly in annual)
    # lightfullpath = os.path.join(greigDir,lightSubDir,lightFileName.format(iyear))
    # dsLight = nc.Dataset(lightfullpath)
    # varsLight = dsLight.variables[varNameLight]

    # varsSalinity_12mo = []
    # varsTemperature_12mo = []
    # varsMixing_12mo = []
    # varsAdvecX_12mo = []
    # varsAdvecY_12mo = []

    # # read and create stack of all months of data
    # for imon in range(startMonth,13):
        # print(imon)
        
        # #SALINITY data
        # if process_salt:
            # salinityfullpath = os.path.join(michaelDir, salinitySubDir, salinityFileName.format(NEMO_run, iyear, buildSortableString(imon,2))) 
            # dsSalinity = nc.Dataset(salinityfullpath)
            # varsSalinity = dsSalinity.variables[varNameSalinity]
            
            # if avg_salt == True:
                # # var names are 'day' but really are 3day
                # SalinityMonth1 = varsSalinity[:,0:salinity_depthlev,:,:]
                # SalinityMonth = np.ma.average(SalinityMonth1, axis = 1) # avg over depths - will reduce to (12,299,132)
                # # replace the dim to match if just one level used
                # #SalinityMonth = np.expand_dims(SalinityMonth2, axis=1)    
                # SalinityMonth1 = []
            # else:
                # SalinityMonth = varsSalinity[:,salinity_depthlev,:,:]

            # if imon == startMonth:
                # varsSalinity_12mo = SalinityMonth
            # else:
                # varsSalinity_12mo = np.concatenate((varsSalinity_12mo, SalinityMonth), axis=0)
        
        # #TEMPERATURE data
        # if process_temp:
            # temperaturefullpath = os.path.join(michaelDir, temperatureSubDir, temperatureFileName.format(NEMO_run, iyear, buildSortableString(imon,2))) 
            # dsTemperature = nc.Dataset(temperaturefullpath)
            # varsTemperature = dsTemperature.variables[varNameTemp]
            
            
            
            # if imon == startMonth:
                # varsTemperature_12mo = varsTemperature
            # else:
                # varsTemperature_12mo = np.concatenate((varsTemperature_12mo, varsTemperature), axis=0)
        
        
        # #MIXING data 
        # if process_mixing:
            # mixingFullPath = os.path.join(michaelDir, mixingSubDir, mixingFileName.format(NEMO_run, iyear, buildSortableString(imon,2)))
            # dsMixing = nc.Dataset(mixingFullPath)
            # varsMixing = dsMixing.variables['mldkz5']
            
            
            
            # if imon == startMonth:
                # varsMixing_12mo = varsMixing
            # else:
                # varsMixing_12mo = np.concatenate((varsMixing_12mo, varsMixing), axis=0)
        
        # #ADVEC data
        # if process_advec:
            # advecXFullPath = os.path.join(michaelDir, advecSubDir, advectXFileName.format(NEMO_run, iyear, buildSortableString(imon,2)))
            # advecYFullPath = os.path.join(michaelDir, advecSubDir, advectYFileName.format(NEMO_run, iyear, buildSortableString(imon,2)))
            # dsAdvecX = nc.Dataset(advecXFullPath)
            # dsAdvecY = nc.Dataset(advecYFullPath)
            # varsAdvecX = dsAdvecX.variables[varNameXadvect]
            # varsAdvecY = dsAdvecY.variables[varNameYadvect]
            
            
            
            # if imon == startMonth:
                # varsAdvecX_12mo = varsAdvecX
                # varsAdvecY_12mo = varsAdvecY
            # else:
                # varsAdvecX_12mo = np.concatenate((varsAdvecX_12mo, varsAdvecX), axis=0)
                # varsAdvecY_12mo = np.concatenate((varsAdvecY_12mo, varsAdvecY), axis=0)

        
        # if (imon == endMonth) and (iyear == endyear):
            # break

        # # why?
        # if imon == 12:
            # break



  
    # # ############# ECOSPACE ASC Forcing files #############
    # leapTF = is_leap_year(iyear)
    # if leapTF:
        # num_days = 366
    # else:
        # num_days = 365
    
    # num_days = num_days // 3 # 3 day 'years'
    # print("number of 'years' per year")
    # print(num_days)
    
    # if not leapTF:
        # num_days += 1 # this helps catch day 364 and 365
    
    # j = 0 # for file naming
    # for iday in range(1,num_days+1):

        # day_strt = (iday-1)+(j*2)
        # day_end = day_strt+2
        # middle_day = day_strt+2
        
        # # catch if last block of days (364,365) is only 2 long
        # if not leapTF:
            # if iday == num_days+1:
                # day_end = day_strt+1
                # middle_day = day_strt+1
            
        # # #############  LIGHT  #############
        # # not done
        # #LightDay = varsLight[day_strt:day_end,:,:]  
        

        # # #############  SALINITY  #############
        # if process_salt:
            
                     
            # SalinityDay1 = varsSalinity_12mo[day_strt:day_end,:,:]
            # SalinityDay = np.ma.average(SalinityDay1, axis = 0)
            # ar_inf = np.where(np.isinf(SalinityDay)) # unused?

            
            # # save to ECOSPACE forcing file
            # if save_ecospace:
                # savepath = os.path.join(greigDir,outputsSubDir,out_code_salt) 
                # if (os.path.isdir(savepath) != True):
                    # os.mkdir(savepath)
                # savefn = saveTemplate.format(out_code_salt,iyear,buildSortableString(middle_day,2))
                # savefullpath = os.path.join(savepath, savefn)
                # print("Saving file " + savefullpath)
                # sigdigfmt = '%0.1f'
                # saveASCFile(savefullpath, SalinityDay, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
            
            # if save_ecosim:
                # # ############# ECOSIM CSV Forcing files #############
                # # ecosim forcing grid requires monthly TS for long-term forcing
                # # so here we avg across map and then expand, inserting 
                
                # varsSalinity_sim = SalinityDay.flatten()
                # varsSalinity_sim = varsSalinity_sim[varsSalinity_sim != 0] # drop zeros for averaging
                # varsSalinity_sim = np.mean(varsSalinity_sim)
                # ecosim_all_salt.append(varsSalinity_sim)
              
                # #else:
                # #    ecosim_all_salt = ecosim_all_salt + varsSalinity_sim
                
                # # varsTemperature_sim = np.ma.average(varsTemperature_12mo, axis = 1)
                # # varsTemperature_sim = np.ma.average(varsTemperature_sim, axis = 1)
                
                # # varsMixing_sim = np.ma.average(varsMixing_12mo, axis = 1)
                # # varsMixing_sim = np.ma.average(varsMixing_sim, axis = 1)
                
                # # varsAdvecX_sim = np.ma.average(varsAdvecX_12mo, axis = 1)
                # # varsAdvecX_sim = np.ma.average(varsAdvecX_sim, axis = 1)
                
                # # varsAdvecY_sim = np.ma.average(varsAdvecY_12mo, axis = 1)
                # # varsAdvecY_sim = np.ma.average(varsAdvecY_sim, axis = 1)
    
    
        # # #############  TEMPERATURE  #############
        # if process_temp:
            # sigdigfmt = '%0.1f'
            # if avg_temp == True:
              # TemperDay1 = varsTemperature_12mo[day_strt:day_end,0:temperature_depthlev,:,:]
              # TemperDay2 = np.ma.average(TemperDay1, axis = 1)
              # TemperDay = np.ma.average(TemperDay2, axis = 0)
            # else:
              # TemperDay1 = varsTemperature_12mo[day_strt:day_end,temperature_depthlev,:,:]
              # TemperDay = np.ma.average(TemperDay1, axis = 0)
            # ar_inf = np.where(np.isinf(TemperDay)) # unused?
            
            # savepath = os.path.join(greigDir,outputsSubDir,out_code_temp) 
            # if (os.path.isdir(savepath) != True):
                # os.mkdir(savepath)
            # savefn = saveTemplate.format(out_code_temp,iyear,buildSortableString(middle_day,2))
            # savefullpath = os.path.join(savepath, savefn)
            # print("Saving file " + savefullpath)
            # saveASCFile(savefullpath, TemperDay, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
    
    
        # # #############  MIXING  #############
        # if process_mixing:
            # MixingDay1 = varsMixing_12mo[day_strt:day_end,:,:]
            # MixingDay = np.ma.average(MixingDay1, axis = 0)
            # savepath = os.path.join(greigDir,outputsSubDir,out_code_mixing) 
            # if (os.path.isdir(savepath) != True):
                # os.mkdir(savepath)
            # savefn = saveTemplate.format(out_code_mixing,iyear,buildSortableString(middle_day,2))
            # savefullpath = os.path.join(savepath, savefn)
            # print("Saving file " + savefullpath)
            # sigdigfmt = '%0.1f'
            # saveASCFile(savefullpath, MixingDay, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
    
    
        # # #############  ADVECTION  #############
        # if process_advec:
            # sigdigfmt = '%0.4f'
            # if avg_advec == True:
              # AdvecXDay1 = varsAdvecX_12mo[day_strt:day_end,0:advec_depthlev,:,:]         
              # AdvecYDay1 = varsAdvecY_12mo[day_strt:day_end,0:advec_depthlev,:,:]
              # AdvecXDay2 = np.ma.average(AdvecXDay1, axis = 1)
              # AdvecYDay2 = np.ma.average(AdvecYDay1, axis = 1)
              # AdvecXDay = np.ma.average(AdvecXDay2, axis = 0)
              # AdvecYDay = np.ma.average(AdvecYDay2, axis = 0)
            # else:
              # AdvecXDay1 = varsAdvecX_12mo[day_strt:day_end,advec_depthlev,:,:]
              # AdvecYDay1 = varsAdvecY_12mo[day_strt:day_end,advec_depthlev,:,:]
              # AdvecXDay = np.ma.average(AdvecXDay1, axis = 0)
              # AdvecYDay = np.ma.average(AdvecYDay1, axis = 0)
            # AdvecXDay = AdvecXDay * advec_mult # convert from m/s to cm/s, and s/365 for daily hack
            # AdvecYDay = AdvecYDay * advec_mult
            # ar_inf = np.where(np.isinf(AdvecXDay)) # unused?

            # # X advec
            # savepath = os.path.join(greigDir,outputsSubDir,out_code_advecX) 
            # if (os.path.isdir(savepath) != True):
                # os.mkdir(savepath)
            # savefn = saveTemplate.format(out_code_advecX,iyear,buildSortableString(middle_day,2))
            # savefullpath = os.path.join(savepath, savefn)
            # print("Saving advec file " + savefullpath)
            # saveASCFile(savefullpath, AdvecXDay, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
            
            # # Y advec
            # savepath = os.path.join(greigDir,outputsSubDir,out_code_advecY) 
            # if (os.path.isdir(savepath) != True):
                # os.mkdir(savepath)
            # savefn = saveTemplate.format(out_code_advecY,iyear,buildSortableString(middle_day,2))
            # savefullpath = os.path.join(savepath, savefn)
            # print("Saving advec file " + savefullpath)
            # saveASCFile(savefullpath, AdvecYDay, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
    
        # j+=1    

# print(ecosim_all_salt)

    
       
        # # is leap year?
        # if imon == 2 and is_leap_year(iyear):
            # num_days = 29
        # else:
            # num_days = 28 if imon == 2 else 30 if imon in [4, 6, 9, 11] else 31
        
        # num_days = num_days // 3
        # print(num_days)
        
        # j = 0 # 
        # for iday in range(1,num_days+1):
        
            # day_strt = (iday-1)+(j*2)
            # day_end = day_strt+2
        
            # middle_day = day_strt+2 # for file name (middle mnth-day of 3 day block)
        
            # # #############  LIGHT  #############
            # # not done
            # #LightDay = varsLight[day_strt:day_end,:,:]  
            
 
            # # #############  SALINITY  #############
            # if process_salt:
                # sigdigfmt = '%0.1f'
                # if avg_salt == True:
                  # SalinityDay1 = varsSalinity[day_strt:day_end,0:salinity_depthlev,:,:]
                  # #print(SalinityDay1.shape) # #print(SalinityDay1.shape) # will be (2,10,299,132)
                  # SalinityDay2 = np.ma.average(SalinityDay1, axis = 1) # avg over depths - will reduce to (2,299,132)
                  # SalinityDay = np.ma.average(SalinityDay2, axis = 0) # avg over time - reduce to (299,132)
                # else:          
                  # SalinityDay1 = varsSalinity[day_strt:day_end,salinity_depthlev,:,:]
                  # SalinityDay = np.ma.average(SalinityDay1, axis = 0)
                # ar_inf = np.where(np.isinf(SalinityDay)) # unused?
                
                # savepath = os.path.join(greigDir,outputsSubDir,out_code_salt) 
                # if (os.path.isdir(savepath) != True):
                    # os.mkdir(savepath)
                # savefn = saveTemplate.format(out_code_salt,iyear,buildSortableString(imon,2),buildSortableString(middle_day,2))
                # savefullpath = os.path.join(savepath, savefn)
                # print("Saving file " + savefullpath)
                # saveASCFile(savefullpath, SalinityDay, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
        
        
            # # #############  TEMPERATURE  #############
            # if process_temp:
                # sigdigfmt = '%0.1f'
                # if avg_temp == True:
                  # TemperDay1 = varsTemperature[day_strt:day_end,0:temperature_depthlev,:,:]
                  # TemperDay2 = np.ma.average(TemperDay1, axis = 1)
                  # TemperDay = np.ma.average(TemperDay2, axis = 0)
                # else:
                  # TemperDay1 = varsTemperature[day_strt:day_end,temperature_depthlev,:,:]
                  # TemperDay = np.ma.average(TemperDay1, axis = 0)
                # ar_inf = np.where(np.isinf(TemperDay)) # unused?
                
                # savepath = os.path.join(greigDir,outputsSubDir,out_code_temp) 
                # if (os.path.isdir(savepath) != True):
                    # os.mkdir(savepath)
                # savefn = saveTemplate.format(out_code_temp,iyear,buildSortableString(imon,2),buildSortableString(middle_day,2))
                # savefullpath = os.path.join(savepath, savefn)
                # print("Saving file " + savefullpath)
                # saveASCFile(savefullpath, TemperDay, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
        
        
            # # #############  MIXING  #############
            # if process_mixing:
                # sigdigfmt = '%0.1f'
                # MixingDay1 = varsMixing[day_strt:day_end,:,:]
                # MixingDay = np.ma.average(MixingDay1, axis = 0)
                # savepath = os.path.join(greigDir,outputsSubDir,out_code_mixing) 
                # if (os.path.isdir(savepath) != True):
                    # os.mkdir(savepath)
                # savefn = saveTemplate.format(out_code_mixing,iyear,buildSortableString(imon,2),buildSortableString(middle_day,2))
                # savefullpath = os.path.join(savepath, savefn)
                # print("Saving file " + savefullpath)
                # saveASCFile(savefullpath, MixingDay, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
        
        
            # # #############  ADVECTION  #############
            # if process_advec:
                # sigdigfmt = '%0.4f'
                # if avg_advec == True:
                  # AdvecXDay1 = varsAdvecX[day_strt:day_end,0:advec_depthlev,:,:]         
                  # AdvecYDay1 = varsAdvecY[day_strt:day_end,0:advec_depthlev,:,:]
                  # AdvecXDay2 = np.ma.average(AdvecXDay1, axis = 1)
                  # AdvecYDay2 = np.ma.average(AdvecYDay1, axis = 1)
                  # AdvecXDay = np.ma.average(AdvecXDay2, axis = 0)
                  # AdvecYDay = np.ma.average(AdvecYDay2, axis = 0)
                # else:
                  # AdvecXDay1 = varsAdvecX[day_strt:day_end,advec_depthlev,:,:]
                  # AdvecYDay1 = varsAdvecY[day_strt:day_end,advec_depthlev,:,:]
                  # AdvecXDay = np.ma.average(AdvecXDay1, axis = 0)
                  # AdvecYDay = np.ma.average(AdvecYDay1, axis = 0)
                # AdvecXDay = AdvecXDay * advec_mult # convert from m/s to cm/s, and s/365 for daily hack
                # AdvecYDay = AdvecYDay * advec_mult
                # ar_inf = np.where(np.isinf(AdvecXDay)) # unused?

                # # X advec
                # savepath = os.path.join(greigDir,outputsSubDir,out_code_advecX) 
                # if (os.path.isdir(savepath) != True):
                    # os.mkdir(savepath)
                # savefn = saveTemplate.format(out_code_advecX,iyear,buildSortableString(imon,2),buildSortableString(middle_day,2))
                # savefullpath = os.path.join(savepath, savefn)
                # print("Saving advec file " + savefullpath)
                # saveASCFile(savefullpath, AdvecXDay, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
                
                # # Y advec
                # savepath = os.path.join(greigDir,outputsSubDir,out_code_advecY) 
                # if (os.path.isdir(savepath) != True):
                    # os.mkdir(savepath)
                # savefn = saveTemplate.format(out_code_advecY,iyear,buildSortableString(imon,2),buildSortableString(middle_day,2))
                # savefullpath = os.path.join(savepath, savefn)
                # print("Saving advec file " + savefullpath)
                # saveASCFile(savefullpath, AdvecYDay, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
        
            # j+=1    
            

