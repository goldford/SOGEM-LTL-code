import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
from calendar import monthrange
import math
from GO_helpers import is_leap_year, buildSortableString, saveASCFile, getDataFrame 

#import matplotlib.pyplot as plt
# module load StdEnv/2020
# module load scipy-stack/2020a

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# created by GO, last edited by GO 2024-05-23

# purpose: prep files for ECOSPACE - use clima as initial conditions
#          input: monthly light in annual files from RDRS climate model, atten coef (K) from mixing depth Z and salin at chosen depth
#          output: computes integrated PAR across water depths, u(I) (growth response) following Merico et al (2004), Steele
#          PAR = photosynthetically active radiation
#     Method codes:
#         - PAR0 - fixed mixed depth, fixed K value
#         - PAR1a - variable mixing depth, fixed K value
#         - PAR1b - variable mixed depth, fixed K value
#         - PAR2a - fixed mix depth, linear salinity->turbidity to modify K by cell, function using salinity vertmean 0-10 m
#         - PAR2b - fixed mix depth, linear salinity->turbidity f(n) using salinity at one depth
#     --> - PAR3 -  variable mix depth (turbocline threshold), turbidity from 2b, salin at one depth or avg depth
#         - PAR3b - no longer using 

#Data in:

# 1/ Mixing Layer data (var mldkz5 / turbocline depth) as daily means in annual NC files, DIRECT from NEMO outputs "xxx_1d_grid_T_2D_1980.nc"
# 2/ Salinity - can be either vertical averages over chosen levels or from a specific depth level, DIRECT from NEMO outputs "xxx_1d_grid_T_selectedlevs_1980.nc"
# 3/ Light as daily means from climate re-analysis (e.g. ERA5) with near-surface downward radiative flux (var msdwswrf Wm-2) interpolated to NEMO grid as annual NC files# 
# file name e.g. "ERA5_NEMOgrid_light_daily_1980.nc".

# Log:
# - GO-2021-12-22 edits to draw from monthly mean files rather than looping through daily NC data 
# - GO2021-12-23 differentiated 1a from 1b as vertmean of salinity (top x layers) and salinity at x m, respectively
#                Added write of K values as ASC for eval
# - GO 2023-03-23 edits to code using PAR2b script (need to re-run PAR3)
# - GO 2023-04-05 moved functions to GO_helpers.py
# - GO 2023-04-06 found that PAR predicted seemed not to match 2015, 2016 in Pawlowicz
# - GO 2023-05-10 added option to switch to over-depth average rather than just selecting a single dpeth
# - GO 2023-05-12 adding calc for u(I) (growth response curve) based on Merico et al (2004), Steele
# - GO 2024-05-17 modifications to read and write to daily instead of monthly means
# - GO 2024-05-21 edits to write out daily files for non-depth integrated salinity (e.g., just 4 m)
# - GO 2024-05-23 mod for 3-day means
# - GO 2024-05-23 fork to this for annual clima
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# //////////////////////////////////// FUNCTIONS //////////////////////////////////////
# now in GO_helpers.py -2023-04-05


# ////////////////////////////////// Basic Params ///////////////////////////////////////
startYear = 1980
endyear = 2018
startMonth = 1
endMonth = 12
NEMO_run = "216" #NEMO run code
PAR_code = "PAR3" #PAR prep code

# declare if single level or pre-calculated mean across depths
# ////////////////////////////////// PAR Ir PARAMS ///////////////////////////////////////
# irradiance PAR across depths
salinity_depthlev = 4
FixedMixingDepth = 10
b = -0.063 
#a = 0.08529 
a = 1.94

avg_salt = False
if avg_salt == True: 
  PAR_code = PAR_code + "_Sal{}mAvg".format(salinity_depthlev)
else:
  PAR_code = PAR_code + "_Sal{}m".format(salinity_depthlev)


#Variables use for Light attenuation
pPAR = 0.44  #proportion of irrad useful for phyto
alb = 0.067  #albedo reflectance
K = 0.05     #baseline K (clear water)

# not completed
# ////////////////////////////////// Growth Response params ///////////////////////////////////////
# eqn from Merico et al 2004 Appendix A pp. 1822
# Light saturation constants W/m-2
I_sd = 15  #diatoms
I_sf = 15  #flag
I_sdf = 15 #dinoflag
I_sc = 45  #coccolith

# relative max growth rates
#u_max_d = 
#u_max_ 

# ///////////////////////////////////// ASC Params ////////////////////////////////////////
ASCheader = "ncols        93\nnrows        151 \nxllcorner    -123.80739\nyllcorner    48.29471 \ncellsize     0.013497347 \nNODATA_value  0.0"


#////////////////// /////////////////// PATHS /////////////////////////////////////////////
greigDir = "/project/6006412/goldford/ECOSPACE/"
michaelDir = "/project/6006412/mdunphy/nemo_results/SalishSea1500/"
lightSubDir = "DATA/light_monthly/"
plumeSubDir = "DATA/"
depthSubDir = "DATA/"

# pre processed by G
mixingSubDir = "DATA/SS1500-RUN{}/NEMO_annual_NC".format(NEMO_run)
salinitySubDir = "DATA/SS1500-RUN{}/NEMO_annual_NC".format(NEMO_run)
outputsSubDir ="DATA/SS1500-RUN{}/ECOSPACE_in_climayr_{}-{}_{}".format(NEMO_run, startYear, endyear, PAR_code)

# (for daily)
#mixingSubDir = "SalishSea1500-RUN{}/CDF".format(NEMO_run)
#salinitySubDir = "SalishSea1500-RUN{}/CDF".format(NEMO_run)
#outputsSubDir ="DATA/SS1500-RUN{}/ECOSPACE_in_annualclima_1980-2018{}".format(NEMO_run, PAR_code)

if (os.path.isdir(os.path.join(greigDir,outputsSubDir)) != True):
  os.mkdir(os.path.join(greigDir,outputsSubDir))

# preprocessed monthlies in yearly files
mixingFileName = 'SalishSea1500-RUN{}_MonthlyMean_grid_T_2D_y{}.nc'
lightFileName = "RDRS21_NEMOgrid_light_monthly_{}.nc"  
salinityFileName ='SalishSea1500-RUN{}_MonthlyMean_grid_T_y{}.nc' 
#salinityFileName ='SalishSea1500-RUN{}_MonthlyMean_grid_T_singledepth10m_{}.nc' #depth=9.5m
#salinityFileName ='SalishSea1500-RUN{}_MonthlyMean_grid_T_avgtop10m_y{}.nc'

# (for daily)
#mixingFileName = 'SalishSea1500-RUN{}_1d_grid_T_2D_y{}m{}.nc'
#lightFileName = "RDRS21_NEMOgrid_light_daily_{}.nc"
#salinityFileName ='SalishSea1500-RUN{}_1d_grid_T_y{}m{}.nc' 


# ///////////////////// VARS ////////////////////////
varNameLight = "solar_rad"
varNameSalinity = 'vosaline'
varNamesMixing = 'mldkz5'
lstVars = ['mldkz5']; 

#Dictionary to convert NEMO names to EwE names
#Used for both Directory name and in the file name
dctVarnameToEwEName = {varNamesMixing:'PAR-VarZ-VarK'}

saveTemplate = "{}_{}-{}.asc" # new ewename, startYear, endYear


#Number of depth strata to use
#Depth indexes are zero based
#So 10 will be the 9 index
#nDepthIndexes = 10 #= 9.001736   10.003407
#nDepthIndexes = 23 #= 31.101034   39.11886
#nDepthIndexes = 26 #= 86.96747   109.73707 


# ///////////////// MAP CLIP ///////////////////////
#array offsets for clipping EwE grid out of NEMO grid
#Calculated by Greig in "Basemap Converter (Py3).ipynb"
upperleft_row_ewe = 253
upperleft_col_ewe = 39
bottomleft_row_ewe = 102
bottomleft_col_ewe = 39


# ///////////////// PLUME MASK /////////////////////
#Create the plume mask
PlumeRegionFilename = os.path.join(greigDir, plumeSubDir, "Fraser_Plume_Region.asc")
dfPlumeRegion = getDataFrame(PlumeRegionFilename,"-9999.00000000")
dfPlumeMask = dfPlumeRegion == 1 #Plume is region one


# ///////////////// LAND MASK /////////////////////
ecospacegrid_f = os.path.join(greigDir, depthSubDir, "ecospacedepthgrid.asc")
df_dep = getDataFrame(ecospacegrid_f,"-9999.00000000")
dfLandMask = df_dep == 0 #mask is where elev =0


# ///////////////// MAIN LOOP /////////////////////
for var in lstVars:
    light_allyrs = []
    mixing_allyrs = []
    salt_allyrs = []
    
    for iyear in range(startYear, endyear+1):
        print(iyear)
        #YearSubDirTemplate = str(iyear) # unused -GO

        #LIGHT data (hrly data in annual, or monthly in annual)
        lightfullpath = os.path.join(greigDir,lightSubDir,lightFileName.format(iyear))
        dsLight = nc.Dataset(lightfullpath)
        varsLight = dsLight.variables[varNameLight]
        LightYear1 = varsLight[:,:,:]
        LightYear = np.ma.average(LightYear1, axis = 0)
        LightYear = np.expand_dims(LightYear, axis=0)
        
        #SALINITY data (3D)
        salinityfullpath = os.path.join(greigDir,salinitySubDir,salinityFileName.format(NEMO_run,iyear)) # old (preprocessed)
        dsSalinity = nc.Dataset(salinityfullpath)
        varsSalinity = dsSalinity.variables[varNameSalinity]
        if avg_salt == True:
            SaltYear1 = varsSalinity[:,0:salinity_depthlev,:,:]
            SaltYear2 = np.ma.average(SaltYear1, axis = 1) # avg over depths - will reduce to (2,299,132)
            SaltYear = np.ma.average(SaltYear2, axis = 0) # avg over time - reduce to (299,132)
            SaltYear = np.expand_dims(SaltYear, axis=0)
        else:
            SaltYear1 = varsSalinity[:,salinity_depthlev,:,:]
            SaltYear = np.ma.average(SaltYear1, axis = 0)
        SaltYear = np.expand_dims(SaltYear, axis=0)
        ar_inf = np.where(np.isinf(SaltYear)) # unused?        
                
        #MIXING data 
        mixingFullPath = os.path.join(greigDir, mixingSubDir, mixingFileName.format(NEMO_run, iyear))
        dsMixing = nc.Dataset(mixingFullPath)
        varsMixing = dsMixing.variables['mldkz5']   
        MixingYear1 = varsMixing[:,:,:]
        MixingYear = np.ma.average(MixingYear1, axis = 0) 
        MixingYear = np.expand_dims(MixingYear, axis=0)        

        if iyear == startYear:
            light_allyrs = LightYear
            mixing_allyrs = MixingYear
            salt_allyrs = SaltYear
        else:
            light_allyrs = np.concatenate((light_allyrs, LightYear), axis=0)
            mixing_allyrs = np.concatenate((mixing_allyrs, MixingYear), axis=0)
            salt_allyrs = np.concatenate((salt_allyrs, SaltYear), axis=0)  


    light_allyrs = np.ma.average(light_allyrs, axis = 0)
    salt_allyrs = np.ma.average(salt_allyrs, axis = 0)
    mixing_allyrs = np.ma.average(mixing_allyrs, axis = 0)
 
   
    # unused code? 
    #x = lambda a : a + 10
    #SalinityMon2 = x(SalinityMon)
    #print(SalinityMon2[1])

    # apply lin regress k w/ salin
    Ksal = a + b * salt_allyrs
    Ksal[Ksal < 0.05] = 0.05

    #mixingZ = FixedMixingDepth

    #Ksal = KLinear(salinity) #Klinear RUN 102a
    #uPAR = L0*(1-math.exp(-Ksal*MixingDepth))/(Ksal*MixingDepth) #vertical avg PAR to mixing lyr depth
    step1 = np.multiply(-Ksal,mixing_allyrs)
    step2 = 1-np.exp(step1)
    step3 = np.multiply(Ksal,mixing_allyrs)
    step4 = np.divide(step2,step3)

    L0 = light_allyrs * pPAR * (1-alb)
    uPAR = np.multiply(L0,step4) #vertical avg PAR to mixing lyr depth
    uPAR[salt_allyrs == 0.0] = 0.0
    Ksal[salt_allyrs == 0.0] = 0.0
    uPAR = np.round(uPAR, 2)
    Ksal = np.round(Ksal, 3)    
       
    # #export PAR
    sigdigfmt = '%0.1f'
    EwEName = dctVarnameToEwEName[varNamesMixing]
    EwEName = EwEName.format(salinity_depthlev) #GO ??
    savepath = os.path.join(greigDir,outputsSubDir,EwEName ) 
    if (os.path.isdir(savepath) != True):
        os.mkdir(savepath)
    savefn = saveTemplate.format(EwEName, startYear, endyear)
    savefullpath = os.path.join(savepath, savefn)
    print("Saving file " + savefullpath)
    saveASCFile(savefullpath, uPAR, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe, sigdigfmt, ASCheader, dfPlumeMask, dfLandMask)

    #export K
    sigdigfmt = '%0.2f'
    EwEName = "RUN" + NEMO_run + "_Kfromsal"
    #EwEName = "RUN102b-Kfromsal_singlelev10m"
    #EwEName = "RUN102a-Kfromsal_vertmeantop10m" #GO alternative
    #EwEName = EwEName.format(salinity_depthlev) #GO ??
    savepath = os.path.join(greigDir,outputsSubDir,EwEName ) 
    if (os.path.isdir(savepath) != True):
        os.mkdir(savepath)
    savefn = saveTemplate.format(EwEName,startYear, endyear)
    savefullpath = os.path.join(savepath, savefn)
    print("Saving file " + savefullpath)
    saveASCFile(savefullpath, Ksal,bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
        
   
