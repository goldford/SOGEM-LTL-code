import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pandas as pd
import csv
import os
from calendar import monthrange
import math
from GO_helpers import is_leap_year, buildSortableString, saveASCFile, getDataFrame 

#import matplotlib.pyplot as plt

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# created by GO, last edited by GO 2024-05-23

# purpose: prep files for ECOSPACE
#          input: 3-daily mean light from RDRS climate model, atten coef (K) from mixing depth Z and salin at chosen depth
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
# - GO 2024-05-27 option to lump final 3-day groups into one so not to exceed 120 blocks per year
# - GO 2025-05-28 added another layer - multiplication of depth-int average PAR and mixing (to get 'volume hab')
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# //////////////////////////////////// FUNCTIONS //////////////////////////////////////
# now in GO_helpers.py -2023-04-05


# ////////////////////////////////// Basic Params ///////////////////////////////////////
startyear = 1980
endyear = 1989
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

# export to ecosim TS for enviro forcing?
save_ecosim = True
# export to ecospace asc?
save_ecospace = True
lump_final = True # lump the 121st,122nd into 120th?

avg_salt = False
if avg_salt == True: 
  PAR_code = PAR_code + "_Sal{}mAvg".format(salinity_depthlev)
else:
  PAR_code = PAR_code + "_Sal{}m".format(salinity_depthlev)
time_code = "3day"

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
ecosim_subdir_out = "DATA/SS1500-RUN{}/ECOSIM_in_{}_{}".format(NEMO_run, time_code, PAR_code)
michaelDir = "/project/6006412/mdunphy/nemo_results/SalishSea1500/"
#lightSubDir = "DATA/light_monthly/"
lightSubDir = "DATA/light_daily/"
plumeSubDir = "DATA/"
depthSubDir = "DATA/"
# old
#mixingSubDir = "DATA/SS1500-RUN{}/NEMO_annual_NC".format(NEMO_run)
#salinitySubDir = "DATA/SS1500-RUN{}/NEMO_annual_NC".format(NEMO_run)
#outputsSubDir ="DATA/SS1500-RUN{}/ECOSPACE_in_{}".format(NEMO_run, PAR_code)
# new (for daily)
mixingSubDir = "SalishSea1500-RUN{}/CDF".format(NEMO_run)
salinitySubDir = "SalishSea1500-RUN{}/CDF".format(NEMO_run)
ecospace_subdir_out ="DATA/SS1500-RUN{}/ECOSPACE_in_{}_{}".format(NEMO_run, time_code, PAR_code)

if (os.path.isdir(os.path.join(greigDir,ecospace_subdir_out)) != True):
  os.mkdir(os.path.join(greigDir,ecospace_subdir_out))

# OLD
#mixingFileName = 'SalishSea1500-RUN{}_MonthlyMean_grid_T_2D_y{}.nc'
#lightFileName = "RDRS21_NEMOgrid_light_monthly_{}.nc"  
#salinityFileName ='SalishSea1500-RUN{}_MonthlyMean_grid_T_y{}.nc' 
#salinityFileName ='SalishSea1500-RUN{}_MonthlyMean_grid_T_singledepth10m_{}.nc' #depth=9.5m
#salinityFileName ='SalishSea1500-RUN{}_MonthlyMean_grid_T_avgtop10m_y{}.nc'

# NEW - for daily
mixingFileName = 'SalishSea1500-RUN{}_1d_grid_T_2D_y{}m{}.nc'
lightFileName = "RDRS21_NEMOgrid_light_daily_{}.nc"
salinityFileName ='SalishSea1500-RUN{}_1d_grid_T_y{}m{}.nc' 

# ///////////////////// VARS ////////////////////////
varNameLight = "solar_rad"
varNameSalinity = 'vosaline'
varNamesMixing = 'mldkz5'
lstVars = ['mldkz5']; 

#Dictionary to convert NEMO names to EwE names
#Used for both Directory name and in the file name
dctVarnameToEwEName = {varNamesMixing:'PAR-VarZ-VarK'}

saveTemplate = "{}_{}_{}.asc" # new ewename, year, dayofyear


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
uPAR_ecosim_3day_all = []; Ksal_ecosim_3day_all = []; uPARxMixing_ecosim_3day_all = []
for iyear in range(startyear, endyear+1):
    print(iyear)
    #YearSubDirTemplate = str(iyear) # unused -GO

    #LIGHT data (hrly data in annual, or monthly in annual)
    lightfullpath = os.path.join(greigDir,lightSubDir,lightFileName.format(iyear))
    dsLight = nc.Dataset(lightfullpath)
    varsLight = dsLight.variables[varNameLight]
    
    #SALINITY data
    #salinityfullpath = os.path.join(greigDir,salinitySubDir,salinityFileName.format(NEMO_run,iyear)) # old (preprocessed)
    #salinityfullpath = os.path.join(michaelDir,salinitySubDir,salinityFileName.format(NEMO_run,iyear)) #new
    #dsSalinity = nc.Dataset(salinityfullpath)
    #varsSalinity = dsSalinity.variables[varNameSalinity]
    
    #MIXING data 
    #mixingFullPath = os.path.join(greigDir, mixingSubDir, mixingFileName.format(NEMO_run, iyear))
    #mixingFullPath = os.path.join(michaelDir, mixingSubDir, mixingFileName.format(NEMO_run, iyear))
    #dsMixing = nc.Dataset(mixingFullPath)
    #varsMixing = dsMixing.variables['mldkz5']       
   
    #GO2021-12-23 to-do: fix how shape is declared
    #monthlyMeanPAR = np.empty([varsSalinity.shape[2],varsSalinity.shape[3]])        
    #monthlyMeanK = np.empty([varsSalinity.shape[3],varsSalinity.shape[2]]) 
    #print(monthlyMeanK.shape)
    #print(monthlyMeanPAR.shape)
    
    varsSalinity_12mo = []
    varsMixing_12mo = []

    # for 3 day blocks, need to read all data into memory (better way?)
    for imon in range(startMonth,13):
        print(imon)
        #SALINITY data
        salinityfullpath = os.path.join(michaelDir, salinitySubDir, salinityFileName.format(NEMO_run, iyear, buildSortableString(imon,2))) 
        dsSalinity = nc.Dataset(salinityfullpath)
        varsSalinity = dsSalinity.variables[varNameSalinity]
        
        if avg_salt == True:
          SalinityMo1 = varsSalinity[:,0:salinity_depthlev,:,:]
          SalinityMo = np.ma.average(SalinityMo1, axis = 1) 
        else:
          SalinityMo = varsSalinity[:,salinity_depthlev,:,:]
       
        if imon == startMonth:
            varsSalinity_12mo = SalinityMo
        else:
            varsSalinity_12mo = np.concatenate((varsSalinity_12mo, SalinityMo), axis=0)
        
        
        #MIXING data 
        mixingFullPath = os.path.join(michaelDir, mixingSubDir, mixingFileName.format(NEMO_run, iyear, buildSortableString(imon,2)))
        dsMixing = nc.Dataset(mixingFullPath)
        varsMixing = dsMixing.variables['mldkz5']
        if imon == startMonth:
            varsMixing_12mo = varsMixing
        else:
            varsMixing_12mo = np.concatenate((varsMixing_12mo, varsMixing), axis=0)
        
        
        if (imon == endMonth) and (iyear == endyear):
            break

        # why?
        if imon == 12:
            break
        
        
    # set up 3 day blocks    
    print(varsSalinity_12mo.shape)
    num_days = varsSalinity_12mo.shape[0]
    print(num_days)

    num_days = num_days // 3 # 3 day 'years'

    leapTF = is_leap_year(iyear)
    if not leapTF:
        num_days += 1 # this helps catch day 364 and 365

    j = 0 # for skipping forward
    for iday in range(1,num_days+1):

        day_strt = (iday-1)+(j*2)
        day_end = day_strt+2
        middle_day = day_strt+2
    
        # exceptions for last blocks of days of year -->
        # lump final 121th and 122th with 120th block?
        if (lump_final) and (iday == 120):
            if not leapTF:
                middle_day = day_strt+3
            else:
                middle_day = day_strt+4
            #Data for month
            LightDay1 = varsLight[day_strt:,:,:]
            SalinityDay1 = varsSalinity_12mo[day_strt:,:,:]
            MixingDay1 = varsMixing_12mo[day_strt:,:,:]
    
        else:
            # catch if last block of days (364,365) is only 2 long
            if not leapTF:
                if iday == num_days+1:
                    day_end = day_strt+1
                    middle_day = day_strt+1
        
            #Data for month
            LightDay1 = varsLight[day_strt:day_end,:,:]
            SalinityDay1 = varsSalinity_12mo[day_strt:day_end,:,:]
            MixingDay1 = varsMixing_12mo[day_strt:day_end,:,:]
        
        LightDay = np.ma.average(LightDay1, axis = 0)       
        SalinityDay = np.ma.average(SalinityDay1, axis = 0)
        ar_inf = np.where(np.isinf(SalinityDay)) # unused?
        MixingDay = np.ma.average(MixingDay1, axis = 0)
        
        # unused code? 
        #x = lambda a : a + 10
        #SalinityMon2 = x(SalinityMon)
        #print(SalinityMon2[1])
    
        # apply lin regress k w/ salin
        Ksal = a + b * SalinityDay
        Ksal[Ksal < 0.05] = 0.05
   
        #mixingZ = FixedMixingDepth
    
        #Ksal = KLinear(salinity) #Klinear RUN 102a
        #uPAR = L0*(1-math.exp(-Ksal*MixingDepth))/(Ksal*MixingDepth) #vertical avg PAR to mixing lyr depth
        step1 = np.multiply(-Ksal,MixingDay)
        step2 = 1-np.exp(step1)
        step3 = np.multiply(Ksal,MixingDay)
        step4 = np.divide(step2,step3)

        L0 = LightDay * pPAR * (1-alb)
        uPAR = np.multiply(L0,step4) #vertical avg PAR to mixing lyr depth
        uPAR[SalinityDay == 0.0] = 0.0
        Ksal[SalinityDay == 0.0] = 0.0
        uPAR = np.round(uPAR, 2)
        Ksal = np.round(Ksal, 3)
        
        uPARxMixing = uPAR * MixingDay # avg light x depth mixing, added 2024-05
        
        if save_ecospace:
            #export PAR
            sigdigfmt = '%0.1f'
            EwEName = dctVarnameToEwEName[varNamesMixing]
            EwEName = EwEName.format(salinity_depthlev) #GO ??
            savepath = os.path.join(greigDir,ecospace_subdir_out,EwEName ) 
            if (os.path.isdir(savepath) != True):
                os.mkdir(savepath)
            savefn = saveTemplate.format(EwEName,iyear,buildSortableString(middle_day,2))
            savefullpath = os.path.join(savepath, savefn)
            print("Saving file " + savefullpath)
            saveASCFile(savefullpath, uPAR, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
        
            #export K
            sigdigfmt = '%0.2f'
            EwEName = "RUN" + NEMO_run + "_Kfromsal"
            #EwEName = "RUN102b-Kfromsal_singlelev10m"
            #EwEName = "RUN102a-Kfromsal_vertmeantop10m" #GO alternative
            #EwEName = EwEName.format(salinity_depthlev) #GO ??
            savepath = os.path.join(greigDir,ecospace_subdir_out,EwEName ) 
            if (os.path.isdir(savepath) != True):
                os.mkdir(savepath)
            savefn = saveTemplate.format(EwEName,iyear,buildSortableString(middle_day,2))
            savefullpath = os.path.join(savepath, savefn)
            print("Saving file " + savefullpath)
            saveASCFile(savefullpath, Ksal,bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
            
            #export PAR x Mixing
            sigdigfmt = '%0.2f'
            EwEName = "RUN" + NEMO_run + "_PARxMixing"
            #EwEName = "RUN102b-Kfromsal_singlelev10m"
            #EwEName = "RUN102a-Kfromsal_vertmeantop10m" #GO alternative
            #EwEName = EwEName.format(salinity_depthlev) #GO ??
            savepath = os.path.join(greigDir,ecospace_subdir_out,EwEName ) 
            if (os.path.isdir(savepath) != True):
                os.mkdir(savepath)
            savefn = saveTemplate.format(EwEName,iyear,buildSortableString(middle_day,2))
            savefullpath = os.path.join(savepath, savefn)
            print("Saving file " + savefullpath)
            saveASCFile(savefullpath, uPARxMixing, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader, dfPlumeMask, dfLandMask)
    
        if save_ecosim:
            # ############# ECOSIM CSV Forcing files #############
            # ecosim forcing grid requires monthly TS for long-term forcing
            # so here we avg across map and then expand, inserting 
            
            uPAR_ecosim_3day = uPAR.flatten()
            Ksal_ecosim_3day = Ksal.flatten()
            uPARxMixing_ecosim_3day = uPARxMixing.flatten()
            
                            
            uPAR_ecosim_3day = uPAR_ecosim_3day[uPAR_ecosim_3day != 0] # drop zeros for averaging
            Ksal_ecosim_3day = Ksal_ecosim_3day[Ksal_ecosim_3day != 0]
            uPARxMixing_ecosim_3day = uPARxMixing_ecosim_3day[uPARxMixing_ecosim_3day != 0] 
            
            uPAR_ecosim_3day = np.mean(uPAR_ecosim_3day)
            Ksal_ecosim_3day = np.mean(Ksal_ecosim_3day)
            uPARxMixing_ecosim_3day = np.mean(uPARxMixing_ecosim_3day)
            
            uPAR_ecosim_3day = np.round(uPAR_ecosim_3day,3)
            Ksal_ecosim_3day = np.round(Ksal_ecosim_3day,3)
            uPARxMixing_ecosim_3day = np.round(uPARxMixing_ecosim_3day,3)
            
            uPAR_ecosim_3day_all.append([iyear,middle_day,uPAR_ecosim_3day])
            Ksal_ecosim_3day_all.append([iyear,middle_day,Ksal_ecosim_3day])
            uPARxMixing_ecosim_3day_all.append([iyear,middle_day,uPARxMixing_ecosim_3day])
        
        if (lump_final) and (iday == 120):
            break
        j += 1
        
if save_ecosim:
        
    # add index corresponding to hacked 3day year
    uPAR_ecosim_3day_all_idx = [[index + 1] + sublist for index, sublist in enumerate(uPAR_ecosim_3day_all)]
    Ksal_ecosim_3day_all_idx = [[index + 1] + sublist for index, sublist in enumerate(Ksal_ecosim_3day_all)]
    uPARxMixing_ecosim_3day_all_idx = [[index + 1] + sublist for index, sublist in enumerate(uPARxMixing_ecosim_3day_all)]
    
    # expand the array since ecosim wants values per month not year in enviro forcing grid
    uPAR_ecosim_all_expnd = [sublist for sublist in uPAR_ecosim_3day_all_idx for _ in range(12)]
    Ksal_ecosim_all_expnd = [sublist for sublist in Ksal_ecosim_3day_all_idx for _ in range(12)]
    uPARxMixing_ecosim_all_expnd = [sublist for sublist in uPARxMixing_ecosim_3day_all_idx for _ in range(12)]
    
    # index by time step (fake 3day 'months' = 6hr)
    uPAR_ecosim_all_expnd_idx = [[i + 1] + row for i, row in enumerate(uPAR_ecosim_all_expnd)]
    Ksal_ecosim_all_expnd_idx = [[i + 1] + row for i, row in enumerate(Ksal_ecosim_all_expnd)]
    uPARxMixing_ecosim_all_expnd_idx = [[i + 1] + row for i, row in enumerate(uPARxMixing_ecosim_all_expnd)]
    
    uPAR_column_names = ['threeday_yrmo','threeday_yr','year', 'dayofyear', 'uPAR']
    Ksal_column_names = ['threeday_yrmo','threeday_yr','year', 'dayofyear', 'Ksal']
    uPARxMixing_column_names = ['threeday_yrmo','threeday_yr','year', 'dayofyear', 'uPARxMixing']

    savepath1 = os.path.join(greigDir, ecosim_subdir_out)
    if (os.path.isdir(savepath1) != True):
        os.mkdir(savepath1)
    savepath = os.path.join(greigDir, ecosim_subdir_out, PAR_code)
    if (os.path.isdir(savepath) != True):
        os.mkdir(savepath)
        
    #uPAR
    savefn = "uPAR_" + PAR_code + "_" + time_code + "_" + str(startyear) + "-" + str(endyear) + ".csv"
    savefullpath = os.path.join(savepath, savefn)
    print("Saving Ecosim file " + savefullpath)

    with open(savefullpath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(uPAR_column_names)
        writer.writerows(uPAR_ecosim_all_expnd_idx)
    
    #Ksal
    savefn = "Ksal_" + PAR_code + "_" + time_code + "_" + str(startyear) + "-" + str(endyear) + ".csv"
    savefullpath = os.path.join(savepath, savefn)
    print("Saving Ecosim file " + savefullpath)

    with open(savefullpath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(Ksal_column_names)
        writer.writerows(Ksal_ecosim_all_expnd_idx)
        
    #uPARxMixing
    savefn = "uPARxMixing_" + PAR_code + "_" + time_code + "_" + str(startyear) + "-" + str(endyear) + ".csv"
    savefullpath = os.path.join(savepath, savefn)
    print("Saving Ecosim file " + savefullpath)

    with open(savefullpath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(uPARxMixing_column_names)
        writer.writerows(uPARxMixing_ecosim_all_expnd_idx)