#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# created by GO 2023, 
# last edited 2023-05-10 -GO
#
# Purpose: interpolate wind onto NEMO grid - useful for analysis and for further work to export to ECOSPACE
# Input: RDRS files w/ adjusted vars from MD /project/6006412/mdunphy/Forcing/RDRS_v2.1_fornemo/ e.g. RDRSv21_y2016.nc
# Output: RDRS NC files (hourly) on NEMO grid 1.3 Gb / yr
#
# Notes:
#   20230510 - created by copying the light script
#            - to do: should be updated to merge light and wind and reduce redundant code (put the helper functions
#            - into GO_helpers.py and modify to make generic)
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import netCDF4 as nc
import numpy as np
import os
from nemointerp import build_matrix, use_matrix

# Output: 
inpath = '/project/6006412/mdunphy/Forcing/RDRS_v2.1_fornemo/'
outpath = '/project/6006412/goldford/ECOSPACE/DATA/wind_hourly/'
basepath = '/project/6006412/mdunphy/Forcing/'

year_start = 1980
year_end = 2018

# match these w/ input file
lat_dimname = "rlat"
lon_dimname = "rlon"
time_dimname = "time"

lat_varname = "lat"
lon_varname = "lon"
time_varname = "time"
wind_u10_varname = "u10m"
wind_v10_varname = "v10m"


if (os.path.isdir(outpath) != True):
  os.mkdir(outpath)
  
# Write empty NC file
def prep_outfile(f, hours_yr, nemoshape, nlat, nlon, outfile):

  g = nc.Dataset(outfile, 'w')
  

  #copy global attributes
  for attname in f.ncattrs():
    setattr(g,attname, getattr(f,attname))
  
  #copy and adjust dimensions
  for dimname,dim in f.dimensions.items():
     if (dimname== lat_dimname):
       g.createDimension(dimname,nemoshape[0])
     elif (dimname == lon_dimname):
       g.createDimension(dimname,nemoshape[1])
     elif (dimname == time_dimname):
       g.createDimension(dimname,None) # make unlimited
     else:
       g.createDimension(dimname,len(dim))
  
  # copy variables
  for varname,ncvar in f.variables.items():
    
    # skip other vars
    if (varname != lat_varname) & (varname != lon_varname) & (varname != time_varname) & (varname != wind_u10_varname) & (varname != wind_v10_varname):
      continue
    
    #print('###################')
    #print(varname)
    #print(ncvar.dimensions)
    
    if ((varname == wind_u10_varname)|(varname == wind_v10_varname)):
    #if varname == "msdwswrf":
      #var = g.createVariable(varname,ncvar.dtype,ncvar.dimensions,fill_value=ncvar._FillValue)
      var = g.createVariable(varname,'f4',ncvar.dimensions,fill_value = 0)
      var.missing_value = 0
      var.scale_factor = 1.0
      var.add_offset = 0.0
    elif (varname == lat_varname) or (varname == lon_varname):
      var = g.createVariable(varname,ncvar.dtype,(lat_dimname,lon_dimname,))
    else: 
      var = g.createVariable(varname,ncvar.dtype,ncvar.dimensions)
    
    for attname in ncvar.ncattrs():
      #print("attributename = " + attname)
      if (attname != "_FillValue") & ((attname != "missing_value")):

        if ((varname != wind_u10_varname)|(varname != wind_v10_varname)):
          setattr(var,attname,getattr(ncvar,attname))

    # variables which are also dimensions
    if (varname != lat_varname) & (varname != lon_varname) & (varname != time_varname): 
      print('check') #GO no need to populate - 20230510
      #for hour in range(0,hours_yr,1):
        #var[hour,:,:] = np.zeros((1, nemoshape[0], nemoshape[1])) - 999
    elif (varname == lat_varname): # lat / lon have simpler interp         
      var[:,:] = nlat
    elif (varname == lon_varname):
      var[:,:] = nlon
    else: #time
      
      ######   @@ DEBUG @@ replace with ncvar[:] (not sure this is needed - GO 20210922)  #####
      #var[:] = ncvar[0:hours_yr]
      var[:] = ncvar[:]
      
  g.close() # required? 
    #print("shape check: " + str(var.shape))

# UPDATE THE FILE with 1 hr of data
def update_outfile(outfile, varname, varvalue, hour):
  with nc.Dataset(outfile,'r+') as ncf:
    ncf[varname][hour,:,:] = np.around(varvalue,decimals=1)

month_hr = {
"Jan": [0,744], "Feb": [744,1416], "Mar": [1416,2160], "Apr":[2160,2880],
"May": [2880,3624], "Jun": [3624,4344], "Jul": [4344,5088], "Aug":[5088,5832],
"Sep": [5832,6552], "Oct": [6552,7296], "Nov": [7296,8016], "Dec": [8016,8760]
}

# for leapyear
month_hr_ly = {
"Jan": [0,744], "Feb": [744,1440], "Mar": [1440,2184], "Apr":[2184,2904],
"May": [2904,3648], "Jun": [3648,4368], "Jul": [4368,5112], "Aug":[5112,5856],
"Sep": [5856,6576], "Oct": [6576,7320], "Nov": [7320,8040], "Dec": [8040,8784]
}

# use example file to build interp matrix
#eg = inpath + 'ERA5_SalishSea_9vars_' + str(year_start) + '.nc'
#  RDRSv21_y2016.nc
eg = inpath + 'RDRSv21_y' + str(year_start) + '.nc'

M1,nemoshape = build_matrix(basepath + 'RDRS_v2.1_fornemo/weights_bilin_RDRSv2.1.nc', eg, xname=lon_dimname, yname=lat_dimname)

print(nemoshape)

# following previous code using coords file for nlat nlon but
# unsure why I don't just get the dimensions from nemoshape  - GO 20210922
with nc.Dataset(basepath + 'grid/coordinates_salishsea_1500m.nc') as ncf:
  nlat = ncf.variables['gphit'][0,...]   # neglected the 1/2 grid box shift here
  nlon = ncf.variables['glamt'][0,...]

for year in range(year_start, year_end + 1, 1):
    print(year)
    #f1 = nc.Dataset(inpath + 'ERA5_SalishSea_9vars_' + str(year) + '.nc', 'r')
    f1 = nc.Dataset(inpath + 'RDRSv21_y' + str(year) + '.nc', 'r')
    
    # prep / write empty output file
    outfile = outpath + 'RDRS21_NEMOgrid_wind_' + str(year) + '.nc'
    
    #leap year 
    if (( year%400 == 0)or (( year%4 == 0 ) and ( year%100 != 0))):
        print("%d is a Leap Year" %year)
        month_hr = month_hr_ly
        hours_yr = 8784
    else:
        hours_yr = 8760
        
    prep_outfile(f1, hours_yr, nemoshape, nlat, nlon, outfile)

    # wind components at 10 m
    wind_u10 = f1.variables[wind_u10_varname]
    wind_v10 = f1.variables[wind_v10_varname]


    i = 0
    for hour in range(0, hours_yr, 1): 
        
        #if (hour%24 == 0):
            #print ("Day: " + str(hour/24))
        
        wind_u10_NEMO = use_matrix(M1, nemoshape, wind_u10[hour,:,:])
        wind_v10_NEMO = use_matrix(M1, nemoshape, wind_v10[hour,:,:])
        
        wind_u10_NEMO = np.expand_dims(wind_u10_NEMO, axis=0)
        wind_v10_NEMO = np.expand_dims(wind_v10_NEMO, axis=0)
        
        wind_u10_NEMO[wind_u10_NEMO < -200] = 0
        wind_v10_NEMO[wind_v10_NEMO < -200] = 0
        
        update_outfile(outfile, wind_u10_varname, wind_u10_NEMO, hour)
        update_outfile(outfile, wind_v10_varname, wind_v10_NEMO, hour)

    f1.close()
