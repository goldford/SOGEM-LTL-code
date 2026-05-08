#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# created by GO 2023, 
# last edited 2023-04-06 -GO
#
# Purpose: interpolate light onto NEMO grid - useful for analysis and for further work to export to ECOSPACE
# Input: RDRS files w/ adjusted vars from MD /project/6006412/mdunphy/Forcing/RDRS_v2.1_fornemo/ e.g. RDRSv21_y2016.nc
# Output: RDRS NC files (hourly) on NEMO grid 1.3 Gb / yr
#
# Notes:
#   20230510 - debugged w/ MD. Several fixes. -999's were causing weird values in output
#            - weird values weren't issue, likely, and instead issue is that some RDRS input
#              files have dates that are the year prior which parse oddly. 
#            - the fix for the above comes in ksh step 4 4_Light_Hourly etc... 
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import netCDF4 as nc
import numpy as np
import os
from nemointerp import build_matrix, use_matrix

# Output: 
inpath = '/project/6006412/mdunphy/Forcing/RDRS_v2.1_fornemo/'
outpath = '/project/6006412/goldford/ECOSPACE/DATA/light_hourly/'
basepath = '/project/6006412/mdunphy/Forcing/'

year_start = 1980
year_end = 2018

# match these w/ input light file
lat_dimname = "rlat"
lon_dimname = "rlon"
time_dimname = "time"

lat_varname = "lat"
lon_varname = "lon"
time_varname = "time"
light_varname = "solar_rad"

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
    if (varname != lat_varname) & (varname != lon_varname) & (varname != time_varname) & (varname != light_varname):
      continue
    
    #print('###################')
    #print(varname)
    #print(ncvar.dimensions)
    
    if varname == light_varname:
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

        if (varname != light_varname):
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
      
  g1.close() # required? 
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
    outfile = outpath + 'RDRS21_NEMOgrid_light_' + str(year) + '.nc'
    
    #leap year 
    if (( year%400 == 0)or (( year%4 == 0 ) and ( year%100 != 0))):
      print("%d is a Leap Year" %year)
      month_hr = month_hr_ly
      hours_yr = 8784
    else:
      hours_yr = 8760

    prep_outfile(f1, hours_yr, nemoshape, nlat, nlon, outfile)
    
    # shortwave mean downward radiation flux
    light = f1.variables[light_varname]


    i = 0
    for hour in range(0, hours_yr, 1): 
        
        #if (hour%24 == 0):
            #print ("Day: " + str(hour/24))
        
        light_NEMO = use_matrix(M1, nemoshape, light[hour,:,:])
        light_NEMO = np.expand_dims(light_NEMO, axis=0)
        light_NEMO[light_NEMO<0] = 0
        update_outfile(outfile, light_varname, light_NEMO, hour)

    f1.close()
