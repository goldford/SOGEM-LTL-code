import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
from calendar import monthrange
import math
import yaml
import datetime as dt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_yaml(yamlfile):
    """ Helper to load a YAML
    """
    def date_to_datetime(loader, node):
        """ The default YAML loader interprets YYYY-MM-DD as a datetime.date object
            Here we override this with a datetime.datetime object with implicit h,m,s=0,0,0 """
        d = yaml.constructor.SafeConstructor.construct_yaml_timestamp(loader,node)
        if type(d) is dt.date:
            d = dt.datetime.combine(d, dt.time(0, 0, 0))
        return d
    yaml.constructor.SafeConstructor.yaml_constructors[u'tag:yaml.org,2002:timestamp'] = date_to_datetime
    with open(yamlfile, 'r') as ya:
        try:
            yamldata = yaml.safe_load(ya)
        except Exception as e:
            print("Error importing YAML file {} {})".format(yamlfile,e))
            raise
    return yamldata


def read_sdomains(domain_file):
    try:
        data = load_yaml(domain_file)
    except FileNotFoundError:
        print("WARNING:\n domain_file {} not found".format(domain_file))

    coords = {}
    for c in data['polygon_coords'].keys():
        coords[c] = np.asarray(data['polygon_coords'][c])
        # make sure the polygon is closed
        if not np.all(coords[c][-1] == coords[c][0]):
            coords[c] = np.vstack([coords[c], coords[c][0, :]])

    return coords


def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def buildSortableString(number, nZeros): 
    newstring = str(number)
    while (len(newstring) < nZeros) :
        tmpstr = "0" + newstring
        newstring = tmpstr
    return newstring   


#def saveASCFile(filename, data, bottomleft_row_ewe, upperleft_row_ewe, upperleft_col_ewe,sigdigfmt,ASCheader):
#
#    trimmedData = []
#    i =  bottomleft_row_ewe 
#    while i < upperleft_row_ewe:
#        trimmedData.append(data[i,upperleft_col_ewe:])
#        i += 1
#    trimmedData.reverse()
#
#    np.savetxt(filename,trimmedData,fmt=sigdigfmt, delimiter=" ", comments='',header=ASCheader)

# GO2021-12-22 fmt= has big effect on ASC file size, added argument for this
def saveASCFile(filename, data,
                bottomleft_row_ewe,
                upperleft_row_ewe, upperleft_col_ewe,
                sigdigfmt, ASCheader,
                dfPlumeMask=None, dfLandMask=None):
    #Greig's bounds clipping from "Basemap Converter (Py3).ipynb"
    #I don't know why this works but it does... -JB
    #(due to how indexing is done in ASC vs NC -GO)
    trimmedData = []
    i =  bottomleft_row_ewe 
    while i < upperleft_row_ewe:
        trimmedData.append(data[i,upperleft_col_ewe:])
        i += 1
    trimmedData.reverse()
    
    dfEwEGrid = pd.DataFrame(data=trimmedData)
    
    if not (dfPlumeMask is None):
      dfEwEGrid = dfEwEGrid.mask(dfPlumeMask)
      
    if not (dfLandMask is None):
      dfEwEGrid = dfEwEGrid.mask(dfLandMask)

    dfEwEGrid.fillna(0.0,inplace=True) # deprecation warning
    # non_obj_cols = dfEwEGrid.select_dtypes(exclude='object').columns #solution leaves nans in?
    # dfEwEGrid[non_obj_cols] = dfEwEGrid[non_obj_cols].fillna(0.0)
    
    np.savetxt(filename, dfEwEGrid.to_numpy(), fmt=sigdigfmt, delimiter=" ", comments='',header=ASCheader) # GO 20211222 - change from %0.5f


#Open a Dataframe
def getDataFrame(fullPath,NaN):
    if os.path.exists(fullPath):
        nas = [NaN]
        #df = pd.read_table(fullPath, skiprows=6, header=None, delim_whitespace=True,na_values=nas) # future warning change - GO 2025
        df = pd.read_table(fullPath, skiprows=6, header=None, sep=r'\s+', na_values=nas)
        return df
