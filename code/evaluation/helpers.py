
import numpy as np
import netCDF4 as nc
import yaml
import datetime as dt
import os
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import patches
import xarray as xr
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

# basemap is currently not compatible with Python 3.10+ as far as I can tell
from mpl_toolkits.basemap import Basemap # to fix install, may need to revert to python 3.10

import pandas as pd

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def buildSortableString(number, nZeros):
    newstring = str(number)
    while (len(newstring) < nZeros) :
        tmpstr = "0" + newstring
        newstring = tmpstr
    return newstring

def haversine(lon1, lat1, lon2, lat2):
    """ This is copied from salishsea_tools """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c  # from ss tools
    # km = 6371 * c  # from earthdist.m
    return km * 1000

def t2u(lont,latt):
    # We don't have a value for the rightmost u points
    lonu = np.zeros(lont.shape)
    latu = np.zeros(lont.shape)
    latu[:,0:-1]=0.5*(latt[:,0:-1] + latt[:,1:])
    lonu[:,0:-1]=0.5*(lont[:,0:-1] + lont[:,1:])
    return lonu,latu


def t2v(lont,latt):
    # We don't have a value for the northmost v points
    lonv = np.zeros(lont.shape)
    latv = np.zeros(lont.shape)
    latv[0:-1,:]=0.5*(latt[0:-1,:] + latt[1:,:])
    lonv[0:-1,:]=0.5*(lont[0:-1,:] + lont[1:,:])
    return lonv,latv


def t2f(lont,latt):
    # We don't have values in the rightmost u pt or northmost v pt
    lonf = np.zeros(lont.shape)
    latf = np.zeros(lont.shape)
    lonf[0:-1,0:-1] = 0.25*(lont[0:-1,0:-1] + lont[1:,0:-1] + lont[0:-1,1:] + lont[1:,1:])
    latf[0:-1,0:-1] = 0.25*(latt[0:-1,0:-1] + latt[1:,0:-1] + latt[0:-1,1:] + latt[1:,1:])
    return lonf,latf


def gete1(lon, lat, expandleft=False):
    if expandleft:
        lon, lat = expandi(lon, lat)
    dx = np.zeros(lon.shape)
    lon1 = lon[0:-1, 0:-1]
    lat1 = lat[0:-1, 0:-1]
    lon2 = lon[0:-1, 1:]
    lat2 = lat[0:-1, 1:]
    dx[:-1, :-1] = haversine(lon1, lat1, lon2, lat2)
    return dx


def gete2(lon, lat, expanddown=False):
    if expanddown:
        lon, lat = expandj(lon, lat)
    dy = np.zeros(lon.shape)
    lon1 = lon[0:-1, 0:-1]
    lat1 = lat[0:-1, 0:-1]
    lon2 = lon[1:, 0:-1]
    lat2 = lat[1:, 0:-1]
    dy[:-1, :-1] = haversine(lon1, lat1, lon2, lat2)
    return dy


def expandi(lon, lat):
    # Expands the grid one to the left by linear extrapolation
    NY, NX = lon.shape
    lon2, lat2 = np.zeros([NY, NX + 1]), np.zeros([NY, NX + 1])
    # Lon
    lon2[:, 1:] = lon
    lon2[:, 0] = lon[:, 0] - (lon[:, 1] - lon[:, 0])
    # Lat
    lat2[:, 1:] = lat
    lat2[:, 0] = lat[:, 0] - (lat[:, 1] - lat[:, 0])
    return lon2, lat2


def expandj(lon, lat):
    # Expands the grid one down by linear extrapolation
    NY, NX = lon.shape
    lon2, lat2 = np.zeros([NY + 1, NX]), np.zeros([NY + 1, NX])
    # Long
    lon2[1:, :] = lon
    lon2[0, :] = lon[0, :] - (lon[1, :] - lon[0, :])
    # Lat
    lat2[1:, :] = lat
    lat2[0, :] = lat[0, :] - (lat[1, :] - lat[0, :])
    return lon2, lat2




def expandf(glamf, gphif):
    # Expand the f points grid so the f points form complete boxes around the t points
    # This is needed because the coordinates file truncates the f points by one.
    NY, NX = glamf.shape[0], glamf.shape[1]
    glamfe = np.zeros([NY+1, NX+1])
    gphife = np.zeros([NY+1, NX+1])
    # Long
    glamfe[1:,1:] = glamf
    glamfe[0,1:] = glamf[0,:] - (glamf[1,:] - glamf[0,:])     # extraoplation
    glamfe[:,0] = glamfe[:,1] - (glamfe[:,2] - glamfe[:,1])   # extraoplation
    # Lat
    gphife[1:,1:] = gphif
    gphife[0,1:] = gphif[0,:] - (gphif[1,:] - gphif[0,:])     # extraoplation
    gphife[:,0] = gphife[:,1] - (gphife[:,2] - gphife[:,1])   # extraoplation
    return glamfe, gphife

def grid_angle(f):
    with nc.Dataset(f, 'r') as cnc:
        glamu = cnc.variables['glamu'][0, ...]
        gphiu = cnc.variables['gphiu'][0, ...]
    # First point
    xA = glamu[0:-1, 0:-1]
    yA = gphiu[0:-1, 0:-1]
    # Second point
    xB = glamu[0:-1, 1:]
    yB = gphiu[0:-1, 1:]
    # Third point: same longitude as second point, same latitude as first point
    xC = xB
    yC = yA
    # Find angle by spherical trig
    # https://en.wikipedia.org/wiki/Solution_of_triangles#Three_sides_given_.28spherical_SSS.29
    R = 6367  # from geo_tools.haversine
    a = haversine(xB, yB, xC, yC) / R
    b = haversine(xA, yA, xC, yC) / R
    c = haversine(xA, yA, xB, yB) / R
    cosA = (np.cos(a) - np.cos(b) * np.cos(c)) / (np.sin(b) * np.sin(c))
    # cosB = (np.cos(b) - np.cos(c)*np.cos(a))/(np.sin(c)*np.sin(a))
    # cosC = (np.cos(c) - np.cos(a)*np.cos(b))/(np.sin(a)*np.sin(b))
    A = np.degrees(np.arccos(cosA))  # A is the angle counterclockwise from from due east
    return A

def writecoords(fname,
                glamt, glamu, glamv, glamf,
                gphit, gphiu, gphiv, gphif,
                e1t, e1u, e1v, e1f,
                e2t, e2u, e2v, e2f):
    # Build a NEMO format coordinates file

    cnc = nc.Dataset(fname, 'w', clobber=True)
    NY, NX = glamt.shape

    # Create the dimensions
    cnc.createDimension('x', NX)
    cnc.createDimension('y', NY)
    cnc.createDimension('time', None)

    # Create the float variables
    cnc.createVariable('nav_lon', 'f', ('y', 'x'), zlib=True, complevel=4)
    cnc.variables['nav_lon'].setncattr('units', 'degrees_east')
    cnc.variables['nav_lon'].setncattr('comment', 'at t points')

    cnc.createVariable('nav_lat', 'f', ('y', 'x'), zlib=True, complevel=4)
    cnc.variables['nav_lat'].setncattr('units', 'degrees_north')
    cnc.variables['nav_lat'].setncattr('comment', 'at t points')

    cnc.createVariable('time', 'f', ('time'), zlib=True, complevel=4)
    cnc.variables['time'].setncattr('units', 'seconds since 0001-01-01 00:00:00')
    cnc.variables['time'].setncattr('time_origin', '0000-JAN-01 00:00:00')
    cnc.variables['time'].setncattr('calendar', 'gregorian')

    # Create the double variables
    varlist = ['glamt', 'glamu', 'glamv', 'glamf', 'gphit', 'gphiu', 'gphiv', 'gphif']
    varlist += ['e1t', 'e1u', 'e1v', 'e1f', 'e2t', 'e2u', 'e2v', 'e2f']
    for v in varlist:
        cnc.createVariable(v, 'd', ('time', 'y', 'x'), zlib=True, complevel=4)
        cnc.variables[v].setncattr('missing_value', 1e+20)

    # Write the data
    cnc.variables['nav_lon'][...] = glamt
    cnc.variables['nav_lat'][...] = gphit
    #
    cnc.variables['glamt'][0, ...] = glamt
    cnc.variables['glamu'][0, ...] = glamu
    cnc.variables['glamv'][0, ...] = glamv
    cnc.variables['glamf'][0, ...] = glamf
    #
    cnc.variables['gphit'][0, ...] = gphit
    cnc.variables['gphiu'][0, ...] = gphiu
    cnc.variables['gphiv'][0, ...] = gphiv
    cnc.variables['gphif'][0, ...] = gphif
    #
    cnc.variables['e1t'][0, ...] = e1t
    cnc.variables['e1u'][0, ...] = e1u
    cnc.variables['e1v'][0, ...] = e1v
    cnc.variables['e1f'][0, ...] = e1f
    #
    cnc.variables['e2t'][0, ...] = e2t
    cnc.variables['e2u'][0, ...] = e2u
    cnc.variables['e2v'][0, ...] = e2v
    cnc.variables['e2f'][0, ...] = e2f

    cnc.close()


def writebathy(filename,glamt,gphit,bathy):

    bnc = nc.Dataset(filename, 'w', clobber=True)
    NY,NX = glamt.shape

    # Create the dimensions
    bnc.createDimension('x', NX)
    bnc.createDimension('y', NY)

    bnc.createVariable('nav_lon', 'f', ('y', 'x'), zlib=True, complevel=4)
    bnc.variables['nav_lon'].setncattr('units', 'degrees_east')

    bnc.createVariable('nav_lat', 'f', ('y', 'x'), zlib=True, complevel=4)
    bnc.variables['nav_lat'].setncattr('units', 'degrees_north')

    bnc.createVariable('Bathymetry', 'd', ('y', 'x'), zlib=True, complevel=4, fill_value=0)
    bnc.variables['Bathymetry'].setncattr('units', 'metres')

    bnc.variables['nav_lon'][:] = glamt
    bnc.variables['nav_lat'][:] = gphit
    bnc.variables['Bathymetry'][:] = bathy

    bnc.close()

#from pypkg
def find_nearest_point(lon,lat,glon,glat,mask,dx):
    """Utility: Finds the indicies of a station.
                lon and lat are the coordinates of the station
                glon and glat are the fields of coordinates from coord file
                    mask is the land mask, since if the closest point is on land it is not valid
                dz is the radius in degrees around which to search"""

    # This funciton only supports lon/lat values that are single floats, not arrays
    # For multiple points, call this function repeatedly (once per point).

    #lon=lon-0.04   # no clue why I put this in at some point

    # Find the points within dx degrees of the specified lon,lat
    b=np.nonzero( (glon[:,:] < lon+dx) & (glon[:,:] > lon-dx) & (glat[:,:] < lat+dx) & (glat[:,:]> lat-dx) & (mask[:,:]>0))
    if (len(b[0]) ==0) :
        return (np.nan, np.nan, np.nan)

    npts = b[0].shape[0]    # how many points are in the range?
    dist = np.zeros(npts)   # initialize variable

    for n in range (npts):
        # Get the indices in b in a shorter form
        # note that since this uses netcdf 3 files and the scipy.io import regime, dimensions go (y,x) rather than (x,y).
        # Make sure this is consistent with the calling program - don't use without verifying that this is sensible
        # If you change the order here, change it a few lines down when you're getting the minimum distance too.
        ix = b[1][n];  iy = b[0][n]

        dist[n] = haversine(lon, lat, glon[iy,ix], glat[iy,ix])

    # Now get the minimum distance.
    ind= np.argmin(dist)            # get the index of the minimum value
    ix = b[1][ind]; iy = b[0][ind]        # get the indicies of the relevant point in the search regions

    # Return a named tuple with all the info: output

    #nearest_pt = collections.namedtuple('nearest_pt', ['id','grp', 'glon', 'glat', 'mask', 'dist', 'ix', 'iy'])
    p = (ix,iy, dist[ind]) #nearest_pt (id,grp,glon[iy,ix], glat[iy,ix], mask[iy,ix], dist[ind], ix, iy)
    return p

def find_nearest_point_from1D(lon_obs,lat_obs,lon_mod,lat_mod,mask,dx):

    """Utility: Find closest model point to obs using type of nearest neighbour search
                lon and lat - the coordinates of the obs or station
                lon_mod and lat_mod - fields of coordinates from model grid
                mask - the land mask, since if the closest point is on land it is not valid
                dx - the radius in degrees around which to search
                assumes lon_mod, lat_mod same shape
                returns - index of closest point, distance in m to pt"""

    # function modified by G Oldford, originally from pyap
    # changelog
    #  - made function for 1D instead of 2D arrays of model lats lons
    # For multiple points, call this function repeatedly (once per point).

    # Find the points within dx degrees of the specified lon,lat
    b = np.nonzero((lon_mod[:] < lon_obs + dx) & (lon_mod[:] > lon_obs - dx) & (lat_mod[:] < lat_obs + dx) & (
                lat_mod[:] > lat_obs - dx) & (mask[:] > 0))

    npts = len(b[0])  # how many points are in the range?

    if (len(b[0]) ==0) :
        #print(lat_obs, lon_obs)
        return (np.nan, np.nan)

    dist = np.zeros(npts)   # initialize variable

    for n in range (npts):
        idx = b[0][n]
        dist[n] = haversine(lon_obs, lat_obs, lon_mod[idx], lat_mod[idx])

    # Now get the minimum distance.
    ind = np.argmin(dist)            # get the index of the minimum value
    # ix = b[1][ind]; iy = b[0][ind]        # get the indicies of the relevant point in the search regions
    idx = b[0][ind] # get the indicies of the relevant point in the search regions

    # Return a named tuple with all the info: output

    #nearest_pt = collections.namedtuple('nearest_pt', ['id','grp', 'glon', 'lat_mod', 'mask', 'dist', 'ix', 'iy'])
    # p = (ix,iy, dist[ind]) #nearest_pt (id,grp,glon[iy,ix], lat_mod[iy,ix], mask[iy,ix], dist[ind], ix, iy)
    # dist should be metres
    p = (idx, dist[ind])  # nearest_pt (id,grp,glon[iy,ix], lat_mod[iy,ix], mask[iy,ix], dist[ind], ix, iy)
    return p

def load_config_yaml(f):
    configpath = os.path.normpath(os.path.dirname(__file__) + "/../config")
    yamldata = load_yaml(os.path.join(configpath, f))
    return yamldata

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

# Function to find the closest date
# def find_closest_date(model_times, target_date):
#     return min(model_times, key=lambda x: abs(x - target_date))

def find_closest_date(model_times, target_date):
    if isinstance(target_date, str):
        target_date = datetime.fromisoformat(target_date)  # e.g., "2012-05-17"
    return min(model_times, key=lambda x: abs(x - target_date))

# Function to find the closest depth
def find_closest_depth(obs_depth, possible_depths):
    closest_depth = min(possible_depths, key=lambda x: abs(x - obs_depth))
    return closest_depth

# reads the subdomain coords
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

def convert_times(times, time_units, time_calendar):
    times = nc.num2date(times, units=time_units, calendar=time_calendar)
    times = np.array([np.datetime64(t) for t in times])
    return times

# computes monthly averages (year-month mean)
def reshape_to_year_month(data, years, months):
    unique_years = np.unique(years)
    unique_months = np.unique(months)

    year_dim = len(unique_years)
    month_dim = len(unique_months)

    reshaped_data = np.empty((year_dim, month_dim, data.shape[1], data.shape[2]))

    for i, year in enumerate(unique_years):
        for j, month in enumerate(unique_months):
            mask = (years == year) & (months == month)
            reshaped_data[i, j, :, :] = data.sel(time=mask).mean(dim='time')

    return reshaped_data

def get_ecospace_data_3(file_path, start_year=None, end_year=None, monthly_averages=False):
    if os.path.exists(file_path):
        try:
            with (nc.Dataset(file_path, 'r') as dataset):
                metadata = {
                    "Dimensions": {dim: len(dataset.dimensions[dim]) for dim in dataset.dimensions},
                    "Variables": {var: dataset.variables[var].dimensions for var in dataset.variables}
                }
                time_var = dataset.variables['time']
                time_units = time_var.units
                time_calendar = time_var.calendar if 'calendar' in time_var.ncattrs() else 'standard'
                times = convert_times(time_var[:], time_units, time_calendar)

                if start_year is not None and end_year is not None:
                    start_date = np.datetime64(f'{start_year}-01-01')
                    end_date = np.datetime64(f'{end_year}-12-31')
                    time_mask = (times >= start_date) & (times <= end_date)
                    times = times[time_mask]
                else:
                    time_mask = slice(None)

                ZF1_ICT = dataset.variables['ZF1-ICT'][time_mask]

                ZC1_EUP = dataset.variables['ZC1-EUP'][time_mask]
                ZC2_AMP = dataset.variables['ZC2-AMP'][time_mask]
                ZC3_DEC = dataset.variables['ZC3-DEC'][time_mask]
                ZC4_CLG = dataset.variables['ZC4-CLG'][time_mask]
                ZC5_CSM = dataset.variables['ZC5-CSM'][time_mask]

                ZS1_JEL = dataset.variables['ZS1-JEL'][time_mask]
                ZS2_CTH = dataset.variables['ZS2-CTH'][time_mask]
                ZS3_CHA = dataset.variables['ZS3-CHA'][time_mask]
                ZS4_LAR = dataset.variables['ZS4-LAR'][time_mask]

                PZ1_CIL = dataset.variables['PZ1-CIL'][time_mask]
                PZ2_DIN = dataset.variables['PZ2-DIN'][time_mask]
                PZ3_HNF = dataset.variables['PZ3-HNF'][time_mask]
                PP1_DIA = dataset.variables['PP1-DIA'][time_mask]
                PP2_NAN = dataset.variables['PP2-NAN'][time_mask]
                PP3_PIC = dataset.variables['PP3-PIC'][time_mask]

                EWE_row = dataset.variables['row'][:]
                EWE_col = dataset.variables['col'][:]
                EWE_depth = dataset.variables['depth'][:]
                row_indices = EWE_row
                col_indices = EWE_col

                ds = xr.Dataset({

                    'ZF1_ICT': (['time', 'EWE_row', 'EWE_col'], ZF1_ICT),

                    'ZC1_EUP': (['time', 'EWE_row', 'EWE_col'], ZC1_EUP),
                    'ZC2_AMP': (['time', 'EWE_row', 'EWE_col'], ZC2_AMP),
                    'ZC3_DEC': (['time', 'EWE_row', 'EWE_col'], ZC3_DEC),
                    'ZC4_CLG': (['time', 'EWE_row', 'EWE_col'], ZC4_CLG),
                    'ZC5_CSM': (['time', 'EWE_row', 'EWE_col'], ZC5_CSM),

                    'ZS1_JEL': (['time', 'EWE_row', 'EWE_col'], ZS1_JEL),
                    'ZS2_CTH': (['time', 'EWE_row', 'EWE_col'], ZS2_CTH),
                    'ZS3_CHA': (['time', 'EWE_row', 'EWE_col'], ZS3_CHA),
                    'ZS4_LAR': (['time', 'EWE_row', 'EWE_col'], ZS4_LAR),

                    'PZ1_CIL': (['time', 'EWE_row', 'EWE_col'], PZ1_CIL),
                    'PZ2_DIN': (['time', 'EWE_row', 'EWE_col'], PZ2_DIN),
                    'PZ3_HNF': (['time', 'EWE_row', 'EWE_col'], PZ3_HNF),
                    'PP1_DIA': (['time', 'EWE_row', 'EWE_col'], PP1_DIA),
                    'PP2_NAN': (['time', 'EWE_row', 'EWE_col'], PP2_NAN),
                    'PP3_PIC': (['time', 'EWE_row', 'EWE_col'], PP3_PIC)
                }, coords={'time': times, 'EWE_row': row_indices, 'EWE_col': col_indices})

                if monthly_averages:
                    ds = ds.resample(time='1M').mean()

                reshaped_data = {}
                for var in ds.data_vars:
                    data = ds[var]
                    years = data['time.year']
                    months = data['time.month']
                    reshaped_data[var] = reshape_to_year_month(data, years, months)

                return metadata, reshaped_data, row_indices, col_indices, EWE_depth
        except OSError as e:
            return {"Error": str(e)}, None, None, None, None
    else:
        return {"Error": "File not found."}, None, None, None, None


class new_utils_GO():

    # interpolated to given depths, takes nc as input
    # adapted from fn in Nanoose ipynb's 202304
    def get_model_interpolated(ncname, mod_dlevs, verbose=False):

        tobs = xr.open_dataset(ncname)
        obs_d = tobs['Pres'].values
        obs_t = tobs['cTemp'].values
        obs_s = tobs['aSal'].values
        ttime = tobs['time'][0].values

        # throw out stuff below 400
        filt = obs_d > 400
        obs_d[filt] = np.nan
        obs_t[filt] = np.nan
        obs_s[filt] = np.nan

        try:
            f = interpolate.interp1d(obs_d, obs_t)  # temperature
            f2 = interpolate.interp1d(obs_d, obs_s)  # salinity

            ## can only interpolate to model points that are within observations
            mod_d = mod_dlevs[(mod_dlevs < max(obs_d)) & (mod_dlevs > min(obs_d))]
            firstind = np.where(mod_dlevs == np.min(mod_d))[0][0]  ## first model index we were able to interpolate to
            interp_t = f(mod_d)  # use interpolation function returned by `interp1d`
            interp_s = f2(mod_d)

            ###
            t_full = np.zeros(40)
            t_full[:] = -999
            t_full[firstind:firstind + len(mod_d)] = interp_t
            t_full[t_full < -900] = np.nan
            t_full[t_full > 100] = np.nan

            s_full = np.zeros(40)
            s_full[:] = -999
            s_full[firstind:firstind + len(mod_d)] = interp_s
            s_full[s_full < -900] = np.nan
            s_full[s_full > 100] = np.nan

            if verbose:
                fig, axs = plt.subplots(1, 2)
                axs = axs.ravel()
                axs[0].plot(obs_t, obs_d, 'ob', interp_t, mod_d, 'or')
                axs[1].plot(obs_s, obs_d, 'ob', interp_s, mod_d, 'or')
                axs[0].set_title('temperature cons')
                axs[1].set_title('salinity abs')
                axs[0].invert_yaxis()
                axs[1].invert_yaxis()

                fig.suptitle(f'blue is CTD, red is interp. to mod depths \n date of cast: {ttime}')
                plt.tight_layout()
                plt.show()
        except:

            t_full = np.zeros(40)
            t_full[:] = -999
            t_full[t_full < -900] = np.nan
            s_full = np.zeros(40)
            s_full[:] = -999
            s_full[s_full < -900] = np.nan

        return t_full, s_full, ttime  # not sure ttime is needed -GO 202304

    # interpolated to given depths, takes np arrays as input
    # adapted from fn in Nanoose ipynb's 202304
    def get_model_interpolated_ar(d_pres, d_salt, d_temp, d_time, mod_dlevs, verbose=False):

        obs_d = d_pres
        obs_t = d_temp
        obs_s = d_salt
        ttime = d_time

        # throw out stuff below 400
        filt = obs_d > 400
        obs_d[filt] = np.nan
        obs_t[filt] = np.nan
        obs_s[filt] = np.nan

        try:
            f = interpolate.interp1d(obs_d, obs_t)  # temperature
            f2 = interpolate.interp1d(obs_d, obs_s)  # salinity

            ## can only interpolate to model points that are within observations
            mod_d = mod_dlevs[(mod_dlevs < max(obs_d)) & (mod_dlevs > min(obs_d))]
            firstind = np.where(mod_dlevs == np.min(mod_d))[0][0]  ## first model index we were able to interpolate to
            interp_t = f(mod_d)  # use interpolation function returned by `interp1d`
            interp_s = f2(mod_d)

            ###
            t_full = np.zeros(40)
            t_full[:] = -999
            t_full[firstind:firstind + len(mod_d)] = interp_t
            t_full[t_full < -900] = np.nan
            t_full[t_full > 100] = np.nan

            s_full = np.zeros(40)
            s_full[:] = -999
            s_full[firstind:firstind + len(mod_d)] = interp_s
            s_full[s_full < -900] = np.nan
            s_full[s_full > 100] = np.nan

            if verbose:
                fig, axs = plt.subplots(1, 2)
                axs = axs.ravel()
                axs[0].plot(obs_t, obs_d, 'ob', interp_t, mod_d, 'or')
                axs[1].plot(obs_s, obs_d, 'ob', interp_s, mod_d, 'or')
                axs[0].set_title('temperature cons')
                axs[1].set_title('salinity abs')
                axs[0].invert_yaxis()
                axs[1].invert_yaxis()

                fig.suptitle(f'blue is CTD, red is interp. to mod depths \n date of cast: {ttime}')
                plt.tight_layout()
                plt.show()
        except:

            t_full = np.zeros(40)
            t_full[:] = -999
            t_full[t_full < -900] = np.nan
            s_full = np.zeros(40)
            s_full[:] = -999
            s_full[s_full < -900] = np.nan

        return t_full, s_full

# Function get SSCast grid indices and lats lons from BATHY file
def get_sscast_grd_idx(file_path):
    if os.path.exists(file_path):
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                metadata = {
                    "Dimensions": {dim: len(dataset.dimensions[dim]) for dim in dataset.dimensions},
                    "Variables": {var: dataset.variables[var].dimensions for var in dataset.variables}
                }

                # Assuming 'lat' and 'lon' are the variable names for latitude and longitude in the file
                lats = dataset.variables['latitude'][:]
                lons = dataset.variables['longitude'][:]

                # Assuming 'gridY' and 'gridX' are the variable names for grid indices in the file
                gridY = dataset.variables['gridY'][:]
                gridX = dataset.variables['gridX'][:]

                bathy = dataset.variables['bathymetry'][:]

            return metadata, lats, lons, gridY, gridX, bathy
        except OSError as e:
            return {"Error": str(e)}, None
    else:
        return {"Error": "File not found."}, None

# Function to get data from salishseacast model output file
def get_sscast_data(file_path):
    if os.path.exists(file_path):
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                metadata = {
                    "Dimensions": {dim: len(dataset.dimensions[dim]) for dim in dataset.dimensions},
                    "Variables": {var: dataset.variables[var].dimensions for var in dataset.variables}
                }
                time_var = dataset.variables['time']
                time_units = time_var.units
                time_calendar = time_var.calendar if 'calendar' in time_var.ncattrs() else 'standard'
                times = nc.num2date(time_var[:], units=time_units, calendar=time_calendar)

                ciliates = dataset.variables['ciliates'][:]
                flagellates = dataset.variables['flagellates'][:]
                diatoms = dataset.variables['diatoms'][:]

            return metadata, times, ciliates, flagellates, diatoms
        except OSError as e:
            return {"Error": str(e)}, None, None, None, None
    else:
        return {"Error": "File not found."}, None, None, None, None

# made a copy of this in helpers 2024-07-30
def make_map(ax, grid, w_map=[-124, -123.9, 47.7, 50.6],
             rotation=39.2,
             par_inc=0.25,
             mer_inc=0.5,
             fs=14,
             bg_color='#e5e5e5'
             ):
    """
    """
    # fcolor='burlywood'
    fcolor = 'white'
    # Make projection
    m = Basemap(ax=ax,
                projection='lcc', resolution='c',
                lon_0=(w_map[1] - w_map[0]) / 2 + w_map[0] + rotation,
                lat_0=(w_map[3] - w_map[2]) / 2 + w_map[2],
                llcrnrlon=w_map[0], urcrnrlon=w_map[1],
                llcrnrlat=w_map[2], urcrnrlat=w_map[3])

    # Add features and labels
    x, y = m(grid.nav_lon.values, grid.nav_lat.values)
    ax.contourf(x, y, grid.Bathymetry, [-0.01, 0.01], colors=fcolor)
    ax.contour(x, y, grid.Bathymetry, [-0.01, 0.01], colors='black', linewidths=0.1)
    ax.contourf(x, y, grid.Bathymetry, [0.011, 500], colors=bg_color)

    #     m.drawmeridians(np.arange(-125.5, -122, mer_inc), labels=[0, 0, 0, 1], linewidth=0.2, fontsize=fs)
    m.drawmeridians(np.arange(-125.5, -122, mer_inc), labels=[0, 0, 0, 0], linewidth=0.2, fontsize=fs)
    #     m.drawparallels(np.arange(48, 51, par_inc), labels=[1, 0, 0, 0], linewidth=0.2, fontsize=fs)
    m.drawparallels(np.arange(47, 51, par_inc), labels=[0, 0, 0, 0], linewidth=0.2, fontsize=fs)

    return m

# made copy of this in helpers 2024-07-30
def adjust_map(ax,
               lat_bl=47.8,
               lon_bl=-123.2,
               lat_br=48.8,
               lon_br=-122.28,
               lat_tl=50.3,
               lon_tl=-124.75,
               lat_bl2=48.2,
               lon_bl2=-123.5,
               label_grid=False
               ):
    # fcolor='burlywood'
    fcolor = 'white'

    w_map = [-124, -123.9, 47.7, 50.6]
    rotation = 39.2
    m = Basemap(ax=ax,
                projection='lcc', resolution='c',
                lon_0=(w_map[1] - w_map[0]) / 2 + w_map[0] + rotation,
                lat_0=(w_map[3] - w_map[2]) / 2 + w_map[2],
                llcrnrlon=w_map[0], urcrnrlon=w_map[1],
                llcrnrlat=w_map[2], urcrnrlat=w_map[3])

    # set width using map units
    # bottom left
    x_bl, y_bl = m(lon_bl, lat_bl)
    x_br, _ = m(lon_br, lat_br)
    ax.set_xlim(x_bl, x_br)

    # top left
    x_tl, y_tl = m(lon_tl, lat_tl)
    x_bl, y_bl = m(lon_bl2, lat_bl2)
    ax.set_ylim(y_bl, y_tl)

    # fix a little path in bottom right
    lccx_TL, lccy_TL = m(-122.83, 49.4)
    lccx_BR, lccy_BR = m(-122.58, 48.7)
    lccx_BL, lccy_BL = m(-122.33, 48.7)
    lccw = lccx_BL - lccx_BR
    lcch = lccy_TL - lccy_BL

    ax.add_patch(patches.Rectangle(
        (lccx_BL, lccy_BL), lccw, lcch,
        facecolor=fcolor, edgecolor='k',
        linewidth=0,
        zorder=0))

    if label_grid:
        #         rotation = 0
        #         ax.annotate("49.5 N", xy=(m(-124.8, 49.5)), xytext=(m(-125, 49.5)),
        #                     xycoords='data', textcoords='data',
        #                     ha='right', va='center', fontsize=8, rotation=rotation)
        fsmer = 7.5
        ax.annotate("50.5N", xy=(m(-123.99, 50.43)), xytext=(m(-123.95, 50.43)),
                    xycoords='data', textcoords='data',
                    ha='left', va='center', fontsize=fsmer, rotation=-30)

        ax.annotate("123.5W", xy=(m(-123.65, 50.05)), xytext=(m(-123.65, 50.15)),
                    xycoords='data', textcoords='data',
                    ha='left', va='center', fontsize=fsmer, rotation=60)
        ax.annotate("50N", xy=(m(-123.5, 49.94)), xytext=(m(-123.42, 49.94)),
                    xycoords='data', textcoords='data',
                    ha='left', va='center', fontsize=fsmer, rotation=-30)
        ax.annotate("49.5N", xy=(m(-123.06, 49.4)), xytext=(m(-122.9, 49.4)),
                    xycoords='data', textcoords='data',
                    ha='left', va='center', fontsize=fsmer, rotation=-30)

        ax.annotate("122.5W", xy=(m(-122.8, 49.15)), xytext=(m(-122.61, 49.15)),
                    xycoords='data', textcoords='data',
                    ha='left', va='center', fontsize=fsmer, rotation=60)

        ax.annotate("49N", xy=(m(-122.65, 49)), xytext=(m(-122.4, 48.93)),
                    xycoords='data', textcoords='data',
                    ha='left', va='center', fontsize=fsmer, rotation=-30)


#         ax.annotate("", xy=(m(-123.5, 50)), xytext=(m(-123.4, 50)),
#                     xycoords='data', textcoords='data',
#                     ha='left', va='center', fontsize=8, rotation=rotation)

#         rotation = -30
#         ax.annotate("50 N", xy=(m(-124.3, 50)), xytext=(m(-124.3, 50.08)),
#                     xycoords='data', textcoords='data',
#                     ha='right', va='center', fontsize=8, rotation=rotation)
#         ax.annotate("49 N", xy=(m(-124.3, 49)), xytext=(m(-124.3, 49.08)),
#                     xycoords='data', textcoords='data',
#                     ha='right', va='center', fontsize=8, rotation=rotation)


def custom_formatter(x, pos):
    return f'{x:.2f}'


def custom_formatter2(x, pos):
    return f'{x:.1f}'

def find_bloom_doy(df, df_field,
                   thrshld_fctr = 1.05,
                   sub_thrshld_fctr = 0.7,
                   average_from = "annual",
                   mean_or_median = "median",
                   exclude_juntojan = True,
                   bloom_early = 68, #suchy's
                   bloom_late = 108 # suchy's
                   ):
# Created 2024 by G Oldford
# Purpose compute bloom date
#
# Based on:based primarily on Suchy et al methods
#
# Params:
#   thrshld_fctr - the threshold above average used to determine bloom
#   sub_thrshld_fctr - if values fall back beyond this proportion of average, then no bloom
#   average_from - is the average from that year or all years?
#   mean_or_median - is the average based on median or mean values?
#   exclude_juntojan - from model outputs, exclude from calculating mean / median values (no satellite data these months)
#
# Updates:
#  2025-05-05 - added option to exclude_decjan

    # Create lists to hold the results
    bloom_dates = []
    bloom_days_of_year = []
    bloom_earlylate = []

    if exclude_juntojan:
        df = df[~df['Date'].dt.month.isin([1,5,6,7,8,9,10,11,12])]

    # Group by year
    df['Year'] = df['Date'].dt.year # redundant?

    if average_from == "all":
        median_value = df[df_field].median()
        mean_value = df[df_field].mean()
    # median_value = 1.65
    # median_value = 1.9
    grouped = df.groupby('Year')

    # Iterate through each year
    for year, group in grouped:

        if average_from=="annual":
            median_value = group[df_field].median()
            mean_value = group[df_field].mean()

        if mean_or_median == "mean":
            threshold = thrshld_fctr * mean_value
        else:
            threshold = thrshld_fctr * median_value

        print("year " + str(year))
        print("threshold " + str(threshold))
        sub_threshold = sub_thrshld_fctr * threshold
        # sub_threshold = sub_thrshld_fctr * threshold

        bloom_found = False

        # Iterate through the time steps of the year
        for i in range(len(group) - 2):  # Ensure we have at least two subsequent steps
            if group.iloc[i][df_field] >= threshold:
                # print("possible bloom found")
                # print(group.iloc[i + 1]['PP1-DIA'])
                # print(group.iloc[i + 2]['PP1-DIA'])
                # print(group.iloc[i + 3]['PP1-DIA'])
                if (
                        (group.iloc[i + 1][df_field] + group.iloc[i + 2][df_field]) / 2
                        > sub_threshold
                ) or (
                        (group.iloc[i + 3][df_field] + group.iloc[i + 4][df_field]) / 2
                        > sub_threshold
                ):
                    # print("bloom passes second crit")

                    bloom_dates.append(group.iloc[i]['Date'])
                    bloom_doy = group.iloc[i]['Date'].timetuple().tm_yday
                    bloom_days_of_year.append(bloom_doy)
                    bloom_found = True
                    print(group.iloc[i + 1][df_field])
                    if bloom_doy + 1.5 <= bloom_early:
                        bloom_earlylate.append("early")
                    elif bloom_doy - 1.5 >= bloom_late:
                        bloom_earlylate.append("late")
                    elif (bloom_doy + 1.5 <= bloom_late - 1) & (bloom_doy - 1.5 >= bloom_early - 1):
                        bloom_earlylate.append("avg")
                    else:
                        bloom_earlylate.append("cusp")
                    break

        # If no bloom is found for the year, append None
        if not bloom_found:
            bloom_dates.append(None)
            bloom_days_of_year.append(None)
            bloom_earlylate.append(None)

    return bloom_dates, bloom_days_of_year, bloom_earlylate


# GO2021-12-22 fmt= has big effect on ASC file size, added argument for this
def saveASCFile(filename, data,
                bottomleft_row_ewe,
                upperleft_row_ewe,
                upperleft_col_ewe,
                sigdigfmt,
                ASCheader,
                dfPlumeMask=None,
                dfLandMask=None):

    # Greig's bounds clipping from "Basemap Converter (Py3).ipynb"
    # I don't know why this works but it does... -JB
    # (due to how indexing is done in ASC vs NC -GO)
    trimmedData = []
    i = bottomleft_row_ewe
    while i < upperleft_row_ewe:
        trimmedData.append(data[i, upperleft_col_ewe:])
        i += 1
    trimmedData.reverse()

    dfEwEGrid = pd.DataFrame(data=trimmedData)

    if not (dfPlumeMask is None):
        dfEwEGrid = dfEwEGrid.mask(dfPlumeMask)

    if not (dfLandMask is None):
        dfEwEGrid = dfEwEGrid.mask(dfLandMask)

    # dfEwEGrid.fillna(0.0, inplace=True) #deprecated
    dfEwEGrid[dfEwEGrid.select_dtypes(include='number').columns] = (
        dfEwEGrid.select_dtypes(include='number').fillna(0.0)
    )

    np.savetxt(filename, dfEwEGrid.to_numpy(), fmt=sigdigfmt, delimiter=" ", comments='',
               header=ASCheader)  # GO 20211222 - change from %0.5f


# Open a Dataframe
def getDataFrame(fullPath, NaN):
    if os.path.exists(fullPath):
        nas = [NaN]
        df = pd.read_table(fullPath, skiprows=6, header=None, delim_whitespace=True, na_values=nas)
        return df