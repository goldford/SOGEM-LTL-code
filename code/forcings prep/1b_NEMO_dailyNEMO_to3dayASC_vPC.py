# Updated version of 1_NEMO_dailyNEMO_to_3dayASC.py
# Script: 1_NEMO_dailyNEMO_to3dayASC_v2025.py
# By G Oldford 2023-2025
# Description:
#   Processes daily NEMO output NetCDF files to generate 3-day averaged ASC files for use in Ecospace.
#   Supports flexible vertical extraction methods, including:
#     (1) Bin-weighted depth averaging using e3t0 vertical layer thicknesses
#     (2) Depth interpolation for arbitrary target depths or depth ranges
#     (3) Near-bottom or bottom-segment (e.g., last 200m) extractions
#
# Features:
#   - Fraser River and plume region masking
#   - User-configurable start/end years, variable selection, and output formatting
#   - Ecosim-compatible CSV time series export
#
# Notes:
#   - Designed for use with SalishSea NEMO outputs
#   - Interpolation can be toggled on/off using `use_interp`
#   - All outputs are saved in ASC raster format with consistent header structure
#   - 2025-05-29 - added paralellization using mp - GO

# for reference, the centre of the cell depths (gdept_0), not accounting for ssh stretch:
#     [[0.5000003   1.5000031   2.5000114   3.5000305   4.5000706   5.5001507
#       6.5003104   7.500623    8.501236    9.502433   10.5047655  11.509312
#       12.518167   13.535412   14.568982   15.634288   16.761173   18.007135
#       19.481785   21.389978   24.100256   28.229916   34.685757   44.517723
#       58.484333   76.58559    98.06296   121.866516  147.08946   173.11449
#       199.57304   226.2603    253.06664   279.93454   298.58588   308.9961
#       360.67453   387.6032    414.5341    441.4661]]
# and the max and min delineation of each vertical cell (deptw_0), not accounting for ssh stretch:
# [[  0.          1.0000012   2.0000064   3.0000193   4.0000467   5.000104
#     6.000217    7.0004406   8.000879    9.001736   10.003407   11.006662
#    12.013008   13.025366   14.049429   15.096255   16.187304   17.364035
#    18.705973   20.363474   22.613064   25.937412   31.101034   39.11886
#    50.963238   67.05207    86.96747   109.73707   134.34593   160.02956
#   186.30528   212.89656   239.65305   266.4952    293.3816    303.79166
#   347.2116    374.1385    401.06845   428.       ]]

# Discussion:
# The challenge of defining "near-bottom" arises from the vertical grid in NEMO,
# where the depth bins increase in thickness toward the bottom (vertical stretching).
# Near the surface, bins are thin (~1m), but deeper layers can span 20m+.
# Selecting just the bottom cell may capture a broad and variable depth range.
# Consider whether averaging over the last 50m (or another threshold) is better
# depending on scientific needs and grid resolution.

import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import re
from GO_helpers import is_leap_year, buildSortableString, saveASCFile, getDataFrame, read_sdomains
from matplotlib.path import Path
from functools import partial
from scipy.interpolate import interp1d
import multiprocessing as mp
import warnings


# warnings.simplefilter(action='ignore', category="FutureWarning")

# Config
USE_MP_PARALLEL = False # mp not working - causing memory related crashes
N_CORES = 3
startyear = 1980
endyear = 2018
NEMO_run = "216"
out_code = "var"
timestep = "3day"
lump_final = True
advec_mult = 100 / 365 * 3
upwel_mult = 1000 / 365 * 3
ASCheader = "ncols        93\nnrows        151 \nxllcorner    -123.80739\nyllcorner    48.29471 \ncellsize     0.013497347 \nNODATA_value  0.0"

# Paths
# root_dir = "/project/6006412"
root_dir = "C:"
root_dir_PC = "D:"
# mesh_f = f"{root_dir}/goldford/data/mesh_mask/mesh_mask_20210406.nc" #server
mesh_f = f"{root_dir}/Users/Greig/Documents/GitHub/HOTSSea_v1_NEMOandSupportCode/desktop/data/mesh mask/mesh_mask_20210406.nc"
# plume_mask_f = f"{root_dir}/goldford/ECOSPACE/DATA/Fraser_Plume_Region.asc" #server
plume_mask_f = f"{root_dir}/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/basemap/Fraser_Plume_Region.asc"
# ecospacegrid_f = f"{root_dir}/goldford/ECOSPACE/DATA/ecospacedepthgrid.asc" #server
ecospacegrid_f = f"{root_dir}/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/basemap/ecospacedepthgrid.asc"
# frmask_f = f"{root_dir}/goldford/ECOSPACE/SCRIPTS/FraserRiverMaskPnts.yml" #server
frmask_f = f"{root_dir}/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/basemap/FraserRiverMaskPnts.yml"
# nemo_nc_p = "/project/6006412/mdunphy/nemo_results/SalishSea1500" #server
nemo_nc_p = f"{root_dir_PC}/nemo_results/SalishSea1500_RUN216"
# asc_out_p = "/project/6006412/goldford/ECOSPACE/DATA" #server
asc_out_p = f"{root_dir_PC}/ecospace_in_test"


use_interp = True  # <-- set this to False to use weighted bin averaging - GO 2025


# Load mesh mask for vertical structure and bottom index calculation
with nc.Dataset(mesh_f, 'r') as mesh:
    e3t0 = mesh.variables['e3t_0'][:]
    tmask = mesh.variables['tmask'][:]
    navlat = mesh.variables['nav_lat'][:]
    navlon = mesh.variables['nav_lon'][:]
    gdept_0 = mesh.variables['gdept_0'][:]
    gdepw_0 = mesh.variables['gdepw_0'][:]

# Fraser plume mask and land mask
plume_mask = getDataFrame(plume_mask_f, "-9999.00000000") == 1
depth_grid = getDataFrame(ecospacegrid_f, "-9999.00000000")
land_mask = depth_grid == 0

# Fraser River polygon mask
frmask_pts = read_sdomains(frmask_f)
frmask = np.ones([1,tmask.shape[2],tmask.shape[3]], dtype=np.int32)
for i in range(navlat.shape[0]):
    for j in range(navlat.shape[1]):
        lat = navlat[i, j]
        lon = navlon[i, j]
        pos = [lat,lon]
        for sd in frmask_pts.keys():
            if Path(frmask_pts[sd]).contains_point(pos):
                frmask[0,i,j] = 0
frmask = np.ma.masked_where(frmask == 0, frmask)

# Precompute bottom indices (new version)
bottom_idxs = np.argmax(tmask[::-1, :, :], axis=0)
bottom_idxs = tmask.shape[0] - 1 - bottom_idxs
bottom_idxs[tmask[0, :, :] == 0] = -1

# Precompute bottom indices (old version for comparison)
bottom_idxs_leg = np.zeros([tmask.shape[2],tmask.shape[3]], dtype=np.int32)
for i in range(tmask.shape[2]):
    for j in range(tmask.shape[3]):
        tmask2 = tmask[0, :, i, j]
        npmax = np.max(tmask2)
        if npmax == 0:
            bottom_idx_leg = -1
        else:
            bottom_idx_leg = np.where(tmask[0, :, i, j])[0][-1] # needs to be tmask
        bottom_idxs_leg[i,j] = bottom_idx_leg

# # Compare old vs new
# if not np.all(bottom_idxs == bottom_idxs_leg):
#     print("WARNING: bottom_idxs mismatch between new and legacy method!")
# else:
#     print("bottom_idxs match!")


def interpolate_and_avg(var3d, gdepth, target_min, target_max, resolution=1.0):
    """
    Interpolates a vertical profile to evenly spaced target depths, then averages over the range.

    Args:
        var3d: 3D array [depth, y, x] of the variable to be averaged
        gdepth: 3D array [depth, y, x] of depth centers
        target_min: Minimum target depth for interpolation
        target_max: Maximum target depth
        resolution: Vertical resolution (default 1.0 m)

    Returns:
        2D array [y, x] of interpolated and averaged values
    """
    target_depths = np.arange(target_min, target_max + resolution, resolution)
    output = np.full(var3d.shape[1:], np.nan)

    for i in range(var3d.shape[1]):  # y
        for j in range(var3d.shape[2]):  # x
            profile = var3d[:, i, j]
            depths = gdepth[:, i, j]

            if np.ma.is_masked(profile):
                profile = profile.compressed()
                depths = depths[~profile.mask]

            if len(profile) > 2:
                f = interp1d(depths, profile, bounds_error=False, fill_value='extrapolate')
                interp_vals = f(target_depths)
                output[i, j] = np.nanmean(interp_vals)

    return output


def weighted_depth_avg(var3d, e3t):
    """
    Performs a vertically weighted average using e3t bin thicknesses over the full water column.

    Args:
        var3d: 3D array [depth, y, x] of variable
        e3t: 3D or 4D array of vertical bin thicknesses (e3t_0)

    Returns:
        2D masked array [y, x] of depth-weighted averages
    """
    if e3t.ndim == 4 and e3t.shape[0] == 1:
        e3t = e3t[0]  # Remove time dimension
    masked = ma.masked_where(e3t == 0, var3d)
    weights = ma.masked_where(e3t == 0, e3t)
    return ma.average(masked, axis=0, weights=weights)


def weighted_depth_avg_range(var3d, e3t, d1, d2):
    """
    Performs a depth-weighted average between two layer indices (d1 to d2).

    Args:
        var3d: 3D array [depth, y, x] of variable
        e3t: e3t bin widths (3D or 4D)
        d1: Start depth index (inclusive)
        d2: End depth index (exclusive)

    Returns:
        2D masked array [y, x] of averaged values over the specified depth range
    """
    if e3t.ndim == 4 and e3t.shape[0] == 1:
        e3t = e3t[0]  # Remove time dimension
    var_slice = var3d[d1:d2]
    weight_slice = e3t[d1:d2]
    masked = ma.masked_where(weight_slice == 0, var_slice)
    weights = ma.masked_where(weight_slice == 0, weight_slice)
    return ma.average(masked, axis=0, weights=weights)


def interpolate_depth_range(var3d, gdepth, min_depth, max_depth, dz=1.0):
    """
    Interpolates to fine vertical resolution (e.g., every 1 m) and averages within a depth range.

    Args:
        var3d: 3D array [depth, y, x]
        gdepth: 4D array [1, depth, y, x] of depth values
        min_depth: Minimum depth to interpolate to
        max_depth: Maximum depth
        dz: Interpolation spacing (default 1 m)

    Returns:
        2D array [y, x] of interpolated and averaged values
    """
    zvals = np.arange(min_depth, max_depth + dz, dz)
    nz, ny, nx = var3d.shape
    out = np.full((ny, nx), np.nan)

    for i in range(ny):
        for j in range(nx):
            raw_profile  = var3d[:, i, j]
            depths = gdepth[0, :, i, j]  # Updated to match shape (1, depth, y, x)

            # Skip if all values are zero or NaN
            if np.all((raw_profile == 0) | np.isnan(raw_profile)):
                continue

            # Mask where values are zero or NaN
            profile = ma.masked_invalid(raw_profile)
            profile = ma.masked_where(profile == 0, profile)

            if np.all(profile.mask):
                continue

            valid = ~profile.mask
            if valid.sum() < 2:
                continue

            if depths[valid].max() < max_depth:
                continue  # Skip grid cells that are too shallow

            f = interp1d(depths[valid], profile[valid], bounds_error=False, fill_value='extrapolate')
            interp_vals = f(zvals)
            out[i, j] = np.nanmean(interp_vals)

    return np.nan_to_num(out, nan=0.0)


def get_depth_indices(gdept_0, min_depth, max_depth):
    """
    Returns the indices corresponding to a depth range based on gdept_0 profile.

    Assumes horizontal uniformity in gdept_0.

    Args:
        gdept_0: 4D array [1, depth, y, x] of cell center depths
        min_depth: Minimum depth (inclusive)
        max_depth: Maximum depth (inclusive)

    Returns:
        Tuple (start_idx, end_idx) for use in slicing depth
    """
    # Assume gdept_0 has shape (1, depth, y, x) and is horizontally uniform
    gdept_1d = gdept_0[0, :, 0, 0]  # Take the profile from one column as representative
    indices = np.where((gdept_1d >= min_depth) & (gdept_1d <= max_depth))[0]
    if len(indices) == 0:
        raise ValueError(f"No depth indices found for range {min_depth} to {max_depth} m")
    return indices[0], indices[-1] + 1


def extract_near_bottom(var3d):
    """
    Extracts the value from the bottom-most valid layer at each horizontal location.

    Args:
        var3d: 3D array [depth, y, x]

    Returns:
        2D array [y, x] of bottom layer values
    """
    out = np.full((var3d.shape[1], var3d.shape[2]), np.nan)
    for i in range(var3d.shape[1]):
        for j in range(var3d.shape[2]):
            bidx = bottom_idxs[i, j]
            if bidx > 0:
                out[i, j] = var3d[bidx, i, j]
    return out


def extract_200m_to_bottom(var3d, e3t):
    # OLD - UNUSED?
    """
    Averages values from approximately 200m above the bottom to the bottom.

    Args:
        var3d: 3D array [depth, y, x]
        e3t: 3D array of bin thicknesses (same shape as var3d)

    Returns:
        2D array [y, x] of depth-weighted averages from 200m above bottom to bottom
    """
    out = np.full((var3d.shape[1], var3d.shape[2]), np.nan)
    for i in range(var3d.shape[1]):
        for j in range(var3d.shape[2]):
            bidx = bottom_idxs[i, j]
            if bidx <= 0:
                continue
            start_idx = 0
            cum_depth = 0
            for k in range(bidx):
                cum_depth += e3t[k, i, j]
                if cum_depth >= 200:
                    start_idx = k
                    break
            segment = var3d[start_idx:bidx+1, i, j]
            weights = e3t[start_idx:bidx+1, i, j]
            if weights.sum() > 0:
                out[i, j] = np.average(segment, weights=weights)
    return out


def extract_last_Nm_from_bottom(var3d, e3t, N):
    """
    Averages over the last N meters of the water column above the seabed.

    Args:
        var3d: 3D array [depth, y, x]
        e3t: 3D array of bin thicknesses
        N: Vertical range in meters from the bottom upward

    Returns:
        2D array [y, x] of depth-weighted averages over bottom N meters
    """
    out = np.full((var3d.shape[1], var3d.shape[2]), np.nan)
    for i in range(var3d.shape[1]):
        for j in range(var3d.shape[2]):
            bidx = bottom_idxs[i, j]
            if bidx <= 0:
                continue
            cum_depth = 0
            start_idx = bidx
            for k in range(bidx, -1, -1):
                cum_depth += e3t[k, i, j]
                if cum_depth >= N:
                    start_idx = k
                    break
            segment = var3d[start_idx:bidx+1, i, j]
            weights = e3t[start_idx:bidx+1, i, j]
            if weights.sum() > 0:
                out[i, j] = np.average(segment, weights=weights)
    return out

def interpolate_at_depth(var3d, gdepth, target_depth):
    """
    Interpolates the value of a variable to a specific depth.

    Args:
        var3d: 3D array [depth, y, x]
        gdepth: 4D array [1, depth, y, x] of depth centers
        target_depth: Depth to interpolate to (in meters)

    Returns:
        2D array [y, x] of interpolated values at the given depth
    """
    nz, ny, nx = var3d.shape
    out = np.full((ny, nx), np.nan)

    for i in range(ny):
        for j in range(nx):
            raw_profile = var3d[:, i, j]
            depths = gdepth[0, :, i, j]

            if np.all((raw_profile == 0) | np.isnan(raw_profile)):
                continue

            profile = ma.masked_invalid(raw_profile)
            profile = ma.masked_where(profile == 0, profile)

            if np.all(profile.mask):
                continue

            valid = ~profile.mask
            if valid.sum() < 2:
                continue

            if depths[valid].max() < target_depth:
                continue

            f = interp1d(depths[valid], profile[valid], bounds_error=False, fill_value=np.nan)
            out[i, j] = f(target_depth)

    return np.nan_to_num(out, nan=0.0)

def interpolate_single_cell(args):
    i, j, var3d, gdepth, target_depth = args
    raw_profile = var3d[:, i, j]
    depths = gdepth[0, :, i, j]

    if np.all((raw_profile == 0) | np.isnan(raw_profile)):
        return (i, j, 0.0)

    profile = ma.masked_invalid(raw_profile)
    profile = ma.masked_where(profile == 0, profile)

    if np.all(profile.mask):
        return (i, j, 0.0)

    valid = ~profile.mask
    if valid.sum() < 2:
        return (i, j, 0.0)

    if depths[valid].max() < target_depth:
        return (i, j, 0.0)

    f = interp1d(depths[valid], profile[valid], bounds_error=False, fill_value=np.nan)
    return (i, j, float(np.nan_to_num(f(target_depth), nan=0.0)))


def interpolate_at_depth_parallel(var3d, gdepth, target_depth, n_cores=N_CORES):
    ny, nx = var3d.shape[1:]
    out = np.full((ny, nx), np.nan)
    args = [(i, j, var3d, gdepth, target_depth) for i in range(ny) for j in range(nx)]

    with mp.Pool(processes=N_CORES) as pool:
        results = pool.map(interpolate_single_cell, args)

    for i, j, value in results:
        out[i, j] = value

    return out


def interpolate_depth_range_single_cell(args):
    i, j, var3d, gdepth, min_depth, max_depth, dz = args
    raw_profile = var3d[:, i, j]
    depths = gdepth[0, :, i, j]

    if np.all((raw_profile == 0) | np.isnan(raw_profile)):
        return (i, j, 0.0)

    profile = ma.masked_invalid(raw_profile)
    profile = ma.masked_where(profile == 0, profile)

    if np.all(profile.mask):
        return (i, j, 0.0)

    valid = ~profile.mask
    if valid.sum() < 2:
        return (i, j, 0.0)

    if depths[valid].max() < max_depth:
        return (i, j, 0.0)

    zvals = np.arange(min_depth, max_depth + dz, dz)
    f = interp1d(depths[valid], profile[valid], bounds_error=False, fill_value='extrapolate')
    interp_vals = f(zvals)
    return (i, j, float(np.nanmean(interp_vals)))


def interpolate_depth_range_parallel(var3d, gdepth, min_depth, max_depth, dz=1.0, n_cores=N_CORES):
    ny, nx = var3d.shape[1:]
    out = np.full((ny, nx), np.nan)
    args = [(i, j, var3d, gdepth, min_depth, max_depth, dz) for i in range(ny) for j in range(nx)]

    with mp.Pool(processes=N_CORES) as pool:
        results = pool.map(interpolate_depth_range_single_cell, args)

    for i, j, value in results:
        out[i, j] = value

    return out

def interpolate_below_depth(var3d, gdepth, min_depth, dz=1.0):
    """
    Interpolates from min_depth to the bottom of the water column at each cell
    and returns the average over that range.

    Args:
        var3d: 3D array [depth, y, x]
        gdepth: 4D array [1, depth, y, x] of depth centers
        min_depth: Minimum depth to start averaging from
        dz: Vertical resolution (default 1 m)

    Returns:
        2D array [y, x] of interpolated averages below min_depth
    """
    nz, ny, nx = var3d.shape
    out = np.full((ny, nx), np.nan)

    for i in range(ny):
        for j in range(nx):
            raw_profile = var3d[:, i, j]
            depths = gdepth[0, :, i, j]

            if np.all((raw_profile == 0) | np.isnan(raw_profile)):
                continue

            profile = ma.masked_invalid(raw_profile)
            profile = ma.masked_where(profile == 0, profile)

            if np.all(profile.mask):
                continue

            valid = ~profile.mask
            if valid.sum() < 2:
                continue

            max_local_depth = depths[valid].max()
            if max_local_depth < min_depth:
                continue  # too shallow for this threshold

            zvals = np.arange(min_depth, max_local_depth + dz, dz)
            f = interp1d(depths[valid], profile[valid], bounds_error=False, fill_value='extrapolate')
            interp_vals = f(zvals)
            out[i, j] = np.nanmean(interp_vals)

    return np.nan_to_num(out, nan=0.0)


min_d_0to4m, max_d_0to4m = get_depth_indices(gdept_0, 1.5, 4) # exclude top-most metre b/c biased
min_d_0to10m, max_d_0to10m = get_depth_indices(gdept_0, 0, 10)
min_d_30to40m, max_d_30to40m = get_depth_indices(gdept_0, 30, 40)

# crashes PC - as of June 16 2025
if USE_MP_PARALLEL:
    func_at150m = partial(interpolate_at_depth_parallel, gdepth=gdept_0, target_depth=150)
    func_temp_avg0to10m = (partial(interpolate_depth_range_parallel, gdepth=gdept_0, min_depth=0, max_depth=10, dz=1.0)
                  if use_interp else
                  partial(weighted_depth_avg_range, e3t=e3t0, d1=min_d_0to10m, d2=max_d_0to10m))
else:
    func_at150m = partial(interpolate_at_depth, gdepth=gdept_0, target_depth=150)
    func_temp_avg0to10m = (partial(interpolate_depth_range, gdepth=gdept_0, min_depth=0, max_depth=10, dz=1.0)
                  if use_interp else
                  partial(weighted_depth_avg_range, e3t=e3t0, d1=min_d_0to10m, d2=max_d_0to10m))

# MAIN LOOP – Load variables, apply extraction, and export
var_defs = {
    # "temp_at35m": {
    #     "filename": "votemper",
    #     "func": partial(interpolate_at_depth_parallel, gdepth=gdept_0, target_depth=35),
    #     "sigdig": '%0.1f'
    # },
    # "temp_at150m": {
    #     "filename": "votemper",
    #     "func": func_at150m,
    #     "sigdig": '%0.1f'
    #},
    # "temp_avg0to10m": {
    #     "filename": "votemper",
    #     "func": func_temp_avg0to10m,
    #     "sigdig": '%0.1f'
    # },

    # "temp_bottom": {
    #     "filename": "votemper",
    #     "func": extract_near_bottom,
    #     "sigdig": '%0.1f'
    # },
    # "temp_last50m": {
    #     "filename": "votemper",
    #     "func": lambda v: extract_last_Nm_from_bottom(v, e3t0, 50),
    #     "sigdig": '%0.1f'
    # },
    # "salt_avg0to30m": {
    #     "filename": "vosaline",
    #     "func": (partial(interpolate_depth_range, gdepth=gdept_0, min_depth=0, max_depth=30)
    #               if use_interp else
    #               partial(weighted_depth_avg_range, e3t=e3t0, d1=min_d, d2=max_d)),
    #     "sigdig": '%0.1f'
    # },
    # "salt_bottom": {
    #     "filename": "vosaline",
    #     "func": extract_near_bottom,
    #     "sigdig": '%0.1f'
    # }
    "temp_avg150toBot": {
        "filename": "votemper",
        "func": partial(interpolate_below_depth, gdepth=gdept_0, min_depth=150, dz=1.0),
        "sigdig": '%0.1f'
    }
}

for var_key, meta in var_defs.items():
    varname = meta["filename"]
    process_func = meta["func"]
    sigdigfmt = meta["sigdig"]
    for iyear in range(startyear, endyear + 1):
        var_12mo = []
        for imon in range(1, 13):
            # file = f"{nemo_nc_p}/SalishSea1500-RUN{NEMO_run}/CDF/SalishSea1500-RUN{NEMO_run}_1d_grid_T_y{iyear}m{buildSortableString(imon,2)}.nc"
            file = f"{nemo_nc_p}/SalishSea1500-RUN{NEMO_run}_1d_grid_T_y{iyear}m{buildSortableString(imon,2)}.nc"

            with nc.Dataset(file) as ds:
                var_dat = ds.variables[varname][:]
            var_12mo.append(var_dat)
        var_full = np.concatenate(var_12mo, axis=0)  # shape (days, depth, y, x)

        leap = is_leap_year(iyear)
        nsteps = 122 if leap else 121
        var_ecosim_all = []

        for j, iday in enumerate(range(1, nsteps + 1)):
            day_strt = (iday - 1) + (j * 2)
            day_end = day_strt + 2
            middle_day = day_strt + 2

            if iday == 121:
                vardays = var_full[day_strt:, :, :, :]
                middle_day += 1
            else:
                vardays = var_full[day_strt:day_end + 1, :, :, :]

            varday_avg = np.ma.mean(vardays, axis=0)
            varday_clip = process_func(varday_avg)  # shape (299, 132)

            # Apply Fraser River mask before saving
            if frmask is not None:
                varday_clip = ma.masked_where(frmask[0] == 0, varday_clip)

            # Save ASC
            varday_clip = np.nan_to_num(varday_clip, nan=0.0)
            path_out = f"{asc_out_p}/SS1500-RUN{NEMO_run}/ECOSPACE_in_{timestep}/{out_code}{var_key}"
            os.makedirs(path_out, exist_ok=True)
            save_name = f"{out_code}{var_key}_{iyear}_{buildSortableString(middle_day, 3)}.asc"
            save_path = os.path.join(path_out, save_name)
            print(f"Saving {save_path}")
            saveASCFile(save_path, varday_clip, 102, 253, 39, sigdigfmt, ASCheader, plume_mask, land_mask)


            # Save mean for Ecosim
            var_flat = varday_clip.compressed()
            var_mean = np.round(np.mean(var_flat), int(sigdigfmt[-2]))
            var_ecosim_all.append([iyear, middle_day, var_mean])

            if lump_final and iday == 121:
                break
            j += 1

        # Save Ecosim time series
        ecosim_dir = f"{asc_out_p}/SS1500-RUN{NEMO_run}/ECOSIM_in_{timestep}/{out_code}{var_key}"
        os.makedirs(ecosim_dir, exist_ok=True)
        ecosim_csv = os.path.join(ecosim_dir, f"{out_code}{var_key}_{timestep}_{startyear}-{endyear}.csv")
        var_ecosim_all_idx = [[i+1] + row for i, row in enumerate(var_ecosim_all)]
        var_ecosim_expnd = [row for row in var_ecosim_all_idx for _ in range(12)]
        var_ecosim_expnd_idx = [[i+1] + r for i, r in enumerate(var_ecosim_expnd)]

        with open(ecosim_csv, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["threeday_yrmo", "threeday_yr", "year", "dayofyear", f"{out_code}{var_key}"])
            writer.writerows(var_ecosim_expnd_idx)





