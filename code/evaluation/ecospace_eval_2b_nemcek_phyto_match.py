"""
Script:    Ecospace Data Prep - 2a Nemcek Phyto Match to Ecospace SSC v2.py
Created:   July 11, 2024
Last Edit: May, 2025
Author:    G. Oldford

Purpose:
   Match phytoplankton observations from Nemcek et al. (2023) with output from:
     - Ecospace ecosystem model
     - SalishSeaCast biogeochemical model

Inputs:
   - Ecospace output NetCDF file (e.g., Scv51-RSPI_AllTemp_Wind_2000-2018.nc)
   - SSC model output NetCDF file (e.g., SalishSeaCast_biology_2008.nc)
   - Nemcek et al. 2023 phytoplankton CSV data (e.g., Nemcek_Supp_Data.csv)

Outputs:
   - CSV file combining Nemcek observations with Ecospace and SSC model outputs
     (e.g., Nemcek_matched_to_model_out_{ecospace_code}.csv)

Data Sources:
   - SalishSeaCast monthly/historical outputs (downloaded via ERDDAP)
     https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSg3DBiologyFields1moV19-05.html

   - Nemcek, N., Hennekes, M., Sastri, A., & Perry, R. I. (2023).
     "Seasonal and spatial dynamics of the phytoplankton community in the Salish Sea, 2015–2019."
     *Progress in Oceanography*, 217, 103108.
     https://doi.org/10.1016/j.pocean.2023.103108

Example Filenames:
   - SSCast model:  SalishSeaCast_biology_2008.nc
   - Nemcek data:   Nemcek_Supp_Data.csv
   - Ecospace model: Scv7-PARMixingNut90Temp_2003-2018.nc

Temporal Coverage:
   - SalishSeaCast:         2008–2018
   - Nemcek et al. (obs):   2015–2019
   - Ecospace model:        2003–2018 (limited to years with validated bloom timing)

Notes:
   - Matches are based on nearest lat/lon cell and closest timestamp (monthly for Ecospace, hourly/monthly for SSC).
   - Domain subregions ("SGI", "SGN", "SGS") are used for stratified analysis.
"""

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import requests
from matplotlib.path import Path
from helpers import (
    load_yaml,
    get_sscast_grd_idx,
    get_sscast_data,
    read_sdomains,
    find_nearest_point,
    find_nearest_point_from1D,
    find_closest_date,
    find_closest_depth
)
import ecospace_eval_config as cfg


# -------------------------------
# Configuration
# -------------------------------

# File names
file_Ecospace = cfg.NC_FILENAME
ecospace_code = cfg.ECOSPACE_SC


file_SSC_mo = cfg.SSC_MO_F
file_SSC_grd = cfg.SSC_GRD_F
file_Nemcek = cfg.NEMCEK_F
file_Nemcek_matched = cfg.NEMCEK_MATCHED_F
file_ecospace_map = cfg.ECOSPACE_MAP_F

# File paths
path_SSC = cfg.SSC_P
path_Ecospace = cfg.ECOSPACE_P
path_Nemcek = cfg.NEMCEK_P
path_evalout = cfg.EVALOUT_P
path_ecospace_map = cfg.ECOSPACE_MAP_P
domain_p = cfg.DOMAIN_P
domain_f = cfg.DOMAIN_F

output_file_path = cfg.NEMCEK_MATCHED_PF

# -------------------------------
# Utility Functions
# -------------------------------

def get_csv_metadata(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return {"Column Names": df.columns.tolist(), "Number of Rows": len(df)}, df
    return {"Error": "File not found."}, None

def get_ecospace_times(file_path):
    if not os.path.exists(file_path):
        return {"Error": "File not found."}, None
    try:
        with nc.Dataset(file_path, 'r') as ds:
            time_var = ds.variables['time']
            units = time_var.units
            calendar = time_var.calendar if 'calendar' in time_var.ncattrs() else 'standard'
            times = nc.num2date(time_var[:], units=units, calendar=calendar)
            return {
                "Dimensions": {dim: len(ds.dimensions[dim]) for dim in ds.dimensions},
                "Variables": list(ds.variables.keys())
            }, times
    except OSError as e:
        return {"Error": str(e)}, None

# -------------------------------
# Utility: ERDDAP Download for SSC
# -------------------------------

def download_ssc_griddap(obs_time, row, col, depth, save_path):
    formatted_dt = obs_time.strftime('%Y-%m-%dT%H:00')
    depth_str = f"{depth:.6f}"
    url = (
        f"https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSg3DBiologyFields1hV19-05.nc?"
        f"ciliates[({formatted_dt}Z):1:({formatted_dt}Z)][({depth_str}):1:({depth_str})][({row}):1:({row})][({col}):1:({col})],"
        f"diatoms[({formatted_dt}Z):1:({formatted_dt}Z)][({depth_str}):1:({depth_str})][({row}):1:({row})][({col}):1:({col})],"
        f"flagellates[({formatted_dt}Z):1:({formatted_dt}Z)][({depth_str}):1:({depth_str})][({row}):1:({row})][({col}):1:({col})]"
    )
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded ERDDAP SSC file: {save_path}")
        return True
    else:
        print(f"Failed to download SSC for {formatted_dt} (status code {response.status_code})")
        return False


def run_eval_2a_nemcek_match() -> None:

    # -------------------------------
    # Load and Inspect Data
    # -------------------------------

    # Paths
    ssc_file_path = os.path.join(path_SSC, file_SSC_mo)
    ecospace_file_path = os.path.join(path_Ecospace, file_Ecospace)
    nemcek_file_path = os.path.join(path_Nemcek, file_Nemcek)

    # Load data
    ssc_metadata, ssc_times, _, _, _ = get_sscast_data(ssc_file_path)
    ecospace_metadata, ecospace_times = get_ecospace_times(ecospace_file_path)
    nemcek_metadata, nemcek_df = get_csv_metadata(nemcek_file_path)

    # Display metadata
    print("SalishSeaCast Metadata:", ssc_metadata)
    print("Ecospace Metadata:", ecospace_metadata)
    print("Nemcek Metadata:", nemcek_metadata)

    # -------------------------------
    # Prepare for Matching
    # -------------------------------

    sdomains = read_sdomains(os.path.join(domain_p, domain_f))
    ecospace_coords_md, ecospace_coords_df = get_csv_metadata(os.path.join(path_ecospace_map, file_ecospace_map))
    model_lats = ecospace_coords_df['lat']
    model_lons = ecospace_coords_df['lon']
    model_rows = ecospace_coords_df['EWE_row']
    model_cols = ecospace_coords_df['EWE_col']
    model_deps = ecospace_coords_df['depth']
    mask_ecospace = model_deps != 0

    ssc_grid_metadata, ssc_lats, ssc_lons, ssc_gridY, ssc_gridX, ssc_grid_bathy = get_sscast_grd_idx(os.path.join(path_SSC, file_SSC_grd))
    mask_ssc = ssc_grid_bathy != 0

    nemcek_df['Date.Time'] = pd.to_datetime(nemcek_df['Date.Time'])
    nemcek_df['closest_ecospace_time'] = pd.to_datetime('1900-01-01')

    model_vars = ['PP1-DIA', 'PP2-NAN', 'PP3-PIC', 'PZ1-CIL', 'PZ2-DIN']
    ssc_vars = ['ssc-DIA', 'ssc-FLA', 'ssc-CIL']
    for var in model_vars + ssc_vars:
        nemcek_df[var] = 0.0

    nemcek_df['sdomain'] = ""
    nemcek_df['ecospace_closest_lat'] = 0.0
    nemcek_df['ecospace_closest_lon'] = 0.0

    # -------------------------------
    # Main Matching Loop
    # -------------------------------

    with nc.Dataset(ecospace_file_path, 'r') as eco_ds:
        for index, row in nemcek_df.iterrows():
            lat, lon, obs_time, obs_depth = row['Lat'], row['Lon'], row['Date.Time'], row['Pressure']

            # Assign subdomain
            for sd, polygon in sdomains.items():
                if Path(polygon).contains_point((lat, lon)):
                    nemcek_df.at[index, 'sdomain'] = sd
                    break

            # Find nearest Ecospace cell
            eco_idx = find_nearest_point_from1D(lon, lat, model_lons, model_lats, mask_ecospace, dx=0.01)
            if eco_idx[0] is np.nan:
                continue
            nemcek_df.at[index, 'ecospace_closest_lat'] = model_lats[eco_idx[0]]
            nemcek_df.at[index, 'ecospace_closest_lon'] = model_lons[eco_idx[0]]
            row_idx, col_idx = model_rows[eco_idx[0]], model_cols[eco_idx[0]]

            # Match Ecospace time
            closest_time = find_closest_date(ecospace_times, obs_time)
            time_idx = np.where(ecospace_times == closest_time)[0][0]
            nemcek_df.at[index, 'closest_ecospace_time'] = closest_time

            # Extract Ecospace values
            for var in model_vars:
                val = eco_ds.variables[var][time_idx, row_idx, col_idx]
                nemcek_df.at[index, var] = np.round(val, 3) if val is not None else np.nan

            # Match SSC grid cell and depth
            ssc_idx = find_nearest_point(lon, lat, ssc_lons, ssc_lats, mask_ssc, dx=0.005)
            if not ssc_idx:
                continue
            ssc_row, ssc_col = ssc_idx[1], ssc_idx[0]
            ssc_depths = [0.5, 1.5, 2.5, 3.5]
            ssc_depth = find_closest_depth(obs_depth, ssc_depths)

            # Build filename for SSC file
            filename_dt = obs_time.strftime('%Y-%m-%d')
            ssc_file = f"SalishSeaCast_biology_v1905_{filename_dt}.nc"
            ssc_path = os.path.join(path_SSC, ssc_file)

            if not os.path.exists(ssc_path):
                success = download_ssc_griddap(obs_time, ssc_row, ssc_col, ssc_depth, ssc_path)
                if not success:
                    continue

            ssc_meta, ssc_times, cil, fla, dia = get_sscast_data(ssc_path)

            if 'Error' in ssc_meta:
                continue

            nemcek_df.at[index, 'ssc-CIL'] = np.round(cil[0, 0, 0, 0], 3)
            nemcek_df.at[index, 'ssc-FLA'] = np.round(fla[0, 0, 0, 0], 3)
            nemcek_df.at[index, 'ssc-DIA'] = np.round(dia[0, 0, 0, 0], 3)

    # -------------------------------
    # Export and Summarize
    # -------------------------------


    nemcek_df.to_csv(output_file_path, index=False)

    print("Saved matched dataset to:", output_file_path)
    print(nemcek_df.head())
    print("Total observations matched:", len(nemcek_df))
    print(nemcek_df['sdomain'].value_counts())
    print("done")

if __name__ == "__main__":
    run_eval_2a_nemcek_match()