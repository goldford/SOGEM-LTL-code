'''
 Script: Ecospace Data Prep - Match Environmental Variables to Zooplankton Observations
Author: G. Oldford
Created: July 2025

Purpose:
  Match environmental variables (e.g., Temp_0to10m) from NetCDF outputs to zooplankton observation data.

Inputs:
  - Environmental NetCDF file (single variable), e.g., Temp_0to10m.nc
     (note these were created using shortcut script 6 (1b) in forcings data prep from ASC to NC!
  - Zooplankton observation CSV, e.g., Zooplankton_B_C_gm2_EWEMODELGRP_Wide.csv

Outputs:
  - CSV file combining zooplankton observations with matched environmental variable.
'''

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from helpers import find_nearest_point_from1D, find_closest_date

# -------------------------------
# Configuration
# -------------------------------
nc_p = "C:/Users/Greig/Sync/PSF/EwE/Georgia Strait 2021/LTL_model/LTL_MODELS/NEMO forcings/NC_3day/"
env_files = [
    "Temp_0to10m.nc",
    "Temp_150toBot.nc",
    "Temp_30to40m.nc"
]
# nc_f = "Temp_150toBot.nc" # "Temp_150toBot.nc", "Temp_0to10m.nc"
# nc_f = os.path.join(nc_p, nc_f)

zoop_p = "C:/Users/Greig/Sync/6. SSMSP Model/Model Greig/Data/4. Zooplankton/Zoop_Perryetal_2021/MODIFIED/"
zoop_f = "Zooplankton_B_C_gm2_EWEMODELGRP_Wide.csv"
zoop_f = os.path.join(zoop_p, zoop_f)

output_p = zoop_p
output_f = "Zooplankton_B_C_gm2_EWEMODELGRP_Wide_NEMO3daymatch.csv"
output_f = os.path.join(output_p, output_f)

# env_var_name = "Temp_150toBot"  # variable name in NetCDF, "Temp_150toBot", "Temp_0to10m"



# -------------------------------
# Load Zooplankton Data
# -------------------------------
zoop_df = pd.read_csv(zoop_f)
# Ensure there is a datetime column for matching
if 'Date' not in zoop_df.columns:
    zoop_df['Date'] = pd.to_datetime(
        dict(year=zoop_df['Year'], month=zoop_df['Month'], day=zoop_df.get('Day', 15))
    )


for env_file in env_files:
    nc_f = os.path.join(nc_p, env_file)
    var_name = os.path.splitext(env_file)[0]
    zoop_df[var_name] = np.nan


    # -------------------------------
    # Open NetCDF and Extract Data
    # -------------------------------
    with nc.Dataset(nc_f, 'r') as ds:
        # Print structure to inspect variables & dimensions
        print(ds)

        # Coordinate/mapping arrays
        lats = ds.variables['lat'][:]           # (row, col)
        lons = ds.variables['lon'][:]           # (row, col)
        rows_idx = ds.variables['EWE_row'][:]   # (row, col)
        cols_idx = ds.variables['EWE_col'][:]   # (row, col)

        # Time variable
        time_var = ds.variables['time']
        times = nc.num2date(time_var[:], units=time_var.units,
                            calendar=getattr(time_var, 'calendar', 'standard'))

        # Environmental variable of interest
        var_data = ds.variables[var_name]   # (time, row, col)

        # Flatten grid arrays for nearest‐neighbor search
        flat_lats = lats.flatten()
        flat_lons = lons.flatten()
        flat_rows = rows_idx.flatten().astype(int)
        flat_cols = cols_idx.flatten().astype(int)
        mask = ~np.isnan(flat_lats)

        # -------------------------------
        # Matching Loop
        # -------------------------------
        for idx, obs in zoop_df.iterrows():
            obs_lat = obs['Latitude.W.']
            obs_lon = obs['Longitude.N.']
            obs_date = obs['Date']

            # Find nearest grid cell index
            eco_idx = find_nearest_point_from1D(
                obs_lon, obs_lat, flat_lons, flat_lats, mask, dx=0.01
            )
            if np.isnan(eco_idx[0]):
                continue

            cell_row = flat_rows[eco_idx[0]]
            cell_col = flat_cols[eco_idx[0]]

            # Find closest time index
            closest_time = find_closest_date(times, obs_date)
            time_idx = np.where(times == closest_time)[0][0]

            # Extract and assign value
            val = var_data[time_idx, cell_row, cell_col]
            zoop_df.at[idx, var_name] = np.round(val, 3)

# -------------------------------
# Export Results
# -------------------------------

os.makedirs(output_p, exist_ok=True)
zoop_df.to_csv(output_f, index=False)
print(f"Saved matched output to: {output_f}")
