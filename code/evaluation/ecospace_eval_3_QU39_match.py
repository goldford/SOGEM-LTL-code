"""
Script Name: Ecospace Data Prep - Match QU39 Observations to Models
Author: G. Oldford
Last Updated: May 2025

Description:
  This script matches plankton observations at station QU39 (del Bel Belluz)
  with modeled outputs from:
    1. Ecospace (EwE)
    2. SalishSeaCast (SSC)
  It extracts the closest grid cell (in space, time, and depth) and
  appends relevant model variables to the observational dataset.

Data Sources:
  - Observations:
    Del Bel Belluz, J. (2024). Protistan plankton time series from
    the northern Salish Sea and central coast, BC, Canada.
    OBIS Dataset: https://obis.org/dataset/071a38b3-0d30-47ba-85aa-d98313460075

  - Models:
    Ecospace outputs from ECOSPACE_OUT directory
    SalishSeaCast v19-05 hourly biology fields (ERDDAP access)

Inputs:
  - QU39_joined.csv: Observation data
  - NetCDF files: Ecospace and SSC output
  - Grid metadata (CSV, bathymetry)

Outputs:
  - CSV with matched QU39 observations and model data

Notes:
  - Switches allow toggling SSC download, SSC matching, and Ecospace matching
"""

# ----- Import libraries -----
from helpers import (
    get_sscast_grd_idx,
    find_nearest_point,
    get_sscast_data,
    find_closest_depth,
    find_nearest_point_from1D,
    find_closest_date
)
import os
import netCDF4 as nc
import pandas as pd
import numpy as np
# import requests
from datetime import datetime
import ecospace_eval_config as cfg


ECOSPACE_SCEN = cfg.ECOSPACE_SC
file_ecospace_out = cfg.NC_FILENAME
file_ecospace_map = cfg.ECOSPACE_MAP_F

do_ecospace_matching = cfg.QU39_DO_ECOSPACE_MATCHING       # Perform Ecospace matching
download_ssc = cfg.QU39_DOWNLOAD_SSC              # Download SSC ERDDAP files
match_ssc = cfg.QU39_MATCH_SSC                  # Match existing SSC files

pathfile_QU39_in = cfg.QU39_IN_P
pathfile_QU39_out_p = cfg.QU39_OUT_P
matched_QU39_out_f = cfg.QU39_MATCHED_OUT_F

path_SSC = cfg.SSC_P
path_SSC_dwnld = os.path.join(path_SSC, 'matched_QU39_dwnld')

path_ecospace_out = cfg.NC_PATH_OUT
path_ecospace_map = cfg.ECOSPACE_MAP_P

file_SSC_bathy = cfg.SSC_BATHY_F
file_SSC_meshm_3D = cfg.SSC_MESH3D_F
file_SSC_meshm_2D = cfg.SSC_MESHM_2D_F

QU39_lat = cfg.QU39_LAT
QU39_lon = cfg.QU39_lON

initial_datetime_placeholder = cfg.QU39_INIT_DT_PLACEHOLDER
fill_value = cfg.QU39_FILL_VAL


# ------------------------------------------------------ #
def match_ecospace_to_QU39(df: pd.DataFrame) -> pd.DataFrame:
    """Match Ecospace model outputs to each observation in the QU39 dataframe."""
    print("Starting Ecospace matching...")

    ecospace_coords_df = pd.read_csv(os.path.join(path_ecospace_map, file_ecospace_map))
    ecospace_lats = ecospace_coords_df['lat']
    ecospace_lons = ecospace_coords_df['lon']
    ecospace_rows = ecospace_coords_df['EWE_row']
    ecospace_cols = ecospace_coords_df['EWE_col']
    ecospace_deps = ecospace_coords_df['depth']
    mask_ecospace = ecospace_deps != 0
    dx = 0.01  # ~1.1 km grid spacing

    with nc.Dataset(os.path.join(path_ecospace_out, file_ecospace_out), 'r') as dataset:
        time_var = dataset.variables['time']
        ecospace_times = nc.num2date(
            time_var[:],
            units=time_var.units,
            calendar=time_var.calendar if 'calendar' in time_var.ncattrs() else 'standard'
        )

        for index, row in df.iterrows():
            # Find closest Ecospace grid cell
            p = find_nearest_point_from1D(QU39_lon, QU39_lat, ecospace_lons, ecospace_lats, mask_ecospace, dx)
            nearest_idx = p[0]

            if not np.isnan(nearest_idx):
                df.at[index, 'ecospace_closest_lat'] = ecospace_lats[nearest_idx]
                df.at[index, 'ecospace_closest_lon'] = ecospace_lons[nearest_idx]
                df.at[index, 'ecospace_closest_gridY'] = ecospace_rows[nearest_idx]
                df.at[index, 'ecospace_closest_gridX'] = ecospace_cols[nearest_idx]
                row_idx = ecospace_rows[nearest_idx]
                col_idx = ecospace_cols[nearest_idx]
            else:
                df.loc[index, ['ecospace_closest_lat', 'ecospace_closest_lon',
                               'ecospace_closest_gridY', 'ecospace_closest_gridX']] = fill_value
                row_idx = col_idx = -999

            # Match to closest model time
            obs_time = row['DateTime']
            if obs_time.year <= 2018:
                closest_time = find_closest_date(ecospace_times, obs_time)
                df.at[index, 'closest_ecospace_time'] = closest_time

                if row_idx != -999 and col_idx != -999:
                    time_idx = np.where(ecospace_times == closest_time)[0][0]
                    df.at[index, 'PP1-DIA'] = dataset['PP1-DIA'][time_idx, row_idx, col_idx]
                    df.at[index, 'PP2-NAN'] = dataset['PP2-NAN'][time_idx, row_idx, col_idx]
                    df.at[index, 'PP3-PIC'] = dataset['PP3-PIC'][time_idx, row_idx, col_idx]
                    df.at[index, 'PZ1-CIL'] = dataset['PZ1-CIL'][time_idx, row_idx, col_idx]
                    df.at[index, 'PZ2-DIN'] = dataset['PZ2-DIN'][time_idx, row_idx, col_idx]

    out_file = os.path.join(pathfile_QU39_out_p, f"{matched_QU39_out_f}_{ECOSPACE_SCEN}.csv")
    df.to_csv(out_file, index=False)
    print(f"Ecospace matching complete. Output written to {out_file}")
    return df


def match_ssc_to_QU39(df: pd.DataFrame) -> pd.DataFrame:
    """Match SSC model outputs to each observation in the QU39 dataframe."""
    print("Starting SSC matching...")

    # Get SSC grid and bathymetry info
    ssc_grid_metadata, ssc_lats, ssc_lons, ssc_gridY, ssc_gridX, ssc_grid_bathy = get_sscast_grd_idx(
        os.path.join(path_SSC, file_SSC_bathy)
    )
    mask_ssc = ssc_grid_bathy != 0
    dx = 0.005  # ~500m resolution

    # Closest SSC grid cell (static for QU39)
    p = find_nearest_point(QU39_lon, QU39_lat, ssc_lons, ssc_lats, mask_ssc, dx)
    ssc_col, ssc_row = p[0], p[1]
    ssc_lat, ssc_lon = ssc_lats[ssc_row, ssc_col], ssc_lons[ssc_row, ssc_col]

    # Possible SSC depths from mesh mask
    with nc.Dataset(os.path.join(path_SSC, file_SSC_meshm_3D)) as mesh:
        gdept_0 = mesh.variables['gdept_0'][:]
    ssc_possible_depths = gdept_0[:, :, ssc_row, ssc_col]

    for index, row in df.iterrows():
        obs_depth = (row['minimumDepthInMeters'] + row['maximumDepthInMeters']) / 2
        ssc_depth = find_closest_depth(obs_depth, ssc_possible_depths[0][:])
        ssc_filename_dt = row['DateTime'].strftime('%Y-%m-%d')
        ssc_filename = os.path.join(path_SSC_dwnld, f"SalishSeaCast_biology_v1905_{ssc_filename_dt}-{obs_depth}.nc")

        ssc_meta, ssc_times, ssc_ciliates, ssc_flagellates, ssc_diatoms = get_sscast_data(ssc_filename)

        if 'Error' in ssc_meta:
            print(f"[Warning] Error with file: {ssc_filename}")
            continue

        df.at[index, 'ssc-DIA'] = ssc_diatoms[0, 0, 0, 0]
        df.at[index, 'ssc-FLA'] = ssc_flagellates[0, 0, 0, 0]
        df.at[index, 'ssc-CIL'] = ssc_ciliates[0, 0, 0, 0]
        df.at[index, 'ssc_closest_gridY'] = ssc_row
        df.at[index, 'ssc_closest_gridX'] = ssc_col
        df.at[index, 'ssc_closest_lat'] = ssc_lat
        df.at[index, 'ssc_closest_lon'] = ssc_lon
        df.at[index, 'closest_ssc_time'] = datetime(*ssc_times[0].timetuple()[:6])

    out_file = os.path.join(pathfile_QU39_out_p, f"{matched_QU39_out_f}_SSC_{ECOSPACE_SCEN}.csv")
    df.to_csv(out_file, index=False)
    print(f"SSC matching complete. Output written to {out_file}")
    return df


def run_qu39_match() -> None:
    # Load input observation data
    print("Loading observational data...")
    df = pd.read_csv(pathfile_QU39_in)
    df['eventDate'] = df['eventDate'].astype(str)
    df['eventTime'] = df['eventTime'].astype(str)
    df['DateTime'] = pd.to_datetime(df['eventDate'] + ' ' + df['eventTime'], format='%Y-%m-%d %H:%M:%SZ')

    # Initialize result columns
    init_fields = {
        'PP1-DIA': fill_value, 'PP2-NAN': fill_value, 'PP3-PIC': fill_value,
        'PZ1-CIL': fill_value, 'PZ2-DIN': fill_value,
        'ecospace_closest_lat': 0.0, 'ecospace_closest_lon': 0.0,
        'ecospace_closest_gridY': 0.0, 'ecospace_closest_gridX': 0.0,
        'ecospace_closest_dep': 0.0, 'closest_ecospace_time': initial_datetime_placeholder,
        'ssc-DIA': fill_value, 'ssc-FLA': fill_value, 'ssc-CIL': fill_value,
        'ssc_closest_lat': 0.0, 'ssc_closest_lon': 0.0,
        'ssc_closest_gridY': 0.0, 'ssc_closest_gridX': 0.0,
        'closest_ssc_time': initial_datetime_placeholder
    }
    for field, val in init_fields.items():
        df[field] = val

    # Run model-data matching
    if do_ecospace_matching:
        df = match_ecospace_to_QU39(df)

    if match_ssc:
        df = match_ssc_to_QU39(df)

    print("All matching complete - done.")

if __name__ == "__main__":
    run_qu39_match()