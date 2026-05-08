
# going 1d NC -> 3 day ASC -> 3 day NC
# it is what it is for now

import os
import pandas as pd
import xarray as xr
import numpy as np


# --- User Configuration ---
NEMO_ASCs_ROOT = "C:/Users/Greig/Documents/github/Ecosystem-Model-Data-Framework/data/forcing/"
NEMO_NC_ROOT = "C:/Users/Greig/Sync/PSF/EwE/Georgia Strait 2021/LTL_model/LTL_MODELS/NEMO forcings/"
RDRS_NC_ROOT = "C:/Users/Greig/Sync/PSF/EwE/Georgia Strait 2021/LTL_model/LTL_MODELS/RDRS forcings/Wind_RDRS/Ecospace/"
REGIONS_ASC = "C:/Users/Greig/Documents/github/Ecosystem-Model-Data-Framework/data/basemap/ecospace_regions_3day.asc"
FIG_OUT_PATH = "..//..//figs//"

BLOOM_SOURCE = "suchy"  # suchy or C09

# --- ASC PROCESSING IF REQUIRED ---
# takes ASC (3 day) converts to NC - only need done once

process_asc = True

year_start = 1980
year_end = 2018
# REGION_ID = 2  # Central SoG (this method can reduce size of NC's)

# IS this used??
#variables = {
    # "PAR": os.path.join(NEMO_NC_ROOT, "PAR" + f"_region{REGION_ID}.nc"),
    # "MixingZ": os.path.join(NEMO_NC_ROOT, "MixingZ" + f"_region{REGION_ID}.nc"),
    # "Wind_Stress_10m_RDRS": os.path.join(RDRS_NC_ROOT, "Wind_Stress_10m_RDRS" + f"_region{REGION_ID}.nc"),
    # "Temp_0to10m": os.path.join(NEMO_NC_ROOT, "Temp_0to10m" + f"_region{REGION_ID}.nc"),
    # "Temp_at150m": os.path.join(NEMO_NC_ROOT, "Temp_at150m" + f"_region{REGION_ID}.nc"),
    # "Temp_150toBot": os.path.join(NEMO_NC_ROOT, "Temp_150toBot" + f"_region{REGION_ID}.nc"),

    #"Temp_30to40m": os.path.join(NEMO_NC_ROOT, "Temp_30to40m" + f"_region{REGION_ID}.nc") # 2025-05 - the 30 to 40 source data in asc is bad - errors

#}

skiprows = 6  # for header
path_NEMO_ASCs_root = "C:/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/forcing/"
# after processing to NC they are too large for github, so moved here
path_NEMO_NCs_root = "C:/Users/Greig/Sync/PSF/EwE/Georgia Strait 2021/LTL_model/LTL_MODELS/NEMO forcings/NC_3day/"
path_RDRS_ASCs_root = "C:/Users/Greig/Sync/PSF/EwE/Georgia Strait 2021/LTL_model/LTL_MODELS/RDRS forcings/Wind_RDRS/Ecospace/NC_3day/"
path_RDRS_NCs_root = "C:/Users/Greig/Sync/PSF/EwE/Georgia Strait 2021/LTL_model/LTL_MODELS/RDRS forcings/Wind_RDRS/Ecospace/NC_3day/"

subpath_NEMO_ASCs_PAR = f"/ECOSPACE_in_3day_PAR3_Sal4m_{year_start}-{year_end}/PAR-VarZ-VarK"
subpath_NEMO_ASCs_vars = f"/ECOSPACE_in_3day_vars_{year_start}-{year_end}/ECOSPACE_in_3day_vars"
subfolder_NEMO_PAR = "/PAR-VarZ-VarK"
subfolder_NEMO_PARxMixing = "/RUN216_PARxMixing"
subfolder_NEMO_temp0to10m = "/vartemp1_C_0-10mAvg"
subfolder_NEMO_tempat150m = "/vartemp_at150m"
subfolder_NEMO_temp150toBot = "/vartemp_avg150toBot"
subfolder_NEMO_temp30to40m = "/vartemp3_C_30-40mAvg"
subfolder_NEMO_mixing = "/varmixing_m"


subfolders = {
    # "PAR": path_NEMO_ASCs_root + f"ECOSPACE_in_3day_PAR3_Sal4m_{year_start}-{year_end}/PAR-VarZ-VarK",
    # "PARxMixing": path_NEMO_ASCs_root + f"ECOSPACE_in_3day_PAR3_Sal4m_{year_start}-{year_end}/RUN216_PARxMixing",
    # "MixingZ": path_NEMO_ASCs_root + f"ECOSPACE_in_3day_vars_{year_start}-{year_end}/varmixing_m",
    # "Wind_Stress_10m_RDRS": path_RDRS_ASCs_root + "stress_",
    # "Wind_Speed_10m_RDRS": path_RDRS_ASCs_root + "speed_",
    # "Temp_0to10m": f"ECOSPACE_in_3day_vars_{year_start}-{year_end}/vartemp1_C_0-10mAvg",
    # "Salt_0to4m": f"ECOSPACE_in_3day_vars_{year_start}-{year_end}/varsalt2_PSU_0-4m",
    # "Temp_0to4m": f"ECOSPACE_in_3day_vars_{year_start}-{year_end}/vartemp2_C_0-4m",
    # "Temp_at150m": f"ECOSPACE_in_3day_vars_{year_start}-{year_end}/{subfolder_NEMO_tempat150m}",
    "Temp_150toBot": f"ECOSPACE_in_3day_vars_{year_start}-{year_end}/{subfolder_NEMO_temp150toBot}",
    # "Temp_30to40m": f"ECOSPACE_in_3day_vars_{year_start}-{year_end}/vartemp3_C_30-40mAvg"
}

ASC_file_fmts = {"PAR": "PAR-VarZ-VarK_{}_{}.asc",  # month, doy
                 "PARxMixing": "RUN216_PARxMixing_{}_{}.asc",
                 "MixingZ": "varmixing_m_{}_{}.asc",
                 "Wind_Stress_10m_RDRS": "RDRS_windstress10m_{}_{}.asc",
                 "Wind_Speed_10m_RDRS": "RDRS_windspeed10m_{}_{}.asc",
                 "Temp_0to10m": "vartemp1_C_0-10mAvg_{}_{}.asc",
                 "Salt_0to4m": "varsalt2_PSU_0-4m_{}_{}.asc",
                 "Temp_0to4m": "vartemp2_C_0-4mAvg_{}_{}.asc",
                 "Temp_at150m": "vartemp_at150m_{}_{}.asc",
                 "Temp_30to40m": "vartemp3_C_30-40mAvg_{}_{}.asc"
                 }

path_regions_asc = "C:/Users/Greig/Documents/github/Ecosystem-Model-Data-Framework/data/basemap"
file_regions_asc = "ecospace_regions_3day.asc"
path_ecospace_map = "C:/Users/Greig/Documents/github/Ecosystem-Model-Data-Framework/data/basemap"
file_ecospace_map = "Ecospace_grid_20210208_rowscols.csv"

if process_asc:
    # ECOSPACE MODEL domain map with lats / lons
    # OID,lat,lon,depth,NEMO_col,NEMO_row,EWE_row,EWE_col
    file_path = os.path.join(path_ecospace_map, file_ecospace_map)
    if os.path.exists(file_path):
        ecospace_map_df = pd.read_csv(file_path)
        metadata = {
            "Column Names": ecospace_map_df.columns.tolist(),
            "Number of Rows": len(ecospace_map_df)
        }
        lats = ecospace_map_df['lat']
        lons = ecospace_map_df['lon']
    else:
        print("Error: File not found.")

    # ECOSPACE regions file prep
    file_path = os.path.join(path_regions_asc, file_regions_asc)
    skiprows = 6
    with open(file_path) as f:
        dat_regions = np.loadtxt(f, skiprows=skiprows)

    rows = dat_regions.shape[0]
    cols = dat_regions.shape[1]
    print(rows, cols)

    # Loop over subfolders and files
    for var, subfolder in subfolders.items():

        ####################################
        if var == 'PARxMixing' or var == 'PAR':
            continue
        # ####################################
        print(var)
        timestamps = []
        file_paths = []
        folder_path = os.path.join(path_NEMO_ASCs_root, subfolder)
        print(folder_path)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.asc'):
                # Extract year and doy from file name
                parts = file_name.split('_')
                if "really_clima" in file_name or "butreally" in file_name:
                    year = int(parts[-5])
                    doy = int(parts[-4].split('.')[0])
                elif "reallyclima" in file_name:
                    year = int(parts[-4])
                    doy = int(parts[-3].split('.')[0])
                else:
                    year = int(parts[-2])
                    doy = int(parts[-1].split('.')[0])

                # Create timestamp
                timestamp = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy - 1)
                timestamps.append(timestamp)

                # Store full file path
                file_paths.append(os.path.join(folder_path, file_name))

        # Sort timestamps and corresponding file paths
        timestamps, file_paths = zip(*sorted(zip(timestamps, file_paths)))

        # Create an empty xarray dataset
        ds = xr.Dataset(
            coords={
                'time': list(timestamps),
                'row': range(1, rows + 1),
                'col': range(1, cols + 1)
            },
            attrs={'description': 'dataset of monthly ASC files'}
        )

        # Add lat, lon, depth, EWE_col, EWE_row as data variables
        ds['lat'] = (('row', 'col'), np.full((rows, cols), np.nan))
        ds['lon'] = (('row', 'col'), np.full((rows, cols), np.nan))
        ds['depth'] = (('row', 'col'), np.full((rows, cols), np.nan))
        ds['EWE_col'] = (('row', 'col'), np.full((rows, cols), np.nan))
        ds['EWE_row'] = (('row', 'col'), np.full((rows, cols), np.nan))

        # Populate the dataset with lat, lon, depth, NEMO_col, NEMO_row from the CSV file
        for _, row in ecospace_map_df.iterrows():
            ewe_row = int(row['EWE_row']) - 1
            ewe_col = int(row['EWE_col']) - 1
            ds['lat'][ewe_row, ewe_col] = row['lat']
            ds['lon'][ewe_row, ewe_col] = row['lon']
            ds['depth'][ewe_row, ewe_col] = row['depth']
            ds['EWE_col'][ewe_row, ewe_col] = row['EWE_col']
            ds['EWE_row'][ewe_row, ewe_col] = row['EWE_row']

        ds[var] = (('time', 'row', 'col'), np.full((len(timestamps), rows, cols), np.nan))

        # Read and load data from each ASC file into the dataset
        for t, file_path in enumerate(file_paths):
            parts = file_path.split('_')
            if "really_clima" in file_path or "butreally" in file_path:
                year = int(parts[-5])
                doy = int(parts[-4].split('.')[0])
            elif "reallyclima" in file_path:
                year = int(parts[-4])
                doy = int(parts[-3].split('.')[0])
            else:
                year = int(parts[-2])
                doy = int(parts[-1].split('.')[0])

            with open(file_path) as f:
                data = np.loadtxt(f, skiprows=skiprows)
                ds[var][t - 1, :, :] = data

        # Save the dataset to a file
        if (var != "Wind_Stress_10m_RDRS") & (var != "Wind_Speed_10m_RDRS"):
            # file_name_out = f'{var}_region{REGION_ID}.nc'
            file_name_out = f'{var}.nc'
            ds.to_netcdf(os.path.join(path_NEMO_NCs_root, file_name_out))
            print(f"Saved {var}.nc to {path_NEMO_NCs_root}")
        else:
            # file_name_out = f'{var}_region{REGION_ID}.nc'
            file_name_out = f'{var}.nc'
            ds.to_netcdf(os.path.join(path_RDRS_NCs_root, file_name_out))
            print(f"Saved {var}.nc to {path_RDRS_NCs_root}")
        print(f"finished {var}")

