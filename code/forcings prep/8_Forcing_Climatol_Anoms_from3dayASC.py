"""
Convert forcings to climatol and anomalies
by: G Oldford, 2025-2026
----------------------------------------------------------
Simply takes the 3-day ASC files prepped previously 
and first computes cell-wise average from all years for 
each 3-day block. Then computes anomalies based on difference
from the average.

Inputs:
- Ecospace ASC files

Notes:
Ecospace limitation does not accept 
negative values by rescaling so normalized anomaly 
scaling centered around the climatological mean (value = 1), 
bounded between 0 and 2.
"""

import os
import numpy as np
from collections import defaultdict
from helpers import getDataFrame, buildSortableString
from GO_helpers import saveASCFile
import pandas as pd

# ----------------------------
# CONFIGURATION
# ----------------------------
DO_ANOMALIES = False
start_year = 1980
end_year = 2018
nsteps = 120  # 3-day time steps per year
grid_shape = (151, 93)  # Dimensions of trimmed ASC grids

# Input variable
var_key = "XLD_normalised" #"RDRS_windstress10m_" #vartemp1_C_0-10mAvg" "vartemp_avg150toBot" "RDRS_windstress_10m" "PAR-VarZ-VarK"

# in paths
forcing_root = "C:/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/forcing" # forcing other than wind
# forcing_root = "C:/Users/Greig/Sync/PSF/EwE/Georgia Strait 2021/LTL_model/LTL_MODELS/RDRS forcings/Wind_RDRS/Ecospace" # wind

# asc_dir = f"{forcing_root}/ECOSPACE_in_3day_PAR3_Sal4m_1980-2018/{var_key}/"
# asc_dir = f"{forcing_root}/stress_/" # wind
# asc_dir = f"{forcing_root}/ECOSPACE_in_3day_vars_1980-2018/{var_key}/"
asc_dir = f"{forcing_root}/ECOSPACE_in_3day_nutrld_fromASC_202601/"

sigdig = 3  # Decimal places to round
num_digits_ecospace_asc_month = 2
ASCheader = "ncols        93\nnrows        151 \nxllcorner    -123.80739\nyllcorner    48.29471 \ncellsize     0.013497347 \nNODATA_value  0.0"

# out paths
clim_output_npz = f"{forcing_root}/climatologies/{var_key}_climatology_3day_mean.npz"
clim_output_asc = f"{forcing_root}/climatologies/{var_key}_rescaledclima_3day.asc"
anomaly_out_dir = f"{forcing_root}/ECOSPACE_in_3day_vars_1980-2018/{var_key}/anomalies"

# Mask files
land_mask_file = f"{forcing_root}/../basemap/ecospacedepthgrid.asc"
plume_mask_file = f"{forcing_root}/../basemap/Fraser_Plume_Region.asc"

# ----------------------------
# LOAD MASKS
# ----------------------------
land_mask = getDataFrame(land_mask_file, "-9999.00000000") == 0
plume_mask = getDataFrame(plume_mask_file, "-9999.00000000") == 1
combined_mask = plume_mask | land_mask

# --------------------------------------
# CUSTOM DATA READER
# --------------------------------------
def getDataFrame_custom(f_n, skiprows=6):
    with open(f_n) as f:
        data = np.loadtxt(f, skiprows=skiprows)

    # Mask invalid values
    data[(data == -9999.0) | (data == 0.0)] = np.nan

    # Fix bottom-left corner issue
    data[140:, :15] = np.nan

    return data

# ----------------------------
# CLIMATOLOGY STACKING
# ----------------------------
clim_stack = defaultdict(list)

for year in range(start_year, end_year + 1):
    print(f"Reading year {year}")
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    for tstep in range(1, nsteps + 1):
        middle_day = (tstep - 1) * 3 + 2
        if tstep >= 120:
            middle_day += 2 if is_leap else 1

        fname = f"{var_key}_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
        fpath = os.path.join(asc_dir, fname)

        if not os.path.exists(fpath):
            if middle_day >= 360:
                middle_day = 363  # some vars have inconsistently named final day-of-year
                if var_key == "RDRS_windstress10m":  # correcting for a file naming glitch!
                    fname = f"{var_key}__{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
                else:
                    fname = f"{var_key}_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
                fpath = os.path.join(asc_dir, fname)
                if not os.path.exists(fpath):  # some files have the final time step named 363, 362, etc
                    print(f"Skipping missing file: {fpath}")
                    continue
            else:
                print(f"Skipping missing file: {fpath}")
                continue
        try:
            data = getDataFrame_custom(fpath)
            masked = np.ma.masked_array(data, mask=combined_mask)
            clim_stack[tstep].append(masked)
        except Exception as e:
            print(f"Failed to process {fpath}: {e}")
            continue

# ----------------------------
# COMPUTE CLIMATOLOGY
# ----------------------------
clim_mean = np.full((nsteps, *grid_shape), np.nan)
for t in range(1, nsteps + 1):
    if clim_stack[t]:
        stack = np.ma.stack(clim_stack[t])
        mean = stack.mean(axis=0)
        mean = mean.filled(np.nan)  # Keep NaNs where data is masked across all years
        clim_mean[t - 1] = np.round(mean, sigdig)
    else:
        print(f"No data for timestep {t}, leaving as NaN")

# ----------------------------
# SAVE OUTPUT
# ----------------------------
os.makedirs(os.path.dirname(clim_output_npz), exist_ok=True)
np.savez(clim_output_npz, clim_mean=clim_mean)
print(f"Saved climatological mean to {clim_output_npz}")

# ----------------------------
# Export 12 monthly ASC climas
# ----------------------------
# === Month to timestep index mapping ===
month_blocks = {
    "Jan": range(0, 10),
    "Feb": range(10, 20),
    "Mar": range(20, 30),
    "Apr": range(30, 40),
    "May": range(40, 50),
    "Jun": range(50, 60),
    "Jul": range(60, 70),
    "Aug": range(70, 80),
    "Sep": range(80, 90),
    "Oct": range(90, 100),
    "Nov": range(100, 111),  # 11 blocks for Nov
    "Dec": range(111, 120),
}

# === Load climatology if not in memory ===
clim = np.load(clim_output_npz)["clim_mean"]  # shape: [120, 151, 93]

# === Output path and ASC header ===
monthly_out_dir = f"{forcing_root}/climatologies/monthly_means"
os.makedirs(monthly_out_dir, exist_ok=True)

for month, indices in month_blocks.items():
    monthly_mean = np.nanmean(clim[list(indices), :, :], axis=0)

    # Optional: round and re-mask if needed
    monthly_mean = np.round(monthly_mean, sigdig)
    monthly_mean[np.isnan(monthly_mean)] = 0.0  # or retain NaN for masked cells

    out_path = os.path.join(monthly_out_dir, f"{var_key}_monthly_mean_{month}.asc")
    np.savetxt(out_path, monthly_mean, fmt='%0.3f', delimiter=" ", comments='', header=ASCheader)
    print(f"Saved {month} to {out_path}")



if not DO_ANOMALIES:
    exit()

exit()

# ----------------------------
# LOAD CLIMATOLOGY AND MASKS
# ----------------------------
clim_mean = np.load(clim_output_npz)["clim_mean"]

# ----------------------------
# PASS 1: Find Global Min/Max Anomaly
# ----------------------------
global_min = np.inf
global_max = -np.inf


# ----------------------------
# LOAD CLIMATOLOGY
# ----------------------------
clim_mean = np.load(clim_output_npz)["clim_mean"]

# ----------------------------
# PASS 1: Find Global Min/Max Anomaly
# ----------------------------
global_min = np.inf
global_max = -np.inf

print("Pass 1: Scanning anomalies to find global min/max...")

for year in range(start_year, end_year + 1):
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    for tstep in range(1, nsteps + 1):
        middle_day = (tstep - 1) * 3 + 2
        if tstep >= 120:
            middle_day += 2 if is_leap else 1

        fname = f"{var_key}_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
        fpath = os.path.join(asc_dir, fname)

        if not os.path.exists(fpath):
            middle_day = 363  # try fallback day
            fname = f"{var_key}_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
            fpath = os.path.join(asc_dir, fname)
            if not os.path.exists(fpath):
                continue

        try:
            data = getDataFrame_custom(fpath)
            masked = np.ma.masked_array(data, mask=combined_mask)
            anomaly = masked - clim_mean[tstep - 1]
            global_min = min(global_min, np.nanmin(anomaly))
            global_max = max(global_max, np.nanmax(anomaly))
        except Exception as e:
            print(f"Failed {fpath}: {e}")
            continue

scale = max(abs(global_min), abs(global_max))
print(f"Global anomaly range: min={global_min:.3f}, max={global_max:.3f}, scale={scale:.3f}")

# ----------------------------
# PASS 2: Rescale and Save Anomalies
# ----------------------------
os.makedirs(anomaly_out_dir, exist_ok=True)
print("Pass 2: Rescaling and saving anomalies...")

for year in range(start_year, end_year + 1):
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    for tstep in range(1, nsteps + 1):
        middle_day = (tstep - 1) * 3 + 2
        if tstep >= 120:
            middle_day += 2 if is_leap else 1

        fname = f"{var_key}_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
        fpath = os.path.join(asc_dir, fname)

        if not os.path.exists(fpath):
            middle_day = 363
            fname = f"{var_key}_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
            fpath = os.path.join(asc_dir, fname)
            if not os.path.exists(fpath):
                continue

        # helps with export of a single annualised mean anomaly (rescaled) to use for spin-up
        # in first 12 time steps
        rescaled_sum = np.zeros(grid_shape)
        rescaled_count = np.zeros(grid_shape)

        try:
            data = getDataFrame_custom(fpath)
            masked = np.ma.masked_array(data, mask=combined_mask)
            anomaly = masked - clim_mean[tstep - 1]

            # Rescale
            scaled = 1 + (anomaly / scale)
            scaled = np.clip(scaled, 0, 2)
            scaled_out = scaled.filled(np.nan)
            scaled_out[np.isnan(masked)] = np.nan

            # using this for the climatological output
            valid_mask = ~scaled.mask
            rescaled_sum[valid_mask] += scaled.data[valid_mask]
            rescaled_count[valid_mask] += 1

            out_name = f"{var_key}_scaled_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
            out_path = os.path.join(anomaly_out_dir, out_name)
            np.savetxt(out_path, np.nan_to_num(scaled_out, nan=0.0), fmt='%0.3f',
                       delimiter=" ", comments='', header=ASCheader)
        except Exception as e:
            print(f"Failed to save {fpath}: {e}")
            continue

# write the mean, annualised, re-scaled anomaly (should be mostly 1's)
mean_rescaled = np.full(grid_shape, np.nan)
valid_cells = rescaled_count > 0
mean_rescaled[valid_cells] = rescaled_sum[valid_cells] / rescaled_count[valid_cells]

# Save to ASC file
np.savetxt(clim_output_asc, np.nan_to_num(mean_rescaled, nan=0.0), fmt='%0.3f',
           delimiter=" ", comments='', header=ASCheader)
print(f"Saved mean annualised  rescaled anomaly to: {clim_output_asc}")


#
#
# # ----------------------------
# # COMPUTE ANOMALIES
# # ----------------------------
# for year in range(start_year, end_year + 1):
#     print(f"Processing anomalies for year {year}")
#     is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
#
#     for tstep in range(1, nsteps + 1):
#         middle_day = (tstep - 1) * 3 + 2
#         if tstep >= 120:
#             middle_day += 2 if is_leap else 1
#
#         fname = f"{var_key}_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
#         fpath = os.path.join(asc_dir, fname)
#
#         if not os.path.exists(fpath):
#             if middle_day >= 360:
#                 middle_day = 363  # some vars have inconsistently named final day-of-year
#                 if var_key == "RDRS_windstress10m":  # correcting for a file naming glitch!
#                     fname = f"{var_key}__{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
#                 else:
#                     fname = f"{var_key}_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
#                 fpath = os.path.join(asc_dir, fname)
#                 if not os.path.exists(fpath):  # some files have the final time step named 363, 362, etc
#                     print(f"Skipping missing file: {fpath}")
#                     continue
#             else:
#                 print(f"Skipping missing file: {fpath}")
#                 continue
#
#         try:
#             data = getDataFrame_custom(fpath)
#             masked = np.ma.masked_array(data, mask=combined_mask)
#
#             clim = clim_mean[tstep - 1]
#             anomaly = masked - clim
#
#             # Calculate scaling factor: symmetric max deviation
#             max_anom = np.nanmax(anomaly)
#             min_anom = np.nanmin(anomaly)
#             scale_range = max(abs(min_anom), abs(max_anom))
#
#             if scale_range == 0 or np.isnan(scale_range):
#                 print(f"Flat anomaly at {year}-{tstep}, setting to 1")
#                 scaled = np.ones_like(clim)
#             else:
#                 scaled = 1 + (anomaly / scale_range)
#                 scaled = np.clip(scaled, 0, 2)
#
#             # Save anomaly as .npz or .npy
#             # Final cleanup for output
#             scaled_out = scaled.filled(np.nan)
#             scaled_out[np.isnan(masked)] = np.nan # re-apply mask explicitly
#             out_name = f"{var_key}_anom_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
#             scaled_z = np.nan_to_num(scaled_out, nan=0.0)
#             np.savetxt(os.path.join(anomaly_out_dir, out_name), scaled_z, fmt='%0.3f',
#                        delimiter=" ", comments='', header=ASCheader)
#
#         except Exception as e:
#             print(f"Failed to process {fpath}: {e}")
#             continue
#
#
# # ----------------------------
# # CREATE OUTPUT DIR
# # ----------------------------
# os.makedirs(anomaly_out_dir, exist_ok=True)