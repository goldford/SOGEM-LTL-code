
"""
By G Oldford 2023-26

Purpose
-------
This script processes daily RDRS v2.1 10 m wind fields (u10m, v10m) on the NEMO grid
and aggregates them into 3-day wind-derived products. While originally written
to support both Ecospace spatial forcing and Ecosim time series forcing, its
primary current use is to generate **nutrient forcing proxy time series for Ecosim**.

Core Functionality
------------------
1. Read daily RDRS wind fields (u10m, v10m) and compute wind speed.
2. Aggregate daily winds into consecutive 3-day blocks.
3. Derive wind-based stress metrics:
   - Wind speed (mean or max over each 3-day block)
   - Wind stress proxy (wind speed squared)
   - Nutrient-loading wind stress proxy, defined as the mean of daily wind stress
     over each 3-day block
4. Optionally override wind forcing outside specified active months.
5. Compute spatial means over the Ecospace domain to produce **3-day time series
   suitable for Ecosim forcing**.
6. Optionally prepend climatological spin-up years to the Ecosim time series.
7. Optionally compute and write long-term climatological wind and stress rasters.

Intended Use (Current)
----------------------
The primary intended use of this script is to generate **Ecosim nutrient forcing
time series**, where wind-driven mixing is represented as a proxy for nutrient
input or vertical exchange. Ecospace raster outputs and climatology products are
legacy or secondary outputs and may be disabled via configuration flags.

Notes
-----
- Wind stress is treated as a proxy forcing variable rather than a physical
  momentum flux.
- Nutrient forcing is based on spatially averaged wind stress over the Ecospace
  domain.
- This script does not explicitly model nutrients; it produces a wind-derived
  proxy intended for use as an external forcing function in Ecosim.

History
-------
Originally developed to support both Ecospace and Ecosim wind forcing products.
Over time, usage has converged on Ecosim nutrient forcing applications, and the
script structure reflects this evolution.
"""

import os
import numpy as np
import pandas as pd
from helpers import getDataFrame, buildSortableString


# --------------------------------------
# CONFIGURATION
# --------------------------------------
startyear = 1980
endyear = 2018
timestep = "3day"
out_code = "var"
# wind already done
timestep = "3day"
out_code = "var"
anomalies = False
var_keys = [
    #"vartemp_at150m",
    #"vartemp_avg150toBot"#,
    # "vartemp_avg150toBot"#,
    # "RDRS_windstress10m",
    # "varsalt1_PSU_0-10mAvg",
    # "vartemp1_C_0-10mAvg",
    # "PAR-VarZ-VarK",
    "varmixing_m"
]

# target percentile for XLD (nutrient forcing)
# this is NOT WORKING - see jote GO 2025-07-29
DO_PERCENTILE = False
percentile = 'filter' # this is not a percentile anymore
stat_label = f"p{percentile}"

# ------------------------------------------
#  nutrient‑multiplier relevant to XLD
# -------------------------------------------

# SC30 - 2, True, 3, 25, False, True, 1, True, 15
# good bloom t fit (0.864) but not good nutrient fit
# ecosim nut 0.95, dia nudged up to 12.1 opt and 6 tmin, 16 tmax

STD_THRESHOLD = 2# 2 good
DO_FILTER_STATIC = True
sigdig = 3
CAP_HIGH = 22 # metres # 20 ok
USE_LOG_STRETCH = False # false ok wss 0.864 (nut 0.93)
USE_PWR_EXPNSN = True
PWR_EXP = 1
# CAP_LOW = 0.4
USE_MOVING_AVG = True
WINDOW_DAYS = 15


NEMO_run = "216"
sigdig = 1  # number of decimal places to round
num_digits_ecospace_asc_month = 2

# --------------------------------------
# FILE PATHS
# --------------------------------------
forcing_root = "C:/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/forcing"
forcing_root_RDRS = "C:/Users/Greig/Sync/PSF/EwE/Georgia Strait 2021/LTL_model/LTL_MODELS/RDRS forcings"


# Mask files (must be 151x93 trimmed ASC format used by saveASCFile)
land_mask_file = "C:/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/basemap/ecospacedepthgrid.asc"
plume_mask_file = "C:/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/basemap/Fraser_Plume_Region.asc"


plume_mask = getDataFrame(plume_mask_file, "-9999.00000000") == 1
land_mask = getDataFrame(land_mask_file, "-9999.00000000") == 0
combined_mask = plume_mask | land_mask


_static_varmask = None  # filled on first use


# CUSTOM DATA READER
def getDataFrame_custom(f_n, skiprows=6):
    with open(f_n) as f:
        data = np.loadtxt(f, skiprows=skiprows)

    # Mask invalid values
    data[(data == -9999.0) | (data == 0.0)] = np.nan

    # Fix bottom-left corner issue
    data[140:, :15] = np.nan

    return data

def build_static_varmask(var_key:str, asc_dir:str, mask:np.ndarray):
    """Return boolean mask of cells with σ(xld) < STD_THRESHOLD (tidally mixed).
    Only a *sample* of files is loaded (every Nth year & first 20 time‑steps)
    for speed; adequate because static cells barely vary in any year.
    """
    global _static_varmask
    if _static_varmask is not None:
        return _static_varmask

    sample_stack = []
    for yr in range(startyear, endyear+1, 5):        # every 5th year
        for day in (2, 95, 188, 281):                # mid‑Feb, Apr, Jul, Oct
            fname = f"{var_key}_{yr}_{buildSortableString(day,2)}.asc"
            fpath = os.path.join(asc_dir, fname)
            if not os.path.exists(fpath):
                continue
            arr = getDataFrame_custom(fpath)
            arr = np.ma.masked_array(arr, mask=mask)
            sample_stack.append(arr)
    if not sample_stack:
        raise RuntimeError("Could not build static variability mask — no sample files found.")

    stack = np.ma.stack(sample_stack, axis=0)  # (time, y, x)
    stdmap = np.nanstd(stack, axis=0)
    print("mean std ", np.nanmean(stdmap))
    print("max std ", np.nanmax(stdmap))
    print("min std ", np.nanmin(stdmap))
    print("med std ", np.nanmedian(stdmap))

    _static_varmask = stdmap < STD_THRESHOLD
    return _static_varmask


# --------------------------------------
# LOOP OVER VARIABLES
# --------------------------------------
for var_key in var_keys:
    print(f"Processing variable: {var_key}")

    if anomalies:
        asc_dir = f"{forcing_root}/ECOSPACE_in_3day_vars_1980-2018/{var_key}/anomalies/"
    else:
        asc_dir = f"{forcing_root}/ECOSPACE_in_3day_vars_1980-2018/{var_key}/"

    if var_key == "PAR-VarZ-VarK":
        asc_dir = f"{forcing_root}/ECOSPACE_in_3day_PAR3_Sal4m_1980-2018/{var_key}/"

    if var_key == "RDRS_windstress10m":
        asc_dir = f"{forcing_root_RDRS}/Wind_RDRS/Ecospace/stress_/"

    if var_key == "vartemp_at150m" or var_key == "vartemp_avg150toBot":
        num_digits_ecospace_asc_month = 3

    if anomalies:
        var_key = var_key + "_anom"

    records = []
    for year in range(startyear, endyear + 1):
        print(year)
        is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        nsteps = 120

        for iday in range(1, nsteps + 1):
            middle_day = (iday - 1) * 3 + 2
            if iday >= 120:
                middle_day += 2 if is_leap else 1

            if var_key == "RDRS_windstress10m": # correcting for a file naming glitch!
                fname = f"{var_key}__{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
            else:
                fname = f"{var_key}_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"

            fpath = os.path.join(asc_dir, fname)

            if not os.path.exists(fpath):
                if middle_day >= 360:
                    middle_day = 363 # some vars have inconsistently named final day-of-year
                    if var_key == "RDRS_windstress10m":  # correcting for a file naming glitch!
                        fname = f"{var_key}__{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
                    else:
                        fname = f"{var_key}_{year}_{buildSortableString(middle_day, num_digits_ecospace_asc_month)}.asc"
                    fpath = os.path.join(asc_dir, fname)
                    if not os.path.exists(fpath): # some files have the final time step named 363, 362, etc
                        print(f"Skipping missing file: {fpath}")
                        continue
                else:
                    print(f"Skipping missing file: {fpath}")
                    continue

            data = getDataFrame_custom(fpath)

            if data.shape != combined_mask.shape:
                raise ValueError(f"Mismatch in shape between data ({data.shape}) and mask ({combined_mask.shape})")

            data = np.ma.masked_array(data, mask=combined_mask)
            # --- option‑1 static‑cell filter only for mixing layer ---
            if var_key == "varmixing_m" and DO_FILTER_STATIC:
                static_mask = build_static_varmask("varmixing_m", asc_dir, combined_mask)
                data = np.ma.masked_where(static_mask, data)
                mean_val = round(np.nanmean(data), sigdig)
                records.append([year, middle_day, mean_val])

            elif var_key == "varmixing_m" and DO_PERCENTILE:
                # vals = data.compressed()  # 1‑D array of valid cells
                # if vals.size == 0:
                #     continue  # or set to np.nan, as you prefer
                # p_val = round(np.nanpercentile(vals, percentile), sigdig)
                # records.append([year, middle_day, p_val])

                # frac = np.count_nonzero(vals >= 20) / vals.size  # gives 0–1
                # records.append([year, middle_day, round(frac, 3)])
                print("not done")


            else:
                mean_val = round(np.nanmean(data), sigdig)
                records.append([year, middle_day, mean_val])

    if var_key == "varmixing_m" and DO_PERCENTILE:

        output_csv = (f"{forcing_root}/ECOSIM_in_3day_vars_1980-2018_fromASC_202506/"
                      f"ECOSIM_in_NEMO_{var_key}_{percentile}th_1980-2018.csv")
        df = pd.DataFrame(
            records, columns=["year", "dayofyear", f"{out_code}{stat_label}_{var_key}"])
        df["threeday_yrmo"] = range(1, len(df) + 1)
        df["threeday_yr"] = df.index + 1
        df = df[["threeday_yrmo", "threeday_yr", "year", "dayofyear", f"{out_code}{stat_label}_{var_key}"]]
        data_col = f"{out_code}{stat_label}_{var_key}"

    # this is used for nutrient forcing
    elif var_key == "varmixing_m" and DO_FILTER_STATIC:
        col = f"var_{var_key}"
        df = pd.DataFrame(records, columns=["year", "dayofyear", col])

        print("max mixing dep", df[col].max())
        print("min mixing dep", df[col].min())
        print("med mixing dep", df[col].median())
        print("avg mixing dep", df[col].mean())

        # # initial scale so <G>=1
        # df[col] = df[col] / df[col].mean()

        # cap winter highs then renormalise again
        df[col] = np.minimum(df[col], CAP_HIGH)

        if USE_LOG_STRETCH:
            eps = 1e-4
            df[col] = np.log(df[col] + eps)

        if USE_PWR_EXPNSN:
            df[col] = df[col]**PWR_EXP

        if USE_MOVING_AVG:
            STEPS = WINDOW_DAYS // 3  # 3‑step centred window
            df[col] = (
                df[col]
                .rolling(window=STEPS, center=True, min_periods=1)
                .mean()
                .round(sigdig)
            )

            # # ---- stretch low end so that the minimum hits CAP_LOW ----
        #   # desired floor after rescale
        # g_min = df[col] .min()
        # if CAP_HIGH == g_min:
        #     scaled = df[col]   # avoid division‑by‑zero, trivial case
        # else:
        #     scaled = CAP_LOW + (df[col] - g_min) * (CAP_HIGH - CAP_LOW) / (CAP_HIGH - g_min)

        df[col] = df[col] / df[col].mean()
        df[col] = df[col].round(sigdig)

        records = df.values.tolist()   # replace with scaled version
        output_csv = (f"{forcing_root}/ECOSIM_in_3day_vars_1980-2018_fromASC_202506/"
                      f"ECOSIM_in_NEMO_{var_key}_stdfilter_1980-2018.csv")
        df = pd.DataFrame(
            records, columns=["year", "dayofyear", f"stdfilter_{var_key}"])
        df["threeday_yrmo"] = range(1, len(df) + 1)
        df["threeday_yr"] = df.index + 1
        df = df[["threeday_yrmo", "threeday_yr", "year", "dayofyear", f"stdfilter_{var_key}"]]
        data_col = f"stdfilter_{var_key}"

    else:

        output_csv = f"{forcing_root}/ECOSIM_in_3day_vars_1980-2018_fromASC_202506/ECOSIM_in_NEMO_{var_key}_1980-2018.csv"
        df = pd.DataFrame(records, columns=["year", "dayofyear", f"{out_code}{var_key}"])
        df["threeday_yrmo"] = range(1, len(df) + 1)
        df["threeday_yr"] = df.index + 1
        df = df[["threeday_yrmo", "threeday_yr", "year", "dayofyear", f"{out_code}{var_key}"]]
        data_col = f"{out_code}{var_key}"


    df.to_csv(output_csv, index=False)
    print(f"Saved CSV for {var_key} to:\n{output_csv}")

    # CREATE DUMMY YEARS 1978 and 1979 for spin-up
    df = pd.read_csv(output_csv)
    df_1980 = df[df["year"] == 1980].copy()

    # Create 1979 and 1978 versions by copying and changing the year
    df_1979 = df_1980.copy()
    df_1979["year"] = 1979
    df_1979["threeday_yrmo"] = df_1979["threeday_yrmo"] - len(df_1980)
    df_1979["threeday_yr"] = df_1979["threeday_yr"] - len(df_1980)

    df_1978 = df_1980.copy()
    df_1978["year"] = 1978
    df_1978["threeday_yrmo"] = df_1979["threeday_yrmo"] - len(df_1980)
    df_1978["threeday_yr"] = df_1979["threeday_yr"] - len(df_1980)

    # Concatenate in new order: 1978, 1979, original
    df_extended = pd.concat([df_1978, df_1979, df], ignore_index=True)
    df_extended.reset_index(drop=True, inplace=True)

    # Recalculate new sequential IDs
    df_extended["threeday_yrmo"] = range(1, len(df_extended) + 1)
    df_extended["threeday_yr"] = df_extended["threeday_yrmo"]

    # Replace first 12 time steps with a single climatological mean over 1980-2018
    mean_val = round(df_extended[(df_extended["year"] >= 1980) & (df_extended["year"] <= 2018)][data_col].mean(),2)
    df_extended.loc[:11, data_col] = mean_val  # Overwrite first 12 rows with same value

    # Save updated CSV
    df_extended.to_csv(output_csv, index=False)
    print(f"Prepended 1978 and 1979 using 1980 data and applied uniform climatological mean to first 12 steps for {var_key}")


