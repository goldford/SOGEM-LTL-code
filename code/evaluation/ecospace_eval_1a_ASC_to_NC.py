"""

By: G Oldford
Created: 2024-2025
Purpose: convert Ecospace ASC out to NetCDF
         For use with the 3Day Model
Input: Biomass maps as ASC files at each time step from Ecospace

Output: A single NetCDF file that combines and compresses the inputs
        Given size issues and github, these are moved manually before subsequent analysis
        to a sync folder (see shortcuts)

"""

import numpy as np
import xarray as xr
import pandas as pd
import os

from helpers import buildSortableString, is_leap_year
from datetime import datetime, timedelta
import ecospace_eval_config as cfg

ECOSPACE_SC = cfg.ECOSPACE_SC
ECOSPACE_SC_FULL = cfg.ECOSPACE_SC_FULL
ECOSPACE_RAW_DIR = cfg.ECOSPACE_RAW_DIR

ECOSPACE_RN_STR = cfg.ECOSPACE_RN_STR_YR
ECOSPACE_RN_END = cfg.ECOSPACE_RN_END_YR
ECOSPACE_RN_STR_MO = cfg.ECOSPACE_RN_STR_MO
ECOSPACE_RN_STR_DA = cfg.ECOSPACE_RN_STR_DA
ECOSPACE_RN_END_MO = cfg.ECOSPACE_RN_END_MO
ECOSPACE_RN_END_DA = cfg.ECOSPACE_RN_END_DA

NC_PATH_OUT = cfg.NC_PATH_OUT
NEMO_EWE_CSV = cfg.NEMO_EWE_CSV

DO_CROSSCHECK = cfg.DO_NC_CROSSCHECK

rows = cfg.EWE_ROWS
cols = cfg.EWE_COLS
skiprows = cfg.SKIPROWS

# mxng_p = "NEMO_prepped_as_ASC/{var}/"
# tmp_p = "NEMO_prepped_as_ASC/{var}/"
# li_p = "ECOSPACE_in_PAR3_Sal10m/{var}/"
# k_p = "ECOSPACE_in_PAR3_Sal10m/RUN203_{var}/"
# sal_p = "NEMO_prepped_as_ASC/{var}/"
# path_data2 = "../data/forcing/"
# li_p2 = "RDRS_light_monthly_ASC/{var}/"
# wi_p = "RDRS_wind_monthly_ASC/{var}/"

# asc to nc
def asc_to_nc_3day(v_f, outfilename, NEMO_EWE_CSV,
                   rows=151, cols=93, skiprows=6,
                   ECOSPACE_RN_STR=2003, ECOSPACE_RN_END=2003,
                   ECOSPACE_RN_STR_MO=1, ECOSPACE_RN_STR_DA=2,
                   ECOSPACE_RN_END_MO=12, ECOSPACE_RN_END_DA=30):

    df = pd.read_csv(NEMO_EWE_CSV)

    # Build 120 "3-day block centers" per year:
    #   - first 119: 2, 5, 8, ..., 356
    #   - last pooled block center: 360 (non-leap) or 361 (leap)
    base_centers = list(range(2, 357, 3))  # 2..356 inclusive (119 vals)

    date_list = []
    pd_timestamps = []
    time_step_model = []
    i = 0

    for yr in range(ECOSPACE_RN_STR, ECOSPACE_RN_END + 1):

        last_center = 361 if is_leap_year(yr) else 360
        days_of_year_year = base_centers + [last_center]  # 120 vals

        for doy in days_of_year_year:
            date1 = datetime(yr, 1, 1) + timedelta(days=doy - 1)
            month1 = date1.month

            if yr == ECOSPACE_RN_END and (month1 == ECOSPACE_RN_END_MO + 1):
                break

            day_of_month = date1.day
            date_list.append([yr, month1, day_of_month, doy])
            pd_timestamps.append(pd.Timestamp(yr, month1, day_of_month))
            time_step_model.append(buildSortableString(i + 1, 5))

            i += 1

        if yr == ECOSPACE_RN_END:
            break

    nutr_specs = getattr(cfg, "NUTR_FLUX_ASC", {}) or {}
    nutr_vars = set(nutr_specs.keys())

    # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    # time = pd.date_range(start='{ECOSPACE_RN_STR}-{ECOSPACE_RN_STR_MO}-{ECOSPACE_RN_STR_DA}'.format(ECOSPACE_RN_STR=ECOSPACE_RN_STR, ECOSPACE_RN_STR_MO=ECOSPACE_RN_STR_MO, ECOSPACE_RN_STR_DA=ECOSPACE_RN_STR_DA),
    #                               end='{ECOSPACE_RN_END}-{ECOSPACE_RN_END_MO}-{ECOSPACE_RN_END_DA}'.format(ECOSPACE_RN_END=ECOSPACE_RN_END, ECOSPACE_RN_END_MO=ECOSPACE_RN_END_MO, ECOSPACE_RN_END_DA=ECOSPACE_RN_END_DA),
    #                               freq='3D')
    # for t in range(1,len(pd_timestamps)+1):
    #     time_step_model.append(buildSortableString(t,5))

    # Create an empty xarray dataset
    ds = xr.Dataset(
        coords={
            'time': pd_timestamps,
            'row': range(1, rows + 1),
            'col': range(1, cols + 1),
            'lat': (('row', 'col'), np.full((rows, cols), np.nan)),
            'lon': (('row', 'col'), np.full((rows, cols), np.nan)),
            'depth': (('row', 'col'), np.full((rows, cols), np.nan)),
            'EWE_col': (('row', 'col'), np.full((rows, cols), np.nan)),
            'EWE_row': (('row', 'col'), np.full((rows, cols), np.nan)),
            'NEMO_col': (('row', 'col'), np.full((rows, cols), np.nan)),
            'NEMO_row': (('row', 'col'), np.full((rows, cols), np.nan)),
        },
        attrs={'description': 'dataset of monthly ASC files'}
    )

    # Populate the dataset with lat, lon, depth, EWE_col, EWE_row from the CSV file
    for _, row in df.iterrows():
        ewe_row = int(row['EWE_row']) - 1
        ewe_col = int(row['EWE_col']) - 1
        ds['lat'][ewe_row, ewe_col] = row['lat']
        ds['lon'][ewe_row, ewe_col] = row['lon']
        ds['depth'][ewe_row, ewe_col] = row['depth']
        ds['EWE_col'][ewe_row, ewe_col] = row['EWE_col']
        ds['EWE_row'][ewe_row, ewe_col] = row['EWE_row']
        ds['NEMO_col'][ewe_row, ewe_col] = row['NEMO_col']
        ds['NEMO_row'][ewe_row, ewe_col] = row['NEMO_row']

    # create empty variable with correct shape
    for v in v_f:
        ds[v] = xr.DataArray(
            np.nan * np.zeros((len(pd_timestamps), rows, cols)),
            dims=('time', 'row', 'col'),
            attrs={'description': f'{v} data'}
        )

    # load these ASCs into a nice xarray dataset
    for v in v_f:
        attribute = v_f[v]
        print(attribute)
        for t in range(0, len(time_step_model)):
            template = v_f[v]
            yr = int(date_list[t][0])
            doy = int(date_list[t][3])

            # Try the common patterns in order:
            # --- Nutrient forcing vars get special handling (start_year, year+doy template) ---
            if v in nutr_vars:
                spec = nutr_specs.get(v, {})
                start_year = int(spec.get("start_year", yr))

                if yr < start_year:
                    # "ignore spinup": forcings are repeated / not meaningful for early years
                    # leave ds[v] as NaN for these times and continue.
                    continue

                f_n = template.format(yr, doy)

            else:
                # Ecospace biomass outputs: single token "00001", "00002", ...
                f_n = template.format(time_step_model[t])

            if not os.path.exists(f_n):
                raise FileNotFoundError(f"Missing ASC for var={v}, t={t}, filename={f_n}")

            ti = pd_timestamps[t]
            yr = ti.year
            mo = ti.month
            da = ti.day

            with open(f_n) as f:
                data = np.loadtxt(f, skiprows=skiprows)

                # homogenize what nans are
                data[data == -9999.0] = ['nan']
                data[data == 0.0] = ['nan']

                # fix issue with bottom left area in map
                data[140:, :15] = ['nan']

                ds[f'{v}'.format(var=v)].loc[{'time': f'{yr}-{mo}-{da}'.format(year=yr, month=mo, day=da)}] = xr.DataArray(
                    data,
                    dims=('row', 'col'),
                    attrs={'description': f'{v} data for year {yr} month {mo} day {da}'.format(var=v, year=yr, month=mo, day=da)}
                )

    ######## SAVE TO NETCDF ########
    # Create an encoding dictionary for compression
    encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}

    # Write the xarray dataset to a compressed NetCDF file
    output_file = outfilename
    ds.to_netcdf(output_file, format='NETCDF4', encoding=encoding)

    # Check if the original and loaded datasets are identical
    if DO_CROSSCHECK:
        ds_loaded = xr.open_dataset(output_file)
        datasets_identical = ds.equals(ds_loaded)
    else:
        datasets_identical = "opted out of check"
    print("Datasets identical? ", datasets_identical)

    return ds, datasets_identical


def run_ecospace_prep() -> None:


    v_f = {"NK1-COH": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-NK1-COH-{}.asc",
           "NK2-CHI": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-NK2-CHI-{}.asc",
           "NK3-FOR": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-NK3-FOR-{}.asc",

           "ZF1-ICT": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-ZF1-ICT-{}.asc",

           "ZC1-EUP": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-ZC1-EUP-{}.asc",
           "ZC2-AMP": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-ZC2-AMP-{}.asc",
           "ZC3-DEC": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-ZC3-DEC-{}.asc",
           "ZC4-CLG": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-ZC4-CLG-{}.asc",
           "ZC5-CSM": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-ZC5-CSM-{}.asc",

           "ZS1-JEL": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-ZS1-JEL-{}.asc",
           "ZS2-CTH": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-ZS2-CTH-{}.asc",
           "ZS3-CHA": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-ZS3-CHA-{}.asc",
           "ZS4-LAR": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-ZS4-LAR-{}.asc",

           "PZ1-CIL": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-PZ1-CIL-{}.asc",
           "PZ2-DIN": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-PZ2-DIN-{}.asc",
           "PZ3-HNF": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-PZ3-HNF-{}.asc",

           "PP1-DIA": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-PP1-DIA-{}.asc",
           "PP2-NAN": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-PP2-NAN-{}.asc",
           "PP3-PIC": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-PP3-PIC-{}.asc", #time step format eg: 00620,

           "BA1-BAC": ECOSPACE_RAW_DIR + "EcospaceMapBiomass-BA1-BAC-{}.asc"

           # detritus isn't written out by Ecospace!

          }

    # ---- Add nutrient flux forcing used as forcing by ecospace to NC file for convenience ----
    extra = getattr(cfg, "NUTR_FLUX_ASC", {})
    for var_name, spec in extra.items():
        if isinstance(spec, dict):
            v_f[var_name] = spec["template"]
        else:
            # allow shorthand: NUTR_FLUX_ASC = {"N_FLUX_MULT": "C:/...-{}.asc"}
            v_f[var_name] = str(spec)

    # corresponding lats and lons to centres of grid cells
    # read the data into an xarray object which is versatile

    out_filename = cfg.NC_FILENAME
    out_filename = os.path.join(NC_PATH_OUT, out_filename)


    original_ds, are_identical = asc_to_nc_3day(v_f, out_filename, NEMO_EWE_CSV,
                                                rows, cols, skiprows,
                                                ECOSPACE_RN_STR, ECOSPACE_RN_END, ECOSPACE_RN_STR_MO, ECOSPACE_RN_STR_DA,
                                                ECOSPACE_RN_END_MO, ECOSPACE_RN_END_DA)
    print("Crosscheck ASC in = NC data out:", are_identical)
    print("done.")

if __name__ == "__main__":
    run_ecospace_prep()