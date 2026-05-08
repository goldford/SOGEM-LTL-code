"""
Ecosim Bloom Timing Evaluation Script
Author: G Oldford, 2025

Description:
This script evaluates bloom timing using **Ecosim model outputs** (1D CSV time series).
It compares Ecosim-derived bloom timing against observational datasets:
- Satellite-derived bloom dates (Suchy et al. 2022)
- Allen 1D model outputs (Collins et al. 2009)

Outputs:
- CSV files summarising bloom timing
- Comparative plots
- Evaluation statistics (RMSE, MAE, Bias, R, Willmott skill)

Note: Designed to be independent from the Ecospace bloom evaluation script.
      Some shared or semi-shared scripts should be combined and put in helpers.
"""

# -------------------------------------------
# Imports
# -------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import ecosim_eval_config as cfg # eval config file


# -------------------------------------------
# Configuration
# -------------------------------------------

SCENARIO = cfg.SCENARIO
ECOSIM_CSV_PATH = cfg.ECOSIM_F_W_NUTRIENTS
STATS_OUT_PATH = cfg.OUTPUT_DIR_EVAL
FIGS_OUT_PATH = cfg.OUTPUT_DIR_FIGS

# User-specified biomass columns for Satellite and C09
BIOMASS_COLS_SATELLITE = cfg.BIOMASS_COLS_SATELLITE
BIOMASS_COLS_C09 = cfg.BIOMASS_COLS_C09

TOTAL_BIOMASS_COL_SATELLITE = cfg.TOTAL_BIOMASS_COL_SATELLITE
TOTAL_BIOMASS_COL_C09 = cfg.TOTAL_BIOMASS_COL_C09

START_FULL_BLM = cfg.START_FULL_BLM
END_FULL_BLM = cfg.END_FULL_BLM

# Bloom detection parameters
THRESHOLD_FACTOR = cfg.SAT_THRESHOLD_FACTOR
SUB_THRESHOLD_FACTOR = cfg.SAT_SUB_THRESHOLD_FACTOR
SAT_LOG_TRNSFRM = cfg.SAT_LOG_TRNSFRM
MEAN_OR_MEDIAN = cfg.SAT_MEAN_OR_MEDIAN
SAT_USE_ANNUALORALL = cfg.SAT_USE_ANNUALORALL

C09_LOG_TRNSFRM = cfg.C09_LOG_TRNSFRM
C09_USE_PCT_MAX = cfg.C09_USE_PCT_MAX
C09_PCT_MAX = cfg.C09_PCT_MAX
C09_PCT_MAX_WINDOW_DAYS = cfg.C09_PCT_MAX_WINDOW_DAYS
C09_USE_ANNUALORALL = cfg.C09_USE_ANNUALORALL



# -------------------------------------------
# Load observation datasets
# -------------------------------------------

def load_observation_bloom_dfs():
    doy_satellite = [100, 68, 50, 83, 115, 115, 100, 100, 92, 88, 92, 92, 55, 77]
    years_satellite = list(range(2003, 2017))
    bloom_el_sat =  ["avg", "avg", "early", "avg",
                     "late", "late", "avg", "avg",
                     "avg", "avg", "avg", "avg",
                     "early", "avg"]
    satellite_df = pd.DataFrame({
        'Year': years_satellite,
        'Day of Year': doy_satellite,
        'Bloom Early Late': bloom_el_sat
    })

    doy_C09 = [94, 78, 81, 82, 87, 76, 75, 87, 99, 76,
               81, 78, 77, 55, 86, 86, 67, 87, 77, 104,
               66, 70, 92, 86, 81, 62, 88, 100, 90, 97,
               104, 103, 98, 88, 86, 77, 88, 104, 77]
    years_C09 = list(range(1980, 2019))

    mean_C09 = np.mean(doy_C09)
    std_C09 = np.std(doy_C09)
    labels_C09 = classify_bloom_by_std(doy_C09, mean_C09, std_C09)

    C09_df = pd.DataFrame({
        'Year': years_C09,
        'Day of Year': doy_C09,
        'Bloom Early Late': labels_C09
    })

    return satellite_df, C09_df


def classify_bloom_by_std(doys, mean_val, std_val):
    results = []
    for doy in doys:
        if doy <= (mean_val - std_val):
            results.append("early")
        elif doy >= (mean_val + std_val):
            results.append("late")
        elif doy <= (mean_val + std_val - 1) and doy >= (mean_val - std_val + 1):
            results.append("avg")
        else:
            results.append("cusp")
    # exploring 'cusp' method
    # for doy in doys:
    #     if doy + 4 <= (mean_val - std_val):
    #         results.append("early")
    #     elif doy - 4 >= (mean_val + std_val):
    #         results.append("late")
    #     elif doy + 4 <= (mean_val + std_val - 1) and doy - 4 >= (mean_val - std_val + 1):
    #         results.append("avg")
    #     else:
    #         results.append("cusp")
    return results


# -------------------------------------------
# Load Ecosim model outputs and compute total biomass
# -------------------------------------------

def load_ecosim_dataset():
    df = pd.read_csv(ECOSIM_CSV_PATH)
    df['Date'] = pd.to_datetime(df['date'])
    df['Year'] = df['Date'].dt.year
    df['Day of Year'] = df['Date'].dt.dayofyear

    # Sum specified biomass columns for Satellite and C09
    df[TOTAL_BIOMASS_COL_SATELLITE] = df[BIOMASS_COLS_SATELLITE].sum(axis=1, skipna=True)
    df[TOTAL_BIOMASS_COL_C09] = df[BIOMASS_COLS_C09].sum(axis=1, skipna=True)

    return df


# -------------------------------------------
# Bloom detection
# -------------------------------------------

# def find_bloom_doy(df, biomass_col, threshold_factor=1.05):
#     bloom_dates = []
#     bloom_doys = []
#
#     for year, group in df.groupby('Year'):
#         ts = group.copy()
#         if LOG_TRANSFORM:
#             ts[biomass_col] = np.log(ts[biomass_col] + 0.01)
#
#         base = ts[biomass_col].median() if MEAN_OR_MEDIAN == "median" else ts[biomass_col].mean()
#         threshold = base * threshold_factor
#
#         bloom = ts[ts[biomass_col] > threshold]
#         if not bloom.empty:
#             bloom_date = bloom.iloc[0]['Date']
#             bloom_dates.append(bloom_date)
#             bloom_doys.append(bloom_date.timetuple().tm_yday)
#         else:
#             bloom_dates.append(None)
#             bloom_doys.append(None)
#
#     return pd.DataFrame({
#         'Year': df['Year'].unique(),
#         'Bloom Date': bloom_dates,
#         'Day of Year': bloom_doys
#     })
# import numpy as np
# import pandas as pd

def find_bloom_doy(
    df,
    biomass_col,
    threshold_factor=THRESHOLD_FACTOR,
    sub_threshold_factor=SUB_THRESHOLD_FACTOR,
    use_median=MEAN_OR_MEDIAN,
    log_transform=False,
    use_annualorall="annual",
    exclude_months=None   # e.g. [1,5,6,7,8,9,10,11,12]
):
    """
    Detects the first sustained spring bloom for each year.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Date' (datetime64) and `biomass_col`.
    biomass_col : str
        Name of the biomass column to analyse.
    threshold_factor : float, optional
        Main bloom threshold = baseline * threshold_factor.
    sub_threshold_factor : float, optional
        Sustained‑bloom check threshold = main_thresh * sub_threshold_factor.
    use_median : bool, optional
        Use median (True) or mean (False) as the baseline statistic.
    log_transform : bool, optional
        Apply log(x+0.01) before analysis.
    exclude_months : list[int] | None, optional
        Calendar months (1–12) to exclude from both baseline and search.
        Pass None to keep all months.

    Returns
    -------
    pd.DataFrame with columns ['Year', 'Bloom Date', 'Day of Year'].
    """
    df = df.copy()

    # Optional month filtering
    if exclude_months:
        df = df[~df['Date'].dt.month.isin(exclude_months)]

    # Optional log‑transform
    if log_transform:
        df[biomass_col] = np.log(df[biomass_col] + 0.01)

    bloom_dates, bloom_doys = [], []

    for yr, grp in df.groupby(df['Date'].dt.year, sort=True):
        grp = grp.sort_values('Date').reset_index(drop=True)

        baseline = grp[biomass_col].median() if use_median else grp[biomass_col].mean()
        main_thresh = baseline * threshold_factor
        sub_thresh  = main_thresh * sub_threshold_factor

        bloom_date = None
        # Walk through the year, looking ahead four steps max
        for i in range(len(grp) - 4):
            if grp.loc[i, biomass_col] >= main_thresh:
                # How many of the next 4 points stay above sub‑threshold?
                window = grp.loc[i+1:i+4, biomass_col]
                if (window >= sub_thresh).sum() >= 2:
                    bloom_date = grp.loc[i, 'Date']
                    break

        if bloom_date is not None:
            bloom_dates.append(bloom_date)
            bloom_doys.append(bloom_date.timetuple().tm_yday)
        else:
            bloom_dates.append(None)
            bloom_doys.append(None)

    return pd.DataFrame({
        'Year': df['Date'].dt.year.sort_values().unique(),
        'Bloom Date': bloom_dates,
        'Day of Year': bloom_doys
    })


def classify_bloom_timing(bloom_df,
                          bloom_early=68,
                          bloom_late=108,
                          margin=1.5):
    """
    Takes the bloom_df (with a 'bloom_doy' column) and returns
    that same DataFrame with an extra 'bloom_timing' column
    in {early, avg, late, cusp, None}.
    """
    def _classify(doy):
        if pd.isna(doy):
            return None
        if doy + margin <= bloom_early:
            return "early"
        if doy - margin >= bloom_late:
            return "late"
        if (doy - margin >= bloom_early) and (doy + margin <= bloom_late):
            return "avg"
        return "cusp"

    df2 = bloom_df.copy()
    df2['Bloom Early Late'] = df2['Day of Year'].apply(_classify)
    return df2


#  Align years
def align_years(df_model, df_obs):
    return df_model[df_model['Year'].isin(df_obs['Year'])]

def find_bloom_doy_pctmax(
    df,
    biomass_col,
    window=6,                 # days
    pct_of_max=0.95,
    log_transform=False,
    use_annualorall= "annual",
    exclude_months=None       # e.g., [1, 12] to ignore Dec–Jan
):
    """
    Bloom timing = first date where the time-smoothed biomass reaches
    >= pct_of_max * (yearly maximum of that smoothed series).

    Notes
    -----
    * Uses a *time-based* rolling window ('6D') so it works for
      daily, 3-day, or irregular time steps.
    * Returns a DataFrame with ['Year', 'Bloom Date', 'Day of Year'].
    """
    out_rows = []

    # Optional month filtering (applies to both baseline and search)
    work = df.copy()
    if exclude_months:
        work = work[~work['Date'].dt.month.isin(exclude_months)]

    # Optional log transform
    if log_transform:
        work[biomass_col] = np.log(work[biomass_col] + 0.01)

    # in case use all years for determining threshold (not typical approach)
    # work2 = work.copy()
    # work2 = work2.sort_values('Date').reset_index(drop=True)
    # ts_all = work2.set_index('Date')[biomass_col].sort_index()
    # smoothed_all = ts_all.rolling(f'{window}D', min_periods=1).mean()

    for yr, grp in work.groupby('Year', sort=True):
        if grp.empty:
            out_rows.append((yr, pd.NaT, np.nan))
            continue

        # Sort and use time-based rolling mean
        grp = grp.sort_values('Date').reset_index(drop=True)
        ts = grp.set_index('Date')[biomass_col].sort_index()

        # '6D' makes this robust to non-daily sampling
        # if use_annualorall == "annual":
        smoothed = ts.rolling(f'{window}D', min_periods=1).mean()
        # else:
        #     smoothed = smoothed_all.copy()

        max_val = smoothed.max()
        if pd.isna(max_val) or max_val <= 0:
            out_rows.append((yr, pd.NaT, np.nan))
            continue

        thresh = pct_of_max * max_val

        # First date at/above threshold
        hit_idx = smoothed.index[smoothed >= thresh]
        if len(hit_idx) > 0:
            first_dt = pd.Timestamp(hit_idx[0])
            out_rows.append((yr, first_dt, first_dt.dayofyear))
        else:
            out_rows.append((yr, pd.NaT, np.nan))

    return pd.DataFrame(out_rows, columns=['Year', 'Bloom Date', 'Day of Year'])



# ------------------------------------------
# -
#  Plotting
# -------------------------------------------
def plot_bloom_comparison(df_model, df_obs, label_model="Ecosim", label_obs="Observation", filename="bloom_timing_ecosim.png"):

    df_merged = df_model.merge(df_obs, on="Year", suffixes=("_Model", "_Obs"))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot model with error bars and line
    plt.errorbar(df_merged['Year'], df_merged['Day of Year_Model'], yerr=1.5,
                 fmt='o', color='blue', label=label_model,
                 markersize=3, capsize=3)
    plt.plot(df_merged['Year'], df_merged['Day of Year_Model'], '-', color='blue',
             markersize=0)

    # Plot observations with error bars and line
    plt.errorbar(df_merged['Year'], df_merged['Day of Year_Obs'], yerr=4,
                 fmt='s', color='darkorange', label=label_obs,
                 markersize=3, capsize=3)
    plt.plot(df_merged['Year'], df_merged['Day of Year_Obs'], '-', color='darkorange',
             markersize=0)

    mean = np.nanmean(df_merged['Day of Year_Obs'])
    std = np.nanstd(df_merged['Day of Year_Obs'])

    print(label_obs)
    print(f"late bloom DoY: {mean+std}")
    print(f"average bloom DoY {mean}")
    print(f"early bloom DoY: {mean-std}")
    plt.axhline(y=mean+std, linestyle='--', color='grey', label='') # late threshold
    plt.axhline(y=mean, linestyle='-', color='grey', label='')      # average
    plt.axhline(y=mean-std, linestyle='--', color='grey', label='') # early threshold

    plt.grid(True)
    # ax.set_ylim([MIN_Y_TICK, df_merged[["Day of Year_Model", "Day of Year_Obs"]].max().max() + 10])
    ax.set_ylim(30,140)
    plt.xlabel("Year")
    plt.ylabel("Day of Year")
    plt.title(f"Bloom Timing Comparison: {label_model} {SCENARIO} vs {label_obs}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(FIGS_OUT_PATH, filename))
    plt.show()
    plt.close()

# NOT DONE
def find_bloom_doy_relative_to_min(df, nutrient_col, rel=0.2):
    """
    For each year, find the first date when df[nutrient_col] ≤ min_value * (1 + rel).
    rel = 0.2 means threshold = min + 20%.
    """
    bloom_dates = []
    bloom_doys  = []

    for year, grp in df.groupby('Year'):
        # compute the yearly minimum
        min_val  = grp[nutrient_col].min()
        init_val = grp.iloc[0][nutrient_col]

        drawdown_total = init_val - min_val
        threshold = min_val + (rel * drawdown_total)

        # find first time below-or-equal to that threshold
        hit = grp[grp[nutrient_col] <= threshold]
        if not hit.empty:
            dt = hit.iloc[0]['Date']
            bloom_dates.append(dt)
            bloom_doys.append(dt.timetuple().tm_yday)
        else:
            bloom_dates.append(None)
            bloom_doys.append(None)

    return pd.DataFrame({
        'Year'       : sorted(df['Year'].unique()),
        'Bloom Date' : bloom_dates,
        'Day of Year': bloom_doys
    })


# -------------------------------------------
# Evaluation
# -------------------------------------------

def willmott1981(obs, mod):
    mod = np.asarray(mod)
    obs = np.asarray(obs)
    num = np.nansum((mod - obs) ** 2)
    obs_mean = np.nanmean(obs)
    dM = np.abs(mod - obs_mean)
    dO = np.abs(obs - obs_mean)
    den = np.nansum((dM + dO) ** 2)
    if den == 0:
        return np.nan
    else:
        return max(0, 1 - num / den)


def evaluate_model(obs, mod, label):
    obs = np.asarray(obs)
    mod = np.asarray(mod)
    valid = ~np.isnan(obs) & ~np.isnan(mod)

    obs_valid = obs[valid]
    mod_valid = mod[valid]

    mse = mean_squared_error(obs_valid, mod_valid)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(obs_valid, mod_valid)
    bias = np.mean(mod_valid - obs_valid)
    r = np.corrcoef(obs_valid, mod_valid)[0, 1] if len(obs_valid) > 1 else np.nan
    skill = willmott1981(obs_valid, mod_valid)
    std_obs = np.std(obs_valid)
    std_mod = np.std(mod_valid)

    return {
        "Label": label,
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "Bias": round(bias, 3),
        "R": round(r, 3) if not np.isnan(r) else np.nan,
        "Willmott Skill": round(skill, 3) if not np.isnan(skill) else np.nan,
        "Obs StdDev": round(std_obs, 3),
        "Model StdDev": round(std_mod, 3)
    }


def export_evaluation_stats(stats_list, out_path, scenario):
    df_stats = pd.DataFrame(stats_list).round(2)
    outfile = os.path.join(STATS_OUT_PATH, f"ecosim_bloom_eval_stats_{scenario}.csv")
    df_stats.to_csv(outfile, index=False)
    print("Exported bloom timing stats to: " + outfile)

def evaluate_bloom_categories(df_obs,
                              df_mod,
                              col_obs='Bloom Early Late',
                              col_mod='Bloom Early Late'):
    df_obs = standardize_columns(df_obs)

    for col in df_obs.columns:
        if "Bloom Early Late" in col and col != "Bloom Early Late":
            df_obs = df_obs.rename(columns={col: "Bloom Early Late"})
            break  # Stop after the first match

    df = df_obs.merge(df_mod, on='Year', suffixes=('_obs', '_mod'))
    agree_count = (df[f'{col_obs}_obs'] == df[f'{col_mod}_mod']).sum()
    total = len(df)
    return agree_count, total


def standardize_columns(df):
    for col in df.columns:
        if "Year" in col and col != "Year" and col != "Day of Year":
            df = df.rename(columns={col: "Year"})
            break  # Stop after the first match
    for col in df.columns:
        if "Day of Year" in col and col != "Year":
            df = df.rename(columns={col: "Day of Year_Obs"})
            break  # Stop after the first match
    return df


def evaluate_overlap_by_timing(df_obs, df_mod, obs_col='Day of Year', mod_col='Day of Year'):

    df_obs = standardize_columns(df_obs)
    for col in df_obs.columns: #standardise
        if "Day of Year" in col and col != "Day of Year":
            df_obs = df_obs.rename(columns={col: "Day of Year"})
            break  # Stop after the first match


    df = df_obs.merge(df_mod, on='Year', suffixes=('_obs', '_mod'))
    # Define bounds
    obs_low = df[f'{obs_col}_obs'] - 4
    obs_high = df[f'{obs_col}_obs'] + 4
    mod_low = df[f'{mod_col}_mod'] - 1.5
    mod_high = df[f'{mod_col}_mod'] + 1.5
    overlap = (mod_high >= obs_low) & (mod_low <= obs_high)
    return overlap.sum(), len(overlap)



# -------------------------------------------
# Main script
# -------------------------------------------
def run_bloom_eval():
    satellite_df, C09_df = load_observation_bloom_dfs()
    ecosim_df = load_ecosim_dataset()

    DRAWDOWN_FRAC = cfg.NUTRIENT_DRAWDOWN_FRAC  # e.g. 0.05
    # INCOMPLETE - seems to work but init nutrients free cant be high
    # nutri_bloom = find_bloom_doy_relative_to_min(
    #     ecosim_df,
    #     nutrient_col='N_Free_Adjusted',
    #     rel=DRAWDOWN_FRAC
    # )
    # nutri_bloom.to_csv(
    #     os.path.join(EVAL, f"ecosim_nutrient_bloom_{SCENARIO}.csv"),
    #     index=False
    # )

    # Bloom detection for Satellite biomass columns
    bloom_df_satellite = find_bloom_doy(ecosim_df,
                                        biomass_col=TOTAL_BIOMASS_COL_SATELLITE,
                                        threshold_factor=THRESHOLD_FACTOR,
                                        log_transform=SAT_LOG_TRNSFRM,
                                        use_annualorall=SAT_USE_ANNUALORALL)
    bloom_df_satellite = classify_bloom_timing(bloom_df_satellite,
                                               bloom_early=68,
                                               bloom_late=108,
                                               margin=1.5)
    bloom_df_satellite.to_csv(os.path.join(STATS_OUT_PATH, f"ecosim_bloom_timing_satellite_{SCENARIO}.csv"), index=False)

    # Bloom detection for C09 biomass columns
    if C09_USE_PCT_MAX: # alt approach added by GO 2025-08-12
        bloom_df_C09 = find_bloom_doy_pctmax(
            ecosim_df,
            biomass_col=TOTAL_BIOMASS_COL_C09,
            window=C09_PCT_MAX_WINDOW_DAYS,
            pct_of_max=C09_PCT_MAX,
            log_transform=C09_LOG_TRNSFRM,
            use_annualorall=C09_USE_ANNUALORALL,
            exclude_months=None
        )
    else:
        bloom_df_C09 = find_bloom_doy(ecosim_df,
                                      biomass_col=TOTAL_BIOMASS_COL_C09,
                                      threshold_factor=THRESHOLD_FACTOR,
                                      log_transform=C09_LOG_TRNSFRM,
                                      use_annualorall=C09_USE_ANNUALORALL)

    bloom_df_C09 = classify_bloom_timing(bloom_df_C09,
                                               bloom_early=C09_df['Day of Year'].mean() - C09_df['Day of Year'].std(),
                                               bloom_late=C09_df['Day of Year'].mean() + C09_df['Day of Year'].std(),
                                               margin=1.5)

    bloom_df_C09.to_csv(os.path.join(STATS_OUT_PATH, f"ecosim_bloom_timing_C09_{SCENARIO}.csv"), index=False)

    # Align model to observation years
    bloom_df_satellite_aligned = bloom_df_satellite[bloom_df_satellite['Year'].isin(satellite_df['Year'])]
    bloom_df_C09_aligned = bloom_df_C09[bloom_df_C09['Year'].isin(C09_df['Year'])]
    bloom_df_C09_aligned = bloom_df_C09_aligned[(bloom_df_C09_aligned['Bloom Date'] >= START_FULL_BLM) &
                                                (bloom_df_C09_aligned['Bloom Date'] <= END_FULL_BLM)]

    C09_df = C09_df[(C09_df['Year'] >= pd.to_datetime(START_FULL_BLM).year) &
                    (C09_df['Year'] <= pd.to_datetime(END_FULL_BLM).year)]

    # Evaluation
    stats_sat = evaluate_model(satellite_df['Day of Year'], bloom_df_satellite_aligned['Day of Year'], "sat")
    stats_C09 = evaluate_model(C09_df['Day of Year'], bloom_df_C09_aligned['Day of Year'], "C09")
    export_evaluation_stats([stats_sat, stats_C09], STATS_OUT_PATH, SCENARIO)

    print("Evaluation Statistics vs Satellite:", stats_sat)
    print("Evaluation Statistics vs C09:", stats_C09)

    # Plotting
    plot_bloom_comparison(bloom_df_satellite_aligned, satellite_df, label_model="Ecosim", label_obs="Satellite",
                          filename=f"ecosim_{SCENARIO}vs_satellite.png")
    plot_bloom_comparison(bloom_df_C09_aligned, C09_df, label_model="Ecosim", label_obs="C09",
                          filename=f"ecosim_{SCENARIO}vs_C09.png")


    # ---------------------------------------------------------------
    # categorical comparison
    # ---------------------------------------------------------------
    # Additional categorical comparisons
    agree_cat_sat, total_cat_sat = evaluate_bloom_categories(satellite_df, bloom_df_satellite_aligned,
                                                             col_obs='Bloom Early Late')
    agree_cat_C09, total_cat_C09 = evaluate_bloom_categories(C09_df, bloom_df_C09_aligned,
                                                                 col_obs='Bloom Early Late')
    print("\nCategorical Agreement:")
    print(f"Satellite: {agree_cat_sat}/{total_cat_sat} years agree in category")
    print(f"C09: {agree_cat_C09}/{total_cat_C09} years agree in category")

    cat_stats = [
        {
            "Label": "Satellite",
            "Type": "Categorical Agreement",
            "Count": agree_cat_sat,
            "Total": total_cat_sat,
            "Proportion": agree_cat_sat / total_cat_sat if total_cat_sat > 0 else np.nan
        },
        {
            "Label": "C09",
            "Type": "Categorical Agreement",
            "Count": agree_cat_C09,
            "Total": total_cat_C09,
            "Proportion": agree_cat_C09 / total_cat_C09 if total_cat_C09 > 0 else np.nan
        }
    ]

    # Overlap by timing comparison
    overlap_sat, n_sat = evaluate_overlap_by_timing(satellite_df, bloom_df_satellite_aligned)
    overlap_C09, n_C09 = evaluate_overlap_by_timing(C09_df, bloom_df_C09_aligned,
                                                        obs_col='Day of Year')

    cat_stats.extend([
        {
            "Label": "Satellite",
            "Type": "Timing Window Overlap",
            "Count": overlap_sat,
            "Total": n_sat,
            "Proportion": overlap_sat / n_sat if n_sat > 0 else np.nan
        },
        {
            "Label": "C09",
            "Type": "Timing Window Overlap",
            "Count": overlap_C09,
            "Total": n_C09,
            "Proportion": overlap_C09 / n_C09 if n_C09 > 0 else np.nan
        }
    ])

    cat_stats_df = pd.DataFrame(cat_stats)
    cat_stats_df["Proportion"] = cat_stats_df["Proportion"].round(2)
    cat_stats_csv_path = os.path.join(STATS_OUT_PATH, f"ecosim_bloom_timing_agreement_{SCENARIO}.csv")
    cat_stats_df.to_csv(cat_stats_csv_path, index=False)
    print(f"Saved categorical/timing agreement stats to {cat_stats_csv_path}")


if __name__ == "__main__":
    run_bloom_eval()

