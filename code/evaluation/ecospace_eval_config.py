
"""
ecospace_eval_config.py
Author: G Oldford, 2026

Description:
This script stores config params that the ecospace eval scripts have in common (single run)

Notes:
    - currently only set up for 3 day time step
"""

import pandas as pd # would rather not do imports here
import os
from typing import Dict, List


# -------------------------------------------
# General settings
# -------------------------------------------
ECOSPACE_SC = "SC300"
ECOSPACE_SC_FULL = "SC300"

ECOPATH_F_NM = "SOGEM-LTL_ewe6_7_18858_v19"
ECOPATH_RAW_OUT_P = "..//..//..//..//SOGEM-LTL-data//data//model_out_raw"
ECOSPACE_RAW_DIR = f"{ECOPATH_RAW_OUT_P}//{ECOPATH_F_NM}//Ecospace_{ECOSPACE_SC_FULL}//asc//"

FIGS_P = "..//..//eval//figs"
EVALOUT_P = "..//..//eval//tables"
FORCINGS_P = "..//..//..//..//SOGEM-LTL-data//data//forcing"
ECOSPACE_MAP_P = "..//..//..//..//SOGEM-LTL-data//data/basemap"
ECOSPACE_MAP_F = "Ecospace_grid_20210208_rowscols.csv"
REF_P = "..//..//..//SOGEM-LTL-data//data//reference"
DOMAIN_P = REF_P
NUTR_P =  REF_P + "//nutr//prepped_combined"


ECOSPACE_RN_STR_YR = 1978
ECOSPACE_RN_END_YR = 2018
ECOSPACE_RN_STR_MO = 1
ECOSPACE_RN_STR_DA = 2
ECOSPACE_RN_END_MO = 12
ECOSPACE_RN_END_DA = 30

NC_FILENAME = ECOSPACE_SC + "_" + str(ECOSPACE_RN_STR_YR) + "-" + str(ECOSPACE_RN_END_YR) + ".nc"

# NC_PATH_OUT = "..//..//data//ecospace_out//"
NC_PATH_OUT = r"..//..//..//SOGEM-LTL-data//data//model_out_prepped//"
ECOSPACE_P = NC_PATH_OUT
NEMO_EWE_CSV = "..//..//..//SOGEM-LTL-data//data//basemap//Ecospace_grid_20210208_rowscols.csv"

DO_NC_CROSSCHECK = False # can be memory intesnive to crosscheck NC with ASC after (they always match anyway)

EWE_ROWS = 151
EWE_COLS = 93
SKIPROWS = 6 # header


# -------------------------------------------
#  nutrient relevant detritus POC workaround
# -------------------------------------------

# ecospsace doesn't output detrital groups, but can export from the output window a csv (1d)
# this links to it
NU_POC_RELATIVE_CSV = rf"{NC_PATH_OUT}/ECOSPACE_Diag_Carb_36Day3Day_May 2025_graph_Relative Biomass_POC_{ECOSPACE_SC}.csv"
# column inside  CSV to use
NU_POC_REL_COL = "DE1-POC"
NU_POC_DIAG_COL_SUBSTR = "DE1-POC"

# nutr flux ASC are loaded into NC for easier access, ignoring spinup; this loads for later
NUTR_FLUX_ASC = {
    "N_FLUX_MULT": {
        "template": r"..//..//data//forcing//ECOSPACE_in_3day_nutrld_fromASC_202601//XLD_normalised_{}_{:02d}.asc",
        "units": "1",
        "description": "Relative nutrient flux multiplier (normalised to 1) used by Ecospace.",
        "zero_to_nan": False,   # retain zeros if any occur
        "freq_days": 3,         # optional metadata, spacing of ASC sequence
        "start_year": 1980,     # useful for future automation
    },
}


# -------------------------------------------
# 1 b - B before and after spin-up
# -------------------------------------------

ES_GROUPS_RELB = {
    "NK1-COH", "NK2-CHI", "NK3-FOR",
    "ZF1-ICT",
    "ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM",
    "ZS1-JEL", "ZS2-CTH", "ZS3-CHA", "ZS4-LAR",
    "PZ1-CIL", "PZ2-DIN", "PZ3-HNF",
    "PP1-DIA", "PP2-NAN", "PP3-PIC",
    "BA1-BAC",
}

ES_GROUPS_ECOPATH_B = {
    "NK1-COH": 0.0046,
    "NK2-CHI": 0.00018,
    "NK3-FOR": 0.28,
    "ZF1-ICT": 0.03,
    "ZC1-EUP": 0.6,
    "ZC2-AMP": 0.52,
    "ZC3-DEC": 0.09,
    "ZC4-CLG": 0.64,
    "ZC5-CSM": 0.5,
    "ZS1-JEL": 0.003,
    "ZS2-CTH": 0.03,
    "ZS3-CHA": 0.26,
    "ZS4-LAR": 0.04,
    "PZ1-CIL": 1.1,
    "PZ2-DIN": 0.87,
    "PZ3-HNF": 0.46,
    "PP1-DIA": 2.31,
    "PP2-NAN": 1.50,
    "PP3-PIC": 0.35,
    "BA1-BAC": 0.17,
    "DE2-DOC": 8,
    "DE1-POC": 10.1
}

ES_START_SPINUP = "1978-01-02"
ES_END_SPINUP =   "1979-12-31"



# -------------------------------------------
# Prep 2 - mcewan seasonal eval
# -------------------------------------------

MW_SHOW_PLTS = True # causes halt of pipeline - optional
MW_SEASON_DEF: Dict[str, List[int]] = {
    "Winter": [11,12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    # "Fall":   [9, 10],  # not used for McEwan comparison
}
MW_GROUP_MAP = {
    "Diatoms": "PP1-DIA",
    "Nano":    "PP2-NAN",
    "Other":   "PP3-PIC",  # adjust if Other aggregates differently
}
MW_GROUP_COLORS = {
    "Diatoms": "#1f77b4",
    "Nano":    "#ff7f0e",
    "Other":   "#2ca02c",
}

MW_DOMAIN_FP = "..//..//..//SOGEM-LTL-data//data/reference//analysis_domains_mcewan.yml"

MW_STATS_OUT = EVALOUT_P
MW_FIGS_OUT  = FIGS_P

MW_DOMAINS = {
    "NSoG": "NSoG",   # Northern Strait of Georgia
    "CSoG": "CSoG",   # Central/Southern Strait of Georgia (rename if needed)
}
MW_OBS_MCEWAN = {
    "NSoG": {
        'Diatoms': {'Winter': 0.77, 'Spring': 6.00, 'Summer': 1.16},
        'Nano':    {'Winter': 1.38, 'Spring': 1.85, 'Summer': 1.90},
        'Other':   {'Winter': 0.12, 'Spring': 0.26, 'Summer': 0.34},
    },
    "CSoG": {
        'Diatoms': {'Winter': 0.65, 'Spring': 8.06, 'Summer': 1.34},
        'Nano':    {'Winter': 1.57, 'Spring': 1.58, 'Summer': 1.53},
        'Other':   {'Winter': 0.12, 'Spring': 0.19, 'Summer': 0.74},
    }
}
# Spatial reduction across the mask: "mean" or "median"
MW_SPATIAL_REDUCTION = "mean"
MW_START_DATE = '2015-01-01'
MW_END_DATE   = '2018-12-31'



# -------------------------------------------
# Prep 2 - nemcek dataset eval
# -------------------------------------------

NM_SHOW_PLTS = False # causes halt of pipeline - optional

SSC_MO_F = "SalishSeaCast_biology_2008.nc"
SSC_GRD_F = "ubcSSnBathymetryV21-08_a29d_efc9_4047.nc"
NEMCEK_F = "Nemcek_Supp_Data.csv"
NEMCEK_MATCHED_F = f"Nemcek_matched_to_ecospace_{ECOSPACE_SC}.csv"
ECOSPACE_MAP_F = "Ecospace_grid_20210208_rowscols.csv"

# File paths
SSC_P = "C:/Users/Greig/Sync/6. SSMSP Model/Model Greig/Data/28. Phytoplankton/SalishSeaCast_BioFields_2008-2018/ORIGINAL"
ECOSPACE_P = "C:/Users/Greig/Sync/PSF/EwE/Georgia Strait 2021/LTL_model/ECOSPACE_OUT"
NEMCEK_P = "C:/Users/Greig/Sync/6. SSMSP Model/Model Greig/Data/28. Phytoplankton/Phytoplankton Salish Sea Nemcek2023 2015-2019/MODIFIED"

ECOSPACE_MAP_P = "C:/Users/Greig/Documents/github/Ecosystem-Model-Data-Framework/data/basemap"
DOMAIN_P = "C:/Users/Greig/Documents/github/Ecosystem-Model-Data-Framework/data/evaluation"
DOMAIN_F = "analysis_domains_jarnikova.yml"

NEMCEK_MATCHED_PF = os.path.join(EVALOUT_P, NEMCEK_MATCHED_F)

NM_LOG_TRANSFORM_OBS = False     # Apply log transform to observational values
NM_LOG_TRANSFORM_MODEL = False  # Apply log transform to model values only for anomaly calc
NM_EPSILON = 1e-4        # Small constant to avoid log(0)
NM_DO_EXPLOR_PLOTS = False # exploratory histograms, data summary
NM_APPLY_NEMCEK_SEASON = False



# -------------------------------------------
# Prep 3 - qu39 dataset eval
# -------------------------------------------

QU39_SHOW_PLTS = False # causes halt of pipeline - optional
QU39_SHOW_SSC_BXPLTS = False
QU39_DO_ECOSPACE_MATCHING = True       # Perform Ecospace matching
QU39_DOWNLOAD_SSC = False              # Download SSC ERDDAP files
QU39_MATCH_SSC = True                  # Match existing SSC files

QU39_IN_P = r'..//..//..//SOGEM-LTL-data//data//reference//phyto//QU39//MODIFIED//QU39_joined.csv'
QU39_OUT_P = r'..//..//..//SOGEM-LTL-data//data//reference//phyto//QU39//MODIFIED'

QU39_MATCHED_OUT_F = 'QU39_joined_matchtoEcospace' # gets tagged w scen later

SSC_DDWNLD_P = os.path.join(SSC_P, 'matched_QU39_dwnld')
SSC_BATHY_F = "ubcSSnBathymetryV21-08_a29d_efc9_4047.nc"
SSC_MESH3D_F = "ubcSSn3DMeshMaskV17-02.nc"
SSC_MESHM_2D_F = "ubcSSn2DMeshMaskV17-02.nc"

# ----- QU39 sampling site) -----
QU39_LAT = 50.0307
QU39_lON = -125.0992

QU39_INIT_DT_PLACEHOLDER = pd.to_datetime('1900-04-23 23:42')
QU39_FILL_VAL = -999.9

QU39_SSC_ECOSPACE_PF = (
    r'..//..//..//SOGEM-LTL-data//data//reference//phyto//QU39//MODIFIED'
)

QU39_INCLUDE_SSC = True
QU39_FIGS_OUT_P = os.path.normpath("..//..//figs//") + os.sep
QU39_STATS_OUT_P = os.path.normpath("..//..//data//evaluation//") + os.sep
QU39_FILL_VALUE = -999

QU39_DO_HIST = False



# -------------------------------------------
# Prep 4 - B Timing
# -------------------------------------------
BT_DOMAIN_CONFIG_PATH = "..//..//..//SOGEM-LTL-data//data/reference"
BT_DOMAIN_FILE = "analysis_domains_suchy.yml"
BT_SAT_MASK_PF = r'..//..//..//SOGEM-LTL-data//data/reference//suchy_ecospace_mask.nc'


BT_RECOMPUTE_BLOOM_TIMING_SAT = True  # Set to True to force recomputation as needed, saves time
BT_RECOMPUTE_BLOOM_TIMING_C09 = True

BT_START_YEAR = 1980 # analysis years (exclude spinup?)
BT_END_YEAR = 2018

BT_CREATE_MASKS = True  # Set to True to regenerate masks

# 20260102 - we may look at just applying this to that one year - or year by year would be ideal
# SGC2 is more accurate to map in suchy pub, but in 2005 clouds mask northern portion, affecting accuracy so SGC3 trims north
# results somewhat sensitive
BT_MASK_REGNM = "SGC4"
BT_USEC09_MASK_FOR_SAT = False # overrides above!
BT_USE_SAT_MASK_CO9 = True
BT_C09_ROW = 100 # overridden by above!
BT_C09_COL = 52

BT_CSV_SUCHY_PF = f"..//..//data//evaluation//ecospace_bloom_timing_SSoG_{ECOSPACE_SC}.csv"
BT_CSV_ALLEN_PF = f"..//..//data//evaluation//ecospace_bloom_timing_C09_{ECOSPACE_SC}.csv"

# if multiple vars listed here, it will sum across them when computing anomalies, bloom timing etc!
#OK with run 96 this is first time model fit has been okay with all pp groups
# VARIABLES_TO_ANALYZE_SAT = ["PP1-DIA"] #for 88_2 - pp1-dai only, annual, keep janfeb
BT_VARS_TO_ANALYZE_SAT = ["PP1-DIA", "PP2-NAN", "PP3-PIC"]
BT_ANNUAL_AVG_METHOD_SAT = "annual" # 'annual' or 'all' - should average bloom compared against be from all years, or just one year
BT_VARS_TO_ANALYZE_C09 = ["PP1-DIA"]
BT_ANNUAL_AVG_METHOD_C09 = "annual" # annual or all (this is always annual for C09 PCT MAX method)

# Bloom detection method
BT_LOG_TRANSFORM_SAT = False #
BT_MEAN_OR_MEDIAN_SAT = "median" # 'median" or "mean"
BT_THRESHOLD_FACTOR_SAT = 1.05
BT_SUB_THRESHOLD_FACTOR_SAT = 0.7

BT_EXCLUDE_DEC_JAN_SAT = False # working
BT_EXCLUDE_DEC_JAN_C09 = False

BT_LOG_TRANSFORM_C09 = False
BT_USE_PCT_MAX_C09 = False # new method added 2025-9-12 GO
BT_PCT_MAX_C09 = 0.9
BT_PCT_MAX_WINDOW_DAYS_C09 = 6


BT_DO_NUTRIENTS = False # another script does this now (#9?)
# OVERRIDE_REDFIELD = True # added by GO to help eval 2025-06-03
BT_MIN_Y_TICK = 30

BT_OVERLAY_ECOSIM = True
BT_ECOSIM_SAT_CSV = os.path.join(
    EVALOUT_P, f"ecosim_bloom_timing_satellite_{ECOSPACE_SC}.csv"
)
BT_ECOSIM_C09_CSV = os.path.join(
    EVALOUT_P, f"ecosim_bloom_timing_C09_{ECOSPACE_SC}.csv"
)
BT_ECOSIM_LABEL = "Ecosim"


# -------------------------------------------
# evaluation 5 - zooplankton eval
# -------------------------------------------
ZP_RECOMPUTE_MATCH = False # WATCH THIS!!

# master show toggle
ZP_SHOW_PLTS = False # SHOWS ALL IF TRUE

# per-plot-type show toggles
ZP_SHOW_PUB_TOTAL_PANEL = False

ZP_SHOW_MODOBS_TOTAL_SCATTER = False
ZP_SHOW_SCATTER_TOWS = False # colorful ones

ZP_SHOW_SEASONAL_BOXPANELS = True

ZP_SHOW_TOTAL_BY_SEASON_SCATTER = False
ZP_SHOW_MODELONLY_ANOM_BARS = False
ZP_SHOW_ANOM_PAIRED_PANEL = True
ZP_SHOW_SCATTER_ANOMS = False
ZP_SHOW_SCATTER_ANOMS_TOTAL4 = False
ZP_SHOW_ANOM_BARS_TOTAL4 = False
ZP_SHOW_ANOM_TOTAL_SINGLE = False

Z_F_SEAS = "Zoopl_SofG_1996-2018_df_summary.csv" # this is output by long R script
Z_F_TOWLEV = "Zooplankton_B_C_gm2_EWEMODELGRP_Wide_NEMO3daymatch.csv" # this is output by short one
Z_P_PREPPED = "..//..//..//SOGEM-LTL-data//data//reference//zoop/Zoop_Perryetal_2021/MODIFIED"
Z_F_MATCH = f"Zooplankton_matched_to_model_out_{ECOSPACE_SC}.csv" # ecospace matched to zoop data

# USER SETTING: choose one season and plot type ('bar' or 'line')
ZP_SEASON_CHOICE = 'Spring'  # e.g., 'Winter','Spring','Summer','Fall'
ZP_PLOT_TYPE = 'bar'     # 'bar' or 'line'
# USER SETTING: toggle friendly labels
ZP_USE_FRIENDLY_LABELS = True  # set False to use cryptic codes
ZP_FRIENDLY_MAP_ZC = {
    'ZF1-ICT': 'Ichthyo',
    'ZC1-EUP': 'Euphausiids',
    'ZC2-AMP': 'Amphipods',
    'ZC3-DEC': 'Decapods',
    'ZC4-CLG': 'Lg. Calanoid Copepods',
    'ZC5-CSM': 'Other Copepods',
    'ZS1-JEL': 'Jellyfish',
    'ZS2-CTH': 'Ctenophores',
    'ZS3-CHA': 'Chaetognaths',
    'ZS4-LAR': 'Larvaceans',
    'ZF1-ICH': 'Ichthyoplankton',
    'misc': 'Other',
    'Total': 'Total'
}

ZP_BOXPLOT_LOG10 = False      # seasonal boxplots only
ZP_SKILL_LOG10 = True       # station/tow skill tables
ZP_TOTAL_SCATTER_LOG10 = True   # optional: station total scatter only

ZP_SHOW_CNTS = False             # annotate n_tows
ZP_ANOM_ALL_SEASONS = True     # True => loop Winter/Spring/Summer/Fall
ZP_MAKE_SCATTER = True          # make scatter figs too
ZP_SCATTER_LOG10 = True         # tow-level scatter uses log10 axes

ZP_BOXPLOT_LEVEL = "tow"   # {"station", "tow"}

# for the two-panel plot for pub - GO
ZP_MAKE_PUB_TOTAL_PANEL = True
ZP_PUB_PANEL_A_MODE = "anomaly_scatter"   # "tow_scatter" or "anomaly_scatter"

# this will filter out tow-level and ALL other year data from the matched anomaly and boxplots etc
ZP_YEAR_START = 2000
ZP_YEAR_END   = 2018

ZP_FULLRN_START = 1980 # used for model-only anoms
ZP_FULLRN_END = 2018


# -------------------------------------------------------------------------
# Anomaly climatology weighting (sampling-effort imbalance)
#
# Problem: tow sampling effort is uneven across years (often far more tows in
# recent years). If the climatology (mean/std) is computed by pooling all tows,
# heavily sampled years dominate the baseline and their anomalies tend to be
# closer to 0 by construction. If the climatology is computed from annual means
# with equal-year weighting, sparsely sampled years (e.g., 1–2 tows) can exert
# disproportionate influence because their annual mean is noisy.
#
# These options control how the climatological mean/std are estimated:
#   - "tow": use all tows pooled (effort-weighted; dominated by sampled years)
#   - "year_equal": use annual means; each year weight=1 (year-balanced)
#   - "year_weighted": use annual means with year weights ~ (n_tows**power),
#       optionally capped, and with a minimum tows-per-year threshold.
# -------------------------------------------------------------------------

# --- revised anomaly-evaluation settings ---
# Use the new season-year summary anomaly engine instead of the legacy tow-level anomaly engine
ZP_ANOM_USE_SEASONAL_SUMMARY = True

# Observation-side seasonal summary options:
#   "mean_raw"     : arithmetic mean on raw biomass scale
#   "log_of_mean"  : take arithmetic mean first, then log-transform that seasonal mean
#   "mean_of_logs" : log-transform tow values first, then average in log space
# Model-side options are the same, but for your intended test use "mean_raw".
ZP_ANOM_OBS_SUMMARY = "mean_of_logs"
ZP_ANOM_MODEL_SUMMARY = "mean_raw"
ZP_ANOM_LOG_BASE = "log10"     # or "ln"

ZP_ANOM_EXCLUDE_STATIONS = [] # experimental

# For the combined "All" anomaly, average available seasonal z-scores within each year
ZP_ANOM_ALL_MIN_SEASONS = 1

# Recommended simple climatology choice for clear write-up
ZP_ANOM_CLIM_MODE = "year_equal"      # or "tow" / "year_equal" / "year_weighted"
ZP_ANOM_CLIM_MIN_TOWS_PER_YEAR = 3      # years with <3 tows won’t define baseline
P_ANOM_CLIM_WEIGHT_POWER = 0.5         # sqrt(n) compromise
ZP_ANOM_CLIM_WEIGHT_CAP = 25            # optional cap

# Turn this on to write side-by-side comparison CSVs for alternative observation summaries
ZP_COMPARE_ANOM_SUMMARIES = True
ZP_COMPARE_ANOM_OBS_SUMMARIES = ("log_of_mean", "mean_of_logs")


# -------------------------------------------------------------------------
# Optional tow-eligibility filter (to mirror Ecosim workflow)
#   - "deep_or_complete": keep tows with (tow_prop >= min_prop) OR (start_depth >= min_start_depth_m)
#   - "none": keep all tows
# NOTE: if changed, set RECOMPUTE_MATCH=True! or delete the old matched CSV.
# -------------------------------------------------------------------------
ZP_TOW_FILTER_MODE = "deep_or_complete"   # {"deep_or_complete", "none"} data from perry's set are already filtered somewhat
ZP_TOW_MIN_PROP = 0.7
ZP_TOW_MIN_START_DEPTH_M = 150.0
ZP_TOW_MAX_BOTTOM_DEPTH_M = None   # optional additional cutoff



# -------------------------------------------


# -------------------------------------------
# ZP - model-only box anomaly
# -------------------------------------------

ZP_MODELONLY_DO = True

ZP_MODELONLY_VERBOSE = True
ZP_MODELONLY_PROGRESS_EVERY = 25   # print every N cells during extraction

# Box bounds (edit these to your preferred CSoG rectangle)
ZP_MODELONLY_LAT_MIN = 49.00
ZP_MODELONLY_LAT_MAX = 49.90
ZP_MODELONLY_LON_MIN = -124.20
ZP_MODELONLY_LON_MAX = -123.10

# Model-only anomaly year window
ZP_MODELONLY_YEAR_START = ZP_FULLRN_START
ZP_MODELONLY_YEAR_END   = ZP_FULLRN_END

# Spatial reduction across selected wet cells
ZP_MODELONLY_SPATIAL_REDUCER = "mean"   # "mean" or "median"

# Reuse your anomaly-summary logic
ZP_MODELONLY_ANOM_SUMMARY = "mean_raw"

# None => combine all seasons within year using existing "All" logic
# or set to "Winter", "Spring", "Summer", "Fall"
ZP_MODELONLY_SEASON = 'Spring'

# ------------------------------------------------------------------
# Define exactly which model-only series to generate.
#
# Each key is the output series name that will appear in plots/csvs.
# Each value is the list of Ecospace groups to sum.
#
# A single-item list gives an individual series.
# A multi-item list gives a total / aggregate series.
# ------------------------------------------------------------------
ZP_MODELONLY_SERIES_DEF = {
    "ZC2-AMP": ["ZC2-AMP"],
    "Total":   ["ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM"],
}

# Which of those named series to plot
ZP_MODELONLY_PLOT_GROUP = "Total"

# Optional pretty labels for figure titles / legends
ZP_MODELONLY_LABELS = {
    "ZC2-AMP": "Amphipods",
    "Total": "Total crustaceans",
}

# Output filenames
ZP_MODELONLY_SERIES_CSV = f"ecospace_zoop_model_box_series_{ZP_MODELONLY_SEASON}_{ECOSPACE_SC}.csv"
ZP_MODELONLY_ANOM_CSV   = f"ecospace_zoop_model_box_anom_{ZP_MODELONLY_SEASON}_{ECOSPACE_SC}.csv"




# =============================================================================
# Evaluation 6 – nutrients (observations vs Ecospace; optional Ecosim overlay)
# =============================================================================

# ---- Inputs ----
NU_F_PREPPED = os.path.join(EVALOUT_P, "nutrients_ios_csop_combined_sampled.csv")

# Plot window (inclusive)
NU_PLT_YR_ST = 2012
NU_PLT_YR_EN = 2018

# Observation filter window (inclusive)
NU_OBS_YR_ST = 2012
NU_OBS_YR_EN = 2018

# Box sampling window (inclusive)
NU_MODEL_SAMPLE_YR_ST = 2000
NU_MODEL_SAMPLE_YR_EN = 2019

# ---- Observation -> inventory settings ----
NU_POOL_BY_CELL = True

NU_ZMIN_M = 0.1
NU_ZMAX_M = 20.0

# Profile handling (used by ecospace_eval_6a_nutrientsmatched.py)
NU_REQUIRE_FULL_COVERAGE = True
NU_SURFACE_PAD_TOL_M = 2.0
NU_DEEP_PAD_TOL_M = 2.0
NU_DEPTH_EXTRAPOLATE = True
NU_ALLOW_SINGLE_DEPTH = False

NU_OBS_VALUE_MODE = "integral" # how to compute depth integration = average or integral (aerial = integral)

NU_OBS_BIN_AVG_TYPE = "mean"  # {"mean", "median"}
NU_MIN_DEPTHS_PER_CAST = 2
NU_MIN_MAXDEPTH_M = 15.0

# Time matching tolerance (days) when pairing obs to model (matched series)
NU_TIME_TOL_DAYS = 7

# Plot controls
NU_SHOW_PLOT = True
NU_INCLUDE_POC_DIAG = False
NU_DEBUG_POC = False
NU_DEBUG_SAVE_DF = False

# Force a full biweekly axis (1..NU_BIWEEK_MAX) even if some bins are missing
NU_FORCE_FULL_BIWEEK_AXIS = True
NU_BIWEEK_MAX = 26


# ---- Initialization anchoring and flux multiplier ----
# Bound/free pools inferred from model biomasses, then decomposed as:
#   N_free(t) = max(0, N_total_init_used - N_bound(t))
# where N_total_init_used and N_bound(t) can optionally incorporate a flux multiplier.
#
# - NU_BOUND_INIT_MODE:
#     "ecopath"   -> use NU_BOUND_INIT_GNM2
#     "t0_series" -> use the first available model slice as the bound anchor
# - NU_FREE_INIT_MODE:
#     "config"            -> use NU_FREE_INIT_GNM2
#     "t0_preserve_total" -> adjust free so total stays NU_TOTAL_INIT_GNM2
#                            while using the chosen bound-init anchor
NU_BOUND_INIT_MODE = "t0_series"          # {"ecopath", "t0_series"}
NU_T0_SPATIAL_REDUCER = "mean"            # {"mean", "median"} across cells at t0 (for t0_series)
NU_T0_PER_SERIES = True                   # compute t0 per series ("matched" vs "box")

NU_FREE_INIT_MODE = "t0_preserve_total"   # {"config", "t0_preserve_total"}
NU_DEBUG_INIT_ANCHOR = False

# Flux multiplier (optional)
NU_USE_FLUX_MULT = True
NU_FLUX_MULT_COL = "N_FLUX_MULT"
NU_FLUX_MULT_FILL = 1.0

# How to apply the multiplier:
# - "scale_free":  N_free_used = N_free * mult
# - "scale_total": N_free_used = (N_total_init_used * mult) - N_bound(t)
NU_FLUX_APPLY_MODE = "scale_free"         # {"scale_free", "scale_total"}


# ---- Model series definitions ----
# "matched": model sampled at each obs location/time (then pooled)
# "box":     model sampled over a fixed rectangle of cells (then pooled)
# NU_MODEL_SERIES = ["box", "matched"]
NU_MODEL_SERIES = ["matched"]

# Box coordinates are given as (row0, row1, col0, col1), inclusive bounds.
# This is a fixed region-of-interest, not a per-observation buffer.
NU_MODEL_SAMPLE_BOX = (90, 98, 46, 55)

# Optional stride when sampling within NU_MODEL_SAMPLE_BOX (1 = sample every cell)
NU_MODEL_SAMPLE_STRIDE = 1

# When aggregating across cells within each (year, biweekly)
NU_MODEL_SAMPLE_SPATIAL_REDUCER = "mean"  # {"mean", "median"}

# If True, require both "matched" and "box" to exist before computing combined outputs
NU_REQUIRE_BOTH_FOR_AGG = False


# ---- Model pools and conversions (g N per g C) ----
NU_MODEL_GROUPS = [
    "NK1-COH", "NK2-CHI", "NK3-FOR",
    "ZF1-ICT",
    "ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM",
    "ZS1-JEL", "ZS2-CTH", "ZS3-CHA", "ZS4-LAR",
    "PZ1-CIL", "PZ2-DIN", "PZ3-HNF",
    "PP1-DIA", "PP2-NAN", "PP3-PIC",
    "BA1-BAC",
    "DE1-POC",
]

# Variables pulled from Ecospace output (must exist in the NetCDF)
NU_EXTRACT_GRPS = NU_MODEL_GROUPS
NU_EXTRACT_EXTRA_VARS = [NU_FLUX_MULT_COL] if NU_USE_FLUX_MULT else []

NU_C_TO_N_LIVING = 0.176
NU_C_TO_N_DOM = 0.15
NU_C_TO_N_POM = 0.07

# Group-specific overrides
NU_C_TO_N_BY_GROUP = {
    "DE1-POC": NU_C_TO_N_POM,
}

def _nu_c_to_n(grp: str) -> float:
    return float(NU_C_TO_N_BY_GROUP.get(grp, NU_C_TO_N_LIVING))


# ---- Initial N inventories (g N m-2) ----
# These define the total inventory reference. If NU_BOUND_INIT_MODE="t0_series",
# the bound anchor is taken from the model, while the total remains fixed.
NU_FREE_INIT_GNM2 = 3.5
NU_BOUND_INIT_GNM2 = sum(float(ES_GROUPS_ECOPATH_B[g]) * _nu_c_to_n(g) for g in NU_MODEL_GROUPS if g in ES_GROUPS_ECOPATH_B)
NU_TOTAL_INIT_GNM2 = float(NU_FREE_INIT_GNM2 + NU_BOUND_INIT_GNM2)

# Convenience: implied initial fraction free (dimensionless)
NU_P_FREE_INIT = float(NU_FREE_INIT_GNM2 / NU_TOTAL_INIT_GNM2) if NU_TOTAL_INIT_GNM2 > 0 else float("nan")


# ---- Outputs (produced by ecospace_eval_6a_nutrientsmatched.py) ----
NU_OBS_BIWEEK_CSV = os.path.join(EVALOUT_P, f"ecospace_{ECOSPACE_SC}_nutrients_biweek_obs.csv")
NU_MODEL_MATCHED_BIWEEK_CSV = os.path.join(EVALOUT_P, f"ecospace_{ECOSPACE_SC}_nutrients_biweek_model_matched.csv")
NU_MODEL_BOX_BIWEEK_CSV = os.path.join(EVALOUT_P, f"ecospace_{ECOSPACE_SC}_nutrients_biweek_model_box.csv")


# ---- Optional: overlay Ecosim series on Ecospace nutrient plots ----
NU_OVERLAY_ECOSIM = True
NU_ECOSIM_NUTRIENTS_CSV = os.path.join(EVALOUT_P, f"ecosim_{ECOSPACE_SC}_onerun_B_dates_seasons_nutrients.csv")
NU_ECOSIM_MODEL_COL = "model_N_free_used_gN_m2"
NU_ECOSIM_LABEL = "Ecosim (1D)"


# =============================================================================
# Evaluation 7 – Taylor plots
# =============================================================================

TY_STATS_FPAT_SPC_FMT = "ecospace_bloom_timing_stats_*.csv"
TY_STATS_FPAT_SIM_FMT = "ecosim_bloom_eval_stats_*.csv"

TY_STATS_F_SPC_FMT = r"^ecospace_bloom_timing_stats_(.+)\.csv$"
TY_STATS_F_SIM_FMT = r"^ecosim_bloom_eval_stats_(.+)\.csv$"
