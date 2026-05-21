"""ecosim_eval_config.py

Shared configuration for Ecosim single-run evaluation scripts.

Conventions
- Time step: 3-day (MAX_TIMESTEPS assumes 3-day outputs).
- Nutrients: evaluated as an areal inventory (g N m-2) over a surface layer
  (default 0.1–20 m), consistent with the Ecospace nutrient evaluation.

Author: G. Oldford
"""

from __future__ import annotations

import os
from typing import Dict


# =============================================================================
# General settings
# =============================================================================

SCENARIO = "SC300"
ECOPATH_F_NM = "SOGEM-LTL_ewe6_7_18858_v19"

YEAR_START_FULLRUN = 1978
YEAR_END_FULLRUN = 2018

TIMESTEP_DAYS = 3
MAX_TIMESTEPS = 120  # per year; for 3-day outputs, values beyond 120 are typically lumped


# ---- Paths (adjust as needed) ----
ECOSIM_F_RAW_SINGLERUN = "biomass_monthly.csv"
ECOSIM_F_RAW_HEADERN = 14
ECOSIM_RAW_DIR = rf"..//..//..//SOGEM-LTL-data//data//model_out_raw//{ECOPATH_F_NM}//ecosim_Ecosim_{SCENARIO}//{ECOSIM_F_RAW_SINGLERUN}"
ECOSIM_PROCESSED_P = "..//..//..//SOGEM-LTL-data//data//model_out_prepped//"


OUTPUT_DIR_EVAL = rf"..//..//..//SOGEM-LTL-data//data//evaluation//"
OUTPUT_DIR_FIGS = "..//..//eval//figs"

ECOSIM_F_PREPPED_SINGLERUN = os.path.join(OUTPUT_DIR_EVAL, f"ecosim_{SCENARIO}_onerun_B_dates_seasons.csv")
NUTRIENTS_P = "..//..//..//SOGEM-LTL-data//data//reference//nutr//prepped_combined//"
NUTRIENTS_F_PREPPED = os.path.join(
    NUTRIENTS_P,
    "nutrients_ios_csop_cast_depthint_0p1to20m.csv"
)


# Season mapping nuance used across multiple evals
MATCH_MCEWAN_SEAS = False  # Feb -> Winter; Jun -> Spring


# =============================================================================
# Evaluation 1 – relative PP multiplier (Ecosim correction)
# =============================================================================

START_SPINUP = "1978-01-01"
END_SPINUP = "1995-12-31"
START_FULL = "1996-01-01"
END_FULL = "2018-12-31"

GROUPS_relPP = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 19, 20,
]

# Ecopath baseline biomasses (g C m-2; areal units)
# Note: Ecosim output time=0 is one step after the Ecopath baseline.
GROUPS_ECOPATH_B = {
    1: 0.0046,   # NK1-COH
    2: 0.00018,  # NK2-CHI
    3: 0.28,     # NK3-FOR
    4: 0.03,     # ZF1-ICT
    5: 0.6,      # ZC1-EUP
    6: 0.52,     # ZC2-AMP
    7: 0.09,     # ZC3-DEC
    8: 0.64,     # ZC4-CLG
    9: 0.5,      # ZC5-CSM
    10: 0.003,   # ZS1-JEL
    11: 0.03,    # ZS2-CTG
    12: 0.26,    # ZS3-CHA
    13: 0.04,    # ZS4-LAR
    14: 1.1,     # PZ1-CIL
    15: 0.87,    # PZ2-DIN
    16: 0.46,    # PZ3-HNF
    17: 2.31,    # PP1-DIA
    18: 1.5,     # PP2-NAN
    19: 0.35,    # PP3-PIC
    20: 0.17,    # BA1-BAC
    21: 4.3,     # DE2-DOC
    22: 10.1,    # DE1-POC
}


# =============================================================================
# Evaluation 2 – seasonal average biomass (phytoplankton)
# =============================================================================

START_DATA_PHYT_SEAS = "2015-01-01"
END_DATE_PHYT_SEAS = "2018-12-31"


# =============================================================================
# Evaluation 3 – nutrients (inventory-based) and seasonal pattern
# =============================================================================

# Nutrient obs-year window (END EXCLUSIVE)
# N_OBS_YR_ST = 2012
# N_OBS_YR_EN = 2019


# new controls for Ecosim nutrient matching
N_MATCH_OBS_IN_TIME = True
N_MATCH_TOL_DAYS = 3
N_OBS_DATE_MIN = START_FULL
N_OBS_DATE_MAX = END_FULL
N_OBS_DATE_REDUCER = "median"   # or "mean"
N_WRITE_MATCHED_CSV = True

# ---- Observation binning ----
OBS_AVG_TYPE = "mean"  # {"mean", "median"}

# ---- Surface layer used for nutrient inventory (match Ecospace config) ----
N_EVAL_ZMIN_M = 0.1
N_EVAL_ZMAX_M = 20.0
N_EVAL_LAYER_THICKNESS_M = float(N_EVAL_ZMAX_M - N_EVAL_ZMIN_M)

# Biweekly axis convention (14-day bins), capped to avoid a 27th bin in late December
N_BIWEEK_MAX = 26

# ---- Unit conversions ----
# 1 umol/L == 1 mmol/m3
# inventory (mmol/m2) = conc (mmol/m3) * thickness (m)
# g N m-2 = mmol/m2 * 0.014   (since 1 mol N = 14 g)
MMOL_TO_GN = 14.0 / 1000.0  # 0.014 g N per mmol N

def umolL_to_gNm2(conc_umol_L: float, thickness_m: float = N_EVAL_LAYER_THICKNESS_M) -> float:
    """Convert a concentration (umol/L) to an areal inventory (g N m-2)."""
    return float(conc_umol_L) * float(thickness_m) * MMOL_TO_GN

# Initial dissolved (free) nutrient concentration (depth-mean over N_EVAL_ZMIN_M–N_EVAL_ZMAX_M)
# Source: ecosim_data_prep_2_nutrients.py (IOS+CSOP combined; depth-avg NO3+NO2, umol/L)
N_FREE_AVG_INIT = 12.21

# Initial dissolved inventory (g N m-2)
N_FREE_INIT_GNM2 = umolL_to_gNm2(N_FREE_AVG_INIT, N_EVAL_LAYER_THICKNESS_M)

# ---- Bound nutrient pools (derived from model biomasses) ----
# Model group columns are numeric strings in the prepped Ecosim CSV.
N_BOUND_GROUPS = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 22,
]

# C -> N multipliers (g N per g C); defaults aligned to Ecospace eval
N_C_TO_N_LIVING = 0.176
N_C_TO_N_DOM = 0.15
N_C_TO_N_POM = 0.07

# Group-specific overrides by numeric group id
N_C_TO_N_BY_GROUP = {
    22: N_C_TO_N_POM,  # DE1-POC
    # 21: N_C_TO_N_DOM,  # DE2-DOC (if/when included)
}

def group_c_to_n(group_id: int) -> float:
    return float(N_C_TO_N_BY_GROUP.get(int(group_id), N_C_TO_N_LIVING))

# Initial N bound in biomass pools (g N m-2), based on Ecopath baseline biomasses
N_BOUND_INIT_GNM2 = sum(
    float(GROUPS_ECOPATH_B[g]) * group_c_to_n(g)
    for g in N_BOUND_GROUPS
    if g in GROUPS_ECOPATH_B
)

# Total inventory implied by initialization (g N m-2)
N_TOTAL_INIT_GNM2 = float(N_FREE_INIT_GNM2 + N_BOUND_INIT_GNM2)

# ---- Legacy scalars (retained for backward compatibility) ----
# Total initial bound-group biomass (g C m-2)
TOT_B_INIT = sum(float(GROUPS_ECOPATH_B[g]) for g in N_BOUND_GROUPS if g in GROUPS_ECOPATH_B)

# Initial bound inventory (g N m-2), legacy name
N_B_INIT = float(N_BOUND_INIT_GNM2)

# Fraction of total inventory that is free (dimensionless), legacy name
P_FREE_INIT = float(N_FREE_INIT_GNM2 / N_TOTAL_INIT_GNM2) if N_TOTAL_INIT_GNM2 > 0 else float("nan")

# ---- Initialization anchoring options (applied in ecosim_eval_3_nutrients.py) ----
# - N_BOUND_INIT_MODE:
#     "ecopath"   -> use N_BOUND_INIT_GNM2 as the initial bound inventory
#     "t0_series" -> use bound inventory inferred from the first model row(s)
# - N_FREE_INIT_MODE:
#     "config"            -> use N_FREE_INIT_GNM2
#     "t0_preserve_total" -> adjust free inventory so total stays N_TOTAL_INIT_GNM2
#                            while using the chosen bound-init anchor
N_BOUND_INIT_MODE = "t0_series"          # {"ecopath", "t0_series"}
N_FREE_INIT_MODE = "t0_preserve_total"   # {"config", "t0_preserve_total"}

# ---- Upward-flux multiplier (optional) ----
# Applies a scalar multiplier to represent seasonal variability in upward nutrient supply.
# If a loading function is also applied inside Ecosim, set USE_N_MULT=False.
USE_N_MULT = True
N_MULT_TYPE = "3day"  # {"seasonal", "monthly", "3day"}
EWE_NUTR_LOADING_FILE = os.path.join(
    "..", "..", "..", "SOGEM-LTL-data", "data", "forcing",
    "ECOSIM_in_3day_vars_1980-2018_fromASC_202506",
    "ECOSIM_in_NEMO_varmixing_m_stdfilter_1980-2018.csv",
)

SEASONAL_N_FLUX_MULT: Dict[str, float] = {
    "Winter": 1.0,
    "Spring": 0.8,
    "Summer": 0.6,
    "Fall": 0.9,
}

# Scenario 129 example
MONTHLY_N_FLUX_MULT: Dict[int, float] = {
    1: 1.37, 2: 1.36, 3: 1.28, 4: 1.0,  5: 0.65, 6: 0.55,
    7: 0.53, 8: 0.47, 9: 0.75, 10: 0.95, 11: 1.4,  12: 1.55,
}

SEASON_MAP: Dict[int, str] = {
    1:  "Winter", 2: "Winter", 12: "Winter",
    3:  "Spring", 4: "Spring", 5:  "Spring",
    6:  "Summer", 7: "Summer", 8:  "Summer",
    9:  "Fall",   10: "Fall",  11: "Winter",
}

# How to apply the multiplier (if USE_N_MULT True)
# - "scale_free":  N_free_used = N_free * mult
# - "scale_total": N_free_used = (N_total_init_used * mult) - N_bound(t)
N_FLUX_APPLY_MODE = "scale_free"  # {"scale_free", "scale_total"}

# Outputs
ECOSIM_F_W_NUTRIENTS = os.path.join(OUTPUT_DIR_EVAL, f"ecosim_{SCENARIO}_onerun_B_dates_seasons_nutrients.csv")

# Plot toggles for ecosim_eval_3_nutrients.py
N_SHOW_PLOT = True
N_PLOT_INCLUDE_OBS = True

# Ecospace overlay on Ecosim nutrient plot

ES_OVERLAY_ENABLE = True
ES_OVERLAY_CONFIG_MODULE = "ecospace_eval_config"
# ES_OVERLAY_SERIES = ["box"]          # {"matched", "box"} (single string or list)
ES_OVERLAY_SERIES = ["matched"]

ES_OVERLAY_MATCHED_CSV = os.path.join(
    OUTPUT_DIR_EVAL,
    f"ecospace_{SCENARIO}_nutrients_biweek_model_matched.csv",
)

ES_OVERLAY_BOX_CSV = os.path.join(
    OUTPUT_DIR_EVAL,
    f"ecospace_{SCENARIO}_nutrients_biweek_model_box.csv",
)

ES_OVERLAY_SPATIAL_REDUCER = "mean"  # {"mean", "median"}


# =============================================================================
# Evaluation 4 – bloom timing
# =============================================================================

START_FULL_BLM = "1980-01-01"
END_FULL_BLM = "2018-12-31"

BIOMASS_COLS_SATELLITE = ["17", "18", "19"]
BIOMASS_COLS_C09 = ["17"]

TOTAL_BIOMASS_COL_SATELLITE = "Biomass_Total_Satellite"
TOTAL_BIOMASS_COL_C09 = "Biomass_Total_C09"

C09_USE_PCT_MAX = False
C09_PCT_MAX = 0.9
C09_LOG_TRNSFRM = False
C09_PCT_MAX_WINDOW_DAYS = 6
C09_USE_ANNUALORALL = "annual"
C09_MEAN_OR_MEDIAN = "NOT SET UP"

SAT_LOG_TRNSFRM = False
SAT_THRESHOLD_FACTOR = 1.05
SAT_SUB_THRESHOLD_FACTOR = 0.7
SAT_MEAN_OR_MEDIAN = "median"
SAT_USE_ANNUALORALL = "annual"

NUTRIENT_DRAWDOWN_FRAC = 0.6  # unused (experimental)


# =============================================================================
# Evaluation 5 – zooplankton evaluation
# =============================================================================

Z_F_SEAS = "Zoopl_SofG_1996-2018_df_summary.csv"
Z_F_TOWLEV = "Zooplankton_B_C_gm2_EWEMODELGRP_Wide_NEMO3daymatch.csv"
Z_P_PREPPED = r"..//..//..//SOGEM-LTL-data//data//reference//zoop//Zoop_Perryetal_2021//MODIFIED"


Z_GROUP_MAP = {
    "ZF1-ICT": 4,
    "ZC1-EUP": 5,
    "ZC2-AMP": 6,
    "ZC3-DEC": 7,
    "ZC4-CLG": 8,
    "ZC5-CSM": 9,
    "ZS1-JEL": 10,
    "ZS2-CTH": 11,
    "ZS3-CHA": 12,
    "ZS4-LAR": 13,
}

ZP_SEASON_CHOICE = "Spring"
ZP_PLOT_TYPE = "bar"  # {"bar", "line"}

ZP_USE_FRIENDLY_LABELS = True
ZP_FRIENDLY_MAP_ZC = {
    "ZF1-ICT": "Ichthyo",
    "ZC1-EUP": "Euphausiids",
    "ZC2-AMP": "Amphipods",
    "ZC3-DEC": "Decapods",
    "ZC4-CLG": "Lg. Calanoid Copepods",
    "ZC5-CSM": "Other Copepods",
    "ZS1-JEL": "Jellyfish",
    "ZS2-CTH": "Ctenophores",
    "ZS3-CHA": "Chaetognaths",
    "ZS4-LAR": "Larvaceans",
    "ZF1-ICH": "Ichthyoplankton",
    "misc": "Other",
    "Total": "Total",
}

ZP_LOG_TRANSFORM = True
ZP_YEAR_START = 2000
ZP_YEAR_END = 2018

ZP_FULLRN_START = 1980
ZP_FULLRN_END = 2018

ZP_SHOW_CNTS = True

# Tow filtering & anomaly options
ZP_TOW_FILTER_MODE = "none"          # {"deep_or_complete", "none"}
ZP_TOW_MIN_PROP = 0.7
ZP_TOW_MIN_START_DEPTH_M = 150.0

ZP_ANOM_CLIM_MODE = "tows"           # {"tows", "yearly_mean"}


