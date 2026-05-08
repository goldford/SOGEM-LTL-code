EVALUATION SCRIPTS - OVERVIEW
==============================

These scripts evaluate Ecosim and Ecospace model outputs against observational
data. The two pipelines are largely independent and can be run separately.


ENTRY POINTS
------------
  ecosim_evaluation_master_pipeline.py   -- orchestrates the Ecosim evaluation
  ecospace_evaluation_master_pipeline.py -- orchestrates the Ecospace evaluation

Each master pipeline imports and calls the individual evaluation scripts in
sequence. Boolean flags at the top of run_pipeline() / run_ecospace_pipeline()
let you toggle individual steps on or off without editing the scripts themselves.


CONFIGURATION
-------------
  ecosim_eval_config.py   -- all settings for the Ecosim pipeline
  ecospace_eval_config.py -- all settings for the Ecospace pipeline

These are the first files to look at when setting up a new run. Key things to
set at the top of each config:

  - Scenario name / Ecopath file name (SCENARIO, ECOPATH_F_NM)
  - Path to raw model output (ECOSIM_RAW_DIR / ECOSPACE_RAW_DIR)
  - Paths for processed outputs and figures (OUTPUT_DIR_*, FIGS_P, EVALOUT_P)
  - Run year range (YEAR_START_FULLRUN / ECOSPACE_RN_STR_YR etc.)

Paths to observational (evaluation) data are also defined in the config files.
Refer to those files to see exactly which datasets are expected and where they
should live.


MODEL OUTPUT REQUIREMENTS
--------------------------
Ecosim:
  A CSV of single-run biomass output (default: biomass_monthly.csv) located
  at the path defined by ECOSIM_RAW_DIR in ecosim_eval_config.py.
  The data prep step (ecosim_dataprep_1_single_run_outputs.py) reads this CSV
  and writes a processed version used by all downstream eval scripts.

Ecospace:
  The raw Ecospace output is a directory of ASC (ASCII grid) files.
  The first pipeline step (ecospace_eval_1a_ASC_to_NC.py) converts these into
  a single NetCDF file (NC_FILENAME, written to NC_PATH_OUT). All subsequent
  Ecospace eval scripts read from that NetCDF.


SCRIPT INVENTORY
----------------
Ecosim pipeline steps (called by ecosim_evaluation_master_pipeline.py):
  ecosim_dataprep_1_single_run_outputs.py  -- parse raw CSV, assign dates/seasons
  ecosim_eval_1_PPmultcalc.py              -- relative PP multiplier check
  ecosim_eval_2_seasonal_phyto.py          -- seasonal phytoplankton biomass
  ecosim_eval_3_nutrients.py               -- nutrient inventory (obs vs model)
  ecosim_eval_4_bloom_timing.py            -- spring bloom timing
  ecosim_eval_5_zoop.py                    -- zooplankton biomass comparison

Ecospace pipeline steps (called by ecospace_evaluation_master_pipeline.py):
  ecospace_eval_1a_ASC_to_NC.py            -- convert ASC output to NetCDF
  ecospace_eval_1b_assessBspinup.py        -- check biomass relative to spin-up
  ecospace_eval_2a_mcewan_phyto_seas.py    -- seasonal phyto (McEwan domains)
  ecospace_eval_2b_nemcek_phyto_match.py  )
  ecospace_eval_2c_nemcek_vs_ecospace.py  ) Nemcek phyto comparison (optional)
  ecospace_eval_3_QU39_match.py            -- match model to QU39 station data
  ecospace_eval_3b_QU39_vs_ecospace.py     -- QU39 vs Ecospace phyto eval
  ecospace_eval_4a_bloomt_CSoG.py          -- bloom timing (CSoG)
  ecospace_eval_5_zoop_eval.py             -- zooplankton evaluation
  ecospace_eval_5b_zoopskilltable_helper.py -- skill table helper (zoop)
  ecospace_eval_6a_nutrientsmatched.py     -- nutrients matched to obs locations
  ecospace_eval_6b_nutrients.py            -- nutrient overlay / summary plot


GETTING STARTED
---------------
1. Set scenario name and all paths in ecosim_eval_config.py and/or
   ecospace_eval_config.py.
2. Confirm model output files are in place (CSV for Ecosim, ASC dir for Ecospace).
3. Run the relevant master pipeline script. Toggle steps on/off via the boolean
   flags in the run_pipeline() call as needed.