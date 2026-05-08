"""
Script:   Master evaluation pipeline for ecosim outputs analysis
By: G Oldford, 2025
Purpose:
   - orchestrates running of evaluation by calling functions etc in other scripts
Input:
  - config file
  - calls several eval scripts
Output:
  - log files
  - results of others scripts
"""

import ecosim_eval_config

# Import the three modules you revised
import ecosim_dataprep_1_single_run_outputs as data_prep
import ecosim_eval_1_PPmultcalc as relPP_eval
import ecosim_eval_2_seasonal_phyto as seasonal_eval
import ecosim_eval_3_nutrients as nutrient_eval
import ecosim_eval_4_bloom_timing as bloom_eval
import ecosim_eval_5_zoop as zoop_eval


def run_pipeline(run_prep=True,
                 run_relPP=True,
                 run_seasonal=True,
                 run_nutrient=True,
                 run_bloom=True,
                 run_zoop=True):
    if run_prep:
        print("=== Running Data Prep Step ===")
        data_prep.run_data_prep()

    if run_relPP:
        print("=== Running Eval of Rel PP mult ===")
        relPP_eval.run_relPP_eval()

    if run_seasonal:
        print("=== Running Seasonal Evaluation Step ===")
        seasonal_eval.run_seasonal_eval()

    if run_nutrient:
        print("=== Running Nutrientg Evaluation Step ===")
        nutrient_eval.run_nutrient_eval()

    if run_bloom:
        print("=== Running Bloom Timing Evaluation Step ===")
        bloom_eval.run_bloom_eval()

    if run_zoop:
        print("=== Running Zooplankton Evaluation Step ===")
        zoop_eval.run_zoop_eval()



    print("=== Pipeline completed successfully. ===")


if __name__ == "__main__":
    run_pipeline()

