"""
Script:   Recommend ecosim PP multiplier
By: G Oldford, 2025
Purpose:
   - evaluate how much primary producers decline relative to initialisation due to EwE 6.7 issue
   - it does not assume that the dynamic enviro forcings are immediately applied, instead assumes average enviro
   fields are applied along with enviro responses for an initial 'faked' spin-up period
   - that means: two periods reflected in Ecosim outputs: initial 'spin-up' and then full run with variable enviro
   - Dividing up helps understand the issue with dynamic enviro versus static initial conditions wrt to enviro resp
Input:
  - CSV of Ecosim output with season and date added
Output:
- the relative PP biomass after initialising the model versus initial conditions
- implies that a PP multiplier could be applied

"""

import pandas as pd
import ecosim_eval_config

OUTPUT_ECOSIM_FILE = ecosim_eval_config.ECOSIM_F_PREPPED_SINGLERUN
START_SPINUP = ecosim_eval_config.START_SPINUP
END_SPINUP = ecosim_eval_config.END_SPINUP
START_FULL = ecosim_eval_config.START_FULL
END_FULL = ecosim_eval_config.END_FULL
GROUPS = ecosim_eval_config.GROUPS_relPP
INIT_VALS = ecosim_eval_config.GROUPS_ECOPATH_B

def run_relPP_eval():

    # === Load data ===
    df = pd.read_csv(OUTPUT_ECOSIM_FILE, skiprows=0)
    df['date'] = pd.to_datetime(df['date'])

    df_su = df[(df['date'] >= START_SPINUP) & (df['date'] <= END_SPINUP)]
    df_fu = df[(df['date'] >= START_FULL) & (df['date'] <= END_FULL)]

    # -------------------------------------------
    # Initial conditions
    print("Group\tInitial\tSpinupMean\tSpinupMult\tFullRunMean\tFullRunMult")

    for group_col in GROUPS:
        if group_col not in INIT_VALS:
            print(f"Warning: Initial value for group {group_col} not found in config.")
            continue

        initial_val = INIT_VALS[group_col]
        spinup_mean = df_su[str(group_col)].mean()
        fullrun_mean = df_fu[str(group_col)].mean()

        spinup_mult = initial_val / spinup_mean if initial_val != 0 else float('nan')
        fullrun_mult = initial_val / fullrun_mean if initial_val != 0 else float('nan')

        print(f"{group_col}\t{initial_val:.3f}\t{spinup_mean:.3f}\t{spinup_mult:.3f}\t{fullrun_mean:.3f}\t{fullrun_mult:.3f}")


if __name__ == "__main__":
    run_relPP_eval()