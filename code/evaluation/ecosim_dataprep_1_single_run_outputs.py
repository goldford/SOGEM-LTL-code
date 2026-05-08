"""
Script:   Process Ecosim Biomass output to CSV that has dates and seasons
By: G Oldford, 2025
Purpose:
   Create a new TS CSV file from the biomass outputs of a single Ecosim run
   that includes dates and seasons
"""

# Import pandas
import pandas as pd
from datetime import datetime, timedelta
import ecosim_eval_config as cfg

# Config vars from master config
SCENARIO = cfg.SCENARIO
FILE_PATH = cfg.ECOSIM_RAW_DIR
OUTPUT_EXPORT_FILE = cfg.ECOSIM_F_PREPPED_SINGLERUN
TIMESTEP_DAYS = cfg.TIMESTEP_DAYS # 3 days (not flexible right now)
HEADER_N = cfg.ECOSIM_F_RAW_HEADERN
MAX_TIMESTEPS = cfg.MAX_TIMESTEPS # per year (will lump >120 with 120)
YEAR_START = cfg.YEAR_START_FULLRUN
YEAR_END = cfg.YEAR_END_FULLRUN
MATCH_MCEWAN_SEAS = cfg.MATCH_MCEWAN_SEAS

# === ADD SEASON COLUMN ===
# matches McEwan
def get_season(date):
    m = date.month
    if MATCH_MCEWAN_SEAS:
        if m in [11, 12, 1, 2]:
            return 'Winter'
        elif m in [3, 4, 5]:
            return 'Spring'
        elif m in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    else:
        if m in [12]:
            return 'Winter'
        elif m in [3]:
            return 'Spring'
        elif m in [8]:
            return 'Summer'
        else:
            return 'Fall'


# Function to calculate date with year rollover
def calc_date(timestep):
    year_offset = (timestep - 1) // MAX_TIMESTEPS
    timestep_in_year = ((timestep - 1) % MAX_TIMESTEPS) + 1
    year = YEAR_START + year_offset
    year_start_date = datetime(year, 1, 1)
    # Middle day of timestep
    date = year_start_date + timedelta(days=(timestep_in_year - 1) * TIMESTEP_DAYS + TIMESTEP_DAYS // 2)
    return date


def run_data_prep():
    # Move all your existing script logic here
    # (indent accordingly)

    # Read the CSV file, skipping the first 14 lines (0-based)
    df = pd.read_csv(FILE_PATH, skiprows=HEADER_N)

    # Show the first few rows to confirm it loaded correctly
    print(df.head())

    # === CALCULATE DATES ===
    timestep_col = df.columns[0]  # identifies 'timestep\\group'
    start_date = datetime(YEAR_START, 1, 1)

    df['date'] = df[timestep_col].apply(calc_date)
    df['season'] = df['date'].apply(get_season)

    # === OUTPUT PREVIEW ===
    # df_date = df[['date', 'season']]
    # print(df[[timestep_col, 'date']].head())

    # === EXPORT TO CSV ===
    df.to_csv(OUTPUT_EXPORT_FILE, index=False)
    print(f"Exported updated dataframe to {OUTPUT_EXPORT_FILE}")


if __name__ == "__main__":
    run_data_prep()