"""
Script:    Nemcek Observation Time Series Prep (Modified)
 By: G Oldford, 2025
 Purpose:
    Create 3-day averaged time series from Nemcek et al. observations,
    with custom subdomain groupings
 Input:
   - Supp data from Nemcek et al publication 10.1016/j.pocean.2023.103108
 Output:
 - Same data, averaged to time step of model, with subdomain tagged
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from matplotlib.path import Path
import matplotlib.pyplot as plt
from helpers import read_sdomains
import matplotlib
matplotlib.use('TkAgg')

# -------------------------------
# Configuration
# -------------------------------

NEMCEK_FILE = "Nemcek_Supp_Data.csv"
NEMCEK_PATH = "C:/Users/Greig/Sync/6. SSMSP Model/Model Greig/Data/28. Phytoplankton/Phytoplankton Salish Sea Nemcek2023 2015-2019/MODIFIED"
OUTPUT_PATH = "C:/Users/Greig/Documents/github/Ecosystem-Model-Data-Framework/data/evaluation"
DOMAIN_PATH = "C:/Users/Greig/Documents/github/Ecosystem-Model-Data-Framework/data/evaluation"
DOMAIN_FILE = "analysis_domains_jarnikova.yml"
OUTPUT_FILE = "Nemcek_obs_timeseries_all_domains.csv"
SUBDOMAINS = ["SGI", "SGN", "SGS"]

# Define custom region groupings
region_map = {
    'SGS': 'Strait of Georgia South',
    'SGI': 'Southern Gulf Islands',
    'SGN': 'Strait of Georgia North'
}

start_year = 2015
end_year = 2019

# -------------------------------
# Load Data
# -------------------------------

nemcek_df = pd.read_csv(os.path.join(NEMCEK_PATH, NEMCEK_FILE))
nemcek_df['Date.Time'] = pd.to_datetime(nemcek_df['Date.Time'])

# -------------------------------
# Assign Subdomains
# -------------------------------

if 'sdomain' not in nemcek_df.columns or nemcek_df['sdomain'].eq("").any():
    sdomains = read_sdomains(os.path.join(DOMAIN_PATH, DOMAIN_FILE))
    nemcek_df['sdomain'] = ""
    for idx, row in nemcek_df.iterrows():
        lat, lon = row['Lat'], row['Lon']
        for sd, polygon in sdomains.items():
            if Path(polygon).contains_point((lat, lon)):
                nemcek_df.at[idx, 'sdomain'] = sd
                break

nemcek_df = nemcek_df[nemcek_df['sdomain'].isin(SUBDOMAINS)].copy()

# -------------------------------
# Create Derived Variables
# -------------------------------

nemcek_df = nemcek_df.rename(columns={'total diatoms': 'Diatoms'})
nanogroup_cols = ['Prasinophytes', 'Cryptophytes', 'Haptophytes', 'Dictyochophytes', 'Raphidophytes']
nemcek_df['Nemcek-NAN'] = nemcek_df[nanogroup_cols].sum(axis=1)

phyto_vars = ['Diatoms', 'Nemcek-NAN']

# -------------------------------
# Depth Averaging
# -------------------------------

group_cols = ['sdomain', 'Date.Time']
depth_avg = nemcek_df.groupby(group_cols)[phyto_vars].mean().reset_index()
depth_counts = nemcek_df.groupby(group_cols).size().reset_index(name='N')
depth_avg = pd.merge(depth_avg, depth_counts, on=['sdomain', 'Date.Time'])
depth_avg['region'] = depth_avg['sdomain'].map(region_map)

# -------------------------------
# Assign 3-Day Block Labels
# -------------------------------

def assign_3day_block(dt):
    year_start = datetime(dt.year, 1, 1)
    delta_days = (dt - year_start).days
    block = delta_days // 3 + 1
    return f"{dt.year}-{str(block).zfill(3)}"

depth_avg['Block3Day'] = depth_avg['Date.Time'].apply(assign_3day_block)

# -------------------------------
# Aggregate by Region and Block
# -------------------------------

block_avg = depth_avg.groupby(['region', 'Block3Day']).agg(
    {**{var: 'mean' for var in phyto_vars}, 'N': 'sum'}
).reset_index()

# -------------------------------
# Add 'ALL' Region
# -------------------------------

block_all = depth_avg.groupby('Block3Day').agg(
    {**{var: 'mean' for var in phyto_vars}, 'N': 'sum'}
).reset_index()
block_all['region'] = 'ALL'
block_avg = pd.concat([block_avg, block_all], ignore_index=True)

# -------------------------------
# Add Missing Blocks
# -------------------------------

def generate_all_blocks(start_year, end_year):
    all_blocks = []
    for year in range(start_year, end_year + 1):
        start = datetime(year, 1, 1)
        day = 0
        block = 1
        while day < 366:
            date = start + pd.Timedelta(days=day)
            if date.year != year:
                break
            all_blocks.append(f"{year}-{str(block).zfill(3)}")
            day += 3
            block += 1
    return all_blocks

all_blocks = generate_all_blocks(start_year, end_year)
all_regions = block_avg['region'].unique()
full_index = pd.MultiIndex.from_product([all_regions, all_blocks], names=['region', 'Block3Day'])
block_avg = block_avg.set_index(['region', 'Block3Day']).reindex(full_index).reset_index()

# -------------------------------
# Force numeric and round
# -------------------------------

for var in phyto_vars + ['N']:
    block_avg[var] = pd.to_numeric(block_avg[var], errors='coerce').round(3)

# -------------------------------
# Save Output
# -------------------------------
block_avg.to_csv(os.path.join(OUTPUT_PATH, OUTPUT_FILE), index=False)
print(f"Saved 3-day block time series (region and ALL) to {OUTPUT_FILE}")
