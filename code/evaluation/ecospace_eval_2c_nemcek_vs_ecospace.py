"""
============================================================================================
Script:    Ecospace Analysis - 2b Nemcek vs Ecospace SSC Phyto v2.py
Created:   July 11, 2024
Revised:   May, 2025
Author:    G. Oldford

Purpose:
   Compare observed phytoplankton data from Nemcek et al. (2023) with model outputs from:
     - Ecospace (spatial ecosystem model)
     - SalishSeaCast (3D physical-biogeochemical model)

Inputs:
   - CSV file output from prior matching script:
       "Nemcek_matched_to_model_out_<ECOSPACE_CODE>.csv"
     containing matched observations and corresponding model estimates.

Outputs:
   - Summary bar plots of sample count by month and season per subdomain
   - Seasonal average and anomaly (z-score) plots by functional group and subdomain
   - PNG figures saved for visual comparisons of functional group trends
   - absolute biomass model vs obs

Data Sources:
   - Observations: Nemcek, N., Hennekes, M., Sastri, A., & Perry, R. I. (2023).
        Seasonal and spatial dynamics of the phytoplankton community in the Salish Sea, 2015–2019.
        *Progress in Oceanography*, 217, 103108.
        https://doi.org/10.1016/j.pocean.2023.103108

   - Model: SalishSeaCast ERDDAP download (see: https://salishsea.eos.ubc.ca/erddap/)

Notes:
   - Requires prior script to generate the matched CSV file
   - Includes a data reduction step to prevent duplicate influence of identical timestamps
============================================================================================
"""

import os
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import ecospace_eval_config as cfg

# -------------------------------
# Configuration and Input Setup
# -------------------------------

ECOSPACE_CODE = cfg.ECOSPACE_SC 
EVAL_P = cfg.EVALOUT_P
INPUT_P = cfg.NEMCEK_P
INPUT_PF = cfg.NEMCEK_MATCHED_PF

LOG_TRANSFORM_OBS = cfg.NM_LOG_TRANSFORM_OBS     # Apply log transform to observational values
LOG_TRANSFORM_MODEL = cfg.NM_LOG_TRANSFORM_MODEL  # Apply log transform to model values only for anomaly calc
EPSILON = cfg.NM_EPSILON       # Small constant to avoid log(0)
DO_EXPLOR_PLOTS = cfg.NM_DO_EXPLOR_PLOTS # exploratory histograms, data summary
APPLY_NEMCEK_SEASON = cfg.NM_APPLY_NEMCEK_SEASON

SHOW_PLTS = cfg.NM_SHOW_PLTS

# Helper: plot bar chart for counts by category
def plot_sample_counts(df_grouped, category, title, filename):
    bar_width = 0.25
    categories = df_grouped[category].unique()
    sdomains = df_grouped['sdomain'].unique()
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, sdomain in enumerate(sdomains):
        subset = df_grouped[df_grouped['sdomain'] == sdomain]
        ax.bar(x + i * bar_width, subset['Sample Count'], bar_width, label=sdomain)

    ax.set_title(title)
    ax.set_xlabel(category)
    ax.set_ylabel('Sample Count')
    ax.legend(title='Subdomain')
    ax.set_xticks(x + bar_width * (len(sdomains) - 1) / 2)
    ax.set_xticklabels(categories)
    plt.tight_layout()
    if SHOW_PLTS:
        plt.show()
    fig.savefig(filename)

# nemcek et al (2023) define a more thoughtful 'season'
def assign_nemcek_season(dt):
    month = dt.month
    day = dt.day
    if (month == 11 and day >= 16) or month in [12, 1, 2] or (month == 3 and day <= 1):
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8] or (month == 9 and day <= 15):
        return 'Summer'
    elif (month == 9 and day >= 16) or month in [10] or (month == 11 and day <= 15):
        return 'Fall'
    else:
        return 'Unknown'  # Shouldn't happen

def plot_seasonal_abs_stacked(df_simplified, out_png):
    """
    Make side-by-side stacked bars (Model vs Obs) of *absolute* seasonal biomass means.
    Uses Nemcek groupings (Obs: total diatoms, Nemcek-NAN, Cyanobacteria;
    Model: PP1-DIA, PP2-NAN, PP3-PIC). Optionally extend with SSC.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    import pandas as pd

    season_order = ['Winter', 'Spring', 'Summer', 'Fall']

    # if needed, attach Season (you already have this earlier, but just in case)
    if 'Season' not in df_simplified.columns:
        df_simplified['Month'] = pd.to_datetime(df_simplified['closest_ecospace_time']).dt.month
        df_simplified['Season'] = df_simplified['Month'].map({
            12:'Winter',1:'Winter',2:'Winter',
            3:'Spring',4:'Spring',5:'Spring',
            6:'Summer',7:'Summer',8:'Summer',
            9:'Fall',10:'Fall',11:'Fall'
        })

    # Define groups (mirror your triplets)
    groups = ['Diatoms','Nanophytoplankton','Picophytoplankton']
    obs_map  = {'Diatoms':'total diatoms','Nanophytoplankton':'Nemcek-NAN','Picophytoplankton':'Cyanobacteria'}
    mod_map  = {'Diatoms':'PP1-DIA','Nanophytoplankton':'PP2-NAN','Picophytoplankton':'PP3-PIC'}
    # Optional SSC:
    ssc_map  = {'Diatoms':'ssc-DIA','Nanophytoplankton':'ssc-FLA','Picophytoplankton':None}

    # Compute seasonal means (absolute, not standardised), pooled across subdomains,
    # or change to groupby(['Season','sdomain']) if you want per-row subplots.
    g = df_simplified.groupby('Season')

    # Build heights for model vs obs (each is stacked by group)
    seasons = [s for s in season_order if s in g.groups]

    model_heights = {grp: [] for grp in groups}
    obs_heights   = {grp: [] for grp in groups}

    for s in seasons:
        sdf = g.get_group(s)
        for grp in groups:
            model_heights[grp].append(sdf[mod_map[grp]].mean(skipna=True))
            obs_heights[grp].append(sdf[obs_map[grp]].mean(skipna=True))

    # Plot side-by-side stacks: even indices = Model, odd = Obs (matches your Ecosim figure style)
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    pair_spacing = 0.4

    x_pos = []
    pos = 0.0
    for _ in seasons:
        x_pos.extend([pos - pair_spacing/2, pos + pair_spacing/2])
        pos += 1.0

    # Initialize bottoms
    bottom_model = [0]*len(seasons)
    bottom_obs   = [0]*len(seasons)

    colors = {'Diatoms':'#1f77b4','Nanophytoplankton':'#ff7f0e','Picophytoplankton':'#2ca02c'}

    # Model stacks (even indices)
    for grp in groups:
        vals = model_heights[grp]
        ax.bar(x_pos[::2], vals, bar_width, bottom=bottom_model, label=f'{grp} (Model)', color=colors[grp])
        bottom_model = [a+b for a,b in zip(bottom_model, vals)]

    # Obs stacks (odd indices)
    for grp in groups:
        vals = obs_heights[grp]
        ax.bar(x_pos[1::2], vals, bar_width, bottom=bottom_obs, color=colors[grp], hatch='//', edgecolor='grey', label=f'{grp} (Obs)')
        bottom_obs = [a+b for a,b in zip(bottom_obs, vals)]

    # Labels & legend
    tick_labels = []
    for s in seasons:
        tick_labels.extend([f'Model-{s}', f'Obs-{s}'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_ylabel('Biomass (g C m$^{-2}$)')
    ax.set_title('Seasonal Phytoplankton Biomass — Model vs Observations')

    legend_patches = [Patch(facecolor=colors['Diatoms'], label='Diatoms'),
                      Patch(facecolor=colors['Nanophytoplankton'], label='Nanophytoplankton'),
                      Patch(facecolor=colors['Picophytoplankton'], label='Picophytoplankton')]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    if SHOW_PLTS:
        plt.show()

def run_eval_2b_nemcek_eval() -> None:

    # -------------------------------
    # Load and Inspect Data
    # -------------------------------

    if not os.path.exists(INPUT_PF):
        raise FileNotFoundError(f"Missing input file: {INPUT_PF}")

    # Load CSV and convert time field
    df = pd.read_csv(INPUT_PF)
    df['Date.Time'] = pd.to_datetime(df['Date.Time'])

    # Print basic metadata
    print("Loaded file:", INPUT_PF)
    print("Columns:", df.columns.tolist())
    print("Total rows:", len(df))

    # Filter to valid subdomains
    valid_sdomains = ['SGS', 'SGN', 'SGI']
    df_filtered = df[df['sdomain'].isin(valid_sdomains)].copy()
    print(f"Filtered to subdomains {valid_sdomains}: {len(df_filtered)} rows")

    # -------------------------------
    # Monthly and Seasonal Sample Counts
    # -------------------------------

    # Add month and season columns
    df_filtered['Month'] = df_filtered['Date.Time'].dt.month

    # apply the more thoughtful Nemcek et al (2023) seasonal definition
    if APPLY_NEMCEK_SEASON:
        df_filtered['Season'] = pd.to_datetime(df_filtered['closest_ecospace_time']).apply(assign_nemcek_season)
    else:
        season_map = {
            1: 'Winter', 2: 'Winter', 3: 'Winter',
            4: 'Spring', 5: 'Spring', 6: 'Spring',
            7: 'Summer', 8: 'Summer', 9: 'Summer',
            10: 'Fall', 11: 'Fall', 12: 'Fall'
        }
        df_filtered['Season'] = df_filtered['Month'].map(season_map)


    # Monthly sample counts
    monthly_counts = df_filtered.groupby(['Month', 'sdomain']).size().reset_index(name='Sample Count')
    monthly_index = pd.MultiIndex.from_product([range(1, 13), valid_sdomains], names=['Month', 'sdomain'])
    monthly_counts = monthly_counts.set_index(['Month', 'sdomain']).reindex(monthly_index, fill_value=0).reset_index()

    if DO_EXPLOR_PLOTS:
        plot_sample_counts(
            monthly_counts,
            category='Month',
            title='Number of Samples by Month and Subdomain',
            filename='..//..//figs//NEMCEK_DIATOM_B_AVG_MONTH_2015-2018.png'
        )

    # Seasonal sample counts
    seasonal_counts = df_filtered.groupby(['Season', 'sdomain']).size().reset_index(name='Sample Count')
    seasonal_index = pd.MultiIndex.from_product([
        ['Winter', 'Spring', 'Summer', 'Fall'], valid_sdomains
    ], names=['Season', 'sdomain'])
    seasonal_counts = seasonal_counts.set_index(['Season', 'sdomain']).reindex(seasonal_index, fill_value=0).reset_index()

    if DO_EXPLOR_PLOTS:
        plot_sample_counts(
            seasonal_counts,
            category='Season',
            title='Number of Samples by Season and Subdomain',
            filename='..//..//figs//DIATOM_B_AVG_SEASONAL_2015-2018.png'
        )

    # Station sample counts
    station_season_counts = (
        df_filtered.groupby(['Station', 'Season'])
        .size()
        .reset_index(name='Sample Count')
    )

    if DO_EXPLOR_PLOTS:
        fig, ax = plt.subplots(figsize=(10, 5))
        station_order = station_season_counts['Station'].unique()
        x = np.arange(len(station_order))
        bar_width = 0.2

        for i, season in enumerate(['Winter', 'Spring', 'Summer', 'Fall']):
            data = station_season_counts[station_season_counts['Season'] == season]
            counts = [data[data['Station'] == st]['Sample Count'].values[0] if st in data['Station'].values else 0 for st in station_order]
            ax.bar(x + i * bar_width, counts, bar_width, label=season)

        ax.set_title('Sample Counts by Station and Season')
        ax.set_xlabel('Station')
        ax.set_ylabel('Sample Count')
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(station_order, rotation=90)
        ax.legend(title='Season')
        plt.tight_layout()

        if SHOW_PLTS:
            plt.show()


    # -------------------------------
    # Functional Group Construction
    # -------------------------------

    # List of raw count fields used in Nemcek-NAN and fields_to_average
    raw_fields = ['Prasinophytes', 'Cryptophytes', 'Haptophytes', 'Dictyochophytes', 'Raphidophytes',
                  'total diatoms', 'Cyanobacteria', 'PP1-DIA', 'PP2-NAN', 'PP3-PIC',
                  'PZ1-CIL', 'ssc-DIA', 'ssc-FLA', 'ssc-CIL']

    # Apply log transformation to observational variables
    if LOG_TRANSFORM_OBS:
        obs_fields = ['Prasinophytes', 'Cryptophytes', 'Haptophytes',
                      'Dictyochophytes', 'Raphidophytes',
                      'total diatoms', 'Cyanobacteria']
        for field in obs_fields:
            if field in df_filtered.columns:
                df_filtered[field] = np.log(df_filtered[field] + EPSILON)

    # Apply log transformation to model variables
    model_fields = ['PP1-DIA', 'PP2-NAN', 'PP3-PIC',
                        'PZ1-CIL', 'ssc-DIA', 'ssc-FLA', 'ssc-CIL']
    if LOG_TRANSFORM_MODEL:
        for field in model_fields:
            if field in df_filtered.columns:
                df_filtered[field] = np.log(df_filtered[field] + EPSILON)
    else:
        # Save raw model values if not transforming them yet
        for field in model_fields:
            if field in df_filtered.columns:
                df_filtered[f'{field}_raw'] = df_filtered[field]


    # Define derived nanophytoplankton group from summed taxa
    df_filtered['Nemcek-NAN'] = df_filtered[['Prasinophytes', 'Cryptophytes', 'Haptophytes',
                                             'Dictyochophytes', 'Raphidophytes']].sum(axis=1)




    # Define fields to include in seasonal analysis
    fields_to_average = [
        'total diatoms', 'Nemcek-NAN', 'Cyanobacteria', 'PP1-DIA', 'PP2-NAN',
        'PP3-PIC', 'PZ1-CIL', 'ssc-DIA', 'ssc-FLA', 'ssc-CIL'
    ]

    print("Preparing seasonal means and anomalies for:")
    print(fields_to_average)


    # -------------------------------
    # Seasonal Mean and Anomaly Calculation
    # -------------------------------

    # Step 1: De-duplicate by averaging values from same time/location
    #         Some samples are taken within the same ecospace cell and same ecospace time
    group_cols = ['closest_ecospace_time', 'ecospace_closest_lon', 'ecospace_closest_lat']
    numeric_columns = df_filtered.select_dtypes(include='number').columns
    non_numeric_columns = df_filtered.select_dtypes(exclude='number').columns

    # Ensure grouping columns are excluded from numeric and non-numeric columns to avoid duplication errors
    numeric_columns = [col for col in numeric_columns if col not in group_cols]
    non_numeric_columns = [col for col in non_numeric_columns if col not in group_cols]

    # Group by time and spatial cell, compute mean of numeric columns
    df_numeric = df_filtered.groupby(group_cols)[numeric_columns].mean().reset_index()

    # For non-numeric fields, take most common value (mode)
    df_non_numeric = df_filtered.groupby(group_cols)[non_numeric_columns].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    ).reset_index()

    # Merge numeric and non-numeric portions on time/space keys
    df_simplified = pd.merge(df_numeric, df_non_numeric, on=group_cols, how='inner')
    print("Length after deduplication:", len(df_simplified))

    # Step 2: Seasonal averages by subdomain
    seasonal_means = df_simplified.groupby(['Season', 'sdomain'])[fields_to_average].mean().reset_index()

    # Step 3: Annual means and standard deviations by subdomain
    annual_stats = df_simplified.groupby('sdomain')[fields_to_average].agg(['mean', 'std']).reset_index()

    # Flatten multi-index columns from aggregation
    annual_stats.columns = ['sdomain'] + [f"{var}_{stat}" for var, stat in annual_stats.columns[1:]]

    # Step 4: Reshape and merge seasonal and annual stats for anomaly calc
    seasonal_melted = seasonal_means.melt(id_vars=['Season', 'sdomain'], var_name='Field', value_name='Seasonal Mean')
    annual_melted = annual_stats.melt(id_vars=['sdomain'], var_name='Field_Statistic', value_name='Value')
    annual_melted[['Field', 'Statistic']] = annual_melted['Field_Statistic'].str.rsplit('_', n=1, expand=True)
    annual_pivot = annual_melted.pivot_table(index=['sdomain', 'Field'], columns='Statistic', values='Value').reset_index()

    # Merge to calculate anomaly
    merged_data = pd.merge(seasonal_melted, annual_pivot, on=['sdomain', 'Field'])

    # Calculate z-scores (anomalies) with conditional log-transform for both obs and model fields
    # fix 2025-05-28
    # merged_data['Anomaly'] = (merged_data['Seasonal Mean'] - merged_data['mean']) / merged_data['std']
    for field in fields_to_average:
        for sdomain in merged_data['sdomain'].unique():
            mask = (merged_data['Field'] == field) & (merged_data['sdomain'] == sdomain)
            mean_val = merged_data.loc[mask, 'mean'].values[0]
            std_val = merged_data.loc[mask, 'std'].values[0]

            is_obs = field in ['total diatoms', 'Cyanobacteria', 'Nemcek-NAN']
            is_model = field in model_fields

            if ((is_obs and LOG_TRANSFORM_OBS) or (is_model and LOG_TRANSFORM_MODEL)):
                merged_data.loc[mask, 'Anomaly'] = (merged_data.loc[mask, 'Seasonal Mean'] - mean_val) / std_val
            elif is_model and not LOG_TRANSFORM_MODEL:
                seasonal_mean = merged_data.loc[mask, 'Seasonal Mean'].values[0]
                merged_data.loc[mask, 'Anomaly'] = (
                    np.log(seasonal_mean + EPSILON) - np.log(mean_val + EPSILON)
                ) / (np.log(std_val + EPSILON))
            elif is_obs and not LOG_TRANSFORM_OBS:
                seasonal_mean = merged_data.loc[mask, 'Seasonal Mean'].values[0]
                merged_data.loc[mask, 'Anomaly'] = (seasonal_mean - mean_val) / std_val


    print("Seasonal anomalies calculated. Example:")
    print(merged_data[merged_data['Field'] == 'PP1-DIA'])


    # -------------------------------
    # Model-obs stats - exploration
    # -------------------------------

    # Modify anomaly_pairs to include SSC model fields as well as Ecospace
    anomaly_triplets = {
        'Diatoms': ('total diatoms', 'PP1-DIA', 'ssc-DIA'),
        'Nanophytoplankton': ('Nemcek-NAN', 'PP2-NAN', 'ssc-FLA'),
        'Picophytoplankton': ('Cyanobacteria', 'PP3-PIC', None)
    }

    normalized_df = df_simplified.copy()
    normalized_df = normalized_df[normalized_df['closest_ecospace_time'] >= '1980-01-01']

    if 'Season' not in normalized_df.columns:
        if APPLY_NEMCEK_SEASON:
            normalized_df['Season'] = pd.to_datetime(normalized_df['closest_ecospace_time']).apply(assign_nemcek_season)
        else:
            normalized_df['Month'] = pd.to_datetime(normalized_df['closest_ecospace_time']).dt.month
            normalized_df['Season'] = normalized_df['Month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })

    # Compute anomalies for all fields
    for group_name, (obs_col, eco_col, ssc_col) in anomaly_triplets.items():
        for var in [obs_col, eco_col, ssc_col]:
            if var is None: continue
            for domain in normalized_df['sdomain'].unique():
                mask = normalized_df['sdomain'] == domain
                mean_val = normalized_df.loc[mask, var].mean()
                std_val = normalized_df.loc[mask, var].std()
                z_col = f"{var}_anom"
                normalized_df.loc[mask, z_col] = (normalized_df.loc[mask, var] - mean_val) / std_val

    anomaly_results = []
    normalized_df['sdomain_Season'] = normalized_df['sdomain'] + '_' + normalized_df['Season']
    strata = [('Season', s) for s in normalized_df['Season'].unique()] + \
             [('sdomain', d) for d in normalized_df['sdomain'].unique()] + \
             [('sdomain_Season', v) for v in normalized_df['sdomain_Season'].unique()] + \
             [('All', 'All')]

    # Compute stats for each model (Ecospace and SSC)
    for group_name, (obs_col, eco_col, ssc_col) in anomaly_triplets.items():
        for model_type, model_col in [('Ecospace', eco_col), ('SalishSeaCast', ssc_col)]:
            if model_col is None:
                continue
            for level, stratum in strata:
                if level == 'All':
                    subset = normalized_df[[f'{obs_col}_anom', f'{model_col}_anom']].dropna()
                else:
                    subset = normalized_df[normalized_df[level] == stratum][[f'{obs_col}_anom', f'{model_col}_anom']].dropna()
                obs = subset[f'{obs_col}_anom'].values
                model = subset[f'{model_col}_anom'].values

                if len(obs) >= 5:
                    bias = np.mean(model - obs)
                    rmse = np.sqrt(mean_squared_error(obs, model))
                    mae = mean_absolute_error(obs, model)
                    try:
                        r, _ = pearsonr(obs, model)
                    except ValueError:
                        r = np.nan
                    wss = 1 - np.sum((model - obs) ** 2) / np.sum((np.abs(model - np.mean(obs)) + np.abs(obs - np.mean(obs))) ** 2)
                    obs_std = np.std(obs)
                    mod_std = np.std(model)
                    centered_rmse = np.sqrt(np.mean(((model - np.mean(model)) - (obs - np.mean(obs))) ** 2))
                else:
                    bias = rmse = mae = r = wss = obs_std = mod_std = centered_rmse = np.nan

                anomaly_results.append({
                    'Group': group_name,
                    'Model': model_type,
                    'Level': level,
                    'Stratum': stratum,
                    'N': len(obs),
                    'Bias': bias,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R': r,
                    'WSS': wss,
                    'Obs_STD': obs_std,
                    'Model_STD': mod_std,
                    'Centered_RMSE': centered_rmse
                })

    stats_df = pd.DataFrame(anomaly_results)
    output_dir = os.path.join('..', '..', 'data', 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'ecospace_{ECOSPACE_CODE}_Nemcek_Seasonal_Model_Statistics.csv')
    stats_df.to_csv(output_file, index=False)
    print(f"[Saved] Seasonal statistics to {output_file}")



    # -------------------------------
    # Multi-Panel Seasonal Anomaly Plot
    # Compute Z-scores per sample and prepare for bar plot
    # -------------------------------

    # Define grouped field sets for visualization
    groups = {
        'Diatoms': ['total diatoms', 'PP1-DIA', 'ssc-DIA'],
        'Nanophytoplankton': ['Nemcek-NAN', 'PP2-NAN', 'ssc-FLA'],
        'Picophytoplankton': ['PP3-PIC', 'Cyanobacteria']
    }

    # Define color scheme
    colors = {
        'Ecospace': 'blue',
        'SalishSeaCast': 'orange',
        'Observations': 'green'
    }

    def get_color(field):
        if field.startswith('PP') or field.startswith('PZ'):
            return colors['Ecospace']
        elif field.startswith('ssc'):
            return colors['SalishSeaCast']
        else:
            return colors['Observations']

    # Start from df_simplified
    normalized_df = df_simplified.copy()

    # Functional group pairings including all 3 sources
    anomaly_triplets = {
        'Diatoms': ('total diatoms', 'PP1-DIA', 'ssc-DIA'),
        'Nanophytoplankton': ('Nemcek-NAN', 'PP2-NAN', 'ssc-FLA'),
        'Picophytoplankton': ('Cyanobacteria', 'PP3-PIC', None)  # No SSC variable for pico group
    }

    # Calculate z-scores for each variable by subdomain
    for group_name, (obs_col, eco_col, ssc_col) in anomaly_triplets.items():
        for var in [obs_col, eco_col, ssc_col]:
            if var is None: continue
            for domain in normalized_df['sdomain'].unique():
                mask = normalized_df['sdomain'] == domain
                mean_val = normalized_df.loc[mask, var].mean()
                std_val = normalized_df.loc[mask, var].std()
                z_col = f"{var}_z"
                normalized_df.loc[mask, z_col] = (normalized_df.loc[mask, var] - mean_val) / std_val

    # Create long-form DataFrame for plotting
    records = []
    for group_name, (obs_col, eco_col, ssc_col) in anomaly_triplets.items():
        for source, col in [('Observations', obs_col), ('Ecospace', eco_col), ('SMELT', ssc_col)]:
            if col is None: continue
            z_col = f"{col}_z"
            grouped = (
                normalized_df.groupby(['Season', 'sdomain'])[z_col]
                .mean()
                .reset_index()
                .rename(columns={z_col: 'Mean_Z'})
            )
            grouped['Source'] = source
            grouped['Group'] = group_name
            records.append(grouped)

    plot_df = pd.concat(records, ignore_index=True)

    # ======================
    # Plotting: Mean z-scores by season and subdomain
    # ======================

    # Plot setup
    sdomains = ['SGN', 'SGS', 'SGI']
    group_order = ['Diatoms', 'Nanophytoplankton', 'Picophytoplankton']
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    sns.set(style='whitegrid')

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 7), sharey='row')
    colors = {'Observations': 'orange', 'Ecospace': 'blue', 'SMELT': 'pink'}
    bw = 0.22 # bar width (adjusted again below)
    label_index = 0
    subplot_labels = ['(a)', '(d)', '(g)', '(b)', '(e)', '(h)', '(c)', '(f)', '(i)']


    for col, group in enumerate(group_order):
        for row, sdomain in enumerate(sdomains):
            ax = axes[row, col]
            subset = plot_df[(plot_df['Group'] == group) & (plot_df['sdomain'] == sdomain)]
            sources = subset['Source'].unique()
            if len(sources) == 3:
                bar_width = 0.18
            else:
                bar_width = bw
            for i, source in enumerate(['Observations', 'Ecospace', 'SMELT']):
                if source not in sources: continue
                data = subset[subset['Source'] == source]
                data = data.set_index('Season').reindex(season_order).reset_index()
                ax.bar(
                    x=np.arange(len(season_order)) + i * bar_width,
                    height=data['Mean_Z'],
                    width=bar_width,
                    label=source,
                    color=colors[source]
                )
            if row == 0:
                ax.set_title(group, fontsize=11)
            if col == 0:
                ax.set_ylabel('Anomaly (z-score)', fontsize=10)
                ax.text(-0.4, 0.5, sdomain, transform=ax.transAxes, fontsize=10, rotation=0,
                        verticalalignment='center', horizontalalignment='right')
            else:
                ax.set_ylabel('')
            if row == len(sdomains) - 1:
                ax.set_xticks(np.arange(len(season_order)) + bar_width)
                ax.set_xticklabels(season_order, rotation=45, fontsize=8)
            else:
                ax.set_xticklabels([])
            ax.text(0.06, 0.95, subplot_labels[row * len(group_order) + col], transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='left')

    fig.legend(colors.keys(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if SHOW_PLTS:
        plt.show()

    # save
    fig.savefig(f'..//..//figs//{ECOSPACE_CODE}_Nemcek_SSC_Ecospace_zscore_panelled.png', bbox_inches='tight')

    # seasonal stacked avg abs
    # THIS DOES NOT WORK YET WITHOUT DEPTH INT AND CONVERSIONS TO C
    # out_png = os.path.join('..', '..', 'figs', f'ecospace_{ECOSPACE_CODE}_Nemcek_Model_vs_Obs_SeasonalAbs.png')
    # plot_seasonal_abs_stacked(df_simplified, out_png)

    print('done')

if __name__ == "__main__":
    run_eval_2b_nemcek_eval()


# OLD
#
# anomaly_results = []
#
# for group_name, (obs_col, model_col) in anomaly_pairs.items():
#     obs_anom = f"{obs_col}_anom"
#     model_anom = f"{model_col}_anom"
#     subset = normalized_df[[obs_anom, model_anom]].dropna()
#     obs = subset[obs_anom].values
#     model = subset[model_anom].values
#
#     if len(obs) >= 5:
#         bias = np.mean(model - obs)
#         rmse = np.sqrt(mean_squared_error(obs, model))
#         mae = mean_absolute_error(obs, model)
#         r, _ = pearsonr(obs, model)
#         wss = 1 - np.sum((model - obs) ** 2) / np.sum((np.abs(model - np.mean(obs)) + np.abs(obs - np.mean(obs))) ** 2)
#     else:
#         bias = rmse = mae = r = wss = np.nan
#
#     anomaly_results.append({
#         'Group': group_name,
#         'Level': 'All',
#         'Stratum': 'All',
#         'N': len(obs),
#         'Bias': bias,
#         'RMSE': rmse,
#         'MAE': mae,
#         'R': r,
#         'WSS': wss
#     })
#
# # Stratified by Season, Subdomain, and Season+Subdomain
# strata = [('Season', s) for s in normalized_df['Season'].unique()] + \
#          [('sdomain', d) for d in normalized_df['sdomain'].unique()] + \
#          [('sdomain_Season', f"{row['sdomain']}_{row['Season']}")
#           for _, row in normalized_df[['sdomain', 'Season']].drop_duplicates().iterrows()]
#
# # Add combined stratum to DataFrame
# normalized_df['sdomain_Season'] = normalized_df['sdomain'] + '_' + normalized_df['Season']
#
# for level, stratum in strata:
#     for group_name, (obs_col, model_col) in anomaly_pairs.items():
#         obs_anom = f"{obs_col}_anom"
#         model_anom = f"{model_col}_anom"
#         subset = normalized_df[normalized_df[level] == stratum][[obs_anom, model_anom]].dropna()
#         obs = subset[obs_anom].values
#         model = subset[model_anom].values
#
#         if len(obs) >= 5:
#             bias = np.mean(model - obs)
#             rmse = np.sqrt(mean_squared_error(obs, model))
#             mae = mean_absolute_error(obs, model)
#             r, _ = pearsonr(obs, model)
#             wss = 1 - np.sum((model - obs) ** 2) / np.sum((np.abs(model - np.mean(obs)) + np.abs(obs - np.mean(obs))) ** 2)
#         else:
#             bias = rmse = mae = r = wss = np.nan
#
#         anomaly_results.append({
#             'Group': group_name,
#             'Level': level,
#             'Stratum': stratum,
#             'N': len(obs),
#             'Bias': bias,
#             'RMSE': rmse,
#             'MAE': mae,
#             'R': r,
#             'WSS': wss
#         })
#
# anomaly_df = pd.DataFrame(anomaly_results)
# print("\nModel performance using normalized anomalies:")
# print(anomaly_df.round(3))

# # Subplot layout setup
# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), sharex=False)
# sdomains = ['SGN', 'SGS', 'SGI']
# group_names = list(groups.keys())
# ordered_seasons = ['Winter', 'Spring', 'Summer', 'Fall']
#
# # Compute consistent y-limits for each subdomain (row)
# y_limits = []
# for sdomain in sdomains:
#     all_anomalies = []
#     for group in group_names:
#         for field in groups[group]:
#             subset = merged_data[(merged_data['sdomain'] == sdomain) & (merged_data['Field'] == field)]
#             all_anomalies.extend(subset['Anomaly'].dropna().values)
#     y_max = max(abs(np.min(all_anomalies)), abs(np.max(all_anomalies)))
#     y_limits.append((-y_max * 1.1, y_max * 1.1))
#
# # Plot loop
# subplot_labels = ['(a)', '(d)', '(g)', '(b)', '(e)', '(h)', '(c)', '(f)', '(i)']
# label_index = 0
#
# for col, group in enumerate(group_names):
#     for row, sdomain in enumerate(sdomains):
#         ax = axes[row, col]
#         bar_width = 0.2
#         x = np.arange(len(ordered_seasons))
#
#         for i, field in enumerate(groups[group]):
#             subset = merged_data[(merged_data['sdomain'] == sdomain) & (merged_data['Field'] == field)]
#             subset = subset.set_index('Season').loc[ordered_seasons].reset_index()
#             ax.bar(x + i * bar_width, subset['Anomaly'], bar_width, label=field, color=get_color(field))
#
#         ax.set_title(f'{sdomain} - {group}')
#         if col == 0:
#             ax.set_ylabel('Anomaly (z-score)', fontsize=9)
#         ax.set_xticks(x + bar_width)
#         ax.set_xticklabels(ordered_seasons, fontsize=8)
#         ax.set_ylim(y_limits[row])
#         ax.text(0.05, 0.95, subplot_labels[label_index], transform=ax.transAxes, fontsize=9,
#                 verticalalignment='top', horizontalalignment='left')
#         label_index += 1
#
# # Custom legend
# custom_lines = [
#     plt.Line2D([0], [0], color=colors['Ecospace'], lw=4),
#     plt.Line2D([0], [0], color=colors['SalishSeaCast'], lw=4),
#     plt.Line2D([0], [0], color=colors['Observations'], lw=4)
# ]
# fig.legend(custom_lines, ['Ecospace', 'SalishSeaCast', 'Observations'],
#            loc='center', ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=11)
#
# plt.tight_layout(rect=[0, 0.05, 1, 1])
# plt.show()
# fig.savefig(f'..//..//figs//{ECOSPACE_CODE}_Nemcek_SSC_Ecosp_seasonal_anomaly_plots.png', bbox_inches='tight')

