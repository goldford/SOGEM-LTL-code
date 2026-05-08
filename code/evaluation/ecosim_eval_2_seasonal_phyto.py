"""
Script:   Evaluate phytoplankton seasonal b versus model
By: G Oldford, 2025
Purpose:
   - Plot seasonal b by group compared to outputs from McEwan
Input:
  - CSV of Ecosim output with season and date added
  - seasonal phyto digitized from McEwan's pub 10.1016/j.ecolmodel.2023.110402
    Sync\6. SSMSP Model\Model Greig\Data\28. Phytoplankton\PhytoplanktonCommunity_McEwanSupp2023_\MODIFIED
Output:
- barplots by phyo group: diatoms, nano, other
- seasonal avg statistics
- not done: seasonal boxplots (McEwan right now data is just barplot - would need to revert to Nemcek)
            ie would need to go to raw data, depth integrate, do chl c conversions, taxa groupings etc
"""
import ecosim_eval_config
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# -------------------------------------------
# User inputs
# -------------------------------------------

SCENARIO = ecosim_eval_config.SCENARIO
START_DATE = ecosim_eval_config.START_DATA_PHYT_SEAS
END_DATE = ecosim_eval_config.END_DATE_PHYT_SEAS
OUTPUT_ECOSIM_FILE = ecosim_eval_config.ECOSIM_F_PREPPED_SINGLERUN
OUTPUT_FIG_PATH = ecosim_eval_config.OUTPUT_DIR_FIGS
OUTPUT_STAT_PATH = ecosim_eval_config.OUTPUT_DIR_EVAL

# -------------------------------------------
# Function definitions
# -------------------------------------------

def load_and_filter_data(file_path, start_date, end_date):
    """Load CSV and filter by start and end date."""
    df = pd.read_csv(file_path, skiprows=0)
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    return df.loc[mask]


def plot_boxplots_by_season(df, value_column, title_prefix=""):
    """Generate a boxplot of the specified value column grouped by season."""
    plt.figure(figsize=(10,6))
    df.boxplot(column=value_column, by='season')
    plt.title(f'{title_prefix} Boxplot of {value_column} by Season')
    plt.suptitle('')  # Removes default pandas boxplot title
    plt.xlabel('Season')
    plt.ylabel(value_column)
    plt.tight_layout()
    plt.show()


def run_seasonal_eval():
    # Hardcoded McEwan observations (carbon units per m²)
    obs_data = {
        'Diatoms': {'Winter': 0.7, 'Spring': 6, 'Summer': 1.24},
        'Nano': {'Winter': 1.51, 'Spring': 1.74, 'Summer': 1.69},
        'Other': {'Winter': 0.11, 'Spring': 0.19, 'Summer': 0.55}
    }

    # Model columns mapping
    group_map = {
        'Diatoms': 17,
        'Nano': 18,
        'Other': 16
    }

    # === Load data ===
    df = pd.read_csv(OUTPUT_ECOSIM_FILE, skiprows=0)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)]

    # -------------------------------------------
    # Seasonal Statistics
    # -------------------------------------------

    seasonal_stats_list = []

    for season in ['Winter', 'Spring', 'Summer']:

        season_df = df[df['season'] == season]

        model_means = season_df[[str(v) for v in group_map.values()]].mean()
        model_means.index = [k for k,v in group_map.items()]
        obs_means = pd.Series({g: obs_data[g][season] for g in group_map.keys()})
        bias = model_means - obs_means

        season_stats = pd.DataFrame({
            'Season': season,
            'Group': model_means.index,
            'ModelMean': model_means.values,
            'ObsMean': obs_means.values,
            'Bias': bias.values
        })
        seasonal_stats_list.append(season_stats)

    seasonal_stats_df = pd.concat(seasonal_stats_list)

    # -------------------------------------------
    # Overall Statistics
    # -------------------------------------------

    model_means_overall = df[[str(v) for v in group_map.values()]].mean()
    model_means_overall.index = [k for k,v in group_map.items()]
    obs_means_overall = pd.Series({g: sum(obs_data[g].values()) / len(obs_data[g]) for g in group_map.keys()})
    bias_overall = model_means_overall - obs_means_overall

    overall_stats_df = pd.DataFrame({
        'Season': 'Overall',
        'Group': model_means_overall.index,
        'ModelMean': model_means_overall.values,
        'ObsMean': obs_means_overall.values,
        'Bias': bias_overall.values
    })

    # -------------------------------------------
    # Combine and Save
    # -------------------------------------------

    final_stats_df = pd.concat([seasonal_stats_df, overall_stats_df], ignore_index=True)

    os.makedirs(OUTPUT_STAT_PATH, exist_ok=True)
    final_stats_df.to_csv(os.path.join(OUTPUT_STAT_PATH, f"ecosim_{SCENARIO}_phyto_seasonal_overall_stats.csv"), index=False)

    print("Seasonal and overall statistics saved to CSV in stats folder.")
    print(final_stats_df)

    # -------------------------------------------
    # Plot
    # -------------------------------------------

    # Prepare Obs data
    obs_df = pd.DataFrame(obs_data).T  # Transpose so seasons become columns, groups as index

    # === Build plot data ===
    seasons = ['Winter', 'Spring', 'Summer']
    groups = ['Diatoms', 'Nano', 'Other']

    # Define group colors
    group_colors = {
        'Diatoms': '#1f77b4',  # blue
        'Nano': '#ff7f0e',  # orange
        'Other': '#2ca02c'  # green
    }

    # Positions
    x_labels = []
    x_pos = []
    model_heights = {g: [] for g in groups}
    obs_heights = {g: [] for g in groups}

    # Spacing
    bar_width = 0.35
    group_spacing = 1.0  # space between seasons
    pair_spacing = 0.4  # space between model and obs within a season

    pos = 0
    for season in seasons:
        # Positions for model and obs within the season
        x_pos.append(pos - pair_spacing / 2)  # model
        x_pos.append(pos + pair_spacing / 2)  # obs
        x_labels.extend([f"Mod.-{season}", f"Obs.-{season}"])

        # Heights
        for group in groups:
            value = seasonal_stats_df[
                (seasonal_stats_df['Season'] == season) &
                (seasonal_stats_df['Group'] == group)
                ]['ModelMean']

            model_heights[group].append(value.iloc[0] if not value.empty else 0)

            value_obs = seasonal_stats_df[
                (seasonal_stats_df['Season'] == season) &
                (seasonal_stats_df['Group'] == group)
                ]['ObsMean']

            obs_heights[group].append(value_obs.iloc[0] if not value_obs.empty else 0)

        pos += group_spacing

    # === Plot ===
    fig, ax = plt.subplots(figsize=(5, 4))

    # Initialize bottoms
    bottom_model = [0] * len(seasons)
    bottom_obs = [0] * len(seasons)

    # Plot model stacked bars (even indices)
    for group in groups:
        values = model_heights[group]
        ax.bar(x_pos[::2], values, bar_width, bottom=bottom_model, color=group_colors[group], label=f'{group} (Model)')
        bottom_model = [i + j for i, j in zip(bottom_model, values)]

    # Plot obs stacked bars (odd indices)
    for group in groups:
        values = obs_heights[group]
        ax.bar(x_pos[1::2], values, bar_width, bottom=bottom_obs, color=group_colors[group], hatch='//', edgecolor='grey',
               label=f'{group} (Obs)')
        bottom_obs = [i + j for i, j in zip(bottom_obs, values)]

    # === Final plot formatting ===
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Mod.' if i % 2 == 0 else 'Obs.'  # even → model, odd → obs
                        for i in range(len(x_pos))],
                       rotation=0, ha='center')

    # ---- add one centred season label per pair ----
    season_y_offset = -0.08  # tweak until it looks right
    for i, season in enumerate(seasons):
        mid_x = (x_pos[2 * i] + x_pos[2 * i + 1]) / 2  # midpoint of model/obs bars
        ax.text(mid_x, season_y_offset, season,
                ha='center', va='top',
                transform=ax.get_xaxis_transform())

    ax.set_ylabel('Biomass (g C m$^{-2}$)')
    ax.set_title(f'Model vs McEwan Seasonal Phytoplankton Biomass ({SCENARIO})')

    # === Custom legend with only three entries ===
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=group_colors['Diatoms'], label='Diatoms'),
        Patch(facecolor=group_colors['Nano'], label='Nano'),
        Patch(facecolor=group_colors['Other'], label='Other')
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)


    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(OUTPUT_FIG_PATH, f"ecosim_{SCENARIO}_phyto_seasonal_mcewan.png"),
                dpi=300)


if __name__ == "__main__":
    run_seasonal_eval()