"""# -------------------------------------------------------------------
Script Name: Ecospace Analysis - QU39 vs Ecospace vs SSC
Author: G. Oldford
Last Updated: Jul 2025

Description:
  This script compares observational phytoplankton data from QU39
  (2016–2019) with matched model outputs from:
    - Ecospace (EwE) – functional group biomass
    - SalishSeaCast (SSC/SMELT) – phytoplankton concentration

  The comparison is visualized using boxplots of normalized anomalies
  (relative to climatological monthly means and standard deviations).

Data Source:
  Del Bel Belluz, J. (2024). Protistan plankton time series from the
  northern Salish Sea and central coast, BC, Canada.
  OBIS Dataset: https://obis.org/dataset/071a38b3-0d30-47ba-85aa-d98313460075

Inputs:
  - CSV file (QU39_joined_matchtoEcospace_SSC_<model_run>.csv)
    containing bottle-sampled phytoplankton counts matched to
    Ecospace and SSC model outputs.

Outputs:
  - Monthly boxplots comparing QU39 vs Ecospace vs SSC (per group)
  - Histograms of raw and log-transformed concentration distributions
  - Count of records by month

Notes:
  - Anomalies are computed as:
      (value - monthly mean) / monthly std deviation
  - Log-transformed values are used for interpretability
  - Figures are saved to the `figs/` directory
  - 2025-05 - cross-check the QU39 pairings (eg class: Bacillo..) are first summed by id_x
            - otherwise comparisons are wonky - comparing total PP1-DIA to diatoms-by-species
              instead of diatoms by class. And averaging monthly is not properly stratifying
              unless you first sum across the whole diatom class for each id_x (station-daycombo)
-------------------------------------------------------------------
"""

# ----------------------------- Configuration ---------------------------- #
import os
import pandas as pd
import calendar
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')
import ecospace_eval_config as cfg

SHOW_PLTS = cfg.QU39_SHOW_PLTS
SHOW_SSC_BXPLTS = cfg.QU39_SHOW_SSC_BXPLTS
MODEL_RUN = cfg.ECOSPACE_SC
INCLUDE_SSC = cfg.QU39_INCLUDE_SSC # IS THIS HOOKED UP?
pathfile_QU39_SSC_Ecospace = cfg.QU39_SSC_ECOSPACE_PF
figs_out_p = cfg.QU39_FIGS_OUT_P
stats_out_p = cfg.QU39_STATS_OUT_P
FILL_VALUE = cfg.QU39_FILL_VALUE
DO_HIST = cfg.QU39_DO_HIST

# Month ordering
months = np.arange(1, 13)
# ------------------------------------------------------------------------ #

def prepare_group_dataset(
    df,
    class_filter,
    obs_field_raw='measurementValue',
    obs_log_field='logQU39',
    model_fields=None,
    id_col='id_x',
    time_col='closest_ecospace_time',
    group_col='class'
):
    """
    Prepare dataset for a functional group:
    - Aggregates observations across species per sample (id_x)
    - Preserves matched model values
    - Applies log-transform
    - Computes anomalies for both observations and model fields

    Parameters:
        df : pd.DataFrame
        class_filter : str or list - taxonomic class(es)
        obs_field_raw : str - raw QU39 observation field
        obs_log_field : str - name for log-transformed observation field
        model_fields : list of str - model columns to retain and transform
        id_col : str - sample ID
        time_col : str - model match datetime field
        group_col : str - grouping column (typically 'class')

    Returns:
        pd.DataFrame
    """
    df_sub = df[df[group_col].isin([class_filter] if isinstance(class_filter, str) else class_filter)].copy()

    if model_fields is None:
        model_fields = ['PP1-DIA', 'PP2-NAN', 'PP3-PIC', 'PZ2-DIN', 'ssc-DIA', 'ssc-FLA', 'ssc-CIL']

    # Aggregate by sample and time
    agg_df = df_sub.groupby([id_col, time_col]).agg({
        obs_field_raw: 'sum',
        **{col: 'first' for col in model_fields if col in df_sub.columns}
    }).reset_index()

    # Assign class label and time fields
    # IMPORTANT FIX: Give a consistent name to the group label
    agg_df[group_col] = class_filter if isinstance(class_filter, str) else 'Nanophytoplankton'
    agg_df['QU39'] = agg_df[obs_field_raw]
    agg_df[obs_log_field] = np.log(agg_df['QU39'].clip(lower=0.001))
    agg_df[time_col] = pd.to_datetime(agg_df[time_col])
    agg_df['closest_ecospace_time'] = agg_df[time_col]  # ensure standardized name
    agg_df['month'] = agg_df[time_col].dt.month
    agg_df['year'] = agg_df[time_col].dt.year

    # Log-transform model fields
    log_model_fields = []
    for field in model_fields:
        if field in agg_df.columns:
            log_field = f'log_{field}'
            agg_df[log_field] = np.log(agg_df[field].clip(lower=0.001))
            log_model_fields.append(log_field)

    # Compute anomalies for observation and model log fields
    all_fields_to_normalize = [obs_log_field] + log_model_fields
    for field in all_fields_to_normalize:
        agg_df = calculate_anomalies(agg_df, field, group_col)

    return agg_df

def calculate_anomalies(df, valuefield, groupon):
    """
    Calculate anomalies of a value field normalized by climatological mean and std dev.

    Parameters:
        df : pd.DataFrame
            The dataframe with columns including valuefield, year, month, and groupon.
        valuefield : str
            The column to compute anomalies on.
        groupon : str
            The taxonomic grouping field (e.g., 'class' or 'family').

    Returns:
        pd.DataFrame with new columns for anomalies and climatology.
    """
    df = df.copy()

    if 'level_0' in df.columns:
        df = df.drop(columns=['level_0'])

    if 'eventID' in df.columns:
        df = df.drop(columns=['eventID'])

    # Filter valid values
    df = df[df[valuefield] > FILL_VALUE]

    # Compute monthly climatology grouped by year/month and groupon
    numeric_cols = df.select_dtypes(include=np.number).columns.difference(
        ['year', 'month', groupon, 'closest_ecospace_time'])
    monthly_stats = df.groupby(['year', 'month', groupon, 'closest_ecospace_time'])[numeric_cols].mean().reset_index()

    # Compute monthly and annual climatologies
    monthly_means = monthly_stats.groupby([groupon, 'month'])[valuefield].mean()
    monthly_medians = monthly_stats.groupby([groupon, 'month'])[valuefield].median()
    annual_mean = monthly_means.groupby(groupon).mean()
    annual_std = monthly_means.groupby(groupon).std()
    annual_median = monthly_medians.groupby(groupon).median()

    # Merge into main dataframe
    df = df.merge(annual_mean.rename(valuefield + '_annual_mean'), on=groupon, how='left')
    df = df.merge(annual_std.rename(valuefield + '_annual_std'), on=groupon, how='left')
    df = df.merge(annual_median.rename(valuefield + '_annual_median'), on=groupon, how='left')

    # Anomaly calculations
    df[valuefield + '_anomaly_fr_mean'] = df[valuefield] - df[valuefield + '_annual_mean']
    df[valuefield + '_anomaly_fr_median'] = df[valuefield] - df[valuefield + '_annual_median']
    df[valuefield + '_anomaly_fr_mean_std_norm'] = (
            df[valuefield + '_anomaly_fr_mean'] / df[valuefield + '_annual_std']
    )
    df[valuefield + '_anomaly_fr_median_std_norm'] = (
            df[valuefield + '_anomaly_fr_median'] / df[valuefield + '_annual_std']
    )
    # df[valuefield + '_log_anomaly_fr_mean_std_norm'] = np.log1p(
    #     df[valuefield + '_anomaly_fr_mean_std_norm'] - df[valuefield + '_anomaly_fr_mean_std_norm'].min()
    # )

    df[valuefield + '_log_anomaly_fr_mean_std_norm'] = np.log(
        df[valuefield + '_anomaly_fr_mean_std_norm'] - df[valuefield + '_anomaly_fr_mean_std_norm'].min() + 0.0001
    )

    return df


def aggregate_means(df, value_fields, groupby_time='closest_ecospace_time', level='monthly'):
    """
    Aggregate fields by either month or week.

    Parameters:
        df : pd.DataFrame
        value_fields : list of str
            Columns to aggregate.
        groupby_time : str
            Datetime field to group by.
        level : str
            One of {'monthly', 'weekly'}.

    Returns:
        pd.DataFrame
    """
    df = df.copy()
    df[groupby_time] = pd.to_datetime(df[groupby_time])
    df['year'] = df[groupby_time].dt.year

    if level == 'monthly':
        df['period'] = df[groupby_time].dt.month
        group_keys = ['year', 'period']
    elif level == 'weekly':
        df['period'] = df[groupby_time].dt.isocalendar().week
        group_keys = ['year', 'period']
    else:
        raise ValueError("Unsupported aggregation level. Use 'monthly' or 'weekly'.")

    grouped = df.groupby(group_keys)[value_fields].mean().reset_index()
    grouped.columns = group_keys + [f'mean_{col}' for col in value_fields]
    return grouped


# NOT USED YET - 2025-05-29
# nemcek et al (2023) define a more thoughtful 'season'
# this follows their approach
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


def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return np.nan


def compute_model_stats(obs, model, seasons=None):
    """
    Compute model-vs-observation statistics.
    If `seasons` is provided, compute stats per season.

    Parameters:
        obs : np.ndarray
        model : np.ndarray
        seasons : array-like of str or None

    Returns:
        list of dicts (seasonal) or single dict (overall)
    """
    if seasons is not None:
        if len(seasons) != len(obs):
            raise ValueError(f"Length mismatch: len(seasons)={len(seasons)}, len(obs)={len(obs)}")

        results = []
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            mask = (np.array(seasons) == season) & ~np.isnan(obs) & ~np.isnan(model)
            if np.sum(mask) < 2:
                continue
            stat = compute_model_stats(obs[mask], model[mask])  # recursive
            stat['Season'] = season
            results.append(stat)
        return results

    # === Overall stats ===
    mask = ~np.isnan(obs) & ~np.isnan(model)
    obs = obs[mask]
    model = model[mask]

    if len(obs) < 2:
        return {}

    obs_std = np.std(obs)
    model_std = np.std(model)
    r, _ = pearsonr(obs, model)
    bias = np.mean(model - obs)
    mae = mean_absolute_error(obs, model)
    rmse = np.sqrt(np.mean((model - obs) ** 2))
    centered_rmse = np.sqrt(np.mean(((model - np.mean(model)) - (obs - np.mean(obs))) ** 2))
    obs_mean = np.mean(obs)
    denom = np.sum((np.abs(model - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    wss = 1 - (np.sum((model - obs) ** 2) / denom) if denom != 0 else np.nan

    return {
        'R': r,
        'obs_std': obs_std,
        'model_std': model_std,
        'centered_RMSE': centered_rmse,
        'RMSE': rmse,
        'Bias': bias,
        'MAE': mae,
        'WSS': wss,
        'N': len(obs)
    }


def run_model_statistics(df, model_obs_pairs,
                         use_anomalies=True,
                         aggregation_level='monthly',
                         seasons_to_exclude=None):
    """
    Compute model-vs-observation statistics.

    Parameters:
        df : pd.DataFrame
        model_obs_pairs : list of dicts
        use_anomalies : bool
            If True, uses normalized anomaly fields.
        aggregation_level : str
            One of {'monthly', 'weekly', 'none'}.

    Returns:
        pd.DataFrame
    """
    results = []

    if seasons_to_exclude is None:
        season_tag = "All"
    else:
        seasons_tag = ""
        for se in seasons_to_exclude:
            seasons_tag = seasons_tag + se
        season_tag = f"All_except_{seasons_tag}"

    for pair in model_obs_pairs:
        taxon = pair['taxon_filter']
        group = pair['group']
        obs_field = pair['obs']
        ecospace_field = pair['ecospace']
        ssc_field = pair['ssc']

        if use_anomalies:
            obs_field += '_anomaly_fr_mean_std_norm'
            ecospace_field += '_anomaly_fr_mean_std_norm'
            if ssc_field:
                ssc_field += '_anomaly_fr_mean_std_norm'

        if isinstance(taxon, list):
            df_sub = df[df['class'] == group]
        else:
            df_sub = df[df['class'] == taxon]
        df_sub = df_sub[pd.to_datetime(df_sub['closest_ecospace_time']).dt.year >= 1980]

        if aggregation_level in ['monthly', 'weekly']:
            value_fields = [obs_field, ecospace_field] + ([ssc_field] if ssc_field else [])
            grouped = aggregate_means(df_sub, value_fields, level=aggregation_level)
            grouped['season'] = grouped['period'].map(assign_season)

            obs_vals = grouped.get(f'mean_{obs_field}', pd.Series(dtype=float)).values
            eco_vals = grouped.get(f'mean_{ecospace_field}', pd.Series(dtype=float)).values
            ssc_vals = grouped.get(f'mean_{ssc_field}', pd.Series(dtype=float)).values if ssc_field else None
            seasons = grouped['season'].values

        elif aggregation_level == 'none':
            df_sub['season'] = df_sub['month'].map(assign_season)
            obs_vals = df_sub[obs_field].values
            eco_vals = df_sub[ecospace_field].values
            ssc_vals = df_sub[ssc_field].values if ssc_field else None
            seasons = df_sub['season'].values

        else:
            raise ValueError("aggregation_level must be 'monthly', 'weekly', or 'none'.")

        # === Overall stats ===
        if seasons_to_exclude:
            valid_mask = ~np.isin(seasons, seasons_to_exclude)
            obs_vals_all = obs_vals[valid_mask]
            eco_vals_all = eco_vals[valid_mask]
            if ssc_vals is not None:
                ssc_vals_all = ssc_vals[valid_mask]
        else:
            obs_vals_all = obs_vals
            eco_vals_all = eco_vals
            if ssc_vals is not None:
                ssc_vals_all = ssc_vals

        if len(obs_vals_all) >= 2 and len(eco_vals_all) >= 2:
            eco_all = compute_model_stats(obs_vals_all, eco_vals_all)
            eco_all.update({'Group': group, 'Model': 'Ecospace', 'Season': season_tag})
            results.append(eco_all)

        if len(obs_vals) >= 2 and len(eco_vals) >= 2:
            seasonal_stats = compute_model_stats(obs_vals, eco_vals, seasons=seasons)
            for stat in seasonal_stats:
                stat.update({'Group': group, 'Model': 'Ecospace'})
                results.append(stat)

        if ssc_field and ssc_vals_all is not None and len(obs_vals_all) >= 2 and len(ssc_vals_all) >= 2:
            ssc_all = compute_model_stats(obs_vals_all, ssc_vals_all)
            ssc_all.update({'Group': group, 'Model': 'SSC', 'Season': season_tag})
            results.append(ssc_all)

        if ssc_field and ssc_vals is not None and len(obs_vals) >= 2 and len(ssc_vals) >= 2:
            seasonal_stats = compute_model_stats(obs_vals, ssc_vals, seasons=seasons)

            if isinstance(seasonal_stats, dict):
                seasonal_stats = [seasonal_stats]

            for stat in seasonal_stats:
                stat.update({'Group': group, 'Model': 'SSC'})
                results.append(stat)

    return pd.DataFrame(results)


def aggregate_observations_by_class(df, class_filter, value_col='measurementValue', id_col='id_x', time_col='closest_ecospace_time'):
    """
    Aggregate observation data by summing values across species for each sample (station-time combo).
    Keeps model data as-is (uses .first()).

    Parameters:
        df : pd.DataFrame - raw dataset
        class_filter : str or list of str - e.g., 'Bacillariophyceae'
        value_col : str - column with observation values (e.g., 'measurementValue')
        id_col : str - unique sample/station ID
        time_col : str - time matching field

    Returns:
        pd.DataFrame - aggregated by sample with model fields preserved
    """
    df_sub = df[df['class'].isin([class_filter] if isinstance(class_filter, str) else class_filter)].copy()

    # Fields to keep from model output
    model_cols = ['PP1-DIA', 'PP2-NAN', 'PP3-PIC', 'PZ2-DIN', 'ssc-DIA', 'ssc-FLA', 'ssc-CIL', time_col]

    agg_df = df_sub.groupby([id_col, time_col]).agg({
        value_col: 'sum',
        **{col: 'first' for col in model_cols if col in df.columns}
    }).reset_index()

    # Rename for compatibility
    agg_df['QU39'] = agg_df[value_col]
    agg_df['class'] = class_filter  # assign aggregated class label for downstream processing

    return agg_df


def plot_monthly_boxplot(
    df,
    field_codes,
    taxon_name,
    groupon,
    plot_label,
    source_names,
    color_map,
    output_file=None,
    include_ssc=False,
    label_position='(a)',
    use_log_anomaly=False
):
    """
    Generate a monthly boxplot comparing anomalies from different sources.

    Parameters:
        df : pd.DataFrame
            Data with anomaly fields already calculated.
        field_codes : list of str
            Fields to plot (e.g., ['log_PP1-DIA', 'logQU39', 'log_ssc-DIA']).
        taxon_name : str
            Name of the taxon to filter for (used with groupon column).
        groupon : str
            Column to group/filter by (e.g., 'class').
        plot_label : str
            Text label for the plot title and output filename.
        source_names : list of str
            Names to show in the legend (should match field_codes order).
        color_map : dict
            Dictionary mapping field_codes to plot colors.
        output_file : str or None
            Full file path to save the plot. If None, does not save.
        include_ssc : bool
            If True, includes the SSC model in plot alignment.
        label_position : str
            Text label to show in plot corner (e.g., '(a)', '(b)').
        use_log_anomaly : bool
            If True, use the log-transformed normalized anomaly field.
    """

    # Setup
    months = np.arange(1, 13)
    if include_ssc and len(field_codes) == 3:
        positions = [months - 0.25, months + 0.25, months]
    elif len(field_codes) == 2:
        positions = [months - 0.2, months + 0.2]
    else:
        positions = [months]

    widths = 0.35 if len(field_codes) == 2 else 0.2

    fig, ax = plt.subplots(figsize=(10, 6) if len(field_codes) == 2 else (10, 6))

    # Filter for taxon of interest
    subset = df[df[groupon].isin([taxon_name] if isinstance(taxon_name, str) else taxon_name)]
    condition = subset[field_codes[0] + '_anomaly_fr_mean'] > FILL_VALUE
    subset = subset.loc[condition]

    # Organize data for plotting
    data_by_month = {month: {} for month in months}
    for field in field_codes:
        for month in months:
            mdata = subset[subset['month'] == month]
            field_to_plot = field + ('_log_anomaly_fr_mean_std_norm' if use_log_anomaly else '_anomaly_fr_mean_std_norm')
            data_by_month[month][field] = mdata[field_to_plot].dropna().values

    # Plot each group
    for i, field in enumerate(field_codes):
        for month in months:
            plt.boxplot(
                data_by_month[month].get(field, []),
                positions=[positions[i][month - 1]],
                widths=widths,
                patch_artist=True,
                boxprops=dict(facecolor=color_map[field], color='black'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                showfliers=False
            )

    # Labels and legend
    plt.title(f'Anomaly from Means Rel to Std Dev by Month for {plot_label} {MODEL_RUN}')
    plt.xlabel('Month')
    plt.ylabel('Anomaly (z-score)')
    plt.xticks(months, [calendar.month_abbr[m] for m in months])
    handles = [plt.Line2D([0], [0], color=color_map[f], lw=4) for f in field_codes]
    plt.legend(handles, source_names, loc='upper right')
    plt.text(0.02, 0.95, label_position, transform=ax.transAxes, fontsize=14, verticalalignment='top')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"[Saved] {output_file}")

    if SHOW_PLTS:
        plt.show()


def plot_log_histogram(df, field, group_filter, group_column, title, output_file):
    """
    Plot log-scaled histogram for a given field and taxonomic group.

    Parameters:
        df : pd.DataFrame
        field : str - column name for value (e.g., 'QU39', 'PP1-DIA')
        group_filter : str - taxonomic group (e.g., 'Bacillariophyceae')
        group_column : str - column name for group ('class', etc.)
        title : str - title of the plot
        output_file : str - path to save the plot
    """
    subset = df[df[group_column] == group_filter]
    values = subset[field] + 1  # avoid log(0)
    bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 50)

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, alpha=0.7, label=group_filter)
    plt.xscale('log')
    plt.xlabel('Measurement Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"[Saved] {output_file}")

    if SHOW_PLTS:
        plt.show()


def plot_monthly_sample_counts(df, group_filter, group_column, output_file):
    """
    Plot a bar chart of unique sample station counts by month for a given group.

    Parameters:
        df : pd.DataFrame
        group_filter : str - e.g. 'Bacillariophyceae'
        group_column : str - e.g. 'class'
        output_file : str - path to save the plot
    """
    subset = df[df[group_column] == group_filter]

    # Count unique station/sample IDs by month
    unique_counts = subset.groupby('month')['id_x'].nunique().reindex(np.arange(1, 13), fill_value=0)

    plt.figure(figsize=(10, 5))
    plt.bar(unique_counts.index, unique_counts.values, color='skyblue', edgecolor='black')
    plt.title(f'Unique Sample Stations by Month for {group_filter}')
    plt.xlabel('Month')
    plt.ylabel('Unique Sample Count')
    plt.xticks(unique_counts.index, [calendar.month_abbr[m] for m in unique_counts.index])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"[Saved] {output_file}")

    if SHOW_PLTS:
        plt.show()


def plot_combined_panel(dfs, field_codes_list, taxon_names, taxon_labels, labels, source_names, color_map, output_file):
    import matplotlib.pyplot as plt
    import calendar

    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.17)
    fig.text(0.004, 0.5, 'Normalized Anomaly', va='center', rotation='vertical', fontsize=10)
    axes = axes.flatten()
    months = list(range(1, 13))

    for i, (df, fields, taxon, taxon_label, label) in enumerate(zip(dfs, field_codes_list, taxon_names, taxon_labels, labels)):
        ax = axes[i]
        subset = df[df['class'] == taxon]
        num_fields = len(fields)
        if num_fields == 2:
            pos_offset = 0.35
            width = 0.3
        else:
            pos_offset = 0.25
            width = 0.2
        for j, field in enumerate(fields):
            for month in months:
                mdata = subset[subset['month'] == month]
                y = mdata[f'{field}_anomaly_fr_mean_std_norm'].dropna()
                pos = month + j * pos_offset - pos_offset
                if len(y) > 0:
                    ax.boxplot(y, positions=[pos], widths=width,
                               patch_artist=True,
                               boxprops=dict(facecolor=color_map[field], color='black', linewidth=0.8),
                               medianprops=dict(color='black', linewidth=0.8),
                               whiskerprops=dict(color='black', linewidth=0.8),
                               capprops=dict(color='black', linewidth=0.8),
                               showfliers=False)
        ax.set_title(f'{taxon_label}')
        ax.text(0.3, 7, f'{label}', fontsize=10) # (a), (b) etc
        ax.set_xticks(months)
        ax.set_xticklabels([calendar.month_abbr[m] for m in months])
        ax.grid(True)

    handles = [plt.Line2D([0], [0], color=color_map[f], lw=4) for f in field_codes_list[0]]
    # fig.legend(handles, source_names, loc='upper left', bbox_to_anchor=(0.37, 0.95), ncol=1, frameon=True)
    fig.tight_layout()
    fig.savefig(output_file)
    print(f"[Saved] Panel to: {output_file}")

    if SHOW_PLTS:
        plt.show()


def run_qu39_eval() -> None:
    generate_panel = True  # toggle to enable combined 2x2 panel output

    print("Loading dataset...")
    df = pd.read_csv(pathfile_QU39_SSC_Ecospace)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
    df = df[pd.to_datetime(df['closest_ecospace_time']).dt.year >= 1980]

    # Derive values
    df['QU39'] = df['measurementValue']

    # Diatoms
    df_dia = prepare_group_dataset(df, 'Bacillariophyceae', model_fields=['PP1-DIA', 'ssc-DIA'])

    # Nanoflagellates
    nanogroup = [
        'Choanoflagellatea', 'Dictyochophyceae', 'Cryptophyceae', 'Metromonadea',
        'Chrysophyceae', 'Telonemea', 'Chlorodendrophyceae', 'Bicosoecophyceae',
        'Xanthophyceae', 'Coccolithophyceae', 'Euglenophyceae', 'Raphidophyceae'
    ]

    df_nan = prepare_group_dataset(df, nanogroup, model_fields=['PP2-NAN', 'ssc-FLA'])
    print("After aggregation, nanophytoplankton samples:", df_nan.shape[0])
    print("Available months:", df_nan['month'].unique())

    # pico
    df_pic = prepare_group_dataset(df, 'Pyramimonadophyceae', model_fields=['PP3-PIC'])

    # dinoflag
    df_dino = prepare_group_dataset(df, 'Dinophyceae', model_fields=['PZ2-DIN'])

    # Diatoms
    if SHOW_SSC_BXPLTS:
        output_file_dia=os.path.join(figs_out_p, 'ecospace_' + MODEL_RUN + '_Diatoms_Monthly_QU39_vs_SSC_2016_2018.png')
        output_file_nan=os.path.join(figs_out_p, 'ecospace_' + MODEL_RUN + '_Nanophytoplankton_Monthly_QU39_vs_Ecospace_vs_SSC_2016_2018.png')
        source_names=['Ecospace', 'SMELT', 'QU39']
        field_codes_dia = ['log_PP1-DIA', 'log_ssc-DIA', 'logQU39']
        field_codes_nan = ['log_PP2-NAN', 'log_ssc-FLA', 'logQU39']
        source_names = ['Ecospace', 'SMELT', 'QU39']
        field_codes_panel = [
            ['log_PP1-DIA', 'log_ssc-DIA', 'logQU39'],
            ['log_PP2-NAN', 'log_ssc-FLA', 'logQU39'],
            ['log_PP3-PIC', 'logQU39'],
            ['log_PZ2-DIN', 'logQU39']
        ]

    else:
        output_file_dia = os.path.join(figs_out_p, 'ecospace_' + MODEL_RUN + '_Diatoms_Monthly_QU39_2016_2018.png')
        output_file_nan = os.path.join(figs_out_p,
                                       'ecospace_' + MODEL_RUN + '_Nanophytoplankton_Monthly_QU39_vs_Ecospace_2016_2018.png')
        source_names = ['SOGEM-LTL', 'Data']
        field_codes_dia = ['log_PP1-DIA', 'logQU39']
        field_codes_nan = ['log_PP2-NAN', 'logQU39']
        field_codes_panel = [
            ['log_PP1-DIA', 'logQU39'],
            ['log_PP2-NAN', 'logQU39'],
            ['log_PP3-PIC', 'logQU39'],
            ['log_PZ2-DIN', 'logQU39']
        ]

    plot_monthly_boxplot(
        df_dia, field_codes_dia, 'Bacillariophyceae', 'class',
        'Diatoms', source_names,
        {'log_PP1-DIA': 'blue', 'log_ssc-DIA': 'pink', 'logQU39': 'orange'},
        output_file=output_file_dia,
        include_ssc=SHOW_SSC_BXPLTS, label_position='(a)'
    )

    # nano
    plot_monthly_boxplot(
        df_nan, field_codes_nan, 'Nanophytoplankton', 'class',
        'Nanophytoplankton', source_names,
        {'log_PP2-NAN': 'blue', 'log_ssc-FLA': 'pink', 'logQU39': 'orange'},
        output_file=output_file_nan,
        include_ssc=SHOW_SSC_BXPLTS, label_position='(b)'
    )

    # Picoplankton
    plot_monthly_boxplot(
        df_pic, ['log_PP3-PIC', 'logQU39'], 'Pyramimonadophyceae', 'class',
        'Picoplankton', ['Ecospace', 'QU39'],
        {'log_PP3-PIC': 'blue', 'logQU39': 'orange'},
        output_file=os.path.join(figs_out_p, 'ecospace_' + MODEL_RUN + '_Picoplankton_Monthly_QU39_vs_Ecospace_2016_2018.png'),
        include_ssc=False, label_position='(c)'
    )

    # Dinoflagellates
    plot_monthly_boxplot(
        df_dino, ['log_PZ2-DIN', 'logQU39'], 'Dinophyceae', 'class',
        'Dinoflagellates', ['Ecospace', 'QU39'],
        {'log_PZ2-DIN': 'blue', 'logQU39': 'orange'},
        output_file=os.path.join(figs_out_p, 'ecospace_' + MODEL_RUN + '_Dinoflagellates_Monthly_QU39_vs_Ecospace_2016_2018.png'),
        include_ssc=False, label_position='(d)'
    )

    if generate_panel:
        panel_output_file = os.path.join(figs_out_p, f'ecospace_{MODEL_RUN}_Monthly_QU39_Boxplot_Panel.png')
        plot_combined_panel(
            dfs=[df_dia, df_nan, df_pic, df_dino],
            field_codes_list=field_codes_panel,
            taxon_names=['Bacillariophyceae', 'Nanophytoplankton', 'Pyramimonadophyceae', 'Dinophyceae'],
            taxon_labels=['Diatoms (PP1-DIA)', 'Nanophytoplankton (PP2-NAN)',
                          'Picophytoplankton (PP3-PIC)', 'Dinoflagellates (PZ2-DIN)'],
            labels=['(a)', '(b)', '(c)', '(d)'],
            source_names=source_names,
            color_map={
                'log_PP1-DIA': 'blue', 'log_ssc-DIA': 'pink', 'logQU39': 'orange',
                'log_PP2-NAN': 'blue', 'log_ssc-FLA': 'pink',
                'log_PP3-PIC': 'blue', 'log_PZ2-DIN': 'blue'
            },
            output_file=panel_output_file
        )

    # Histograms
    if DO_HIST:
        plot_log_histogram(df_dia, 'QU39', 'Bacillariophyceae', 'class', 'Distribution - QU39', figs_out_p + 'Histo_QU39_Diatoms.png')
        plot_log_histogram(df_dia, 'PP1-DIA', 'Bacillariophyceae', 'class', 'Distribution - Ecospace', figs_out_p + 'Histo_PP1_DIA.png')
        plot_log_histogram(df_dia, 'ssc-DIA', 'Bacillariophyceae', 'class', 'Distribution - SSC', figs_out_p + 'Histo_SSC_DIA.png')

        # Monthly sample count
        plot_monthly_sample_counts(df_dia, 'Bacillariophyceae', 'class', figs_out_p + 'Monthly_Sample_Count_Diatoms.png')

    print("✅ All plots generated.")

    # ----------------------------------------
    # Define group-specific input to stats
    # ----------------------------------------

    model_obs_pairs = [
        {
            'group': 'Diatoms',
            'taxon_filter': 'Bacillariophyceae',
            'obs': 'logQU39',
            'ecospace': 'log_PP1-DIA',
            'ssc': 'log_ssc-DIA',
            'df': df_dia
        },
        {
            'group': 'Nanophytoplankton',
            'taxon_filter': nanogroup,
            'obs': 'logQU39',
            'ecospace': 'log_PP2-NAN',
            'ssc': 'log_ssc-FLA',
            'df': df_nan
        },
        {
            'group': 'Picoplankton',
            'taxon_filter': 'Pyramimonadophyceae',
            'obs': 'logQU39',
            'ecospace': 'log_PP3-PIC',
            'ssc': None,
            'df': df_pic
        },
        {
            'group': 'Dinoflagellates',
            'taxon_filter': 'Dinophyceae',
            'obs': 'logQU39',
            'ecospace': 'log_PZ2-DIN',
            'ssc': None,
            'df': df_dino
        }
    ]

    # ----------------------------------------
    # Run model-vs-observation statistics
    # ----------------------------------------

    # crosscheck for nan
    expected_cols = ['logQU39_anomaly_fr_mean_std_norm', 'log_PP2-NAN_anomaly_fr_mean_std_norm']
    print("Columns in df_nan:", df_nan.columns)
    missing = [col for col in expected_cols if col not in df_nan.columns]
    print("Missing columns:", missing)

    results = []
    for pair in model_obs_pairs:
        stats = run_model_statistics(
            pair['df'],
            model_obs_pairs=[{
                'group': pair['group'],
                'taxon_filter': pair['taxon_filter'],
                'obs': pair['obs'],
                'ecospace': pair['ecospace'],
                'ssc': pair['ssc']
            }],
            use_anomalies=True,
            aggregation_level='monthly',
            seasons_to_exclude = None # NOT WORKING, meant for pooled stats to exclude season optionally
        )
        results.append(stats)

    # Combine and save
    stats_df = pd.concat(results, ignore_index=True)
    print('stats generated!')
    print(stats_df)

    stats_outfile = os.path.join(stats_out_p, f"ecospace_{MODEL_RUN}_QU39_ModelStats_ObsVsModel.csv")

    stats_df = pd.concat(results, ignore_index=True)
    print('stats generated!')
    print(stats_df)

    # Round numeric statistics for export
    round_cols = ['R', 'obs_std', 'model_std', 'centered_RMSE',
                  'RMSE', 'Bias', 'MAE', 'WSS']

    for col in round_cols:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].round(2)

    if 'N' in stats_df.columns:
        stats_df['N'] = stats_df['N'].astype(int)

    stats_outfile = os.path.join(stats_out_p, f"ecospace_{MODEL_RUN}_QU39_ModelStats_ObsVsModel.csv")
    stats_df.to_csv(stats_outfile, index=False)
    print(f"[Saved] Model statistics written to: {stats_outfile}")


if __name__ == "__main__":
    run_qu39_eval()