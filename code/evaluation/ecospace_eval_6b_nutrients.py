"""ecospace_eval_6b_nutrients.py (REDO)

Goal
----
Infer an Ecospace "free dissolved N" time series (g N m-2) from model biomass pools
that are exported in carbon units (g C m-2), and overlay a biweekly climatology
against depth-integrated nutrient observations produced by ecospace_eval_6a_nutrientsmatched.py.

Key idea
--------
Ecospace tracks nutrients implicitly. We approximate a conserved, vertically-integrated
N budget over the evaluation layer (e.g., 0–20 m):

    N_free(t) = N_free_init + N_bound_init - N_bound(t)

where:
- N_free_init is the initial dissolved inventory (g N m-2) for the layer
  (e.g., from 18 umol/L climatological average -> 3.5 g N m-2).
- N_bound_init is initial N bound in model biomass pools included in the sum (g N m-2)
  (user-provided 1.56 g N m-2, OR computed from Ecopath B_init + C->N factors).
- N_bound(t) is the time-varying N bound in those biomass pools, obtained by converting
  each model group B (g C m-2) to g N m-2 using group-specific C->N multipliers.

Detritus
--------
If detrital pools (DOM/POM) are not exported by Ecospace, they cannot be included in
N_bound(t). For now this script defaults to using only non-detrital groups exported
in the matched table.

If detrital columns exist in NetCDF / matched table later (e.g., DE1-POC / DE2-DOC),
add them to the conversion map and they will automatically be included.

"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import ecospace_eval_config as cfg
import matplotlib

# Whether to show figures interactively (default False for batch/HPC runs)
NU_SHOW_PLOT = bool(getattr(cfg, "NU_SHOW_PLOT", False))

# Use a non-interactive backend unless explicitly showing plots
if not NU_SHOW_PLOT:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt



# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SCENARIO = cfg.ECOSPACE_SC
OUTPUT_DIR_EVAL = cfg.EVALOUT_P
OUTPUT_DIR_FIGS = cfg.FIGS_P

# for compensating for how ecospace does not include detrital outputs
# but I want to consider POC to contain nutrients that are 'bound' (vs unbound DOC)
# this requires exporting POC direct from EwE Ecospace run output window (1D)
NU_INCLUDE_POC_DIAG = bool(getattr(cfg, "NU_INCLUDE_POC_DIAG", False))
NU_POC_RELATIVE_CSV = getattr(cfg, "NU_POC_RELATIVE_CSV", None)
ES_GROUPS_ECOPATH_B = getattr(cfg, "ES_GROUPS_ECOPATH_B", {}) or {}
NU_POC_BASELINE_GC_M2 = ES_GROUPS_ECOPATH_B.get("DE1-POC", None)
NU_POC_REL_COL = getattr(cfg, "NU_POC_REL_COL", "DE1-POC")
NU_POC_DIAG_COL_SUBSTR = str(getattr(cfg, "NU_POC_DIAG_COL_SUBSTR", "DE1-POC"))

NU_USE_FLUX_MULT = bool(getattr(cfg, "NU_USE_FLUX_MULT", False))
NU_FLUX_MULT_COL = str(getattr(cfg, "NU_FLUX_MULT_COL", "N_FLUX_MULT"))
NU_FLUX_MULT_FILL = getattr(cfg, "NU_FLUX_MULT_FILL", None)
NU_FLUX_APPLY_MODE = str(getattr(cfg, "NU_FLUX_APPLY_MODE", "scale_free")).lower()


# Produced by ecospace_eval_6a_nutrientsmatched.py
PAIRED_CAST_CSV = os.path.join(
    OUTPUT_DIR_EVAL,
    f"ecospace_{SCENARIO}_nutrients_cast_model_matched.csv",
)

PAIRED_BIWEEK_CSV = os.path.join(
    OUTPUT_DIR_EVAL,
    f"ecospace_{SCENARIO}_nutrients_biweek_model_matched.csv",
)

BOX_BIWEEK_CSV = os.path.join(
    OUTPUT_DIR_EVAL,
    f"ecospace_{SCENARIO}_nutrients_biweek_model_box.csv",
)

NU_PRIMARY_PAIRED_SERIES = str(getattr(cfg, "NU_PRIMARY_PAIRED_SERIES", "matched_cast" if os.path.exists(PAIRED_CAST_CSV) else "matched"))
NU_PLOT_MODEL_SERIES = list(getattr(cfg, "NU_MODEL_SERIES", [NU_PRIMARY_PAIRED_SERIES]))

# Optional year filter
NU_PLT_YR_ST = getattr(cfg, "NU_PLT_YR_ST", None)
NU_PLT_YR_EN = getattr(cfg, "NU_PLT_YR_EN", None)  # exclusive

# Obs statistic within (year, biweekly) across cells
OBS_AVG_TYPE = getattr(cfg, "OBS_AVG_TYPE", getattr(cfg, "NU_OBS_AVG_TYPE", "mean"))  # mean|median
MODEL_AVG_TYPE = getattr(cfg, "MODEL_AVG_TYPE", "mean")  # mean|median

# ---- Conversion factors (g N per g C) ----
C_TO_N_LIVING = float(getattr(cfg, "NU_C_TO_N_LIVING", 0.176))
C_TO_N_DOM = float(getattr(cfg, "NU_C_TO_N_DOM", 0.15))
C_TO_N_POM = float(getattr(cfg, "NU_C_TO_N_POM", 0.07))

# Optional explicit overrides per group name, e.g.:
# NU_GROUP_C_TO_N = {"BA1-BAC": 0.20}
# Merge per-group overrides from either name (some configs use NU_C_TO_N_BY_GROUP)
NU_GROUP_C_TO_N: Dict[str, float] = {
    **dict(getattr(cfg, 'NU_C_TO_N_BY_GROUP', {}) or {}),
    **dict(getattr(cfg, 'NU_GROUP_C_TO_N', {}) or {}),
}

# Candidate detrital column names if they exist
DOM_COL_CANDIDATES = list(getattr(cfg, "NU_DOM_COLS", ["DE2-DOC", "DE1-DOC", "DE1-DOM", "DE2-DOM"]))
POM_COL_CANDIDATES = list(getattr(cfg, "NU_POM_COLS", ["DE1-POC", "DE2-POC", "DE1-POM", "DE2-POM"]))

# ---- Initial inventory terms (g N m-2) ----
# Prefer explicit values if aadded them to ecospace_eval_config.py.
# (If only have a concentration in umol/L, convert to an areal inventory in gN/m2 first.)
NU_FREE_INIT_GNM2 = float(getattr(cfg, "NU_FREE_INIT_GNM2", 3.5))

# If provided, use directly. Otherwise compute from cfg.ES_GROUPS_ECOPATH_B.
NU_BOUND_INIT_GNM2 = getattr(cfg, "NU_BOUND_INIT_GNM2", None)

# Plot display toggle (interactive). If running headless/HPC, set False.

# If True, force climatology index to include all biweekly bins 1..NU_BIWEEK_MAX
NU_FORCE_FULL_BIWEEK_AXIS = bool(getattr(cfg, "NU_FORCE_FULL_BIWEEK_AXIS", True))
NU_BIWEEK_MAX = int(getattr(cfg, "NU_BIWEEK_MAX", 26))

# Aggregation behavior:
# - True (default): keep only rows where BOTH obs and model exist for that (cell,time)
# - False: allow model-only rows into model climatology and obs-only rows into obs climatology
NU_REQUIRE_BOTH_FOR_AGG = bool(getattr(cfg, "NU_REQUIRE_BOTH_FOR_AGG", True))

# Debug toggles
NU_DEBUG_POC = bool(getattr(cfg, 'NU_DEBUG_POC', False))
NU_DEBUG_SAVE_DF = bool(getattr(cfg, 'NU_DEBUG_SAVE_DF', False))
NU_DEBUG_SAVE_MAX_ROWS = int(getattr(cfg, 'NU_DEBUG_SAVE_MAX_ROWS', 2000))


# Which model groups to treat as biomass pools contributing to N_bound(t).
# If not set, we use the intersection of matched-table columns with:
#   cfg.ES_GROUPS_RELB (preferred) or cfg.NU_INCLUDE_GRPS (fallback)
NU_MODEL_GROUPS = list(getattr(cfg, "NU_MODEL_GROUPS", []))

# When True, include detrital columns if present in the matched table.
NU_INCLUDE_DETRITUS_IF_AVAILABLE = bool(getattr(cfg, "NU_INCLUDE_DETRITUS_IF_AVAILABLE", True))

NU_BOUND_INIT_MODE = getattr(cfg, "NU_BOUND_INIT_MODE", "ecopath")
NU_T0_SPATIAL_REDUCER = getattr(cfg, "NU_T0_SPATIAL_REDUCER", "median")
NU_T0_PER_SERIES = bool(getattr(cfg, "NU_T0_PER_SERIES", True))
NU_FREE_INIT_MODE = getattr(cfg, "NU_FREE_INIT_MODE", "config")
NU_DEBUG_INIT_ANCHOR = bool(getattr(cfg, "NU_DEBUG_INIT_ANCHOR", False))


# --- Optional: overlay Ecosim (1D) nutrient series on the same plot ---
NU_OVERLAY_ECOSIM = bool(getattr(cfg, "NU_OVERLAY_ECOSIM", False))
NU_ECOSIM_NUTRIENTS_CSV = getattr(cfg, "NU_ECOSIM_NUTRIENTS_CSV", None)
NU_ECOSIM_MODEL_COL = str(getattr(cfg, "NU_ECOSIM_MODEL_COL", "model_N_free_used_gN_m2"))
NU_ECOSIM_LABEL = str(getattr(cfg, "NU_ECOSIM_LABEL", "ecosim"))



# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def _compute_skill_metrics(
    df: pd.DataFrame,
    *,
    obs_col: str = "avg_obs",
    model_col: str = "avg_model",
) -> pd.DataFrame:
    """
    Compute standard paired skill metrics from a paired table.
    Expects one row per paired comparison unit (e.g., year x biweekly).
    """
    d = df[[obs_col, model_col]].copy()
    d[obs_col] = pd.to_numeric(d[obs_col], errors="coerce")
    d[model_col] = pd.to_numeric(d[model_col], errors="coerce")
    d = d.dropna(subset=[obs_col, model_col]).copy()

    if d.empty:
        return pd.DataFrame([{
            "n": 0,
            "bias": np.nan,
            "obs_std": np.nan,
            "model_std": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "r": np.nan,
            "WSS": np.nan,
        }])

    obs = d[obs_col].to_numpy(dtype=float)
    mod = d[model_col].to_numpy(dtype=float)
    resid = mod - obs

    n = len(d)
    bias = float(np.mean(resid))
    obs_std = float(np.std(obs, ddof=1)) if n > 1 else np.nan
    model_std = float(np.std(mod, ddof=1)) if n > 1 else np.nan
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r = float(np.corrcoef(obs, mod)[0, 1]) if n > 1 else np.nan

    obs_mean = float(np.mean(obs))
    denom = np.sum((np.abs(mod - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    wss = float(1.0 - np.sum((mod - obs) ** 2) / denom) if denom > 0 else np.nan

    return pd.DataFrame([{
        "n": n,
        "bias": bias,
        "obs_std": obs_std,
        "model_std": model_std,
        "MAE": mae,
        "RMSE": rmse,
        "r": r,
        "WSS": wss,
    }])


SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
SEASON_COLORS = {
    "Winter": "tab:blue",
    "Spring": "tab:green",
    "Summer": "tab:orange",
    "Fall": "tab:red",
}


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def _biweekly_to_season(biweekly: float) -> str | float:
    if pd.isna(biweekly):
        return np.nan
    day_of_year = int((int(biweekly) - 1) * 14 + 7)
    ts = pd.Timestamp("2001-01-01") + pd.to_timedelta(day_of_year - 1, unit="D")
    return _month_to_season(int(ts.month))


def _infer_season_series(df: pd.DataFrame) -> pd.Series:
    if "season" in df.columns:
        return df["season"].astype(str).str.capitalize()
    if "Season" in df.columns:
        return df["Season"].astype(str).str.capitalize()
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        return dt.dt.month.map(lambda m: _month_to_season(int(m)) if pd.notna(m) else np.nan)
    if "model_time" in df.columns:
        dt = pd.to_datetime(df["model_time"], errors="coerce")
        return dt.dt.month.map(lambda m: _month_to_season(int(m)) if pd.notna(m) else np.nan)
    if "biweekly" in df.columns:
        bw = pd.to_numeric(df["biweekly"], errors="coerce")
        return bw.map(_biweekly_to_season)
    return pd.Series(np.nan, index=df.index, dtype=object)


def _add_best_fit_line(ax, x: np.ndarray, y: np.ndarray, *, color: str, linewidth: float = 1.4, linestyle: str = "-") -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 2:
        return
    if np.nanmin(x) == np.nanmax(x):
        return
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    yy = slope * xx + intercept
    ax.plot(xx, yy, color=color, linewidth=linewidth, linestyle=linestyle, zorder=4)


def _plot_skill_scatter(
    df: pd.DataFrame,
    *,
    obs_col: str = "avg_obs",
    model_col: str = "avg_model",
    out_png: str,
    title: str,
    stats_df: pd.DataFrame | None = None,
):
    """
    Scatter plot from the same paired table used for skill stats.
    Points are colored by season when season can be inferred, with a separate
    best-fit line for each season.
    """
    keep_cols = [obs_col, model_col]
    for extra in ("year", "biweekly", "date", "model_time", "season", "Season"):
        if extra in df.columns and extra not in keep_cols:
            keep_cols.append(extra)
    d = df[keep_cols].copy()

    d[obs_col] = pd.to_numeric(d[obs_col], errors="coerce")
    d[model_col] = pd.to_numeric(d[model_col], errors="coerce")
    d = d.dropna(subset=[obs_col, model_col]).copy()

    if d.empty:
        print(f"[skill scatter] No paired data available for {title}")
        return

    d["season_plot"] = _infer_season_series(d)

    fig, ax = plt.subplots(figsize=(6.2, 5.8))

    lo = np.nanmin([d[obs_col].min(), d[model_col].min()])
    hi = np.nanmax([d[obs_col].max(), d[model_col].max()])
    span = hi - lo
    pad = 0.05 * span if span > 0 else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1, color="0.5", zorder=1)

    plotted_any = False
    for season in SEASON_ORDER:
        g = d[d["season_plot"] == season].copy()
        if g.empty:
            continue
        color = SEASON_COLORS[season]
        ax.scatter(
            g[obs_col],
            g[model_col],
            alpha=0.5,
            s=15,
            color=color,
            label=f"{season} (n={len(g)})",
            zorder=3,
        )
        _add_best_fit_line(
            ax,
            g[obs_col].to_numpy(dtype=float),
            g[model_col].to_numpy(dtype=float),
            color=color,
        )
        plotted_any = True

    if not plotted_any:
        ax.scatter(d[obs_col], d[model_col], alpha=0.5, s=22, zorder=3)
        _add_best_fit_line(ax, d[obs_col].to_numpy(dtype=float), d[model_col].to_numpy(dtype=float), color="k")
    else:
        ax.legend(frameon=True, fontsize=8, loc="upper left")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Observed N (g N m$^{-2}$)")
    ax.set_ylabel("Modelled N (g N m$^{-2}$)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if stats_df is not None and not stats_df.empty:
        s = stats_df.iloc[0]
        txt = (
            f"n = {int(s['n'])}\n"
            f"r = {s['r']:.2f}\n"
            f"RMSE = {s['RMSE']:.2f}\n"
            f"MAE = {s['MAE']:.2f}\n"
            f"WSS = {s['WSS']:.2f}"
        )
        ax.text(
            0.98, 0.02, txt,
            transform=ax.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _prepare_paired_df_for_series(series: str = "matched") -> pd.DataFrame:
    """
    Build the most raw paired Ecospace nutrient table available for one series.

    For 'matched', this is the cell-biweekly matched table with model free-N reconstructed
    row by row. Rows lacking either obs or model are dropped only at the very end.
    """
    df = _load_series_table(series)

    # Best available obs column in g N m-2
    obs_col = _pick_obs_gN_col(df)
    df = _ensure_obs_col(df, obs_col).copy()

    # Infer model-bound N from biomass columns
    group_cols = _infer_group_cols(df)
    c2n = _build_c_to_n_map(group_cols)
    bound_init_ecopath = _compute_bound_init_gN_m2(group_cols, c2n)

    df["model_N_bound_gN_m2"] = _compute_bound_N(df, group_cols, c2n)

    # Optional: add POC diagnostics into bound N
    if NU_INCLUDE_POC_DIAG:
        if not NU_POC_RELATIVE_CSV or not os.path.exists(NU_POC_RELATIVE_CSV):
            raise FileNotFoundError(f"NU_POC_RELATIVE_CSV not found: {NU_POC_RELATIVE_CSV}")

        b0 = dict(getattr(cfg, "ES_GROUPS_ECOPATH_B", {}) or {})
        if "DE1-POC" not in b0:
            raise ValueError("Missing 'DE1-POC' in cfg.ES_GROUPS_ECOPATH_B (baseline gC/m2).")

        B0_poc_gc = float(b0["DE1-POC"])
        c2n_poc = float(NU_GROUP_C_TO_N.get("DE1-POC", C_TO_N_POM))
        rel = _load_diag_relative_series_poc(NU_POC_RELATIVE_CSV, rel_col=NU_POC_REL_COL)

        if "model_t_idx" in df.columns:
            step = pd.to_numeric(df["model_t_idx"], errors="coerce")
        else:
            mt = pd.to_datetime(df["model_time"], errors="coerce")
            uniq = pd.Series(mt.dropna().unique()).sort_values()
            step_map = pd.Series(range(len(uniq)), index=uniq.values)
            step = mt.map(step_map)

        rel_on_rows = step.map(rel).astype(float).fillna(0.0)
        B_poc_gc = B0_poc_gc * (1.0 + rel_on_rows)
        df["model_POC_N_bound_gN_m2"] = B_poc_gc * c2n_poc
        df["model_N_bound_gN_m2"] = df["model_N_bound_gN_m2"] + df["model_POC_N_bound_gN_m2"]

    # Choose bound-init anchor
    if NU_BOUND_INIT_MODE.lower() == "t0_series":
        bound_init_used = _series_t0_bound_init_gN_m2(df)
    elif NU_BOUND_INIT_MODE.lower() == "ecopath":
        bound_init_used = bound_init_ecopath
    else:
        raise ValueError("NU_BOUND_INIT_MODE must be 'ecopath' or 't0_series'")

    # Choose free-init anchor
    free_init = float(NU_FREE_INIT_GNM2)
    if NU_FREE_INIT_MODE.lower() == "t0_preserve_total":
        total_ref = float(NU_FREE_INIT_GNM2 + bound_init_ecopath)
        bound_t0 = bound_init_used if NU_BOUND_INIT_MODE.lower() == "t0_series" else _series_t0_bound_init_gN_m2(df)
        free_init = float(max(0.0, total_ref - bound_t0))
    elif NU_FREE_INIT_MODE.lower() != "config":
        raise ValueError("NU_FREE_INIT_MODE must be 'config' or 't0_preserve_total'")

    # Infer free dissolved N
    df["model_N_free_gN_m2"] = free_init + bound_init_used - df["model_N_bound_gN_m2"]
    df.loc[df["model_N_free_gN_m2"] < 0, "model_N_free_gN_m2"] = 0.0
    model_col_used = "model_N_free_gN_m2"

    # Optional nutrient-flux scaling
    if NU_USE_FLUX_MULT and (NU_FLUX_MULT_COL in df.columns):
        mult = pd.to_numeric(df[NU_FLUX_MULT_COL], errors="coerce")
        if NU_FLUX_MULT_FILL is not None:
            mult = mult.fillna(float(NU_FLUX_MULT_FILL))
        df["model_N_flux_mult"] = mult

        if NU_FLUX_APPLY_MODE == "scale_free":
            df["model_N_free_used_gN_m2"] = df["model_N_free_gN_m2"] * df["model_N_flux_mult"]
        elif NU_FLUX_APPLY_MODE == "scale_total":
            total_init = float(getattr(cfg, "NU_TOTAL_INIT_GNM2", free_init + bound_init_used))
            df["model_N_free_used_gN_m2"] = (total_init * df["model_N_flux_mult"]) - df["model_N_bound_gN_m2"]
        else:
            raise ValueError(f"Unknown NU_FLUX_APPLY_MODE={NU_FLUX_APPLY_MODE!r}")

        df.loc[df["model_N_free_used_gN_m2"] < 0, "model_N_free_used_gN_m2"] = 0.0
        model_col_used = "model_N_free_used_gN_m2"

    # Standardize names used by the skill helpers
    raw = df.copy()
    raw["avg_obs"] = pd.to_numeric(raw[obs_col], errors="coerce")
    raw["avg_model"] = pd.to_numeric(raw[model_col_used], errors="coerce")

    # This is the actual paired table used for raw skill
    raw = raw.dropna(subset=["avg_obs", "avg_model"]).copy()

    return raw


def _write_ecospace_skill_from_series(series: str = "matched") -> tuple[str, str, str, str]:
    """
    Write BOTH:
      1) raw matched skill/stat products
      2) year x biweekly aggregated skill/stat products
    """
    raw = _prepare_paired_df_for_series(series)

    # ---------- RAW MATCHED ----------
    stats_raw = _compute_skill_metrics(raw, obs_col="avg_obs", model_col="avg_model")
    stats_raw.insert(0, "support", "raw_matched")
    stats_raw.insert(0, "series", series)

    out_stats_raw = os.path.join(
        OUTPUT_DIR_EVAL,
        f"ecospace_{SCENARIO}_nutrient_skill_raw_{series}.csv"
    )
    out_pairs_raw = os.path.join(
        OUTPUT_DIR_EVAL,
        f"ecospace_{SCENARIO}_nutrient_skill_pairs_raw_{series}.csv"
    )
    out_scatter_raw = os.path.join(
        OUTPUT_DIR_FIGS,
        f"ecospace_{SCENARIO}_nutrient_scatter_raw_{series}.png"
    )

    raw.to_csv(out_pairs_raw, index=False)
    stats_raw.to_csv(out_stats_raw, index=False)

    _plot_skill_scatter(
        raw,
        obs_col="avg_obs",
        model_col="avg_model",
        out_png=out_scatter_raw,
        title=f"Ecospace nutrient skill ({series}, raw matched)",
        stats_df=stats_raw,
    )

    # ---------- OLD WAY: YEAR x BIWEEK ----------
    year_biweek = _aggregate_year_biweek(
        raw,
        obs_col="avg_obs",
        model_col="avg_model",
        require_both=True,
    )

    stats_ybw = _compute_skill_metrics(year_biweek, obs_col="avg_obs", model_col="avg_model")
    stats_ybw.insert(0, "support", "year_biweek")
    stats_ybw.insert(0, "series", series)

    out_stats_ybw = os.path.join(
        OUTPUT_DIR_EVAL,
        f"ecospace_{SCENARIO}_nutrient_skill_yearbiweek_{series}.csv"
    )
    out_pairs_ybw = os.path.join(
        OUTPUT_DIR_EVAL,
        f"ecospace_{SCENARIO}_nutrient_skill_pairs_yearbiweek_{series}.csv"
    )
    out_scatter_ybw = os.path.join(
        OUTPUT_DIR_FIGS,
        f"ecospace_{SCENARIO}_nutrient_scatter_yearbiweek_{series}.png"
    )

    year_biweek.to_csv(out_pairs_ybw, index=False)
    stats_ybw.to_csv(out_stats_ybw, index=False)

    _plot_skill_scatter(
        year_biweek,
        obs_col="avg_obs",
        model_col="avg_model",
        out_png=out_scatter_ybw,
        title=f"Ecospace nutrient skill ({series}, year × biweekly)",
        stats_df=stats_ybw,
    )

    print(f"[ecospace nutrients] Raw skill stats saved → {out_stats_raw}")
    print(f"[ecospace nutrients] Raw paired table saved → {out_pairs_raw}")
    print(f"[ecospace nutrients] Raw scatter saved → {out_scatter_raw}")
    print(f"[ecospace nutrients] Year×biweekly skill stats saved → {out_stats_ybw}")
    print(f"[ecospace nutrients] Year×biweekly paired table saved → {out_pairs_ybw}")
    print(f"[ecospace nutrients] Year×biweekly scatter saved → {out_scatter_ybw}")

    return out_stats_raw, out_stats_ybw, out_scatter_raw, out_scatter_ybw

def _load_series_table(series: str) -> pd.DataFrame:
    if series == "matched_cast":
        return _load_paired_table(PAIRED_CAST_CSV)
    if series == "matched":
        return _load_paired_table(PAIRED_BIWEEK_CSV)
    if series == "box":
        return _load_paired_table(BOX_BIWEEK_CSV)
    raise ValueError(f"Unknown model series {series!r}. Use 'matched_cast', 'matched', or 'box'.")


def _ensure_obs_col(df: pd.DataFrame, obs_col: str) -> pd.DataFrame:
    # Box tables often have no obs column; we need it present so _aggregate_year_biweek() can run.
    if obs_col in df.columns:
        return df
    d = df.copy()
    d[obs_col] = np.nan
    return d



def _load_diag_relative_series_poc(csv_path: str, *, rel_col: str = "DE1-POC") -> pd.Series:
    """
    Parse Ecospace Diagnostics 'Relative Biomass' CSV and return a Series indexed by model step (int)
    containing the relative anomaly for DE1-POC.

    Assumptions:
      - First row after header contains labels like '22: DE1-POC'
      - Column 'Pane' contains integer model step (0,1,2,...) but includes a non-numeric sentinel at end
      - Values are normalized to 0 at t=0 and represent fractional change relative to baseline:
            rel(t) = B(t)/B0 - 1
        so B(t) = B0 * (1 + rel(t))
    """
    df = pd.read_csv(csv_path, header=0)

    # Row 0 holds the "Data" labels like "22: DE1-POC"
    label_row = df.iloc[0].astype(str)

    # Find the column that corresponds to rel_col (e.g., "DE1-POC")
    target_idx = None
    for j, lab in enumerate(label_row):
        if rel_col in lab:
            target_idx = j
            break
    if target_idx is None:
        raise ValueError(
            f"Could not find rel_col={rel_col!r} in the label row of {csv_path}. "
            f"Example labels: {list(label_row.values)[:8]}"
        )

    # Keep rows after the label row
    d = df.iloc[1:].copy()

    # Clean 'Pane' (model step); drop sentinel/non-numeric rows
    d["Pane_num"] = pd.to_numeric(d["Pane"], errors="coerce")
    d = d.dropna(subset=["Pane_num"])
    d = d[d["Pane_num"] < 1e9]  # drop the huge float sentinel
    d["step"] = d["Pane_num"].astype(int)

    # Extract the series values
    ser = pd.to_numeric(d.iloc[:, target_idx], errors="coerce")
    ser.index = d["step"].values
    ser.name = "poc_rel"

    return ser


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_obs_gN_col(df: pd.DataFrame) -> str:
    """Return the best available obs column in g N m-2."""
    if "avg_nitrogen_gN_m2_obs" in df.columns:
        return "avg_nitrogen_gN_m2_obs"

    # fallback: compute from mmol/m^2 inventory if present
    if "avg_nitrogen_mmol_m2_obs" in df.columns:
        df["avg_nitrogen_gN_m2_obs"] = df["avg_nitrogen_mmol_m2_obs"] * (14.0067 / 1000.0)
        return "avg_nitrogen_gN_m2_obs"

    # older pipelines sometimes used a unitless name
    if "avg_nitrogen_obs" in df.columns:
        return "avg_nitrogen_obs"  # WARNING: unknown units

    raise ValueError(
        "Could not find an obs column to use. Expected 'avg_nitrogen_gN_m2_obs' (preferred) "
        "or 'avg_nitrogen_mmol_m2_obs'."
    )


def _infer_group_cols(df: pd.DataFrame) -> list[str]:
    """Choose model biomass columns to include in N_bound(t)."""

    if NU_MODEL_GROUPS:
        cols = [c for c in NU_MODEL_GROUPS if c in df.columns]
        if not cols:
            raise ValueError(
                f"NU_MODEL_GROUPS was provided but none were found in the matched table. "
                f"First few columns: {list(df.columns[:30])}"
            )
        return cols

    base_candidates: Iterable[str]
    if hasattr(cfg, "ES_GROUPS_RELB"):
        base_candidates = list(getattr(cfg, "ES_GROUPS_RELB"))
    else:
        base_candidates = list(getattr(cfg, "NU_INCLUDE_GRPS", []))

    cols = [c for c in base_candidates if c in df.columns]

    # Optionally include detritus (if present in matched table)
    if NU_INCLUDE_DETRITUS_IF_AVAILABLE:
        for c in DOM_COL_CANDIDATES + POM_COL_CANDIDATES:
            if c in df.columns and c not in cols:
                cols.append(c)

    if not cols:
        raise ValueError(
            "Could not infer any model biomass group columns from the matched table. "
            "Consider setting cfg.NU_MODEL_GROUPS explicitly."
        )

    return cols


def _build_c_to_n_map(group_cols: list[str]) -> Dict[str, float]:
    """Group-specific gN/gC conversion map."""
    c2n = {g: C_TO_N_LIVING for g in group_cols}

    for g in group_cols:
        if g in DOM_COL_CANDIDATES:
            c2n[g] = C_TO_N_DOM
        if g in POM_COL_CANDIDATES:
            c2n[g] = C_TO_N_POM

    # Explicit overrides win
    for g, v in NU_GROUP_C_TO_N.items():
        if g in c2n:
            c2n[g] = float(v)

    return c2n


def _compute_bound_N(df: pd.DataFrame, group_cols: list[str], c2n: Dict[str, float]) -> pd.Series:
    """Compute N_bound(t) from biomass pools (g C m-2 -> g N m-2)."""
    d = df[group_cols].apply(pd.to_numeric, errors="coerce")

    # Defensively clip negative biomass
    d = d.clip(lower=0)

    # Weighted sum with per-column conversion
    out = pd.Series(0.0, index=df.index, dtype=float)
    for g in group_cols:
        out = out + d[g].fillna(0.0) * float(c2n[g])

    # If all groups are NaN in a row, mark bound N as NaN
    all_nan = d.isna().all(axis=1)
    out.loc[all_nan] = np.nan

    return out


def _compute_bound_init_gN_m2(group_cols: list[str], c2n: Dict[str, float]) -> float:
    """Initial bound N for the same pools used in N_bound(t)."""
    if NU_BOUND_INIT_GNM2 is not None:
        return float(NU_BOUND_INIT_GNM2)

    # Compute from Ecopath B_init if available
    if hasattr(cfg, "ES_GROUPS_ECOPATH_B"):
        b0 = dict(getattr(cfg, "ES_GROUPS_ECOPATH_B"))
        s = 0.0
        missing = []
        for g in group_cols:
            if g in b0:
                s += float(b0[g]) * float(c2n[g])
            else:
                missing.append(g)

        # If detrital columns are in group_cols but not in ES_GROUPS_ECOPATH_B, we ignore them here
        # unless NU_BOUND_INIT_GNM2 provided explicitly.
        if missing:
            print(
                "[6b] NOTE: Some group columns are not in cfg.ES_GROUPS_ECOPATH_B and were ignored "
                f"when computing bound_init: {missing}"
            )

        return float(s)

    raise ValueError(
        "NU_BOUND_INIT_GNM2 was not provided and cfg.ES_GROUPS_ECOPATH_B is not available. "
        "Add cfg.NU_BOUND_INIT_GNM2 (g N m-2) or provide ES_GROUPS_ECOPATH_B."
    )


def _load_paired_table(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Paired nutrient table not found: {csv_path}\n"
            "Run ecospace_eval_6a_nutrientsmatched.py first."
        )

    df = pd.read_csv(csv_path)

    # Minimal required structure
    for c in ("year", "biweekly"):
        if c not in df.columns:
            raise ValueError(f"Paired table missing required column: {c!r}")

    df["year"] = df["year"].astype(int)
    df["biweekly"] = df["biweekly"].astype(int)

    if NU_PLT_YR_ST is not None:
        df = df[df["year"] >= int(NU_PLT_YR_ST)]
    if NU_PLT_YR_EN is not None:
        df = df[df["year"] < int(NU_PLT_YR_EN)]

    return df


def _aggregate_year_biweek(df: pd.DataFrame, obs_col: str, model_col: str, *, require_both: Optional[bool] = None) -> pd.DataFrame:

    """Collapse from (year, biweekly, cell) to (year, biweekly).

    If NU_REQUIRE_BOTH_FOR_AGG=True, rows missing either obs or model are dropped.
    If False, we keep obs and model aggregation *independently* by allowing NaNs.

    Output always contains columns:
      year, biweekly, avg_obs, avg_model, n_cells_obs, n_cells_model
    """
    if require_both is None:
        require_both = NU_REQUIRE_BOTH_FOR_AGG

    if OBS_AVG_TYPE not in {"mean", "median"}:
        raise ValueError(f"OBS_AVG_TYPE must be 'mean' or 'median', got: {OBS_AVG_TYPE}")
    if MODEL_AVG_TYPE not in {"mean", "median"}:
        raise ValueError(f"MODEL_AVG_TYPE must be 'mean' or 'median', got: {MODEL_AVG_TYPE}")

    obs_func = "mean" if OBS_AVG_TYPE == "mean" else "median"
    mod_func = "mean" if MODEL_AVG_TYPE == "mean" else "median"

    d = df.copy()

    if require_both:
        d = d.dropna(subset=[obs_col, model_col], how="any")

        out = (
            d.groupby(["year", "biweekly"], as_index=False)
            .agg(
                avg_obs=(obs_col, obs_func),
                avg_model=(model_col, mod_func),
                n_cells_obs=(obs_col, "count"),
                n_cells_model=(model_col, "count"),
            )
            .sort_values(["year", "biweekly"])
            .reset_index(drop=True)
        )
        return out

    # Independent aggregation (do not require both)
    g = d.groupby(["year", "biweekly"], as_index=False)
    out = (
        g.agg(
            avg_obs=(obs_col, obs_func),
            avg_model=(model_col, mod_func),
            n_cells_obs=(obs_col, "count"),
            n_cells_model=(model_col, "count"),
        )
        .sort_values(["year", "biweekly"])
        .reset_index(drop=True)
    )

    # NOTE: with independent aggregation, avg_obs can be NaN where n_cells_obs=0, etc.
    return out


def _compute_climatology(year_biweek: pd.DataFrame) -> pd.DataFrame:
    g = year_biweek.groupby("biweekly")

    clima = pd.DataFrame({"biweekly": sorted(year_biweek["biweekly"].unique())}).set_index("biweekly")

    clima["obs_center"] = g["avg_obs"].mean() if OBS_AVG_TYPE == "mean" else g["avg_obs"].median()
    clima["obs_q10"] = g["avg_obs"].quantile(0.1)
    clima["obs_q90"] = g["avg_obs"].quantile(0.9)

    clima["model_center"] = g["avg_model"].mean() if MODEL_AVG_TYPE == "mean" else g["avg_model"].median()
    clima["model_q10"] = g["avg_model"].quantile(0.1)
    clima["model_q90"] = g["avg_model"].quantile(0.9)

    # context
    clima["n_years_obs"] = g["avg_obs"].count()
    clima["n_years_model"] = g["avg_model"].count()

    # Optionally force 1..NU_BIWEEK_MAX so late-year bins show up as gaps instead of disappearing
    if NU_FORCE_FULL_BIWEEK_AXIS:
        full = pd.Index(range(1, NU_BIWEEK_MAX + 1), name="biweekly")
        clima = clima.reindex(full)

    return clima



def _compute_ecosim_climatology(csv_path: str, *, value_col: str) -> pd.DataFrame:
    """Compute a biweekly climatology from an Ecosim nutrient-series CSV (1D)."""
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("Ecosim CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    df["biweekly"] = ((df["day_of_year"] - 1) // 14 + 1)

    # pick the model series column
    if value_col not in df.columns:
        # common fallbacks
        for alt in ["model_N_free_used_gN_m2", "model_N_free_gN_m2", "N_Free_Adjusted", "N_Free"]:
            if alt in df.columns:
                value_col = alt
                break
        else:
            raise ValueError(f"Could not find model series column '{value_col}' (or fallbacks) in {csv_path}")

    # restrict to the same plot window (if desired)
    if NU_PLT_YR_ST is not None:
        df = df[df["year"] >= int(NU_PLT_YR_ST)]
    if NU_PLT_YR_EN is not None:
        df = df[df["year"] < int(NU_PLT_YR_EN)]

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Reduce within (year, biweekly) first so each year counts equally
    yrbw = df.groupby(["year", "biweekly"], as_index=False).agg(avg_model=(value_col, "mean"))
    # Provide dummy obs column so _compute_climatology stays reusable
    yrbw["avg_obs"] = np.nan

    clima = _compute_climatology(yrbw)
    return clima


def _plot_overlay(clima: pd.DataFrame, *, scenario: str) -> str:
    clima = clima.copy()
    clima.index = clima.index.astype(int)

    x = clima.index.to_numpy()

    plt.figure(figsize=(10, 5))

    # ---- Model ----
    m_fill = np.isfinite(clima["model_q10"].to_numpy()) & np.isfinite(clima["model_q90"].to_numpy())
    m_line = np.isfinite(clima["model_center"].to_numpy())
    if m_fill.any():
        plt.fill_between(
            x,
            clima["model_q10"].to_numpy(),
            clima["model_q90"].to_numpy(),
            where=m_fill,
            interpolate=True,
            alpha=0.2,
            label="Model 10–90%",
        )
    if m_line.any():
        plt.plot(x[m_line], clima["model_center"].to_numpy()[m_line], label="Model center")

    # ---- Obs ----
    o_fill = np.isfinite(clima["obs_q10"].to_numpy()) & np.isfinite(clima["obs_q90"].to_numpy())
    o_line = np.isfinite(clima["obs_center"].to_numpy())
    if o_fill.any():
        plt.fill_between(
            x,
            clima["obs_q10"].to_numpy(),
            clima["obs_q90"].to_numpy(),
            where=o_fill,
            interpolate=True,
            alpha=0.2,
            label="Obs 10–90%",
        )
    if o_line.any():
        plt.plot(x[o_line], clima["obs_center"].to_numpy()[o_line], label="Obs center")

    # Month ticks (as before)
    month_start_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_ticks = [((d - 1) // 14 + 1) for d in month_start_doy]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    ax = plt.gca()
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)

    # Keep the axis extending through the last biweekly bin
    ax.set_xlim(1, max(NU_BIWEEK_MAX, max(month_ticks)))
    ax.set_xlabel("Month")

    plt.ylabel("N (g N m$^{-2}$)")
    plt.title(f"Biweekly nutrient climatology (obs vs inferred model free N) – Ecospace {scenario}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # if NU_SHOW_PLOT:

    plt.show()

    _ensure_dir(OUTPUT_DIR_FIGS)
    out_png = os.path.join(OUTPUT_DIR_FIGS, f"nutrient_climatology_overlay_ecospace{scenario}_freeNredo.png")
    plt.savefig(out_png, dpi=200)


    plt.close()

    return out_png

def _plot_overlay_multi(
    obs_clima: pd.DataFrame,
    model_climas: dict[str, pd.DataFrame],
    *,
    out_png: str,
    scenario: str
) -> None:
    # --- Normalize indices (so month ticks behave the same) ---
    if obs_clima is not None and not obs_clima.empty:
        obs_clima = obs_clima.copy()
        obs_clima.index = obs_clima.index.astype(int)

    plt.figure(figsize=(10, 5))

    # ---- Obs (same logic as _plot_overlay) ----
    if obs_clima is not None and not obs_clima.empty:
        x = obs_clima.index.to_numpy()

        o_fill = np.isfinite(obs_clima["obs_q10"].to_numpy()) & np.isfinite(obs_clima["obs_q90"].to_numpy())
        o_line = np.isfinite(obs_clima["obs_center"].to_numpy())

        if o_fill.any():
            plt.fill_between(
                x,
                obs_clima["obs_q10"].to_numpy(),
                obs_clima["obs_q90"].to_numpy(),
                where=o_fill,
                interpolate=True,
                alpha=0.2,
                label="Obs 10–90%",
            )
        if o_line.any():
            plt.plot(x[o_line], obs_clima["obs_center"].to_numpy()[o_line], label="Obs center")

    # ---- Models (each series gets center + band with finite masks) ----
    for label, cl in model_climas.items():
        if cl is None or cl.empty:
            print(f"[plot] {label}: empty climatology")
            continue

        cl = cl.copy()
        cl.index = cl.index.astype(int)
        x = cl.index.to_numpy()

        m_fill = np.isfinite(cl["model_q10"].to_numpy()) & np.isfinite(cl["model_q90"].to_numpy())
        m_line = np.isfinite(cl["model_center"].to_numpy())

        print(f"[plot] {label}: model_center finite bins = {int(m_line.sum())} / {len(m_line)}")

        if m_fill.any():
            # keep band out of legend to reduce clutter, or comment out `label=None`
            plt.fill_between(
                x,
                cl["model_q10"].to_numpy(),
                cl["model_q90"].to_numpy(),
                where=m_fill,
                interpolate=True,
                alpha=0.2,
                label=f"{label} 10–90%",
            )
        if m_line.any():
            plt.plot(x[m_line], cl["model_center"].to_numpy()[m_line], label=f"{label} center")

    # ---- Month ticks (copied from _plot_overlay) ----
    month_start_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_ticks = [((d - 1) // 14 + 1) for d in month_start_doy]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    ax = plt.gca()
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)

    ax.set_xlim(1, max(NU_BIWEEK_MAX, max(month_ticks)))
    ax.set_xlabel("Month")

    plt.ylabel("N (g N m$^{-2}$)")
    plt.title(f"Biweekly nutrient climatology (obs vs inferred model free N) – Ecospace {scenario}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    _ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=200)

    if NU_SHOW_PLOT:
        plt.show()
    plt.close()

def plot_total_biomass_swings(
    series: str = "box",                  # "box" or "matched"
    spatial_reduction: str = "mean",       # "mean" | "median" | "sum"
    include_poc_diag: bool = False,        # optional: add DE1-POC from diagnostics
) -> str:
    """
    Plot total biomass (sum across groups) and its timestep-to-timestep swing.

    Notes:
      - Uses the same group list logic as 6b (cfg.NU_MODEL_GROUPS via _infer_group_cols).
      - If your CSV doesn't actually include *every* 3-day model step (depends on 6a),
        this will show swings at whatever timesteps exist in that table.
    """
    df = _load_series_table(series)  # 6b already knows "matched" vs "box" CSVs :contentReference[oaicite:3]{index=3}
    group_cols = _infer_group_cols(df)  # picks cfg.NU_MODEL_GROUPS if set :contentReference[oaicite:4]{index=4}

    # ---- pick a timestep key (prefer model_t_idx if present) ----
    if "model_t_idx" in df.columns:
        tkey = pd.to_numeric(df["model_t_idx"], errors="coerce")
        x_label = "Model step (t_idx)"
    elif "model_time" in df.columns:
        tkey = pd.to_datetime(df["model_time"], errors="coerce")
        x_label = "Model time"
    else:
        # fallback: not ideal, but keeps it working
        tkey = df["year"].astype(int) * 100 + df["biweekly"].astype(int)
        x_label = "YearBiweek (fallback)"

    # ---- sum biomass across groups (gC/m2) per row ----
    B = df[group_cols].apply(pd.to_numeric, errors="coerce").clip(lower=0)
    df = df.copy()
    df["tkey"] = tkey
    df["Bsum_gc_m2"] = B.sum(axis=1, min_count=1)

    # ---- optional: include POC from diagnostics the same way 6b aligns it ----
    # 6b aligns POC to model steps using model_t_idx (preferred) or model_time mapping :contentReference[oaicite:5]{index=5}
    if include_poc_diag:
        rel = _load_diag_relative_series_poc(NU_POC_RELATIVE_CSV, rel_col=NU_POC_REL_COL)
        B0_poc_gc = float(getattr(cfg, "ES_GROUPS_ECOPATH_B", {})["DE1-POC"])
        # rel(t) = B/B0 - 1  =>  B = B0 * (1 + rel)
        poc_gc = B0_poc_gc * (1.0 + df["tkey"].map(rel).astype(float).fillna(0.0))
        df["Bsum_gc_m2"] = df["Bsum_gc_m2"] + poc_gc

    df = df.dropna(subset=["tkey", "Bsum_gc_m2"])

    # ---- aggregate across space at each timestep ----
    g = df.groupby("tkey")["Bsum_gc_m2"]
    spatial_reduction = spatial_reduction.lower()
    if spatial_reduction == "mean":
        ts = g.mean()
        y_label = "Mean total biomass across cells (g C m$^{-2}$)"
    elif spatial_reduction == "median":
        ts = g.median()
        y_label = "Median total biomass across cells (g C m$^{-2}$)"
    elif spatial_reduction == "sum":
        ts = g.sum()
        y_label = "Sum over sampled cells (g C m$^{-2}$ × n_cells)"
    else:
        raise ValueError("spatial_reduction must be: 'mean', 'median', or 'sum'")

    ts = ts.sort_index()

    # ---- swings ----
    dB = ts.diff()
    pct = ts.pct_change() * 100.0

    # quick diagnostics you can print/log
    print(f"[biomass] series={series}, reduction={spatial_reduction}, n_steps={len(ts)}")
    print(f"[biomass] |ΔB| median={float(dB.abs().median()):.4g}, 90%={float(dB.abs().quantile(0.9)):.4g}, max={float(dB.abs().max()):.4g}")
    print(f"[biomass] |%Δ| median={float(pct.abs().median()):.4g}%, 90%={float(pct.abs().quantile(0.9)):.4g}%, max={float(pct.abs().max()):.4g}%")

    # ---- plot ----
    plt.figure(figsize=(11, 6))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(ts.index, ts.values)
    ax1.set_ylabel(y_label)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(pct.index, pct.values)
    ax2.axhline(0, linewidth=0.8)
    ax2.set_ylabel("Step swing (%Δ)")
    ax2.set_xlabel(x_label)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    _ensure_dir(OUTPUT_DIR_FIGS)
    out_png = os.path.join(OUTPUT_DIR_FIGS, f"biomass_total_swing_{SCENARIO}_{series}_{spatial_reduction}.png")
    plt.savefig(out_png, dpi=200)

    if NU_SHOW_PLOT:
        plt.show()
    plt.close()
    print(f"Saved biomass swing plot → {out_png}")
    return out_png


def _series_t0_bound_init_gN_m2(df: pd.DataFrame) -> float:
    """
    Compute bound_init (gN/m2) from the first model timestep present in the series table.

    Requirements:
      - df must already have 'model_N_bound_gN_m2' computed.
      - Prefer 'model_t_idx' if present; else use earliest 'model_time'.
      - If 'ewe_row'/'ewe_col' exist, take earliest timestep per cell, then reduce across cells.
        Otherwise, just reduce across all rows at the earliest timestep.

    Returns: scalar bound_init (gN/m2)
    """
    if "model_N_bound_gN_m2" not in df.columns:
        raise ValueError("df is missing model_N_bound_gN_m2; compute bound first.")

    d = df.dropna(subset=["model_N_bound_gN_m2"]).copy()
    if d.empty:
        raise ValueError("No non-NaN model_N_bound_gN_m2 values to compute t0 bound_init.")

    have_cells = ("ewe_row" in d.columns) and ("ewe_col" in d.columns)

    # ---- choose timestep key ----
    if "model_t_idx" in d.columns:
        d["__tkey"] = pd.to_numeric(d["model_t_idx"], errors="coerce")
    elif "model_time" in d.columns:
        d["__tkey"] = pd.to_datetime(d["model_time"], errors="coerce")
    else:
        raise ValueError("Need 'model_t_idx' or 'model_time' in df to compute t0.")

    d = d.dropna(subset=["__tkey"])
    if d.empty:
        raise ValueError("All timestep keys were NaN (model_t_idx/model_time).")

    # ---- select t0 rows ----
    if have_cells:
        # earliest timestep per cell
        idx = d.groupby(["ewe_row", "ewe_col"])["__tkey"].idxmin()
        d0 = d.loc[idx]
    else:
        # earliest timestep overall
        t0 = d["__tkey"].min()
        d0 = d.loc[d["__tkey"] == t0]

    # ---- reduce across space ----
    reducer = NU_T0_SPATIAL_REDUCER.lower()
    if reducer == "median":
        out = float(d0["model_N_bound_gN_m2"].median())
    elif reducer == "mean":
        out = float(d0["model_N_bound_gN_m2"].mean())
    else:
        raise ValueError("NU_T0_SPATIAL_REDUCER must be 'median' or 'mean'")

    if NU_DEBUG_INIT_ANCHOR:
        # show the chosen t0 and summary
        t0_show = d0["__tkey"].median() if have_cells else d0["__tkey"].iloc[0]
        print(f"[init-anchor] t0={t0_show}  n_rows={len(d0)}  bound_init_t0={out:.4g} gN/m2")
        print(f"[init-anchor] bound@t0 min/med/max: "
              f"{float(d0['model_N_bound_gN_m2'].min()):.4g} / "
              f"{float(d0['model_N_bound_gN_m2'].median()):.4g} / "
              f"{float(d0['model_N_bound_gN_m2'].max()):.4g}")

    return out



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_nutrient_overlay() -> None:
    """Pipeline entrypoint (keeps the same name used by ecospace_evaluation_master_pipeline.py)."""

    # --- Load the primary paired support once to define obs_col + group_cols consistently ---
    df_matched = _load_series_table(NU_PRIMARY_PAIRED_SERIES)
    obs_col = _pick_obs_gN_col(df_matched)

    group_cols = _infer_group_cols(df_matched)
    c2n = _build_c_to_n_map(group_cols)
    bound_init = _compute_bound_init_gN_m2(group_cols, c2n)

    # --- Optional: add POC detritus baseline bound-N (diagnostics series) ---
    poc_bound_init = 0.0
    if NU_INCLUDE_POC_DIAG:
        b0 = dict(getattr(cfg, "ES_GROUPS_ECOPATH_B", {}) or {})
        if "DE1-POC" not in b0:
            raise ValueError("Missing 'DE1-POC' in cfg.ES_GROUPS_ECOPATH_B (baseline gC/m2).")
        B0_poc_gc = float(b0["DE1-POC"])

        # This is where NU_C_TO_N_POM matters
        c2n_poc = float(NU_GROUP_C_TO_N.get("DE1-POC", C_TO_N_POM))

        poc_bound_init = B0_poc_gc * c2n_poc
        bound_init = float(bound_init + poc_bound_init)

    def compute_clima_for_df(df_in: pd.DataFrame) -> pd.DataFrame:
        df = _ensure_obs_col(df_in, obs_col).copy()

        # Compute time-varying bound N and inferred free N (same logic already use)
        df["model_N_bound_gN_m2"] = _compute_bound_N(df, group_cols, c2n)

        # --- Optional: add POC detritus bound-N time series from diagnostics (non-spatial) ---
        if NU_INCLUDE_POC_DIAG:
            if not NU_POC_RELATIVE_CSV or not os.path.exists(NU_POC_RELATIVE_CSV):
                raise FileNotFoundError(f"NU_POC_RELATIVE_CSV not found: {NU_POC_RELATIVE_CSV}")

            b0 = dict(getattr(cfg, "ES_GROUPS_ECOPATH_B", {}) or {})
            B0_poc_gc = float(b0["DE1-POC"])
            c2n_poc = float(NU_GROUP_C_TO_N.get("DE1-POC", C_TO_N_POM))

            rel = _load_diag_relative_series_poc(NU_POC_RELATIVE_CSV, rel_col=NU_POC_REL_COL)

            # Best alignment: use model_t_idx if present (from 6a); otherwise fall back to model_time mapping
            if "model_t_idx" in df.columns:
                step = pd.to_numeric(df["model_t_idx"], errors="coerce")
            else:
                mt = pd.to_datetime(df["model_time"], errors="coerce")
                uniq = pd.Series(mt.dropna().unique()).sort_values()
                step_map = pd.Series(range(len(uniq)), index=uniq.values)
                step = mt.map(step_map)

            # rel is normalized to 0 at baseline: B = B0*(1+rel)
            rel_on_rows = step.map(rel).astype(float).fillna(0.0)
            B_poc_gc = B0_poc_gc * (1.0 + rel_on_rows)
            df["model_POC_N_bound_gN_m2"] = B_poc_gc * c2n_poc

            if NU_DEBUG_POC:
                frac_nan = float(df["model_POC_N_bound_gN_m2"].isna().mean())
                print(
                    f"[POC] c2n={c2n_poc:.4g} B0={B0_poc_gc:.4g} gC/m2; "
                    f"rel index max={int(rel.index.max()) if len(rel.index) else -1}; "
                    f"rows NaN frac={frac_nan:.3f}"
                )
                print(
                    f"[POC] N_bound range: {float(df['model_POC_N_bound_gN_m2'].min()):.4g} "
                    f"to {float(df['model_POC_N_bound_gN_m2'].max()):.4g} gN/m2"
                )
                cols = [c for c in ["year", "biweekly", "model_t_idx", "model_time", "model_POC_N_bound_gN_m2"] if c in df.columns]
                print("[POC] head:", df[cols].head(3).to_dict(orient="records"))

            if NU_DEBUG_SAVE_DF:
                out_dbg = os.path.join(OUTPUT_DIR_EVAL, f"ecospace_{SCENARIO}_nutrients_debug_poc.csv")
                keep = [
                    c for c in [
                        "year", "biweekly", "ewe_row", "ewe_col", "model_t_idx", "model_time",
                        obs_col, "model_N_bound_gN_m2", "model_POC_N_bound_gN_m2", "model_N_free_gN_m2"
                    ]
                    if c in df.columns
                ]
                df.head(NU_DEBUG_SAVE_MAX_ROWS)[keep].to_csv(out_dbg, index=False)
                print("[POC] wrote debug CSV ->", out_dbg)

            # Add into total bound N (this is the key step you’re missing)
            df["model_N_bound_gN_m2"] = df["model_N_bound_gN_m2"] + df["model_POC_N_bound_gN_m2"]

        # --- bound_init selection (ecopath baseline vs first timestep in series table) ---
        bound_init_ecopath = float(bound_init)  # includes optional POC baseline already added above
        bound_init_used = bound_init_ecopath

        def _bound_t0_from_df(dfx: pd.DataFrame) -> float:
            """Compute bound_init from the first model timestep present in THIS df."""
            if "model_N_bound_gN_m2" not in dfx.columns:
                raise ValueError("Expected model_N_bound_gN_m2 to be computed before calling _bound_t0_from_df().")

            d = dfx.dropna(subset=["model_N_bound_gN_m2"]).copy()
            if d.empty:
                raise ValueError("No non-NaN model_N_bound_gN_m2 values to compute t0 bound_init.")

            have_cells = ("ewe_row" in d.columns) and ("ewe_col" in d.columns)

            # Choose timestep key
            if "model_t_idx" in d.columns:
                d["__tkey"] = pd.to_numeric(d["model_t_idx"], errors="coerce")
            else:
                d["__tkey"] = pd.to_datetime(d["model_time"], errors="coerce")

            d = d.dropna(subset=["__tkey"])
            if d.empty:
                raise ValueError("All timestep keys were NaN (model_t_idx/model_time).")

            # Earliest timestep per cell (if cell columns exist), otherwise earliest overall
            if have_cells:
                d = d.sort_values("__tkey")
                d0 = d.groupby(["ewe_row", "ewe_col"], as_index=False).first()
            else:
                t0 = d["__tkey"].min()
                d0 = d.loc[d["__tkey"] == t0]

            reducer = str(getattr(cfg, "NU_T0_SPATIAL_REDUCER", "median")).lower()
            if reducer == "median":
                return float(d0["model_N_bound_gN_m2"].median())
            elif reducer == "mean":
                return float(d0["model_N_bound_gN_m2"].mean())
            else:
                raise ValueError("NU_T0_SPATIAL_REDUCER must be 'median' or 'mean'")

        if NU_BOUND_INIT_MODE.lower() == "t0_series":
            # Compute bound_init from the first timestep in THIS df
            bound_init_used = _bound_t0_from_df(df)

            if NU_DEBUG_INIT_ANCHOR:
                print(
                    f"[init-anchor] bound_init ecopath={bound_init_ecopath:.4g} "
                    f"vs t0_series={bound_init_used:.4g} gN/m2"
                )

        elif NU_BOUND_INIT_MODE.lower() != "ecopath":
            raise ValueError("NU_BOUND_INIT_MODE must be 'ecopath' or 't0_series'")

        # --- free_init selection ---
        free_init = float(NU_FREE_INIT_GNM2)

        if NU_FREE_INIT_MODE.lower() == "t0_preserve_total":
            # Preserve the old definition of total inventory based on config + ecopath baseline,
            # but ensure the budget closes at t0 by adjusting free_init.
            total_ref = float(NU_FREE_INIT_GNM2 + bound_init_ecopath)

            # Compute mean/median bound at t0 (already computed if NU_BOUND_INIT_MODE=="t0_series",
            # otherwise compute it here without changing bound_init_used)
            bound_t0 = bound_init_used if NU_BOUND_INIT_MODE.lower() == "t0_series" else _bound_t0_from_df(df)
            free_init = float(max(0.0, total_ref - bound_t0))

            if NU_DEBUG_INIT_ANCHOR:
                print(
                    f"[init-anchor] total_ref={total_ref:.4g}  bound_t0={bound_t0:.4g}  "
                    f"free_init(adjusted)={free_init:.4g}"
                )

        elif NU_FREE_INIT_MODE.lower() != "config":
            raise ValueError("NU_FREE_INIT_MODE must be 'config' or 't0_preserve_total'")

        df["model_N_free_gN_m2"] = free_init + bound_init_used - df["model_N_bound_gN_m2"]
        df.loc[df["model_N_free_gN_m2"] < 0, "model_N_free_gN_m2"] = 0.0
        model_col_used = "model_N_free_gN_m2"

        # Optional flux scaling
        if NU_USE_FLUX_MULT and (NU_FLUX_MULT_COL in df.columns):
            mult = pd.to_numeric(df[NU_FLUX_MULT_COL], errors="coerce")
            if NU_FLUX_MULT_FILL is not None:
                mult = mult.fillna(float(NU_FLUX_MULT_FILL))
            df["model_N_flux_mult"] = mult

            if NU_FLUX_APPLY_MODE == "scale_free":
                df["model_N_free_used_gN_m2"] = df["model_N_free_gN_m2"] * df["model_N_flux_mult"]
            elif NU_FLUX_APPLY_MODE == "scale_total":
                total_init = float(getattr(cfg, "NU_TOTAL_INIT_GNM2", free_init + bound_init_used))
                df["model_N_free_used_gN_m2"] = (total_init * df["model_N_flux_mult"]) - df["model_N_bound_gN_m2"]
            else:
                raise ValueError(f"Unknown NU_FLUX_APPLY_MODE={NU_FLUX_APPLY_MODE!r}")

            df.loc[df["model_N_free_used_gN_m2"] < 0, "model_N_free_used_gN_m2"] = 0.0
            model_col_used = "model_N_free_used_gN_m2"

        year_biweek = _aggregate_year_biweek(df, obs_col=obs_col, model_col=model_col_used)
        return _compute_climatology(year_biweek)

    # --- Obs climatology (use matched table) ---
    obs_clima = compute_clima_for_df(df_matched)

    # --- Model climatologies to overlay ---
    model_climas: dict[str, pd.DataFrame] = {}
    for series in NU_PLOT_MODEL_SERIES:
        df_s = _load_series_table(series)
        cl = compute_clima_for_df(df_s)
        model_climas[series] = cl

    # --- Optional: add Ecosim climatology to the overlay ---
    print("1")
    print(NU_ECOSIM_NUTRIENTS_CSV)
    print("2")
    print(NU_OVERLAY_ECOSIM)
    if NU_OVERLAY_ECOSIM and NU_ECOSIM_NUTRIENTS_CSV:
        if os.path.exists(NU_ECOSIM_NUTRIENTS_CSV):
            try:
                cl_ecosim = _compute_ecosim_climatology(
                    NU_ECOSIM_NUTRIENTS_CSV, value_col=NU_ECOSIM_MODEL_COL
                )
                model_climas[NU_ECOSIM_LABEL] = cl_ecosim
                print(f"[ecosim overlay] added: {NU_ECOSIM_LABEL} from {NU_ECOSIM_NUTRIENTS_CSV}")
            except Exception as e:
                print(f"[ecosim overlay] failed: {e}")
        else:
            print(f"[ecosim overlay] missing CSV: {NU_ECOSIM_NUTRIENTS_CSV}")

    # --- Output ---
    _ensure_dir(OUTPUT_DIR_FIGS)
    suffix = "flux" if NU_USE_FLUX_MULT else "raw"
    out_png_multi = os.path.join(
        OUTPUT_DIR_FIGS,
        f"ecospace__{SCENARIO}_nutrient_climatology_overlay_freeN_multi_{suffix}.png",
    )

    _plot_overlay_multi(obs_clima, model_climas, out_png=out_png_multi, scenario=SCENARIO)
    print(f"Saved multi-series overlay plot → {out_png_multi}")

    skill_series_to_write: list[str] = []
    for s in [NU_PRIMARY_PAIRED_SERIES, "matched_cast", "matched"]:
        if s not in skill_series_to_write:
            try:
                _load_series_table(s)
                skill_series_to_write.append(s)
            except (FileNotFoundError, ValueError):
                pass

    for s in skill_series_to_write:
        if s == "box":
            continue
        _write_ecospace_skill_from_series(s)

    return out_png_multi

if __name__ == "__main__":
    run_nutrient_overlay()
    plot_total_biomass_swings(series="box", spatial_reduction="mean", include_poc_diag=False)

