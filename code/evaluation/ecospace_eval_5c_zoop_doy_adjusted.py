"""
Ecospace zooplankton anomalies (obs vs model), adjusted for sampling day of year.

Why: the tow-matched seasonal anomalies in ecospace_eval_5_zoop_eval.py partly
reflect *when* each year was sampled. Within a season biomass changes quickly,
so a year sampled late in spring looks "high" in both obs and model. That
inflates obs-model agreement (spring Total r = 0.83 -> ~0 once removed).

Method (per season, per group, tow level):
  1. log10(biomass + 1e-6) for obs and for the tow-matched model value.
  2. Fit a cubic in day-of-season, pooled over all years, separately for obs
     and model; keep each tow's residual.
     This removes the mean seasonal curve but keeps year-specific departures
     (e.g. an early spring still shows up as a positive residual).
  3. Average residuals within each year.
  4. z-score across years (year_equal climatology, years with >= 3 tows
     define the mean/SD), as in the main pipeline.

Same tow filter, years, seasons and figure layouts as ecospace_eval_5_zoop_eval.py.
Uncorrected r (pipeline method: obs mean_of_logs, model mean_raw) is shown for
reference on each panel.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ecospace_eval_config as cfg

# ----------------------------------------------------------------------------- settings
SC = cfg.ECOSPACE_SC
MATCHED_CSV = os.path.join(r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data\evaluation",
                           f"Zooplankton_matched_to_model_out_{SC}.csv")
FIGS_P = os.path.normpath(os.path.join(os.path.dirname(__file__), cfg.FIGS_P))
TABLES_P = os.path.normpath(os.path.join(os.path.dirname(__file__), cfg.EVALOUT_P))
Y0, Y1 = cfg.ZP_YEAR_START, cfg.ZP_YEAR_END
GROUPS = ["ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM"]
SEASONS = ["Winter", "Spring", "Summer", "Fall"]
MIN_TOWS_CLIM = cfg.ZP_ANOM_CLIM_MIN_TOWS_PER_YEAR
LOG_OFFSET = 1e-6
POLY_DEG = 3
LABELS = cfg.ZP_FRIENDLY_MAP_ZC
C_OBS, C_MOD = "darkorange", "blue"   # as in the publication panel
PUB_SEASON = cfg.ZP_SEASON_CHOICE


# ----------------------------------------------------------------------------- helpers
def wss(o, m):
    return 1 - np.sum((o - m) ** 2) / np.sum((np.abs(o - o.mean()) + np.abs(m - o.mean())) ** 2)


def skill(o, m):
    o, m = np.asarray(o, float), np.asarray(m, float)
    k = np.isfinite(o) & np.isfinite(m)
    o, m = o[k], m[k]
    return dict(N=len(o), r=np.corrcoef(o, m)[0, 1], WSS=wss(o, m),
                RMSE=float(np.sqrt(np.mean((o - m) ** 2))), MB=float(np.mean(m - o)))


def zscore_year_equal(v: pd.Series, n: pd.Series) -> pd.Series:
    good = v[n >= MIN_TOWS_CLIM]
    return (v - good.mean()) / good.std(ddof=1)


def day_of_season(d: pd.DataFrame) -> np.ndarray:
    doy = pd.to_datetime(d["Date"]).dt.dayofyear.to_numpy(float)
    return np.where(d["Month"].to_numpy() == 12, doy - 365, doy)  # keep winter continuous


def doy_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    t = (x - x.mean()) / max(x.std(), 1.0)
    X = np.column_stack([t ** k for k in range(POLY_DEG + 1)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def load_tows() -> pd.DataFrame:
    df = pd.read_csv(MATCHED_CSV)
    rng = df["Tow_start_depth.m."].abs() - df["Tow_end_depth.m."].abs()
    keep = (rng / df["Bottom_depth.m."] >= cfg.ZP_TOW_MIN_PROP) | \
           (df["Tow_start_depth.m."] >= cfg.ZP_TOW_MIN_START_DEPTH_M)
    df = df[keep & df["Year"].between(Y0, Y1)].copy()
    df["Total"] = df[GROUPS].sum(axis=1)
    df["EWE-Total"] = df[[f"EWE-{g}" for g in GROUPS]].sum(axis=1)
    df["dos"] = day_of_season(df)
    return df


def build_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for season in SEASONS:
        s = df[df["Season"] == season]
        for g in GROUPS + ["Total"]:
            d = s[s[g].notna() & s[f"EWE-{g}"].notna()].copy()
            lo = np.log10(d[g].to_numpy() + LOG_OFFSET)
            lm = np.log10(d[f"EWE-{g}"].to_numpy() + LOG_OFFSET)
            d["obs_res"] = doy_residual(lo, d["dos"].to_numpy())
            d["mod_res"] = doy_residual(lm, d["dos"].to_numpy())
            d["obs_log"] = lo
            a = d.groupby("Year").agg(n_tows=("Index", "nunique"),
                                      obs_res=("obs_res", "mean"), mod_res=("mod_res", "mean"),
                                      obs_unadj=("obs_log", "mean"),               # pipeline: mean_of_logs
                                      mod_unadj=(f"EWE-{g}", "mean"),              # pipeline: mean_raw
                                      mean_doy=("dos", "mean"))
            a["obs_anom"] = zscore_year_equal(a.obs_res, a.n_tows)
            a["model_anom"] = zscore_year_equal(a.mod_res, a.n_tows)
            a["obs_anom_unadj"] = zscore_year_equal(a.obs_unadj, a.n_tows)
            a["model_anom_unadj"] = zscore_year_equal(a.mod_unadj, a.n_tows)
            a["season"], a["group"] = season, g
            out.append(a.reset_index().rename(columns={"Year": "year"}))
    return pd.concat(out, ignore_index=True)


def skill_table(an: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, g), d in an.groupby(["season", "group"], sort=False):
        a, u = skill(d.obs_anom, d.model_anom), skill(d.obs_anom_unadj, d.model_anom_unadj)
        rows.append(dict(season=s, group=g, N=a["N"], r=a["r"], WSS=a["WSS"], RMSE=a["RMSE"], MB=a["MB"],
                         r_unadjusted=u["r"], WSS_unadjusted=u["WSS"]))
    return pd.DataFrame(rows)


def _bars(ax, d, legend=False):
    x = np.arange(len(d)); w = 0.35
    ax.bar(x - w / 2, d.obs_anom, w, color=C_OBS, label="Observations")
    ax.bar(x + w / 2, d.model_anom, w, color=C_MOD, label="SOGEM-LTL")
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(d.year.astype(int), rotation=45, ha="right")
    ax.grid(alpha=0.4)
    if legend:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.3 * (hi - lo))   # headroom so the legend doesn't cover bars
        ax.legend(loc="upper right", ncol=2, fontsize=9)


def _stat_text(ax, d, keys=("r",), loc=(0.98, 0.04), ha="right", va="bottom"):
    a, u = skill(d.obs_anom, d.model_anom), skill(d.obs_anom_unadj, d.model_anom_unadj)
    lines = []
    if "r" in keys:
        lines.append(f"r = {a['r']:.2f} (unadj. {u['r']:.2f})")
    if "WSS" in keys:
        lines.append(f"WSS = {a['WSS']:.2f} (unadj. {u['WSS']:.2f})")
    if "N" in keys:
        lines.append(f"N = {a['N']}")
    ax.text(*loc, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=9)


def _save(fig, name):
    p = os.path.join(FIGS_P, name)
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


# ----------------------------------------------------------------------------- figures
def fig_pub_panel(an, season=PUB_SEASON, group="Total"):
    d = an[(an.season == season) & (an.group == group)].sort_values("year")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 9), gridspec_kw=dict(height_ratios=[1.25, 1]))
    a1.scatter(d.obs_anom, d.model_anom, color="tab:blue", alpha=0.85)
    for _, r in d.iterrows():
        a1.annotate(str(int(r.year)), (r.obs_anom, r.model_anom), fontsize=7, color="0.25")
    lim = np.nanmax(np.abs(np.r_[d.obs_anom, d.model_anom])) * 1.15
    a1.plot([-lim, lim], [-lim, lim], color="grey", lw=1)
    k = np.isfinite(d.obs_anom) & np.isfinite(d.model_anom)
    b = np.polyfit(d.obs_anom[k], d.model_anom[k], 1)
    a1.plot([-lim, lim], np.polyval(b, [-lim, lim]), "k--", lw=1)
    a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim); a1.set_aspect("equal")
    a1.set_xlabel("Observed anomaly (z-score)"); a1.set_ylabel("Model anomaly (z-score)")
    a1.grid(alpha=0.4); a1.text(0.01, 0.97, "(a)", transform=a1.transAxes, fontweight="bold", va="top")
    _stat_text(a1, d, keys=("r", "N"))
    _bars(a2, d, legend=True)
    a2.set_ylabel("Anomaly (z-score)"); a2.set_xlabel("Year")
    a2.text(0.01, 0.97, "(b)", transform=a2.transAxes, fontweight="bold", va="top")
    _stat_text(a2, d, keys=("WSS",), loc=(0.08, 0.97), ha="left", va="top")
    fig.suptitle(f"Ecospace {SC} | {Y0}–{Y1} | {season} | {LABELS.get(group, group)} | "
                 f"day-of-year adjusted", fontsize=11)
    fig.tight_layout()
    _save(fig, f"ecospace_{SC}_zoop_pub_total_single_season_{season}_zoop_ZC_DOYadj.png")


def fig_total_4panel(an, group="Total"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for ax, s in zip(axes.flat, SEASONS):
        d = an[(an.season == s) & (an.group == group)].sort_values("year")
        _bars(ax, d, legend=(s == "Winter")); ax.set_title(s)
        _stat_text(ax, d, keys=("r", "WSS"), loc=(0.98, 0.03))
    for ax in axes[:, 0]:
        ax.set_ylabel("Anomaly (z-score)")
    fig.suptitle(f"Ecospace {SC} | Annual anomalies (Obs vs Model) | {Y0}–{Y1} | "
                 f"{LABELS.get(group, group)} | day-of-year adjusted")
    fig.tight_layout()
    _save(fig, f"ecospace_{SC}_zoop_anomaly_bars_total4panel_ZC_DOYadj.png")


def fig_group_panel(an, season):
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharey=True)
    for ax, g in zip(axes.flat, GROUPS + ["Total"]):
        d = an[(an.season == season) & (an.group == g)].sort_values("year")
        _bars(ax, d, legend=(g == "Total")); ax.set_title(LABELS.get(g, g))
        _stat_text(ax, d, keys=("r",), loc=(0.98, 0.03))
    for ax in axes[:, 0]:
        ax.set_ylabel("Anomaly (z-score)")
    fig.suptitle(f"Ecospace {SC} | Obs vs Model annual anomalies | {Y0}–{Y1} | {season} | "
                 f"day-of-year adjusted")
    fig.tight_layout()
    _save(fig, f"ecospace_{SC}_zoop_anomalies_panel_{season}_zoop_ZC_DOYadj.png")


def main():
    an = build_anomalies(load_tows())
    an.to_csv(os.path.join(TABLES_P, f"ecospace_zoop_anom_DOYadj_{SC}.csv"), index=False)
    st = skill_table(an)
    st.to_csv(os.path.join(TABLES_P, f"ecospace_zoop_anom_DOYadj_skill_{SC}.csv"), index=False)
    print(st.round(2).to_string(index=False))
    fig_pub_panel(an)
    fig_total_4panel(an)
    for s in SEASONS:
        fig_group_panel(an, s)


if __name__ == "__main__":
    main()
