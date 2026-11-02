"""
Zooplankton interannual-comparison diagnostics (revision, Oct 2026).

Produces
  1. Table: season-by-season obs-vs-model correlation of crustacean anomalies under
     progressively stricter treatment of survey sampling composition:
       (a) naive: pipeline method (season-year mean of tow-matched values, z-scored)
       (b) date-adjusted: within-season day-of-year curve removed at tow level
       (c) standardized: observed year index from OLS with year + day-of-year (cubic)
           + log10(tow depth range) + day/night + net type + station effects,
           compared with a model index that has no sampling artefacts
           (seasonal window mean at the sampled grid cells, tow-count weighted, from the NetCDF)
     plus r(mean sampling day, naive observed anomaly).
  2. Bloom timing vs spring crustacean biomass (manuscript Sect. 3.4; Supplement S.3.6 table and figure):
     date-adjusted tow-level biomass (obs, Ecospace, Ecosim; 2000-2018 survey years) and full-hindcast
     spring means (Ecospace at surveyed cells, Ecosim domain; 1980-2018), each against bloom day of year.
  3. Figure: naive spring anomalies vs mean sampling day of year.

Why: naive spring agreement (r = 0.83) was largely produced by year-to-year differences
in sampling dates.

Usage: python ecospace_eval_5d_zoop_sampling_diagnostics.py [--phenology]
  --phenology runs item 2 only (about 15 s).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats

import ecospace_eval_config as cfg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "SOGEM-LTL-data", "data"))
MATCHED = os.path.join(DATA, "evaluation", f"Zooplankton_matched_to_model_out_{cfg.ECOSPACE_SC}.csv")
RAW = os.path.join(DATA, "reference", "zoop", "Zoop_Perryetal_2021", "MODIFIED",
                   "Zoopl_SofG_1996-2018_EN_sizeclassesseparate.csv")
NC = os.path.join(DATA, "model_out_prepped", cfg.NC_FILENAME)
ECOSIM = os.path.join(DATA, "evaluation", f"ecosim_{cfg.ECOSPACE_SC}_onerun_B_dates_seasons.csv")
ECOSPACE_BLOOM = os.path.join(DATA, "evaluation", f"ecospace_bloom_timing_C09_{cfg.ECOSPACE_SC}.csv")
ECOSIM_BLOOM = os.path.join(DATA, "evaluation", f"ecosim_bloom_timing_C09_{cfg.ECOSPACE_SC}.csv")
FIGS = os.path.normpath(os.path.join(HERE, cfg.FIGS_P))
TABS = os.path.normpath(os.path.join(HERE, cfg.EVALOUT_P))

G = ["ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM"]
ECOSIM_COL = {"ZC1-EUP": "5", "ZC2-AMP": "6", "ZC3-DEC": "7", "ZC4-CLG": "8", "ZC5-CSM": "9"}
SEAS = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Fall": [9, 10, 11]}
Y0, Y1 = cfg.ZP_YEAR_START, cfg.ZP_YEAR_END
OFF = 1e-6
MIN_TOWS = cfg.ZP_ANOM_CLIM_MIN_TOWS_PER_YEAR
C_OBS, C_MOD, C_SIM = "darkorange", "blue", "royalblue"

# observed bloom dates (as in ecosim_eval_4_bloom_timing.load_observation_bloom_dfs)
SAT = pd.Series([100, 68, 50, 83, 115, 115, 100, 100, 92, 88, 92, 92, 55, 77], index=range(2003, 2017))
C09 = pd.Series([94, 78, 81, 82, 87, 76, 75, 87, 99, 76, 81, 78, 77, 55, 86, 86, 67, 87, 77, 104,
                 66, 70, 92, 86, 81, 62, 88, 100, 90, 97, 104, 103, 98, 88, 86, 77, 88, 104, 77],
                index=range(1980, 2019))


# ----------------------------------------------------------------------------- data
def load_tows() -> pd.DataFrame:
    m = pd.read_csv(MATCHED)
    rng = m["Tow_start_depth.m."].abs() - m["Tow_end_depth.m."].abs()
    keep = (rng / m["Bottom_depth.m."] >= cfg.ZP_TOW_MIN_PROP) | (m["Tow_start_depth.m."] >= cfg.ZP_TOW_MIN_START_DEPTH_M)
    m = m[keep & m.Year.between(Y0, Y1)].copy()
    m = m.merge(pd.read_csv(RAW, usecols=["Index", "Net_type"]).drop_duplicates("Index"), on="Index", how="left")
    m["Total"] = m[G].sum(axis=1)
    m["EWE-Total"] = m[[f"EWE-{g}" for g in G]].sum(axis=1)
    m["towrng"] = rng.loc[m.index] if False else (m["Tow_start_depth.m."].abs() - m["Tow_end_depth.m."].abs())
    doy = pd.to_datetime(m.Date).dt.dayofyear.to_numpy(float)
    m["doy"] = doy
    m["dos"] = np.where(m.Month == 12, doy - 365, doy)
    m["night"] = (m.Twilight == "Night").astype(float)
    m["net"] = m.Net_type.str.split().str[0]
    # Ecosim value at each tow (nearest 3-day step)
    es = pd.read_csv(ECOSIM, parse_dates=["date"])
    es["ecosim_Total"] = es[list(ECOSIM_COL.values())].sum(axis=1)
    m["date"] = pd.to_datetime(m.Date)
    m = pd.merge_asof(m.sort_values("date"), es[["date", "ecosim_Total"]].sort_values("date"),
                      on="date", direction="nearest", tolerance=pd.Timedelta(days=3))
    return m


def doy_resid(y, x):
    t = (x - x.mean()) / max(x.std(), 1.0)
    X = np.column_stack([t ** k for k in range(4)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def zyear(v, n):
    good = v[n >= MIN_TOWS]
    return (v - good.mean()) / good.std(ddof=1)


def r_ci(a, b):
    j = pd.concat([pd.Series(a), pd.Series(b)], axis=1).dropna()
    if len(j) < 5:
        return dict(r=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(j))
    r, p = stats.pearsonr(j.iloc[:, 0], j.iloc[:, 1])
    z, se = np.arctanh(r), 1 / np.sqrt(len(j) - 3)
    return dict(r=r, lo=np.tanh(z - 1.96 * se), hi=np.tanh(z + 1.96 * se), p=p, n=len(j))


def year_index(d, ycol, terms):
    years = sorted(d.Year.unique())
    X = [np.ones(len(d))] + [(d.Year == y).astype(float).values for y in years[1:]]
    if "doy" in terms:
        z = ((d.dos - d.dos.mean()) / max(d.dos.std(), 1)).values
        X += [z, z ** 2, z ** 3]
    if "towrng" in terms:
        X.append(np.log10(d.towrng.values))
    if "night" in terms:
        X.append(d.night.values)
    if "net" in terms:
        X += [(d.net == n).astype(float).values for n in sorted(d.net.dropna().unique())[1:]]
    if "station" in terms:
        X += [(d.Station == s).astype(float).values for s in sorted(d.Station.unique())[1:]]
    X = np.column_stack(X)
    b, *_ = np.linalg.lstsq(X, d[ycol].values, rcond=None)
    eff = np.r_[0.0, b[1:len(years)]]
    return pd.Series(eff - eff.mean(), index=years)


def model_window_indices(m: pd.DataFrame) -> dict:
    """log10 seasonal-window mean of model biomass at sampled cells (weighted by tow count)."""
    ds = xr.open_dataset(NC)
    lat, lon = ds.lat.values, ds.lon.values

    def cell(la, lo):
        d = (lat - la) ** 2 + ((lon - lo) * np.cos(np.deg2rad(la))) ** 2
        d = np.where(np.isfinite(d), d, np.inf)
        return np.unravel_index(np.argmin(d), d.shape)

    cells = m[["ecospace_closest_lat", "ecospace_closest_lon"]].drop_duplicates()
    cells["rc"] = [cell(a, b) for a, b in cells.values]
    mm = m.merge(cells, on=["ecospace_closest_lat", "ecospace_closest_lon"])
    uniq = sorted(set(mm.rc))
    w = np.array([(mm.rc == rc).sum() for rc in uniq], float)
    rows = np.array([r for r, _ in uniq]); cols = np.array([c for _, c in uniq])
    t = pd.to_datetime(ds.time.values)
    k = (t.year >= Y0) & (t.year <= Y1)
    tt = t[k]
    series = {}
    for g in G:
        series[g] = ds[g].isel(time=np.where(k)[0]).values[:, rows, cols]
    series["Total"] = sum(series[g] for g in G)
    out = {}
    for grp, arr in series.items():
        for s, months in SEAS.items():
            vals = {}
            for y in range(Y0, Y1 + 1):
                sel = np.isin(tt.month, months) & (tt.year == y)
                if sel.any():
                    vals[y] = np.log10(np.average(arr[sel].mean(axis=0), weights=w) + OFF)
            out[(grp, s)] = pd.Series(vals)
    return out


# ----------------------------------------------------------------------------- analyses
def diagnostics_table(m, mwin) -> pd.DataFrame:
    rows = []
    for grp in ["Total"] + G:
        for s in SEAS:
            d = m[(m.Season == s) & m[grp].notna() & m[f"EWE-{grp}"].notna()].copy()
            d["lo"] = np.log10(d[grp] + OFF)
            d["lm"] = np.log10(d[f"EWE-{grp}"] + OFF)
            d["lo_r"] = doy_resid(d.lo.values, d.dos.values)
            d["lm_r"] = doy_resid(d.lm.values, d.dos.values)
            a = d.groupby("Year").agg(n=("Index", "nunique"), obs=("lo", "mean"), mod=(f"EWE-{grp}", "mean"),
                                      obs_r=("lo_r", "mean"), mod_r=("lm_r", "mean"), mean_doy=("dos", "mean"))
            naive = r_ci(zyear(a.obs, a.n), zyear(a["mod"], a.n))
            date_r = r_ci(zyear(a.mean_doy, a.n), zyear(a.obs, a.n))
            adj = r_ci(zyear(a.obs_r, a.n), zyear(a.mod_r, a.n))
            oi = year_index(d, "lo", ["doy", "towrng", "night", "net", "station"])
            mw = mwin[(grp, s)]
            std = r_ci(oi, mw[mw.index.isin(oi.index)])
            rows.append(dict(group=grp, season=s, n_years=naive["n"],
                             r_naive=naive["r"], r_sampling_day_vs_obs=date_r["r"],
                             r_date_adjusted=adj["r"], ci_date_adjusted=f"[{adj['lo']:.2f}, {adj['hi']:.2f}]",
                             r_standardized_vs_model_window=std["r"],
                             ci_standardized=f"[{std['lo']:.2f}, {std['hi']:.2f}]"))
    return pd.DataFrame(rows)


def full_model_spring(m, y0=1980, y1=2018) -> pd.DataFrame:
    """log10 spring (Mar-May) mean crustacean biomass for every hindcast year, free of survey sampling:
    Ecospace at the surveyed grid cells (tow-count weighted) and Ecosim (domain, non-spatial)."""
    ds = xr.open_dataset(NC)
    lat, lon = ds.lat.values, ds.lon.values

    def cell(la, lo):
        d = (lat - la) ** 2 + ((lon - lo) * np.cos(np.deg2rad(la))) ** 2
        return np.unravel_index(np.argmin(np.where(np.isfinite(d), d, np.inf)), d.shape)

    cells = m[["ecospace_closest_lat", "ecospace_closest_lon"]].drop_duplicates()
    cells["rc"] = [cell(a, b) for a, b in cells.values]
    mm = m.merge(cells, on=["ecospace_closest_lat", "ecospace_closest_lon"])
    uniq = sorted(set(mm.rc))
    w = np.array([(mm.rc == rc).sum() for rc in uniq], float)
    rows = np.array([r for r, _ in uniq]); cols = np.array([c for _, c in uniq])
    t = pd.to_datetime(ds.time.values)
    k = np.where(np.isin(t.month, SEAS["Spring"]) & (t.year >= y0) & (t.year <= y1))[0]
    # load each variable once, then subset in memory (point-wise reads re-decompress chunks and are very slow)
    tot = sum(ds[g].values[k][:, rows, cols] for g in G)
    yrs = t[k].year
    espc = pd.Series({y: np.log10(np.average(tot[yrs == y].mean(axis=0), weights=w) + OFF) for y in range(y0, y1 + 1)})
    es = pd.read_csv(ECOSIM, parse_dates=["date"])
    es = es[es.date.dt.month.isin(SEAS["Spring"]) & es.date.dt.year.between(y0, y1)]
    esim = np.log10(es[list(ECOSIM_COL.values())].sum(axis=1).groupby(es.date.dt.year).mean() + OFF)
    return pd.DataFrame({"ecospace_full": espc, "ecosim_full": esim})


def spring_residual_years(m):
    sp = m[m.Season == "Spring"].copy()
    out = {}
    for lab, col in [("obs", "Total"), ("ecospace", "EWE-Total"), ("ecosim", "ecosim_Total")]:
        d = sp[sp[col].notna()]
        r = doy_resid(np.log10(d[col].values + OFF), d.dos.values)
        out[lab] = pd.Series(r, index=d.index).groupby(d.Year).mean()
    out["mean_doy"] = sp.groupby("Year").dos.mean()
    out["naive_obs"] = sp.groupby("Year").apply(lambda d: np.mean(np.log10(d.Total + OFF)))
    out["naive_mod"] = sp.groupby("Year")["EWE-Total"].mean()
    out["n"] = sp.groupby("Year").Index.nunique()
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- figures
def _panel(ax, x, y, color, marker, label, letter, xlabel, ylabel, stat_loc="upper"):
    j = pd.concat([x, y], axis=1).dropna()
    ax.scatter(j.iloc[:, 0], j.iloc[:, 1], s=36, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=3)
    if len(j) >= 5:
        b = np.polyfit(j.iloc[:, 0], j.iloc[:, 1], 1)
        xx = np.linspace(j.iloc[:, 0].min(), j.iloc[:, 0].max(), 10)
        ax.plot(xx, np.polyval(b, xx), color="0.35", lw=1, ls="--")
        rc = r_ci(j.iloc[:, 0], j.iloc[:, 1])
        ptxt = "p < 0.01" if rc["p"] < 0.01 else f"p = {rc['p']:.2f}"
        y0, va = (0.95, "top") if stat_loc == "upper" else (0.04, "bottom")
        ax.text(0.97, y0, f"r = {rc['r']:.2f}\n{ptxt}\nn = {rc['n']}", transform=ax.transAxes,
                ha="right", va=va, fontsize=8.5)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.text(0.02, 0.97, letter, transform=ax.transAxes, fontweight="bold", va="top")
    ax.grid(alpha=0.3)


def fig_bloom_zoop(sr):
    eb = pd.read_csv(ECOSPACE_BLOOM).set_index("Year")["Day of Year"]
    sb = pd.read_csv(ECOSIM_BLOOM).set_index("Year")["Day of Year"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    _panel(ax[0], SAT.reindex(sr.index), sr.obs, C_OBS, "o", "Observed zooplankton\nvs satellite bloom day", "(a)",
           "Spring bloom day of year", "Spring crustacean biomass\n(log10 residual from seasonal curve)")
    _panel(ax[1], eb.reindex(sr.index), sr.ecospace, C_MOD, "s", "SOGEM-LTL Ecospace\nvs own bloom day", "(b)",
           "Spring bloom day of year", "")
    _panel(ax[2], sb.reindex(sr.index), sr.ecosim, C_SIM, "^", "SOGEM-LTL Ecosim\nvs own bloom day", "(c)",
           "Spring bloom day of year", "")
    fig.tight_layout()
    p = os.path.join(FIGS, f"ecospace_{cfg.ECOSPACE_SC}_zoop_spring_vs_bloomday.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); print("saved", p)


def fig_sampling_day(sr):
    z = lambda v: zyear(v, sr.n)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    _panel(ax[0], sr.mean_doy, z(sr.naive_obs), C_OBS, "o", "Observed", "(a)",
           "Mean spring sampling day of year", "Spring anomaly (z-score),\nseason-year mean of matched tows", stat_loc="lower")
    _panel(ax[1], sr.mean_doy, z(sr.naive_mod), C_MOD, "s", "SOGEM-LTL Ecospace (tow-matched)", "(b)",
           "Mean spring sampling day of year", "", stat_loc="lower")
    fig.tight_layout()
    p = os.path.join(FIGS, f"ecospace_{cfg.ECOSPACE_SC}_zoop_spring_anom_vs_samplingday.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); print("saved", p)


def bloom_zoop_links(m):
    """Bloom timing vs spring crustacean biomass (manuscript Sect. 3.4; Supplement S.3.6 table and figure)."""
    sr = spring_residual_years(m)
    eb = pd.read_csv(ECOSPACE_BLOOM).set_index("Year")["Day of Year"]
    sb = pd.read_csv(ECOSIM_BLOOM).set_index("Year")["Day of Year"]
    fm = full_model_spring(m)
    fm.join(eb.rename("ecospace_bloom")).join(sb.rename("ecosim_bloom")).to_csv(
        os.path.join(TABS, f"ecospace_zoop_full_model_spring_vs_bloom_{cfg.ECOSPACE_SC}.csv"), index_label="Year")
    ph = pd.DataFrame([
        dict(comparison="Ecospace full-model spring zoop vs Ecospace bloom (1980-2018)",
             **r_ci(eb.reindex(fm.index), fm.ecospace_full)),
        dict(comparison="Ecosim full-model spring zoop vs Ecosim bloom (1980-2018)",
             **r_ci(sb.reindex(fm.index), fm.ecosim_full)),
        dict(comparison="obs zoop vs satellite bloom", **r_ci(SAT.reindex(sr.index), sr.obs)),
        dict(comparison="obs zoop vs C09 bloom", **r_ci(C09.reindex(sr.index), sr.obs)),
        dict(comparison="Ecospace zoop vs Ecospace bloom", **r_ci(eb.reindex(sr.index), sr.ecospace)),
        dict(comparison="Ecosim zoop vs Ecosim bloom", **r_ci(sb.reindex(sr.index), sr.ecosim)),
        dict(comparison="Ecospace bloom vs satellite", **r_ci(eb.reindex(sr.index), SAT.reindex(sr.index))),
        dict(comparison="sampling day vs satellite bloom", **r_ci(sr.mean_doy, SAT.reindex(sr.index))),
        dict(comparison="sampling day vs C09 bloom", **r_ci(sr.mean_doy, C09.reindex(sr.index))),
    ])
    ph.to_csv(os.path.join(TABS, f"ecospace_zoop_bloom_phenology_links_{cfg.ECOSPACE_SC}.csv"), index=False)
    print(ph.round(2).to_string(index=False))
    fig_bloom_zoop(sr)
    return sr


def main(phenology_only=False):
    m = load_tows()
    if not phenology_only:
        mwin = model_window_indices(m)
        tab = diagnostics_table(m, mwin)
        tab.to_csv(os.path.join(TABS, f"ecospace_zoop_interannual_diagnostics_{cfg.ECOSPACE_SC}.csv"), index=False)
        print(tab.round(2).to_string(index=False))
    sr = bloom_zoop_links(m)
    if not phenology_only:
        fig_sampling_day(sr)


if __name__ == "__main__":
    import sys
    main(phenology_only="--phenology" in sys.argv)
