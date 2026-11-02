"""
ecospace_eval_4e_sat_chl_timeseries.py

Regional (SGC2, central Strait of Georgia) 8-day time series of satellite chl-a vs
SOGEM-LTL phytoplankton biomass, 2003-2016.

- Satellite: MODIS 8-day DINEOF gap-filled chl-a (Costa lab; as in Suchy et al. 2022);
  weekly spatial median over SGC2 pixels.
- Ecospace: total phytoplankton (PP1-DIA + PP2-NAN + PP3-PIC); spatial median over SGC2
  cells at each 3-day step, averaged into the satellite's 8-day weeks.
- Ecosim: total phytoplankton (groups 17-19), averaged into the same weeks.
Satellite weeks with a regional median below SAT_MIN_CHL (mg m-3) are treated as invalid
composites (one case: 2004-11-08, median 0.001 mg m-3).
Each series is divided by its own median over the weeks with satellite data that year
(~mid-Feb to mid-Nov), as in the bloom-threshold definition (Figure 6).

Outputs
-------
- ecospace_<SC>_sat_chl_timeseries.png  (a) 2003-2016 series, (b) weekly climatology
- ecospace_<SC>_sat_chl_timeseries.csv
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib.pyplot as plt
from matplotlib.path import Path

import ecospace_eval_config as cfg

SC = cfg.ECOSPACE_SC
NC_MODEL = os.path.join(cfg.NC_PATH_OUT, cfg.NC_FILENAME)
SAT_NC = os.path.join("..", "..", "..", "SOGEM-LTL-data", "data", "reference", "satellite_chl",
                      "modis_chl_8day_interp_2003_2016.nc")
POLY = os.path.join("..", "..", "..", "SOGEM-LTL-data", "data", "reference", "analysis_domains_suchy.yml")
ECOSIM_CSV = os.path.join(cfg.EVALOUT_P, f"ecosim_{SC}_onerun_B_dates_seasons_nutrients.csv")
ECOSIM_COLS = ["17", "18", "19"]
OUT_PNG = os.path.join(cfg.FIGS_P, f"ecospace_{SC}_sat_chl_timeseries.png")
OUT_CSV = os.path.join(cfg.EVALOUT_P, f"ecospace_{SC}_sat_chl_timeseries.csv")

YEARS = np.arange(2003, 2017)
REGION = "SGC2"
VARS = cfg.BT_VARS_TO_ANALYZE_SAT
WK_START = np.arange(46) * 8 + 1
SAT_MIN_CHL = 0.05
COL = {"sat": "#E07B00", "ecospace": "#1f4e9c", "ecosim": "#6aa7e0"}


def weekly_mean(dates, vals):
    wk = np.minimum((pd.DatetimeIndex(dates).dayofyear.values - 1) // 8, 45)
    return np.array([np.nanmean(vals[wk == i]) if (wk == i).any() else np.nan for i in range(46)])


def build():
    poly = Path(np.asarray(yaml.safe_load(open(POLY))["polygon_coords"][REGION]))
    s = xr.open_dataset(SAT_NC)
    la, lo = np.meshgrid(s.lat.values, s.lon.values, indexing="ij")
    reg_s = poly.contains_points(np.c_[la.ravel(), lo.ravel()]).reshape(la.shape)
    ts = pd.DatetimeIndex(s.time.values)
    sat_all = np.nanmedian(s.chl.values[:, reg_s], axis=1)
    sat_all[sat_all < SAT_MIN_CHL] = np.nan

    ds = xr.open_dataset(NC_MODEL)
    reg_m = poly.contains_points(np.c_[ds.lat.values.ravel(), ds.lon.values.ravel()]).reshape(ds.lat.shape) \
        & np.isfinite(ds.depth.values) & (ds.depth.values > 0)
    es = pd.read_csv(ECOSIM_CSV, parse_dates=["date"])
    es["tot"] = es[ECOSIM_COLS].sum(axis=1)

    rows = []
    for yr in YEARS:
        sat = np.full(46, np.nan)
        k = ts.year == yr
        sat[(ts[k].dayofyear.values - 1) // 8] = sat_all[k]
        avail = np.isfinite(sat)
        y = ds.sel(time=str(yr))
        esp = weekly_mean(y.time.values, np.nanmedian(sum(y[g].values for g in VARS)[:, reg_m], axis=1))
        e = es[es.date.dt.year == yr]
        sim = weekly_mean(e.date.values, e.tot.values)
        for name, v in (("sat", sat), ("ecospace", esp), ("ecosim", sim)):
            med = np.nanmedian(v[avail])
            for w in np.flatnonzero(avail):
                rows.append(dict(year=yr, week=w + 1, doy=WK_START[w],
                                 date=pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=int(WK_START[w]) - 1),
                                 source=name, value=v[w], ratio=v[w] / med))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    return df


def main():
    df = build()
    wide = df.pivot_table(index=["year", "week", "date"], columns="source", values="ratio").reset_index()

    # agreement: raw weekly log ratios and anomalies from each source's weekly climatology
    stats_r = {}
    for m in ("ecospace", "ecosim"):
        f = wide[["sat", m]].dropna()
        r_raw = np.corrcoef(np.log(f["sat"]), np.log(f[m]))[0, 1]
        an = wide.copy()
        for c in ("sat", m):
            an[c] = np.log(an[c]) - an.groupby("week")[c].transform(lambda x: np.log(x).mean())
        fa = an[["sat", m]].dropna()
        r_an = np.corrcoef(fa["sat"], fa[m])[0, 1]
        stats_r[m] = (r_raw, r_an)
        cv = lambda x: np.std(np.log(x))
        print(f"{m:9s} weekly log-ratio r = {r_raw:.2f} (n={len(f)}); anomaly r = {r_an:.2f}; "
              f"SD of log ratio: sat {cv(f['sat']):.2f}, {m} {cv(f[m]):.2f}")

    LOGTICKS = ([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    fig = plt.figure(figsize=(9.5, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])

    # (a) 2003-2016 weekly series (satellite and Ecospace)
    ax = fig.add_subplot(gs[0, :])
    for yr in YEARS:
        g = wide[wide.year == yr]
        ax.plot(g["date"], g["sat"], color=COL["sat"], lw=1.0, label="Satellite chl a" if yr == YEARS[0] else None)
        ax.plot(g["date"], g["ecospace"], color=COL["ecospace"], lw=1.4, label="Ecospace" if yr == YEARS[0] else None)
    ax.axhline(1.05, color="0.35", lw=0.7, ls=":")
    ax.set_yscale("log"); ax.set_ylim(0.15, 8); ax.set_yticks(LOGTICKS[0]); ax.set_yticklabels(LOGTICKS[1])
    ax.minorticks_off()
    ax.set_ylabel("× annual median")
    ax.set_xlim(pd.Timestamp(2003, 1, 1), pd.Timestamp(2017, 1, 1))
    ax.legend(ncol=2, fontsize=8, loc="upper left", frameon=False)
    ax.text(0.995, 0.04, "(a)", transform=ax.transAxes, ha="right", fontweight="bold")
    ax.grid(alpha=0.25)

    # (b) weekly climatology
    ax2 = fig.add_subplot(gs[1, :2])
    gw = wide.groupby("week")
    x = WK_START[np.array(sorted(wide.week.unique())) - 1]
    for c, lab in (("sat", "Satellite chl a"), ("ecospace", "Ecospace"), ("ecosim", "Ecosim")):
        q = gw[c].quantile([0.25, 0.5, 0.75]).unstack()
        ax2.plot(x, q[0.5], color=COL[c], lw=1.8 if c != "ecosim" else 1.2, ls="-" if c != "ecosim" else "--", label=lab)
        if c != "ecosim":
            ax2.fill_between(x, q[0.25], q[0.75], color=COL[c], alpha=0.18, lw=0)
    ax2.axhline(1.05, color="0.35", lw=0.7, ls=":")
    ax2.set_yscale("log"); ax2.set_ylim(0.15, 8); ax2.set_yticks(LOGTICKS[0]); ax2.set_yticklabels(LOGTICKS[1])
    ax2.minorticks_off()
    mt = [pd.Timestamp(2001, m, 1).dayofyear for m in range(2, 13)]
    ax2.set_xticks(mt); ax2.set_xticklabels([pd.Timestamp(2001, m, 1).strftime("%b") for m in range(2, 13)])
    ax2.set_xlim(40, 335)
    ax2.set_ylabel("× annual median"); ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="upper right", frameon=False, ncol=3)
    ax2.text(0.01, 0.04, "(b)", transform=ax2.transAxes, fontweight="bold")

    # (c) weekly scatter
    ax3 = fig.add_subplot(gs[1, 2])
    lim = [0.15, 8]
    for c, lab in (("ecosim", "Ecosim"), ("ecospace", "Ecospace")):
        f = wide[["sat", c]].dropna()
        b = np.polyfit(np.log10(f["sat"]), np.log10(f[c]), 1)
        print(f"{c}: log-log OLS slope {b[0]:.2f}, intercept {b[1]:.2f}")
        ax3.scatter(f["sat"], f[c], s=7, alpha=0.5, color=COL[c], lw=0,
                    label=f"{lab}: r = {stats_r[c][0]:.2f} (anom. {stats_r[c][1]:.2f}), slope {b[0]:.2f}")
        xx = np.array(lim)
        ax3.plot(xx, 10 ** np.polyval(b, np.log10(xx)), color=COL[c], lw=1.4, ls="--" if c == "ecosim" else "-")
    ax3.plot(lim, lim, color="0.5", lw=0.7)
    ax3.axhline(1, color="0.8", lw=0.6); ax3.axvline(1, color="0.8", lw=0.6)
    ax3.set_xscale("log"); ax3.set_yscale("log"); ax3.set_xlim(lim); ax3.set_ylim(lim)
    for a in (ax3.xaxis, ax3.yaxis):
        a.set_major_locator(plt.FixedLocator(LOGTICKS[0])); a.set_major_formatter(plt.FixedFormatter(LOGTICKS[1]))
        a.set_minor_locator(plt.NullLocator())
    ax3.set_aspect("equal")
    ax3.set_xlabel("Satellite (× median)"); ax3.set_ylabel("Model (× median)")
    ax3.legend(fontsize=6.5, loc="upper left", frameon=False, handletextpad=0.2, markerscale=1.8)
    ax3.text(0.97, 0.04, "(c)", transform=ax3.transAxes, ha="right", fontweight="bold")
    ax3.grid(alpha=0.25)
    fig.savefig(OUT_PNG, dpi=300); plt.close(fig)
    print("saved", OUT_PNG, "and", OUT_CSV)


if __name__ == "__main__":
    main()
