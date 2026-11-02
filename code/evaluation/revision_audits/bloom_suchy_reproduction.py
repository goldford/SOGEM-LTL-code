"""
bloom_suchy_reproduction.py

1) Reproduce Suchy et al. (2022) spring bloom initiation from the DINEOF-interpolated
   MODIS 8-day chl-a (2003-2016) over the SGC2 polygon, testing recipe variants:
     scale       : linear chl vs log10 chl
     aggregation : spatial mean vs median per 8-day week
     persistence : 'suchy' (>=1 of next 2 weeks > 0.7*thr)
                   'code'  (mean of next 2, or of the 2 after, > 0.7*thr; helpers.find_bloom_doy)
   Threshold = 1.05 x median over the weeks with data (~mid-Feb to mid-Nov).
2) Apply the best recipe to Ecospace (SGC2 and SGC4) and Ecosim on the same 8-day
   weeks and window; also report the current Figure 6 setup for reference.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from matplotlib.path import Path
from scipy import stats

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "SOGEM-LTL-data", "data"))
SAT = os.path.join(ROOT, "reference", "satellite_chl", "modis_chl_8day_interp_2003_2016.nc")
POLY = os.path.join(ROOT, "reference", "analysis_domains_suchy.yml")
ESP = os.path.join(ROOT, "model_out_prepped", "ecospace_SC300_1978-2018.nc")
ESIM = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "eval", "tables",
                                     "ecosim_SC300_onerun_B_dates_seasons_nutrients.csv"))
OUT = os.path.join(os.path.dirname(__file__), "bloom_suchy_reproduction_results.txt")

YEARS = np.arange(2003, 2017)
SUCHY_DOY = np.array([100, 68, 50, 83, 115, 115, 100, 100, 92, 88, 92, 92, 55, 77])
WK_START = np.arange(46) * 8 + 1          # DOY of the first day of each 8-day week
PP = ["PP1-DIA", "PP2-NAN", "PP3-PIC"]
ESIM_COLS = ["17", "18", "19"]

lines = []


def say(s=""):
    print(s); lines.append(s)


def polys():
    return {k: Path(np.asarray(v)) for k, v in yaml.safe_load(open(POLY))["polygon_coords"].items()}


def bloom_week(v, rule):
    """v: weekly values (NaN where no data). Returns index of bloom week or None."""
    ok = np.isfinite(v)
    if ok.sum() < 5:
        return None
    thr = 1.05 * np.nanmedian(v)
    sub = 0.7 * thr
    idx = np.flatnonzero(ok)
    w = v[idx]
    for j in range(len(w) - 2):
        if w[j] < thr:
            continue
        if rule == "suchy":
            if w[j + 1] > sub or w[j + 2] > sub:
                return idx[j]
        else:
            a = (w[j + 1] + w[j + 2]) / 2 > sub
            b = j + 4 < len(w) and (w[j + 3] + w[j + 4]) / 2 > sub
            if a or b:
                return idx[j]
    return None


def series_doy(weekly, rule):
    """weekly: dict year -> (46,) array. Returns bloom start DOY per year."""
    out = []
    for yr in YEARS:
        k = bloom_week(weekly[yr], rule)
        out.append(np.nan if k is None else WK_START[k])
    return np.array(out, dtype=float)


def score(doy, ref=SUCHY_DOY):
    f = np.isfinite(doy)
    wk = (doy[f] - 1) // 8
    wref = (ref[f] - 1) // 8
    r = stats.pearsonr(doy[f], ref[f])[0] if f.sum() > 3 else np.nan
    return dict(n=int(f.sum()), r=r, bias=float(np.mean(doy[f] - ref[f])), mae=float(np.mean(np.abs(doy[f] - ref[f]))),
                exact_wk=int((wk == wref).sum()), within1=int((np.abs(wk - wref) <= 1).sum()))


def fmt(name, s):
    return (f"{name:48s} n={s['n']:2d}  r={s['r']:5.2f}  bias={s['bias']:6.1f} d  MAE={s['mae']:5.1f} d  "
            f"same week {s['exact_wk']:2d}/14  +-1 week {s['within1']:2d}/14")


# ------------------------------------------------------------------ satellite
P = polys()
s = xr.open_dataset(SAT)
la, lo = np.meshgrid(s.lat.values, s.lon.values, indexing="ij")
m_sat = P["SGC2"].contains_points(np.c_[la.ravel(), lo.ravel()]).reshape(la.shape)
chl = s.chl.values[:, m_sat]                                  # (time, npix)
t = pd.DatetimeIndex(s.time.values)
say(f"SGC2 satellite pixels: {m_sat.sum()}  (finite in a full week: {int(np.isfinite(chl[t.dayofyear == 97]).sum(1).max())})")

sat_weekly = {}
for scale in ("lin", "log"):
    for agg in ("mean", "median"):
        d = {}
        for yr in YEARS:
            c = chl[t.year == yr]
            if scale == "log":
                c = np.log10(c)
            with np.errstate(all="ignore"):
                d[yr] = np.nanmean(c, 1) if agg == "mean" else np.nanmedian(c, 1)
        sat_weekly[(scale, agg)] = d

say("\n=== 1. Reproducing Suchy et al. (2022) from the satellite files (SGC2) ===")
best = None
for (scale, agg), d in sat_weekly.items():
    for rule in ("suchy", "code"):
        doy = series_doy(d, rule)
        sc = score(doy)
        say(fmt(f"sat {scale:3s} {agg:6s} persistence={rule}", sc) + f"   DOY: {doy.astype(int).tolist()}")
        key = (sc["exact_wk"], sc["within1"], -sc["mae"])
        if best is None or key > best[0]:
            best = (key, scale, agg, rule, doy)
_, B_SCALE, B_AGG, B_RULE, SAT_BEST = best
say(f"Suchy published DOY:{' ' * 63}{SUCHY_DOY.tolist()}")
say(f"best recipe: scale={B_SCALE}, aggregation={B_AGG}, persistence={B_RULE}")

# weeks with satellite data in each year (model uses the same weeks)
avail = {yr: np.isfinite(sat_weekly[("lin", "mean")][yr]) for yr in YEARS}

# ------------------------------------------------------------------ models
ds = xr.open_dataset(ESP)
lat, lon, dep = ds.lat.values, ds.lon.values, ds.depth.values
m_esp = {k: P[k].contains_points(np.c_[lat.ravel(), lon.ravel()]).reshape(lat.shape) & (dep > 0)
         for k in ("SGC2", "SGC4")}

es = pd.read_csv(ESIM, parse_dates=["date"])
es["tot"] = es[ESIM_COLS].sum(axis=1)


def to_weekly(dates, vals, yr, scale):
    doy = pd.DatetimeIndex(dates).dayofyear.values
    wk = np.minimum((doy - 1) // 8, 45)
    out = np.array([np.nanmean(vals[wk == i]) if (wk == i).any() else np.nan for i in range(46)])
    out[~avail[yr]] = np.nan
    return np.log10(out) if scale == "log" else out


def model_weekly(kind, scale, region=None):
    d = {}
    for yr in YEARS:
        if kind == "ecospace":
            y = ds.sel(time=str(yr))
            tot = sum(y[g].values for g in PP)
            v = np.nanmean(tot[:, m_esp[region]], 1) if B_AGG == "mean" else np.nanmedian(tot[:, m_esp[region]], 1)
            d[yr] = to_weekly(y.time.values, v, yr, scale)
        else:
            y = es[es.date.dt.year == yr]
            d[yr] = to_weekly(y.date.values, y.tot.values, yr, scale)
    return d


def model_3day_suchywindow(kind, region=None):
    """3-day output, Suchy window (DOY of satellite weeks 7-41), suchy-style persistence on steps."""
    out = []
    for yr in YEARS:
        if kind == "ecospace":
            y = ds.sel(time=str(yr)); dates = y.time.values
            v = np.nanmedian(sum(y[g].values for g in PP)[:, m_esp[region]], 1)
        else:
            y = es[es.date.dt.year == yr]; dates = y.date.values; v = y.tot.values
        doy = pd.DatetimeIndex(dates).dayofyear.values
        lo_, hi_ = WK_START[avail[yr]].min(), WK_START[avail[yr]].max() + 7
        w = (doy >= lo_) & (doy <= hi_)
        k = bloom_week(v[w], B_RULE)
        out.append(np.nan if k is None else doy[w][k])
    return np.array(out, dtype=float)


say("\n=== 2. Models with the best satellite recipe (8-day weeks, same window) ===")
rows = {}
for name, d in [("Ecospace SGC2", model_weekly("ecospace", B_SCALE, "SGC2")),
                ("Ecospace SGC4 (2 cells)", model_weekly("ecospace", B_SCALE, "SGC4")),
                ("Ecosim", model_weekly("ecosim", B_SCALE))]:
    doy = series_doy(d, B_RULE); rows[name] = doy
    say(fmt(f"{name} vs Suchy", score(doy)) + f"   DOY: {np.nan_to_num(doy, nan=-1).astype(int).tolist()}")
    say(fmt(f"{name} vs reproduced satellite", score(doy, SAT_BEST)))

say("\n=== 3. Models, 3-day output, Suchy window, Suchy-style median (no 8-day binning) ===")
for name, doy in [("Ecospace SGC2", model_3day_suchywindow("ecospace", "SGC2")),
                  ("Ecosim", model_3day_suchywindow("ecosim"))]:
    say(fmt(f"{name} vs Suchy", score(doy)) + f"   DOY: {np.nan_to_num(doy, nan=-1).astype(int).tolist()}")

say("\n=== 4. Current Figure 6 setup, for reference ===")
e_pub = pd.read_csv(os.path.join(ROOT, "evaluation", "ecospace_bloom_timing_SSoG_SC300.csv")).set_index("Year")["Day of Year"].reindex(YEARS).values
say(fmt("Ecospace SGC4, 3-day, full-year median", score(e_pub.astype(float))))
f_sim = os.path.join(os.path.dirname(ESIM), "ecosim_bloom_timing_satellite_SC300.csv")
if os.path.exists(f_sim):
    q = pd.read_csv(f_sim)
    col = [c for c in q.columns if "Day of Year" in c][0]
    ycol = [c for c in q.columns if c.lower().startswith("year")][0]
    say(fmt("Ecosim, 3-day, full-year median", score(q.set_index(ycol)[col].reindex(YEARS).values.astype(float))))

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("saved", OUT)
