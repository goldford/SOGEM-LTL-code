"""Standardized observed zooplankton indices vs model indices, with controls for sampling composition."""
import numpy as np, pandas as pd, xarray as xr
from scipy import stats

pd.set_option("display.width", 250)
DE = r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data\evaluation"
RAW = r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data\reference\zoop\Zoop_Perryetal_2021\MODIFIED\Zoopl_SofG_1996-2018_EN_sizeclassesseparate.csv"
NC = r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data\model_out_prepped\ecospace_SC300_1978-2018.nc"
G = ["ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM"]
SEAS = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Fall": [9, 10, 11]}
OFF = 1e-6

# ---------------------------------------------------------------- tows
m = pd.read_csv(f"{DE}/Zooplankton_matched_to_model_out_SC300.csv")
rng = m["Tow_start_depth.m."].abs() - m["Tow_end_depth.m."].abs()
m = m[((rng / m["Bottom_depth.m."] >= 0.7) | (m["Tow_start_depth.m."] >= 150)) & m.Year.between(2000, 2018)].copy()
meta = pd.read_csv(RAW, usecols=["Index", "Net_type"]).drop_duplicates("Index")
m = m.merge(meta, on="Index", how="left")
m["Total"] = m[G].sum(axis=1)
m["EWE-Total"] = m[[f"EWE-{g}" for g in G]].sum(axis=1)
m["towrng"] = m["Tow_start_depth.m."].abs() - m["Tow_end_depth.m."].abs()
doy = pd.to_datetime(m.Date).dt.dayofyear.to_numpy(float)
m["dos"] = np.where(m.Month == 12, doy - 365, doy)
m["night"] = (m.Twilight == "Night").astype(float)
m["net"] = m.Net_type.str.split().str[0]
# season-year: Dec belongs to the following winter? pipeline keeps calendar Year; keep that.

# ---------------------------------------------------------------- model fixed-window indices from NetCDF
ds = xr.open_dataset(NC)
lat, lon = ds.lat.values, ds.lon.values
def cell(la, lo):
    d = (lat - la) ** 2 + ((lon - lo) * np.cos(np.deg2rad(la))) ** 2
    d = np.where(np.isfinite(d), d, np.inf)
    return np.unravel_index(np.argmin(d), d.shape)
cells = m[["ecospace_closest_lat", "ecospace_closest_lon"]].drop_duplicates()
cells["rc"] = [cell(a, b) for a, b in cells.values]
m = m.merge(cells, on=["ecospace_closest_lat", "ecospace_closest_lon"])
uniq = sorted(set(m.rc))
rows = np.array([r for r, c in uniq]); cols = np.array([c for r, c in uniq])
geo1_rc = m.loc[m.Station == "GEO1", "rc"].mode()[0]
t = pd.to_datetime(ds.time.values)
sel_t = (t.year >= 1999) & (t.year <= 2018)
mod_pts = {}
for g in G:
    arr = ds[g].isel(time=np.where(sel_t)[0]).values  # (t,row,col) -- big but bounded
    mod_pts[g] = arr[:, rows, cols]                     # (t, ncell)
    del arr
tt = t[sel_t]
tot = sum(mod_pts[g] for g in G)
cell_idx = {rc: i for i, rc in enumerate(uniq)}
# cell weights: number of tows per cell (sampling-weighted domain) and equal-weight
w_tow = np.array([(m.rc == rc).sum() for rc in uniq], float)

def model_window_index(season, series_2d, weights=None, single=None):
    months = SEAS[season]
    yrs = np.where(tt.month == 12, tt.year + 0, tt.year)  # calendar-year convention like obs
    out = {}
    for y in range(2000, 2019):
        k = np.isin(tt.month, months) & (yrs == y)
        if not k.any():
            continue
        x = series_2d[k]
        if single is not None:
            v = x[:, cell_idx[single]].mean()
        else:
            v = np.average(x.mean(axis=0), weights=weights)
        out[y] = np.log10(v + OFF)
    return pd.Series(out)

# ---------------------------------------------------------------- regression helpers
def year_index(d, ycol, terms):
    """OLS of ycol on year dummies + terms; return year effects (centered) and SE."""
    d = d.copy()
    years = sorted(d.Year.unique())
    X = [np.ones(len(d))]
    for y in years[1:]:
        X.append((d.Year == y).astype(float).values)
    if "doy" in terms:
        z = (d.dos - d.dos.mean()) / max(d.dos.std(), 1)
        X += [z.values, z.values ** 2, z.values ** 3]
    if "towrng" in terms:
        X.append(np.log10(d.towrng.values))
    if "night" in terms:
        X.append(d.night.values)
    if "net" in terms:
        for n in sorted(d.net.unique())[1:]:
            X.append((d.net == n).astype(float).values)
    if "station" in terms:
        for s in sorted(d.Station.unique())[1:]:
            X.append((d.Station == s).astype(float).values)
    X = np.column_stack(X)
    y = d[ycol].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    eff = np.r_[0.0, b[1:len(years)]]
    res = y - X @ b
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    coef = dict(zip(terms, [None] * len(terms)))
    extra = {}
    k = len(years)
    if "doy" in terms: k += 3
    if "towrng" in terms: extra["log10_towrng_coef"] = b[k]; k += 1
    if "night" in terms: extra["night_coef"] = b[k]; k += 1
    return pd.Series(eff - eff.mean(), index=years), extra, np.sqrt(res @ res / dof)

def r_ci(a, b):
    j = pd.concat([a, b], axis=1).dropna()
    if len(j) < 5:
        return np.nan, np.nan, np.nan, len(j)
    r, p = stats.pearsonr(j.iloc[:, 0], j.iloc[:, 1])
    z, se = np.arctanh(r), 1 / np.sqrt(len(j) - 3)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), len(j)

def fmt(x):
    r, lo, hi, n = x
    return f"{r:+.2f} [{lo:+.2f},{hi:+.2f}] n={n}" if np.isfinite(r) else "   n/a"

# ---------------------------------------------------------------- run
VARIANTS = [("year only", []), ("+doy", ["doy"]), ("+doy+towrng", ["doy", "towrng"]),
            ("+doy+towrng+night+net", ["doy", "towrng", "night", "net"]),
            ("+all+station FE", ["doy", "towrng", "night", "net", "station"])]
results = []
for grp in ["Total"] + G:
    mod_series = tot if grp == "Total" else mod_pts[grp]
    for season in SEAS:
        d = m[(m.Season == season) & m[grp].notna()].copy()
        d["lo"] = np.log10(d[grp] + OFF)
        d["lm"] = np.log10(d[f"EWE-{grp}"] + OFF)
        mw_tow = model_window_index(season, mod_series, weights=w_tow)
        mw_eq = model_window_index(season, mod_series)
        mw_tow = mw_tow[mw_tow.index.isin(d.Year.unique())]
        mw_eq = mw_eq[mw_eq.index.isin(d.Year.unique())]
        mod_towmatched, _, _ = year_index(d, "lm", ["doy"])
        for name, terms in VARIANTS:
            oi, extra, sd = year_index(d, "lo", terms)
            results.append(dict(group=grp, season=season, obs_variant=name,
                                r_vs_model_towmatched_doy=r_ci(oi, mod_towmatched)[0],
                                r_vs_model_window_sampledcells=r_ci(oi, mw_tow)[0],
                                r_vs_model_window_allsampledcells_eq=r_ci(oi, mw_eq)[0],
                                ci_window=fmt(r_ci(oi, mw_tow)), n_years=oi.size,
                                obs_index_sd=oi.std(), **extra))
res = pd.DataFrame(results)
res.to_csv(r"C:\Users\Greig\AppData\Local\Temp\claude\C--Users-Greig-Documents-GitHub-SOGEM-LTL-code\6d3f3c21-8737-4a15-ac6f-f7ee23211ad0\scratchpad\audit3_results.csv", index=False)
show = res[res.group == "Total"][["season", "obs_variant", "r_vs_model_towmatched_doy", "r_vs_model_window_sampledcells",
                                  "ci_window", "obs_index_sd", "log10_towrng_coef", "night_coef"]]
print("TOTAL CRUSTACEANS: standardized obs year index vs model")
print(show.round(2).to_string(index=False))
print("\nBest obs variant (+all+station FE) vs fixed-window model, by group:")
b = res[res.obs_variant == "+all+station FE"].pivot(index="group", columns="season", values="r_vs_model_window_sampledcells")
print(b[list(SEAS)].round(2).to_string())

# ---------------------------------------------------------------- GEO1 only
print("\nGEO1 ONLY (single deep station, one model cell)")
for grp in ["Total"] + G:
    mod_series = tot if grp == "Total" else mod_pts[grp]
    row = []
    for season in SEAS:
        d = m[(m.Station == "GEO1") & (m.Season == season) & m[grp].notna()].copy()
        if d.Year.nunique() < 6:
            row.append(f"{season}: n_yr={d.Year.nunique()}"); continue
        d["lo"] = np.log10(d[grp] + OFF)
        oi, _, _ = year_index(d, "lo", ["doy"] if len(d) > d.Year.nunique() + 4 else [])
        mw = model_window_index(season, mod_series, single=geo1_rc)
        r = r_ci(oi, mw[mw.index.isin(oi.index)])
        row.append(f"{season}: {fmt(r)}")
    print(f"  {grp:8s} " + " | ".join(row))
print("\nGEO1 tows per season-year:")
print(m[m.Station == "GEO1"].pivot_table(index="Year", columns="Season", values="Index", aggfunc="count").fillna(0).astype(int)[list(SEAS)].T.to_string())
