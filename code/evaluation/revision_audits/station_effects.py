"""Spatial component of the zooplankton model-observation mismatch.

For stations with >= 8 matched tows, station effects are estimated separately for observed and modelled
log10 biomass from an OLS fit with station, year and month as factors (so station contrasts are not
confounded with when each station was sampled). The two sets of station effects are then compared,
and related to water depth, tow depth range and latitude. Model grid-cell depth is checked against
the reported bottom depth to rule out a grid-depth mismatch.
Output: station_effects_results.txt, station_effects_total.csv (this folder).
"""
import os, sys
import numpy as np, pandas as pd, xarray as xr
from scipy import stats
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ecospace_eval_5d_zoop_sampling_diagnostics as d5

OFF = 1e-4
MIN_TOWS = 8
out = []


def say(s):
    print(s); out.append(s)


m = d5.load_tows()
m = m[m["EWE-Total"].notna()].copy()
ds = xr.open_dataset(d5.NC)
tree = cKDTree(np.c_[ds["lat"].values.ravel(), ds["lon"].values.ravel()])
_, ix = tree.query(np.c_[m.ecospace_closest_lat, m.ecospace_closest_lon])
m["celldepth"] = ds["depth"].values.ravel()[ix]


def station_effects(d, y):
    X = pd.get_dummies(d[["Station", "Year", "Month"]].astype(str)).astype(float)
    X = X.drop(columns=[[c for c in X if c.startswith(p)][0] for p in ("Year_", "Month_")])
    b, *_ = np.linalg.lstsq(X.to_numpy(), d[y].to_numpy(), rcond=None)
    s = pd.Series(b, index=X.columns)
    s = s[[c for c in s.index if c.startswith("Station_")]].rename(lambda c: c[len("Station_"):])
    return s - s.mean()


r_ = np.log10(m.celldepth / m["Bottom_depth.m."])
say(f"log10(model cell depth / station bottom depth): median {r_.median():.2f}, IQR {r_.quantile(.25):.2f} to {r_.quantile(.75):.2f}")
for g in ["Total"] + d5.G:
    d = m.assign(lo=np.log10(m[g].fillna(0) + OFF), lm=np.log10(m[f"EWE-{g}"] + OFF))
    n = d.Station.value_counts()
    d = d[d.Station.isin(n[n >= MIN_TOWS].index)]
    so, sm = station_effects(d, "lo"), station_effects(d, "lm")
    r = stats.pearsonr(so, sm)
    say(f"{g:8s} n stations {len(so):2d}; SD of station effects obs {so.std():.2f}, model {sm.std():.2f}; "
        f"r(obs, model) {r[0]:+.2f} (p = {r[1]:.2f})")
    if g == "Total":
        st = d.groupby("Station").agg(bottom_depth=("Bottom_depth.m.", "mean"), cell_depth=("celldepth", "mean"),
                                      tow_range=("towrng", "mean"), lat=("Latitude.W.", "mean"), n_tows=("Index", "size"))
        st["eff_obs"], st["eff_model"] = so, sm
        for c in ["bottom_depth", "tow_range", "lat"]:
            say(f"   r(station effect, {c}): obs {stats.pearsonr(st.eff_obs, st[c])[0]:+.2f}, "
                f"model {stats.pearsonr(st.eff_model, st[c])[0]:+.2f}")
        st.sort_values("lat").round(3).to_csv(os.path.join(HERE, "station_effects_total.csv"))
open(os.path.join(HERE, "station_effects_results.txt"), "w", encoding="utf-8").write("\n".join(out))
