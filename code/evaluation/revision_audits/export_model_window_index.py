"""Export model indices free of sampling artefacts (seasonal and annual means at sampled cells, tow-count weighted)
plus the filtered tow table with covariates, for the GAM standardisation in R."""
import os, sys
import numpy as np, pandas as pd, xarray as xr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ecospace_eval_5d_zoop_sampling_diagnostics as d5
OUT = os.path.dirname(os.path.abspath(__file__))
m = d5.load_tows()
cols = ["Index", "Year", "Month", "Season", "Station", "Region", "dos", "doy", "towrng", "night", "net",
        "ecospace_closest_lat", "ecospace_closest_lon"] + ["Total"] + d5.G + ["EWE-Total"] + [f"EWE-{g}" for g in d5.G]
m[cols].to_csv(os.path.join(OUT, "gam_input_tows.csv"), index=False)
w = d5.model_window_indices(m)
rows = [dict(group=g, season=s, Year=y, log10_model=v) for (g, s), ser in w.items() for y, v in ser.items()]
# annual (all months) index
ds = xr.open_dataset(d5.NC)
lat, lon = ds.lat.values, ds.lon.values
def cell(la, lo):
    d = (lat - la) ** 2 + ((lon - lo) * np.cos(np.deg2rad(la))) ** 2
    return np.unravel_index(np.nanargmin(np.where(np.isfinite(d), d, np.inf)), d.shape)
cells = m[["ecospace_closest_lat", "ecospace_closest_lon"]].drop_duplicates()
cells["rc"] = [cell(a, b) for a, b in cells.values]
mm = m.merge(cells, on=["ecospace_closest_lat", "ecospace_closest_lon"])
uniq = sorted(set(mm.rc)); wt = np.array([(mm.rc == rc).sum() for rc in uniq], float)
r_ = np.array([a for a, _ in uniq]); c_ = np.array([b for _, b in uniq])
t = pd.to_datetime(ds.time.values); k = (t.year >= 2000) & (t.year <= 2018)
arrs = {g: ds[g].isel(time=np.where(k)[0]).values[:, r_, c_] for g in d5.G}
arrs["Total"] = sum(arrs.values())
for g, a in arrs.items():
    for y in range(2000, 2019):
        sel = t[k].year == y
        rows.append(dict(group=g, season="Annual", Year=y, log10_model=np.log10(np.average(a[sel].mean(axis=0), weights=wt) + 1e-6)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "gam_model_window_index.csv"), index=False)
print("exported", len(m), "tows;", len(rows), "model index rows")
