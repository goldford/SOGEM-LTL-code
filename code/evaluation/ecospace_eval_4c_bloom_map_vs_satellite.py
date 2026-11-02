"""
ecospace_eval_4c_bloom_map_vs_satellite.py

Per-cell spring bloom timing: MODIS 8-day chl-a (Costa lab) vs Ecospace, 2003-2016.

Both sides are put on the Ecospace grid and the same 8-day bins:
- satellite: nearest MODIS pixel (~1 km) to each Ecospace cell centre;
- Ecospace: total phytoplankton (PP1-DIA + PP2-NAN + PP3-PIC), 3-day output averaged
  into the satellite's 8-day composites (DOY 1, 9, 17, ...).
The bloom rule is the one used for the satellite comparison in 4a/4b
(threshold = BT_THRESHOLD_FACTOR_SAT x annual median; persistence above
BT_SUB_THRESHOLD_FACTOR_SAT x threshold), applied per cell and year; bloom day is
the composite midpoint DOY.

Cells are excluded when the Ecospace peak biomass is below PEAK_FLOOR_FRAC of the
domain median peak (Fraser plume / Burrard Inlet, see 4b) or when no satellite
pixel lies within MAX_DIST_KM.

Outputs
-------
- ecospace_<SC>_bloom_doy_map_vs_sat.nc  (cached per-cell bloom DOY, both sources)
- ecospace_<SC>_bloom_doy_map_vs_sat.png
- ecospace_<SC>_bloom_doy_map_vs_sat_years.png  (selected years)
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.spatial import cKDTree
from scipy import stats

import ecospace_eval_config as cfg
from ecospace_eval_4b_bloom_map import bloom_doy_year

SC = cfg.ECOSPACE_SC
NC_MODEL = os.path.join(cfg.NC_PATH_OUT, cfg.NC_FILENAME)
SAT_NC = os.path.join("..", "..", "..", "SOGEM-LTL-data", "data", "reference", "satellite_chl",
                      "modis_chl_8day_interp_2003_2016.nc")
CACHE = os.path.join(cfg.EVALOUT_P, f"ecospace_{SC}_bloom_doy_map_vs_sat.nc")
FIG = os.path.join(cfg.FIGS_P, f"ecospace_{SC}_bloom_doy_map_vs_sat.png")
FIG_YRS = os.path.join(cfg.FIGS_P, f"ecospace_{SC}_bloom_doy_map_vs_sat_years.png")

YEARS = np.arange(2003, 2017)
SEL_YEARS = [2005, 2008, 2015]
VARS = cfg.BT_VARS_TO_ANALYZE_SAT
PEAK_FLOOR_FRAC = 0.1
MAX_DIST_KM = 1.5
RECOMPUTE = False
# satellite composites exist only for 8-day weeks 7-41 (~DOY 49-328); both sources use this
# window so the annual-median threshold is computed over the same season
WK = slice(6, 41)


def sat_on_grid(lat_m, lon_m, water):
    s = xr.open_dataset(SAT_NC)
    la, lo = np.meshgrid(s.lat.values, s.lon.values, indexing="ij")
    k = np.cos(np.deg2rad(49.5))
    valid = np.isfinite(s.chl.isel(time=slice(0, 46)).values).any(axis=0)
    tree = cKDTree(np.c_[la[valid], lo[valid] * k])
    dist, idx = tree.query(np.c_[lat_m[water], lon_m[water] * k])
    ok = dist * 111.0 <= MAX_DIST_KM
    iy, ix = np.where(valid)
    chl = s.chl.values[:, iy[idx], ix[idx]]          # (time, ncell_water)
    chl[:, ~ok] = np.nan
    return s.time.values, s.doy_mid.values, chl, ok


def model_8day(ds, water, yr):
    y = ds.sel(time=str(yr))
    tot = sum(y[g].values.astype("float32") for g in VARS).reshape(y.sizes["time"], -1)[:, water.ravel()]
    doy = pd.DatetimeIndex(y.time.values).dayofyear.values
    b = np.minimum((doy - 1) // 8, 45)
    out = np.stack([np.nanmean(tot[b == i], axis=0) if (b == i).any() else np.full(tot.shape[1], np.nan)
                    for i in range(46)])
    return out


def compute():
    ds = xr.open_dataset(NC_MODEL)
    depth = ds["depth"].values
    water = np.isfinite(depth) & (depth > 0)
    lat_m, lon_m = ds["lat"].values, ds["lon"].values
    t_sat, mid_sat, chl, ok = sat_on_grid(lat_m, lon_m, water)
    yrs_sat = pd.DatetimeIndex(t_sat).year.values
    nr, nc = water.shape
    B = {k: np.full((YEARS.size, nr, nc), np.nan, dtype="float32") for k in ("sat", "mod")}
    pk = np.full((YEARS.size, nr, nc), np.nan, dtype="float32")
    cov = np.full((YEARS.size, nr, nc), np.nan, dtype="float32")
    mid = (np.arange(46) * 8 + 4.5).clip(max=365)
    for k, yr in enumerate(YEARS):
        cs = chl[yrs_sat == yr]
        if cs.shape[0] != 46:
            print(f"{yr}: {cs.shape[0]} satellite composites; skipped")
            continue
        cm = model_8day(ds, water, yr)
        for key, v in (("sat", cs), ("mod", cm)):
            o = np.full(nr * nc, np.nan, dtype="float32")
            o[water.ravel()] = bloom_doy_year(v[WK], mid[WK])
            B[key][k] = o.reshape(nr, nc)
        o = np.full(nr * nc, np.nan, dtype="float32"); o[water.ravel()] = np.nanmax(cm, axis=0); pk[k] = o.reshape(nr, nc)
        o = np.full(nr * nc, np.nan, dtype="float32"); o[water.ravel()] = np.isfinite(cs[2:16]).mean(axis=0)
        cov[k] = o.reshape(nr, nc)
        print(yr, "median bloom DOY sat", np.nanmedian(B["sat"][k]), "model", np.nanmedian(B["mod"][k]))
    okg = np.zeros(nr * nc, bool); okg[np.flatnonzero(water.ravel())[ok]] = True
    out = xr.Dataset(
        {"bloom_sat": (("year", "row", "col"), B["sat"]), "bloom_mod": (("year", "row", "col"), B["mod"]),
         "peak_mod": (("year", "row", "col"), pk), "sat_cover_FebJun": (("year", "row", "col"), cov),
         "sat_matched": (("row", "col"), okg.reshape(nr, nc))},
        coords={"year": YEARS, "lat": (("row", "col"), lat_m), "lon": (("row", "col"), lon_m)})
    out.to_netcdf(CACHE)
    return out


def _map(ax, lon, lat, z, cmap, lev, lab, tag, fig, weak, cbar=True):
    norm = BoundaryNorm(lev, plt.get_cmap(cmap).N, extend="both")
    pm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(z), cmap=cmap, norm=norm, shading="auto")
    ax.pcolormesh(lon, lat, np.ma.masked_where(~weak, weak), cmap="Greys", vmin=0, vmax=2, shading="auto")
    ax.set_facecolor("0.92")
    ax.set_aspect(1 / np.cos(np.deg2rad(49.5)))
    ax.text(0.03, 0.03, tag, transform=ax.transAxes, fontsize=10)
    ax.tick_params(labelsize=7)
    if cbar:
        cb = fig.colorbar(pm, ax=ax, orientation="horizontal", shrink=0.9, pad=0.03)
        cb.set_label(lab, fontsize=8); cb.ax.tick_params(labelsize=7)
    return pm


def main():
    d = xr.open_dataset(CACHE) if (os.path.exists(CACHE) and not RECOMPUTE) else compute()
    lon, lat = d.lon.values, d.lat.values
    pk = np.nanmean(d.peak_mod.values, axis=0)
    weak = np.isfinite(pk) & (pk < PEAK_FLOOR_FRAC * np.nanmedian(pk))
    keep = np.isfinite(pk) & ~weak & d.sat_matched.values
    bs = np.where(keep, d.bloom_sat.values, np.nan)
    bm = np.where(keep, d.bloom_mod.values, np.nan)

    ms, mm = np.nanmean(bs, 0), np.nanmean(bm, 0)
    ss, sm = np.nanstd(bs, 0), np.nanstd(bm, 0)
    # per-cell interannual correlation
    a, b = bs - ms, bm - mm
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nansum(a * b, 0) / np.sqrt(np.nansum(a ** 2, 0) * np.nansum(b ** 2, 0))
    r[~keep] = np.nan

    f = np.isfinite(ms) & np.isfinite(mm)
    rs = stats.pearsonr(ms[f], mm[f])
    print(f"cells compared: {f.sum()}")
    print(f"mean bloom DOY, domain median: satellite {np.nanmedian(ms):.1f}, Ecospace {np.nanmedian(mm):.1f}; "
          f"median bias (model - sat) {np.nanmedian(mm - ms):.1f} d")
    print(f"spatial r of mean-DOY maps: {rs[0]:.2f}")
    print(f"interannual SD, domain median: satellite {np.nanmedian(ss):.1f}, Ecospace {np.nanmedian(sm):.1f}")
    print(f"per-cell interannual r: median {np.nanmedian(r):.2f}; share r>0.5: {np.nanmean(r[keep] > 0.5):.2f}")
    regs = np.nanmedian(bs[:, keep], axis=1); regm = np.nanmedian(bm[:, keep], axis=1)
    print("domain-median bloom DOY by year (sat, model):", list(zip(YEARS.tolist(), regs.round(0), regm.round(0))))
    print(f"domain-median series r = {stats.pearsonr(regs, regm)[0]:.2f}")

    fig, axs = plt.subplots(2, 3, figsize=(9.5, 8.6), constrained_layout=True)
    lev_m, lev_s = np.arange(50, 131, 5), np.arange(0, 31, 3)
    _map(axs[0, 0], lon, lat, ms, "viridis", lev_m, "", "(a) satellite", fig, weak, cbar=False)
    _map(axs[0, 1], lon, lat, mm, "viridis", lev_m, "Mean bloom day of year", "(b) Ecospace", fig, weak)
    _map(axs[0, 2], lon, lat, mm - ms, "RdBu_r", np.arange(-40, 41, 8), "Ecospace − satellite (days)", "(c)", fig, weak)
    _map(axs[1, 0], lon, lat, ss, "magma_r", lev_s, "", "(d) satellite", fig, weak, cbar=False)
    _map(axs[1, 1], lon, lat, sm, "magma_r", lev_s, "Interannual SD (days)", "(e) Ecospace", fig, weak)
    _map(axs[1, 2], lon, lat, r, "RdBu_r", np.arange(-1, 1.01, 0.2), "Interannual r (2003–2016)", "(f)", fig, weak)
    for ax in axs[:, 1:].ravel():
        ax.set_yticklabels([])
    fig.savefig(FIG, dpi=300); plt.close(fig); print("saved", FIG)

    fig, axs = plt.subplots(2, len(SEL_YEARS), figsize=(3.2 * len(SEL_YEARS), 8.2), constrained_layout=True)
    for j, yr in enumerate(SEL_YEARS):
        k = int(np.where(YEARS == yr)[0][0])
        _map(axs[0, j], lon, lat, bs[k], "viridis", lev_m, "", f"satellite {yr}", fig, weak, cbar=False)
        pm = _map(axs[1, j], lon, lat, bm[k], "viridis", lev_m, "", f"Ecospace {yr}", fig, weak, cbar=False)
    cb = fig.colorbar(pm, ax=axs, orientation="horizontal", shrink=0.6, pad=0.02)
    cb.set_label("Bloom day of year", fontsize=8)
    fig.savefig(FIG_YRS, dpi=300); plt.close(fig); print("saved", FIG_YRS)


if __name__ == "__main__":
    main()
