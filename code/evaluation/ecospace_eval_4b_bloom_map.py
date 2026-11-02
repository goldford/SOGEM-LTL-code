"""
ecospace_eval_4b_bloom_map.py

Spatial spring bloom timing from Ecospace.

For every water cell and year, the bloom day of year is found with the same
threshold rule used for the satellite comparison in 4a (helpers.find_bloom_doy):
  threshold = BT_THRESHOLD_FACTOR_SAT x annual median of total phytoplankton
  (PP1-DIA + PP2-NAN + PP3-PIC); the bloom is the first time step at or above the
  threshold where the mean of the next two (or the following two) steps stays above
  BT_SUB_THRESHOLD_FACTOR_SAT x threshold.

Outputs
-------
- ecospace_<SC>_bloom_doy_map.nc  (bloom_doy[year,row,col]; cached)
- ecospace_<SC>_bloom_doy_map.png (panel figure)
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

import ecospace_eval_config as cfg

SC = cfg.ECOSPACE_SC
NC_IN = os.path.join(cfg.NC_PATH_OUT, cfg.NC_FILENAME)
CACHE = os.path.join(cfg.EVALOUT_P, f"ecospace_{SC}_bloom_doy_map.nc")
FIG = os.path.join(cfg.FIGS_P, f"ecospace_{SC}_bloom_doy_map.png")
MASK_PF = cfg.BT_SAT_MASK_PF

YR_ST, YR_EN = cfg.BT_START_YEAR, cfg.BT_END_YEAR
VARS = cfg.BT_VARS_TO_ANALYZE_SAT
THR_F = cfg.BT_THRESHOLD_FACTOR_SAT
SUB_F = cfg.BT_SUB_THRESHOLD_FACTOR_SAT
RECOMPUTE = False
# cells whose mean annual peak biomass is below this fraction of the domain median peak are treated as no bloom
PEAK_FLOOR_FRAC = 0.1


def bloom_doy_year(v, doy):
    """v: (t, ncell) biomass for one year; doy: (t,). Returns bloom DOY per cell (NaN if none)."""
    nt = v.shape[0]
    thr = THR_F * np.nanmedian(v, axis=0)
    sub = SUB_F * thr
    pad = np.full((4, v.shape[1]), np.nan)
    vp = np.vstack([v, pad])
    above = v[: nt - 2] >= thr
    nxt1 = (vp[1: nt - 1] + vp[2: nt]) / 2 > sub
    nxt2 = (vp[3: nt + 1] + vp[4: nt + 2]) / 2 > sub
    ok = above & (nxt1 | nxt2)
    first = ok.argmax(axis=0)
    out = doy[: nt - 2][first].astype(float)
    out[~ok.any(axis=0)] = np.nan
    return out


def compute():
    ds = xr.open_dataset(NC_IN)
    water = np.isfinite(ds["depth"].values) & (ds["depth"].values > 0)
    rows, cols = ds.sizes["row"], ds.sizes["col"]
    years = np.arange(YR_ST, YR_EN + 1)
    res = np.full((years.size, rows, cols), np.nan, dtype="float32")
    peak = np.full((years.size, rows, cols), np.nan, dtype="float32")
    for k, yr in enumerate(years):
        y = ds.sel(time=str(yr))
        tot = sum(y[g].values.astype("float32") for g in VARS)
        v = tot.reshape(tot.shape[0], -1)[:, water.ravel()]
        doy = pd.DatetimeIndex(y.time.values).dayofyear.values
        out = np.full(rows * cols, np.nan, dtype="float32")
        out[water.ravel()] = bloom_doy_year(v, doy)
        res[k] = out.reshape(rows, cols)
        pk = np.full(rows * cols, np.nan, dtype="float32"); pk[water.ravel()] = np.nanmax(v, axis=0)
        peak[k] = pk.reshape(rows, cols)
        print(yr, np.nanmedian(res[k]))
    da = xr.Dataset(
        {"bloom_doy": (("year", "row", "col"), res), "peak_B": (("year", "row", "col"), peak)},
        coords={"year": years, "lat": (("row", "col"), ds["lat"].values),
                "lon": (("row", "col"), ds["lon"].values), "depth": (("row", "col"), ds["depth"].values)},
    )
    da.to_netcdf(CACHE)
    return da


def main():
    da = xr.open_dataset(CACHE) if (os.path.exists(CACHE) and not RECOMPUTE) else compute()
    b = da["bloom_doy"].values.copy()
    pk = np.nanmean(da["peak_B"].values, axis=0)
    weak = np.isfinite(pk) & (pk < PEAK_FLOOR_FRAC * np.nanmedian(pk))
    b[:, weak] = np.nan
    lon, lat = da["lon"].values, da["lat"].values
    mask = xr.open_dataset(MASK_PF)["mask"].values.astype(bool)

    # regional (satellite-region) series, used for coherence
    reg = np.nanmedian(b[:, mask], axis=1)
    mean = np.nanmean(b, axis=0)
    sd = np.nanstd(b, axis=0)
    ba = b - np.nanmean(b, axis=0)
    ra = reg - reg.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nansum(ba * ra[:, None, None], axis=0) / np.sqrt(
            np.nansum(ba ** 2, axis=0) * np.nansum(ra ** 2))
    r[~np.isfinite(mean)] = np.nan

    fig, axs = plt.subplots(1, 3, figsize=(9.5, 4.6), constrained_layout=True)
    specs = [
        (mean, "viridis", np.arange(60, 121, 5), "Mean bloom day of year"),
        (sd, "magma_r", np.arange(0, 31, 3), "Interannual SD (days)"),
        (r, "RdBu_r", np.arange(-1, 1.01, 0.2), "r with bloom day at ★"),
    ]
    for ax, (z, cmap, lev, lab), tag in zip(axs, specs, "abc"):
        norm = BoundaryNorm(lev, plt.get_cmap(cmap).N, extend="both")
        pm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(z), cmap=cmap, norm=norm, shading="auto")
        ax.pcolormesh(lon, lat, np.ma.masked_where(~weak, weak), cmap="Greys", vmin=0, vmax=2, shading="auto")
        ax.plot(lon[mask].mean(), lat[mask].mean(), marker="*", ms=9, mfc="w", mec="k", mew=0.8)
        ax.set_facecolor("0.92")
        ax.set_xlim(np.nanmin(lon[np.isfinite(pk)]) - 0.05, np.nanmax(lon[np.isfinite(pk)]) + 0.05)
        ax.set_ylim(np.nanmin(lat[np.isfinite(pk)]) - 0.05, np.nanmax(lat[np.isfinite(pk)]) + 0.05)
        ax.set_aspect(1 / np.cos(np.deg2rad(49.5)))
        ax.text(0.03, 0.03, f"({tag})", transform=ax.transAxes, fontsize=10)
        ax.tick_params(labelsize=7)
        if tag != "a":
            ax.set_yticklabels([])
        cb = fig.colorbar(pm, ax=ax, orientation="horizontal", shrink=0.9, pad=0.02)
        cb.set_label(lab, fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.savefig(FIG, dpi=300)
    print("saved", FIG)
    print(f"regional median bloom DOY: mean {reg.mean():.1f}, sd {reg.std():.1f}")
    print(f"domain mean-DOY range (5-95%): {np.nanpercentile(mean, [5, 95])}")
    print(f"weak (no-bloom) water cells: {weak.sum()} of {np.isfinite(pk).sum()}")
    print(f"share of cells with r>0.5: {np.nanmean(r > 0.5):.2f}; no-bloom cell-years: {np.isnan(b[:, np.isfinite(mean)]).mean():.3f}")


if __name__ == "__main__":
    main()
