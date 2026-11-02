"""
ecospace_eval_4d_bloom_snapshot_maps.py

Illustrative spatial comparison of one spring bloom: MODIS 8-day chl-a (DINEOF
gap-filled; Costa lab, as used by Suchy et al. 2022) vs Ecospace total phytoplankton
(PP1-DIA + PP2-NAN + PP3-PIC).

Each source is expressed relative to its own median over the SGC2 region for the
weeks with satellite data that year (~mid-Feb to mid-Nov), i.e. the quantity behind
the bloom threshold in Figure 6 (bloom = 1.05 x median). Ecospace 3-day output is
averaged into the satellite's 8-day weeks (DOY 1, 9, 17, ...).

Satellite chl-a (near-surface concentration) and Ecospace biomass (areal pool) differ
in kind, so each row has its own colour scale; both diverge at the bloom threshold
(1.05).

Output: ecospace_<SC>_bloom_snapshots_<YEAR>.png
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.path import Path
from scipy.spatial import cKDTree

import ecospace_eval_config as cfg

SC = cfg.ECOSPACE_SC
NC_MODEL = os.path.join(cfg.NC_PATH_OUT, cfg.NC_FILENAME)
SAT_NC = os.path.join("..", "..", "..", "SOGEM-LTL-data", "data", "reference", "satellite_chl",
                      "modis_chl_8day_interp_2003_2016.nc")
POLY = os.path.join("..", "..", "..", "SOGEM-LTL-data", "data", "reference", "analysis_domains_suchy.yml")
FIG = os.path.join(cfg.FIGS_P, "ecospace_{sc}_bloom_snapshots_{yr}.png")

SNAP_YEARS = [2010, 2015, 2016]
OVERVIEW_YEARS = list(range(2003, 2017))
LAND = "#ddd6c8"                               # land (GSHHS full resolution)
NODATA = "0.55"                                # Fraser plume cells (negligible model phytoplankton)
VARS = cfg.BT_VARS_TO_ANALYZE_SAT
REGION = "SGC2"
PEAK_FLOOR_FRAC = 0.1                          # as in 4b: near-zero Ecospace cells (Fraser plume) shown grey
MAX_DIST_KM = 1.5
WK_START = np.arange(46) * 8 + 1

LEVELS = {"sat": [0.25, 0.5, 0.75, 0.9, 1.05, 1.5, 2, 3, 4],
          "mod": [0.75, 0.85, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3, 1.5]}
LEVELS_SPRING = {"sat": [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0],
                 "mod": [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]}
SPRING_WEEKS = np.arange(6, 19)                # 0-based week index; starts DOY 49..145 (~18 Feb-31 May)
ROW_LABEL = {"sat": "Satellite chl a", "mod": "Ecospace phytoplankton B"}


def cmap_norm(levels):
    cm = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0.05, 0.95, len(levels) + 1)))
    return cm, BoundaryNorm(levels, cm.N, extend="both")


def region_poly():
    v = np.asarray(yaml.safe_load(open(POLY))["polygon_coords"][REGION])
    if not np.allclose(v[0], v[-1]):
        v = np.vstack([v, v[0]])
    return v


def model_weekly(ds, cells, yr):
    y = ds.sel(time=str(yr))
    tot = sum(y[g].values for g in VARS)
    doy = pd.DatetimeIndex(y.time.values).dayofyear.values
    wk = np.minimum((doy - 1) // 8, 45)
    out = np.full((46,) + cells.shape, np.nan, dtype="float32")
    for i in range(46):
        if (wk == i).any():
            out[i] = np.nanmean(tot[wk == i], axis=0)
    out[:, ~cells] = np.nan
    return out


def sat_weekly(s, yr):
    t = pd.DatetimeIndex(s.time.values)
    c = s.chl.values[t.year == yr]
    wk = (t[t.year == yr].dayofyear.values - 1) // 8
    out = np.full((46,) + c.shape[1:], np.nan, dtype="float32")
    out[wk] = c
    return out


def ratio_to_median(w, reg, avail):
    """Divide a weekly field by the median of its regional weekly medians over the available weeks."""
    reg_series = np.nanmedian(w[:, reg], axis=1)
    med = np.nanmedian(reg_series[avail])
    return w / med, reg_series / med


def sat_onset_week(ser):
    """Suchy et al. (2022) rule on the regional weekly series: first week >= 1.05 x median with
    at least one of the next two weeks > 0.7 x threshold. Returns 0-based week index."""
    idx = np.flatnonzero(np.isfinite(ser)); v = ser[idx]
    for j in range(len(v) - 2):
        if v[j] >= 1.05 and (v[j + 1] > 0.735 or v[j + 2] > 0.735):
            return int(idx[j])
    return None


def prepare(ds, s, yr):
    lat_m, lon_m = ds["lat"].values, ds["lon"].values
    mw = model_weekly(ds, np.ones(lat_m.shape, bool), yr)
    cells = np.isfinite(mw).any(axis=0)
    peak = np.nanmax(mw, axis=0)
    weak = cells & (peak < PEAK_FLOOR_FRAC * np.nanmedian(peak[cells]))
    poly_v = region_poly(); poly = Path(poly_v)
    reg_m = poly.contains_points(np.c_[lat_m.ravel(), lon_m.ravel()]).reshape(lat_m.shape) & cells
    la_s, lo_s = np.meshgrid(s.lat.values, s.lon.values, indexing="ij")
    reg_s = poly.contains_points(np.c_[la_s.ravel(), lo_s.ravel()]).reshape(la_s.shape)
    k = np.cos(np.deg2rad(49.5))
    dist, _ = cKDTree(np.c_[lat_m[cells & ~weak], lon_m[cells & ~weak] * k]).query(np.c_[la_s.ravel(), lo_s.ravel() * k])
    foot = (dist * 111.0 <= MAX_DIST_KM).reshape(la_s.shape)
    sw = sat_weekly(s, yr)
    avail = np.isfinite(np.nanmedian(sw[:, reg_s], axis=1))
    rs, rs_ser = ratio_to_median(sw, reg_s, avail)
    rm, rm_ser = ratio_to_median(mw, reg_m, avail)
    rs[:, ~foot] = np.nan
    rm[:, weak] = np.nan
    extent = (np.nanmin(lon_m[cells]) - 0.03, np.nanmax(lon_m[cells]) + 0.03,
              np.nanmin(lat_m[cells]) - 0.03, np.nanmax(lat_m[cells]) + 0.03)
    return dict(lat_m=lat_m, lon_m=lon_m, la_s=la_s, lo_s=lo_s, cells=cells, weak=weak, poly_v=poly_v,
                rs=rs, rm=rm, rs_ser=rs_ser, rm_ser=rm_ser, avail=avail, extent=extent)


GSHHS_F = os.path.expanduser(os.path.join("~", ".local", "share", "cartopy", "shapefiles", "gshhs", "f", "GSHHS_f_L1.shp"))
SEA_OUT = "#eef2f6"                            # sea outside the model domain
COAST = "0.35"
PLACE_LABELS = [("Vancouver Island", -124.55, 49.12, -38), ("Fraser R.", -122.78, 49.24, 0)]
_LAND = None


def land_polygons(extent):
    """Full-resolution GSHHS land polygons (cartopy cache), clipped to extent; list of (N, 2) lon/lat arrays."""
    global _LAND
    if _LAND is None:
        import shapefile
        from shapely.geometry import box, shape
        bb = box(extent[0], extent[2], extent[1], extent[3])
        polys = []
        sf = shapefile.Reader(GSHHS_F)
        for s in sf.iterShapes():
            x0, y0, x1, y1 = s.bbox
            if x1 < extent[0] or x0 > extent[1] or y1 < extent[2] or y0 > extent[3]:
                continue
            g = shape(s.__geo_interface__)
            if not g.intersects(bb):
                continue
            g = g.intersection(bb)
            for part in getattr(g, "geoms", [g]):
                if part.geom_type == "Polygon" and not part.is_empty:
                    polys.append(np.asarray(part.exterior.coords))
        _LAND = polys
    return _LAND


def draw_land(ax, extent, labels=False):
    from matplotlib.collections import PolyCollection
    ax.add_collection(PolyCollection(land_polygons(extent), facecolor=LAND, edgecolor=COAST, linewidth=0.25,
                                     zorder=3, rasterized=True))
    if labels:
        for txt, x, y, rot in PLACE_LABELS:
            ax.text(x, y, txt, fontsize=6, style="italic", color="0.3", rotation=rot, ha="center", va="center",
                    zorder=4)


def panel(ax, P, kind, w, label=True, fs=7, places=False):
    lo, la, R = (P["lo_s"], P["la_s"], P["rs"]) if kind == "sat" else (P["lon_m"], P["lat_m"], P["rm"])
    cm, nm = cmap_norm(LEVELS[kind])
    ax.set_facecolor(SEA_OUT)
    sea = np.where(P["cells"], 1.0, np.nan)
    ax.pcolormesh(P["lon_m"], P["lat_m"], np.ma.masked_invalid(sea), cmap=ListedColormap(["0.82"]),
                  shading="auto", rasterized=True)
    pm = ax.pcolormesh(lo, la, np.ma.masked_invalid(R[w]), cmap=cm, norm=nm, shading="auto", rasterized=True)
    ax.pcolormesh(P["lon_m"], P["lat_m"], np.ma.masked_where(~P["weak"], P["weak"]),
                  cmap=ListedColormap([NODATA]), shading="auto", rasterized=True)
    draw_land(ax, P["extent"], labels=places)
    ax.plot(P["poly_v"][:, 1], P["poly_v"][:, 0], color="k", lw=0.6, ls="--", zorder=5)
    e = P["extent"]
    ax.set_xlim(e[:2]); ax.set_ylim(e[2:])
    ax.set_aspect(1 / np.cos(np.deg2rad(49.5)))
    ax.set_xticks([]); ax.set_yticks([])
    if label:
        ser = P["rs_ser"] if kind == "sat" else P["rm_ser"]
        ax.text(0.04, 0.04, f"×{ser[w]:.2f}", transform=ax.transAxes, fontsize=fs,
                bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.2))
    return pm


def strip(ds, s, yr):
    P = prepare(ds, s, yr)
    on = sat_onset_week(P["rs_ser"])
    first = max(on - 1, int(np.flatnonzero(P["avail"])[0]))
    cols = list(range(first, first + 4))
    fig, axs = plt.subplots(2, 4, figsize=(8.7, 5.9), constrained_layout=True)
    for i, kind in enumerate(("sat", "mod")):
        for j, w in enumerate(cols):
            pm = panel(axs[i, j], P, kind, w)
            if j == 0:
                axs[i, j].set_ylabel(ROW_LABEL[kind], fontsize=9)
            if i == 0:
                d0 = pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=int(WK_START[w]) - 1)
                axs[i, j].set_title(f"{d0:%d %b} – {d0 + pd.Timedelta(days=7):%d %b}" + (" (onset)" if w == on else ""),
                                    fontsize=8.5)
        cb = fig.colorbar(pm, ax=axs[i, :], orientation="vertical", shrink=0.9, pad=0.01, ticks=LEVELS[kind])
        cb.set_label("× median", fontsize=8); cb.ax.tick_params(labelsize=7)
    p = FIG.format(sc=SC, yr=yr)
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"saved {p}; satellite onset week starts DOY {WK_START[on]}")
    print("   SGC2 ratio sat/model:", " ".join(f"{WK_START[w]}:{P['rs_ser'][w]:.2f}/{P['rm_ser'][w]:.2f}" for w in range(max(on - 2, 6), on + 5)))


def overview(ds, s):
    fig, axs = plt.subplots(2, len(OVERVIEW_YEARS), figsize=(1.25 * len(OVERVIEW_YEARS), 3.6), constrained_layout=True)
    for j, yr in enumerate(OVERVIEW_YEARS):
        P = prepare(ds, s, yr)
        on = sat_onset_week(P["rs_ser"])
        for i, kind in enumerate(("sat", "mod")):
            pm = panel(axs[i, j], P, kind, on + 1, fs=5.5)
            if j == 0:
                axs[i, j].set_ylabel("Satellite" if kind == "sat" else "Ecospace", fontsize=8)
        d1 = pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=int(WK_START[on + 1]) - 1)
        axs[0, j].set_title(f"{yr}\nwk of {d1:%d %b}", fontsize=7)
    p = os.path.join(cfg.FIGS_P, f"ecospace_{SC}_bloom_snapshots_overview.png")
    fig.savefig(p, dpi=220); plt.close(fig); print("saved", p, "(week after satellite onset, each year)")


def spring_mean(ds, s):
    """C: spring (weeks starting DOY 49-145) mean ratio to the annual median, averaged over OVERVIEW_YEARS."""
    acc = {"sat": [], "mod": []}
    for yr in OVERVIEW_YEARS:
        P = prepare(ds, s, yr)
        for kind, R in (("sat", P["rs"]), ("mod", P["rm"])):
            acc[kind].append(np.nanmean(R[SPRING_WEEKS], axis=0))
    P["rs"] = np.nanmean(acc["sat"], 0)[None]
    P["rm"] = np.nanmean(acc["mod"], 0)[None]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 4.4), constrained_layout=True)
    for ax, kind, lab in ((axs[0], "sat", "(a) Satellite chl a"), (axs[1], "mod", "(b) Ecospace phytoplankton B")):
        pm = panel(ax, P, kind, 0, label=False)
        ax.set_title(lab, fontsize=9, loc="left")
        lev = LEVELS_SPRING[kind]
        cm, nm = cmap_norm(lev)
        ax.collections[1].set_cmap(cm); ax.collections[1].set_norm(nm)
        cb = fig.colorbar(ax.collections[1], ax=ax, orientation="horizontal", shrink=0.85, pad=0.02, ticks=lev)
        cb.set_label("Feb–May mean × annual median", fontsize=8); cb.ax.tick_params(labelsize=6.5)
    p = os.path.join(cfg.FIGS_P, f"ecospace_{SC}_spring_mean_ratio_map.png")
    fig.savefig(p, dpi=300); plt.close(fig); print("saved", p)


def block_climatology(ds, s, blocks, tag, levels):
    """Maps of the across-year median (OVERVIEW_YEARS) of each block's mean ratio.
    blocks: list of (label, [0-based week indices]); week assigned by its midpoint date."""
    acc = {"sat": {b: [] for b, _ in blocks}, "mod": {b: [] for b, _ in blocks}}
    for yr in OVERVIEW_YEARS:
        P = prepare(ds, s, yr)
        for b, wks in blocks:
            acc["sat"][b].append(np.nanmean(P["rs"][wks], axis=0))
            acc["mod"][b].append(np.nanmean(P["rm"][wks], axis=0))
    nb = len(blocks)
    fig, axs = plt.subplots(2, nb, figsize=(2.1 * nb + 0.9, 5.9), constrained_layout=True)
    for i, kind in enumerate(("sat", "mod")):
        stack = np.stack([np.nanmedian(np.stack(acc[kind][b]), axis=0) for b, _ in blocks])
        if kind == "sat":
            P["rs"] = stack
        else:
            P["rm"] = stack
        for j, (b, _) in enumerate(blocks):
            ax = axs[i, j]
            panel(ax, P, kind, j, label=False, places=(i == 0 and j == 0))
            cm, nm = cmap_norm(levels[kind])
            ax.collections[1].set_cmap(cm); ax.collections[1].set_norm(nm)
            reg = Path(P["poly_v"])
            if kind == "sat":
                inside = reg.contains_points(np.c_[P["la_s"].ravel(), P["lo_s"].ravel()]).reshape(P["la_s"].shape)
            else:
                inside = reg.contains_points(np.c_[P["lat_m"].ravel(), P["lon_m"].ravel()]).reshape(P["lat_m"].shape)
            ax.text(0.04, 0.04, f"×{np.nanmedian(stack[j][inside]):.2f}", transform=ax.transAxes, fontsize=7,
                    bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.2))
            if j == 0:
                ax.set_ylabel(ROW_LABEL[kind], fontsize=9)
            if i == 0:
                ax.set_title(b, fontsize=9)
        cb = fig.colorbar(axs[i, 0].collections[1], ax=axs[i, :], orientation="vertical", shrink=0.9, pad=0.01,
                          ticks=levels[kind])
        cb.set_label("× annual median", fontsize=8); cb.ax.tick_params(labelsize=7)
    # spatial pattern agreement per block (log ratios; satellite sampled at nearest pixel to each model cell)
    k = np.cos(np.deg2rad(49.5))
    okm = P["cells"] & ~P["weak"]
    fs = np.isfinite(P["rs"][0])
    _, idx = cKDTree(np.c_[P["la_s"][fs], P["lo_s"][fs] * k]).query(np.c_[P["lat_m"][okm], P["lon_m"][okm] * k])
    for j, (b, _) in enumerate(blocks):
        a = np.log(P["rs"][j][fs][idx]); m = np.log(P["rm"][j][okm])
        f = np.isfinite(a) & np.isfinite(m)
        print(f"  {b:12s} spatial r = {np.corrcoef(a[f], m[f])[0, 1]:.2f} (n={f.sum()})")
    p = os.path.join(cfg.FIGS_P, f"ecospace_{SC}_spring_blocks_{tag}.png")
    fig.savefig(p, dpi=300); plt.close(fig); print("saved", p)


DIFF_LEVELS = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]


def block_climatology_diff(ds, s, blocks, tag, levels):
    """As block_climatology, plus a third row: difference of standardised spatial patterns.
    Satellite is sampled at the nearest pixel to each Ecospace cell (within MAX_DIST_KM); within each block,
    log values of each source are converted to z-scores across cells where both exist; row 3 = z_Ecospace - z_satellite."""
    acc = {"sat": {b: [] for b, _ in blocks}, "mod": {b: [] for b, _ in blocks}}
    for yr in OVERVIEW_YEARS:
        P = prepare(ds, s, yr)
        for b, wks in blocks:
            acc["sat"][b].append(np.nanmean(P["rs"][wks], axis=0))
            acc["mod"][b].append(np.nanmean(P["rm"][wks], axis=0))
    S = np.stack([np.nanmedian(np.stack(acc["sat"][b]), axis=0) for b, _ in blocks])
    Mm = np.stack([np.nanmedian(np.stack(acc["mod"][b]), axis=0) for b, _ in blocks])
    P["rs"], P["rm"] = S, Mm

    # satellite on the Ecospace grid
    k = np.cos(np.deg2rad(49.5))
    okm = P["cells"] & ~P["weak"]
    fs = np.isfinite(S[0])
    dist, idx = cKDTree(np.c_[P["la_s"][fs], P["lo_s"][fs] * k]).query(np.c_[P["lat_m"][okm], P["lon_m"][okm] * k])
    near = dist * 111.0 <= MAX_DIST_KM
    D = np.full(Mm.shape, np.nan, dtype="float32")
    rvals = []
    for j in range(len(blocks)):
        a = np.full(okm.sum(), np.nan); a[near] = np.log(S[j][fs][idx[near]])
        m = np.log(Mm[j][okm])
        f = np.isfinite(a) & np.isfinite(m)
        za = (a - a[f].mean()) / a[f].std(); zm = (m - m[f].mean()) / m[f].std()
        d = np.full(okm.sum(), np.nan); d[f] = zm[f] - za[f]
        D[j][okm] = d
        rvals.append(np.corrcoef(a[f], m[f])[0, 1])
        print(f"  {blocks[j][0]:8s} spatial r = {np.corrcoef(a[f], m[f])[0, 1]:.2f}; "
              f"|dz| > 1 in {np.mean(np.abs(d[f]) > 1):.0%} of cells (n={f.sum()})")

    nb = len(blocks)
    fig, axs = plt.subplots(3, nb, figsize=(2.1 * nb + 0.9, 8.6), constrained_layout=True)
    reg = Path(P["poly_v"])
    for i, kind in enumerate(("sat", "mod")):
        stack = S if kind == "sat" else Mm
        if kind == "sat":
            inside = reg.contains_points(np.c_[P["la_s"].ravel(), P["lo_s"].ravel()]).reshape(P["la_s"].shape)
        else:
            inside = reg.contains_points(np.c_[P["lat_m"].ravel(), P["lon_m"].ravel()]).reshape(P["lat_m"].shape)
        for j, (b, _) in enumerate(blocks):
            ax = axs[i, j]
            panel(ax, P, kind, j, label=False, places=(i == 0 and j == 0))
            cm, nm = cmap_norm(levels[kind])
            ax.collections[1].set_cmap(cm); ax.collections[1].set_norm(nm)
            ax.text(0.04, 0.04, f"×{np.nanmedian(stack[j][inside]):.2f}", transform=ax.transAxes, fontsize=7,
                    bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.2))
            if j == 0:
                ax.set_ylabel(ROW_LABEL[kind], fontsize=9)
            if i == 0:
                ax.set_title(b, fontsize=9)
        cb = fig.colorbar(axs[i, 0].collections[1], ax=axs[i, :], orientation="vertical", shrink=0.9, pad=0.01,
                          ticks=levels[kind])
        cb.set_label("× annual median", fontsize=8); cb.ax.tick_params(labelsize=7)
    # row 3: standardised pattern difference
    P["rm"] = D
    cmd = ListedColormap(plt.get_cmap("PuOr_r")(np.linspace(0.05, 0.95, len(DIFF_LEVELS) + 1)))
    nmd = BoundaryNorm(DIFF_LEVELS, cmd.N, extend="both")
    for j in range(nb):
        ax = axs[2, j]
        panel(ax, P, "mod", j, label=False)
        ax.collections[1].set_cmap(cmd); ax.collections[1].set_norm(nmd)
        ax.text(0.04, 0.04, f"r = {rvals[j]:.2f}", transform=ax.transAxes, fontsize=7,
                bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.2))
        if j == 0:
            ax.set_ylabel("Pattern difference\n(Ecospace − satellite)", fontsize=9)
    cb = fig.colorbar(axs[2, 0].collections[1], ax=axs[2, :], orientation="vertical", shrink=0.9, pad=0.01,
                      ticks=DIFF_LEVELS)
    cb.set_label("Δ standardised anomaly (SD)", fontsize=8); cb.ax.tick_params(labelsize=7)
    p = os.path.join(cfg.FIGS_P, f"ecospace_{SC}_spring_blocks_{tag}.png")
    fig.savefig(p, dpi=300); plt.close(fig); print("saved", p)


def week_blocks(kind):
    mid = pd.Timestamp(2001, 1, 1) + pd.to_timedelta(WK_START - 1 + 3.5, unit="D")
    if kind == "monthly":
        return [(pd.Timestamp(2001, m, 1).strftime("%B"), [w for w in range(46) if mid[w].month == m]) for m in (3, 4, 5, 6)]
    edges = [(3, 1), (3, 16), (4, 1), (4, 16), (5, 1), (5, 16), (6, 1)]
    out = []
    for (m0, d0), (m1, d1) in zip(edges[:-1], edges[1:]):
        a, b = pd.Timestamp(2001, m0, d0), pd.Timestamp(2001, m1, d1)
        out.append((f"{a:%d %b}–{b - pd.Timedelta(days=1):%d %b}", [w for w in range(46) if a <= mid[w] < b]))
    return out


BLOCK_LEVELS = {"sat": [0.5, 0.67, 0.8, 0.9, 1.05, 1.25, 1.5, 2, 3],
                "mod": [0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3, 1.5]}


def main():
    ds = xr.open_dataset(NC_MODEL)
    s = xr.open_dataset(SAT_NC)
    block_climatology_diff(ds, s, week_blocks("monthly"), "monthly", BLOCK_LEVELS)


def _snapshots_and_overview():
    ds = xr.open_dataset(NC_MODEL)
    s = xr.open_dataset(SAT_NC)
    for yr in SNAP_YEARS:
        strip(ds, s, yr)
    overview(ds, s)
    spring_mean(ds, s)


def _old_main():
    ds = xr.open_dataset(NC_MODEL)
    lat_m, lon_m = ds["lat"].values, ds["lon"].values
    mw = model_weekly(ds, np.ones(lat_m.shape, bool), SNAP_YEAR)
    cells = np.isfinite(mw).any(axis=0)                       # cells with model output
    peak = np.nanmax(mw, axis=0)
    weak = cells & (peak < PEAK_FLOOR_FRAC * np.nanmedian(peak[cells]))
    poly_v = region_poly(); poly = Path(poly_v)
    reg_m = poly.contains_points(np.c_[lat_m.ravel(), lon_m.ravel()]).reshape(lat_m.shape) & cells

    s = xr.open_dataset(SAT_NC)
    la_s, lo_s = np.meshgrid(s.lat.values, s.lon.values, indexing="ij")
    reg_s = poly.contains_points(np.c_[la_s.ravel(), lo_s.ravel()]).reshape(la_s.shape)
    k = np.cos(np.deg2rad(49.5))
    dist, _ = cKDTree(np.c_[lat_m[cells & ~weak], lon_m[cells & ~weak] * k]).query(np.c_[la_s.ravel(), lo_s.ravel() * k])
    foot = (dist * 111.0 <= MAX_DIST_KM).reshape(la_s.shape)

    sw = sat_weekly(s, SNAP_YEAR)
    avail = np.isfinite(np.nanmedian(sw[:, reg_s], axis=1))
    rs, rs_ser = ratio_to_median(sw, reg_s, avail)
    rm, rm_ser = ratio_to_median(mw, reg_m, avail)
    rs[:, ~foot] = np.nan
    rm[:, weak] = np.nan

    extent = (np.nanmin(lon_m[cells]) - 0.03, np.nanmax(lon_m[cells]) + 0.03,
              np.nanmin(lat_m[cells]) - 0.03, np.nanmax(lat_m[cells]) + 0.03)
    cols = [(d - 1) // 8 for d in SNAP_WEEK_STARTS]
    fig, axs = plt.subplots(2, len(cols), figsize=(1.95 * len(cols) + 0.9, 5.9), constrained_layout=True)
    for i, (kind, R, ser, la, lo) in enumerate((("sat", rs, rs_ser, la_s, lo_s), ("mod", rm, rm_ser, lat_m, lon_m))):
        cm, nm = cmap_norm(LEVELS[kind])
        for j, w in enumerate(cols):
            ax = axs[i, j]
            pm = ax.pcolormesh(lo, la, np.ma.masked_invalid(R[w]), cmap=cm, norm=nm, shading="auto", rasterized=True)
            ax.pcolormesh(lon_m, lat_m, np.ma.masked_where(~weak, weak), cmap="Greys", vmin=0, vmax=2.2,
                          shading="auto", rasterized=True)
            ax.set_facecolor("0.93")
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
            ax.set_aspect(1 / np.cos(np.deg2rad(49.5)))
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.04, 0.04, f"{REGION} ×{ser[w]:.2f}", transform=ax.transAxes, fontsize=7,
                    bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5))
            if j == 0:
                ax.plot(poly_v[:, 1], poly_v[:, 0], color="k", lw=0.7, ls="--")
                ax.set_ylabel(ROW_LABEL[kind], fontsize=9)
            if i == 0:
                d0 = pd.Timestamp(SNAP_YEAR, 1, 1) + pd.Timedelta(days=int(WK_START[w]) - 1)
                ax.set_title(f"{d0:%d %b} – {d0 + pd.Timedelta(days=7):%d %b}", fontsize=8.5)
        cb = fig.colorbar(pm, ax=axs[i, :], orientation="vertical", shrink=0.9, pad=0.01, ticks=LEVELS[kind])
        cb.set_label("× median", fontsize=8); cb.ax.tick_params(labelsize=7)
    p = FIG.format(sc=SC, yr=SNAP_YEAR)
    fig.savefig(p, dpi=300); plt.close(fig); print("saved", p)
    print(f"{SNAP_YEAR} {REGION} ratio by week (start DOY): sat / ecospace")
    for w in range(8, 16):
        print(f"  {WK_START[w]:3d}  {rs_ser[w]:.2f} / {rm_ser[w]:.2f}")


if __name__ == "__main__":
    main()
