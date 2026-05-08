"""
1d_nutrients_EcospaceASC_NEMO3day_toEcospaceASC_3day.py

Goal
----
Generate a 3-day time series of Ecospace ASC forcing maps (e.g., nutrient-loading proxy)
using the same core transforms as the 1c Ecosim nutrient-forcing workflow, but applied
to every grid cell.

Pipeline (mirrors 1c nutrient forcing)
--------------------------------------
For each 3-day timestep raster:
  - read input ASC (trimmed Ecospace grid)
  - apply combined mask (land + optional plume)
  - optionally identify "static" (low-variability) cells and treat them as neutral (=1)
Then, across the full time stack:
  - cap highs (CAP_HIGH)
  - optional log stretch (log(x + eps))
  - optional power transform (x ** PWR_EXP)
  - optional moving average smoothing over time (per-cell)
  - normalize to mean=1 using either:
      * global normalization (one scalar mean over all valid cells & times), OR
      * per-cell normalization (each cell divided by its own time-mean)
Finally:
  - write one ASC per timestep for Ecospace to load as a forcing time series

Notes
-----
- First run this on 1981 - 2018. Then do script 7 (dummy forcing)
- Masked cells (land/plume) are written as NODATA_VALUE (default -9999).
- If DO_FILTER_STATIC=True, static cells are written as 1.0 (neutral multiplier).
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# If you already rely on your helpers, you can swap these in.
# from helpers import buildSortableString


# -----------------------------
# CONFIG
# -----------------------------
@dataclass
class Cfg:
    # Years / time step conventions
    startyear: int = 1980
    endyear: int = 2018
    nsteps_per_year: int = 120  # 3-day blocks

    # Variable key + input directory convention (same as 1c uses)
    var_key: str = "varmixing_m"
    anomalies: bool = False  # if True, expects *_anom in filenames / directories

    # Paths (edit to your machine)
    forcing_root: str = r"C:/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/forcing"
    forcing_root_RDRS: str = r"C:/Users/Greig/Sync/PSF/EwE/Georgia Strait 2021/LTL_model/LTL_MODELS/RDRS forcings"

    # Input ASC directory override (if None, derived from var_key like 1c)
    asc_dir_override: Optional[str] = None

    # Output directory (will create if missing)
    out_dir: str = r"C:/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/forcing/ECOSPACE_in_3day_nutrld_fromASC_202601"
    out_prefix: str = "XLD_normalised_"  # output file prefix; final name: {prefix}{year}_{DOY}.asc

    # Mask files (trimmed 151x93 ASC format on Ecospace grid)
    land_mask_file: str = r"C:/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/basemap/ecospacedepthgrid.asc"
    plume_mask_file: str = r"C:/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/basemap/Fraser_Plume_Region.asc"
    use_plume_mask: bool = False

    # Output NODATA
    NODATA_VALUE: float = -9999.0

    # Static-cell filtering (tidally mixed / low-variability regions)
    DO_FILTER_STATIC: bool = True
    STD_THRESHOLD: float = 2.0   # σ threshold used to define "static"
    static_sample_year_step: int = 5   # every Nth year when building static mask
    static_sample_nsteps: int = 20     # number of early timesteps to sample per year

    # Nutrient-multiplier transforms (mirrors 1c)
    CAP_HIGH: float = 22.0
    USE_LOG_STRETCH: bool = False
    LOG_EPS: float = 1e-4
    USE_PWR_EXPNSN: bool = True
    PWR_EXP: float = 1.0

    # Optional smoothing over time (per cell). Use 1 to disable.
    MOVING_AVG_WIN: int = 1  # e.g., 3, 5, 9 ...

    # Normalization mode: "global" or "per_cell"
    NORM_MODE: str = "global"  # or "per_cell"

    # Filename day padding (1c uses 3 for most; adjust if needed)
    num_digits_doy: int = 2

    # Handling last timestep naming quirks (try nearby DOY values if file missing)
    doy_fallback_offsets: Tuple[int, ...] = (0, 1, 2, -1, -2)


# -----------------------------
# ASC IO helpers
# -----------------------------
def read_asc(path: str, n_header: int = 6) -> Tuple[List[str], np.ndarray]:
    """
    Read ESRI ASCII grid. Returns (header_lines, data_array).
    """
    with open(path, "r") as f:
        header = [next(f) for _ in range(n_header)]
        data = np.loadtxt(f)

    return header, data


def write_asc(path: str, header_lines: List[str], data: np.ndarray, fmt: str = "%.6f") -> None:
    """
    Write ESRI ASCII grid with provided header lines.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for line in header_lines:
            f.write(line)
        np.savetxt(f, data, fmt=fmt)


def pad_int(n: int, width: int) -> str:
    return str(n).zfill(width)


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# -----------------------------
# Mask + static filter helpers
# -----------------------------
def load_masks(cfg: Cfg) -> Tuple[np.ndarray, List[str]]:
    """
    Build combined mask (True = masked) and return template header from land_mask_file.
    land mask: uses ecospacedepthgrid convention where depth==0 indicates land.
    plume mask: Fraser_Plume_Region==1 indicates plume region (masked if enabled).
    """
    tmpl_header, land = read_asc(cfg.land_mask_file)
    _, plume = read_asc(cfg.plume_mask_file)

    land_mask = (land == 0)
    plume_mask = (plume == 1) if cfg.use_plume_mask else np.zeros_like(land_mask, dtype=bool)

    combined = land_mask | plume_mask
    return combined, tmpl_header


def getDataFrame_custom_like_1c(arr: np.ndarray) -> np.ndarray:
    """
    Mimic the 1c 'custom' reader behavior on an already-loaded array:
    - treat -9999 and 0 as invalid -> NaN
    - apply bottom-left corner fix (hard-coded indices as in 1c)
    """
    data = arr.astype(np.float32, copy=True)
    data[(data == -9999.0) | (data == 0.0)] = np.nan
    # bottom-left corner fix from 1c
    if data.shape[0] >= 141 and data.shape[1] >= 15:
        data[140:, :15] = np.nan
    return data


def build_static_varmask(cfg: Cfg, asc_dir: str, combined_mask: np.ndarray) -> np.ndarray:
    """
    Identify low-variability ("static") cells using a sampled subset of files:
      - every cfg.static_sample_year_step years
      - first cfg.static_sample_nsteps timesteps each sampled year

    Returns boolean mask (True = static cell).
    """
    sample_stack = []

    for year in range(cfg.startyear, cfg.endyear + 1, cfg.static_sample_year_step):
        leap = is_leap_year(year)
        for iday in range(1, min(cfg.nsteps_per_year, cfg.static_sample_nsteps) + 1):
            middle_day = (iday - 1) * 3 + 2
            if iday >= cfg.nsteps_per_year:
                middle_day += 2 if leap else 1

            fpath = resolve_input_asc(cfg, asc_dir, cfg.var_key, year, middle_day)
            if fpath is None or (not os.path.exists(fpath)):
                continue

            _, raw = read_asc(fpath)
            data = getDataFrame_custom_like_1c(raw)
            if data.shape != combined_mask.shape:
                raise ValueError(f"Mask/data shape mismatch: {data.shape} vs {combined_mask.shape}")
            data = np.ma.masked_array(data, mask=combined_mask).filled(np.nan)

            sample_stack.append(data)

    if not sample_stack:
        raise RuntimeError("Static mask build failed: no sample files found. Check asc_dir / filenames.")

    stack = np.stack(sample_stack, axis=0)  # (t, y, x)
    stdmap = np.nanstd(stack, axis=0)
    static_mask = stdmap < cfg.STD_THRESHOLD
    print(f"[static] mean σ={np.nanmean(stdmap):.4f}, median σ={np.nanmedian(stdmap):.4f}, "
          f"static frac={(np.nanmean(static_mask.astype(float))):.3f}")
    return static_mask


# -----------------------------
# Input file resolution
# -----------------------------
def infer_asc_dir(cfg: Cfg, var_key: str) -> str:
    """
    Match 1c conventions (extend as needed).
    """
    if cfg.asc_dir_override:
        return cfg.asc_dir_override

    # 1c conventions (common cases)
    if var_key == "RDRS_windstress10m":
        return os.path.join(cfg.forcing_root_RDRS, "Wind_RDRS", "Ecospace", "stress_")

    if var_key == "PAR-VarZ-VarK":
        return os.path.join(cfg.forcing_root, "ECOSPACE_in_3day_PAR3_Sal4m_1980-2018", var_key)

    # default
    return os.path.join(cfg.forcing_root, "ECOSPACE_in_3day_vars_1980-2018", var_key)


def resolve_input_asc(cfg: Cfg, asc_dir: str, var_key: str, year: int, middle_day: int) -> Optional[str]:
    """
    Try the filename patterns seen in 1c:
      - {var_key}_{year}_{DOY}.asc
      - {var_key}__{year}_{DOY}.asc
    plus a small DOY fallback window to handle last-step naming quirks.
    """
    for off in cfg.doy_fallback_offsets:
        doy = middle_day + off
        if doy < 1:
            continue
        doy_str = pad_int(doy, cfg.num_digits_doy)

        # pattern A
        fname_a = f"{var_key}_{year}_{doy_str}.asc"
        fpath_a = os.path.join(asc_dir, fname_a)
        if os.path.exists(fpath_a):
            return fpath_a

        # pattern B (double underscore)
        fname_b = f"{var_key}__{year}_{doy_str}.asc"
        fpath_b = os.path.join(asc_dir, fname_b)
        if os.path.exists(fpath_b):
            return fpath_b
        else:
            print(fname_b)

    return None


def build_time_index(cfg: Cfg, asc_dir: str, var_key: str) -> List[Tuple[int, int, str]]:
    """
    Build an ordered list of (year, middle_day, fpath) for all available files.
    """
    time_meta: List[Tuple[int, int, str]] = []
    for year in range(cfg.startyear, cfg.endyear + 1):
        leap = is_leap_year(year)
        for iday in range(1, cfg.nsteps_per_year + 1):
            middle_day = (iday - 1) * 3 + 2
            if iday >= cfg.nsteps_per_year:
                middle_day += 2 if leap else 1

            fpath = resolve_input_asc(cfg, asc_dir, var_key, year, middle_day)
            if fpath is None:
                print(f"[warn] missing {var_key} file for {year} DOY~{middle_day} in {asc_dir}")
                continue
            time_meta.append((year, middle_day, fpath))

    if not time_meta:
        raise RuntimeError("No input files found. Check asc_dir, var_key, year range, and filename pattern.")
    return time_meta


# -----------------------------
# Time smoothing (nan-safe rolling mean)
# -----------------------------
def rolling_nanmean(stack: np.ndarray, win: int) -> np.ndarray:
    """
    Centered rolling mean along axis=0 (time) for a 3D array (T, Y, X), nan-safe.
    Uses pad(mode='edge') so output length == input length when win is odd.
    """
    if win <= 1:
        return stack
    if win % 2 == 0:
        raise ValueError("MOVING_AVG_WIN should be odd for centered smoothing (e.g., 3,5,9).")

    pad = win // 2
    # nan-safe: track sums and counts separately
    valid = np.isfinite(stack)
    x = np.where(valid, stack, 0.0).astype(np.float64)
    c = valid.astype(np.float64)

    xpad = np.pad(x, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    cpad = np.pad(c, ((pad, pad), (0, 0), (0, 0)), mode="edge")

    sx = np.cumsum(xpad, axis=0)
    sc = np.cumsum(cpad, axis=0)

    sum_win = sx[win:] - sx[:-win]
    cnt_win = sc[win:] - sc[:-win]

    out = np.where(cnt_win > 0, sum_win / cnt_win, np.nan).astype(np.float32)
    return out


# -----------------------------
# Main processing
# -----------------------------
def main(cfg: Cfg) -> None:
    var_key = cfg.var_key + ("_anom" if cfg.anomalies else "")
    asc_dir = infer_asc_dir(cfg, var_key)
    print(f"[info] Input ASC dir: {asc_dir}")

    combined_mask, template_header = load_masks(cfg)

    # Build time index
    time_meta = build_time_index(cfg, asc_dir, var_key)
    print(f"[info] Found {len(time_meta)} rasters")

    # Load one file to confirm shape
    _, raw0 = read_asc(time_meta[0][2])
    data0 = getDataFrame_custom_like_1c(raw0)
    if data0.shape != combined_mask.shape:
        raise ValueError(f"Mask/data shape mismatch: {data0.shape} vs {combined_mask.shape}")
    nrows, ncols = data0.shape

    # Static filter mask (optional)
    static_mask = None
    if cfg.DO_FILTER_STATIC:
        static_mask = build_static_varmask(cfg, asc_dir, combined_mask)

    # Allocate full stack (float32 to keep memory reasonable)
    T = len(time_meta)
    stack = np.full((T, nrows, ncols), np.nan, dtype=np.float32)

    # Load all rasters
    for t, (year, doy, fpath) in enumerate(time_meta):
        _, raw = read_asc(fpath)
        data = getDataFrame_custom_like_1c(raw)

        data = np.where(combined_mask, np.nan, data)
        if static_mask is not None:
            # keep them as nan during transforms/normalization; we’ll set to 1.0 at the end
            data = np.where(static_mask, np.nan, data)

        stack[t, :, :] = data

        if (t + 1) % 200 == 0 or (t + 1) == T:
            print(f"[load] {t+1}/{T}")

    # Transform pipeline (mirrors 1c)
    stack = np.minimum(stack, cfg.CAP_HIGH).astype(np.float32)

    if cfg.USE_LOG_STRETCH:
        stack = np.log(stack + cfg.LOG_EPS).astype(np.float32)

    if cfg.USE_PWR_EXPNSN:
        stack = (stack ** cfg.PWR_EXP).astype(np.float32)

    if cfg.MOVING_AVG_WIN > 1:
        stack = rolling_nanmean(stack, cfg.MOVING_AVG_WIN)

    # Normalization
    if cfg.NORM_MODE.lower() == "global":
        mu = np.nanmean(stack)
        if not np.isfinite(mu) or mu == 0:
            raise RuntimeError("Global mean is invalid (nan/0). Check masks and inputs.")
        stack = (stack / mu).astype(np.float32)
        print(f"[norm] global mean={mu:.6f} -> normalized mean ~1")

    elif cfg.NORM_MODE.lower() == "per_cell":
        mu_map = np.nanmean(stack, axis=0)  # (Y, X)
        mu_map = np.where(np.isfinite(mu_map) & (mu_map != 0), mu_map, 1.0).astype(np.float32)
        stack = (stack / mu_map[None, :, :]).astype(np.float32)
        print("[norm] per-cell normalization applied (each cell mean ~1)")

    else:
        raise ValueError("NORM_MODE must be 'global' or 'per_cell'")

    # Finalize output rasters:
    # - static cells become neutral 1.0
    # - masked cells become NODATA
    out_stack = stack.copy()
    if static_mask is not None:
        out_stack[:, static_mask] = 1.0
    out_stack = np.where(combined_mask, cfg.NODATA_VALUE, out_stack)

    # Ensure NODATA line in header matches cfg.NODATA_VALUE (best-effort)
    header = template_header.copy()
    for i, line in enumerate(header):
        if line.lower().startswith("nodata_value"):
            header[i] = f"NODATA_value {cfg.NODATA_VALUE}\n"

    # Write outputs
    os.makedirs(cfg.out_dir, exist_ok=True)
    for t, (year, doy, _fpath) in enumerate(time_meta):
        doy_str = pad_int(doy, cfg.num_digits_doy)
        out_name = f"{cfg.out_prefix}{year}_{doy_str}.asc"
        out_path = os.path.join(cfg.out_dir, out_name)
        write_asc(out_path, header, out_stack[t, :, :], fmt="%.6f")

        if (t + 1) % 200 == 0 or (t + 1) == T:
            print(f"[write] {t+1}/{T}")

    print("[done] wrote Ecospace forcing rasters to:", cfg.out_dir)


if __name__ == "__main__":
    cfg = Cfg()
    # Example quick switches:
    # cfg.NORM_MODE = "per_cell"
    # cfg.USE_LOG_STRETCH = True
    # cfg.MOVING_AVG_WIN = 5
    main(cfg)
