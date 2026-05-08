from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt


# --- EDIT THESE TWO TO MATCH YOUR RUN ---
OUT_DIR = Path(r"C:/Users/Greig/Documents/GitHub/Ecosystem-Model-Data-Framework/data/forcing/ECOSPACE_in_3day_nutrld_fromASC_202601")
PREFIX  = "XLD_normalised_"   # same as cfg.out_prefix
# ---------------------------------------


def read_esri_asc(path: Path, n_header: int = 6):
    """Return (header_lines:list[str], data:2D np.ndarray, nodata_value:float)."""
    header = []
    nodata = None
    with path.open("r") as f:
        for _ in range(n_header):
            line = f.readline()
            header.append(line)
            if "nodata" in line.lower():
                # e.g., "NODATA_value -9999"
                try:
                    nodata = float(line.split()[-1])
                except Exception:
                    nodata = None

    data = np.loadtxt(path, skiprows=n_header)
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    return header, data, nodata


def parse_year_doy(fname: str):
    """
    Expect: {PREFIX}{year}_{doy}.asc
    e.g., XLD_normalised_1980_02.asc
    """
    m = re.match(rf"^{re.escape(PREFIX)}(\d{{4}})_(\d+)\.asc$", fname)
    if not m:
        return None
    year = int(m.group(1))
    doy  = int(m.group(2))
    return year, doy


def doy_to_date(year: int, doy: int) -> datetime:
    # DOY=1 => Jan 1
    return datetime(year, 1, 1) + timedelta(days=doy - 1)


def main():
    files = []
    for p in OUT_DIR.glob(f"{PREFIX}*.asc"):
        parsed = parse_year_doy(p.name)
        if parsed is None:
            continue
        year, doy = parsed
        files.append((year, doy, p))

    if not files:
        raise RuntimeError(f"No files found in {OUT_DIR} matching prefix '{PREFIX}'")

    files.sort(key=lambda x: (x[0], x[1]))

    dates = []
    means = []

    for i, (year, doy, path) in enumerate(files, start=1):
        _, data, _ = read_esri_asc(path)
        m = np.nanmean(data)  # spatial mean excluding NaNs
        dates.append(doy_to_date(year, doy))
        means.append(m)

        if i % 300 == 0 or i == len(files):
            print(f"read {i}/{len(files)}")

    dates = np.array(dates)
    means = np.array(means)

    # --- Plot: full time series ---
    plt.figure()
    plt.plot(dates, means)
    plt.title("Spatial mean per 3-day timestep")
    plt.xlabel("Date")
    plt.ylabel("Spatial mean (NaNs excluded)")
    plt.tight_layout()

    # --- Plot: seasonal climatology (by DOY block) ---
    # Group by day-of-year (integer)
    doys = np.array([d.timetuple().tm_yday for d in dates])
    uniq = np.unique(doys)

    clim_mean = np.array([np.nanmean(means[doys == u]) for u in uniq])
    clim_p10  = np.array([np.nanpercentile(means[doys == u], 10) for u in uniq])
    clim_p90  = np.array([np.nanpercentile(means[doys == u], 90) for u in uniq])

    plt.figure()
    plt.plot(uniq, clim_mean, label="mean")
    plt.fill_between(uniq, clim_p10, clim_p90, alpha=0.2, label="10–90%")
    plt.title("Seasonal climatology of spatial mean (across years)")
    plt.xlabel("Day of year")
    plt.ylabel("Spatial mean (NaNs excluded)")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()