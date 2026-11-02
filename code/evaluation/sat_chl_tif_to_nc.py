"""
sat_chl_tif_to_nc.py

Stack the MODIS chlorophyll-a GeoTIFFs (Costa lab, UVic; "MODIS_Chla for Villy")
into one NetCDF per set.

Weekly sets hold 46 files per year (YY_chl_WW.tif or YYYY_chl_WW.tif, WW = 1..46),
assumed to be 8-day composites starting on DOY 1, 9, 17, ... (standard MODIS
8-day binning). Time is stored as the composite start date; 'doy_mid' is the
composite midpoint day of year.

Values are kept as float32 (chl-a, mg m-3); non-finite and <= 0 values become NaN.
Coordinates are pixel centres taken from each file's GeoTIFF tie point and pixel scale.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import xarray as xr
import tifffile

SRC = r"C:\Users\Greig\Sync\6. SSMSP Model\Model Greig\Data\28. Phytoplankton\Satellite Chl Costa 13 yrs 2021\MODIS_Chla for Villy\WEEK_TIF_NICOLE_03_16\WEEK_TIF_NICOLE_03_16"
OUT_P = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "SOGEM-LTL-data", "data", "reference", "satellite_chl"))
SETS = {"interp": "Interpolated", "uninterp": "Uninterpolated"}
PAT = re.compile(r"^(\d{2}|\d{4})_chl_(\d{1,2})\.tif$")


def read_tif(path):
    with tifffile.TiffFile(path) as t:
        pg = t.pages[0]
        a = pg.asarray().astype("float32")
        tags = {k.name: k.value for k in pg.tags.values()}
    sx, sy = tags["ModelPixelScaleTag"][:2]
    x0, y0 = tags["ModelTiepointTag"][3:5]
    nd = tags.get("GDAL_NODATA")
    if nd is not None:
        a[a == float(nd)] = np.nan
    a[~np.isfinite(a) | (a <= 0) | (a > 1e6)] = np.nan
    ny, nx = a.shape
    lon = x0 + sx * (np.arange(nx) + 0.5)
    lat = y0 - sy * (np.arange(ny) + 0.5)
    return a, lat, lon


def convert(key):
    d = os.path.join(SRC, SETS[key])
    files = []
    for f in os.listdir(d):
        m = PAT.match(f)
        if m:
            yr = int(m.group(1)); yr = yr + 2000 if yr < 100 else yr
            files.append((yr, int(m.group(2)), f))
    files.sort()
    print(f"{key}: {len(files)} files, {files[0][0]}-{files[-1][0]}")
    arrs, times, mids = [], [], []
    lat = lon = None
    for yr, wk, f in files:
        a, la, lo = read_tif(os.path.join(d, f))
        if lat is None:
            lat, lon = la, lo
        elif a.shape != (lat.size, lon.size):
            raise ValueError(f"grid mismatch in {f}: {a.shape}")
        start = pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=8 * (wk - 1))
        arrs.append(a); times.append(start); mids.append(min(8 * (wk - 1) + 4.5, 365.0))
    ds = xr.Dataset(
        {"chl": (("time", "lat", "lon"), np.stack(arrs))},
        coords={"time": times, "lat": lat, "lon": lon, "doy_mid": ("time", np.array(mids, dtype="float32"))},
        attrs={"source": "MODIS-Aqua chl-a 8-day composites, Costa lab (UVic), 'MODIS_Chla for Villy' WEEK_TIF_NICOLE_03_16",
               "set": SETS[key], "created_by": "sat_chl_tif_to_nc.py"},
    )
    ds["chl"].attrs = {"units": "mg m-3", "long_name": "chlorophyll-a concentration"}
    os.makedirs(OUT_P, exist_ok=True)
    out = os.path.join(OUT_P, f"modis_chl_8day_{key}_{files[0][0]}_{files[-1][0]}.nc")
    ds.to_netcdf(out, encoding={"chl": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    cov = np.isfinite(ds.chl).mean(("lat", "lon")).values
    print(f"saved {out}; mean finite fraction per composite {cov.mean():.3f} (min {cov.min():.3f}, max {cov.max():.3f})")
    # missing weeks per year
    per = pd.Series(1, index=pd.DatetimeIndex(times)).groupby(lambda t: t.year).sum()
    print("composites per year:", per[per != 46].to_dict() or "all 46")


if __name__ == "__main__":
    for k in (sys.argv[1:] or SETS):
        convert(k)
