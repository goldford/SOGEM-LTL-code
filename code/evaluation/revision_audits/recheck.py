"""Faithful rebuild of Ecosim spring anomalies + fair version of check 3 for both models."""
import numpy as np, pandas as pd
from scipy import stats

D = r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data"
G = ["ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM"]
GMAP = {"ZC1-EUP": "5", "ZC2-AMP": "6", "ZC3-DEC": "7", "ZC4-CLG": "8", "ZC5-CSM": "9"}
Y0, Y1 = 2000, 2018


def ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def rr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    r, p = stats.pearsonr(x[m], y[m]); lo, hi = ci(r, m.sum())
    return f"{r:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3f} n={m.sum()}"


def partial(x, y, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return stats.pearsonr(rx, ry)[0]


def doy_resid(y, doy):
    t = (doy - 105) / 30.
    X = np.column_stack([np.ones_like(t), t, t**2, t**3])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


# ======================= ECOSIM: rebuild pipeline exactly =======================
obs = pd.read_csv(f"{D}/reference/zoop/Zoop_Perryetal_2021/MODIFIED/Zooplankton_B_C_gm2_EWEMODELGRP_Wide_NEMO3daymatch.csv")
obs["date"] = pd.to_datetime(obs[["Year", "Month", "Day"]])
obs["season"] = obs["Month"].map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                                  6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"})
mod = pd.read_csv(f"{D}/evaluation/ecosim_SC300_onerun_B_dates_seasons.csv", parse_dates=["date"])
mod = mod.rename(columns={v: f"EWE-{k}" for k, v in GMAP.items()})[["date"] + [f"EWE-{g}" for g in G]]
mt = pd.merge_asof(obs.sort_values("date"), mod.sort_values("date"), on="date",
                   direction="nearest", tolerance=pd.Timedelta(days=3)).dropna(subset=[f"EWE-{g}" for g in G])
mt["Total"] = mt[G].sum(axis=1); mt["EWE-Total"] = mt[[f"EWE-{g}" for g in G]].sum(axis=1)
mt["year"] = mt.date.dt.year; mt["doy"] = mt.date.dt.dayofyear
sp = mt[(mt.season == "Spring") & mt.year.between(Y0, Y1)].copy()
saved = pd.read_csv(f"{D}/evaluation/ecosim_zoop_anomalies_SC300_Spring_ZC.csv")

# full-season (all Mar-May timesteps) model, ln, z-scored over 2000-2018
mam = mod[mod.date.dt.month.isin([3, 4, 5]) & mod.date.dt.year.between(Y0, Y1)].copy()
mam["year"] = mam.date.dt.year
full = mam.groupby("year")[[f"EWE-{g}" for g in G]].mean()
full["EWE-Total"] = full.sum(axis=1)

print(f"ECOSIM spring tows (unfiltered, as in config): {len(sp)}")
print(f"{'group':8s} | rebuilt r (saved r) | r(DOY,obs) | partial | deseasoned obs vs deseasoned tow-model | deseasoned obs vs full-season model")
for g in ["Total"] + G:
    d = sp[sp[g].notna()].copy()
    d["o"] = np.log(d[g] + 1e-6); d["m"] = np.log(d[f"EWE-{g}"] + 1e-6)
    # pipeline: pooled-tow clim, z per tow, mean per year
    d["oz"] = (d.o - d.o.mean()) / d.o.std(); d["mz"] = (d.m - d.m.mean()) / d.m.std()
    d["or"] = doy_resid(d.o.values, d.doy.values); d["mr"] = doy_resid(d.m.values, d.doy.values)
    a = d.groupby("year").agg(oz=("oz", "mean"), mz=("mz", "mean"), doy=("doy", "mean"),
                              orr=("or", "mean"), mr=("mr", "mean"))
    a = a.join(np.log(full[f"EWE-{g}"]).rename("full"))
    s = saved[saved.group == g].set_index("year")
    r_reb = stats.pearsonr(a.oz, a.mz)[0]; r_sav = stats.pearsonr(s.obs_anom, s.model_anom)[0]
    print(f"{g:8s} | {r_reb:+.2f} ({r_sav:+.2f}) | {stats.pearsonr(a.doy, a.oz)[0]:+.2f} | "
          f"{partial(a.oz.values, a.mz.values, a.doy.values):+.2f} | {rr(a.orr.values, a.mr.values)} | {rr(a.orr.values, a.full.values)}")

# ======================= ECOSPACE: fair check 3 =======================
m = pd.read_csv(f"{D}/evaluation/Zooplankton_matched_to_model_out_SC300.csv")
rng = m["Tow_start_depth.m."].abs() - m["Tow_end_depth.m."].abs()
m = m[((rng / m["Bottom_depth.m."] >= 0.7) | (m["Tow_start_depth.m."] >= 150)) & m.Year.between(Y0, Y1)].copy()
m["Total"] = m[G].sum(axis=1)
m["doy"] = pd.to_datetime(m.Date).dt.dayofyear
sp = m[m.Season == "Spring"]
box = pd.read_csv(f"{D}/evaluation/ecospace_zoop_model_box_anom_Spring_SC300.csv")
print("\nECOSPACE: deseasoned obs (log10, DOY cubic) vs full-season central-SoG box model anomaly")
for g in ["Total", "ZC2-AMP"]:
    d = sp[sp[g].notna()].copy()
    d["or"] = doy_resid(np.log10(d[g].values + 1e-6), d.doy.values)
    a = d.groupby("Year")["or"].mean().rename("orr").reset_index()
    a = a.merge(box[box.group == g].rename(columns={"year": "Year"}), on="Year")
    print(f"  {g:8s} {rr(a.orr.values, a.anom.values)}")
