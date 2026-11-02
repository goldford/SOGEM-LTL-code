"""Check whether spring zooplankton anomaly skill is driven by sampling dates."""
import numpy as np
import pandas as pd
from scipy import stats

D = r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data\evaluation"
GROUPS = ["ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM"]
Y0, Y1 = 2000, 2018


def r_p(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan, np.nan, int(m.sum())
    r, p = stats.pearsonr(x[m], y[m])
    return r, p, int(m.sum())


def partial_r(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return stats.pearsonr(rx, ry)[0]


# ---------------- ECOSPACE: rebuild spring anomalies ----------------
m = pd.read_csv(f"{D}/Zooplankton_matched_to_model_out_SC300.csv")
print("Season x Month (obs file):")
print(pd.crosstab(m.Season, m.Month), "\n")
rng = m["Tow_start_depth.m."].abs() - m["Tow_end_depth.m."].abs()
keep = (rng / m["Bottom_depth.m."] >= 0.7) | (m["Tow_start_depth.m."] >= 150)
m = m[keep & m.Year.between(Y0, Y1)].copy()
m["Total"] = m[GROUPS].sum(axis=1)
m["EWE-Total"] = m[[f"EWE-{g}" for g in GROUPS]].sum(axis=1)
m["doy"] = pd.to_datetime(m["Date"]).dt.dayofyear
sp = m[m.Season == "Spring"].copy()
print(f"Spring tows {Y0}-{Y1}: {len(sp)}")

# Within-season date dependence at tow level
print("\nTow-level: correlation of biomass with day-of-year within spring")
for g in ["Total"] + GROUPS:
    ro = r_p(sp.doy.values, np.log10(sp[g].values + 1e-6))
    rm = r_p(sp.doy.values, np.log10(sp[f"EWE-{g}"].values + 1e-6))
    print(f"  {g:8s} obs r={ro[0]:+.2f} (p={ro[1]:.3f})   model r={rm[0]:+.2f} (p={rm[1]:.3f})")

yr = sp.groupby("Year").agg(n=("Index", "nunique"), doy=("doy", "mean"),
                            doy_sd=("doy", "std"), dmin=("doy", "min"), dmax=("doy", "max"))
rows = []
for g in ["Total"] + GROUPS:
    a = sp.groupby("Year").apply(lambda d: pd.Series({
        "obs": np.mean(np.log10(d[g] + 1e-6)),
        "mod": np.mean(d[f"EWE-{g}"]),
    })).join(yr)
    good = a[a.n >= 3]
    a["obs_anom"] = (a.obs - good.obs.mean()) / good.obs.std(ddof=1)
    a["mod_anom"] = (a["mod"] - good["mod"].mean()) / good["mod"].std(ddof=1)
    a["group"] = g
    rows.append(a.reset_index())
es = pd.concat(rows)

print("\nPer-year spring sampling (Ecospace matched set):")
print(yr.round(1).to_string())

# model-only box anomaly (full season, central SoG box, 1980-2018)
box = pd.read_csv(f"{D}/ecospace_zoop_model_box_anom_Spring_SC300.csv")
box = box[box.year.between(Y0, Y1)].rename(columns={"year": "Year", "anom": "box_anom"})

print("\nECOSPACE spring (z-scores; obs=mean_of_logs, model=mean_raw, year_equal clim)")
print(f"{'group':8s} {'r(obs,mod)':>10s} {'r(DOY,obs)':>11s} {'r(DOY,mod)':>11s} {'partial':>8s} "
      f"{'r(obs,box)':>11s} {'r(mod,box)':>11s}")
for g, d in es.groupby("group", sort=False):
    o, mo, doy = d.obs_anom.values, d.mod_anom.values, d.doy.values
    r1 = r_p(o, mo); r2 = r_p(doy, o); r3 = r_p(doy, mo)
    pr = partial_r(o, mo, doy)
    b = d.merge(box[box.group == g], on="Year")
    rb = r_p(b.obs_anom.values, b.box_anom.values) if len(b) else (np.nan, np.nan, 0)
    rmb = r_p(b.mod_anom.values, b.box_anom.values) if len(b) else (np.nan, np.nan, 0)
    print(f"{g:8s} {r1[0]:+.2f} (n={r1[2]}) {r2[0]:+.2f} p={r2[1]:.2f} {r3[0]:+.2f} p={r3[1]:.2f} "
          f"{pr:+.2f}   " + (f"{rb[0]:+.2f} p={rb[1]:.3f} {rmb[0]:+.2f}" if len(b) else "   (no box series)"))

# ---------------- ECOSIM: full-season spring means vs obs ----------------
print("\nECOSIM spring")
ob = pd.read_csv(f"{D}/ecosim_zoop_anomalies_SC300_Spring_ZC.csv")
ecs = pd.read_csv(f"{D}/ecosim_SC300_onerun_B_dates_seasons.csv", parse_dates=["date"])
print("Ecosim model file season labels by month:")
print(pd.crosstab(ecs.date.dt.month, ecs.season).to_string(), "\n")
gmap = {"ZC1-EUP": "5", "ZC2-AMP": "6", "ZC3-DEC": "7", "ZC4-CLG": "8", "ZC5-CSM": "9"}
ecs["Year"] = ecs.date.dt.year
mam = ecs[ecs.date.dt.month.isin([3, 4, 5]) & ecs.Year.between(Y0, Y1)]
full = mam.groupby("Year")[list(gmap.values())].mean().rename(columns={v: k for k, v in gmap.items()})
full["Total"] = full[GROUPS].sum(axis=1)
full = np.log(full + 1e-6)
fz = (full - full.mean()) / full.std(ddof=1)
print(f"{'group':8s} {'r(obs,tow-matched mod)':>23s} {'r(obs,full-MAM mod)':>21s} {'r(DOY,obs)':>11s} {'partial':>8s}")
for g, d in ob.groupby("group", sort=False):
    d = d.rename(columns={"year": "Year"}).merge(fz[[g]].rename(columns={g: "full"}), on="Year")
    d = d.merge(yr[["doy"]], left_on="Year", right_index=True, how="left")
    r1 = r_p(d.obs_anom.values, d.model_anom.values)
    r2 = r_p(d.obs_anom.values, d.full.values)
    r3 = r_p(d.doy.values, d.obs_anom.values)
    pr = partial_r(d.obs_anom.values, d.model_anom.values, d.doy.values)
    print(f"{g:8s} {r1[0]:+.2f} (n={r1[2]})            {r2[0]:+.2f} p={r2[1]:.3f}      "
          f"{r3[0]:+.2f} p={r3[1]:.2f} {pr:+.2f}")
