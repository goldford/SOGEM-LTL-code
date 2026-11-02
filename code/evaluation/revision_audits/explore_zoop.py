"""Exploratory look at observed vs SOGEM-LTL Ecospace crustacean zooplankton (revision, Oct 2026).

Figures (eval/figs/explore_zoop_*.png) and a text summary (revision_audits/explore_zoop_results.txt):
  1 sampling   : tows by year x month, stations, tow depth range, day/night - the design behind the anomalies
  2 seasonal   : tow-level log10 biomass vs day of year, obs and model, by group (seasonal cycle shape)
  3 mismatch   : per-tow log10(obs/model) by year, month, station, day/night
  4 indices    : annual index series - Perry's published anomaly, our GAM obs index, model (window), z-scored
  5 groups     : GAM-standardised annual obs index vs model by group
  6 composition: group shares of crustacean biomass by year, obs vs model
Also: which factors explain the per-tow mismatch (drop-one R2), and the effect of applying Perry's euphausiid x3.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ecospace_eval_5d_zoop_sampling_diagnostics as d5

G = d5.G
LAB = {"ZC1-EUP": "Euphausiids", "ZC2-AMP": "Amphipods", "ZC3-DEC": "Decapod larvae", "ZC4-CLG": "Large calanoids",
       "ZC5-CSM": "Small/medium calanoids", "Total": "Total crustaceans"}
CO, CM, CP = "#d95f02", "#1f5fbf", "#555555"   # obs, model, Perry
OFF = 1e-4
FIG = d5.FIGS
out = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)


plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})

m = d5.load_tows()
m = m[m["EWE-Total"].notna()].copy()
for g in G + ["Total"]:
    m[f"lo_{g}"] = np.log10(m[g].fillna(0) + OFF)
    m[f"lm_{g}"] = np.log10(m[f"EWE-{g}"] + OFF)
    m[f"lr_{g}"] = m[f"lo_{g}"] - m[f"lm_{g}"]
m["period"] = np.where(m.Year <= 2007, "2000-07", "2008-18")
say(f"{len(m)} matched tows, {m.Year.min()}-{m.Year.max()}, {m.Station.nunique()} stations")

# ---------------------------------------------------------------- euphausiid x3 (Perry rule) per tow
raw = pd.read_csv(d5.RAW)
raw.columns = [c.replace("(", ".").replace(")", ".") for c in raw.columns]
eu = raw[raw.Order == "Euphausiacea"].copy()
eu["mgm3"] = eu["Abundance.m3."] * eu["Average_biomass_per_individual.mg."]
eu["stage"] = eu.Size_class_or_life_stage.str.strip().str.split().str[-1]
eu["adj"] = np.where((eu.Twilight == "Daylight") & eu.stage.isin(["F", "M", "s3"]), 3.0, 1.0)
fac = (eu.assign(a=eu.mgm3 * eu.adj).groupby("Index").a.sum() / eu.groupby("Index").mgm3.sum()).rename("eup_fac")
m = m.merge(fac, left_on="Index", right_index=True, how="left")
m["eup_fac"] = m.eup_fac.fillna(1.0)
m["ZC1-EUP_x3"] = m["ZC1-EUP"] * m.eup_fac
m["Total_x3"] = m.Total + m["ZC1-EUP_x3"].fillna(0) - m["ZC1-EUP"].fillna(0)
say(f"Euphausiid x3 (Perry rule: daytime F/M/s3): median per-tow factor on ZC1 {m.eup_fac.median():.2f} "
    f"(day {m.eup_fac[m.night == 0].median():.2f}, night {m.eup_fac[m.night == 1].median():.2f}); "
    f"total crustaceans x{np.median(m.Total_x3 / m.Total):.2f} (median)")
for g in G + ["Total"]:
    say(f"  obs/model ratio, median per tow, {LAB[g]:24s}: {10 ** np.nanmedian(m[f'lr_{g}']):6.2f}"
        + (f"   (with euph x3: {np.nanmedian((m['ZC1-EUP_x3'] if g == 'ZC1-EUP' else m.Total_x3) / m[f'EWE-{g}']):.2f})"
           if g in ("ZC1-EUP", "Total") else ""))

# ---------------------------------------------------------------- 1 sampling design
fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
ct = m.groupby(["Year", "Month"]).size().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
im = ax[0, 0].imshow(np.log10(ct.T.replace(0, np.nan)), aspect="auto", cmap="Blues", origin="lower",
                     extent=[ct.index.min() - .5, ct.index.max() + .5, .5, 12.5])
ax[0, 0].set(title="a  Tows per year and month (log10)", ylabel="Month", yticks=range(1, 13))
plt.colorbar(im, ax=ax[0, 0], shrink=.8)
top = m.Station.value_counts().index[:12]
st = m[m.Station.isin(top)].groupby(["Station", "Year"]).size().unstack(fill_value=0).loc[top]
ax[0, 1].imshow(np.log10(st.replace(0, np.nan)), aspect="auto", cmap="Greens",
                extent=[st.columns.min() - .5, st.columns.max() + .5, len(top) - .5, -.5])
ax[0, 1].set(title="b  Tows per year at the 12 most-sampled stations", yticks=range(len(top)), yticklabels=top)
yr = m.groupby("Year")
ax[1, 0].plot(yr.towrng.mean(), "o-", color=CP, ms=4, lw=1.2)
ax[1, 0].set(title="c  Mean tow depth range (m)", xlabel="Year")
ax[1, 1].plot(yr.night.mean() * 100, "o-", color=CP, ms=4, lw=1.2, label="% night tows")
ax[1, 1].plot((yr.Region.apply(lambda s: (s == "Northern Strait of Georgia").mean())) * 100, "s--", color="#999",
              ms=4, lw=1.2, label="% northern SoG")
ax[1, 1].set(title="d  Day/night and region mix", xlabel="Year", ylabel="%"); ax[1, 1].legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "explore_zoop_1_sampling.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 2 seasonal cycle
fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
for a, g in zip(ax.flat, ["Total"] + G):
    a.scatter(m.doy, m[f"lo_{g}"], s=4, color=CO, alpha=.25, lw=0)
    a.scatter(m.doy, m[f"lm_{g}"], s=4, color=CM, alpha=.25, lw=0)
    mo = m.groupby("Month")
    mid = mo.doy.median()
    a.plot(mid, mo[f"lo_{g}"].median(), "-o", color=CO, ms=3, lw=1.6, label="Observed (monthly median)")
    a.plot(mid, mo[f"lm_{g}"].median(), "-s", color=CM, ms=3, lw=1.6, label="Model (tow-matched)")
    a.set_title(LAB[g]); a.set_ylabel("log10 g C m$^{-2}$")
    lo = np.nanpercentile(np.r_[m[f"lo_{g}"], m[f"lm_{g}"]], 1); a.set_ylim(bottom=max(lo, -3))
for a in ax[1]: a.set_xlabel("Day of year")
ax[0, 0].legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "explore_zoop_2_seasonal_cycle.png"), dpi=200); plt.close(fig)
say("\nSeasonal cycle: month of maximum median biomass (obs / model)")
for g in ["Total"] + G:
    mo = m.groupby("Month")
    say(f"  {LAB[g]:24s} obs peak month {mo[f'lo_{g}'].median().idxmax():2d}, model {mo[f'lm_{g}'].median().idxmax():2d};"
        f" seasonal amplitude (max-min monthly median, log10) obs {np.ptp(mo[f'lo_{g}'].median()):.2f}, model {np.ptp(mo[f'lm_{g}'].median()):.2f}")

# ---------------------------------------------------------------- 3 mismatch
fig, ax = plt.subplots(2, 2, figsize=(10, 6.5))
yrs = sorted(m.Year.unique())
ax[0, 0].boxplot([m.loc[m.Year == y, "lr_Total"] for y in yrs], positions=yrs, widths=.6, showfliers=False,
                 medianprops=dict(color=CO))
ax[0, 0].axhline(0, color="#bbb", lw=.8)
ax[0, 0].set(title="a  log10(obs / model), total crustaceans, by year", xticks=yrs[::2], xticklabels=yrs[::2])
ax[0, 1].boxplot([m.loc[m.Month == k, "lr_Total"] for k in range(1, 13)], positions=range(1, 13), showfliers=False,
                 medianprops=dict(color=CO))
ax[0, 1].axhline(0, color="#bbb", lw=.8); ax[0, 1].set(title="b  ... by month", xlabel="Month")
ax[1, 0].boxplot([m.loc[m.Station == s, "lr_Total"] for s in top], showfliers=False, medianprops=dict(color=CO))
ax[1, 0].set_xticks(range(1, len(top) + 1)); ax[1, 0].set_xticklabels(top, rotation=60)
ax[1, 0].axhline(0, color="#bbb", lw=.8); ax[1, 0].set(title="c  ... by station (12 most sampled)")
for g, c in zip(G, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]):
    s = m.groupby("Year")[f"lr_{g}"].median()
    ax[1, 1].plot(s, "-o", ms=3, lw=1.2, color=c, label=LAB[g])
ax[1, 1].axhline(0, color="#bbb", lw=.8)
ax[1, 1].set(title="d  Median log10(obs / model) by year and group", xlabel="Year"); ax[1, 1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "explore_zoop_3_mismatch.png"), dpi=200); plt.close(fig)

# drop-one R2 for log ratio
import itertools
say("\nWhat explains per-tow log10(obs/model)? OLS with factors; R2 lost when each is dropped")


def X_of(d, terms):
    cols = [np.ones(len(d))]
    for t in terms:
        if t in ("Year", "Month", "Station", "net"):
            dm = pd.get_dummies(d[t].astype(str), drop_first=True).astype(float).to_numpy(); cols.append(dm)
        else:
            cols.append(d[[t]].to_numpy(float))
    return np.column_stack(cols)


def r2(d, y, terms):
    X = X_of(d, terms); b, *_ = np.linalg.lstsq(X, d[y].to_numpy(), rcond=None)
    e = d[y].to_numpy() - X @ b
    return 1 - e.var() / d[y].var()


terms = ["Year", "Month", "Station", "night", "lrng", "net"]
for g in ["Total"] + G:
    d = m.assign(lrng=np.log10(m.towrng)).dropna(subset=[f"lr_{g}"])
    d = d[np.isfinite(d[f"lr_{g}"])]
    full = r2(d, f"lr_{g}", terms)
    drops = {t: full - r2(d, f"lr_{g}", [u for u in terms if u != t]) for t in terms}
    say(f"  {LAB[g]:24s} full R2 {full:.2f}; unique: " + ", ".join(f"{k} {v:.2f}" for k, v in drops.items()))
for p, dd in m.groupby("period"):
    say(f"  period {p}: median obs/model (total) {10 ** dd.lr_Total.median():.2f}; obs median {10 ** dd.lo_Total.median():.2f},"
        f" model median {10 ** dd.lm_Total.median():.2f} g C m-2")

# ---------------------------------------------------------------- 4 annual indices
gi = pd.read_csv(os.path.join(d5.TABS, "ecospace_zoop_gam_index_values_SC300.csv"))
pa = pd.read_csv(os.path.join(HERE, "perry_annual_anomaly_comparison.csv"), index_col=0)
z = lambda s: (s - s.mean()) / s.std()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.axhline(0, color="#ddd", lw=.8)
ax.plot(z(pa.Perry_TotBiom), "-", color=CP, lw=1.4, label="Perry et al. 2021 published total-zooplankton anomaly")
ax.plot(z(pa.ours_obs_GAMannual), "-o", color=CO, ms=4, lw=1.4, label="Observed crustaceans, GAM-standardised index")
ax.plot(z(pa.model_window_annual), "-s", color=CM, ms=4, lw=1.4, label="SOGEM-LTL crustaceans, annual mean at sampled cells")
ax.set(ylabel="z-score", title="Annual crustacean zooplankton indices, 2000-2018")
ax.legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "explore_zoop_4_annual_indices.png"), dpi=200); plt.close(fig)
say("\nLinear trend 2008-2018 (z units / decade): " + ", ".join(
    f"{k} {stats.linregress(s.loc[2008:2018].index, z(s).loc[2008:2018]).slope * 10:+.2f}"
    for k, s in [("Perry", pa.Perry_TotBiom), ("obs GAM", pa.ours_obs_GAMannual), ("model", pa.model_window_annual)]))
say("Linear trend 2000-2007: " + ", ".join(
    f"{k} {stats.linregress(s.loc[2000:2007].index, z(s).loc[2000:2007]).slope * 10:+.2f}"
    for k, s in [("Perry", pa.Perry_TotBiom), ("obs GAM", pa.ours_obs_GAMannual), ("model", pa.model_window_annual)]))
# detrended agreement
for k in ["Perry_TotBiom", "ours_obs_GAMannual"]:
    j = pa[[k, "model_window_annual"]].dropna()
    rd = stats.pearsonr(*(signal.detrend(j[c].to_numpy()) for c in j))[0]
    dif = j.diff().dropna(); rdf = stats.pearsonr(dif.iloc[:, 0], dif.iloc[:, 1])[0]
    say(f"  {k} vs model window: raw r {j.corr().iloc[0, 1]:+.2f}, linear-detrended r {rd:+.2f}, first-difference r {rdf:+.2f}")

# ---------------------------------------------------------------- 5 groups annual
fig, ax = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True)
for a, g in zip(ax.flat, ["Total"] + G):
    s = gi[(gi.group == g) & (gi.season == "Annual")].set_index("Year")
    sd = s.o2.std()
    a.errorbar(s.index, z(s.o2), yerr=1.96 * s.se2 / sd, fmt="o", color=CO, ms=3, lw=.8, label="Observed (GAM index, 95% CI)")
    a.plot(s.index, z(s.log10_model), "-s", color=CM, ms=3, lw=1.2, label="Model (sampled cells)")
    r = d5.r_ci(s.o2, s.log10_model)
    a.set_title(f"{LAB[g]}  (r = {r['r']:+.2f})"); a.axhline(0, color="#ddd", lw=.8)
ax[0, 0].legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "explore_zoop_5_group_indices.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 6 composition
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
cols = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
for a, pre, ttl in [(ax[0], "", "Observed"), (ax[1], "EWE-", "Model (tow-matched)")]:
    sh = m.groupby("Year")[[pre + g for g in G]].mean()
    sh = sh.div(sh.sum(axis=1), axis=0)
    a.stackplot(sh.index, sh.T.values, colors=cols, labels=[LAB[g] for g in G], edgecolor="white", lw=.5)
    a.set(title=f"{ttl}: share of crustacean biomass", xlabel="Year", ylim=(0, 1))
ax[1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1, 1), loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "explore_zoop_6_composition.png"), dpi=200); plt.close(fig)
osh = m[G].mean() / m[G].mean().sum(); msh = m[[f"EWE-{g}" for g in G]].mean(); msh = msh / msh.sum()
say("\nMean composition obs vs model: " + ", ".join(f"{LAB[g]} {osh[g]:.0%}/{msh['EWE-' + g]:.0%}" for g in G))

# ---------------------------------------------------------------- euph x3 effect on interannual comparison
say("\nEffect of Perry's euphausiid x3 on interannual agreement (Perry-method annual anomaly, tow-matched):")


def pa_(d, col):
    sm = d.groupby(["Year", "Season"])[col].mean().reset_index(); sm["l"] = np.log10(sm[col] + OFF)
    sm["a"] = sm.l - sm.Season.map(sm.groupby("Season").l.mean())
    return sm.groupby("Year").a.mean()


for lbl, oc, mc in [("ZC1 as is", "ZC1-EUP", "EWE-ZC1-EUP"), ("ZC1 x3", "ZC1-EUP_x3", "EWE-ZC1-EUP"),
                    ("Total as is", "Total", "EWE-Total"), ("Total x3", "Total_x3", "EWE-Total")]:
    d = m.dropna(subset=[oc]); r = d5.r_ci(pa_(d, oc), pa_(d, mc))
    say(f"  {lbl:12s} r = {r['r']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
open(os.path.join(HERE, "explore_zoop_results.txt"), "w", encoding="utf-8").write("\n".join(out))
