"""Annual crustacean zooplankton anomalies, observed vs SOGEM-LTL Ecospace, 2015-2018.

2015-2018 are the only survey years with sampling in every season (Table: zoop_sampling_summary_year_month.csv).
Method (Perry et al. 2021 / ICES), applied identically to observed and tow-matched modelled total crustacean biomass:
  season-year value = log10 of the arithmetic mean of tows; winter = December of the preceding year to February;
  seasonal anomaly  = value minus the 2015-2018 mean for that season (each year weighted equally);
  annual anomaly    = mean of the seasonal anomalies in that year.
Robustness: GAM-standardised annual observed index (gam_standardized_index.R) vs modelled annual mean biomass at the
sampled grid cells.
Outputs: eval/tables/ecospace_zoop_annual_anom_2015_2018_SC300.csv, eval/figs/ecospace_SC300_zoop_annual_anom_2015_2018.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ecospace_eval_5d_zoop_sampling_diagnostics as d5

Y0, Y1 = 2015, 2018
OFF = 1e-6
SEASONS = ["Winter", "Spring", "Summer", "Fall"]

m = d5.load_tows()
m = m[m["EWE-Total"].notna()].copy()
m["sYear"] = np.where(m.Month == 12, m.Year + 1, m.Year)          # December belongs to the following winter
m = m[m.sYear.between(Y0, Y1)]

sy = m.groupby(["sYear", "Season"]).agg(obs=("Total", "mean"), mod=("EWE-Total", "mean"), n=("Index", "size")).reset_index()
for c in ["obs", "mod"]:
    sy[c] = np.log10(sy[c] + OFF)
    sy[c] -= sy.groupby("Season")[c].transform("mean")
ann = sy.groupby("sYear").agg(obs=("obs", "mean"), mod=("mod", "mean"), n_seasons=("Season", "nunique"), n_tows=("n", "sum"))

r, p = stats.pearsonr(ann["obs"], ann["mod"])
gi = pd.read_csv(os.path.join(d5.TABS, "ecospace_zoop_gam_index_values_SC300.csv"))
g = gi[(gi.group == "Total") & (gi.season == "Annual") & gi.Year.between(Y0, Y1)].set_index("Year")
rg, pg = stats.pearsonr(g.o2, g.log10_model)

out = ann.rename(columns={"obs": "obs_anom_log10", "mod": "model_anom_log10"}).round(3)
out.index.name = "Year"
out.to_csv(os.path.join(d5.TABS, f"ecospace_zoop_annual_anom_{Y0}_{Y1}_{d5.cfg.ECOSPACE_SC}.csv"))
print(out.to_string())
print(f"Perry/ICES method, tow-matched: r = {r:.2f}, p = {p:.2f}, n = {len(ann)}")
print(f"GAM-standardised index vs model at sampled cells: r = {rg:.2f}, p = {pg:.2f}")

fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.axhline(0, color="#bbbbbb", lw=0.8)
ax.plot(ann.index, ann.obs, "-o", color="darkorange", lw=1.5, ms=6, label="Observed")
ax.plot(ann.index, ann["mod"], "-s", color="blue", lw=1.5, ms=5.5, label="SOGEM-LTL Ecospace")
ax.set_xticks(ann.index)
ax.set_ylabel("Annual anomaly (log$_{10}$)")
ax.set_title(f"r = {r:.2f}, p = {p:.2f}, n = {len(ann)} years", fontsize=9, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower left")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fp = os.path.join(d5.FIGS, f"ecospace_{d5.cfg.ECOSPACE_SC}_zoop_annual_anom_{Y0}_{Y1}.png")
fig.savefig(fp, dpi=300)
print("saved", fp)
