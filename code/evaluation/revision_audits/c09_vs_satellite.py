"""Skill of the C09 1D bloom-timing model (Collins et al. 2009; Allen & Wolfe 2013) against satellite-derived bloom
dates (Suchy et al. 2022), 2003-2016, using the same statistics reported for SOGEM-LTL (manuscript Table 5).

Both series are copied from load_observation_bloom_dfs() in ecosim_eval_4_bloom_timing.py.
Categories: satellite labels as published; C09 classified with the Suchy et al. (2022) thresholds
(early <= day 68, late >= day 108).
"""
import numpy as np
from scipy import stats

SAT_YEARS = list(range(2003, 2017))
SAT_DOY = np.array([100, 68, 50, 83, 115, 115, 100, 100, 92, 88, 92, 92, 55, 77])
SAT_CAT = ["avg", "avg", "early", "avg", "late", "late", "avg", "avg", "avg", "avg", "avg", "avg", "early", "avg"]
C09_DOY_1980_2018 = [94, 78, 81, 82, 87, 76, 75, 87, 99, 76, 81, 78, 77, 55, 86, 86, 67, 87, 77, 104,
                     66, 70, 92, 86, 81, 62, 88, 100, 90, 97, 104, 103, 98, 88, 86, 77, 88, 104, 77]


def category(doy):
    return "early" if doy <= 68 else ("late" if doy >= 108 else "avg")


c09 = np.array(C09_DOY_1980_2018[SAT_YEARS[0] - 1980: SAT_YEARS[-1] - 1980 + 1])
diff = c09 - SAT_DOY
r, p = stats.pearsonr(c09, SAT_DOY)
wss = 1 - np.sum(diff ** 2) / np.sum((np.abs(c09 - SAT_DOY.mean()) + np.abs(SAT_DOY - SAT_DOY.mean())) ** 2)
matches = sum(category(c) == s for c, s in zip(c09, SAT_CAT))

print(f"C09 vs satellite, {SAT_YEARS[0]}-{SAT_YEARS[-1]} (n = {len(c09)})")
print(f"r = {r:.2f} (p = {p:.3f}), MAE = {np.abs(diff).mean():.1f} d, bias = {diff.mean():.1f} d, "
      f"RMSE = {np.sqrt((diff ** 2).mean()):.1f} d, WSS = {wss:.2f}, category matches = {matches}/{len(c09)}")
