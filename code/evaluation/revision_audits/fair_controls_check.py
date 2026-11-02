"""Same location/date controls on both obs and tow-matched model; obs-only controls (tow range, night, net) on obs."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ecospace_eval_5d_zoop_sampling_diagnostics as d5
m = d5.load_tows(); OFF = 1e-6
G = ["Total", "ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM"]
rows = []
for grp in G:
    for s in ["Winter", "Spring", "Summer", "Fall"]:
        d = m[(m.Season == s) & m[grp].notna() & m[f"EWE-{grp}"].notna()].copy()
        d["lo"] = np.log10(d[grp] + OFF); d["lm"] = np.log10(d[f"EWE-{grp}"] + OFF)
        for lab, to, tm in [("DOY both", ["doy"], ["doy"]),
                            ("DOY+station both", ["doy", "station"], ["doy", "station"]),
                            ("DOY+station both; obs also towrng+night+net", ["doy", "station", "towrng", "night", "net"], ["doy", "station"]),
                            ("DOY both; obs also towrng", ["doy", "towrng"], ["doy"])]:
            oi = d5.year_index(d, "lo", to); mi = d5.year_index(d, "lm", tm)
            rc = d5.r_ci(oi, mi)
            rows.append(dict(group=grp, season=s, controls=lab, r=rc["r"], lo=rc["lo"], hi=rc["hi"], p=rc["p"], n=rc["n"]))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fair_controls_results.csv"), index=False)
print(res[res.group == "Total"].pivot_table(index="controls", columns="season", values="r", sort=False)[["Winter","Spring","Summer","Fall"]].round(2).to_string())
print()
x = res[res.controls == "DOY+station both; obs also towrng+night+net"]
print("Full fair controls, by group (r; * p<0.05):")
x = x.assign(v=x.r.round(2).astype(str) + np.where(x.p < 0.05, "*", ""))
print(x.pivot(index="group", columns="season", values="v")[["Winter","Spring","Summer","Fall"]].to_string())
