"""Is the date adjustment over-correcting? Compare spring obs-vs-model anomaly r under corrections of
increasing flexibility, and under a joint fit where year effects and the seasonal curve are estimated together
(so the curve cannot absorb between-year differences)."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ecospace_eval_5d_zoop_sampling_diagnostics as d5

OFF = 1e-6
m = d5.load_tows()


def zy(v, n):
    g = v[n >= 3]
    return (v - g.mean()) / g.std(ddof=1)


def resid(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def design(d, kind):
    t = ((d.dos - d.dos.mean()) / d.dos.std()).values
    one = np.ones(len(d))
    if kind == "month FE":
        return np.column_stack([one] + [(d.Month == mo).astype(float).values for mo in sorted(d.Month.unique())[1:]])
    deg = {"linear DOY (1 df)": 1, "quadratic DOY (2 df)": 2, "cubic DOY (3 df)": 3}[kind]
    return np.column_stack([t ** k for k in range(deg + 1)])


def joint_year_index(d, col, kind):
    """year FE + seasonal term fitted jointly; return year effects."""
    yrs = sorted(d.Year.unique())
    S = design(d, kind)[:, 1:]  # drop intercept
    X = np.column_stack([np.ones(len(d))] + [(d.Year == y).astype(float).values for y in yrs[1:]] + [S])
    b, *_ = np.linalg.lstsq(X, d[col].values, rcond=None)
    e = np.r_[0, b[1:len(yrs)]]
    return pd.Series(e - e.mean(), index=yrs)


rows = []
for season in ["Winter", "Spring", "Summer", "Fall"]:
    d = m[(m.Season == season) & m.Total.notna()].copy()
    d["lo"] = np.log10(d.Total + OFF); d["lm"] = np.log10(d["EWE-Total"] + OFF)
    n = d.groupby("Year").Index.nunique()
    # none (conventional, log both sides)
    a = d.groupby("Year")[["lo", "lm"]].mean()
    rows.append(dict(season=season, correction="none (log both)", method="residual-then-average",
                     r=stats.pearsonr(zy(a.lo, n), zy(a.lm, n))[0]))
    for kind in ["month FE", "linear DOY (1 df)", "quadratic DOY (2 df)", "cubic DOY (3 df)"]:
        X = design(d, kind)
        d["ro"] = resid(d.lo.values, X); d["rm"] = resid(d.lm.values, X)
        a = d.groupby("Year")[["ro", "rm"]].mean()
        rows.append(dict(season=season, correction=kind, method="residual-then-average",
                         r=stats.pearsonr(zy(a.ro, n), zy(a.rm, n))[0]))
        oi, mi = joint_year_index(d, "lo", kind), joint_year_index(d, "lm", kind)
        rows.append(dict(season=season, correction=kind, method="joint year+season fit",
                         r=stats.pearsonr(oi, mi)[0]))
res = pd.DataFrame(rows)
piv = res.pivot_table(index=["correction", "method"], columns="season", values="r", sort=False)[["Winter", "Spring", "Summer", "Fall"]]
print(piv.round(2).to_string())
res.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "overcorrection_check_results.csv"), index=False)
