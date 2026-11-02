"""
QU39 phytoplankton: how much of the obs-model agreement is seasonal?

Uses the same preparation as ecospace_eval_3b_QU39_vs_ecospace.py (per-sample class totals, log transform,
departures from the group annual mean scaled by the SD of the monthly means, then month-year means) and compares
  r_model        : observations vs Ecospace (reproduces Table 6 r)
  r_model_clim   : observations vs the Ecospace monthly climatology (seasonal cycle only)
  r_obs_clim     : observations vs their own monthly climatology (upper bound for a purely seasonal predictor)
  r_deseasoned   : month-year departures from each series' monthly climatology, obs vs model
Reported in manuscript Sect. 3.2.
Output: eval/tables/ecospace_<run>_QU39_seasonal_check.csv
"""
import os
import numpy as np
import pandas as pd

import ecospace_eval_config as cfg

RUN = cfg.ECOSPACE_SC
SRC = os.path.join(cfg.QU39_SSC_ECOSPACE_PF, f"QU39_joined_matchtoEcospace_{RUN}.csv")
OUT = os.path.join(os.path.normpath(cfg.EVALOUT_P), f"ecospace_{RUN}_QU39_seasonal_check.csv")

GROUPS = {
    "Diatoms": (["Bacillariophyceae"], "PP1-DIA"),
    "Nanophytoplankton": (["Choanoflagellatea", "Dictyochophyceae", "Cryptophyceae", "Metromonadea", "Chrysophyceae",
                           "Telonemea", "Chlorodendrophyceae", "Bicosoecophyceae", "Xanthophyceae",
                           "Coccolithophyceae", "Euglenophyceae", "Raphidophyceae"], "PP2-NAN"),
    "Picoplankton": (["Pyramimonadophyceae"], "PP3-PIC"),
    "Dinoflagellates": (["Dinophyceae"], "PZ2-DIN"),
}


def standardised_month_year(df, col):
    """Month-year means of the log value, as departures from the annual mean of monthly means, scaled by their SD."""
    my = df.groupby(["year", "month"])[col].mean()
    mm = my.groupby(level="month").mean()
    return (my - mm.mean()) / mm.std()


def monthly_clim(s):
    return s.groupby(level="month").transform("mean")


def main():
    df = pd.read_csv(SRC)
    df["t"] = pd.to_datetime(df["closest_ecospace_time"])
    df = df[df.t.dt.year >= 1980]
    rows = []
    for name, (classes, field) in GROUPS.items():
        d = df[df["class"].isin(classes)]
        d = d.groupby(["id_x", "t"]).agg(obs=("measurementValue", "sum"), mod=(field, "first")).reset_index()
        d["year"], d["month"] = d.t.dt.year, d.t.dt.month
        d["lobs"] = np.log(d.obs.clip(lower=1e-3))
        d["lmod"] = np.log(d["mod"].clip(lower=1e-3))
        o, m = standardised_month_year(d, "lobs"), standardised_month_year(d, "lmod")
        oc, mc = monthly_clim(o), monthly_clim(m)
        rows.append(dict(group=name, n_month_years=len(o),
                         r_model=o.corr(m), r_model_clim=o.corr(mc), r_obs_clim=o.corr(oc),
                         r_deseasoned=(o - oc).corr(m - mc)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    print(res.round(2).to_string(index=False))
    print("saved", OUT)


if __name__ == "__main__":
    main()
