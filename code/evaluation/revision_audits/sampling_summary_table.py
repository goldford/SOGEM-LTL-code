"""Supplement table: number of matched zooplankton tows by year and month, with stations sampled per year
(deep/complete vertical tows, 2000-2018, as used in the evaluation)."""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ecospace_eval_5d_zoop_sampling_diagnostics as d5

m = d5.load_tows()
mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
t = m.groupby(["Year", "Month"]).size().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
t.columns = mon
t["Tows"] = t[mon].sum(axis=1)
t["Stations"] = m.groupby("Year").Station.nunique()
t["Night (%)"] = (m.groupby("Year").night.mean() * 100).round(0).astype(int)
tot = t[mon + ["Tows"]].sum()
tot["Stations"] = m.Station.nunique()
tot["Night (%)"] = int(round(m.night.mean() * 100))
t.loc["Total"] = tot
t.index.name = "Year"
t.to_csv(os.path.join(d5.TABS, "zoop_sampling_summary_year_month.csv"))
print(t.to_string())
