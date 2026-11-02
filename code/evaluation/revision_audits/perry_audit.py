"""Audit our zooplankton data preparation against Perry et al. (2021, PLoS ONE 16:e0245941).

Uses Perry's own per-tow table (S2 Table sheet 1 = MODIFIED/Perry2021_Supp2_Tab1.csv, dry mass g m-2 by
Perry group, with their daytime adult-euphausiid x3) and their published annual anomalies (S2 sheet 3).

Checks
  1. Tow selection: which matched tows are / are not in Perry's 760, and why.
  2. Per-tow biomass recomputed from the raw file (abundance x mass x tow range) vs Perry's per-tow values.
  3. Euphausiid net-avoidance factor: Perry vs raw, by day/night.
  4. Copepod grouping: our ZC4-CLG / ZC5-CSM vs Perry's large / medium / small calanoids.
  5. What goes into ZF1-ICT (Chordata) and 'misc'.
  6. Perry's published annual total-biomass anomaly vs our observed series and vs the model.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ecospace_eval_5d_zoop_sampling_diagnostics as d5

P = os.path.join(d5.DATA, "reference", "zoop", "Zoop_Perryetal_2021")
OUT = os.path.join(HERE, "perry_audit_results.txt")
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); lines.append(s)


# ------------------------------------------------------------------ load
raw = pd.read_csv(d5.RAW)
raw.columns = [c.replace("(", ".").replace(")", ".") for c in raw.columns]
raw["rng"] = raw["Tow_start_depth.m."].abs() - raw["Tow_end_depth.m."].abs()
raw["mgm3"] = raw["Abundance.m3."] * raw["Average_biomass_per_individual.mg."]
raw["dw_gm2"] = raw.mgm3 * raw.rng * 1e-3
raw["stage"] = raw.Size_class_or_life_stage.str.strip().str.split().str[-1]
perry = pd.read_csv(os.path.join(P, "MODIFIED", "Perry2021_Supp2_Tab1.csv"), encoding="utf-8-sig")
m = d5.load_tows()  # 716 matched, deep/complete tows 2000-2018

say("=" * 90, "\n1. TOW SELECTION")
say(f"Perry tows: {len(perry)} ({perry.yr.min()}-{perry.yr.max()}); regions: {perry.region.value_counts().to_dict()}")
say(f"Perry mesh (um): {perry['mesh.um'].value_counts().to_dict()}")
say(f"Perry nets: {perry['net.type'].value_counts().to_dict()}")
say(f"Matched tows (ours, 2000-2018, deep/complete filter): {len(m)}; regions: {m.Region.value_counts().to_dict()}")
inP = m.Index.isin(perry.key)
say(f"Matched tows also in Perry's set: {inP.sum()} / {len(m)}")
meta = raw.drop_duplicates("Index").set_index("Index")
notP = m.loc[~inP, "Index"]
nm = meta.loc[meta.index.intersection(notP)]
say("Matched tows NOT in Perry's set - region:", nm.Region.value_counts().to_dict())
say("   mesh:", nm["Mesh_size.micro m."].value_counts().to_dict() if "Mesh_size.micro m." in nm else "")
say("   net:", nm.Net_type.value_counts().to_dict())
say("   bottom depth (m) quantiles:", nm["Bottom_depth.m."].quantile([0, .25, .5, .75, 1]).round(0).tolist())
say("   by year:", nm.Year.value_counts().sort_index().to_dict())
p00 = perry[perry.yr >= 2000]
miss = p00[~p00.key.isin(m.Index)]
say(f"Perry tows 2000-2018 NOT in our matched set: {len(miss)} / {len(p00)}; in raw file at all: "
    f"{miss.key.isin(raw.Index).sum()}; region: {miss.region.value_counts().to_dict()}")
say("   by year:", miss.yr.value_counts().sort_index().to_dict())
mesh_col = [c for c in raw.columns if c.startswith("Mesh")][0]
say(f"Raw-file mesh sizes in matched tows: {meta.loc[meta.index.intersection(m.Index), mesh_col].value_counts().to_dict()}")

# ------------------------------------------------------------------ 2. per-tow biomass
say("=" * 90, "\n2. PER-TOW DRY BIOMASS: raw recomputation vs Perry (common tows)")
say("NB Perry's per-tow table (S2 sheet 1) is in mg m-3 (see its definitions sheet), not g m-2.")
tot_raw = raw.groupby("Index").mgm3.sum()
ch_raw = raw[raw.Phylum == "Chaetognatha"].groupby("Index").mgm3.sum()
j = perry.set_index("key")[["Total.Biomass", "Euphs", "Chaetognatha", "twilight"]].join(tot_raw.rename("raw_total"), how="inner")
j = j.join(ch_raw.rename("raw_chaet"))
rc = (j.Chaetognatha / j.raw_chaet)[j.raw_chaet > 0]
say(f"n common tows {len(j)}; chaetognaths Perry/raw (mg m-3): min {rc.min():.4f}, max {rc.max():.4f}  -> identical source")
rt = j["Total.Biomass"] / j.raw_total
say(f"Total Perry/raw: median {rt.median():.3f}, IQR {rt.quantile(.25):.3f}-{rt.quantile(.75):.3f} (excess = euphausiid x3)")

# ------------------------------------------------------------------ 3. euphausiids
say("=" * 90, "\n3. EUPHAUSIIDS")
eu = raw[raw.Order == "Euphausiacea"].copy()
say("Euphausiid stage codes and share of raw euphausiid dry biomass:",
    (eu.groupby("stage").mgm3.sum() / eu.mgm3.sum()).round(3).sort_values(ascending=False).head(8).to_dict())
eu_raw = eu.groupby("Index").mgm3.sum()
adult = eu[eu.stage.isin(["F", "M"])].groupby("Index").mgm3.sum()
ad_s3 = eu[eu.stage.isin(["F", "M", "s3"])].groupby("Index").mgm3.sum()
j = j.join(eu_raw.rename("eu_raw")).join(adult.rename("eu_FM")).join(ad_s3.rename("eu_FMs3"))
j = j.fillna({"eu_raw": 0, "eu_FM": 0, "eu_FMs3": 0})
day = (j.twilight == "Daylight").to_numpy()
cands = {"x3 daytime F+M": j.eu_raw + np.where(day, 2 * j.eu_FM, 0),
         "x3 daytime F+M+s3": j.eu_raw + np.where(day, 2 * j.eu_FMs3, 0),
         "x3 all F+M (day+night)": j.eu_raw + 2 * j.eu_FM}
for tw, idx in [("Daylight", day), ("Night", ~day)]:
    g = j[idx & (j.eu_raw > 0)]
    s = f"  {tw:9s} n={len(g):4d}  Perry/raw median {np.median(g.Euphs / g.eu_raw):.3f}"
    for k, v in cands.items():
        rr_ = g.Euphs / v[idx & (j.eu_raw > 0)]
        s += f" | Perry/({k}) med {np.median(rr_):.3f}, within 1%: {np.mean(abs(rr_ - 1) < .01):.0%}"
    say(s)
say(f"Euphausiid share of total (Perry, mg m-3, with x3): {perry.Euphs.sum() / perry['Total.Biomass'].sum():.2f}"
    f"  (paper: 26% of g m-2);  raw without x3: {eu.mgm3.sum() / raw.mgm3.sum():.2f}")
say(f"Matched tows: Daylight {(m.Twilight == 'Daylight').sum()}, Night {(m.Twilight == 'Night').sum()}, "
    f"other {m.Twilight.value_counts().drop(['Daylight', 'Night'], errors='ignore').to_dict()}")

# ------------------------------------------------------------------ 4. copepods
say("=" * 90, "\n4. CALANOID GROUPING (dry mass, all raw tows)")
cop = raw[raw.Order.isin(["Calanoida", "Poecilostomatoida", "Cyclopoida"])].copy()
cop["ours"] = np.select(
    [cop.Family.isin(["Calanidae", "Eucalanidae"]) | (cop.Order == "Poecilostomatoida"),
     (cop.Order == "Calanoida") | cop.Family.isin(["Oithonidae"])], ["ZC4-CLG", "ZC5-CSM"], "other")
tab = cop.groupby(["ours", "Genus"]).dw_gm2.sum().unstack(0).fillna(0)
tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index[:12]]
say((tab / raw.dw_gm2.sum() * 100).round(2).to_string())
say("  (% of all raw dry biomass)")
both = perry[["CalCops.larg", "CalCops.med", "CalCops.smal"]].sum() / perry["Total.Biomass"].sum() * 100
say("Perry shares (% total): ", both.round(1).to_dict())
ours = cop.groupby("ours").dw_gm2.sum() / raw.dw_gm2.sum() * 100
say("Our ZC4/ZC5 shares (% total, dry, no euph x3):", ours.round(1).to_dict())
cal = cop[cop.Genus == "Calanus"]
say("Calanus species in ZC4-CLG (dry share of ZC4):",
    (cal.groupby("Species").dw_gm2.sum() / cop[cop.ours == "ZC4-CLG"].dw_gm2.sum()).round(3).to_dict())
poe = cop[cop.Order == "Poecilostomatoida"]
say(f"Poecilostomatoida (Oncaea, Corycaeidae etc., ~0.5-1 mm) in ZC4-CLG: "
    f"{poe.dw_gm2.sum() / cop[cop.ours == 'ZC4-CLG'].dw_gm2.sum() * 100:.1f}% of ZC4 dry biomass")

# ------------------------------------------------------------------ 5. chordata / misc
say("=" * 90, "\n5. ZF1-ICT and misc")
ch = raw[(raw.Phylum == "Chordata") & (raw.Class != "Appendicularia")]
say("Chordata (non-appendicularian) by class, % of their dry biomass:",
    (ch.groupby("Class").dw_gm2.sum() / ch.dw_gm2.sum() * 100).round(1).to_dict())

# ------------------------------------------------------------------ 6. annual anomalies
say("=" * 90, "\n6. PERRY'S PUBLISHED ANNUAL ANOMALY (TotBiom, S2 sheet 3)")
s2 = pd.read_excel(os.path.join(P, "ORIGINAL", "pone.0245941.s002.xlsx"), sheet_name="3. Zooplankton anomalies")
s2 = s2.set_index("Year")
tb = s2["TotBiom"].dropna()
say("Perry TotBiom anomaly:", tb.round(2).to_dict())


pp = perry.copy()
pp["Season"] = pp.mon.map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                           6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"})
pp["gm2"] = pp["Total.Biomass"] * (pp["startz.m"].abs() - pp["endz.m"].abs()) / 1000
recipes = {}
for unit in ["mgm3", "gm2"]:
    for dec in ["same year", "next year"]:
        for summ in ["log of mean", "mean of logs"]:
            d = pp.assign(v=pp["Total.Biomass"] if unit == "mgm3" else pp.gm2,
                          Y=np.where((pp.mon == 12) & (dec == "next year"), pp.yr + 1, pp.yr))
            if summ == "log of mean":
                sm = d.groupby(["Y", "Season"]).v.mean().reset_index(); sm["l"] = np.log10(sm.v)
            else:
                sm = d.assign(l=np.log10(d.v)).groupby(["Y", "Season"]).l.mean().reset_index()
            sm["a"] = sm.l - sm.Season.map(sm[sm.Y.between(1996, 2010)].groupby("Season").l.mean())
            ok = sm.groupby("Y").Season.apply(lambda s: s.isin(["Spring", "Summer"]).any())
            a = sm.groupby("Y").a.mean()[ok]
            jj = pd.concat([a, tb], axis=1).dropna()
            recipes[(unit, dec, summ)] = (np.corrcoef(jj.iloc[:, 0], jj.iloc[:, 1])[0, 1], (jj.iloc[:, 0] - jj.iloc[:, 1]).abs().max())
say("Reproducing Perry's TotBiom from their per-tow table (r, max abs diff):")
for k, (r_, mx) in recipes.items():
    say(f"   {k[0]:5s} | Dec->{k[1]:9s} | {k[2]:12s}: r = {r_:.4f}, max diff {mx:.4f}")


def perry_anom(d, col, years=(1996, 2010)):
    """Perry/ICES: log10 of arithmetic seasonal mean; minus seasonal clim; annual = mean of seasons;
    a year counts only if it has spring or summer."""
    sm = d.groupby(["Year", "Season"])[col].mean().reset_index()
    sm["l"] = np.log10(sm[col] + d5.OFF)
    cy = sm[sm.Year.between(*years)]
    clim = cy.groupby("Season").l.mean() if len(cy) else sm.groupby("Season").l.mean()
    sm["a"] = sm.l - sm.Season.map(clim)
    ok = sm.groupby("Year").Season.apply(lambda s: s.isin(["Spring", "Summer"]).any())
    return sm.groupby("Year").a.mean()[ok]


crus = ["ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM"]
mt = m.copy()
mt["EWE-Total"] = mt[[f"EWE-{g}" for g in crus]].sum(axis=1)
obs_pa = perry_anom(mt, "Total", years=(2000, 2018))
mod_pa = perry_anom(mt, "EWE-Total", years=(2000, 2018))
gi = pd.read_csv(os.path.join(d5.TABS, "ecospace_zoop_gam_index_values_SC300.csv"))
ga = gi[(gi.group == "Total") & (gi.season == "Annual")].set_index("Year")
pt = pd.DataFrame({"Perry_TotBiom": tb, "ours_obs_PerryMethod": obs_pa, "model_towmatched_PerryMethod": mod_pa,
                   "ours_obs_GAMannual": ga.o2, "model_window_annual": ga.log10_model}).loc[2000:2018]


def rr(a, b):
    r = d5.r_ci(pt[a], pt[b]); return f"r = {r['r']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] n={r['n']}"


say("Perry TotBiom vs our obs crustaceans (Perry method, matched tows):", rr("Perry_TotBiom", "ours_obs_PerryMethod"))
say("Perry TotBiom vs our obs GAM annual index:                     ", rr("Perry_TotBiom", "ours_obs_GAMannual"))
say("Perry TotBiom vs model window annual index:                    ", rr("Perry_TotBiom", "model_window_annual"))
say("Perry TotBiom vs model tow-matched (Perry method):             ", rr("Perry_TotBiom", "model_towmatched_PerryMethod"))
say("Our obs (Perry method) vs model tow-matched (Perry method):     ", rr("ours_obs_PerryMethod", "model_towmatched_PerryMethod"))
pt.to_csv(os.path.join(HERE, "perry_annual_anomaly_comparison.csv"))
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
