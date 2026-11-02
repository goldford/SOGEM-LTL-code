import numpy as np, pandas as pd
from scipy import stats
D = r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data\evaluation"
G = ["ZC1-EUP","ZC2-AMP","ZC3-DEC","ZC4-CLG","ZC5-CSM"]
m = pd.read_csv(f"{D}/Zooplankton_matched_to_model_out_SC300.csv")
rng = m["Tow_start_depth.m."].abs()-m["Tow_end_depth.m."].abs()
m = m[((rng/m["Bottom_depth.m."]>=0.7)|(m["Tow_start_depth.m."]>=150)) & m.Year.between(2000,2018)].copy()
m["Total"]=m[G].sum(1); m["EWE-Total"]=m[[f"EWE-{g}" for g in G]].sum(1)
m["doy"]=pd.to_datetime(m.Date).dt.dayofyear
sp=m[m.Season=="Spring"].copy()
print("Spring tows by region:\n", pd.crosstab(sp.Year, sp.Region).to_string(), "\n")
def resid(y, X):
    b,*_=np.linalg.lstsq(X,y,rcond=None); return y-X@b
def design(d, region):
    t=(d.doy.values-105)/30.
    cols=[np.ones(len(d)),t,t**2,t**3]
    if region:
        for r in sorted(d.Region.unique())[1:]: cols.append((d.Region==r).astype(float).values)
    return np.column_stack(cols)
for region in (False, True):
    print("== DOY cubic", "+ region" if region else "only")
    for g in ["Total"]+G:
        s2=sp[sp[g].notna()]; X=design(s2,region)
        o=resid(np.log10(s2[g].values+1e-6),X); mo=resid(np.log10(s2[f"EWE-{g}"].values+1e-6),X)
        a=pd.DataFrame({"Year":s2.Year.values,"o":o,"m":mo}).groupby("Year").agg(o=("o","mean"),m=("m","mean"),n=("o","size"))
        r,p=stats.pearsonr(a.o,a.m); a3=a[a.n>=3]; r3,p3=stats.pearsonr(a3.o,a3.m)
        print(f"  {g:8s} r={r:+.2f} p={p:.3f} (n={len(a)})   years with >=3 tows: r={r3:+.2f} p={p3:.3f} (n={len(a3)})   model resid SD={a.m.std():.3f} obs resid SD={a.o.std():.3f}")
