import numpy as np, pandas as pd
from scipy import stats
D=r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data\evaluation"
G=["ZC1-EUP","ZC2-AMP","ZC3-DEC","ZC4-CLG","ZC5-CSM"]
m=pd.read_csv(f"{D}/Zooplankton_matched_to_model_out_SC300.csv")
rng=m["Tow_start_depth.m."].abs()-m["Tow_end_depth.m."].abs()
m=m[((rng/m["Bottom_depth.m."]>=0.7)|(m["Tow_start_depth.m."]>=150))&m.Year.between(2000,2018)].copy()
m["Total"]=m[G].sum(1); m["EWE-Total"]=m[[f"EWE-{g}" for g in G]].sum(1)
doy=pd.to_datetime(m.Date).dt.dayofyear.values.astype(float); m["dos"]=np.where(m.Month==12,doy-365,doy)
def res(y,x):
    t=(x-x.mean())/x.std(); X=np.column_stack([t**k for k in range(4)]); b,*_=np.linalg.lstsq(X,y,rcond=None); return y-X@b
print("Tow-level log10 Total: r raw vs r after removing within-season DOY curve")
for s in ["Winter","Spring","Summer","Fall"]:
    d=m[m.Season==s]; o=np.log10(d.Total+1e-6).values; mo=np.log10(d["EWE-Total"]+1e-6).values
    r0=stats.pearsonr(o,mo)[0]; r1=stats.pearsonr(res(o,d.dos.values),res(mo,d.dos.values))[0]
    print(f"  {s:7s} n={len(d):3d}  raw r={r0:+.2f}  DOY-removed r={r1:+.2f}")
o=np.log10(m.Total+1e-6).values; mo=np.log10(m["EWE-Total"]+1e-6).values
print(f"  All     n={len(m)}  raw r={stats.pearsonr(o,mo)[0]:+.2f}")
