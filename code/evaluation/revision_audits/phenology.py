import numpy as np, pandas as pd
from scipy import stats
D=r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data"
G=["ZC1-EUP","ZC2-AMP","ZC3-DEC","ZC4-CLG","ZC5-CSM"]
c09=pd.Series([94,78,81,82,87,76,75,87,99,76,81,78,77,55,86,86,67,87,77,104,66,70,92,86,81,62,88,100,90,97,104,103,98,88,86,77,88,104,77],index=range(1980,2019),name="c09")
sat=pd.Series([100,68,50,83,115,115,100,100,92,88,92,92,55,77],index=range(2003,2017),name="sat")
esim=pd.read_csv(f"{D}/evaluation/ecosim_bloom_timing_C09_SC300.csv").set_index("Year")["Day of Year"].rename("ecosim_bloom")
espc=pd.read_csv(f"{D}/evaluation/ecospace_bloom_timing_C09_SC300.csv").set_index("Year")["Day of Year"].rename("ecospace_bloom")
def rr(a,b):
    m=a.notna()&b.notna(); r,p=stats.pearsonr(a[m],b[m]); return f"{r:+.2f} (p={p:.2f}, n={m.sum()})"
def resid(y,doy):
    t=(doy-105)/30.; X=np.column_stack([np.ones_like(t),t,t**2,t**3]); b,*_=np.linalg.lstsq(X,y,rcond=None); return y-X@b
m=pd.read_csv(f"{D}/evaluation/Zooplankton_matched_to_model_out_SC300.csv")
rng=m["Tow_start_depth.m."].abs()-m["Tow_end_depth.m."].abs()
m=m[((rng/m["Bottom_depth.m."]>=0.7)|(m["Tow_start_depth.m."]>=150))&m.Year.between(2000,2018)]
sp=m[m.Season=="Spring"].copy(); sp["doy"]=pd.to_datetime(sp.Date).dt.dayofyear
sp["Total"]=sp[G].sum(1); sp["EWE-Total"]=sp[[f"EWE-{g}" for g in G]].sum(1)
# ecosim model at same tows
mod=pd.read_csv(f"{D}/evaluation/ecosim_SC300_onerun_B_dates_seasons.csv",parse_dates=["date"])
mod["ecosim_tot"]=mod[["5","6","7","8","9"]].sum(1)
sp["date"]=pd.to_datetime(sp.Date)
sp=pd.merge_asof(sp.sort_values("date"),mod[["date","ecosim_tot"]].sort_values("date"),on="date",direction="nearest")
sp["obs_r"]=resid(np.log10(sp.Total.values+1e-6),sp.doy.values)
sp["espc_r"]=resid(np.log10(sp["EWE-Total"].values+1e-6),sp.doy.values)
sp["esim_r"]=resid(np.log10(sp.ecosim_tot.values+1e-6),sp.doy.values)
y=sp.groupby("Year").agg(obs_r=("obs_r","mean"),espc_r=("espc_r","mean"),esim_r=("esim_r","mean"),samp_doy=("doy","mean"))
y=y.join([c09,sat,esim,espc])
print("Bloom-timing agreement, 2000-2018 (does the model get early/late springs right?)")
print("  Ecosim bloom vs C09:  ", rr(y.ecosim_bloom,y.c09), "  vs satellite:", rr(y.ecosim_bloom,y.sat))
print("  Ecospace bloom vs C09:", rr(y.ecospace_bloom,y.c09), "  vs satellite:", rr(y.ecospace_bloom,y.sat))
print("\nDate-adjusted spring zooplankton (Total) vs bloom day (negative r = early bloom -> more zoop)")
print("  OBS zoop vs C09 bloom:      ", rr(y.obs_r,y.c09))
print("  OBS zoop vs satellite bloom:", rr(y.obs_r,y.sat))
print("  Ecosim zoop vs own bloom:   ", rr(y.esim_r,y.ecosim_bloom))
print("  Ecospace zoop vs own bloom: ", rr(y.espc_r,y.ecospace_bloom))
print("\nIs cruise timing linked to bloom timing?")
print("  sampling day vs C09:", rr(y.samp_doy,y.c09), "  vs satellite:", rr(y.samp_doy,y.sat))
print("\nResidual SD: obs %.3f  ecosim %.3f  ecospace %.3f" % (y.obs_r.std(),y.esim_r.std(),y.espc_r.std()))
print(y.round(2).to_string())
