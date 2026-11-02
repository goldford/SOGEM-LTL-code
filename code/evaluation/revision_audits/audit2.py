import numpy as np, pandas as pd
pd.set_option("display.width",250)
R=r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data\reference\zoop\Zoop_Perryetal_2021\MODIFIED"
raw=pd.read_csv(f"{R}/Zoopl_SofG_1996-2018_EN_sizeclassesseparate.csv")
eu=raw[raw.Order=="Euphausiacea"]
print("Euphausiid size/stage strings (tail token):", eu.Size_class_or_life_stage.str.split().str[-1].value_counts().head(12).to_dict())
t=raw.drop_duplicates("Index").copy()
t["src"]=t.Index.str[:3]
sp=t[(t.Month.between(3,5))&(t.Year>=2000)]
print("\nSPRING tows by year: index source prefix, net type, night share, mean bottom depth, mean tow range, n stations")
g=sp.groupby("Year")
out=pd.DataFrame({"n":g.size(),
 "src":g.src.agg(lambda s:"/".join(f"{k}{v}" for k,v in s.value_counts().items())),
 "net":g.Net_type.agg(lambda s:"/".join(f"{k.split()[0]}{v}" for k,v in s.value_counts().items())),
 "night%":g.Twilight.apply(lambda s:round(100*(s=="Night").mean())),
 "botm":g["Bottom_depth(m)"].mean().round(),
 "towrng":(g["Tow_start_depth(m)"].mean()-g["Tow_end_depth(m)"].mean()).round(),
 "nstn":g.Station.nunique(),
 "months":g.Month.agg(lambda s:"".join(str(m) for m in sorted(s.unique())))})
print(out.to_string())
print("\nStations with most spring tows 2000-2018:"); print(sp.Station.value_counts().head(12).to_string())
print("\nAll-season tows per station and years covered (top 12):")
a=t[t.Year>=2000].groupby("Station").agg(n=("Index","size"),years=("Year","nunique"),botm=("Bottom_depth(m)","mean"),region=("Region","first")).sort_values("n",ascending=False)
print(a.head(12).to_string())
