import pandas as pd, numpy as np
raw=pd.read_csv(r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data\reference\zoop\Zoop_Perryetal_2021\MODIFIED\Zoopl_SofG_1996-2018_EN_sizeclassesseparate.csv")
print("size_class non-null rows overall:", raw.size_class.notna().sum(), raw.size_class.value_counts().to_dict())
eu=raw[raw.Order=="Euphausiacea"].copy()
eu["stage"]=eu.Size_class_or_life_stage.str.split().str[-1]
eu["B"]=eu["Abundance(m3)"]*eu["Average_biomass_per_individual(mg)"]
sh=eu.groupby("stage").B.sum()/eu.B.sum()
print("Share of euphausiid biomass by stage:", sh.round(3).to_dict())
ad=eu.stage.isin(["F","M","s3"])
pertow=eu.groupby("Index").apply(lambda d:(d.B*np.where(d.stage.isin(["F","M","s3"]),3,1)).sum()/d.B.sum())
print("Per-tow euphausiid multiplier if x3 applied to F/M/s3: median %.2f, IQR %.2f-%.2f"%(pertow.median(),pertow.quantile(.25),pertow.quantile(.75)))
