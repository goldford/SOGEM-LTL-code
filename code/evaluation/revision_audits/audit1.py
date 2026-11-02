import numpy as np, pandas as pd
R=r"C:\Users\Greig\Documents\GitHub\SOGEM-LTL-data\data\reference\zoop\Zoop_Perryetal_2021\MODIFIED"
raw=pd.read_csv(f"{R}/Zoopl_SofG_1996-2018_EN_sizeclassesseparate.csv")
print("rows",len(raw),"tows",raw.Index.nunique())
eu=raw[raw.Order.str.lower()=="euphausiacea"]
print("Euphausiid Order spellings:",eu.Order.unique()," size_class values:",eu.size_class.value_counts(dropna=False).to_dict())
print("Twilight by tow:",raw.drop_duplicates("Index").Twilight.value_counts().to_dict())
print("Net types:",raw.drop_duplicates("Index").Net_type.value_counts().to_dict())
print("Mesh:",raw.drop_duplicates("Index")["Mesh_size(micro m)"].value_counts().to_dict())
w=pd.read_csv(f"{R}/Zooplankton_B_C_gm2_EWEMODELGRP_Wide.csv")
G=["ZC1-EUP","ZC2-AMP","ZC3-DEC","ZC4-CLG","ZC5-CSM","ZS2-CTH","ZS3-CHA","ZS4-LAR"]
print("\nWide file tows:",len(w)); print("NA (absent) fraction per group:"); print(w[G].isna().mean().round(3).to_string())
print("exact zeros:", (w[G]==0).sum().to_dict())
# check euphausiid x3 actually applied: compare wide ZC1 with recomputation
eu=raw[raw.Order=="Euphausiacea"].copy()
eu["rng"]=eu["Tow_start_depth(m)"].abs()-eu["Tow_end_depth(m)"].abs()
for mult,lab in [(1,"no x3"),(3,"x3 on s3")]:
    a=eu["Abundance(m3)"]*np.where(eu.size_class=="s3",mult,1)
    c=(a*eu["Average_biomass_per_individual(mg)"]*eu.rng*1e-3*0.43).groupby(eu.Index).sum()
    j=pd.concat([c.rename("re"),w.set_index("Index")["ZC1-EUP"]],axis=1).dropna()
    print(f"ZC1 recompute {lab}: median ratio wide/recompute = {np.median(j['ZC1-EUP']/j.re):.3f}")
