import os
import pandas as pd
from GO_helpers import buildSortableString, saveASCFile, getDataFrame # 2023-04


greigDir = "/project/6006412/goldford/ECOSPACE/"
depthSubDir = "DATA/"
ecospacegrid_f = os.path.join(greigDir, depthSubDir, "ecospacedepthgrid.asc")
df_dep = getDataFrame(ecospacegrid_f,"-9999.00000000")
dfLandMask = df_dep == 0 #Plume is region one

print(dfLandMask)