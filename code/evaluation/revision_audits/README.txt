Supporting analyses for the revised manuscript (zooplankton evaluation, bloom-timing reference skill).
Paths are relative to this folder and the sibling SOGEM-LTL-data repository. Run from this folder with the
project environment (pandas, scipy, xarray); the GAM step needs R with mgcv.

sampling_summary_table.py      Tows by year and month, stations and night share (Supplement Table S34)
                               -> eval/tables/zoop_sampling_summary_year_month.csv
perry_audit.py                 Checks tow selection, per-tow biomass and annual anomalies against Perry et al. (2021)
                               (Methods 2.4: "reproduced their published annual anomalies")
zoop_annual_2015_2018.py       Annual crustacean anomalies 2015-2018, observed vs tow-matched Ecospace
                               (Results 3.4; Supplement Table S35, Figure S6); also reports the GAM-index check
                               -> eval/tables/ecospace_zoop_annual_anom_2015_2018_SC300.csv
                               -> eval/figs/ecospace_SC300_zoop_annual_anom_2015_2018.png
export_model_window_index.py   Exports tow table and model seasonal/annual indices used by the GAM step
                               (writes gam_input_tows.csv and gam_model_window_index.csv here; not versioned)
gam_standardized_index.R       mgcv standardised observed index (year + s(day of year) + station) vs model
                               -> eval/tables/ecospace_zoop_gam_index_values_SC300.csv (read by zoop_annual_2015_2018.py)
station_effects.py             Station vs year/month share of the obs/model mismatch (Supplement S.3.6)
c09_vs_satellite.py            C09 bloom-timing skill against the satellite series (manuscript Table 5, reference row)

Related scripts in ../ :
ecospace_eval_5d_zoop_sampling_diagnostics.py   Bloom timing vs spring crustacean biomass (Results 3.4;
                                                Supplement Table S38, Figure S10); --phenology runs this part only
                                                -> eval/tables/ecospace_zoop_bloom_phenology_links_SC300.csv
                                                -> eval/tables/ecospace_zoop_full_model_spring_vs_bloom_SC300.csv
                                                -> eval/figs/ecospace_SC300_zoop_spring_vs_bloomday.png
ecospace_eval_3c_QU39_seasonal_check.py         Seasonal vs deseasoned QU39 agreement (Results 3.2)
                                                -> eval/tables/ecospace_SC300_QU39_seasonal_check.csv

Order: sampling_summary_table.py, export_model_window_index.py, gam_standardized_index.R, zoop_annual_2015_2018.py.
