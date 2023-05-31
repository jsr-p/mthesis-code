@REM all K
make r_script_k SUFFIX=_new K=30 COL_SUFFIX=_fullk FEATURE_SUFFIX=_fullk30
make r_script_k SUFFIX=_new K=50 COL_SUFFIX=_fullk FEATURE_SUFFIX=_fullk50

@REM default k
make r_script_k SUFFIX=_new K=30 COL_SUFFIX=_defaultk FEATURE_SUFFIX=_defaultk30
make r_script_k SUFFIX=_new K=50 COL_SUFFIX=_defaultk FEATURE_SUFFIX=_defaultk50

@REM Radius
make r_script_radius SUFFIX=_new RADIUS=200 COL_SUFFIX=_defaultradius FEATURE_SUFFIX=_defaultradius200
make r_script_radius SUFFIX=_new RADIUS=400 COL_SUFFIX=_defaultradius FEATURE_SUFFIX=_defaultradius400

@REM Deciles for k=50 and radius=200
make r_script_k_deciles SUFFIX=_new K=30 COL_SUFFIX=_defaultk FEATURE_SUFFIX=_defaultk30
make r_script_radius_deciles SUFFIX=_new RADIUS=200 COL_SUFFIX=_defaultradius FEATURE_SUFFIX=_defaultradius200