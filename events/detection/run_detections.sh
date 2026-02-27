#!/bin/bash

python detection.py \
        ../gw/gw_params_narrow.dat \
        ./outdir_FIM/full_narrow_ETL1_ETL2_gwfish.dat \
        ../kilonova/kn_params_narrow.dat \
        ../grb/grb_params_narrow.dat \


python detection.py \
        ../gw/gw_params_narrow.dat \
        ./outdir_FIM/full_narrow_ETL1_ETL2_CE_gwfish.dat \
        ../kilonova/kn_params_narrow.dat \
        ../grb/grb_params_narrow.dat \


python detection.py \
        ../gw/gw_params_wide.dat \
        ./outdir_FIM/full_wide_ETL1_ETL2_gwfish.dat \
        ../kilonova/kn_params_wide.dat \
        ../grb/grb_params_wide.dat \


python detection.py \
        ../gw/gw_params_wide.dat \
        ./outdir_FIM/full_wide_ETL1_ETL2_CE_gwfish.dat \
        ../kilonova/kn_params_wide.dat \
        ../grb/grb_params_wide.dat \
