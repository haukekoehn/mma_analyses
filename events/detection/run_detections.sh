#!/bin/bash

python detection.py \
        /home/aya/work/hkoehn/mma_analysis/events/gw/gw_params_narrow.dat \
        ./outdir_FIM/full_narrow_ETL1_ETL2_gwfish.dat \
        /home/aya/work/hkoehn/mma_analysis/events/kilonova/kn_params_narrow.dat \
        /home/aya/work/hkoehn/mma_analysis/events/grb/grb_params_narrow.dat \


python detection.py \
        /home/aya/work/hkoehn/mma_analysis/events/gw/gw_params_narrow.dat \
        ./outdir_FIM/full_narrow_ETL1_ETL2_CE_gwfish.dat \
        /home/aya/work/hkoehn/mma_analysis/events/kilonova/kn_params_narrow.dat \
        /home/aya/work/hkoehn/mma_analysis/events/grb/grb_params_narrow.dat \


python detection.py \
        /home/aya/work/hkoehn/mma_analysis/events/gw/gw_params_wide.dat \
        ./outdir_FIM/full_wide_ETL1_ETL2_gwfish.dat \
        /home/aya/work/hkoehn/mma_analysis/events/kilonova/kn_params_wide.dat \
        /home/aya/work/hkoehn/mma_analysis/events/grb/grb_params_wide.dat \


python detection.py \
        /home/aya/work/hkoehn/mma_analysis/events/gw/gw_params_wide.dat \
        ./outdir_FIM/full_wide_ETL1_ETL2_CE_gwfish.dat \
        /home/aya/work/hkoehn/mma_analysis/events/kilonova/kn_params_wide.dat \
        /home/aya/work/hkoehn/mma_analysis/events/grb/grb_params_wide.dat \
