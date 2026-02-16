#!/bin/bash

python detection.py \
        /home/aya/work/hkoehn/mma_analysis/catalogs/full_narrow.dat \
        /home/aya/work/hkoehn/mma_analysis/catalogs/outdir_FIM/full_narrow_ETL1_ETL2_gwfish.dat \
        /home/aya/work/hkoehn/mma_analysis/kilonova/full_narrow_kn_parameters.dat \
        /home/aya/work/hkoehn/mma_analysis/grb/full_narrow_grb_parameters.dat \


python detection.py \
        /home/aya/work/hkoehn/mma_analysis/catalogs/full_narrow.dat \
        /home/aya/work/hkoehn/mma_analysis/catalogs/outdir_FIM/full_narrow_ETL1_ETL2_CE_gwfish.dat \
        /home/aya/work/hkoehn/mma_analysis/kilonova/full_narrow_kn_parameters.dat \
        /home/aya/work/hkoehn/mma_analysis/grb/full_narrow_grb_parameters.dat \


python detection.py \
        /home/aya/work/hkoehn/mma_analysis/catalogs/full_wide.dat \
        /home/aya/work/hkoehn/mma_analysis/catalogs/outdir_FIM/full_wide_ETL1_ETL2_gwfish.dat \
        /home/aya/work/hkoehn/mma_analysis/kilonova/full_wide_kn_parameters.dat \
        /home/aya/work/hkoehn/mma_analysis/grb/full_wide_grb_parameters.dat \


python detection.py \
        /home/aya/work/hkoehn/mma_analysis/catalogs/full_wide.dat \
        /home/aya/work/hkoehn/mma_analysis/catalogs/outdir_FIM/full_wide_ETL1_ETL2_CE_gwfish.dat \
        /home/aya/work/hkoehn/mma_analysis/kilonova/full_wide_kn_parameters.dat \
        /home/aya/work/hkoehn/mma_analysis/grb/full_wide_grb_parameters.dat \
