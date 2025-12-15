#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -p rome
#SBATCH -N 1
#SBATCH -n 128

cd /gpfs/work4/0/dynql/Caravan-Qual/

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /gpfs/home6/ejones/.conda/envs/myenv


### ---------------------------------------- ###
###       PROCESS INDIVIDUAL DATASETS        ###
### ---------------------------------------- ###

###GRQA
#python scripts/add_data/add_wq_dataset.py GRQA
#python scripts/add_data/add_gauge_id.py GRQA

###GEMS
#python scripts/add_data/add_wq_dataset.py GEMS
#python scripts/add_data/add_gauge_id.py GEMS

###Waterbase
#python scripts/add_data/add_wq_dataset.py Waterbase
#python scripts/add_data/add_gauge_id.py Waterbase

###WQP
#python scripts/add_data/add_wq_dataset.py WQP
#python scripts/add_data/add_gauge_id.py WQP

###EMPODAT
#python scripts/add_data/add_wq_dataset.py EMPODAT
#python scripts/add_data/add_gauge_id.py EMPODAT

###Elbe Rhine RIWA
#python scripts/add_data/add_wq_dataset.py Elbe_Rhine_RIWA
#python scripts/add_data/add_gauge_id.py Elbe_Rhine_RIWA

###UK-EA
#python scripts/add_data/add_wq_dataset.py UK_EA
#python scripts/add_data/add_gauge_id.py UK_EA

###Camels-CH-Chem
#python scripts/add_data/add_wq_dataset.py Camels_CH_Chem
#python scripts/add_data/add_gauge_id.py Camels_CH_Chem

###CNEMC
#python scripts/add_data/add_wq_dataset.py CNEMC
#python scripts/add_data/add_gauge_id.py CNEMC

###extra
#python scripts/add_data/add_wq_dataset.py extra
#python scripts/add_data/add_gauge_id.py extra

### -------------------------------------- ###
###       PROCESS COMBINED DATASETS        ###
### -------------------------------------- ###

###process gauge_id for all sites
#python scripts/add_data/add_gauge_id.py 


###optional: get raw site info from all raw sites (i.e. original site name, lat, lon and source)
#awk -F',' '
#NR==1 {
#    print $1","$2","$3","$11","$4 > "auxiliary/wq_data/site_info.csv"
#    next
#}
#{
#    key = $1","$2","$3
#    if (!seen[key]++)
#        print $1","$2","$3","$11","$4 >> "auxiliary/wq_data/site_info.csv"
#}
#' auxiliary/wq_data/combined_wqms_dataset.csv


###process shapefiles for each LINKNO (whole database)
##this will first process at native resolution of TDXhydro, before simplifying into wqms_basin_shapes.gpkg
#python scripts/add_data/add_wqms_shps.py

#ogr2ogr -f GPKG auxiliary/wqms-gpkg/wqms_basin_shapes.gpkg auxiliary/wqms-gpkg/wqms_TDXhydro_catchments.gpkg \
#  -nln catchments \
#  -simplify 0.001 \
#  -nlt MULTIPOLYGON \
#  -makevalid

###extract hydroatlas attributes
python scripts/add_data/extract_HydroATLAS_attributes.py