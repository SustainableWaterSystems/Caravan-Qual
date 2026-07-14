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

###GLORICH
#python scripts/add_data/add_wq_dataset.py GLORICH
#python scripts/add_data/add_gauge_id.py GLORICH

###Waterbase
#python scripts/add_data/add_wq_dataset.py Waterbase
#python scripts/add_data/add_gauge_id.py Waterbase

###WQP
#python scripts/add_data/add_wq_dataset.py WQP
#python scripts/add_data/add_gauge_id.py WQP

###EMPODAT
#python scripts/add_data/add_wq_dataset.py EMPODAT
#python scripts/add_data/add_gauge_id.py EMPODAT

###UK-EA
#python scripts/add_data/add_wq_dataset.py UK_EA
#python scripts/add_data/add_gauge_id.py UK_EA

###CESI
#python scripts/add_data/add_wq_dataset.py CESI
#python scripts/add_data/add_gauge_id.py CESI

###CNEMC
#python scripts/add_data/add_wq_dataset.py CNEMC
#python scripts/add_data/add_gauge_id.py CNEMC

###extra
#python scripts/add_data/add_wq_dataset.py extra
#python scripts/add_data/add_gauge_id.py extra

###Elbe Rhine RIWA
#python scripts/add_data/add_wq_dataset.py Elbe_Rhine_RIWA
#python scripts/add_data/add_gauge_id.py Elbe_Rhine_RIWA

###IWRMC
#python scripts/add_data/add_wq_dataset.py IWRMC
#python scripts/add_data/add_gauge_id.py IWRMC

###Camels-CH-Chem
#python scripts/add_data/add_wq_dataset.py Camels-CH-Chem
#python scripts/add_data/add_gauge_id.py Camels-CH-Chem

###Wilkinson
#python scripts/add_data/add_wq_dataset.py Wilkinson
#python scripts/add_data/add_gauge_id.py Wilkinson

###HK_EPD
#python scripts/add_data/add_wq_dataset.py HK_EPD
#python scripts/add_data/add_gauge_id.py HK_EPD


### -------------------------------------- ###
###       PROCESS COMBINED DATASETS        ###
### -------------------------------------- ###

###process gauge_id for all sites
python scripts/add_data/add_gauge_id.py 

###optional: get raw site info from all raw sites (i.e. original site name, lat, lon and source)
#awk -F',' '
#NR==1 {
#    #store column indices based on header names
#    for (i=1; i<=NF; i++) {
#        col[$i] = i
#    }
#    print "site_id,lat,lon,source,wqms_id" > "auxiliary/wq_data/site_info.csv"
#    next
#}
#{
#    key = $col["site_id"]","$col["lat"]","$col["lon"]
#    if (!seen[key]++)
#        print $col["site_id"]","$col["lat"]","$col["lon"]","$col["source"]","$col["wqms_id"] >> "auxiliary/wq_data/site_info.csv"
#}
#' auxiliary/wq_data/combined_wqms_dataset.csv


###process shapefiles for each LINKNO (whole database) (first at native resolution of TDXhydro, before simplifying into wqms_basin_shapes.gpkg)
#python scripts/add_data/add_wqms_shps.py

#ogr2ogr -f GPKG auxiliary/wqms-gpkg/wqms_basin_shapes.gpkg auxiliary/wqms-gpkg/wqms_TDXhydro_catchments.gpkg \
#  -nln catchments \
#  -simplify 0.001 \
#  -nlt MULTIPOLYGON \
#  -makevalid

###extract hydroatlas attributes
#python scripts/add_data/extract_HydroATLAS_attributes.py
