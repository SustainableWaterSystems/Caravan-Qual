#!/bin/bash
#SBATCH -t 120:00:00
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH -n 192

cd /gpfs/work4/0/dynql/Caravan-Qual/

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /gpfs/home6/ejones/.conda/envs/Caravan-Qual

###Processing Caravan-Qual###

##.csv
#python scripts/Caravan-Qual/create_Caravan-Qual_csvs.py &
#wait

##Caravan-Qual (process .zarr)
#python scripts/Caravan-Qual/create_Caravan-Qual_zarr.py &
#wait


