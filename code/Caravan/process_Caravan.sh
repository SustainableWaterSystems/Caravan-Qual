#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p rome
#SBATCH -N 1
#SBATCH -n 128

cd /gpfs/work4/0/dynql/Caravan-Qual/

###---------------------------------------------------###
###                   Functions                       ###
###---------------------------------------------------###

download_and_extract() {
    local url="$1"
    local archive_name="$2"
    
    #determine folder name by removing extensions
    local folder_name="${archive_name%.tar.gz}"
    folder_name="${folder_name%.zip}"
    
    #skip if folder already exists
    if [[ -d "$folder_name" ]]; then
        echo "? Skipping: '$folder_name/' already exists."
        return
    fi
    
    #download if archive doesn't exist
    if [[ -f "$archive_name" ]]; then
        echo "? Archive '$archive_name' already exists. Skipping download."
    else
        echo "? Downloading $archive_name..."
        curl -L -C - --retry 10 --retry-delay 10 --retry-connrefused -o "$archive_name" "$url" || {
            echo "? Error: Failed to download $archive_name"
            exit 1
        }
    fi
    
    #extract archive
    echo "? Extracting $archive_name..."
    case "$archive_name" in
        *.tar.gz)
            tar -xzf "$archive_name"
            ;;
        *.zip)
            UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q "$archive_name"
            ;;
        *)
            echo "? Error: Unsupported archive format: $archive_name"
            exit 1
            ;;
    esac
    
    #clean up archive
    rm "$archive_name"
    echo "? Extraction complete: Files are in '$folder_name/'"
}

create_caravan_dirs() {
    local dataset="$1"
    mkdir -p "Caravan-nc/licenses/$dataset/"
    mkdir -p "Caravan-nc/attributes/$dataset/"
    mkdir -p "Caravan-nc/shapefiles/$dataset/"
    mkdir -p "Caravan-nc/timeseries/netcdf/$dataset/"
}


###----------------------------------------------------------------###
###                Download Caravan and extensions                 ###
###----------------------------------------------------------------###

#echo "Starting downloading Caravan and extensions..."

##Caravan dataset
#echo "Downloading Caravan"
#download_and_extract \
#    "https://zenodo.org/record/15529786/files/Caravan-nc.tar.gz?download=1" \
#    "Caravan-nc.tar.gz"
#echo "Caravan successfully downloaded!"

##GRDC extension
#echo "Downloading GRDC extension"
#download_and_extract \
#    "https://zenodo.org/record/15349031/files/GRDC_Caravan_extension_nc.zip?download=1" \
#    "GRDC_Caravan_extension_nc.zip"
#create_caravan_dirs "grdc"
#mkdir -p "Caravan-nc/timeseries/netcdf/grdc_raw/"

#mv GRDC_Caravan_extension_nc/licenses/grdc/* Caravan-nc/licenses/grdc/
#mv GRDC_Caravan_extension_nc/attributes/grdc/* Caravan-nc/attributes/grdc/
#mv GRDC_Caravan_extension_nc/shapefiles/grdc/* Caravan-nc/shapefiles/grdc/
#mv GRDC_Caravan_extension_nc/timeseries/netcdf/grdc/* Caravan-nc/timeseries/netcdf/grdc_raw/

#rm -rf GRDC_Caravan_extension_nc/
#echo "GRDC extension successfully downloaded!"

##Caravan-DE
#echo "Downloading Caravan-DE"
#download_and_extract \
#    "https://zenodo.org/records/14755229/files/caravan_de.zip?download=1" \
#    "de_Caravan_extension_nc.zip"

#create_caravan_dirs "camelsde"
#mv licenses/* Caravan-nc/licenses/ && rm -rf licenses/
#mv attributes/* Caravan-nc/attributes/ && rm -rf attributes/
#mv shapefiles/* Caravan-nc/shapefiles/ && rm -rf shapefiles/
#mv timeseries/netcdf/* Caravan-nc/timeseries/netcdf/ && rm -rf timeseries/
#echo "Caravan-DE extension successfully downloaded!"

##Caravan-DK
#echo "Downloading Caravan-DK"
#download_and_extract \
#    "https://zenodo.org/records/15200118/files/Caravan_extension_DK.zip?download=1" \
#    "dk_Caravan_extension_nc.zip"

#create_caravan_dirs "camelsdk"
#mv licenses/* Caravan-nc/licenses/ && rm -rf licenses/
#mv attributes/* Caravan-nc/attributes/ && rm -rf attributes/
#mv shapefiles/* Caravan-nc/shapefiles/ && rm -rf shapefiles/
#mv timeseries/netcdf/* Caravan-nc/timeseries/netcdf/ && rm -rf timeseries/
#echo "Caravan-DK extension successfully downloaded!"

##Caravan-IS
#echo "Downloading Caravan-IS"
#download_and_extract \
#    "https://zenodo.org/records/15181680/files/Caravan_extension_Israel_Ver4.zip?download=1" \
#    "is_Caravan_extension_nc.zip"
#
#create_caravan_dirs "camelsil"
#mv Caravan_extension_Israel_Ver4/licenses/il/* Caravan-nc/licenses/camelsil/
#mv Caravan_extension_Israel_Ver4/attributes/il/* Caravan-nc/attributes/camelsil/
#mv Caravan_extension_Israel_Ver4/shapefiles/il/* Caravan-nc/shapefiles/camelsil/
#
#for f in Caravan_extension_Israel_Ver4/timeseries/netcdf/il/il_*.nc; do
#    if [[ -f "$f" ]]; then
#        mv "$f" "$(dirname "$f")/camels$(basename "$f")"
#    fi
#done

#mv Caravan_extension_Israel_Ver4/timeseries/netcdf/il/* Caravan-nc/timeseries/netcdf/camelsil/
#rm -rf Caravan_extension_Israel_Ver4/

#echo "Caravan-IS extension successfully downloaded!"

##Caravan-CH
#echo "Downloading Caravan-CH"
#download_and_extract \
#    "https://zenodo.org/records/15025258/files/Caravan_extension_CH.zip?download=1" \
#    "camels_ch.zip"

#create_caravan_dirs "camelsch"
#mv Caravan_extension_CH/Caravan_extension_CH/licenses/camelsch/* Caravan-nc/licenses/camelsch/
#mv Caravan_extension_CH/Caravan_extension_CH/attributes/camelsch/* Caravan-nc/attributes/camelsch/
#mv Caravan_extension_CH/Caravan_extension_CH/shapefiles/camelsch/* Caravan-nc/shapefiles/camelsch/
#mv Caravan_extension_CH/Caravan_extension_CH/timeseries/netcdf/camelsch/* Caravan-nc/timeseries/netcdf/camelsch/
#rm -rf Caravan_extension_CH/
#echo "Caravan-CH extension successfully downloaded!"

##Caravan-ES
#echo "Downloading Caravan-ES"
#download_and_extract \
#    "https://zenodo.org/records/15040948/files/CAMELS-ES_v110.zip?download=1" \
#    "camels_es.zip"

#create_caravan_dirs "camelses"
#mv v110/licenses/camelses/* Caravan-nc/licenses/camelses/
#mv v110/attributes/camelses/* Caravan-nc/attributes/camelses/
#mv v110/shapefiles/camelses/* Caravan-nc/shapefiles/camelses/
#mv v110/timeseries/netcdf/camelses/* Caravan-nc/timeseries/netcdf/camelses/
#rm -rf v110/
#echo "Caravan-ES extension successfully downloaded!"

##Caravan-IND
#echo "Downloading Caravan-IND"
#wget "https://zenodo.org/record/14999580/files/CAMELS_IND_All_Catchments.zip?download=1" \
#    -O Caravan-nc/CAMELS_IND_All_Catchments.zip
#unzip -q Caravan-nc/CAMELS_IND_All_Catchments.zip -d Caravan-nc/camels_ind
#rm Caravan-nc/CAMELS_IND_All_Catchments.zip
#echo "Caravan-IND extension successfully downloaded!"

##Caravan-FR
#echo "Caravan-FR must be manually downloaded..."
#echo "Can be retrieved from: https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/WH7FJR"
#echo "Downloaded folder is called dataverse_files.zip, which should be located in the working directory defined at the start of this script."

#mkdir -p Caravan-nc/camels_fr
#unzip dataverse_files.zip -d camels_fr/
#unzip camels_fr/CAMELS_FR_attributes.zip -d camels_fr/
#unzip camels_fr/CAMELS_FR_geography.zip -d camels_fr/
#unzip camels_fr/CAMELS_FR_time_series.zip -d camels_fr/
#rm dataverse_files.zip
#rm camels_fr/ADDITIONAL_LICENSES.zip \
#   camels_fr/CAMELS-FR_description.ods \
#   camels_fr/CAMELS_FR_attributes.zip \
#   camels_fr/CAMELS_FR_geography.zip \
#   camels_fr/CAMELS_FR_time_series.zip \
#   camels_fr/MANIFEST.TXT \
#   camels_fr/NEWS.md \
#   camels_fr/README.md
#mv camels_fr/ Caravan-nc/

##Caravan-CZ
#echo "Downloading Caravan-CZ"
#download_and_extract \
#    "https://zenodo.org/records/17769325/files/Caravan-Extension-CZ.zip?download=1" \
#    "caravan_cz.zip"

#create_caravan_dirs "camelscz"
#mv Caravan-Extension-CZ/license/camelscz/* Caravan-nc/licenses/camelscz/
#mv Caravan-Extension-CZ/attributes/camelscz/* Caravan-nc/attributes/camelscz/
#mv Caravan-Extension-CZ/shapefiles/camelscz/* Caravan-nc/shapefiles/camelscz/
#mv Caravan-Extension-CZ/timeseries/netcdf/camelscz/* Caravan-nc/timeseries/netcdf/camelscz/
#rm -rf Caravan-Extension-CZ/

##LamaH-Ice
#echo "Downloading LamaH-Ice"
#download_and_extract \
#    "https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/data/contents/LamaH-Ice_Caravan_Extension_v15.zip" \
#    "lamah_ice.zip"

#create_caravan_dirs "lamahice"
#mv LamaH-Ice_Caravan_Extension_v15/licenses/lamahice/* Caravan-nc/licenses/lamahice/
#mv LamaH-Ice_Caravan_Extension_v15/attributes/lamahice/* Caravan-nc/attributes/lamahice/
#mv LamaH-Ice_Caravan_Extension_v15/shapefiles/lamahice/* Caravan-nc/shapefiles/lamahice/
#mv LamaH-Ice_Caravan_Extension_v15/timeseries/netcdf/lamahice/* Caravan-nc/timeseries/netcdf/lamahice/
#rm -rf LamaH-Ice_Caravan_Extension_v15/

###----------------------------------------------------------------###
###              Process Caravan and extensions (python)           ###
###----------------------------------------------------------------###
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/home6/ejones/.conda/envs/myenv

#echo "Processing extensions that require additional handling..."
#python scripts/Caravan/process_extensions.py

#rm -r Caravan-nc/camels_fr/ #remove Caravan-FR raw files
#rm -r Caravan-nc/camels_ind/ #remove Caravan-IND raw files
#rm -r Caravan-nc/timeseries/netcdf/grdc_raw/ #remove (unprocessed) GRDC netcdfs

###----------------------------------------------------------------###
###                 Relocate to auxiliary folder                   ###
###----------------------------------------------------------------###
#mv Caravan-nc/* /gpfs/work4/0/dynql/Caravan-Qual/auxiliary/Caravan/
#rmdir Caravan-nc/

#echo "Finished downloading Caravan + extensions!"

###----------------------------------------------------------------###
###                      Gauge assignments                         ###
###----------------------------------------------------------------###
echo "Processing gauge station assignments..."
python scripts/Caravan/process_all_gauges.py
echo "Finished processing gauge station assignments!"