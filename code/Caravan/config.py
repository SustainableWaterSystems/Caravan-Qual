#!/usr/bin/env python3

import os
from pathlib import Path
from datetime import datetime

class Config:
    
    # Base directories
    BASE_DIR = Path("/gpfs/work4/0/dynql/Caravan-Qual/")
    CARAVAN_DIR = BASE_DIR / "auxiliary/Caravan/"
    AUX_DATA_DIR = BASE_DIR / "auxiliary"
    
    # Temporary download directories
    RAW_CARAVAN_DIR = BASE_DIR / "Caravan-nc"
    TEMP_CAMELSFR_DIR = RAW_CARAVAN_DIR / "camels_fr"
    TEMP_CAMELSIND_DIR = RAW_CARAVAN_DIR / "camels_ind"
    
    # Caravan subdirectories
    LICENSES_DIR = CARAVAN_DIR / "licenses"
    ATTRIBUTES_DIR = CARAVAN_DIR / "attributes"
    SHAPEFILES_DIR = CARAVAN_DIR / "shapefiles"
    TIMESERIES_DIR = CARAVAN_DIR / "timeseries" / "netcdf"
    
    # Output files
    COMBINED_CARAVAN_ATTRS = ATTRIBUTES_DIR / "combined_attributes_caravan.csv"
    COMBINED_HYDROATLAS_ATTRS = ATTRIBUTES_DIR / "combined_attributes_hydroatlas.csv"
    COMBINED_OTHER_ATTRS = ATTRIBUTES_DIR / "combined_attributes_other.csv"
    SITE_INFO_OUTPUT = CARAVAN_DIR / "caravan_site_info.csv"
    
    # Auxiliary data files
    COUNTRY_SHP = AUX_DATA_DIR / "WorldBank" / "WB_countries_Admin0_10m.shp"
    HYDROBASIN_SHP = AUX_DATA_DIR / "HydroATLAS" / "BasinATLAS_v10_lev12.shp"
    RIVERS_GPKG = AUX_DATA_DIR / "geoglows_TDXhydro" / "global_streams_simplified.gpkg"
    BRANCH_MAP_CSV = AUX_DATA_DIR / "geoglows_TDXhydro" / "merged_branches.csv"
    CATCHMENTS_GPKG_DIR = AUX_DATA_DIR / "geoglows_TDXhydro" / "catchments_gpkg"
    
    # NetCDF settings
    NETCDF_TIME_UNITS = "days since 1951-01-01 00:00:00"
    NETCDF_CALENDAR = "proleptic_gregorian"
    NETCDF_START_YEAR = 1951
    NETCDF_END_YEAR = 2020
    NETCDF_START_DATE = datetime(NETCDF_START_YEAR, 1, 1)
    NETCDF_END_DATE = datetime(NETCDF_END_YEAR, 12, 31)
    
    # Processing options
    GET_LINKNO_FROM_GPKG = True
    SHOW_PROGRESS = True

    @classmethod
    def create_dataset_dirs(cls, dataset_name):
        """Create standard directory structure for a dataset"""
        (cls.LICENSES_DIR / dataset_name).mkdir(parents=True, exist_ok=True)
        (cls.ATTRIBUTES_DIR / dataset_name).mkdir(parents=True, exist_ok=True)
        (cls.SHAPEFILES_DIR / dataset_name).mkdir(parents=True, exist_ok=True)
        (cls.TIMESERIES_DIR / dataset_name).mkdir(parents=True, exist_ok=True)