import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import zarr
from numcodecs import Zstd
from datetime import datetime
from netCDF4 import Dataset
import glob
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial
import geopandas as gpd

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

#Input directories and files
input_dir_wqms = "/gpfs/work4/0/dynql/Caravan-Qual/"
input_dir_caravan = "/gpfs/work4/0/dynql/Caravan-Qual/auxiliary/Caravan/"
input_weather_zarr = "/gpfs/work4/0/dynql/weather_data.zarr/"

caravan_timeseries_path = os.path.join(input_dir_caravan, "timeseries/netcdf/")
caravan_site_info = os.path.join(input_dir_caravan, "caravan_site_info.csv")
csv_dir = os.path.join(input_dir_wqms, "wqms-csv")
site_info = os.path.join(input_dir_wqms, "wqms_site_info.csv")
catchment_attrs_csv = os.path.join(input_dir_wqms, "auxiliary/attributes/linkno_hydroatlas_attributes_Caravan.csv")
units_file_csv = os.path.join(input_dir_wqms, "auxiliary/wq_data/wq_variable_list.csv")
geoglows_gdb = os.path.join(input_dir_wqms, "auxiliary/geoglows_TDXhydro/geoglows-v2-map-optimized.gdb")

#Output directories and file paths
output_dir_wqms = "/gpfs/work4/0/dynql/Caravan-Qual/"
os.makedirs(output_dir_wqms, exist_ok=True)

output_zarr_dir = os.path.join(output_dir_wqms, "Caravan-Qual.zarr")
output_wq_linkages_path = os.path.join(output_dir_wqms, "Caravan-Qual_linkages.parquet")

#Define time range
START_DATE = pd.Timestamp('1980-01-01').date()
END_DATE = pd.Timestamp('2025-09-30').date()
WQ_UNITS = {}

#Define chunking strategy
ZARR_CHUNKS = {
    'time': 16710,
    'gauge_id': 500,
    'wqms_id': 2500,
    'LINKNO': 100,
}

#parallelisation settings
N_WORKERS = min(cpu_count() - 1, 32)  #max 32 workers

###---------------------------------------------------###
###                   Functions                       ###
###---------------------------------------------------###

def load_wq_units(units_file_csv):
    if os.path.exists(units_file_csv):
        df_units = pd.read_csv(units_file_csv)
        return dict(zip(df_units['variable_code'], df_units['target_unit']))
    else:
        raise FileNotFoundError(f"Units file not found: {units_file_csv}")


def get_wq_units(param_name):
    """Get units for a water quality parameter."""
    return WQ_UNITS.get(param_name, 'unknown')


def load_geoglows_data(gdb_path):
    print(f"Loading GeoGLOWS data from {gdb_path}")
    gdf = gpd.read_file(gdb_path)
    print(f"  Loaded {len(gdf)} features with {len(gdf.columns)} attributes")

    if 'LINKNO' not in gdf.columns:
        raise ValueError(f"LINKNO column not found. Available columns: {list(gdf.columns)}")
    
    #select specific columns to add to zarr
    required_cols = ['LINKNO', 'strmOrder', 'DSContArea', 'TDXHydroRegion', 
                     'TopologicalOrder', 'LengthGeodesicMeters', 'TerminalLink', 
                     'musk_k', 'musk_x']
    
    missing_cols = [col for col in required_cols if col not in gdf.columns]
    if missing_cols:
        print(f"  Warning: Missing columns in GeoGLOWS data: {missing_cols}")
        available_cols = [col for col in required_cols if col in gdf.columns]
    else:
        available_cols = required_cols
    
    print(f"  Selecting {len(available_cols)} GeoGLOWS columns: {available_cols}")
    
    #convert to regular dataframe
    df = pd.DataFrame(gdf[available_cols].drop(columns='geometry', errors='ignore'))
    df['LINKNO'] = df['LINKNO'].astype('Int64')
    
    if df['LINKNO'].duplicated().any():
        print(f"  Warning: Found {df['LINKNO'].duplicated().sum()} duplicate LINKNO entries, keeping first occurrence")
        df = df.drop_duplicates(subset='LINKNO', keep='first')
    
    print(f"  Processed {len(df)} unique LINKNO records\n")
    return df


def build_netcdf_file_map(base_path):
    print(f"Building NetCDF file map from {base_path}")
    netcdf_files = glob.glob(os.path.join(base_path, "**/*.nc"), recursive=True)
    gauge_to_path = {}
    for file_path in netcdf_files:
        gauge_id = os.path.splitext(os.path.basename(file_path))[0]
        gauge_to_path[gauge_id.lower()] = file_path
    print(f"  Found {len(gauge_to_path)} NetCDF files\n")
    return gauge_to_path


def load_gauge_metadata(caravan_site_info_csv, netcdf_file_map):
    print(f"Loading gauge metadata from {caravan_site_info_csv}")
    df_gauge_meta = pd.read_csv(caravan_site_info_csv, dtype={"gauge_id": str})
    
    if "area" not in df_gauge_meta.columns:
        raise ValueError("'area' column not found")
    
    #check for lat/lon columns
    if "gauge_lat" not in df_gauge_meta.columns or "gauge_lon" not in df_gauge_meta.columns:
        raise ValueError("'gauge_lat' and 'gauge_lon' columns required in caravan_site_info")
    
    #filter for valid gauges
    df_valid = df_gauge_meta[df_gauge_meta['area'].notna()].copy()
    gauge_ids = [gid for gid in df_valid['gauge_id'].values if gid.lower() in netcdf_file_map]
    gauge_ids_sorted = sorted(gauge_ids)
    
    print(f"  Found {len(gauge_ids_sorted)} valid gauges\n")
    return df_gauge_meta, gauge_ids_sorted


def load_linkages_and_wqms_ids(site_info_csv, catchment_attrs_csv, sites_to_process=None):
    df_linkages = pd.read_csv(site_info_csv, dtype={"wqms_id": str, "LINKNO": "Int64", "merged_LINKNO": "Int64", "gauge_id": str})
    
    #Check for lat/lon columns
    if "wqms_lat" not in df_linkages.columns or "wqms_lon" not in df_linkages.columns:
        raise ValueError("'wqms_lat' and 'wqms_lon' columns required in wqms_site_info")
    
    #Check for country and hydrobasin columns
    if "country_name" not in df_linkages.columns:
        raise ValueError("'country_name' column required in wqms_site_info")
    if "hydrobasin_level12" not in df_linkages.columns:
        raise ValueError("'hydrobasin_level12' column required in wqms_site_info")
    
    #Check for merged_LINKNO column
    if "merged_LINKNO" not in df_linkages.columns:
        print("  Warning: 'merged_LINKNO' column not found in wqms_site_info")
    
    if sites_to_process is not None:
        wqms_ids = df_linkages[df_linkages['wqms_id'].isin(sites_to_process)]['wqms_id'].unique()
    else:
        wqms_ids = df_linkages['wqms_id'].unique()
    
    wqms_ids_sorted = sorted([str(wid) for wid in wqms_ids])
    
    print(f"Loading catchment attributes from {catchment_attrs_csv}")
    df_attrs = pd.read_csv(catchment_attrs_csv, dtype={"LINKNO": "Int64"})
    
    print(f"  Found {len(wqms_ids_sorted)} WQMS stations\n")
    print(f"Linkages between wqms_id, LINKNO and gauge_id loaded from {site_info_csv}")
    return df_linkages, df_attrs, wqms_ids_sorted


def merge_geoglows_attributes(df_attrs, df_geoglows):
    print(f"Merging GeoGLOWS attributes with existing catchment attributes...")
    
    #get LINKNO values that exist in both datasets
    common_linknos = set(df_attrs['LINKNO'].dropna()) & set(df_geoglows['LINKNO'].dropna())
    print(f"  Found {len(common_linknos)} common LINKNO values between datasets")
    
    #merge on LINKNO
    df_merged = df_attrs.merge(df_geoglows, on='LINKNO', how='left', suffixes=('', '_geoglows'))
    
    return df_merged


def get_param_names(csv_dir):
    param_files = [f for f in os.listdir(csv_dir) if f.lower().endswith(".csv")]
    return sorted([os.path.splitext(f)[0] for f in param_files])


def load_weather_variables(weather_zarr_path):    
    print(f"Loading weather variables from {weather_zarr_path}")
    ds_weather = xr.open_zarr(weather_zarr_path)
    weather_vars = list(ds_weather.data_vars.keys())
    ds_weather.close()
    print(f"  Found {len(weather_vars)} weather variables\n")
    return weather_vars


###---------------------------------------------------###
###                Initialize Zarr Store              ###
###---------------------------------------------------###

def initialize_zarr_store(output_zarr_dir, gauge_ids, wqms_ids, dates_sorted, wq_params, 
                         df_attrs, df_gauge_meta, df_linkages, weather_vars):
    """Initialize Zarr store with streamflow, water quality, catchment attributes, AND weather data."""
    print(f"Initialising Zarr store at {output_zarr_dir}")
    
    if os.path.exists(output_zarr_dir):
        print(f"  Clearing existing store...")
        shutil.rmtree(output_zarr_dir)
    
    time_coord = pd.to_datetime(dates_sorted)
    n_time = len(dates_sorted)
    n_gauges = len(gauge_ids)
    n_wqms = len(wqms_ids)
    
    df_attrs_clean = df_attrs[df_attrs['LINKNO'].notna()].sort_values('LINKNO').reset_index(drop=True)
    linkno_values = df_attrs_clean['LINKNO'].values.astype('i4')
    n_linkno = len(linkno_values)
    
    #Prepare gauge lat/lon arrays (aligned with gauge_ids)
    gauge_meta_dict = df_gauge_meta.set_index("gauge_id").to_dict("index")
    gauge_lats = np.array([gauge_meta_dict.get(gid, {}).get('gauge_lat', np.nan) for gid in gauge_ids], dtype='f4')
    gauge_lons = np.array([gauge_meta_dict.get(gid, {}).get('gauge_lon', np.nan) for gid in gauge_ids], dtype='f4')
    
    #Prepare wqms lat/lon arrays (aligned with wqms_ids)
    wqms_meta_dict = df_linkages.set_index("wqms_id").to_dict("index")
    wqms_lats = np.array([wqms_meta_dict.get(wid, {}).get('wqms_lat', np.nan) for wid in wqms_ids], dtype='f4')
    wqms_lons = np.array([wqms_meta_dict.get(wid, {}).get('wqms_lon', np.nan) for wid in wqms_ids], dtype='f4')
    
    #Prepare wqms country and hydrobasin arrays (aligned with wqms_ids)
    wqms_countries = np.array([wqms_meta_dict.get(wid, {}).get('country_name', '') for wid in wqms_ids], dtype='U100')
    wqms_hydrobasins = np.array([wqms_meta_dict.get(wid, {}).get('hydrobasin_level12', '') for wid in wqms_ids], dtype='U100')
    
    #Prepare merged_LINKNO array (aligned with wqms_ids)
    wqms_merged_linknos_raw = [wqms_meta_dict.get(wid, {}).get('merged_LINKNO', np.nan) for wid in wqms_ids]
    wqms_merged_linknos = np.array(wqms_merged_linknos_raw, dtype='f8')  # float64 to handle NaN

    #create streamflow dataset with coordinates
    print(f"  Writing streamflow array with coordinates...")
    ds_streamflow = xr.Dataset({
        'streamflow': xr.DataArray(
            np.full((n_gauges, n_time), np.nan, dtype='f4'),
            dims=['gauge_id', 'time'],
            coords={
                'gauge_id': np.array(gauge_ids, dtype='U50'), 
                'time': time_coord
            },
            attrs={'units': 'm3/s', 'long_name': 'Streamflow'}
        ),
        'gauge_lat': xr.DataArray(
            gauge_lats,
            dims=['gauge_id'],
            coords={'gauge_id': np.array(gauge_ids, dtype='U50')},
            attrs={'units': 'degrees_north', 'long_name': 'Gauge latitude'}
        ),
        'gauge_lon': xr.DataArray(
            gauge_lons,
            dims=['gauge_id'],
            coords={'gauge_id': np.array(gauge_ids, dtype='U50')},
            attrs={'units': 'degrees_east', 'long_name': 'Gauge longitude'}
        )
    })
    
    encoding = {
        'streamflow': {'chunks': (ZARR_CHUNKS['gauge_id'], ZARR_CHUNKS['time'])},
        'gauge_lat': {'chunks': (ZARR_CHUNKS['gauge_id'],)},
        'gauge_lon': {'chunks': (ZARR_CHUNKS['gauge_id'],)}
    }
    ds_streamflow.to_zarr(output_zarr_dir, mode='w', encoding=encoding, consolidated=False, zarr_format=2)
    del ds_streamflow
    
    #create water quality arrays with coordinates
    print(f"  Writing {len(wq_params)} water quality parameter arrays with coordinates...")
    
    for idx, param_name in enumerate(wq_params):
        print(f"    - {param_name} ({idx+1}/{len(wq_params)})")
        
        #get units per parameter
        units = get_wq_units(param_name)
        
        ds_wq = xr.Dataset({
            param_name: xr.DataArray(
                np.full((n_wqms, n_time), np.nan, dtype='f4'),
                dims=['wqms_id', 'time'],
                coords={
                    'wqms_id': np.array(wqms_ids, dtype='U50'), 
                    'time': time_coord
                },
                attrs={'units': units, 'long_name': param_name}
            )
        })
        
        #Add wqms coordinates only on first parameter to avoid duplication
        if idx == 0:
            ds_wq['wqms_lat'] = xr.DataArray(
                wqms_lats,
                dims=['wqms_id'],
                coords={'wqms_id': np.array(wqms_ids, dtype='U50')},
                attrs={'units': 'degrees_north', 'long_name': 'WQMS station latitude'}
            )
            ds_wq['wqms_lon'] = xr.DataArray(
                wqms_lons,
                dims=['wqms_id'],
                coords={'wqms_id': np.array(wqms_ids, dtype='U50')},
                attrs={'units': 'degrees_east', 'long_name': 'WQMS station longitude'}
            )
            ds_wq['country_name'] = xr.DataArray(
                wqms_countries,
                dims=['wqms_id'],
                coords={'wqms_id': np.array(wqms_ids, dtype='U50')},
                attrs={'long_name': 'Country name'}
            )
            ds_wq['hydrobasin_level12'] = xr.DataArray(
                wqms_hydrobasins,
                dims=['wqms_id'],
                coords={'wqms_id': np.array(wqms_ids, dtype='U50')},
                attrs={'long_name': 'HydroBasin Level 12 ID'}
            )
            ds_wq['merged_LINKNO'] = xr.DataArray(
                wqms_merged_linknos,
                dims=['wqms_id'],
                coords={'wqms_id': np.array(wqms_ids, dtype='U50')},
                attrs={'long_name': 'Merged LINKNO'}
            )
            encoding = {
                param_name: {'chunks': (ZARR_CHUNKS['wqms_id'], ZARR_CHUNKS['time'])},
                'wqms_lat': {'chunks': (ZARR_CHUNKS['wqms_id'],)},
                'wqms_lon': {'chunks': (ZARR_CHUNKS['wqms_id'],)},
                'country_name': {'chunks': (ZARR_CHUNKS['wqms_id'],)},
                'hydrobasin_level12': {'chunks': (ZARR_CHUNKS['wqms_id'],)},
                'merged_LINKNO': {'chunks': (ZARR_CHUNKS['wqms_id'],)}
            }
        else:
            encoding = {param_name: {'chunks': (ZARR_CHUNKS['wqms_id'], ZARR_CHUNKS['time'])}}
        
        ds_wq.to_zarr(output_zarr_dir, mode='a', encoding=encoding, consolidated=False, zarr_format=2)
        del ds_wq
    
    #Create catchment attributes arrays (HydroATLAS + GeoGLOWS)
    print(f"  Writing catchment attributes (HydroATLAS + GeoGLOWS)...")
    
    attr_data_vars = {}
    for col in df_attrs_clean.columns:
        if col == 'LINKNO':
            continue
        
        values = df_attrs_clean[col].values
        
        # Determine dtype
        if df_attrs_clean[col].dtype == 'object' or df_attrs_clean[col].dtype.name.startswith('string'):
            max_len = df_attrs_clean[col].astype(str).str.len().max()
            dtype = f'U{max(max_len, 10)}'
        elif df_attrs_clean[col].dtype in ['int64', 'int32', 'Int64']:
            dtype = 'i4'
        else:
            dtype = 'f4'
        
        # Store all attributes without prefix
        attr_data_vars[col] = xr.DataArray(
            values.astype(dtype),
            dims=['LINKNO'],
            coords={'LINKNO': linkno_values},
            attrs={'long_name': col}
        )
    
    ds_attrs = xr.Dataset(attr_data_vars)
    encoding_attrs = {var: {'chunks': (len(linkno_values),)} for var in attr_data_vars.keys()}
    ds_attrs.to_zarr(output_zarr_dir, mode='a', encoding=encoding_attrs, consolidated=False, zarr_format=2)
    del ds_attrs
    
    #Create weather variables (indexed by LINKNO, time)
    print(f"  Writing {len(weather_vars)} weather variables...")
    
    for idx, var_name in enumerate(weather_vars):
        print(f"    - {var_name} ({idx+1}/{len(weather_vars)})")
        
        ds_weather = xr.Dataset({
            var_name: xr.DataArray(
                np.full((n_linkno, n_time), np.nan, dtype='f4'),
                dims=['LINKNO', 'time'],
                coords={
                    'LINKNO': linkno_values,
                    'time': time_coord
                },
                attrs={'long_name': var_name, 'source': 'ERA5-Land'}
            )
        })
        
        encoding = {var_name: {'chunks': (ZARR_CHUNKS['LINKNO'], ZARR_CHUNKS['time'])}}
        ds_weather.to_zarr(output_zarr_dir, mode='a', encoding=encoding, consolidated=False, zarr_format=2)
        del ds_weather
    
    print(f"  Zarr store initialised:")
    print(f"    - {n_gauges} gauges")
    print(f"    - {n_wqms} WQMS stations")
    print(f"    - {n_linkno} LINKNOs")
    print(f"    - {n_time} time steps")
    print(f"    - {len(wq_params)} water quality parameters")
    print(f"    - {len(attr_data_vars)} catchment attributes")
    print(f"    - {len(weather_vars)} weather variables")
    print()

###---------------------------------------------------###
###                 Process Data                      ###
###---------------------------------------------------###

def process_streamflow_data(output_zarr_dir, gauge_ids, netcdf_file_map, gauge_meta_dict, start_date, n_time):
    """
    Process streamflow data from NetCDF files into Zarr store.
    """
    print(f"Processing streamflow data for {len(gauge_ids)} gauges...")
    
    z = zarr.open_group(output_zarr_dir, mode='r+')
    streamflow_array = z['streamflow']
    
    chunk_size = ZARR_CHUNKS['gauge_id']
    n_chunks = (len(gauge_ids) + chunk_size - 1) // chunk_size
    
    processed = 0
    total_records_written = 0
    date_range_mismatches = 0
    missing_area = 0
    overlap_but_no_data = 0
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(gauge_ids))
        chunk_gauges = gauge_ids[start_idx:end_idx]
        
        #prepare array
        chunk_data = np.full((len(chunk_gauges), n_time), np.nan, dtype='f4')
        
        #populate with streamflow data
        for i, gauge_id in enumerate(chunk_gauges):
            nc_path = netcdf_file_map.get(gauge_id.lower())
            if nc_path is None:
                continue
            
            #get catchment area for conversion
            gauge_meta = gauge_meta_dict.get(gauge_id)
            if gauge_meta is None or 'area' not in gauge_meta or pd.isna(gauge_meta['area']):
                missing_area += 1
                continue
            
            area_km2 = float(gauge_meta['area'])
            conversion_factor = area_km2 * 1000.0 / 86400.0
            
            try:
                with Dataset(nc_path, 'r') as ds:
                    if 'streamflow' not in ds.variables:
                        continue
                    
                    date_var = ds.variables['date']
                    dates_raw = date_var[:]
                    
                    if hasattr(date_var, 'units'):
                        time_units = date_var.units
                        calendar = getattr(date_var, 'calendar', 'standard')
                        
                        from netCDF4 import num2date
                        dates_nc = num2date(dates_raw, units=time_units, calendar=calendar, 
                                          only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                        dates_nc = pd.to_datetime(dates_nc)
                        
                    else:
                        dates_nc = pd.to_datetime(dates_raw, errors='coerce')
                    
                    streamflow = ds.variables['streamflow'][:]
                    
                    valid_dates_mask = ~pd.isna(dates_nc)
                    if not valid_dates_mask.any():
                        continue
                    
                    dates_nc_valid = dates_nc[valid_dates_mask]
                    streamflow_valid = streamflow[valid_dates_mask]
                    
                    nc_start = dates_nc_valid.min().date()
                    nc_end = dates_nc_valid.max().date()
                    
                    output_start = start_date
                    output_end = start_date + pd.Timedelta(days=n_time-1)
                    
                    if nc_end < output_start or nc_start > output_end:
                        date_range_mismatches += 1
                        continue
                    
                    records_written = 0
                    for j, (date_nc, sf_value) in enumerate(zip(dates_nc_valid, streamflow_valid)):
                        date_val = date_nc.date()
                        days_from_start = (date_val - output_start).days
                        
                        if 0 <= days_from_start < n_time:
                            if hasattr(sf_value, 'mask'):
                                if not sf_value.mask:
                                    sf_m3s = float(sf_value) * conversion_factor
                                    chunk_data[i, days_from_start] = sf_m3s
                                    records_written += 1
                            elif not np.isnan(sf_value):
                                sf_m3s = sf_value * conversion_factor
                                chunk_data[i, days_from_start] = sf_m3s
                                records_written += 1
                    
                    if records_written > 0:
                        total_records_written += records_written
                        processed += 1
                    else:
                        overlap_but_no_data += 1
            
            except Exception as e:
                print(f"    Error processing {gauge_id}: {e}")
                continue
        
        streamflow_array[start_idx:end_idx, :] = chunk_data
        pct = ((chunk_idx + 1) / n_chunks) * 100
        print(f"  [{chunk_idx+1:3d}/{n_chunks}] {pct:5.1f}% | Success: {processed}")
    
    pct = (processed / len(gauge_ids)) * 100 if len(gauge_ids) > 0 else 0
    print(f"Finished processing streamflow data ({pct:.1f}% success rate, {total_records_written:,} records written)")
    if date_range_mismatches > 0:
        print(f"  Note: {date_range_mismatches} gauges skipped due to no date overlap")
    if overlap_but_no_data > 0:
        print(f"  Note: {overlap_but_no_data} gauges had date overlap but no valid data in target range")
    if missing_area > 0:
        print(f"  Note: {missing_area} gauges skipped due to missing catchment area")
    print()


def process_single_wq_parameter(param_name, output_zarr_dir, wqms_ids, csv_dir, 
                                start_date, n_time, chunk_size, sites_to_process=None):
    """Process a single water quality parameter and write to Zarr store."""
    
    csv_path = os.path.join(csv_dir, f"{param_name}.csv")
    wqms_to_idx = {w: i for i, w in enumerate(wqms_ids)}
    
    try:
        df_param = pd.read_csv(csv_path, parse_dates=["dates"], date_format="%Y-%m-%d")
        df_param["dates"] = pd.to_datetime(df_param["dates"], errors="coerce")
        
        if sites_to_process is not None:
            df_param = df_param[df_param["wqms_id"].isin(sites_to_process)]
        
        z = zarr.open_group(output_zarr_dir, mode='r+')
        wq_array = z[param_name]
        
        grouped = df_param.groupby("wqms_id")
        
        n_chunks = (len(wqms_ids) + chunk_size - 1) // chunk_size
        stations_processed = 0
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(wqms_ids))
            chunk_wqms = wqms_ids[start_idx:end_idx]
            
            chunk_data = np.full((len(chunk_wqms), n_time), np.nan, dtype='f4')
            
            for i, wqms_id in enumerate(chunk_wqms):
                if wqms_id not in grouped.groups:
                    continue
                
                df_obs = grouped.get_group(wqms_id)
                df_obs = df_obs[df_obs["obs"].notna()]
                
                if df_obs.empty:
                    continue
                
                for _, row in df_obs.iterrows():
                    days_from_start = (row["dates"].date() - start_date).days
                    if 0 <= days_from_start < n_time:
                        chunk_data[i, days_from_start] = row["obs"]
                
                stations_processed += 1
            
            wq_array[start_idx:end_idx, :] = chunk_data
        
        return (param_name, stations_processed, True, None)
    
    except Exception as e:
        return (param_name, 0, False, str(e))


def process_wq_data(output_zarr_dir, wqms_ids, csv_dir, wq_params, start_date, n_time, sites_to_process=None):
    print(f"Processing water quality data for {len(wq_params)} parameters using {N_WORKERS} workers...")
    
    chunk_size = ZARR_CHUNKS['wqms_id']
    
    process_func = partial(process_single_wq_parameter,
                          output_zarr_dir=output_zarr_dir,
                          wqms_ids=wqms_ids,
                          csv_dir=csv_dir,
                          start_date=start_date,
                          n_time=n_time,
                          chunk_size=chunk_size,
                          sites_to_process=sites_to_process)
    
    processed = 0
    total_stations = 0
    
    with Pool(processes=N_WORKERS) as pool:
        for idx, (param_name, stations_processed, success, error) in enumerate(pool.imap(process_func, wq_params)):
            pct = ((idx + 1) / len(wq_params)) * 100
            
            if success:
                print(f"  [{idx+1:3d}/{len(wq_params)}] {pct:5.1f}% | {param_name:20s} | {stations_processed:4d} stations")
                processed += 1
                total_stations += stations_processed
            else:
                print(f"  [{idx+1:3d}/{len(wq_params)}] {pct:5.1f}% | {param_name:20s} | ERROR: {error}")
    
    pct = (processed / len(wq_params)) * 100 if len(wq_params) > 0 else 0
    print(f"Finished processing water quality data ({pct:.1f}% success rate, {total_stations} total stations)\n")


def populate_weather_data(input_weather_path, output_zarr_dir, linkno_values, weather_vars, start_date, end_date):
    
    print(f"Processing weather data for {len(weather_vars)} variables...")
    
    ds_weather = xr.open_zarr(input_weather_path)
    weather_time = pd.to_datetime(ds_weather.time.values)
    time_mask = (weather_time >= pd.to_datetime(start_date)) & (weather_time <= pd.to_datetime(end_date))
    time_indices = np.where(time_mask)[0]
    
    #Create LINKNO mapping
    weather_linkno_str = ds_weather.linkno.values.astype(str)
    target_linkno_str = linkno_values.astype(str)
    
    weather_linkno_to_idx = {linkno: idx for idx, linkno in enumerate(weather_linkno_str)}
    linkno_mapping = {}
    for out_idx, tln in enumerate(target_linkno_str):
        if tln in weather_linkno_to_idx:
            linkno_mapping[out_idx] = weather_linkno_to_idx[tln]
    
    print(f"  Matched {len(linkno_mapping)} out of {len(linkno_values)} LINKNOs")
    
    #Open output zarr for writing
    z_out = zarr.open_group(output_zarr_dir, mode='r+')
    
    #Process each weather variable
    for idx, var_name in enumerate(weather_vars):
        print(f"  [{idx+1:2d}/{len(weather_vars)}] Processing {var_name}...")
        
        var_data = ds_weather[var_name].values
        out_array = z_out[var_name]
        
        chunk_size = ZARR_CHUNKS['LINKNO']
        n_linkno_out = out_array.shape[0]
        n_chunks = (n_linkno_out + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_linkno_out)
            
            chunk_data = np.full((end_idx - start_idx, len(time_indices)), np.nan, dtype='f4')
            
            for i in range(start_idx, end_idx):
                if i in linkno_mapping:
                    input_idx = linkno_mapping[i]
                    chunk_data[i - start_idx, :] = var_data[input_idx, time_mask]
            
            out_array[start_idx:end_idx, :] = chunk_data
    
    print(f"Finished processing weather data\n")


###---------------------------------------------------------###
###       Enrich metadata for better station selection      ###
###---------------------------------------------------------###

def process_station_metadata(wqms_id, wqms_to_idx, output_zarr_dir, wq_vars):
    """Process metadata for a single station."""
    try:
        ds = xr.open_zarr(output_zarr_dir, consolidated=False)
        idx = wqms_to_idx[wqms_id]
        
        all_dates = []
        param_counts = {}
        param_obs = {}
        param_starts = {}
        param_ends = {}
        
        for var in wq_vars:
            param_name = var
            series = ds[var].isel(wqms_id=idx).to_pandas()
            valid_data = series[series.notna()]
            
            obs_count = len(valid_data)
            param_obs[param_name] = int(obs_count)
            param_counts[param_name] = obs_count
            
            if obs_count > 0:
                param_starts[param_name] = valid_data.index.min().strftime('%Y-%m-%d')
                param_ends[param_name] = valid_data.index.max().strftime('%Y-%m-%d')
                all_dates.extend(valid_data.index.tolist())
            else:
                param_starts[param_name] = None
                param_ends[param_name] = None
        
        total_obs = int(sum(param_counts.values()))
        
        if all_dates:
            all_dates = pd.to_datetime(all_dates)
            start_date = all_dates.min().strftime('%Y-%m-%d')
            end_date = all_dates.max().strftime('%Y-%m-%d')
            observation_years = int(all_dates.year.nunique())
        else:
            start_date = None
            end_date = None
            observation_years = 0
        
        parameters_measured = int(sum(1 for count in param_counts.values() if count > 0))
        
        if param_counts and max(param_counts.values()) > 0:
            most_observed_parameter = max(param_counts, key=param_counts.get)
        else:
            most_observed_parameter = None
        
        ds.close()
        
        return {
            'wqms_id': wqms_id,
            'total_observations': total_obs,
            'start_date': start_date,
            'end_date': end_date,
            'observation_years': observation_years,
            'parameters_measured': parameters_measured,
            'most_observed_parameter': most_observed_parameter,
            'param_obs': param_obs,
            'param_starts': param_starts,
            'param_ends': param_ends
        }
    
    except Exception as e:
        print(f"Error processing {wqms_id}: {e}")
        return None


def add_observation_metadata_to_linkages(output_zarr_dir, df_linkages, wqms_ids):
    """Add comprehensive observation metadata to the linkages dataframe using parallel processing."""
    print("Calculating observation metadata for each WQMS station (parallelised)...")
    
    ds = xr.open_zarr(output_zarr_dir, consolidated=False)
    wq_vars = [v for v in ds.data_vars if v not in ['streamflow', 'wqms_lat', 'wqms_lon', 'gauge_lat', 'gauge_lon', 
                                                      'country_name', 'hydrobasin_level12', 'merged_LINKNO']]
    # Exclude weather variables from WQ metadata calculation
    # Weather variables can be identified as those with LINKNO dimension
    wq_vars = [v for v in wq_vars if 'wqms_id' in ds[v].dims]
    ds.close()
    
    wqms_to_idx = {w: i for i, w in enumerate(wqms_ids)}
    total_stations = len(wqms_ids)
    
    process_func = partial(process_station_metadata,
                          wqms_to_idx=wqms_to_idx,
                          output_zarr_dir=output_zarr_dir,
                          wq_vars=wq_vars)
    
    print(f"  Using {N_WORKERS} workers to process {total_stations:,} stations...")
    
    results = []
    with Pool(processes=N_WORKERS) as pool:
        for i, result in enumerate(pool.imap(process_func, wqms_ids), 1):
            if result is not None:
                results.append(result)
            
            if i % 500 == 0 or i == total_stations:
                pct = (i / total_stations) * 100
                print(f"  Progress: {i:,}/{total_stations:,} ({pct:.1f}%)")
    
    print(f"  Successfully processed {len(results):,} stations")
    
    total_obs_dict = {r['wqms_id']: r['total_observations'] for r in results}
    start_date_dict = {r['wqms_id']: r['start_date'] for r in results}
    end_date_dict = {r['wqms_id']: r['end_date'] for r in results}
    observation_years_dict = {r['wqms_id']: r['observation_years'] for r in results}
    parameters_measured_dict = {r['wqms_id']: r['parameters_measured'] for r in results}
    most_observed_parameter_dict = {r['wqms_id']: r['most_observed_parameter'] for r in results}
    
    param_obs_dict = {param: {} for param in wq_vars}
    param_start_dict = {param: {} for param in wq_vars}
    param_end_dict = {param: {} for param in wq_vars}
    
    for r in results:
        for param_name, obs_count in r['param_obs'].items():
            param_obs_dict[param_name][r['wqms_id']] = obs_count
        for param_name, start_date in r['param_starts'].items():
            param_start_dict[param_name][r['wqms_id']] = start_date
        for param_name, end_date in r['param_ends'].items():
            param_end_dict[param_name][r['wqms_id']] = end_date
    
    df_linkages['total_observations'] = df_linkages['wqms_id'].map(total_obs_dict)
    df_linkages['start_date'] = df_linkages['wqms_id'].map(start_date_dict)
    df_linkages['end_date'] = df_linkages['wqms_id'].map(end_date_dict)
    df_linkages['observation_years'] = df_linkages['wqms_id'].map(observation_years_dict)
    df_linkages['parameters_measured'] = df_linkages['wqms_id'].map(parameters_measured_dict)
    df_linkages['most_observed_parameter'] = df_linkages['wqms_id'].map(most_observed_parameter_dict)
    
    param_dfs = []
    
    for param_name, obs_dict in param_obs_dict.items():
        param_dfs.append(
            pd.Series(df_linkages['wqms_id'].map(obs_dict), name=f'obs_{param_name}')
        )
    
    for param_name, start_dict in param_start_dict.items():
        param_dfs.append(
            pd.Series(df_linkages['wqms_id'].map(start_dict), name=f'start_{param_name}')
        )
    
    for param_name, end_dict in param_end_dict.items():
        param_dfs.append(
            pd.Series(df_linkages['wqms_id'].map(end_dict), name=f'end_{param_name}')
        )
    
    df_params = pd.concat(param_dfs, axis=1)
    df_linkages = pd.concat([df_linkages, df_params], axis=1)
    
    if 'wqms_lat' not in df_linkages.columns or 'wqms_lon' not in df_linkages.columns:
        ds = xr.open_zarr(output_zarr_dir, consolidated=False)
        wqms_coords = {
            wqms_id: {
                'lat': float(ds['wqms_lat'].isel(wqms_id=idx).values),
                'lon': float(ds['wqms_lon'].isel(wqms_id=idx).values)
            }
            for wqms_id, idx in wqms_to_idx.items()
        }
        df_linkages['wqms_lat'] = df_linkages['wqms_id'].map(lambda x: wqms_coords[x]['lat'])
        df_linkages['wqms_lon'] = df_linkages['wqms_id'].map(lambda x: wqms_coords[x]['lon'])
        ds.close()
    
    if 'country_name' not in df_linkages.columns or 'hydrobasin_level12' not in df_linkages.columns:
        ds = xr.open_zarr(output_zarr_dir, consolidated=False)
        if 'country_name' not in df_linkages.columns:
            country_dict = {
                wqms_id: str(ds['country_name'].isel(wqms_id=idx).values)
                for wqms_id, idx in wqms_to_idx.items()
            }
            df_linkages['country_name'] = df_linkages['wqms_id'].map(country_dict)
        
        if 'hydrobasin_level12' not in df_linkages.columns:
            hydrobasin_dict = {
                wqms_id: str(ds['hydrobasin_level12'].isel(wqms_id=idx).values)
                for wqms_id, idx in wqms_to_idx.items()
            }
            df_linkages['hydrobasin_level12'] = df_linkages['wqms_id'].map(hydrobasin_dict)
        ds.close()
    
    df_linkages['has_streamflow'] = df_linkages['gauge_id'].notna()
    
    return df_linkages


###---------------------------------------------------###
###       Linking wqms_id, LINKNO and gauge_id        ###
###---------------------------------------------------###

def create_linkages(df_linkages, output_wq_path, sites_to_process=None):
    print(f"Creating parquet file with metadata...")
    
    if sites_to_process is None:
        df_wq = df_linkages.copy()
    else:
        df_wq = df_linkages[df_linkages["wqms_id"].isin(sites_to_process)].copy()
    
    df_wq.to_parquet(output_wq_path, index=False, compression="snappy")

###---------------------------------------------------###
###                    Runner                         ###
###---------------------------------------------------###

if __name__ == "__main__":
    
    print(f"=== Caravan-WQMS Zarr Creation ===")
    
    # Load units from CSV file
    WQ_UNITS = load_wq_units(units_file_csv)
    
    # Load GeoGLOWS data
    df_geoglows = load_geoglows_data(geoglows_gdb)
    
    # Load weather variables (conditional)
    weather_vars = load_weather_variables(input_weather_zarr)
    
    #Load metadata
    netcdf_file_map = build_netcdf_file_map(caravan_timeseries_path)
    df_gauge_meta, gauge_ids = load_gauge_metadata(caravan_site_info, netcdf_file_map)
    
    #Load linkages and WQMS IDs
    df_linkages, df_attrs, wqms_ids = load_linkages_and_wqms_ids(
        site_info, catchment_attrs_csv, sites_to_process=None
    )
    
    # Merge GeoGLOWS attributes with existing catchment attributes
    df_attrs = merge_geoglows_attributes(df_attrs, df_geoglows)
    
    wq_params = get_param_names(csv_dir)
    
    #Create date range
    dates_sorted = pd.date_range(start=START_DATE, end=END_DATE, freq='D').date.tolist()
    n_time = len(dates_sorted)
    
    #Initialize zarr store with coordinates (includes WQ, streamflow, attributes, AND weather if enabled)
    initialize_zarr_store(output_zarr_dir, gauge_ids, wqms_ids, dates_sorted, wq_params, 
                         df_attrs, df_gauge_meta, df_linkages, weather_vars)
    
    #Process streamflow and water quality data
    gauge_meta_dict = df_gauge_meta.set_index("gauge_id").to_dict("index")
    process_streamflow_data(output_zarr_dir, gauge_ids, netcdf_file_map, gauge_meta_dict, START_DATE, n_time)
    process_wq_data(output_zarr_dir, wqms_ids, csv_dir, wq_params, START_DATE, n_time, sites_to_process=None)
    
    #Process weather data (conditional)
    linkno_values = df_attrs[df_attrs['LINKNO'].notna()].sort_values('LINKNO')['LINKNO'].values.astype('i4')
    populate_weather_data(input_weather_zarr, output_zarr_dir, linkno_values, weather_vars, START_DATE, END_DATE)
    
    #Process linkages parquet
    df_linkages = add_observation_metadata_to_linkages(output_zarr_dir, df_linkages, wqms_ids)
    create_linkages(df_linkages, output_wq_linkages_path, sites_to_process=None)

    
    #Consolidate zarr metadata
    zarr.consolidate_metadata(output_zarr_dir)
    
    print("Processing complete!")