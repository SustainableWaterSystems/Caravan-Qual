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

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

# Input directories
input_dir_caravan = "/gpfs/work4/0/dynql/Caravan-Qual/auxiliary/Caravan/"
caravan_timeseries_path = os.path.join(input_dir_caravan, "timeseries/netcdf/")
caravan_site_info = os.path.join(input_dir_caravan, "caravan_site_info.csv")

# Output directory
output_dir = "/gpfs/work4/0/dynql/Caravan-Qual/"
output_zarr_dir = os.path.join(output_dir, "Caravan.zarr")

# Define time range
START_DATE = pd.Timestamp('1951-01-01').date()
END_DATE = pd.Timestamp('2025-09-30').date()

# Define chunking strategy
ZARR_CHUNKS = {
    'time': 27302,
    'gauge_id': 500,
}

# Parallelization settings
N_WORKERS = min(cpu_count() - 1, 64)

###---------------------------------------------------###
###                   Functions                       ###
###---------------------------------------------------###

def build_netcdf_file_map(base_path):
    """Build a mapping of gauge_id to NetCDF file paths."""
    print(f"Building NetCDF file map from {base_path}")
    netcdf_files = glob.glob(os.path.join(base_path, "**/*.nc"), recursive=True)
    gauge_to_path = {}
    for file_path in netcdf_files:
        gauge_id = os.path.splitext(os.path.basename(file_path))[0]
        gauge_to_path[gauge_id.lower()] = file_path  # Store with lowercase key
    print(f"  Found {len(gauge_to_path)} NetCDF files\n")
    return gauge_to_path


def load_gauge_metadata(caravan_site_info_csv, netcdf_file_map):
    """Load gauge metadata and filter for valid gauges."""
    print(f"Loading gauge metadata from {caravan_site_info_csv}")
    df_gauge_meta = pd.read_csv(caravan_site_info_csv, dtype={"gauge_id": str})
    
    print(f"  Total gauges in CSV: {len(df_gauge_meta)}")
    
    # Check for required columns
    required_cols = ["gauge_id", "gauge_lat", "gauge_lon", "area"]
    missing_cols = [col for col in required_cols if col not in df_gauge_meta.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter for valid gauges with area data
    df_valid = df_gauge_meta[df_gauge_meta['area'].notna()].copy()
    print(f"  Gauges with area: {len(df_valid)}")
    
    # Match using lowercase comparison
    gauge_ids = [gid for gid in df_valid['gauge_id'].values if gid.lower() in netcdf_file_map]
    gauge_ids_sorted = sorted(gauge_ids)
    
    print(f"  Gauges with NetCDF files: {len(gauge_ids_sorted)}\n")
    return df_gauge_meta, gauge_ids_sorted


def scan_netcdf_variables(netcdf_file_map, gauge_ids, sample_size=10):
    """Scan a sample of NetCDF files to identify all variables."""
    print(f"Scanning NetCDF files to identify variables (sampling {sample_size} files)...")
    
    all_variables = set()
    sample_gauges = gauge_ids[:sample_size] if len(gauge_ids) > sample_size else gauge_ids
    
    for gauge_id in sample_gauges:
        nc_path = netcdf_file_map[gauge_id.lower()]
        try:
            with Dataset(nc_path, 'r') as ds:
                for var_name in ds.variables.keys():
                    if var_name not in ['date', 'time']:
                        all_variables.add(var_name)
        except Exception as e:
            print(f"  Warning: Error reading {gauge_id}: {e}")
    
    variables = sorted(list(all_variables))
    print(f"  Found {len(variables)} variables: {variables}\n")
    return variables


def get_variable_attributes(netcdf_file_map, gauge_ids, variables):
    """Extract variable attributes (units, long_name) from NetCDF files."""
    print("Extracting variable attributes from NetCDF files...")
    
    var_attrs = {var: {'units': 'unknown', 'long_name': var} for var in variables}
    
    # Sample a few files to get attributes
    sample_gauges = gauge_ids[:5]
    
    for gauge_id in sample_gauges:
        nc_path = netcdf_file_map[gauge_id.lower()]
        try:
            with Dataset(nc_path, 'r') as ds:
                for var_name in variables:
                    if var_name in ds.variables:
                        var = ds.variables[var_name]
                        if hasattr(var, 'units') and var_attrs[var_name]['units'] == 'unknown':
                            var_attrs[var_name]['units'] = var.units
                        if hasattr(var, 'long_name') and var_attrs[var_name]['long_name'] == var_name:
                            var_attrs[var_name]['long_name'] = var.long_name
        except Exception as e:
            continue
    
    print("  Variable attributes extracted\n")
    return var_attrs


def initialize_zarr_store(output_zarr_dir, gauge_ids, dates_sorted, variables, 
                         var_attrs, df_gauge_meta):
    """Initialize Zarr store with all variables and coordinates."""
    print(f"Initializing Zarr store at {output_zarr_dir}")
    
    if os.path.exists(output_zarr_dir):
        print(f"  Clearing existing store...")
        shutil.rmtree(output_zarr_dir)
    
    time_coord = pd.to_datetime(dates_sorted)
    n_time = len(dates_sorted)
    n_gauges = len(gauge_ids)
    
    # Prepare gauge lat/lon arrays (aligned with gauge_ids)
    gauge_meta_dict = df_gauge_meta.set_index("gauge_id").to_dict("index")
    gauge_lats = np.array([gauge_meta_dict.get(gid, {}).get('gauge_lat', np.nan) 
                           for gid in gauge_ids], dtype='f4')
    gauge_lons = np.array([gauge_meta_dict.get(gid, {}).get('gauge_lon', np.nan) 
                           for gid in gauge_ids], dtype='f4')
    gauge_areas = np.array([gauge_meta_dict.get(gid, {}).get('area', np.nan) 
                           for gid in gauge_ids], dtype='f4')
    
    # Create dataset with coordinates
    print(f"  Creating dataset with {len(variables)} variables...")
    
    data_vars = {}
    encoding = {}
    
    # Add coordinate variables
    data_vars['gauge_lat'] = xr.DataArray(
        gauge_lats,
        dims=['gauge_id'],
        coords={'gauge_id': np.array(gauge_ids, dtype='U50')},
        attrs={'units': 'degrees_north', 'long_name': 'Gauge latitude'}
    )
    data_vars['gauge_lon'] = xr.DataArray(
        gauge_lons,
        dims=['gauge_id'],
        coords={'gauge_id': np.array(gauge_ids, dtype='U50')},
        attrs={'units': 'degrees_east', 'long_name': 'Gauge longitude'}
    )
    data_vars['area'] = xr.DataArray(
        gauge_areas,
        dims=['gauge_id'],
        coords={'gauge_id': np.array(gauge_ids, dtype='U50')},
        attrs={'units': 'km2', 'long_name': 'Catchment area'}
    )
    
    encoding['gauge_lat'] = {'chunks': (ZARR_CHUNKS['gauge_id'],)}
    encoding['gauge_lon'] = {'chunks': (ZARR_CHUNKS['gauge_id'],)}
    encoding['area'] = {'chunks': (ZARR_CHUNKS['gauge_id'],)}
    
    # Add data variables
    for var_name in variables:
        data_vars[var_name] = xr.DataArray(
            np.full((n_gauges, n_time), np.nan, dtype='f4'),
            dims=['gauge_id', 'time'],
            coords={
                'gauge_id': np.array(gauge_ids, dtype='U50'), 
                'time': time_coord
            },
            attrs={
                'units': var_attrs[var_name]['units'],
                'long_name': var_attrs[var_name]['long_name']
            }
        )
        encoding[var_name] = {'chunks': (ZARR_CHUNKS['gauge_id'], ZARR_CHUNKS['time'])}
    
    ds = xr.Dataset(data_vars)
    ds.to_zarr(output_zarr_dir, mode='w', encoding=encoding, 
               consolidated=False, zarr_format=2)
    
    print(f"  Zarr store initialized:")
    print(f"    - {n_gauges} gauges")
    print(f"    - {n_time} time steps")
    print(f"    - {len(variables)} variables")
    print()


def process_single_variable(var_info):
    """Process a single variable for all gauges"""
    var_name, netcdf_file_map, gauge_ids, start_date, n_time, output_zarr_dir = var_info
    
    z = zarr.open_group(output_zarr_dir, mode='r+')
    var_array = z[var_name]
    
    chunk_size = ZARR_CHUNKS['gauge_id']
    n_chunks = (len(gauge_ids) + chunk_size - 1) // chunk_size
    
    gauges_with_data = 0
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(gauge_ids))
        chunk_gauges = gauge_ids[start_idx:end_idx]
        
        chunk_data = np.full((len(chunk_gauges), n_time), np.nan, dtype='f4')
        
        for i, gauge_id in enumerate(chunk_gauges):
            nc_path = netcdf_file_map.get(gauge_id.lower())
            if nc_path is None:
                continue
            
            try:
                with Dataset(nc_path, 'r') as ds:
                    if var_name not in ds.variables or 'date' not in ds.variables:
                        continue
                    
                    # Parse dates
                    date_var = ds.variables['date']
                    dates_raw = date_var[:]
                    
                    if hasattr(date_var, 'units'):
                        time_units = date_var.units
                        calendar = getattr(date_var, 'calendar', 'standard')
                        from netCDF4 import num2date
                        dates_nc = num2date(dates_raw, units=time_units, calendar=calendar, 
                                          only_use_cftime_datetimes=False, 
                                          only_use_python_datetimes=True)
                        dates_nc = pd.to_datetime(dates_nc)
                    else:
                        dates_nc = pd.to_datetime(dates_raw, errors='coerce')
                    
                    var_data = ds.variables[var_name][:]
                    
                    valid_dates_mask = ~pd.isna(dates_nc)
                    if not valid_dates_mask.any():
                        continue
                    
                    dates_nc_valid = dates_nc[valid_dates_mask]
                    var_data_valid = var_data[valid_dates_mask]
                    
                    # Check date overlap
                    nc_start = dates_nc_valid.min().date()
                    nc_end = dates_nc_valid.max().date()
                    output_start = start_date
                    output_end = start_date + pd.Timedelta(days=n_time-1)
                    
                    if nc_end < output_start or nc_start > output_end:
                        continue
                    
                    # Map to output time axis
                    has_data = False
                    for j, (date_nc, value) in enumerate(zip(dates_nc_valid, var_data_valid)):
                        date_val = date_nc.date()
                        days_from_start = (date_val - output_start).days
                        
                        if 0 <= days_from_start < n_time:
                            if hasattr(value, 'mask'):
                                if not value.mask:
                                    chunk_data[i, days_from_start] = float(value)
                                    has_data = True
                            elif not np.isnan(value):
                                chunk_data[i, days_from_start] = float(value)
                                has_data = True
                    
                    if has_data:
                        gauges_with_data += 1
            
            except Exception as e:
                continue
        
        var_array[start_idx:end_idx, :] = chunk_data
    
    return (var_name, gauges_with_data)


def populate_zarr_data(output_zarr_dir, gauge_ids, netcdf_file_map, 
                       variables, start_date, n_time):
    """Populate Zarr store by processing each variable in parallel"""
    print(f"Processing {len(variables)} variables for {len(gauge_ids)} gauges using {N_WORKERS} workers...")
    
    var_infos = [
        (var_name, netcdf_file_map, gauge_ids, start_date, n_time, output_zarr_dir)
        for var_name in variables
    ]
    
    with Pool(processes=N_WORKERS) as pool:
        for idx, (var_name, gauges_with_data) in enumerate(pool.imap_unordered(process_single_variable, var_infos)):
            pct = ((idx + 1) / len(variables)) * 100
            print(f"  [{idx+1:3d}/{len(variables)}] {pct:5.1f}% | {var_name:30s} | {gauges_with_data:,} gauges with data")
    
    print(f"Finished processing all gauge data\n")


###---------------------------------------------------###
###                    Runner                         ###
###---------------------------------------------------###

if __name__ == "__main__":
    
    print(f"=== Caravan NetCDF to Zarr Conversion ===\n")
    
    # Build file map and load metadata
    netcdf_file_map = build_netcdf_file_map(caravan_timeseries_path)
    df_gauge_meta, gauge_ids = load_gauge_metadata(caravan_site_info, netcdf_file_map)
    
    # Scan NetCDF files to identify variables
    variables = scan_netcdf_variables(netcdf_file_map, gauge_ids, sample_size=20)
    var_attrs = get_variable_attributes(netcdf_file_map, gauge_ids, variables)
    
    # Create date range
    dates_sorted = pd.date_range(start=START_DATE, end=END_DATE, freq='D').date.tolist()
    n_time = len(dates_sorted)
    
    # Initialize zarr store
    initialize_zarr_store(output_zarr_dir, gauge_ids, dates_sorted, variables, 
                         var_attrs, df_gauge_meta)
    
    # Populate zarr with data
    populate_zarr_data(output_zarr_dir, gauge_ids, netcdf_file_map, 
                      variables, START_DATE, n_time)
    
    # Consolidate zarr metadata
    print("Consolidating Zarr metadata...")
    zarr.consolidate_metadata(output_zarr_dir)
    
    print("Conversion complete!")