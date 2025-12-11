#!/usr/bin/env python3
import pandas as pd
import numpy as np
import geopandas as gpd
from netCDF4 import Dataset, num2date, date2num
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent))
from config import Config

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                   Functions                       ###
###---------------------------------------------------###

def create_netcdf_timeseries(dates, streamflow, output_path, station_id, 
                            time_units, calendar, dataset_name="Caravan"):

    time_vals = date2num(dates.to_list() if hasattr(dates, 'to_list') else dates,
                        units=time_units, calendar="standard").astype("int32")
    
    with Dataset(output_path, "w", format="NETCDF4") as nc:
        #Create dimensions
        nc.createDimension("date", len(dates))
        
        #Create time variable
        time_var = nc.createVariable("date", "i4", ("date",))
        time_var.units = time_units
        time_var.calendar = calendar
        time_var[:] = time_vals
        
        #Create streamflow variable
        sf_var = nc.createVariable("streamflow", "f4", ("date",), fill_value=np.nan)
        sf_var[:] = streamflow.astype("float32")
        sf_var.units = "mm/day"
        sf_var.long_name = "Streamflow"
        sf_var.standard_name = "water_volume_transport_in_river_channel"
        
        #Global attributes
        nc.title = f"Streamflow for station {station_id}"
        nc.institution = dataset_name
        nc.source = f"{dataset_name} dataset"
        nc.Conventions = "CF-1.6"
        nc.creation_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

def process_grdc():
    print("Processing GRDC extension")
    
    input_dir = Config.TIMESERIES_DIR / "grdc_raw"
    output_dir = Config.TIMESERIES_DIR / "grdc"
    
    output_dir.mkdir(exist_ok=True)
    nc_files = list(input_dir.glob("*.nc"))
    
    print(f"Found {len(nc_files)} files to process")
    success = 0
    
    for nc_file in nc_files:
        try:
            with Dataset(nc_file, "r") as src:
                if "streamflow" not in src.variables:
                    continue
                
                #get dates
                date_var = src.variables["date"]
                orig_units = getattr(date_var, "units", None)
                orig_calendar = getattr(date_var, "calendar", Config.NETCDF_CALENDAR)
                all_dates = num2date(date_var[:], units=orig_units, calendar=orig_calendar)
                
                #filter to fixed range
                mask = (all_dates >= Config.NETCDF_START_DATE) & (all_dates <= Config.NETCDF_END_DATE)
                if not np.any(mask):
                    continue
                
                start_idx = np.argmax(mask)
                end_idx = len(mask) - np.argmax(mask[::-1]) - 1
                
                #get streamflow valid range
                sf = src.variables["streamflow"][:]
                valid_mask = ~np.isnan(sf) & mask
                if not np.any(valid_mask):
                    continue
                
                sf_start = np.argmax(valid_mask)
                sf_end = len(valid_mask) - np.argmax(valid_mask[::-1]) - 1
                
                #create output
                output_path = output_dir / nc_file.name.replace("GRDC", "grdc")
                with Dataset(output_path, "w", format="NETCDF4") as dst:

                    for name, dim in src.dimensions.items():
                        if name == "date":
                            dst.createDimension(name, end_idx - start_idx + 1)
                        else:
                            dst.createDimension(name, len(dim) if not dim.isunlimited() else None)
                    
                    for name, var in src.variables.items():
                        fill_value = getattr(var, "_FillValue", None)
                        out_var = dst.createVariable(name, var.dtype, var.dimensions, fill_value=fill_value)
                        
                        if name == "date":
                            fixed_dates = all_dates[start_idx:end_idx+1]
                            new_vals = date2num(fixed_dates, units=Config.NETCDF_TIME_UNITS, 
                                              calendar=Config.NETCDF_CALENDAR).astype(np.int32)
                            out_var[:] = new_vals
                            out_var.units = Config.NETCDF_TIME_UNITS
                            out_var.calendar = Config.NETCDF_CALENDAR
                        elif name == "streamflow":
                            sf_out = np.full(end_idx - start_idx + 1, np.nan, dtype=var.dtype)
                            offset = sf_start - start_idx
                            sf_out[offset:offset + (sf_end - sf_start + 1)] = var[sf_start:sf_end+1]
                            out_var[:] = sf_out
                        elif var.dimensions and var.dimensions[0] == "date":
                            out_var[:] = var[start_idx:end_idx+1]
                        else:
                            out_var[:] = var[:]
                        
                        for attr in var.ncattrs():
                            if attr != "_FillValue":
                                setattr(out_var, attr, getattr(var, attr))
                    
                    for attr in src.ncattrs():
                        setattr(dst, attr, getattr(src, attr))
                
                success += 1
        
        except Exception as e:
            print(f"Error with {nc_file.name}: {e}")
            continue                


def process_camelsfr():
    print("Processing CAMELS-FR extension")
    
    if not Config.TEMP_CAMELSFR_DIR.exists():
        print(f"Directory not found: {Config.TEMP_CAMELSFR_DIR}")
        print("Download from: https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/WH7FJR\n")
        return
    
    Config.create_dataset_dirs("camelsfr")
    
    #Extract shapefile
    gpkg_path = Config.TEMP_CAMELSFR_DIR / "CAMELS_FR_geography" / "CAMELS_FR_catchment_boundaries.gpkg"
    output_shp = Config.SHAPEFILES_DIR / "camelsfr" / "camelsfr_basin_shapes.shp"
    gdf = gpd.read_file(gpkg_path)
    gdf.to_file(output_shp)
    
    #Create attributes CSV
    input_csv = Config.TEMP_CAMELSFR_DIR / "CAMELS_FR_attributes" / "static_attributes" / "CAMELS_FR_station_general_attributes.csv"
    output_csv = Config.ATTRIBUTES_DIR / "camelsfr" / "attributes_other_camelsfr.csv"
    
    df = pd.read_csv(input_csv, sep=";")
    output_df = pd.DataFrame({
        'gauge_id': 'camelsfr_' + df['sta_code_h3'].astype(str),
        'gauge_name': df['sta_label'],
        'country': 'France',
        'gauge_lat': df['sta_y_w84'],
        'gauge_lon': df['sta_x_w84'],
        'area': df['sta_area_snap']
    })
    output_df.to_csv(output_csv, index=False)
    
    #Process timeseries
    in_dir = Config.TEMP_CAMELSFR_DIR / "CAMELS_FR_time_series" / "daily"
    out_dir = Config.TIMESERIES_DIR / "camelsfr"
    
    processed = 0
    for csv_file in in_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, sep=";", comment="#")
            if "tsd_date" not in df.columns or "tsd_q_mm" not in df.columns:
                continue
            
            df["date"] = pd.to_datetime(df["tsd_date"], format="%Y%m%d", errors='coerce')
            df = df.dropna(subset=['date'])
            
            if df.empty or np.all(np.isnan(df["tsd_q_mm"])):
                continue
            
            station_id = csv_file.stem.split("_")[-1]
            nc_path = out_dir / f"camelsfr_{station_id}.nc"
            
            create_netcdf_timeseries(
                df["date"], df["tsd_q_mm"].values, nc_path, station_id,
                Config.NETCDF_TIME_UNITS, Config.NETCDF_CALENDAR, "CAMELS-FR"
            )
            processed += 1
            
        except Exception as e:
            print(f"Error with {csv_file.name}: {e}")


def process_camelsind():
    print("Processing CAMELS-IND extension")
    
    Config.create_dataset_dirs("camelsind")
    
    #Process streamflow
    csv_path = Config.TEMP_CAMELSIND_DIR / "streamflow_timeseries" / "streamflow_observed.csv"
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.drop(columns=["year", "month", "day"])
    
    #Create NetCDF files
    out_dir = Config.TIMESERIES_DIR / "camelsind"
    valid_sites = []
    
    for site in df.columns:
        if site == "date":
            continue
        
        streamflow = df[site].to_numpy(dtype="float32")
        if np.all(np.isnan(streamflow)):
            continue
        
        valid_sites.append(str(site))
        nc_path = out_dir / f"camelsind_{site}.nc"
        
        create_netcdf_timeseries(
            df["date"], streamflow, nc_path, site,
            Config.NETCDF_TIME_UNITS, Config.NETCDF_CALENDAR, "CAMELS-IND"
        )
    
    print(f"  Created {len(valid_sites)} NetCDF files")
    
    #Copy shapefiles
    source_path = Config.TEMP_CAMELSIND_DIR / "shapefiles_catchment" / "merged"
    output_base = Config.SHAPEFILES_DIR / "camelsind" / "camelsind_basin_shapes"
    
    if source_path.exists():
        source_base = source_path / "all_catchments"
        copied = 0
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            source_file = source_base.with_suffix(ext)
            if source_file.exists():
                shutil.copy2(source_file, output_base.with_suffix(ext))
                copied += 1
    
    #Create attributes CSV
    topo_csv = Config.TEMP_CAMELSIND_DIR / "attributes_csv" / "camels_ind_topo.csv"
    name_csv = Config.TEMP_CAMELSIND_DIR / "attributes_csv" / "camels_ind_name.csv"
    output_csv = Config.ATTRIBUTES_DIR / "camelsind" / "attributes_other_camelsind.csv"
    
    topo_df = pd.read_csv(topo_csv)
    name_df = pd.read_csv(name_csv)
    
    combined = pd.merge(
        topo_df[["gauge_id", "cwc_lat", "cwc_lon", "cwc_area"]],
        name_df[["gauge_id", "cwc_site_name"]],
        on="gauge_id", how="outer"
    )
    
    combined["country"] = "India"
    combined["gauge_id"] = "camelsind_" + combined["gauge_id"].astype(str)
    combined = combined.rename(columns={
        "cwc_lat": "gauge_lat",
        "cwc_lon": "gauge_lon",
        "cwc_area": "area",
        "cwc_site_name": "gauge_name"
    })
    
    #Filter to valid sites
    valid_prefixed = ["camelsind_" + str(s) for s in valid_sites]
    combined = combined[combined["gauge_id"].isin(valid_prefixed)]
    combined = combined[["gauge_id", "gauge_name", "country", "gauge_lat", "gauge_lon", "area"]]
    combined.to_csv(output_csv, index=False)
      

###---------------------------------------------------###
###                   Main                            ###
###---------------------------------------------------###

def main():
    process_grdc()
    process_camelsfr()
    process_camelsind()
    
if __name__ == "__main__":
    main()