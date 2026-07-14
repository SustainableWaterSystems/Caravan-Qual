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
        #create dimensions
        nc.createDimension("date", len(dates))
        
        #create time variable
        time_var = nc.createVariable("date", "i4", ("date",))
        time_var.units = time_units
        time_var.calendar = calendar
        time_var[:] = time_vals
        
        #create streamflow variable
        sf_var = nc.createVariable("streamflow", "f4", ("date",), fill_value=np.nan)
        sf_var[:] = streamflow.astype("float32")
        sf_var.units = "mm/day"
        sf_var.long_name = "Streamflow"
        sf_var.standard_name = "water_volume_transport_in_river_channel"
        
        #global attributes
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
    
    #extract shapefile
    gpkg_path = Config.TEMP_CAMELSFR_DIR / "CAMELS_FR_geography" / "CAMELS_FR_catchment_boundaries.gpkg"
    output_shp = Config.SHAPEFILES_DIR / "camelsfr" / "camelsfr_basin_shapes.shp"
    gdf = gpd.read_file(gpkg_path)
    gdf.to_file(output_shp)
    
    #create attributes .csv
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
    
    #process timeseries
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
    
    #load in attributes
    topo_csv = Config.TEMP_CAMELSIND_DIR / "attributes_csv" / "camels_ind_topo.csv"
    name_csv = Config.TEMP_CAMELSIND_DIR / "attributes_csv" / "camels_ind_name.csv"
    output_csv = Config.ATTRIBUTES_DIR / "camelsind" / "attributes_other_camelsind.csv"
    
    topo_df = pd.read_csv(topo_csv)
    name_df = pd.read_csv(name_csv)
    
    combined = pd.merge(
        topo_df[["gauge_id", "cwc_lat", "cwc_lon", "ghi_area"]],
        name_df[["gauge_id", "cwc_site_name"]],
        on="gauge_id", how="outer"
    )
    
    combined["country"] = "India"
    combined["gauge_id"] = "camelsind_" + combined["gauge_id"].astype(str)
    combined = combined.rename(columns={
        "cwc_lat": "gauge_lat",
        "cwc_lon": "gauge_lon",
        "ghi_area": "area",
        "cwc_site_name": "gauge_name"
    })
    
    #area lookup for unit conversion (using ghi_area, as per camelsind documentation)
    area_lookup = dict(zip(combined["gauge_id"].astype(str), combined["area"]))
    
    #process streamflow (from m3/s to mm/day)
    csv_path = Config.TEMP_CAMELSIND_DIR / "streamflow_timeseries" / "streamflow_observed.csv"
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.drop(columns=["year", "month", "day"])
    
    out_dir = Config.TIMESERIES_DIR / "camelsind"
    valid_sites = []
    
    for site in df.columns:
        if site == "date":
            continue
        
        prefixed_id = f"camelsind_{site}"
        if prefixed_id not in area_lookup:
            continue
        
        flow_m3s = df[site].to_numpy(dtype="float64")
        if np.all(np.isnan(flow_m3s)):
            continue
        
        area_km2 = area_lookup[prefixed_id]
        if pd.isna(area_km2) or area_km2 <= 0:
            print(f"  Skipping {site}: missing/invalid area for unit conversion")
            continue
        
        #apply conversion (m3/s to mm/day)
        flow_mmday = (flow_m3s * 86400 / (area_km2 * 1e6) * 1000).astype("float32")
        
        valid_sites.append(str(site))
        nc_path = out_dir / f"camelsind_{site}.nc"
        
        create_netcdf_timeseries(
            df["date"], flow_mmday, nc_path, site,
            Config.NETCDF_TIME_UNITS, Config.NETCDF_CALENDAR, "CAMELS-IND"
        )
    
    print(f"  Created {len(valid_sites)} NetCDF files")
    
    #copy shapefiles
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
    
    #filter to valid sites
    valid_prefixed = ["camelsind_" + str(s) for s in valid_sites]
    combined = combined[combined["gauge_id"].isin(valid_prefixed)]
    combined = combined[["gauge_id", "gauge_name", "country", "gauge_lat", "gauge_lon", "area"]]
    combined.to_csv(output_csv, index=False)


def process_camelsnz():
    print("Processing CAMELS-NZ extension")
    
    Config.create_dataset_dirs("camelsnz")
    
    #process attributes
    input_csv = Config.TEMP_CAMELSNZ_DIR / "1.CAMELS_NZ_Catchment_information.csv"
    output_csv = Config.ATTRIBUTES_DIR / "camelsnz" / "attributes_other_camelsnz.csv"
    
    df = pd.read_csv(input_csv)
    df_out = df.rename(columns={
        "Station_ID": "gauge_id",
        "Station Name": "gauge_name",
        "Latitude (WGS 84)": "gauge_lat",
        "Longitude(WGS 84)": "gauge_lon",
        "uparea": "area"
    })[["gauge_id", "gauge_name", "gauge_lat", "gauge_lon", "area"]]
    
    df_out["gauge_id"] = "camelsnz_" + df_out["gauge_id"].astype(str)
    df_out["country"] = "New Zealand"
    df_out = df_out[["gauge_id", "gauge_name", "country", "gauge_lat", "gauge_lon", "area"]]
    df_out.to_csv(output_csv, index=False)
    
    #build area lookup for streamflow conversion
    area_lookup = dict(zip(df_out["gauge_id"].astype(str), df_out["area"]))
    
    #copy shapefiles
    input_shp = Config.TEMP_CAMELSNZ_DIR / "camel_stationsNZ.shp"
    output_shp = Config.SHAPEFILES_DIR / "camelsnz" / "camelsnz_basin_shapes.shp"
    
    input_stem = input_shp.stem
    output_stem = output_shp.stem
    shapefile_extensions = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
    
    for ext in shapefile_extensions:
        src = input_shp.parent / f"{input_stem}{ext}"
        dst = output_shp.parent / f"{output_stem}{ext}"
        if src.exists():
            shutil.copy(src, dst)
    
    #process streamflow
    streamflow_csv_dir = Config.TEMP_CAMELSNZ_DIR
    output_nc_dir = Config.TIMESERIES_DIR / "camelsnz"
    
    for csv_file in streamflow_csv_dir.glob("daily_flow_station_id_*.csv"):
        station_id = csv_file.stem.split("_")[-1]
        prefixed_id = f"camelsnz_{station_id}"
        
        if prefixed_id not in area_lookup:
            continue
        
        try:
            df_q = pd.read_csv(csv_file)
            
            if not {"time", "flow"}.issubset(df_q.columns):
                continue
            
            if len(df_q) == 0:
                continue
            
            area_km2 = area_lookup[prefixed_id]
            dates = pd.to_datetime(df_q["time"]).dt.to_pydatetime()
            
            #Convert flow m3/s to mm/day
            flow_m3s = df_q["flow"].astype(float)
            flow_mmday = flow_m3s * 86400 / (area_km2 * 1e6) * 1000
            
            output_nc = output_nc_dir / f"camelsnz_{station_id}.nc"
            
            create_netcdf_timeseries(
                dates, flow_mmday, output_nc, station_id,
                Config.NETCDF_TIME_UNITS, Config.NETCDF_CALENDAR, "CAMELS-NZ"
            )
            
        except (pd.errors.EmptyDataError, ValueError) as e:
            print(f"Error with {csv_file.name}: {e}")
            continue

def process_camelspe():
    print("Processing CAMELS-PE extension")
    
    Config.create_dataset_dirs("camelspe")
    
    def strip_pe(raw_id):
        s = str(raw_id)
        return s[3:] if s.startswith("PE_") else s
    
    #process attributes
    stations_csv = Config.TEMP_CAMELSPE_DIR / "CAMELS-PE/01_metadata" / "stations.csv"
    topo_csv     = Config.TEMP_CAMELSPE_DIR / "CAMELS-PE/02_attributes" / "topographic_attributes.csv"
    output_csv   = Config.ATTRIBUTES_DIR / "camelspe" / "attributes_other_camelspe.csv"
    
    stations_df = pd.read_csv(stations_csv)
    topo_df     = pd.read_csv(topo_csv)
    
    stations_df["gauge_id"] = stations_df["gauge_id"].apply(strip_pe)
    topo_df["gauge_id"]     = topo_df["gauge_id"].apply(strip_pe)
    
    combined = pd.merge(
        stations_df[["gauge_id", "gauge_name", "gauge_lat", "gauge_lon"]],
        topo_df[["gauge_id", "area"]],
        on="gauge_id", how="outer"
    )
    
    combined["country"]  = "Peru"
    combined["gauge_id"] = "camelspe_" + combined["gauge_id"].astype(str)
    combined = combined[["gauge_id", "gauge_name", "country", "gauge_lat", "gauge_lon", "area"]]
    
    #process streamflow (already in mm/day)
    timeseries_dir = Config.TEMP_CAMELSPE_DIR / "CAMELS-PE/03_timeseries" / "by_catchment"
    out_dir = Config.TIMESERIES_DIR / "camelspe"
    valid_sites = []
    
    for csv_file in sorted(timeseries_dir.glob("PE_*.csv")):
        site = strip_pe(csv_file.stem)   # "PE_110139" -> "110139"
        
        try:
            df_q = pd.read_csv(csv_file)
            
            if not {"date", "flow_obs"}.issubset(df_q.columns):
                continue
            
            if len(df_q) == 0:
                continue
            
            dates = pd.to_datetime(df_q["date"])
            flow_mmday = df_q["flow_obs"].astype("float32")
            
            if np.all(np.isnan(flow_mmday)):
                continue
            
            valid_sites.append(site)
            nc_path = out_dir / f"camelspe_{site}.nc"
            
            create_netcdf_timeseries(
                dates, flow_mmday, nc_path, site,
                Config.NETCDF_TIME_UNITS, Config.NETCDF_CALENDAR, "CAMELS-PE"
            )
            
        except (pd.errors.EmptyDataError, ValueError) as e:
            print(f"Error with {csv_file.name}: {e}")
            continue
    
    print(f"  Created {len(valid_sites)} NetCDF files")
    
    #shapefiles
    catchments_gpkg = Config.TEMP_CAMELSPE_DIR / "CAMELS-PE/04_geospatial" / "camels_pe_catchments.gpkg"
    output_shp      = Config.SHAPEFILES_DIR / "camelspe" / "camelspe_basin_shapes.shp"

    if catchments_gpkg.exists():
        gdf = gpd.read_file(catchments_gpkg)
        gdf["gauge_id"] = gdf["gauge_id"].apply(strip_pe)
        gdf["gauge_id"] = "camelspe_" + gdf["gauge_id"].astype(str)
        
        valid_prefixed = ["camelspe_" + str(s) for s in valid_sites]
        gdf = gdf[gdf["gauge_id"].isin(valid_prefixed)]
        gdf = gdf[["gauge_id", "name", "area", "geometry"]]
        
        output_shp.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shp)
        print(f"  Saved {len(gdf)} catchment boundaries to {output_shp}")
    else:
        print(f"  Warning: {catchments_gpkg} not found")
    
    #filter and save
    valid_prefixed = ["camelspe_" + str(s) for s in valid_sites]
    combined = combined[combined["gauge_id"].isin(valid_prefixed)]
    combined.to_csv(output_csv, index=False)

def process_camelspl():
    print("Processing CAMELS-PL extension")
    
    Config.create_dataset_dirs("camelspl")
    
    ENCODING = "utf-8"
    
    #attributes
    topo_csv   = Config.TEMP_CAMELSPL_DIR / "CAMELS-PL/CAMELS_PL_topographic_attributes.csv"
    output_csv = Config.ATTRIBUTES_DIR / "camelspl" / "attributes_other_camelspl.csv"
    
    topo_df = pd.read_csv(topo_csv, encoding=ENCODING, dtype={"gauge_id": str},
                          sep=None, engine="python")
    
    #NOTE: gauge_lon/gauge_lat columns in the source file appear incorrectly labelled?
    topo_df = topo_df.rename(columns={"gauge_lon": "gauge_lat", "gauge_lat": "gauge_lon"})
    
    combined = topo_df.rename(columns={"area_metadata": "area"})[
        ["gauge_id", "gauge_name", "gauge_lat", "gauge_lon", "area"]
    ].copy()
    
    combined["country"]  = "Poland"
    combined["gauge_id"] = "camelspl_" + combined["gauge_id"]
    combined = combined[["gauge_id", "gauge_name", "country", "gauge_lat", "gauge_lon", "area"]]
    
    #streamflow (already mm/day)
    timeseries_dir = Config.TEMP_CAMELSPL_DIR / "CAMELS-PL" / "timeseries"
    out_dir = Config.TIMESERIES_DIR / "camelspl"
    valid_sites = []
    
    for csv_file in sorted(timeseries_dir.glob("CAMELS_PL_hydromet_timeseries_*.csv")):
        site = csv_file.stem.replace("CAMELS_PL_hydromet_timeseries_", "")
        
        try:
            df_q = pd.read_csv(csv_file, encoding=ENCODING, sep=None, engine="python")
            
            if not {"date", "discharge_spec_obs"}.issubset(df_q.columns):
                continue
            
            if len(df_q) == 0:
                continue
            
            dates = pd.to_datetime(df_q["date"], format="%Y-%m-%d")
            flow_mmday = df_q["discharge_spec_obs"].astype("float32")
            
            if np.all(np.isnan(flow_mmday)):
                continue
            
            valid_sites.append(site)
            nc_path = out_dir / f"camelspl_{site}.nc"
            
            create_netcdf_timeseries(
                dates, flow_mmday, nc_path, site,
                Config.NETCDF_TIME_UNITS, Config.NETCDF_CALENDAR, "CAMELS-PL"
            )
            
        except (pd.errors.EmptyDataError, ValueError) as e:
            print(f"Error with {csv_file.name}: {e}")
            continue
    
    print(f"  Created {len(valid_sites)} NetCDF files")
    
    #catchment shapefiles; reprojected from EPSG:2180 to WGS84
    catchments_shp = (Config.TEMP_CAMELSPL_DIR / "CAMELS-PL/CAMELS_PL_catchment_boundaries"
                      / "catchments" / "CAMELS_PL_catchments.shp")
    output_shp     = Config.SHAPEFILES_DIR / "camelspl" / "camelspl_basin_shapes.shp"
    
    if catchments_shp.exists():
        gdf = gpd.read_file(catchments_shp)
        gdf["gauge_id"] = gdf["gauge_id"].astype(str)
        gdf = gdf.to_crs(epsg=4326)
        gdf["gauge_id"] = "camelspl_" + gdf["gauge_id"]
        
        valid_prefixed = ["camelspl_" + s for s in valid_sites]
        gdf = gdf[gdf["gauge_id"].isin(valid_prefixed)]
        
        gdf = gdf[["gauge_id", "geometry"]]
        
        output_shp.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shp)
        print(f"  Saved {len(gdf)} catchment boundaries to {output_shp}")
    else:
        print(f"  Warning: {catchments_shp} not found")
    
    #filter and save
    valid_prefixed = ["camelspl_" + s for s in valid_sites]
    combined = combined[combined["gauge_id"].isin(valid_prefixed)]
    combined.to_csv(output_csv, index=False)

def process_camelslux():
    print("Processing CAMELS-LUX extension")
    
    Config.create_dataset_dirs("camelslux")
    
    ENCODING = "utf-8"
    
    def normalize_id(raw_id):
        """'ID_01' -> '1', or a bare int/str '01' -> '1'. Strips both the ID_prefix and any zero-padding so CSV-derived and shapefile-derived (stored as Integer) gauge_ids match consistently."""
        s = str(raw_id)
        if s.startswith("ID_"):
            s = s[3:]
        return str(int(s))
    
    #attributes
    meta_csv   = Config.TEMP_CAMELSLUX_DIR / "CAMELS_LUX_meta_attributes.csv"
    output_csv = Config.ATTRIBUTES_DIR / "camelslux" / "attributes_other_camelslux.csv"
    
    meta_df = pd.read_csv(meta_csv, encoding=ENCODING, dtype={"gauge_id": str},
                          sep=None, engine="python")
    
    meta_df["gauge_id"] = meta_df["gauge_id"].apply(normalize_id)
    
    combined = meta_df.rename(columns={
        "Station":  "gauge_name",
        "Lat":      "gauge_lat",
        "Lon":      "gauge_lon",
        "area_km2": "area",
    })[["gauge_id", "gauge_name", "gauge_lat", "gauge_lon", "area"]].copy()
    
    combined["country"]  = "Luxembourg"
    combined["gauge_id"] = "camelslux_" + combined["gauge_id"]
    combined = combined[["gauge_id", "gauge_name", "country", "gauge_lat", "gauge_lon", "area"]]
    
    #streamflow (already mm/day)
    timeseries_dir = Config.TEMP_CAMELSLUX_DIR / "timeseries" / "daily"
    out_dir = Config.TIMESERIES_DIR / "camelslux"
    valid_sites = []
    
    for csv_file in sorted(timeseries_dir.glob("CAMELS_LUX_hydromet_timeseries__daily_ID_*.csv")):
        raw_id = csv_file.stem.replace("CAMELS_LUX_hydromet_timeseries__daily_", "")
        site   = normalize_id(raw_id)
        
        try:
            df_q = pd.read_csv(csv_file, encoding=ENCODING, sep=None, engine="python")
            
            if not {"Date", "Qspec"}.issubset(df_q.columns):
                print(f"  Skipping {csv_file.name}: expected columns not found: "
                      f"got {df_q.columns.tolist()}")
                continue
            
            if len(df_q) == 0:
                continue
            
            dates      = pd.to_datetime(df_q["Date"], format="%Y-%m-%d")
            flow_mmday = df_q["Qspec"].astype("float32")
            
            if np.all(np.isnan(flow_mmday)):
                continue
            
            valid_sites.append(site)
            nc_path = out_dir / f"camelslux_{site}.nc"
            
            create_netcdf_timeseries(
                dates, flow_mmday, nc_path, site,
                Config.NETCDF_TIME_UNITS, Config.NETCDF_CALENDAR, "CAMELS-LUX"
            )
            
        except (pd.errors.EmptyDataError, ValueError) as e:
            print(f"Error with {csv_file.name}: {e}")
            continue
    
    print(f"  Created {len(valid_sites)} NetCDF files")
    
    #catchments
    catchments_shp = Config.TEMP_CAMELSLUX_DIR / "catchments_CAMELS-LUX.shp"
    output_shp     = Config.SHAPEFILES_DIR / "camelslux" / "camelslux_basin_shapes.shp"
    
    if catchments_shp.exists():
        gdf = gpd.read_file(catchments_shp)
        gdf["gauge_id"] = gdf["gauge_id"].apply(normalize_id)
        gdf["gauge_id"] = "camelslux_" + gdf["gauge_id"]
        
        valid_prefixed = ["camelslux_" + s for s in valid_sites]
        gdf = gdf[gdf["gauge_id"].isin(valid_prefixed)]
        
        gdf = gdf[["gauge_id", "geometry"]]
        
        output_shp.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shp)
        print(f"  Saved {len(gdf)} catchment boundaries to {output_shp}")
    else:
        print(f"  Warning: {catchments_shp} not found")
    
    #filter and save
    valid_prefixed = ["camelslux_" + s for s in valid_sites]
    combined = combined[combined["gauge_id"].isin(valid_prefixed)]
    combined.to_csv(output_csv, index=False, encoding="utf-8-sig")

def process_camelsfi():
    print("Processing CAMELS-FI extension")

    Config.create_dataset_dirs("camelsfi")

    #attributes
    meta_csv = Config.TEMP_CAMELSFI_DIR / "CAMELS-FI" / "data" / "CAMELS_FI_meta_attributes.csv"
    output_csv     = Config.ATTRIBUTES_DIR / "camelsfi" / "attributes_other_camelsfi.csv"

    meta_df = pd.read_csv(meta_csv, dtype={"gauge_id": str})

    combined = meta_df[["gauge_id", "gauge_name", "gauge_lat", "gauge_lon", "area"]].copy()
    combined["country"]  = "Finland"
    combined["gauge_id"] = "camelsfi_" + combined["gauge_id"]
    combined = combined[["gauge_id", "gauge_name", "country", "gauge_lat", "gauge_lon", "area"]]

    #streamflow (already mm/day)
    timeseries_dir = Config.TEMP_CAMELSFI_DIR / "CAMELS-FI" / "data" / "timeseries"

    out_dir = Config.TIMESERIES_DIR / "camelsfi"
    valid_sites = []

    for csv_file in sorted(timeseries_dir.glob("CAMELS_FI_hydromet_timeseries_*.csv")):
        site = csv_file.stem \
            .replace("CAMELS_FI_hydromet_timeseries_", "") \
            .rsplit("_", 1)[0]

        try:
            df_q = pd.read_csv(csv_file)

            if not {"date", "discharge_spec"}.issubset(df_q.columns):
                print(f"  Skipping {csv_file.name}: expected columns not found "
                      f"got {df_q.columns.tolist()}")
                continue

            if len(df_q) == 0:
                continue

            dates      = pd.to_datetime(df_q["date"], format="%Y-%m-%d")
            flow_mmday = df_q["discharge_spec"].astype("float32")

            if np.all(np.isnan(flow_mmday)):
                continue

            valid_sites.append(site)
            nc_path = out_dir / f"camelsfi_{site}.nc"

            create_netcdf_timeseries(
                dates, flow_mmday, nc_path, site,
                Config.NETCDF_TIME_UNITS, Config.NETCDF_CALENDAR, "CAMELS-FI"
            )

        except (pd.errors.EmptyDataError, ValueError) as e:
            print(f"Error with {csv_file.name}: {e}")
            continue

    print(f"  Created {len(valid_sites)} NetCDF files")

    #catchments
    catchments_gpkg = Config.TEMP_CAMELSFI_DIR / "CAMELS-FI" / "data" / "CAMELS_FI_catchment_boundaries.gpkg"
    output_shp      = Config.SHAPEFILES_DIR / "camelsfi" / "camelsfi_basin_shapes.shp"

    if catchments_gpkg.exists():
        gdf = gpd.read_file(catchments_gpkg, layer="catchments")
        gdf["gauge_id"] = gdf["gauge_id"].astype(str)
        gdf = gdf.to_crs(epsg=4326)
        gdf["gauge_id"] = "camelsfi_" + gdf["gauge_id"]

        valid_prefixed = ["camelsfi_" + s for s in valid_sites]
        gdf = gdf[gdf["gauge_id"].isin(valid_prefixed)]

        gdf = gdf[["gauge_id", "geometry"]]

        output_shp.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shp)
        print(f"  Saved {len(gdf)} catchment boundaries -> {output_shp}")
    else:
        print(f"  Warning: {catchments_gpkg} not found: skipping shapefile export")

    #filter and save
    valid_prefixed = ["camelsfi_" + s for s in valid_sites]
    combined = combined[combined["gauge_id"].isin(valid_prefixed)]
    combined.to_csv(output_csv, index=False, encoding="utf-8-sig")

###---------------------------------------------------###
###                   Main                            ###
###---------------------------------------------------###

def main():
    #process_grdc()
    #process_camelsfr()
    #process_camelsind()
    #process_camelsnz()
    #process_camelspe()
    #process_camelspl()
    #process_camelslux()
    process_camelsfi()
    
if __name__ == "__main__":
    main()