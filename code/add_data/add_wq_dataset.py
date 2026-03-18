# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import geopandas as gpd
import glob
from shapely.geometry import Point
from shapely import set_precision
from shapely.ops import nearest_points
from tqdm import tqdm
import warnings

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

#directories
input_dir_wqms = "/gpfs/work4/0/dynql/Caravan-Qual/"
input_dir_aux_data = "/gpfs/work4/0/dynql/Caravan-Qual/auxiliary/"
database = sys.argv[1]

#water quality data file paths
site_info_file = os.path.join(input_dir_wqms, "wqms_site_info.csv")
existing_wq_data_file = os.path.join(input_dir_aux_data, "wq_data", "combined_wqms_dataset.csv")
wq_data_folder = os.path.join(input_dir_aux_data, "wq_data", database)

#auxillary file paths
country_identifier = pd.read_csv(os.path.join(input_dir_aux_data, "WorldBank/Country_Region_Econ_ID.txt"), sep="\t", header=0, quotechar='"')
country_shp = gpd.read_file(os.path.join(input_dir_aux_data, "WorldBank/WB_countries_Admin0_10m.shp"))
hydrobasin_shp = gpd.read_file(os.path.join(input_dir_aux_data, "HydroATLAS/BasinATLAS_v10_lev12.shp"))
catchments_gpkg_dir = os.path.join(input_dir_aux_data, "geoglows_TDXhydro/catchments_gpkg")


###---------------------------------------------------###
###           Add water quality data                  ###
###---------------------------------------------------###

def load_site_info():
    """Load site info that has been prepared in a suitable way for Caravan-Qual"""

    required_columns = [
        "wqms_id",
        "wqms_lat",
        "wqms_lon",
        "country_name",
        "hydrobasin_level12",
        "LINKNO",
        "merged_LINKNO"
    ]

    if not os.path.exists(site_info_file):
        pd.DataFrame(columns=required_columns).to_csv(site_info_file, index=False)

    df = pd.read_csv(site_info_file, dtype={
        "wqms_id": str,
        "wqms_lat": float,
        "wqms_lon": float,
        "country_name": str,
        "hydrobasin_level12": str,
        "LINKNO": "Int64",
        "merged_LINKNO": "Int64"
    })

    return df

def load_wq_data(folder):
    """Load water quality data that has been prepared in a suitable way for Caravan-Qual"""

    files = glob.glob(os.path.join(folder, "*.csv"))
    df_list = []
    for f in files:
        df_list.append(pd.read_csv(f, dtype={
            'lat': float, 'lon': float, 'dates': str, 'site_id': str,
            'limit_flag': str, 'raw_obs': float, 'raw_unit': str,
            'detection_limit': float,
            'source': str, 'variable': str
        }))
    return pd.concat(df_list, ignore_index=True)

def assign_country_hydrobasin(sites_gdf, country_shp, hydrobasin_shp):
    """get country name and hydrobasin_id via spatial overlay with catchment polygons, then nearest neighbor for unmatched sites"""

    #clean geometries
    country_shp = country_shp.copy()
    hydrobasin_shp = hydrobasin_shp.copy()
    country_shp['geometry'] = country_shp['geometry'].buffer(0)
    hydrobasin_shp['geometry'] = hydrobasin_shp['geometry'].buffer(0)

    #reproject to metric CRS (distance calculations)
    sites_proj = sites_gdf.to_crs(epsg=3857)
    country_proj = country_shp.to_crs(epsg=3857)
    hydro_proj = hydrobasin_shp.to_crs(epsg=3857)

    #initialise columns
    sites_proj['country_name'] = None
    sites_proj['hydrobasin_level12'] = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        #assign country name and hydrobasin by direct spatial intersection
        sites_country = gpd.sjoin(
            sites_proj,
            country_proj[['NAME_EN', 'geometry']],
            how='left',
            predicate='intersects'
        )

        sites_country = sites_country.groupby(sites_country.index).first()
        sites_proj.loc[sites_country.index, 'country_name'] = sites_country['NAME_EN']

        sites_hydro = gpd.sjoin(
            sites_proj,
            hydro_proj[['HYBAS_ID', 'geometry']],
            how='left',
            predicate='intersects'
        )
        sites_hydro = sites_hydro.groupby(sites_hydro.index).first()
        sites_proj.loc[sites_hydro.index, 'hydrobasin_level12'] = sites_hydro['HYBAS_ID']

        #assign nearest country name and hydrobasin for unmatched sites
        unmatched_country = sites_proj[sites_proj['country_name'].isna()]
        if len(unmatched_country) > 0:
            nearest_country = gpd.sjoin_nearest(
                unmatched_country,
                country_proj[['NAME_EN', 'geometry']],
                how='left',
                distance_col=None
            )
            sites_proj.loc[nearest_country.index, 'country_name'] = nearest_country['NAME_EN']

        unmatched_hydro = sites_proj[sites_proj['hydrobasin_level12'].isna()]
        if len(unmatched_hydro) > 0:
            nearest_hydro = gpd.sjoin_nearest(
                unmatched_hydro,
                hydro_proj[['HYBAS_ID', 'geometry']],
                how='left',
                distance_col=None
            )
            sites_proj.loc[nearest_hydro.index, 'hydrobasin_level12'] = nearest_hydro['HYBAS_ID']

    return sites_proj


def build_sites_gdf(df, lon_col='lon', lat_col='lat'):
    """Convert dataframe (with coordinate columns) into a GeoDataFrame and project to EPSG:4326"""    
    
    if df[lat_col].abs().max() <= 90 and df[lon_col].abs().max() <= 180:
        crs = "EPSG:4326"
    else:
        crs = "EPSG:3857"

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs
    )

    if crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def assign_linkno_from_catchments(sites_df, catchments_gpkg_dir, lon_col='lon', lat_col='lat', show_progress=True):
    """Assign LINKNO via spatial overlay with catchment polygons, then nearest neighbor for unmatched sites"""

    sites_gdf = build_sites_gdf(sites_df, lon_col=lon_col, lat_col=lat_col)

    print(f"Getting TDX-hydro polygons (.gpkgs) from: {catchments_gpkg_dir}")
    gpkg_files = glob.glob(os.path.join(catchments_gpkg_dir, "*.gpkg"))
    
    if not gpkg_files:
        print(f"? No GPKG files found in {catchments_gpkg_dir}")
        return sites_gdf

    sites_gdf['LINKNO'] = pd.NA
    sites_proj = sites_gdf.to_crs(epsg=3857)
    
    assigned_overlap = 0
    total_sites = len(sites_gdf)

    #Assign by spatial overlap
    print("Assigning LINKNO by spatial overlap...")
    iterator = enumerate(gpkg_files)
    if show_progress:
        iterator = tqdm(iterator, total=len(gpkg_files), desc="  Overlap assignment")

    for i, gpkg_file in iterator:
        try:
            #reproject sites to catchment CRS for bbox filtering
            catchments_meta = gpd.read_file(gpkg_file, rows=1)
            catchments_crs = catchments_meta.crs

            sites_in_catch_crs = sites_gdf.to_crs(catchments_crs)
            sites_bounds = tuple(sites_in_catch_crs.total_bounds)

            #only read catchments within bbox
            catchments = gpd.read_file(gpkg_file, bbox=sites_bounds)

            linkno_col = next((col for col in ['LINKNO', 'linkno'] if col in catchments.columns), None)
            if linkno_col is None:
                continue
            
            catchments = catchments[[linkno_col, 'geometry']].rename(columns={linkno_col: 'LINKNO'})
            catchments['LINKNO'] = pd.to_numeric(catchments['LINKNO'], errors='coerce')
            catchments = catchments.dropna(subset=['LINKNO'])
            if catchments.empty:
                continue

            #reproject catchments for spatial join
            catchments = catchments.to_crs(epsg=3857)

            #select unassigned sites
            unassigned_sites = sites_proj[sites_proj['LINKNO'].isna()]
            if unassigned_sites.empty:
                break

            #spatial join by overlap
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                joined = gpd.sjoin(
                    unassigned_sites,
                    catchments[['LINKNO', 'geometry']],
                    how='left',
                    predicate='within'
                )

            if 'LINKNO_right' in joined.columns:
                matched_mask = joined['LINKNO_right'].notna()
                if matched_mask.any():
                    first_matches = joined[matched_mask].groupby(level=0).first()
                    sites_proj.loc[first_matches.index, 'LINKNO'] = first_matches['LINKNO_right']
                    sites_gdf.loc[first_matches.index, 'LINKNO'] = first_matches['LINKNO_right']
                    assigned_overlap += len(first_matches)

        except Exception as e:
            if show_progress:
                tqdm.write(f"Error processing {os.path.basename(gpkg_file)}: {str(e)}")
            continue

    print(f"  Sites assigned by overlap: {assigned_overlap}/{total_sites} ({assigned_overlap/total_sites*100:.1f}%)")

    #assign remaining sites by proximity (regional batching for memory efficiency)
    unassigned_after_overlap = sites_proj['LINKNO'].isna().sum()
    
    if unassigned_after_overlap > 0:
        print(f"Assigning {unassigned_after_overlap} remaining sites by proximity...")
        
        assigned_proximity = 0
        
        #process each gpkg file region-by-region
        iterator = enumerate(gpkg_files)
        if show_progress:
            iterator = tqdm(iterator, total=len(gpkg_files), desc="  Processing regions")

        for i, gpkg_file in iterator:
            try:
                #check for unassigned sites
                still_unassigned_wgs84 = sites_gdf[sites_gdf['LINKNO'].isna()]
                if still_unassigned_wgs84.empty:
                    break
                
                #read catchment metadata
                catchments_meta = gpd.read_file(gpkg_file, rows=1)
                catchments_crs = catchments_meta.crs

                #get unassigned sites bounds in catchment CRS
                sites_in_catch_crs = still_unassigned_wgs84.to_crs(catchments_crs)
                sites_bounds = tuple(sites_in_catch_crs.total_bounds)

                #only read catchments within bbox
                catchments = gpd.read_file(gpkg_file, bbox=sites_bounds) #only read catchments within bbox

                if catchments.empty:
                    continue

                linkno_col = next((col for col in ['LINKNO', 'linkno'] if col in catchments.columns), None)
                if linkno_col is None:
                    continue
                
                catchments = catchments[[linkno_col, 'geometry']].rename(columns={linkno_col: 'LINKNO'})
                catchments['LINKNO'] = pd.to_numeric(catchments['LINKNO'], errors='coerce')
                catchments = catchments.dropna(subset=['LINKNO'])
                
                if catchments.empty:
                    continue
                
                #peproject to metric CRS
                catchments = catchments.to_crs(epsg=3857)
                
                #get still unassigned sites in this region
                still_unassigned = sites_proj[sites_proj['LINKNO'].isna()].copy()
                
                #Filter sites to those within reasonable distance of this catchment file
                bbox_buffer = 100000  #100km in meters
                minx, miny, maxx, maxy = catchments.total_bounds
                region_bbox = (minx - bbox_buffer, miny - bbox_buffer, 
                              maxx + bbox_buffer, maxy + bbox_buffer)
                
                #filter sites to this region
                sites_in_region = still_unassigned.cx[region_bbox[0]:region_bbox[2], 
                                                       region_bbox[1]:region_bbox[3]]
                
                if sites_in_region.empty:
                    continue
                
                #nearest neighbor matching for this region
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    joined = gpd.sjoin_nearest(
                        sites_in_region,
                        catchments[['LINKNO', 'geometry']],
                        how='left',
                        distance_col='proximity_distance',
                        max_distance=bbox_buffer  # Only match within 100km
                    )
                
                if 'LINKNO_right' in joined.columns:
                    matched_mask = joined['LINKNO_right'].notna()
                    if matched_mask.any():
                        first_matches = joined[matched_mask].groupby(level=0).first()
                        
                        #only assign if not already assigned
                        to_assign = first_matches.index.intersection(sites_proj[sites_proj['LINKNO'].isna()].index)
                        
                        if len(to_assign) > 0:
                            sites_proj.loc[to_assign, 'LINKNO'] = first_matches.loc[to_assign, 'LINKNO_right']
                            sites_gdf.loc[to_assign, 'LINKNO'] = first_matches.loc[to_assign, 'LINKNO_right']
                            assigned_proximity += len(to_assign)

            except Exception as e:
                if show_progress:
                    tqdm.write(f"Error processing {os.path.basename(gpkg_file)}: {str(e)}")
                continue

        print(f"  Sites assigned by proximity: {assigned_proximity}/{unassigned_after_overlap} sites ({assigned_proximity/unassigned_after_overlap*100:.1f}%)")
        
        #report sites that are still unassigned (i.e. beyond 100km from any catchment)
        still_unassigned_final = sites_proj['LINKNO'].isna().sum()
        if still_unassigned_final > 0:
            print(f"  Warning: {still_unassigned_final} sites remain unassigned (>100km from any catchment)")
    else:
        assigned_proximity = 0
        print("  All sites assigned by overlap, no proximity assignment needed.")

    sites_gdf['LINKNO'] = pd.to_numeric(sites_gdf['LINKNO'], errors='coerce').astype('Int64') #convert LINKNO to integer

    print(f"LINKNO assignment complete:")
    print(f"  Total sites assigned: {assigned_overlap + assigned_proximity}/{total_sites} ({(assigned_overlap + assigned_proximity)/total_sites*100:.1f}%)")
    print(f"  Sites unassigned: {total_sites - assigned_overlap - assigned_proximity}")

    #ensure output coordinates are in degrees
    sites_gdf = sites_gdf.to_crs(epsg=4326)
    sites_gdf['lon'] = sites_gdf.geometry.x
    sites_gdf['lat'] = sites_gdf.geometry.y

    return sites_gdf


def snap_sites_to_rivers_within_linkno(sites_gdf, rivers, link_branch_dict, show_progress=True):
    """Snap water quality sites to the stream network (closest point within LINKNO polygon)"""
        
    #filter sites that have a LINKNO assigned
    sites_with_linkno = sites_gdf[sites_gdf['LINKNO'].notna()].copy()
    sites_without_linkno = sites_gdf[sites_gdf['LINKNO'].isna()].copy()
    
    print(f"Sites with LINKNO: {len(sites_with_linkno)}")
    print(f"Sites without LINKNO: {len(sites_without_linkno)}")
    
    if sites_with_linkno.empty:
        print("No sites have LINKNO assigned, skipping river snapping")
        return sites_gdf
    
    #reproject both to metric CRS
    sites_proj = sites_with_linkno.to_crs(epsg=3857).copy()
    rivers_proj = rivers.to_crs(epsg=3857).copy()
    
    #initialise output columns
    sites_proj['snapped_lon'] = sites_proj.geometry.x
    sites_proj['snapped_lat'] = sites_proj.geometry.y
    sites_proj['snap_distance'] = 0.0
    sites_proj['merged_LINKNO'] = pd.NA
    
    #group sites by LINKNO for efficient processing
    linkno_groups = sites_proj.groupby('LINKNO')
    
    iterator = linkno_groups
    if show_progress:
        iterator = tqdm(linkno_groups, desc="Snapping by LINKNO", total=len(linkno_groups))
    
    snapped_count = 0
    for linkno, group_sites in iterator:
        try:
            #filter rivers to only those with matching LINKNO
            matching_rivers = rivers_proj[rivers_proj['LINKNO'] == linkno].copy()
            
            if matching_rivers.empty:
                if show_progress:
                    tqdm.write(f"No rivers found for LINKNO {linkno} ({len(group_sites)} sites)")
                continue
            
            matching_rivers["river_geom"] = matching_rivers.geometry
            
            #nearest join within the identified LINKNO
            joined = gpd.sjoin_nearest(
                group_sites,
                matching_rivers[["LINKNO", "geometry", "river_geom"]],
                how="left",
                distance_col="snap_distance"
            ).reset_index().drop_duplicates(subset="index").set_index("index")
            
            #snap to nearest river segment
            for idx, row in joined.iterrows():
                site_geom = row.geometry
                river_geom = row["river_geom"]
                
                if river_geom is not None and river_geom.geom_type in ["LineString", "MultiLineString"]:
                    snapped_point = river_geom.interpolate(river_geom.project(site_geom))                    
                    snapped_point_wgs84 = gpd.GeoSeries([snapped_point], crs=3857).to_crs(epsg=4326).iloc[0]
                    
                    sites_proj.loc[idx, 'snapped_lon'] = snapped_point_wgs84.x
                    sites_proj.loc[idx, 'snapped_lat'] = snapped_point_wgs84.y
                    sites_proj.loc[idx, 'snap_distance'] = row['snap_distance']
                    
                    #assign merged_LINKNO
                    merged_linkno = link_branch_dict.get(linkno, -1)
                    sites_proj.loc[idx, 'merged_LINKNO'] = merged_linkno
                    
                    snapped_count += 1
                
        except Exception as e:
            if show_progress:
                tqdm.write(f"Error processing LINKNO {linkno}: {str(e)}")
            continue
    
    #update coordinates
    sites_gdf.loc[sites_with_linkno.index, 'wqms_lon'] = sites_proj['snapped_lon']
    sites_gdf.loc[sites_with_linkno.index, 'wqms_lat'] = sites_proj['snapped_lat']
    sites_gdf.loc[sites_with_linkno.index, 'merged_LINKNO'] = sites_proj['merged_LINKNO']
    
    #for sites without LINKNO, keep original coordinates and set merged_LINKNO as NA
    if not sites_without_linkno.empty:
        sites_gdf.loc[sites_without_linkno.index, 'wqms_lon'] = sites_without_linkno.geometry.x
        sites_gdf.loc[sites_without_linkno.index, 'wqms_lat'] = sites_without_linkno.geometry.y
        sites_gdf.loc[sites_without_linkno.index, 'merged_LINKNO'] = pd.NA
    
    #convert merged_LINKNO to Int64 to preserve NA values
    sites_gdf['merged_LINKNO'] = pd.to_numeric(sites_gdf['merged_LINKNO'], errors='coerce').astype('Int64')
    
    #drop geometry
    sites_gdf = sites_gdf.drop(columns="geometry", errors="ignore")
    
    print(f"River snapping complete:")
    print(f"Successfully snapped: {snapped_count} sites")
    if snapped_count > 0:
        print(f"Average snap distance: {sites_proj['snap_distance'].mean():.1f} meters")
    
    return sites_gdf
    

def merge_with_database(sites_gdf, wq_data, wq_database_site_info, country_identifier, coord_precision=5):
    """Merge processed data with existing data"""
    
    #create snapped_coord_id for merging (5 decimal places for approx 1m accuracy)
    for df in [wq_database_site_info, sites_gdf]:
        df['wqms_lat'] = df['wqms_lat'].round(coord_precision)
        df['wqms_lon'] = df['wqms_lon'].round(coord_precision)

    for df in [wq_database_site_info, sites_gdf]:
        df['snapped_coord_id'] = (
            df['wqms_lat'].astype(str) + "_" + df['wqms_lon'].astype(str)
        )

    existing_link_map = (
        wq_database_site_info
        .set_index('snapped_coord_id')['wqms_id']
        .to_dict()
    )
    sites_gdf['wqms_id'] = sites_gdf['snapped_coord_id'].map(existing_link_map).astype('object')

    #identify new sites (i.e. no wqms_id assigned yet)
    new_sites = sites_gdf[sites_gdf['wqms_id'].isna()].copy()
    new_sites = new_sites.drop_duplicates(subset='snapped_coord_id')
    
    if not new_sites.empty:
        #merge FID from country_identifier
        new_sites = new_sites.merge(
            country_identifier[['Country', 'FID']],
            left_on='country_name', right_on='Country',
            how='left'
        )
        
        #drop any sites without valid FID (missing country, mismatched country, etc.)
        sites_dropped = new_sites[new_sites['FID'].isna()]
        if not sites_dropped.empty:
            print(f"\nDropping {len(sites_dropped)} sites without valid FID/country assignment")
            new_sites = new_sites[new_sites['FID'].notna()]
        
        if new_sites.empty:
            print("No sites with valid FID remaining - skipping database merge")
            return wq_database_site_info, wq_data
        
        #convert FID to zero-padded string
        new_sites['FID_str'] = new_sites['FID'].astype(int).astype(str).str.zfill(3)
        
        #track last assigned number per FID in existing database
        existing_fid_max = (
            wq_database_site_info
            .merge(country_identifier[['Country', 'FID']], left_on='country_name', right_on='Country', how='left')
            .assign(FID_str=lambda df: df['FID'].astype(int).astype(str).str.zfill(3))
            .groupby('FID_str')['wqms_id']
            .apply(lambda x: x.str[-5:].astype(int).max() if not x.empty else 0)
            .to_dict()
        )
        
        #assign new wqms_id per snapped_coord_id
        assigned_link_ids = {}
        for _, row in new_sites.iterrows():
            fid = row['FID_str']
            snapped_coord_id = row['snapped_coord_id']
            next_num = existing_fid_max.get(fid, 0) + 1
            existing_fid_max[fid] = next_num
            row_num_str = str(next_num).zfill(5)
            assigned_link_ids[snapped_coord_id] = "wqms_" + fid + row_num_str
        
        #map new wqms_id back to new_sites
        new_sites['wqms_id'] = new_sites['snapped_coord_id'].map(assigned_link_ids)
        
        #drop columns
        new_sites = new_sites.drop(columns=['FID', 'Country', 'FID_str'], errors='ignore')
        
        #merge new sites into database
        new_sites = new_sites.drop_duplicates(subset='wqms_id')
        wq_database_site_info = pd.concat([wq_database_site_info, new_sites], ignore_index=True)
    
    #update sites_gdf with new wqms_ids
    if not new_sites.empty:
        new_id_mapping = new_sites.set_index('snapped_coord_id')['wqms_id'].to_dict()
        mask = sites_gdf['wqms_id'].isna()
        sites_gdf.loc[mask, 'wqms_id'] = sites_gdf.loc[mask, 'snapped_coord_id'].map(new_id_mapping)
    
    #dictionary for mapping site_ids to wqms_ids
    site_to_wqms_map = sites_gdf.set_index('site_id')['wqms_id'].to_dict()
    
    #merge wq_data using site_id
    wq_data['wqms_id'] = wq_data['site_id'].map(site_to_wqms_map).astype('object')
    
    #drop observations without valid wqms_id
    obs_dropped = wq_data['wqms_id'].isna().sum()
    if obs_dropped > 0:
        print(f"Dropping {obs_dropped} observations without valid wqms_id")
        wq_data = wq_data[wq_data['wqms_id'].notna()]
            
    #merge wqms_lat and wqms_lon 
    wq_data = wq_data.merge(
        wq_database_site_info[['wqms_id', 'wqms_lat', 'wqms_lon']],
        on='wqms_id',
        how='left'
    )
    
    #tidy database
    wq_data = wq_data[['site_id', 'lat', 'lon', 'wqms_id', 'wqms_lat', 'wqms_lon', 
                       'dates', 'limit_flag', 'raw_obs', 'raw_unit', 'detection_limit',
                       'source', 'variable']]
    
    return wq_database_site_info, wq_data


def save_results(wq_database_site_info, wq_data, input_dir_wqms, database, existing_wq_data_file):
    """Save site info results and append water quality data to raw observational data store"""

    #clean columns for site_info
    site_info_columns = [
        "wqms_id",
        "wqms_lat",
        "wqms_lon",
        "country_name",
        "hydrobasin_level12",
        "LINKNO",
        "merged_LINKNO"
    ]

    #write out the full site_info
    site_info_full = wq_database_site_info[site_info_columns]
    site_info_full.to_csv(site_info_file, index=False)
    print(f"Full site info saved to: {site_info_file}")

    #subset site_info to target database records
    unique_wqms_ids = wq_data["wqms_id"].unique()
    filtered_site_info = site_info_full[site_info_full["wqms_id"].isin(unique_wqms_ids)]

    filtered_file = os.path.join(input_dir_aux_data, f"wq_data/wqms_site_info_for_{database}.csv")
    filtered_site_info.to_csv(filtered_file, index=False)
    print(f"Filtered site info saved to: {filtered_file}")

    #append or write WQ data (i.e. processed observations)
    if os.path.exists(existing_wq_data_file):
        existing_cols = pd.read_csv(existing_wq_data_file, nrows=0).columns.tolist()
        wq_data = wq_data[existing_cols]
        wq_data.to_csv(existing_wq_data_file, mode="a", header=False, index=False)
        print(f"Appended WQ data to existing file: {existing_wq_data_file}")
    else:
        wq_data.to_csv(existing_wq_data_file, index=False)
        print(f"Created new WQ data file: {existing_wq_data_file}")

        
###---------------------------------------------------###
###                     Main                          ###
###---------------------------------------------------###

def main():
    print("Loading existing site info...")
    wq_database_site_info = load_site_info()
    
    print("Loading new WQ data...")
    wq_data = load_wq_data(wq_data_folder)
    
    print("Preparing GeoDataFrame for processing...")
    unique_sites = wq_data.drop_duplicates('site_id').copy()
    geometry = [Point(xy) for xy in zip(unique_sites['lon'], unique_sites['lat'])]
    sites_gdf = gpd.GeoDataFrame(unique_sites, geometry=geometry, crs="EPSG:4326")
    
    print("Adding country and hydrobasin info...")
    sites_gdf = assign_country_hydrobasin(sites_gdf, country_shp, hydrobasin_shp)
    
    print("Loading river network and branch mapping...")
    rivers = gpd.read_file(os.path.join(input_dir_aux_data, "geoglows_TDXhydro/global_streams_simplified.gpkg"))
    branch_map = pd.read_csv(os.path.join(input_dir_aux_data, "geoglows_TDXhydro/merged_branches.csv"))
    link_branch_map = branch_map.assign(LINKNOs=branch_map["merged_LINKNOs"].str.split(";")).explode("LINKNOs")
    link_branch_map["LINKNOs"] = link_branch_map["LINKNOs"].astype(int)
    link_branch_dict = dict(zip(link_branch_map["LINKNOs"], link_branch_map["branch_id"]))

    print("Assigning LINKNO from catchment polygons (overlap + proximity)...")
    sites_gdf = assign_linkno_from_catchments(sites_gdf, catchments_gpkg_dir, show_progress=True)
    
    print("Snapping sites to rivers within same LINKNO...")
    sites_gdf = snap_sites_to_rivers_within_linkno(sites_gdf, rivers, link_branch_dict, show_progress=True)
    
    print("Merging with existing database...")
    wq_database_site_info, wq_data = merge_with_database(
        sites_gdf, wq_data, wq_database_site_info, country_identifier
    )

    print("Saving results...")
    save_results(wq_database_site_info, wq_data, input_dir_wqms, database, existing_wq_data_file)

    print(f"Finished processing database: {database}")

if __name__ == "__main__":
    main()