#!/usr/bin/env python3
import pandas as pd
import geopandas as gpd
import glob
import sys
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point
import warnings

sys.path.insert(0, str(Path(__file__).parent))
from config import Config

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                   Functions                       ###
###---------------------------------------------------###

def combine_attribute_files(input_dir, patterns, output_files):
    """Combine attribute files matching patterns into single CSVs."""
    
    for pattern, output_file in zip(patterns, output_files):
        all_files = glob.glob(str(Path(input_dir) / "*" / pattern))
        df_list = []
        
        for f in all_files:
            try:
                df = pd.read_csv(f)
                if "gauge_id" in df.columns:
                    df["gauge_id"] = df["gauge_id"].str.lower()
                df_list.append(df)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")
        
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            print(f"Saved {output_file} with {len(combined_df)} rows")
        else:
            print(f"No files found for pattern {pattern}")


def build_geodataframe(df, lon_col='gauge_lon', lat_col='gauge_lat', crs='EPSG:4326'):
    """Build a GeoDataFrame from coordinates"""
    
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    return gdf


def assign_country_hydrobasin(sites_gdf, country_shp, hydrobasin_shp):
    """Assign country and hydrobasin IDs to gauge stations (spatial overlay, then nearest neighbour)."""
    
    #clean geometries
    country_shp = country_shp.copy()
    hydrobasin_shp = hydrobasin_shp.copy()
    country_shp['geometry'] = country_shp['geometry'].buffer(0)
    hydrobasin_shp['geometry'] = hydrobasin_shp['geometry'].buffer(0)
    
    #reproject to metric CRS
    sites_proj = sites_gdf.to_crs(epsg=3857)
    country_proj = country_shp.to_crs(epsg=3857)
    hydro_proj = hydrobasin_shp.to_crs(epsg=3857)
    
    #initialise columns
    sites_proj['country_name'] = None
    sites_proj['hydrobasin_level12'] = None
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        #spatial intersection
        sites_country = gpd.sjoin(
            sites_proj, country_proj[['NAME_EN', 'geometry']],
            how='left', predicate='intersects'
        ).groupby(level=0).first()
        sites_proj.loc[sites_country.index, 'country_name'] = sites_country['NAME_EN']
        
        sites_hydro = gpd.sjoin(
            sites_proj, hydro_proj[['HYBAS_ID', 'geometry']],
            how='left', predicate='intersects'
        ).groupby(level=0).first()
        sites_proj.loc[sites_hydro.index, 'hydrobasin_level12'] = sites_hydro['HYBAS_ID']
        
        #nearest neighbor for unmatched
        unmatched_country = sites_proj[sites_proj['country_name'].isna()]
        if len(unmatched_country) > 0:
            nearest_country = gpd.sjoin_nearest(
                unmatched_country, country_proj[['NAME_EN', 'geometry']], how='left'
            )
            sites_proj.loc[nearest_country.index, 'country_name'] = nearest_country['NAME_EN']
        
        unmatched_hydro = sites_proj[sites_proj['hydrobasin_level12'].isna()]
        if len(unmatched_hydro) > 0:
            nearest_hydro = gpd.sjoin_nearest(
                unmatched_hydro, hydro_proj[['HYBAS_ID', 'geometry']], how='left'
            )
            sites_proj.loc[nearest_hydro.index, 'hydrobasin_level12'] = nearest_hydro['HYBAS_ID']
    
    return sites_proj


def assign_linkno_from_catchments(sites_gdf, show_progress=True):
    """Assign LINKNO to gauge stations (spatial overlay [with catchment polygons], then nearest neighbor)"""
    
    sites_gdf['LINKNO'] = pd.NA
    sites_proj = sites_gdf.to_crs(epsg=3857)
    gpkg_files = list(Config.CATCHMENTS_GPKG_DIR.glob("*.gpkg"))
    
    print(f"Processing {len(gpkg_files)} catchment files")
    
    assigned_overlap = 0
    total = len(sites_gdf)
    
    #spatial overlap
    print("Assigning by spatial overlap...")
    iterator = tqdm(gpkg_files, desc="  Overlap assignment") if show_progress else gpkg_files
    
    for gpkg_file in iterator:
        try:
            #read with bbox filter
            meta = gpd.read_file(gpkg_file, rows=1)
            sites_temp = sites_gdf.to_crs(meta.crs)
            bbox = tuple(sites_temp.total_bounds)
            
            catchments = gpd.read_file(gpkg_file, bbox=bbox)
            
            #find LINKNO column
            linkno_col = next((c for c in ['LINKNO', 'linkno'] if c in catchments.columns), None)
            if linkno_col is None:
                continue
            
            catchments = catchments[[linkno_col, 'geometry']].rename(columns={linkno_col: 'LINKNO'})
            catchments['LINKNO'] = pd.to_numeric(catchments['LINKNO'], errors='coerce')
            catchments = catchments.dropna(subset=['LINKNO']).to_crs(epsg=3857)
            
            if catchments.empty:
                continue
            
            #spatial join for unassigned sites
            unassigned = sites_proj[sites_proj['LINKNO'].isna()]
            if unassigned.empty:
                break
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                joined = gpd.sjoin(
                    unassigned, 
                    catchments[['LINKNO', 'geometry']], 
                    how='left', 
                    predicate='within'
                )
            
            if 'LINKNO_right' in joined.columns:
                matched = joined['LINKNO_right'].notna()
                if matched.any():
                    first = joined[matched].groupby(level=0).first()
                    sites_proj.loc[first.index, 'LINKNO'] = first['LINKNO_right']
                    sites_gdf.loc[first.index, 'LINKNO'] = first['LINKNO_right']
                    assigned_overlap += len(first)
        
        except Exception as e:
            print(f"Error with {gpkg_file.name}: {e}")
            continue
    
    print(f"  LINKNO assigned by overlap: {assigned_overlap}/{total} sites ({assigned_overlap/total*100:.1f}%)")
    
    #assign remaining sites by proximity
    unassigned_after_overlap = sites_proj['LINKNO'].isna().sum()
    
    if unassigned_after_overlap > 0:
        print(f"Assigning {unassigned_after_overlap} remaining sites by proximity...")
        
        assigned_proximity = 0
        max_catchments_per_batch = 100000  #limit catchments loaded at once
        
        #process each gpkg file (but collect catchments in batches)
        for gpkg_file in tqdm(gpkg_files, desc="  Processing regions") if show_progress else gpkg_files:
            try:
                #check for still unassigned sites
                still_unassigned_wgs84 = sites_gdf[sites_gdf['LINKNO'].isna()]
                if still_unassigned_wgs84.empty:
                    break
                
                #read catchment metadata
                meta = gpd.read_file(gpkg_file, rows=1)
                
                #get unassigned sites bounds in catchment CRS
                sites_temp = still_unassigned_wgs84.to_crs(meta.crs)
                bbox = tuple(sites_temp.total_bounds)
                
                #read catchments within bbox
                catchments = gpd.read_file(gpkg_file, bbox=bbox)
                
                if catchments.empty:
                    continue
                
                #get LINKNO column
                linkno_col = next((c for c in ['LINKNO', 'linkno'] if c in catchments.columns), None)
                if linkno_col is None:
                    continue
                
                catchments = catchments[[linkno_col, 'geometry']].rename(columns={linkno_col: 'LINKNO'})
                catchments['LINKNO'] = pd.to_numeric(catchments['LINKNO'], errors='coerce')
                catchments = catchments.dropna(subset=['LINKNO'])
                
                if catchments.empty:
                    continue
                
                catchments = catchments.to_crs(epsg=3857) #reproject to metric CRS
                
                #get (still) unassigned sites in this region (using expanded bbox)
                still_unassigned = sites_proj[sites_proj['LINKNO'].isna()].copy()
                
                # Filter sites to those within reasonable distance of this catchment file
                bbox_buffer = 100000  #100km in metres
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
                        max_distance=bbox_buffer
                    )
                
                if 'LINKNO_right' in joined.columns:
                    matched = joined['LINKNO_right'].notna()
                    if matched.any():
                        first = joined[matched].groupby(level=0).first()
                        
                        #only assign if not already assigned
                        to_assign = first.index.intersection(sites_proj[sites_proj['LINKNO'].isna()].index)
                        
                        if len(to_assign) > 0:
                            sites_proj.loc[to_assign, 'LINKNO'] = first.loc[to_assign, 'LINKNO_right']
                            sites_gdf.loc[to_assign, 'LINKNO'] = first.loc[to_assign, 'LINKNO_right']
                            assigned_proximity += len(to_assign)
            
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Error with {gpkg_file.name}: {e}")
                continue
        
        print(f"  LINKNO assigned by proximity: {assigned_proximity}/{unassigned_after_overlap} sites ({assigned_proximity/unassigned_after_overlap*100:.1f}%)")
        
        #report any sites still unassigned
        still_unassigned_final = sites_proj['LINKNO'].isna().sum()
        if still_unassigned_final > 0:
            print(f"  Warning: {still_unassigned_final} sites remain unassigned (>100km from any catchment)")
    else:
        assigned_proximity = 0
        print("  All sites assigned by overlap, no proximity assignment needed.")
    
    sites_gdf['LINKNO'] = pd.to_numeric(sites_gdf['LINKNO'], errors='coerce').astype('Int64')
    sites_gdf = sites_gdf.to_crs(epsg=4326)
    sites_gdf['gauge_lon'] = sites_gdf.geometry.x
    sites_gdf['gauge_lat'] = sites_gdf.geometry.y
    
    print(f"  Total LINKNO assigned: {assigned_overlap + assigned_proximity}/{total} sites ({(assigned_overlap + assigned_proximity)/total*100:.1f}%)")
    
    return sites_gdf


def snap_sites_to_rivers_within_linkno(sites_gdf, rivers, link_branch_dict, show_progress=True):
    """Snap sites to nearest river segment within the same LINKNO."""
    
    sites_with_linkno = sites_gdf[sites_gdf['LINKNO'].notna()].copy()
    sites_without_linkno = sites_gdf[sites_gdf['LINKNO'].isna()].copy()
    
    print(f"Sites with LINKNO: {len(sites_with_linkno)}")
    print(f"Sites without LINKNO: {len(sites_without_linkno)}")
    
    if sites_with_linkno.empty:
        print("No sites have LINKNO assigned, skipping river snapping")
        return sites_gdf
    
    #reproject gauge stations and rivers to metric CRS
    sites_proj = sites_with_linkno.to_crs(epsg=3857).copy()
    rivers_proj = rivers.to_crs(epsg=3857).copy()
    
    #initialise output columns
    sites_proj['gauge_lon_snapped'] = sites_proj.geometry.x
    sites_proj['gauge_lat_snapped'] = sites_proj.geometry.y
    sites_proj['snap_distance'] = 0.0
    sites_proj['merged_LINKNO'] = None
    
    linkno_groups = sites_proj.groupby('LINKNO')
    iterator = tqdm(linkno_groups, desc="Snapping by LINKNO", total=len(linkno_groups)) if show_progress else linkno_groups
    
    snapped_count = 0
    for linkno, group_sites in iterator:
        try:
            matching_rivers = rivers_proj[rivers_proj['LINKNO'] == linkno].copy()
            
            if matching_rivers.empty:
                if show_progress:
                    tqdm.write(f"No rivers found for LINKNO {linkno}")
                continue
            
            matching_rivers["river_geom"] = matching_rivers.geometry
            
            joined = gpd.sjoin_nearest(
                group_sites, matching_rivers[["LINKNO", "geometry", "river_geom"]],
                how="left", distance_col="snap_distance"
            ).reset_index().drop_duplicates(subset="index").set_index("index")
            
            for idx, row in joined.iterrows():
                river_geom = row["river_geom"]
                if river_geom is not None and river_geom.geom_type in ["LineString", "MultiLineString"]:
                    snapped_point = river_geom.interpolate(river_geom.project(row.geometry))
                    snapped_point_wgs84 = gpd.GeoSeries([snapped_point], crs=3857).to_crs(epsg=4326).iloc[0]
                    
                    sites_proj.loc[idx, 'gauge_lon_snapped'] = snapped_point_wgs84.x
                    sites_proj.loc[idx, 'gauge_lat_snapped'] = snapped_point_wgs84.y
                    sites_proj.loc[idx, 'snap_distance'] = row['snap_distance']
                    sites_proj.loc[idx, 'merged_LINKNO'] = link_branch_dict.get(linkno, -1)
                    snapped_count += 1
        except Exception as e:
            if show_progress:
                tqdm.write(f"Error processing LINKNO {linkno}: {str(e)}")
            continue
    
    #update GeoDataFrame
    sites_gdf.loc[sites_with_linkno.index, 'gauge_lon_snapped'] = sites_proj['gauge_lon_snapped']
    sites_gdf.loc[sites_with_linkno.index, 'gauge_lat_snapped'] = sites_proj['gauge_lat_snapped']
    sites_gdf.loc[sites_with_linkno.index, 'merged_LINKNO'] = sites_proj['merged_LINKNO']
    
    if not sites_without_linkno.empty:
        sites_gdf.loc[sites_without_linkno.index, 'gauge_lon_snapped'] = pd.NA
        sites_gdf.loc[sites_without_linkno.index, 'gauge_lat_snapped'] = pd.NA
        sites_gdf.loc[sites_without_linkno.index, 'merged_LINKNO'] = pd.NA
    
    sites_gdf = sites_gdf.drop(columns="geometry", errors="ignore")
    
    print(f"Successfully snapped: {snapped_count} sites")
    if snapped_count > 0:
        print(f"Average snap distance: {sites_proj['snap_distance'].mean():.1f} meters")
    
    return sites_gdf

###---------------------------------------------------###
###                   Main                            ###
###---------------------------------------------------###

def main():

    print("Processing gauge stations")

    #step 1. Combining attribute files
    print("Combining attribute files...")
    patterns = [
        "attributes_caravan_*.csv",
        "attributes_hydroatlas_*.csv",
        "attributes_other_*.csv"
    ]
    outputs = [
        Config.COMBINED_CARAVAN_ATTRS,
        Config.COMBINED_HYDROATLAS_ATTRS,
        Config.COMBINED_OTHER_ATTRS
    ]
    combine_attribute_files(Config.ATTRIBUTES_DIR, patterns, outputs)
    print(f"  Attribute files combined.")
        
    #step 2. Load sites
    print("Loading gauge stations...")
    sites_df = pd.read_csv(Config.COMBINED_OTHER_ATTRS)
    print(f"  Loaded {len(sites_df)} stations.")
    
    #step 3. Build GeoDataFrame
    print("Creating GeoDataFrame...")
    sites_gdf = build_geodataframe(sites_df)
    print(f"  GeoDataFrame created.")
    
    #step 4. Assign country and hydrobasin
    print("Assigning country and hydrobasin...")
    country_shp = gpd.read_file(Config.COUNTRY_SHP)
    hydrobasin_shp = gpd.read_file(Config.HYDROBASIN_SHP)
    sites_gdf = assign_country_hydrobasin(sites_gdf, country_shp, hydrobasin_shp)
    print(f" Country and hydrobasin assigned.")
    
    #step 5: Assign LINKNO (overlap first, then proximity for unmatched)
    print("Assigning LINKNOs...")
    sites_gdf = assign_linkno_from_catchments(sites_gdf, Config.SHOW_PROGRESS)
    print(f"  LINKNOs assigned.")
    
    #step 6: Snap coordinates to rivers
    print("Snapping coordinates to river network...")
    rivers = gpd.read_file(Config.RIVERS_GPKG).to_crs(epsg=3857)
    branch_map = pd.read_csv(Config.BRANCH_MAP_CSV)
    
    link_branch = branch_map.assign(
        LINKNOs=branch_map["merged_LINKNOs"].str.split(";")
    ).explode("LINKNOs")
    link_branch["LINKNOs"] = link_branch["LINKNOs"].astype(int)
    link_branch_dict = dict(zip(link_branch["LINKNOs"], link_branch["branch_id"]))
    
    sites_gdf = snap_sites_to_rivers_within_linkno(sites_gdf, rivers, link_branch_dict, Config.SHOW_PROGRESS)
    print(f"  Coordinates snapped to river network...")
    
    #step 7. Save output
    print("Saving results...")
    sites_gdf.to_csv(Config.SITE_INFO_OUTPUT, index=False)
    print(f"  Saved to: {Config.SITE_INFO_OUTPUT}")
    
    #summary
    print(f"\n=== Final Summary ===")
    print(f"Total sites: {len(sites_gdf)}")
    
    if 'country_name' in sites_gdf.columns:
        assigned = sites_gdf['country_name'].notna().sum()
        print(f"Country assigned: {assigned} ({assigned/len(sites_gdf)*100:.1f}%)")
    
    if 'hydrobasin_level12' in sites_gdf.columns:
        assigned = sites_gdf['hydrobasin_level12'].notna().sum()
        print(f"Hydrobasin assigned: {assigned} ({assigned/len(sites_gdf)*100:.1f}%)")
    
    if 'LINKNO' in sites_gdf.columns:
        assigned = sites_gdf['LINKNO'].notna().sum()
        print(f"LINKNO assigned: {assigned} ({assigned/len(sites_gdf)*100:.1f}%)")
    
    if 'merged_LINKNO' in sites_gdf.columns:
        assigned = sites_gdf['merged_LINKNO'].notna().sum()
        print(f"Snapped to rivers: {assigned} ({assigned/len(sites_gdf)*100:.1f}%)")

    print("\nGauge processing complete")

if __name__ == "__main__":
    main()