# -*- coding: utf-8 -*-
import os
import sys
import pickle
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
try:
    from shapely.ops import union_all as _union_all
    def union_geoms(geoms):
        return _union_all(geoms)
except Exception:
    def union_geoms(geoms):
        return unary_union(list(geoms))

from datetime import datetime
import warnings
from tqdm import tqdm
import glob
from multiprocessing import Pool, cpu_count
import gc
import psutil
import time

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

input_dir_wqms = "/gpfs/work4/0/dynql/Caravan-Qual/"
input_dir_aux_data = "/gpfs/work4/0/dynql/Caravan-Qual/auxiliary/"

GRAPH_FILE      = os.path.join(input_dir_aux_data, "geoglows_TDXhydro/graph_G.pkl")
CATCH_GPKG_DIR  = os.path.join(input_dir_aux_data, "geoglows_TDXhydro/catchments_gpkg/")
SITE_CSV        = os.path.join(input_dir_wqms, "wqms_site_info.csv")

OUTPUT_DIR      = os.path.join(input_dir_aux_data, "wqms-gpkg/individual_catchments/")
FINAL_OUTPUT    = os.path.join(input_dir_aux_data, "wqms-gpkg/wqms_TDXhydro_catchments.gpkg")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FINAL_OUTPUT), exist_ok=True)

COMBINE_NOW = True #option to combine individual files at the end
INCREMENTAL_MODE = True #just append polygons from new LINKNOs to existing files

G_GLOBAL = None

###---------------------------------------------------###
###               Add wqms shapefiles                 ###
###---------------------------------------------------###

def init_worker(graph):
    global G_GLOBAL
    G_GLOBAL = graph


def get_upstream_network(linkno, linkno_to_geom, local_cache, reuse_stats=None):

    #if LINKNO already computed, just return
    if linkno in local_cache:
        geom, upstream_count = local_cache[linkno]
        if reuse_stats is not None:
            reuse_stats['local_hits'] += 1
        return geom, upstream_count, True
    
    #iterative approach with cache reuse
    geom_list = []
    total_upstream_count = 0
    to_process = [linkno]
    processed = set()
    
    while to_process:
        current_linkno = to_process.pop()
        
        if current_linkno in processed:
            continue
        processed.add(current_linkno)
        
        #check if linkno is already present in cache
        if current_linkno in local_cache:
            cached_geom, cached_count = local_cache[current_linkno]
            if cached_geom is not None:
                geom_list.append(cached_geom)
            total_upstream_count += cached_count
            
            if reuse_stats is not None:
                reuse_stats['upstream_cached'] += 1
                reuse_stats['upstream_total'] += cached_count
            
            continue
        
        #add geometry if avaliable
        if current_linkno in linkno_to_geom:
            geom_list.append(linkno_to_geom[current_linkno])
            total_upstream_count += 1
        
        #add upstream to stack (if not already cached)
        for upstream_linkno in G_GLOBAL.predecessors(current_linkno):
            if upstream_linkno not in processed and upstream_linkno not in local_cache:
                to_process.append(upstream_linkno)
            elif upstream_linkno in local_cache:
                #if cached upstream is found, add directly
                cached_geom, cached_count = local_cache[upstream_linkno]
                if cached_geom is not None:
                    geom_list.append(cached_geom)
                total_upstream_count += cached_count
                
                if reuse_stats is not None:
                    reuse_stats['upstream_cached'] += 1
                    reuse_stats['upstream_total'] += cached_count
    
    geom = union_geoms(geom_list) if geom_list else None
    local_cache[linkno] = (geom, total_upstream_count)
    
    return geom, total_upstream_count, False


def process_gpkg_worker(gpkg_file, linkno_set, output_dir, incremental=True):
    """Process a single GPKG file to generate catchments per LINKNO."""
    
    global G_GLOBAL
    start_time = time.time()
    base_name = os.path.splitext(os.path.basename(gpkg_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.gpkg")

    #existing results
    local_cache = {}
    existing_linknos = set()
    
    if os.path.exists(output_file):
        print(f"  Found existing results for {base_name}, loading...")
        try:
            prior_gdf = gpd.read_file(output_file, layer="catchments")
            for _, row in prior_gdf.iterrows():
                linkno = int(row['LINKNO'])
                existing_linknos.add(linkno)
                local_cache[linkno] = (
                    row['geometry'],
                    int(row.get('upstream_count', 0))
                )
            print(f"  Loaded {len(local_cache)} existing LINKNOs from {base_name}")
            del prior_gdf
            gc.collect()
        except Exception as e:
            print(f"  Warning: Could not load prior results from {base_name}: {e}")
            existing_linknos = set()
            local_cache = {}

    print(f" starting {base_name}...")
    try:
        
        #read gpkg once for this worker/file
        gdf = gpd.read_file(gpkg_file)
        if gdf.empty:
            elapsed = time.time() - start_time
            print(f"  {base_name} empty ({elapsed:.1f}s)", flush=True)
            return base_name, 'EMPTY', elapsed

        #check LINKNO column exists
        link_col = next((c for c in gdf.columns if c.upper() == "LINKNO"), None)
        if link_col is None:
            elapsed = time.time() - start_time
            return base_name, 'NO_LINKNO', elapsed
        if link_col != "LINKNO":
            gdf.rename(columns={link_col: "LINKNO"}, inplace=True)
        gdf["LINKNO"] = pd.to_numeric(gdf["LINKNO"], errors='coerce').astype('Int64')
        gdf = gdf.dropna(subset=["LINKNO"])
        gdf["LINKNO"] = gdf["LINKNO"].astype(int)

        #check Which LINKNOs are in this gpkg
        gdf_links = set(gdf["LINKNO"].values)
        linknos_requested = linkno_set & gdf_links
        
        if not linknos_requested:
            elapsed = time.time() - start_time
            return base_name, 'NO_STATIONS', elapsed

        #only process new LINKNOs and append results to existing .gpkg
        if incremental:
            linknos_to_process = linknos_requested - existing_linknos
            linknos_already_done = linknos_requested & existing_linknos
            
            if not linknos_to_process:
                elapsed = time.time() - start_time
                print(f"  {base_name}: all {len(linknos_already_done)} LINKNOs already done ({elapsed:.1f}s)")
                return base_name, 'SKIPPED', elapsed
            
            print(f"  {base_name}: {len(linknos_to_process)} new LINKNOs to process")
        else:
            linknos_to_process = linknos_requested
            linknos_already_done = set()

        #in-memory map LINKNO to geometry
        linkno_to_geom = {int(r["LINKNO"]): r.geometry for _, r in gdf.iterrows()}
        del gdf
        gc.collect()

        results = []
        for linkno in linknos_to_process:
            try:
                geom, upstream_count, was_cached = get_upstream_network(
                    linkno, linkno_to_geom, local_cache, reuse_stats=None
                )
                if geom is not None:
                    results.append({
                        "LINKNO": linkno,
                        "upstream_count": upstream_count,
                        "geometry": geom
                    })
            except Exception as e:
                print(f"    Error processing LINKNO {linkno}: {e}")
                continue

        if not results:
            elapsed = time.time() - start_time
            return base_name, 'NO_RESULTS', elapsed

        #save results
        new_gdf = gpd.GeoDataFrame(results, crs="EPSG:4326")
        
        if incremental and os.path.exists(output_file):
            #append to existing file
            try:
                existing_gdf = gpd.read_file(output_file, layer="catchments")
                combined_gdf = pd.concat([existing_gdf, new_gdf], ignore_index=True)
                combined_gdf.to_file(output_file, driver="GPKG", layer="catchments")
                del existing_gdf, combined_gdf
            except Exception as e:
                print(f"    Warning: Could not append to {base_name}, overwriting: {e}")
                new_gdf.to_file(output_file, driver="GPKG", layer="catchments")
        else:
            #create new file
            new_gdf.to_file(output_file, driver="GPKG", layer="catchments")
        
        del new_gdf
        gc.collect()
        
        elapsed = time.time() - start_time
        print(f"  {base_name}: completed {len(results)} LINKNOs ({elapsed:.1f}s)")
        return base_name, 'COMPLETED', elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ERROR in {base_name}: {e}")
        return base_name, 'ERROR', elapsed


def combine_individual_files(output_dir, final_output):

    print(f"Combining individual files from {output_dir}")
    individual_files = glob.glob(os.path.join(output_dir, "catchments_*.gpkg"))
    if not individual_files:
        print("No individual files found to combine!")
        return

    all_gdfs = []
    total_records = 0

    for i, file_path in enumerate(tqdm(individual_files, desc="Loading files")):
        try:
            gdf = gpd.read_file(file_path, layer="catchments")
            total_records += len(gdf)
            all_gdfs.append(gdf)
            if (i + 1) % 50 == 0:
                mem = psutil.Process().memory_info().rss / (1024**3)
                print(f"  Loaded {i+1}/{len(individual_files)} files, {total_records} records, {mem:.1f}GB memory")
        except Exception as e:
            print(f"  Error loading {os.path.basename(file_path)}: {e}")

    if all_gdfs:
        print(f"  Concatenating {len(all_gdfs)} dataframes...")
        final_gdf = pd.concat(all_gdfs, ignore_index=True)
        
        #check for (and handle, if needed) duplicates
        initial_count = len(final_gdf)
        duplicate_count = final_gdf.duplicated(subset=['LINKNO']).sum()
        
        if duplicate_count > 0:
            print(f"  Found {duplicate_count} duplicate LINKNOs, keeping first occurrence...")
            final_gdf = final_gdf.drop_duplicates(subset=['LINKNO'], keep='first')
        
        print(f"  Writing final output to {final_output}")
        final_gdf.to_file(final_output, driver="GPKG", layer="catchments")
        print(f"  Combined file saved: {len(final_gdf)} unique LINKNOs")
        
        if duplicate_count > 0:
            print(f"  (Removed {initial_count - len(final_gdf)} duplicate records)")
        
        del all_gdfs, final_gdf
        gc.collect()
    else:
        print("No data to combine!")


###---------------------------------------------------###
###                     Main                          ###
###---------------------------------------------------###

if __name__ == "__main__":
    
    #load river network graph
    print("Loading river network graph...")
    with open(GRAPH_FILE, "rb") as f:
        G_main = pickle.load(f)
    print(f"  Graph loaded: {G_main.number_of_nodes()} nodes, {G_main.number_of_edges()} edges.")

    #load water quality station data
    print("Loading and preparing station data...")
    stations = pd.read_csv(SITE_CSV)
    
    #check for LINKNO
    if "LINKNO" not in stations.columns:
        raise SystemExit(f"CSV missing required column: LINKNO")

    #clean and prepare LINKNO 
    stations = stations.dropna(subset=["LINKNO"])
    stations["LINKNO"] = pd.to_numeric(stations["LINKNO"], errors='coerce').astype('Int64')
    stations = stations.dropna(subset=["LINKNO"])
    stations["LINKNO"] = stations["LINKNO"].astype(int)
    unique_linknos = set(stations["LINKNO"].unique())

    print(f"  Total station records: {len(stations)}")
    print(f"  Unique LINKNOs to process: {len(unique_linknos)}")

    #get GPKG files, sort by size
    gpkg_files = glob.glob(os.path.join(CATCH_GPKG_DIR, "*.gpkg"))
    if not gpkg_files:
        raise SystemExit(f"No GPKG files found in {CATCH_GPKG_DIR}")

    gpkg_files_with_size = [(f, os.path.getsize(f)) for f in gpkg_files]
    gpkg_files_with_size.sort(key=lambda x: x[1], reverse=False)
    gpkg_files = [f for f, _ in gpkg_files_with_size]

    print(f"  Found {len(gpkg_files)} GPKG files.")

    #run multiprocessing
    NUM_PROCESSES = 4
    process_args = [(f, unique_linknos, OUTPUT_DIR, INCREMENTAL_MODE) for f in gpkg_files]
    
    completed = 0
    start_time = time.time()
    
    with Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(G_main,)) as pool:
        for result in pool.starmap(process_gpkg_worker, process_args):
            base_name, status, elapsed = result
            completed += 1
            
            if completed % 10 == 0:
                total_elapsed = time.time() - start_time
                print(f"\n>>> Progress: {completed}/{len(gpkg_files)} files ({completed/len(gpkg_files)*100:.1f}%), {total_elapsed/60:.1f} min elapsed\n")

    if COMBINE_NOW:
        combine_individual_files(OUTPUT_DIR, FINAL_OUTPUT)
    else:
        print(f"Final output not combined.")

    print("Processing complete!")