# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import geopandas as gpd
import networkx as nx
import pickle
from tqdm import tqdm
from shapely.geometry import Point, MultiLineString
from shapely.ops import transform
import pyproj

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

#Directories
input_dir_wqms = "/gpfs/work4/0/dynql/Caravan-Qual/"
input_dir_aux_data = "/gpfs/work4/0/dynql/Caravan-Qual/auxiliary/"
input_dir_Caravan = "/gpfs/work4/0/dynql/Caravan-Qual/auxiliary/Caravan/"   

include_full_grdc = False #option to include or exclude grdc sites that do not have a permissive license.
    
###---------------------------------------------------###
###              Add streamflow data                  ###
###---------------------------------------------------###
   
def load_data(wqms_csv, caravan_csv, graph_file):

    #wqms sites
    print("Loading WQMS sites...")
    wqms = pd.read_csv(wqms_csv)
    
    #clean up columns (if already existing)
    columns_to_remove = ['gauge_id', 'gauge_distance_km', 'match_method', 'num_candidates']
    existing_columns = [col for col in columns_to_remove if col in wqms.columns]
    if existing_columns:
        print(f"    Removing existing columns: {existing_columns}")
        wqms = wqms.drop(columns=existing_columns)
    
    gdf_wqms = gpd.GeoDataFrame(
        wqms,
        geometry=gpd.points_from_xy(wqms["wqms_lon"], wqms["wqms_lat"]),
        crs="EPSG:4326"
    )
    print(f"{len(gdf_wqms)} WQMS sites loaded")

    #Caravan sites
    print("Loading Caravan sites...")
    caravan = pd.read_csv(caravan_csv)
    gdf_caravan = gpd.GeoDataFrame(
        caravan,
        geometry=gpd.points_from_xy(caravan["gauge_lon_snapped"], caravan["gauge_lat_snapped"]),
        crs="EPSG:4326"
    )
    print(f"{len(gdf_caravan)} Caravan sites loaded")

    #Load river network graph
    print("Loading river network graph...")
    with open(graph_file, "rb") as f:
        G = pickle.load(f)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return gdf_wqms, gdf_caravan, G


def find_candidates(wq_row, caravan_df, match_stage):
    """Select candidate gauges for matching"""
    if match_stage == "same_linkno":
        candidates = caravan_df[caravan_df["LINKNO"] == wq_row["LINKNO"]].copy()
    elif match_stage == "same_merged_linkno":
        candidates = caravan_df[caravan_df["merged_LINKNO"] == wq_row["merged_LINKNO"]].copy()
    else:
        return pd.DataFrame()
    return candidates


def calculate_river_distance(source_link, target_link, G, wq_point=None, gauge_point=None, G_rev=None):
    """Calculate along-river distance between wqms_id and gauge_id in meters."""

    #transform to metric CRS (Web Mercator)
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

    ##process wqms_id and gauge_id that are located on the same segment
    if source_link == target_link:
        geom = G.nodes[source_link]["geometry"]
        
        if isinstance(geom, MultiLineString):
            wq_m = transform(project, Point(wq_point.x, wq_point.y))
            best_wq_dist = float('inf')
            best_wq_geom = None
            best_wq_pos = None
            
            for line_geom in geom.geoms:
                line_geom_m = transform(project, line_geom)
                dist_to_line = line_geom_m.distance(wq_m)
                
                if dist_to_line < best_wq_dist:
                    best_wq_dist = dist_to_line
                    best_wq_geom = line_geom_m
                    best_wq_pos = line_geom_m.project(wq_m)
            
            gauge_m = transform(project, Point(gauge_point.x, gauge_point.y))
            best_gauge_dist = float('inf')
            best_gauge_geom = None
            best_gauge_pos = None
            
            for line_geom in geom.geoms:
                line_geom_m = transform(project, line_geom)
                dist_to_line = line_geom_m.distance(gauge_m)
                
                if dist_to_line < best_gauge_dist:
                    best_gauge_dist = dist_to_line
                    best_gauge_geom = line_geom_m
                    best_gauge_pos = line_geom_m.project(gauge_m)
            
            if best_wq_geom is best_gauge_geom:
                return abs(best_gauge_pos - best_wq_pos)
            
            #If different sub-segments, sum distances through the MultiLineString
            else:
                #convert back to find which sub-segments in the original geometry
                wq_segment_idx = None
                gauge_segment_idx = None
                
                for idx, line_geom in enumerate(geom.geoms):
                    line_geom_m = transform(project, line_geom)
                    if line_geom_m.equals(best_wq_geom):
                        wq_segment_idx = idx
                    if line_geom_m.equals(best_gauge_geom):
                        gauge_segment_idx = idx
                
                if wq_segment_idx is not None and gauge_segment_idx is not None:
                    #sum lengths of intermediate segments
                    start_idx = min(wq_segment_idx, gauge_segment_idx)
                    end_idx = max(wq_segment_idx, gauge_segment_idx)
                    
                    total_dist = 0.0
                    for idx in range(start_idx, end_idx + 1):
                        line_geom = geom.geoms[idx]
                        line_geom_m = transform(project, line_geom)
                        
                        if idx == wq_segment_idx:
                            #estimate partial distance from wqms_id to end of segment
                            total_dist += line_geom_m.length - best_wq_pos
                        elif idx == gauge_segment_idx:
                            #estimate partial distance from start to gauge_id
                            total_dist += best_gauge_pos
                        else:
                            #get full segment length
                            total_dist += line_geom_m.length
                    
                    return total_dist
                
                #fallback: use direct distance between projections
                return abs(best_gauge_pos - best_wq_pos)
        
        else:
            geom_m = transform(project, geom)
            wq_m = transform(project, Point(wq_point.x, wq_point.y))
            gauge_m = transform(project, Point(gauge_point.x, gauge_point.y))

            wq_pos = geom_m.project(wq_m)
            gauge_pos = geom_m.project(gauge_m)
            dist_m = abs(gauge_pos - wq_pos)
            
            return dist_m

    ##process wqms_id and gauge_id that are not located on the same segment
    paths = []

    #look downstream
    try:
        paths.append(nx.shortest_path(G, source=source_link, target=target_link, weight="weight"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    #look upstream
    if G_rev is not None:
        try:
            paths.append(nx.shortest_path(G_rev, source=source_link, target=target_link, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    if not paths:
        return None

    min_dist = None
    for path in paths:
        dist_m = 0.0
        for i, node in enumerate(path):
            geom = G.nodes[node]["geometry"]
            length = G.nodes[node]["length_m"]
            if isinstance(geom, MultiLineString):
                geom = geom.geoms[0]
            geom_m = transform(project, geom)

            #distance from wqms_id to end of the segment
            if i == 0 and wq_point is not None:
                wq_m = transform(project, Point(wq_point.x, wq_point.y))
                dist_m += geom_m.length - geom_m.project(wq_m)
            #distance from segment start to gauge_id
            elif i == len(path) - 1 and gauge_point is not None:
                gauge_m = transform(project, Point(gauge_point.x, gauge_point.y))
                dist_m += geom_m.project(gauge_m)
            #use stored length in meters for intermediate segments
            else:
                dist_m += length

        if min_dist is None or dist_m < min_dist:
            min_dist = dist_m

    return min_dist


def match_single_wqms_site(wq_row, caravan_df, G, G_rev=None):
    """Match wqms_id to nearest Caravan gauge using along-river distance"""
    #check for same LINKNO
    candidates = find_candidates(wq_row, caravan_df, "same_linkno")
    match_method = "same_linkno"

    #check for same merged_LINKNO (if no candidates)
    if candidates.empty:
        candidates = find_candidates(wq_row, caravan_df, "same_merged_linkno")
        match_method = "same_merged_linkno"

    if candidates.empty:
        return {"gauge_id": None, "gauge_distance_km": None, "match_method": "no_match", "num_candidates": 0}

    distances = []
    wq_point = Point(wq_row['wqms_lon'], wq_row['wqms_lat'])

    for idx, candidate in candidates.iterrows():
        gauge_point = Point(candidate['gauge_lon_snapped'], candidate['gauge_lat_snapped'])
        
        river_dist = calculate_river_distance(
            wq_row["LINKNO"], candidate["LINKNO"], G,
            wq_point=wq_point, gauge_point=gauge_point, G_rev=G_rev
        )
        
        if river_dist is not None:
            distances.append({"idx": idx, "gauge_id": candidate.get("gauge_id", idx), "distance_m": river_dist})

    if not distances:
        return {"gauge_id": None, "gauge_distance_km": None,
                "match_method": f"{match_method}_no_path", "num_candidates": len(candidates)}

    nearest = min(distances, key=lambda x: x["distance_m"])
    
    return {"gauge_id": nearest["gauge_id"], "gauge_distance_km": round(nearest["distance_m"]/1000.0, 4),
            "match_method": match_method, "num_candidates": len(candidates)}
            

def run_matching(gdf_wqms, gdf_caravan, G, G_rev=None):
    """Run the matching for all wqms_ids"""
    print("Starting matching process...")
    results = []

    for idx, wq_row in tqdm(gdf_wqms.iterrows(), total=len(gdf_wqms), desc="Matching sites"):
        match_result = match_single_wqms_site(wq_row, gdf_caravan, G, G_rev)
        match_result["wqms_idx"] = idx
        results.append(match_result)

    results_df = pd.DataFrame(results)
    final_df = pd.concat([gdf_wqms.reset_index(drop=True), results_df.drop("wqms_idx", axis=1)], axis=1)
    return final_df, results_df


def print_matching_summary(results_df):
    """Print matching statistics"""
    print("\n=== MATCHING SUMMARY ===")
    total_sites = len(results_df)
    matched_sites = len(results_df[results_df["gauge_id"].notna()])
    print(f"Total WQMS sites: {total_sites}")
    print(f"Successfully matched: {matched_sites} ({matched_sites/total_sites*100:.1f}%)")
    print(f"Unmatched: {total_sites - matched_sites}")
    print("\nMatching method breakdown:")
    for method, count in results_df["match_method"].value_counts().items():
        print(f"  {method}: {count} ({count/total_sites*100:.1f}%)")

    matched_results = results_df[results_df["gauge_id"].notna()]
    if len(matched_results) > 0:
        distances = matched_results["gauge_distance_km"]
        print(f"\nDistance statistics (km): Mean={distances.mean():.2f}, Median={distances.median():.2f}, "
              f"Min={distances.min():.2f}, Max={distances.max():.2f}")

###---------------------------------------------------###
###                     Main                          ###
###---------------------------------------------------###

def main():
    
    if len(sys.argv) > 1:
        arg_name = sys.argv[1]
        wqms_csv = os.path.join(input_dir_aux_data, "wq_data", f"wqms_site_info_for_{arg_name}.csv")
        print(f"Processing only wqms sites for database: {arg_name}")
    else:
        wqms_csv = os.path.join(input_dir_wqms, "wqms_site_info.csv")
        print("Processing all wqms sites")
    
    print(f"Using WQMS file: {wqms_csv}")

    if include_full_grdc:
        print("Including GRDC stations that have a non-permissive sharing license")
        caravan_csv = os.path.join(input_dir_Caravan, "caravan_site_info_full_grdc.csv")
    else:    
        caravan_csv = os.path.join(input_dir_Caravan, "caravan_site_info.csv")
    
    graph_file = os.path.join(input_dir_aux_data, "geoglows_TDXhydro/graph_G.pkl")

    gdf_wqms, gdf_caravan, G = load_data(wqms_csv, caravan_csv, graph_file)
    G_rev = G.reverse(copy=False)

    final_df, results_df = run_matching(gdf_wqms, gdf_caravan, G, G_rev)
    final_df = final_df.drop(columns=["geometry", "match_method", "num_candidates"], errors='ignore')
    final_df = final_df.sort_values(by="wqms_id").reset_index(drop=True)

    print_matching_summary(results_df)

    if include_full_grdc:
        if len(sys.argv) > 1:
            wqms_csv = os.path.join(input_dir_aux_data, "wq_data", f"wqms_site_info_for_{arg_name}_full_grdc.csv")
        else:
            wqms_csv = os.path.join(input_dir_wqms, "wqms_site_info_full_grdc.csv")
        print(f"\nSaving results to {wqms_csv}...")
        final_df.to_csv(wqms_csv, index=False)
    else:
        print(f"\nSaving results to {wqms_csv}...")
        final_df.to_csv(wqms_csv, index=False)
    print("Complete!")

    return final_df, results_df

if __name__ == "__main__":
    final_df, results_df = main()
