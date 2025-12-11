import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm

warnings.filterwarnings('ignore')

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

#Define input and output paths
BASIN_SHAPEFILE = '/gpfs/work4/0/dynql/Caravan-Qual/auxiliary/wqms-gpkg/wqms_basin_shapes.gpkg'
HYDROATLAS_FOLDER = '/gpfs/work4/0/dynql/Caravan-Qual/auxiliary/HydroATLAS/'
OUTPUT_FILE = '/gpfs/work4/0/dynql/Caravan-Qual/auxiliary/attributes/linkno_hydroatlas_attributes_Caravan.csv'

#Basin identification and overlap thresholds
LINKNO_FIELD = 'LINKNO'
MIN_OVERLAP_THRESHOLD_DEFAULT = 5
MIN_OVERLAP_PERCENTAGE = 0.5


###---------------------------------------------------###
###                HydroATLAS setup                   ###
###---------------------------------------------------###

#HydroATLAS shapefile patterns and property categories
SHAPEFILE_PATTERNS = {
    'level12': '*lev12*.shp',
    'level09': '*lev09*.shp',
    'level07': '*lev07*.shp',
}

MAJORITY_PROPERTIES = [
    'clz_cl_smj', 'cls_cl_smj', 'glc_cl_smj', 'pnv_cl_smj', 
    'wet_cl_smj', 'tbi_cl_smj', 'tec_cl_smj', 'fmh_cl_smj', 
    'fec_cl_smj', 'lit_cl_smj',
]

POUR_POINT_PROPERTIES = [
    'dis_m3_pmn', 'dis_m3_pmx', 'dis_m3_pyr',
    'lkv_mc_usu', 'rev_mc_usu', 'ria_ha_usu', 'riv_tc_usu',
    'pop_ct_usu', 'dor_pc_pva',
]

IGNORE_PROPERTIES = [
    'COAST', 'DIST_MAIN', 'DIST_SINK', 'ENDO', 'MAIN_BAS',
    'NEXT_SINK', 'ORDER_', 'PFAF_ID', 'SORT',
    'geometry', 'geometry_ea',
]

PROCESSING_PROPERTIES = ['HYBAS_ID', 'NEXT_DOWN', 'SUB_AREA', 'UP_AREA']

#Processing configuration (defined by basin size)
PROCESSING_CONFIG = {
    '0-to-2000': {
        'min_overlap_threshold': 0,
        'batch_size': 5,
        'hydroatlas_level': 'level12',
        'area_greater_than': None,
        'area_not_greater_than': 2000,
    },
    '2000-to-50000': {
        'min_overlap_threshold': 5,
        'batch_size': 10,
        'hydroatlas_level': 'level12',
        'area_greater_than': 2000,
        'area_not_greater_than': 50000,
    },
    '50000-to-100000': {
        'min_overlap_threshold': 5,
        'batch_size': 10,
        'hydroatlas_level': 'level09',
        'area_greater_than': 50000,
        'area_not_greater_than': 100000,
    },
    '100000-to-1000000': {
        'min_overlap_threshold': 5,
        'batch_size': 5,
        'hydroatlas_level': 'level07',
        'area_greater_than': 100000,
        'area_not_greater_than': 1000000,
    },
    '1000000-to-inf': {
        'min_overlap_threshold': 5,
        'batch_size': 1,
        'hydroatlas_level': 'level07',
        'area_greater_than': 1000000,
        'area_not_greater_than': None,
    },
}

###---------------------------------------------------###
###               Output format                       ###
###---------------------------------------------------###

OUTPUT_COLUMNS = [
    'LINKNO',
    'sgr_dk_sav',
    'glc_pc_s06',
    'glc_pc_s07',
    'nli_ix_sav',
    'glc_pc_s04',
    'glc_pc_s05',
    'glc_pc_s02',
    'glc_pc_s03',
    'glc_pc_s01',
    'pet_mm_syr',
    'glc_pc_s08',
    'glc_pc_s09',
    'swc_pc_s09',
    'ele_mt_smx',
    'tbi_cl_smj',
    'swc_pc_s01',
    'swc_pc_s02',
    'swc_pc_s03',
    'swc_pc_s04',
    'swc_pc_s05',
    'swc_pc_s06',
    'swc_pc_s07',
    'swc_pc_s08',
    'crp_pc_sse',
    'glc_pc_s22',
    'glc_pc_s20',
    'glc_pc_s21',
    'wet_pc_sg1',
    'wet_pc_sg2',
    'pac_pc_sse',
    'swc_pc_s10',
    'swc_pc_s11',
    'swc_pc_s12',
    'clz_cl_smj',
    'gwt_cm_sav',
    'glc_pc_s17',
    'glc_pc_s18',
    'hft_ix_s93',
    'glc_pc_s15',
    'ire_pc_sse',
    'glc_pc_s16',
    'glc_pc_s13',
    'prm_pc_sse',
    'glc_pc_s14',
    'glc_pc_s11',
    'glc_pc_s12',
    'glc_pc_s10',
    'kar_pc_sse',
    'slp_dg_sav',
    'glc_pc_s19',
    'tmp_dc_s07',
    'tmp_dc_s08',
    'tmp_dc_s05',
    'tmp_dc_s06',
    'tmp_dc_s09',
    'for_pc_sse',
    'aet_mm_s06',
    'aet_mm_s05',
    'aet_mm_s08',
    'aet_mm_s07',
    'aet_mm_s09',
    'tmp_dc_s10',
    'tmp_dc_s11',
    'aet_mm_s02',
    'aet_mm_s01',
    'tmp_dc_s12',
    'aet_mm_s04',
    'aet_mm_s03',
    'lit_cl_smj',
    'tmp_dc_s03',
    'tmp_dc_s04',
    'tmp_dc_s01',
    'tmp_dc_s02',
    'cls_cl_smj',
    'pre_mm_syr',
    'pnv_pc_s01',
    'pnv_pc_s04',
    'pnv_pc_s05',
    'pnv_pc_s02',
    'rdd_mk_sav',
    'ele_mt_smn',
    'pnv_pc_s03',
    'pnv_pc_s08',
    'pnv_pc_s09',
    'pnv_pc_s06',
    'pnv_pc_s07',
    'wet_cl_smj',
    'snw_pc_syr',
    'pnv_pc_s11',
    'pnv_pc_s12',
    'pnv_pc_s10',
    'pnv_pc_s15',
    'pnv_pc_s13',
    'pnv_pc_s14',
    'cmi_ix_syr',
    'pet_mm_s11',
    'pet_mm_s12',
    'pet_mm_s10',
    'tmp_dc_smn',
    'wet_pc_s08',
    'wet_pc_s09',
    'slt_pc_sav',
    'wet_pc_s02',
    'wet_pc_s03',
    'wet_pc_s01',
    'hdi_ix_sav',
    'wet_pc_s06',
    'wet_pc_s07',
    'wet_pc_s04',
    'wet_pc_s05',
    'fec_cl_smj',
    'glc_cl_smj',
    'swc_pc_syr',
    'hft_ix_s09',
    'soc_th_sav',
    'gdp_ud_sav',
    'gdp_ud_ssu',
    'tmp_dc_smx',
    'cly_pc_sav',
    'pet_mm_s02',
    'pet_mm_s03',
    'pet_mm_s01',
    'snw_pc_smx',
    'ppd_pk_sav',
    'pet_mm_s08',
    'aet_mm_s11',
    'pet_mm_s09',
    'aet_mm_s10',
    'pet_mm_s06',
    'pet_mm_s07',
    'aet_mm_s12',
    'pet_mm_s04',
    'pet_mm_s05',
    'inu_pc_slt',
    'ero_kh_sav',
    'aet_mm_syr',
    'cmi_ix_s10',
    'cmi_ix_s11',
    'cmi_ix_s12',
    'ari_ix_sav',
    'tmp_dc_syr',
    'tec_cl_smj',
    'fmh_cl_smj',
    'inu_pc_smn',
    'pnv_cl_smj',
    'pre_mm_s08',
    'pre_mm_s09',
    'run_mm_syr',
    'pre_mm_s06',
    'pre_mm_s07',
    'pre_mm_s04',
    'pre_mm_s05',
    'snd_pc_sav',
    'pre_mm_s02',
    'pre_mm_s03',
    'ele_mt_sav',
    'pre_mm_s01',
    'pre_mm_s10',
    'pre_mm_s11',
    'pre_mm_s12',
    'urb_pc_sse',
    'lka_pc_sse',
    'snw_pc_s01',
    'snw_pc_s02',
    'snw_pc_s03',
    'snw_pc_s04',
    'snw_pc_s05',
    'snw_pc_s06',
    'gla_pc_sse',
    'snw_pc_s07',
    'snw_pc_s08',
    'snw_pc_s09',
    'inu_pc_smx',
    'snw_pc_s10',
    'snw_pc_s11',
    'snw_pc_s12',
    'cmi_ix_s07',
    'cmi_ix_s08',
    'cmi_ix_s05',
    'cmi_ix_s06',
    'cmi_ix_s09',
    'cmi_ix_s03',
    'cmi_ix_s04',
    'cmi_ix_s01',
    'cmi_ix_s02',
    'pst_pc_sse',
    'dis_m3_pmn',
    'dis_m3_pmx',
    'dis_m3_pyr',
    'lkv_mc_usu',
    'rev_mc_usu',
    'ria_ha_usu',
    'riv_tc_usu',
    'pop_ct_usu',
    'dor_pc_pva',
    'area',
    'area_fraction_used_for_aggregation',
]

###---------------------------------------------------###
###               Helper functions                    ###
###---------------------------------------------------###

def validate_inputs():
    """Validate that all input paths and files exist."""
    print("Validating input paths...")
    
    if not Path(BASIN_SHAPEFILE).exists():
        raise FileNotFoundError(f"Basin shapefile not found: {BASIN_SHAPEFILE}")
    
    if not Path(HYDROATLAS_FOLDER).exists():
        raise FileNotFoundError(f"HydroATLAS folder not found: {HYDROATLAS_FOLDER}")
    
    print("All input paths validated successfully!")


def load_basin_data():
    """Load basin shapefile and prepare it for processing."""
    print(f"Loading basin data from {BASIN_SHAPEFILE}...")
    basins = gpd.read_file(BASIN_SHAPEFILE)
    
    if LINKNO_FIELD not in basins.columns:
        raise ValueError(f"'{LINKNO_FIELD}' field not found in basin shapefile")
    
    basins['geometry'] = basins['geometry'].buffer(0)
    basins['area_km2'] = basins.geometry.to_crs('EPSG:6933').area / 1e6
    
    print(f"  Loaded {len(basins)} basins")
    print(f"  Area range: {basins['area_km2'].min():.2f} to {basins['area_km2'].max():.2f} km2")
    
    return basins


def load_existing_linknos():
    """Load LINKNOs from existing output file to skip reprocessing."""
    output_path = Path(OUTPUT_FILE)
    
    if not output_path.exists():
        print("No existing output file found - will process all basins")
        return set()
    
    try:
        existing_df = pd.read_csv(OUTPUT_FILE)
        if LINKNO_FIELD in existing_df.columns:
            existing_linknos = set(existing_df[LINKNO_FIELD])
            print(f"Found {len(existing_linknos)} existing LINKNOs in output file")
            return existing_linknos
    except Exception as e:
        print(f"Warning: Could not read existing file: {e}")
    
    return set()


def load_hydroatlas_data(level):
    """Load HydroATLAS data for the specified level."""
    pattern = SHAPEFILE_PATTERNS[level]
    print(f"  Loading HydroATLAS data (level: {level})...")
    
    shapefiles = list(Path(HYDROATLAS_FOLDER).rglob(pattern))
    
    if not shapefiles:
        raise FileNotFoundError(f"No HydroATLAS shapefiles found matching pattern '{pattern}'")
    
    print(f"    Found {len(shapefiles)} shapefile(s)")
    
    hydroatlas_parts = []
    for shp in shapefiles:
        gdf = gpd.read_file(shp)
        gdf['geometry'] = gdf['geometry'].buffer(0)
        
        hydroatlas_parts.append(gdf)
    
    hydroatlas = pd.concat(hydroatlas_parts, ignore_index=True)
    hydroatlas = gpd.GeoDataFrame(hydroatlas, geometry='geometry', crs=hydroatlas_parts[0].crs)
    
    hydroatlas = hydroatlas.to_crs('EPSG:6933')
    hydroatlas['geometry_ea'] = hydroatlas.geometry
    hydroatlas['SUB_AREA_EA'] = hydroatlas.geometry_ea.area / 1e6
    
    hydroatlas = hydroatlas.to_crs('EPSG:4326')
    
    print(f"    Loaded {len(hydroatlas)} HydroATLAS polygons")
    
    return hydroatlas


def find_intersecting_polygons(basins, hydroatlas, min_overlap_threshold):
    """Find intersecting polygons between basins and HydroATLAS."""
    print("Finding intersecting polygons...")
    
    hydroatlas_ea = hydroatlas.to_crs('EPSG:6933')
    basins_ea = basins.to_crs('EPSG:6933')
    
    print("Validating geometries after reprojection...")
    hydroatlas_ea['geometry_ea'] = hydroatlas_ea.geometry_ea.buffer(0)
    basins_ea['geometry'] = basins_ea.geometry.buffer(0)
    
    spatial_index = hydroatlas_ea.sindex
    
    intersections = defaultdict(lambda: defaultdict(list))
    
    for idx, basin_row in tqdm(basins_ea.iterrows(), total=len(basins_ea), desc="Processing basins"):
        basin_id = basin_row[LINKNO_FIELD]
        basin_geom = basin_row.geometry
        
        #validate basin geometry
        if not basin_geom.is_valid:
            basin_geom = basin_geom.buffer(0)
        
        #store original basin area in equal-area projection
        basin_area_km2 = basin_geom.area / 1e6
        
        possible_matches_idx = list(spatial_index.intersection(basin_geom.bounds))
        possible_matches = hydroatlas_ea.iloc[possible_matches_idx]
        
        if len(possible_matches) == 0:
            continue
        
        for _, hydro_row in possible_matches.iterrows():
            hydro_geom = hydro_row.geometry_ea
            
            #validate hydro geometry
            if not hydro_geom.is_valid:
                hydro_geom = hydro_geom.buffer(0)
            
            try:
                if not basin_geom.intersects(hydro_geom):
                    continue
            except Exception as e:
                #if intersects check fails, try with buffered geometries
                try:
                    basin_geom = basin_geom.buffer(0)
                    hydro_geom = hydro_geom.buffer(0)
                    if not basin_geom.intersects(hydro_geom):
                        continue
                except:
                    print(f"Warning: Could not check intersection for basin {basin_id}: {e}")
                    continue
            
            try:
                intersection = basin_geom.intersection(hydro_geom)
                if intersection.is_empty:
                    continue
                
                overlap_area = intersection.area / 1e6
                
                if overlap_area < min_overlap_threshold:
                    continue
                
                intersections[basin_id]['overlap_areas'].append(overlap_area)
                intersections[basin_id]['area_fragments'].append(hydro_row['SUB_AREA_EA'])
                
                for col in hydro_row.index:
                    if col not in IGNORE_PROPERTIES and col not in ['geometry', 'geometry_ea', 'SUB_AREA_EA']:
                        intersections[basin_id][col].append(hydro_row[col])
                
            except Exception as e:
                print(f"Warning: Error processing intersection for basin {basin_id}: {e}")
                continue
    
    return intersections


def compute_pour_point_properties(basin_data, min_overlap_threshold, basin_id):
    """Compute pour point properties for a basin."""
    pour_point_results = {}
    overlap_areas = basin_data.get('overlap_areas', [])
    
    if not overlap_areas:
        return pour_point_results
    
    largest_overlap_idx = np.argmax(overlap_areas)
    
    for prop in POUR_POINT_PROPERTIES:
        if prop in basin_data and len(basin_data[prop]) > largest_overlap_idx:
            value = basin_data[prop][largest_overlap_idx]
            pour_point_results[prop] = value if value != -999 else np.nan
    
    return pour_point_results


def aggregate_basin_attributes(intersections, min_overlap_threshold):
    """Aggregate HydroATLAS attributes for each basin."""
    print("Aggregating attributes for basins...")
    
    aggregated_results = {}
    
    for basin_id, basin_data in tqdm(intersections.items(), desc="Aggregating basins"):
        basin_results = {LINKNO_FIELD: basin_id}
        
        overlap_areas = basin_data.get('overlap_areas', [])
        if not overlap_areas:
            aggregated_results[basin_id] = basin_results
            continue
        
        weights = np.array(overlap_areas, dtype=float)
        total_overlap_area = np.sum(weights)  # FIXED: Store total overlap area before masking
        
        mask = weights >= min_overlap_threshold
        masked_weights = weights[mask]
        
        if len(masked_weights) == 0:
            aggregated_results[basin_id] = basin_results
            continue
        
        skip_keys = set(
            ['overlap_areas', 'area_fragments'] + 
            PROCESSING_PROPERTIES + 
            POUR_POINT_PROPERTIES +
            IGNORE_PROPERTIES
        )
        
        for key, values in basin_data.items():
            if key in skip_keys:
                continue
            
            try:
                values_array = np.array(values, dtype=float)
            except (ValueError, TypeError):
                print(f"Warning: Basin {basin_id}: Skipping non-numeric property '{key}'")
                continue
            
            if len(values_array) != len(weights):
                continue
            
            masked_values = values_array[mask]
            
            if key == "wet_cl_smj":
                masked_values = np.where(masked_values == -999, 13, masked_values)
            
            valid_mask = masked_values != -999
            if not np.any(valid_mask):
                basin_results[key] = np.nan
                continue
            
            valid_values = masked_values[valid_mask]
            valid_weights = masked_weights[valid_mask]
            
            if key in MAJORITY_PROPERTIES:
                unique_values = np.unique(valid_values)
                vote_weights = np.array([
                    np.sum(valid_weights[valid_values == val]) for val in unique_values
                ])
                basin_results[key] = unique_values[np.argmax(vote_weights)]
            else:
                basin_results[key] = np.average(valid_values, weights=valid_weights)
        
        pour_point_features = compute_pour_point_properties(
            basin_data=basin_data,
            min_overlap_threshold=min_overlap_threshold,
            basin_id=basin_id
        )
        basin_results.update(pour_point_features)
        
        #Use correct column names and calculation
        basin_results['area'] = total_overlap_area
        basin_results['area_fraction_used_for_aggregation'] = (
            sum(masked_weights) / total_overlap_area if total_overlap_area > 0 else 0
        )
        
        aggregated_results[basin_id] = basin_results
    
    return aggregated_results


def format_output_dataframe(df):
    """Format the output dataframe with specified columns and order."""
    print("Formatting output dataframe...")
    
    if df.index.name == LINKNO_FIELD:
        df = df.reset_index()
    
    available_cols = [col for col in OUTPUT_COLUMNS if col in df.columns]
    missing_cols = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns in output: {missing_cols}")
    
    output_df = df[available_cols].copy()
    
    print("Rounding numeric columns to 5 decimal places...")
    for col in output_df.columns:
        if pd.api.types.is_numeric_dtype(output_df[col]):
            output_df[col] = output_df[col].round(5)
    
    return output_df


def merge_with_existing(new_df, output_file):
    """Merge new results with existing data, overwriting duplicates."""
    output_path = Path(output_file)
    
    if not output_path.exists():
        print("No existing file - writing new results directly")
        return new_df
    
    try:
        existing_df = pd.read_csv(output_file)
        print(f"Loaded {len(existing_df)} existing records")
        
        new_linknos = set(new_df[LINKNO_FIELD])
        existing_df = existing_df[~existing_df[LINKNO_FIELD].isin(new_linknos)]
        print(f"Removed {len(new_linknos)} old records to be replaced")
        
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.sort_values(LINKNO_FIELD).reset_index(drop=True)
        
        print(f"Final dataset contains {len(combined_df)} records")
        return combined_df
        
    except Exception as e:
        print(f"Warning: Could not merge with existing file: {e}")
        print("Writing new results only")
        return new_df


###---------------------------------------------------###
###                       Main                        ###
###---------------------------------------------------###

def main():
    
    try:
        validate_inputs()
        
        output_path = Path(OUTPUT_FILE).parent
        output_path.mkdir(parents=True, exist_ok=True)
        
        basins = load_basin_data()
        
        #load existing LINKNOs to skip already-processed basins
        existing_linknos = load_existing_linknos()
        
        #filter out processed basins
        if existing_linknos:
            original_count = len(basins)
            basins = basins[~basins[LINKNO_FIELD].isin(existing_linknos)]
            skipped_count = original_count - len(basins)
            print(f"\n{'='*60}")
            print(f"SKIPPING {skipped_count} already-processed LINKNOs")
            print(f"PROCESSING {len(basins)} new/remaining LINKNOs")
            print(f"{'='*60}\n")
            
            if len(basins) == 0:
                print("All basins have already been processed!")
                return
        
        #process all groups
        groups_to_process = list(PROCESSING_CONFIG.items())
        
        #collect results across groups
        all_results = {}
        
        for group_name, config in reversed(groups_to_process):
            print(f"\n{'='*60}")
            print(f"Processing group: {group_name}")
            print(f"{'='*60}")
            
            group_basins = basins.copy()
            
            if config['area_greater_than'] is not None:
                group_basins = group_basins[group_basins['area_km2'] > config['area_greater_than']]
            
            if config['area_not_greater_than'] is not None:
                group_basins = group_basins[group_basins['area_km2'] <= config['area_not_greater_than']]
            
            if len(group_basins) == 0:
                print("No basins in this group, skipping...")
                continue
            
            print(f"Processing {len(group_basins)} basins")
            
            try:
                hydroatlas = load_hydroatlas_data(config['hydroatlas_level'])
            except FileNotFoundError as e:
                print(f"Warning: Skipping group {group_name}: {e}")
                continue
            
            intersections = find_intersecting_polygons(
                basins=group_basins,
                hydroatlas=hydroatlas,
                min_overlap_threshold=config['min_overlap_threshold']
            )
            
            group_results = aggregate_basin_attributes(
                intersections=intersections,
                min_overlap_threshold=config['min_overlap_threshold']
            )
            
            #merge into all_results
            all_results.update(group_results)
            
            print(f"  Processed {len(group_results)} basins in group {group_name}")
        
        if not all_results:
            print("No new results generated!")
            return
        
        #create dataframe from new results
        print("\nCreating output for new results...")        
        df = pd.DataFrame.from_dict(all_results, orient="index")
        df = df.reset_index(drop=True)
        new_output_df = format_output_dataframe(df)
        
        #merge with existing data
        print("\nMerging with existing data...")
        final_output_df = merge_with_existing(new_output_df, OUTPUT_FILE)
        
        #save output
        final_output_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Results saved to: {OUTPUT_FILE}")
        print(f"  New LINKNOs processed: {len(new_output_df)}")
        print(f"  Total LINKNOs in file: {len(final_output_df)}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()