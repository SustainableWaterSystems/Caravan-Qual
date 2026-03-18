import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from scipy import stats
import glob

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

#input directories and files
input_dir = "/gpfs/work4/0/dynql/Caravan-Qual/"

site_info = os.path.join(input_dir, "wqms_site_info.csv")
variable_list_path = os.path.join(input_dir, "auxiliary", "wq_data", "wq_variable_list.csv")
combined_wq_file = os.path.join(input_dir, "auxiliary", "wq_data", "combined_wqms_dataset.csv")

caravan_zarr_path = os.path.join(input_dir, "Caravan.zarr")
caravan_attributes = os.path.join(input_dir, "caravan_site_info.csv")

#output directory (for .csv files)
csv_dir = os.path.join(input_dir, "wqms-csv")
os.makedirs(csv_dir, exist_ok=True)

#options
ADD_STREAMFLOW = True #include streamflow in .csvs

NPROC_CSV = 25 #number of processes for creating .csvs

ROS_MIN_DETECTS = 5 #minimum number of detects required for ROS            
ROS_MIN_DETECT_FRACTION = 0.5 #minimum proportion of detects required for ROS  

OUTLIER_IQR_MULTIPLIER = 5.0 #interquartile range multiplier for statistical outlier detection   
OUTLIER_MIN_N = 10 #minimum number of observations required to run statistical outlier detection             


###---------------------------------------------------###
###                    Functions                      ###
###---------------------------------------------------###

def round_to_sig_figs(x, sig_figs=5):
    """Round a value to a given number of significant figures."""
    
    if x == 0 or (isinstance(x, float) and np.isnan(x)):
        return x
    magnitude = np.floor(np.log10(np.abs(x)))
    return round(x, -int(magnitude) + (sig_figs - 1))


def detect_outliers(df_group, global_min, global_max, iqr_multiplier=3.0, min_n=10, log_transform=False):
    """Detect physical and statistical outliers."""

    #apply absolute physical bounds
    mask_physical = (df_group["obs"] >= global_min) & (df_group["obs"] <= global_max)
    
    if len(df_group) < min_n or mask_physical.sum() < min_n:
        return mask_physical #too few observations for statistical outlier detection (i.e. only use physical bounds)
    
    #get physically plausible values
    values = df_group.loc[mask_physical, "obs"].copy()
    
    #apply log transformation (variables defined in variable_list)
    if log_transform:
        values_for_iqr = np.log10(values + 1e-6) #add small constant to observations to avoid log(0)
    else:
        values_for_iqr = values
    
    #calculate IQR
    Q1 = values_for_iqr.quantile(0.25)
    Q3 = values_for_iqr.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR == 0:
        return mask_physical #no variation in data (i.e. keep all physically plausible values)
    
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    #apply bounds in the transformed space (log or normal)
    if log_transform:
        obs_transformed = np.log10(df_group["obs"] + 1e-6)
    else:
        obs_transformed = df_group["obs"]
    
    #combine both outlier detector criteria
    mask_statistical = (obs_transformed >= lower_bound) & (obs_transformed <= upper_bound)
    mask_combined = mask_physical & mask_statistical
    
    return mask_combined


def ros_substitution(values, is_censored, is_detected):
    """Regression on Order Statistics (ROS) imputation for left-censored observations."""

    values = np.array(values)
    is_censored = np.array(is_censored)
    is_detected = np.array(is_detected)

    #separate clean detected and censored observations
    detected = values[is_detected]
    censored_limits = values[is_censored]
    
    if len(detected) == 0:
        return values * 0.5 #no detected values (i.e. apply direct substituion: LOD/2)
    
    #log-transform detected values for regression
    log_detected = np.log(detected)
    
    #generate plotting positions for detected values
    n_detected = len(detected)
    detected_sorted_idx = np.argsort(detected)
    plotting_positions = (np.arange(1, n_detected + 1)) / (n_detected + 1)
    
    #fit linear regression
    norm_quantiles = stats.norm.ppf(plotting_positions)
    slope, intercept = np.polyfit(norm_quantiles, log_detected[detected_sorted_idx], 1)
    
    #get the rank of each censored observation rank among all observations
    all_values = np.concatenate([detected, censored_limits])
    all_censored = np.concatenate([np.zeros(len(detected), dtype=bool), 
                                   np.ones(len(censored_limits), dtype=bool)])
    
    sorted_idx = np.argsort(all_values)
    ranks = np.empty(len(all_values), dtype=int)
    ranks[sorted_idx] = np.arange(len(all_values))
    
    #compute plotting positions for censored observations
    censored_ranks = ranks[len(detected):]
    censored_plotting_pos = (censored_ranks + 1) / (len(all_values) + 1)
    
    #predict log-concentrations for censored values
    censored_norm_quantiles = stats.norm.ppf(censored_plotting_pos)
    log_censored_predicted = intercept + slope * censored_norm_quantiles
    censored_predicted = np.array([round_to_sig_figs(v, 5) for v in np.exp(log_censored_predicted)])
    
    #combine detected and imputed values
    imputed_values = values.copy()
    imputed_values[is_censored] = censored_predicted
    
    return imputed_values


def apply_ros_or_substitution(df_group):
    """Apply LOD imputation (ROS or direct substitution, depending on data avaliability"""

    if "flag" not in df_group.columns or "detection_limit" not in df_group.columns:
        return df_group, {"method": "none", "n_censored": 0, "n_detected": len(df_group)}
    
    #identify censored ("<") and outlier ("*") observations
    is_censored = (df_group["flag"] == "<").values
    is_outlier  = (df_group["flag"] == "*").values
    
    #"clean" detections (i.e. observations neither censored nor flagged as outlier)
    is_detected = ~is_censored & ~is_outlier
    n_censored  = is_censored.sum()
    n_detected  = is_detected.sum()
    n_total     = len(df_group)

    if n_censored == 0:
        return df_group, {"method": "none", "n_censored": 0, "n_detected": n_detected}

    #create imputation_method column
    df_group = df_group.copy()
    if "imputation_method" not in df_group.columns:
        df_group["imputation_method"] = ""

    #for censored observations, use detection_limit as the value for imputation
    working_values = df_group["obs"].copy()
    working_values[is_censored] = df_group.loc[is_censored, "detection_limit"]

    #check ROS criteria
    detect_fraction = n_detected / n_total
    use_ros = (n_detected >= ROS_MIN_DETECTS and
               detect_fraction >= ROS_MIN_DETECT_FRACTION)

    if use_ros:
        try:
            #apply ROS using detection limits for censored values (with observations flagged as outliers excluded)
            imputed = ros_substitution(working_values.values, is_censored, is_detected)
            df_group["obs"] = imputed
            df_group.loc[is_censored, "imputation_method"] = "ROS"
            method = "ros"
        except Exception as e:
            #otherwise, apply direct substitution (i.e. LOD/2)
            print(f"    ROS failed, using substitution: {e}")
            df_group.loc[is_censored, "obs"] = (
                df_group.loc[is_censored, "detection_limit"] * 0.5
            ).apply(lambda x: round_to_sig_figs(x, 5))
            df_group.loc[is_censored, "imputation_method"] = "LOD/2"
            method = "substitution_fallback"
    else:
        #use substitution method (LOD/2)
        df_group.loc[is_censored, "obs"] = (
            df_group.loc[is_censored, "detection_limit"] * 0.5
        ).apply(lambda x: round_to_sig_figs(x, 5))
        df_group.loc[is_censored, "imputation_method"] = "LOD/2"
        method = "substitution"
    
    stats_dict = {
        "method": method,
        "n_censored": n_censored,
        "n_detected": n_detected,
        "n_total": n_total,
        "censoring_fraction": n_censored / n_total
    }
    
    return df_group, stats_dict


###-----------------------------------------------------###
###        Clean and process water quality data         ###
###-----------------------------------------------------###

def clean_wq_data():
    """Clean raw water quality data and save to .csv (per-variable)"""
    
    #define variable limits
    var_limits = pd.read_csv(variable_list_path)
    var_limits["variable_code"] = var_limits["variable_code"].astype(str)
    
    if "log_transform" not in var_limits.columns:
        print("Warning: 'log_transform' column not found in variable list. Defaulting to False for all variables.")
        var_limits["log_transform"] = False
    
    #read in full (raw) wqms dataset
    df_all = pd.read_csv(combined_wq_file, dtype=str)
    df_all["raw_obs"] = pd.to_numeric(df_all["raw_obs"], errors="coerce")
    df_all["detection_limit"] = pd.to_numeric(df_all["detection_limit"], errors="coerce")
    df_all["dates"] = pd.to_datetime(df_all["dates"], format="mixed", dayfirst=True, errors="coerce")
    
    #create obs and unit data columns (called raw_obs and raw_unit in unprocessed dataset)
    df_all["obs"] = pd.to_numeric(df_all["raw_obs"], errors="coerce")
    df_all["unit"] = df_all["raw_unit"]
    
    #rename limit_flag to flag
    if "limit_flag" in df_all.columns:
        df_all = df_all.rename(columns={"limit_flag": "flag"})
    
    #process water quality data per (unique) variable
    variables = df_all["variable"].unique()
    
    for variable_code in variables:
        df = df_all[df_all["variable"] == variable_code].copy()
        print(f"\nProcessing variable '{variable_code}' with {len(df)} rows ...")
        original_len = len(df)
        
        #step 1: Handle censored vs. detected observations
        before = len(df)
        mask_valid = (
            df["obs"].notna() |  #detected observation
            ((df["flag"] == "<") & df["detection_limit"].notna())  #censored observation
        )
        
        df = df[mask_valid]
        dropped_invalid = before - len(df)
        print(f"  Dropped rows with invalid obs/detection_limit combination: {dropped_invalid}")
        
        #step 2: flag outliers using hybrid approach (physical bounds + per-site IQR)
        limits = var_limits[var_limits["variable_code"] == variable_code]
        
        if not limits.empty:
            plausible_min = limits["plausible_min"].values[0]
            plausible_max = limits["plausible_max"].values[0]
            
            #get log_transform setting for this variable
            use_log_transform = limits["log_transform"].values[0]
            if isinstance(use_log_transform, str):
                use_log_transform = use_log_transform.upper() in ['TRUE', 'YES', '1', 'T', 'Y']
            else:
                use_log_transform = bool(use_log_transform)
            
            #apply hybrid method per site (only on detected observations)
            outlier_mask = pd.Series(False, index=df.index)  # True = outlier
            outlier_stats = {"physical_only": 0, "statistical": 0, "physical_drops": 0, "statistical_drops": 0}
            
            for wqms_id, group in df.groupby("wqms_id"):
                #only check outliers for detected observations
                detected_mask = group["obs"].notna()
                
                if detected_mask.sum() == 0:
                    continue  #no detected observations to check
                
                detected_group = group[detected_mask]
                
                group_keep_mask = detect_outliers(
                    detected_group, 
                    plausible_min, 
                    plausible_max,
                    iqr_multiplier=OUTLIER_IQR_MULTIPLIER,
                    min_n=OUTLIER_MIN_N,
                    log_transform=use_log_transform
                )
                
                #mark outliers (where keep_mask is False) in the outlier_mask
                outlier_mask.loc[detected_group.index] = ~group_keep_mask.values
                
                #track which method was limiting
                physical_drops = ((detected_group["obs"] < plausible_min) | (detected_group["obs"] > plausible_max)).sum()
                total_drops = (~group_keep_mask).sum()
                
                if physical_drops > 0:
                    outlier_stats["physical_only"] += 1
                    outlier_stats["physical_drops"] += physical_drops
                if total_drops > physical_drops:
                    outlier_stats["statistical"] += 1
                    outlier_stats["statistical_drops"] += (total_drops - physical_drops)
            
            #flag outliers with "*" in the flag column (only for detected obs, not censored "<")
            if "flag" not in df.columns:
                df["flag"] = ""
            df.loc[outlier_mask & (df["flag"] != "<"), "flag"] = "*"
            
            flagged_plausible = outlier_mask.sum()
            log_note = " (log-transformed IQR)" if use_log_transform else ""
            print(f"  Flagged outliers outside plausible range [{plausible_min}, {plausible_max}]{log_note}: {flagged_plausible}")
            print(f"    Physical boundary violations: {outlier_stats['physical_drops']} obs from {outlier_stats['physical_only']} sites")
            print(f"    Additional statistical outliers: {outlier_stats['statistical_drops']} obs from {outlier_stats['statistical']} sites")
            
            #flag censored observations with detection limits outside plausible range
            censored_mask = (df["flag"] == "<") & df["detection_limit"].notna()
            invalid_detection_limits = censored_mask & (
                (df["detection_limit"] < plausible_min) | 
                (df["detection_limit"] > plausible_max)
            )
            df.loc[invalid_detection_limits, "flag"] = "*"
            flagged_detection_limits = invalid_detection_limits.sum()
            if flagged_detection_limits > 0:
                print(f"  Flagged censored observations with detection limits outside plausible range: {flagged_detection_limits}")
        else:
            print(f"  Warning: No plausible limits found for variable '{variable_code}' - skipping outlier flagging")
        
        #step 3: process observations with flag == "<" using ROS or substitution
        if "flag" in df.columns and "detection_limit" in df.columns:
            mask_censored = df["flag"] == "<"
            n_censored_total = mask_censored.sum()
            
            if n_censored_total > 0:
                print(f"  Processing {n_censored_total} censored observations (<)...")
                
                #track imputation methods used
                method_counts = {"ros": 0, "substitution": 0, "substitution_fallback": 0}
                censored_by_method = {"ros": 0, "substitution": 0, "substitution_fallback": 0}
                
                #apply ROS or substitution per site
                processed_groups = []
                for wqms_id, group in df.groupby("wqms_id"):
                    processed_group, stats_dict = apply_ros_or_substitution(group)
                    processed_groups.append(processed_group)
                    
                    if stats_dict["n_censored"] > 0:
                        method_counts[stats_dict["method"]] += 1
                        censored_by_method[stats_dict["method"]] += stats_dict["n_censored"]
                
                df = pd.concat(processed_groups, ignore_index=True)
                
                print(f"    ROS applied to {method_counts['ros']} sites ({censored_by_method['ros']} censored obs)")
                print(f"    Substitution applied to {method_counts['substitution']} sites ({censored_by_method['substitution']} censored obs)")
                if method_counts['substitution_fallback'] > 0:
                    print(f"    Substitution fallback for {method_counts['substitution_fallback']} sites ({censored_by_method['substitution_fallback']} censored obs)")
        
        #step 4: remove exact duplicates (i.e. same wqms_id, date and obs)
        before = len(df)
        duplicate_cols = ["variable", "wqms_id", "dates", "unit", "obs"]
        if "flag" in df.columns:
            duplicate_cols.append("flag")
        if "imputation_method" in df.columns:
            duplicate_cols.append("imputation_method")
        
        df = df.drop_duplicates(subset=duplicate_cols)
        exact_drops = before - len(df)
        print(f"  Removed exact duplicate rows: {exact_drops} dropped")
        
        #step 5: remove implausible dates (e.g. future)
        today = pd.Timestamp.today().normalize()
        before = len(df)
        df = df[df["dates"] <= today]
        future_drops = before - len(df)
        print(f"  Removed rows with future dates: {future_drops} dropped")
        
        #step 6: reorder columns
        if "flag" not in df.columns:
            df["flag"] = ""
        if "detection_limit" not in df.columns:
            df["detection_limit"] = np.nan
        if "imputation_method" not in df.columns:
            df["imputation_method"] = ""
        
        df = df[["wqms_id", "dates", "obs", "unit", "flag", "detection_limit", "imputation_method", "variable"]]
        
        #step 7: Save process water quality data per variable
        out_path = os.path.join(csv_dir, f"{variable_code}.csv")
        df.to_csv(out_path, index=False)
        
        #summary
        final_len = len(df)
        print(f"  Original rows: {original_len}")
        print(f"  Final rows:    {final_len}")
        print(f"  Total dropped: {original_len - final_len}")
        print(f"Cleaned and saved: {out_path}")


###-----------------------------------------------------###
###    Supplement water quality data with streamflow    ###
###-----------------------------------------------------###

def load_caravan_zarr(zarr_path):
    """Load Caravan zarr store and create gauge_id lookup"""
    
    print(f"Loading Caravan.zarr from {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=True)
    
    #get gauge_ids and areas
    gauge_ids = ds.gauge_id.values
    areas = ds.area.values
    
    #create lookup dictionaries
    gauge_id_to_idx = {str(gid).lower(): i for i, gid in enumerate(gauge_ids)}
    gauge_id_to_area = {str(gid).lower(): areas[i] for i, gid in enumerate(gauge_ids)}
    
    print(f"  Loaded {len(gauge_ids)} gauges from Caravan.zarr")
    
    return ds, gauge_id_to_idx, gauge_id_to_area


def process_csv_with_streamflow(param_tuple):
    """Add streamflow data to water quality .csv files"""
    
    param_name, df_param = param_tuple
    
    print(f" Processing {param_name}")
    
    df_to_process = df_param
    df_to_process = df_to_process.merge(df_sites[["wqms_id", "gauge_id"]], on="wqms_id", how="left")
    
    #collect unique gauge_ids that are needed
    unique_gauge_ids = set()
    for gauge_id, group in df_to_process.groupby("gauge_id"):
        if pd.isna(gauge_id) or gauge_id.strip() == "":
            continue
        gauge_id_lower = str(gauge_id).strip().lower()
        
        # Check if any sites in this group are valid
        valid_sites_in_group = []
        for _, row in group.iterrows():
            if (row["wqms_id"], gauge_id) in valid_site_gauge_pairs:
                valid_sites_in_group.append(row["wqms_id"])
        
        if valid_sites_in_group and gauge_id_lower in GAUGE_ID_TO_IDX:
            unique_gauge_ids.add(gauge_id_lower)
    
    #load streamflow data for needed gauges
    gauge_sf_dict = {}
    
    for gauge_id_lower in unique_gauge_ids:
        if gauge_id_lower not in GAUGE_ID_TO_IDX:
            print(f"  Gauge {gauge_id_lower} not found in Caravan.zarr")
            continue
        
        try:
            gauge_idx = GAUGE_ID_TO_IDX[gauge_id_lower]
            area = GAUGE_ID_TO_AREA.get(gauge_id_lower)
            
            if pd.isna(area):
                print(f"  Skipping gauge {gauge_id_lower} - missing area data")
                continue
            
            sf_series = DS_CARAVAN.streamflow.isel(gauge_id=gauge_idx).to_pandas() #extract streamflow for this gauge
            sf_series = (sf_series * area * 1000) / 86400 #convert from mm/day to m3/s
            
            gauge_sf_dict[gauge_id_lower] = sf_series
            
        except Exception as e:
            print(f"  Error reading streamflow for gauge {gauge_id_lower}: {e}")
            continue
    
    def get_streamflow(row):
        gauge_id = row["gauge_id"]
        wqms_id = row["wqms_id"]
        if pd.isna(gauge_id) or (wqms_id, gauge_id) not in valid_site_gauge_pairs:
            return np.nan
        gauge_id_lower = str(gauge_id).strip().lower()
        if gauge_id_lower in gauge_sf_dict:
            return gauge_sf_dict[gauge_id_lower].get(row["dates"], np.nan)
        return np.nan
    
    df_to_process["streamflow"] = df_to_process.apply(get_streamflow, axis=1).apply(
        lambda x: round_to_sig_figs(x, 5) if pd.notna(x) else x
    )
    df_to_process.drop(columns=["gauge_id"], inplace=True)
    
    columns_order = ["wqms_id", "dates", "obs", "unit", "flag", "detection_limit", "imputation_method", "variable", "streamflow"]
    columns_order = [col for col in columns_order if col in df_to_process.columns]
    df_to_process = df_to_process[columns_order]
    
    out_csv_path = os.path.join(csv_dir, f"{param_name}.csv")
    df_to_process = df_to_process.sort_values(['wqms_id', 'dates'])
    df_to_process.to_csv(out_csv_path, index=False, na_rep="")


def init_worker(global_valid_site_gauge_pairs, ds_caravan, gauge_id_to_idx, gauge_id_to_area):
    """Initialise worker process with global variables"""
    
    global valid_site_gauge_pairs, DS_CARAVAN, GAUGE_ID_TO_IDX, GAUGE_ID_TO_AREA
    valid_site_gauge_pairs = global_valid_site_gauge_pairs
    DS_CARAVAN = ds_caravan
    GAUGE_ID_TO_IDX = gauge_id_to_idx
    GAUGE_ID_TO_AREA = gauge_id_to_area


def add_streamflow_to_csvs():
    """Add streamflow data to cleaned water quality CSV files using data stored in Caravan.zarr"""
    
    print(f" Adding streamflow data to .csv files for all sites with a matching gauge_id")
    
    global df_sites, valid_site_gauge_pairs, ds_caravan, gauge_id_to_idx, gauge_id_to_area
    
    #load Caravan.zarr
    ds_caravan, gauge_id_to_idx, gauge_id_to_area = load_caravan_zarr(caravan_zarr_path)
    
    #load site information
    df_sites = pd.read_csv(site_info, dtype={"gauge_id": str})
    df_attrs = pd.read_csv(caravan_attributes, dtype={"gauge_id": str})
    df_sites = df_sites.merge(df_attrs[["gauge_id", "area"]], on="gauge_id", how="left")
    
    #all sites with a non-null gauge_id are valid
    valid_site_gauge_pairs = set()
    for _, row in df_sites.iterrows():
        if pd.notna(row["gauge_id"]) and str(row["gauge_id"]).strip() != "":
            valid_site_gauge_pairs.add((row["wqms_id"], row["gauge_id"]))
    
    valid_gauge_ids = set(gauge_id for _, gauge_id in valid_site_gauge_pairs)
    
    total_sites = len(df_sites[df_sites["gauge_id"].notna()])
    print(f"Total sites with gauge_id: {total_sites}")
    print(f"Unique gauge_ids: {len(valid_gauge_ids)}")
    
    #load all water quality data
    print(f"Updating all water quality .csv files with streamflow")
    wq_data_dict_local = {}
    for csv_file in os.listdir(csv_dir):
        if not csv_file.lower().endswith(".csv"):
            continue
        param = os.path.splitext(csv_file)[0]
        df_param = pd.read_csv(os.path.join(csv_dir, csv_file), parse_dates=["dates"], date_format="%Y-%m-%d")
        df_param["dates"] = pd.to_datetime(df_param["dates"], errors="coerce")
        for wqms_id, group in df_param.groupby("wqms_id"):
            wq_data_dict_local.setdefault(wqms_id, {})[param] = group
    
    param_dict = {}
    for wqms_id, param_dfs in wq_data_dict_local.items():
        for param, df in param_dfs.items():
            param_dict.setdefault(param, []).append(df)
    
    param_tuples = [
        (param, pd.concat(dfs, ignore_index=True)) 
        for param, dfs in param_dict.items()
    ]
    
    #process with multiprocessing, passing zarr dataset to workers
    with Pool(processes=NPROC_CSV, maxtasksperchild=1, initializer=init_worker,
              initargs=(valid_site_gauge_pairs, ds_caravan, gauge_id_to_idx, gauge_id_to_area)) as pool:
        pool.map(process_csv_with_streamflow, param_tuples)
    
    #close zarr dataset
    ds_caravan.close()


###---------------------------------------------------###
###                   Main                            ###
###---------------------------------------------------###

if __name__ == "__main__":
    
    print(f"Starting processing water quality data")
    clean_wq_data()
    print(f"Finished processing water quality data")
    
    if ADD_STREAMFLOW:
        add_streamflow_to_csvs()
        print(f"Finished adding streamflow data to .csv files")
    else:
        print(f"Streamflow not added to .csv files")