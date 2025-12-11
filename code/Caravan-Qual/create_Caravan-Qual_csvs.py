import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from netCDF4 import Dataset, date2num
from datetime import datetime
from multiprocessing import Pool, cpu_count
from scipy import stats
import glob

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

#Define input directories and files
input_dir = "/gpfs/work4/0/dynql/Caravan-Qual/"
input_dir_caravan = "/gpfs/work4/0/dynql/Caravan-Qual/auxiliary/Caravan/"

site_info = os.path.join(input_dir, "wqms_site_info.csv")
variable_list_path = os.path.join(input_dir, "auxiliary", "wq_data", "wq_variable_list.csv")
combined_wq_file = os.path.join(input_dir, "auxiliary", "wq_data", "combined_wqms_dataset.csv")

caravan_timeseries_path = os.path.join(input_dir_caravan, "timeseries", "netcdf")
caravan_attributes = os.path.join(input_dir_caravan, "attributes", "combined_attributes_other.csv")

#Output folder (for .csv files)
csv_dir = os.path.join(input_dir, "wqms-csv")
os.makedirs(csv_dir, exist_ok=True)

#Distance threshold (km) for linking to streamflow gauges
distance_threshold_km = 10.0
nproc_csv = 25
add_streamflow = True

#ROS parameters
ros_min_detects = 5            #define minimum number of detected values
ros_min_detect_fraction = 0.5  #define minimum fraction of detects (50%)

#Outlier detection parameters
outlier_iqr_multiplier = 5.0   #define IQR threshold (e.g. 5.0 = very conservative; 3.0 = conservative; 1.5 = standard)
outlier_min_n = 10             #define minimum number of observations required for performing statistical outlier detection


###---------------------------------------------------###
###                    Functions                      ###
###---------------------------------------------------###

def detect_outliers(df_group, global_min, global_max, iqr_multiplier=3.0, min_n=10, log_transform=False):

    #apply absolute physical bounds
    mask_physical = (df_group["obs"] >= global_min) & (df_group["obs"] <= global_max)
    
    if len(df_group) < min_n or mask_physical.sum() < min_n:
        #too few observations for outlier detection - only use physical bounds
        return mask_physical
    
    #get physically plausible values
    values = df_group.loc[mask_physical, "obs"].copy()
    
    #apply log transformation (variables defined in variable_list)
    if log_transform:
        #add small constant to observations to avoid log(0)
        values_for_iqr = np.log10(values + 1e-6)
    else:
        values_for_iqr = values
    
    #calculate IQR
    Q1 = values_for_iqr.quantile(0.25)
    Q3 = values_for_iqr.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR == 0:
        #no variation in data (i.e. keep all physically plausible values)
        return mask_physical
    
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


def ros_substitution(values, is_censored):

    values = np.array(values)
    is_censored = np.array(is_censored)
    
    #separate detected and LOD ("<") observations
    detected = values[~is_censored]
    censored_limits = values[is_censored]
    
    if len(detected) == 0:
        #no detected values - apply direct substituion (i.e. LOD/2)
        return values * 0.5
    
    #log-transform detected values for regression
    log_detected = np.log(detected)
    
    #generate plotting positions for detected values
    n_detected = len(detected)
    detected_sorted_idx = np.argsort(detected)
    plotting_positions = (np.arange(1, n_detected + 1)) / (n_detected + 1)
    
    #fit linear regression: log(concentration) ~ Normal quantiles
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
    censored_predicted = np.exp(log_censored_predicted)
    
    #combine detected and imputed values
    imputed_values = values.copy()
    imputed_values[is_censored] = censored_predicted
    
    return imputed_values


def apply_ros_or_substitution(df_group):

    if "limit_flag" not in df_group.columns:
        return df_group, {"method": "none", "n_censored": 0, "n_detected": len(df_group)}
    
    is_censored = (df_group["limit_flag"] == "<").values
    n_censored = is_censored.sum()
    n_detected = (~is_censored).sum()
    n_total = len(df_group)
    
    if n_censored == 0:
        return df_group, {"method": "none", "n_censored": 0, "n_detected": n_detected}
    
    #check ROS criteria
    detect_fraction = n_detected / n_total
    use_ros = (n_detected >= ros_min_detects and 
               detect_fraction >= ros_min_detect_fraction)
    
    if use_ros:
        try:
            #apply ROS
            imputed = ros_substitution(df_group["obs"].values, is_censored)
            df_group = df_group.copy()
            df_group["obs"] = imputed
            method = "ros"
        except Exception as e:
            #otherwise, apply direct subsitituion (i.e. LOD/2)
            print(f"    ROS failed, using substitution: {e}")
            df_group = df_group.copy()
            df_group.loc[is_censored, "obs"] = df_group.loc[is_censored, "obs"] * 0.5
            method = "substitution_fallback"
    else:
        #use substitution method
        df_group = df_group.copy()
        df_group.loc[is_censored, "obs"] = df_group.loc[is_censored, "obs"] * 0.5
        method = "substitution"
    
    stats_dict = {
        "method": method,
        "n_censored": n_censored,
        "n_detected": n_detected,
        "n_total": n_total,
        "censoring_fraction": n_censored / n_total
    }
    
    return df_group, stats_dict


###---------------------------------------------------###
###        Clean and process water quality data       ###
###---------------------------------------------------###

def clean_wq_data():
    """Clean raw water quality data and save to .csv (per-variable)"""
    
    #Define variable limits
    var_limits = pd.read_csv(variable_list_path)
    var_limits["variable_code"] = var_limits["variable_code"].astype(str)
    
    if "log_transform" not in var_limits.columns:
        print("Warning: 'log_transform' column not found in variable list. Defaulting to False for all variables.")
        var_limits["log_transform"] = False
    
    #read in full (raw) wqms dataset
    df_all = pd.read_csv(combined_wq_file, dtype=str)
    df_all["raw_obs"] = pd.to_numeric(df_all["raw_obs"], errors="coerce")
    df_all["dates"] = pd.to_datetime(df_all["dates"], format="%d/%m/%Y", errors="coerce")
    
    #create obs and unit data columns (called raw_obs and raw_unit in unprocessed dataset)
    df_all["obs"] = pd.to_numeric(df_all["raw_obs"], errors="coerce")
    df_all["unit"] = df_all["raw_unit"]
    
    #process water quality data per (unique) variable
    variables = df_all["variable"].unique()
    
    for variable_code in variables:
        df = df_all[df_all["variable"] == variable_code].copy()
        print(f"\nProcessing variable '{variable_code}' with {len(df)} rows ...")
        original_len = len(df)
        
        #Step 1: remove rows with NA in obs
        before = len(df)
        df = df.dropna(subset=["obs"])
        dropped_na = before - len(df)
        print(f"  Dropped NA in obs: {dropped_na}")
        
        #Step 2: remove outliers using hybrid approach (physical bounds + per-site IQR)
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
            
            before = len(df)
            
            #apply hybrid method per site
            keep_mask = pd.Series(False, index=df.index)
            outlier_stats = {"physical_only": 0, "statistical": 0, "physical_drops": 0, "statistical_drops": 0}
            
            for wqms_id, group in df.groupby("wqms_id"):
                group_mask = detect_outliers(
                    group, 
                    plausible_min, 
                    plausible_max,
                    iqr_multiplier=outlier_iqr_multiplier,
                    min_n=outlier_min_n,
                    log_transform=use_log_transform
                )
                keep_mask.loc[group.index] = group_mask.values
                
                #track which method was limiting
                physical_drops = ((group["obs"] < plausible_min) | (group["obs"] > plausible_max)).sum()
                total_drops = (~group_mask).sum()
                
                if physical_drops > 0:
                    outlier_stats["physical_only"] += 1
                    outlier_stats["physical_drops"] += physical_drops
                if total_drops > physical_drops:
                    outlier_stats["statistical"] += 1
                    outlier_stats["statistical_drops"] += (total_drops - physical_drops)
            
            df = df[keep_mask]
            dropped_plausible = before - len(df)
            
            log_note = " (log-transformed IQR)" if use_log_transform else ""
            print(f"  Dropped outliers outside plausible range [{plausible_min}, {plausible_max}]{log_note}: {dropped_plausible}")
            print(f"    Physical boundary violations: {outlier_stats['physical_drops']} obs from {outlier_stats['physical_only']} sites")
            print(f"    Additional statistical outliers: {outlier_stats['statistical_drops']} obs from {outlier_stats['statistical']} sites")
        else:
            print(f"  Warning: No plausible limits found for variable '{variable_code}' - skipping outlier filtering")
        
        #Step 3: process observations with limit_flag == "<" using ROS or substitution
        if "limit_flag" in df.columns:
            mask_censored = df["limit_flag"] == "<"
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
        
        #Step 4: remove exact duplicates (i.e. same wqms_id, date and obs) and average where multiple (different) obs per day (i.e. same wqms_id, date)
        before = len(df)
        df = df.drop_duplicates(subset=["variable", "wqms_id", "dates", "unit", "obs"])
        exact_drops = before - len(df)
        print(f"  Removed exact duplicate rows: {exact_drops} dropped")
        
        before = len(df)
        df = df.groupby(["variable", "wqms_id", "dates", "unit"], as_index=False).agg({"obs": "mean"})
        grouped_drops = before - len(df)
        print(f"  Combined duplicate site/date rows with differing obs: {grouped_drops} merged")
        
        #Step 5: remove implausible dates (e.g. future)
        today = pd.Timestamp.today().normalize()
        before = len(df)
        df = df[df["dates"] <= today]
        future_drops = before - len(df)
        print(f"  Removed rows with future dates: {future_drops} dropped")
        
        #Step 6: reorder columns
        df = df[["wqms_id", "dates", "obs", "unit", "variable"]]
        
        #Step 7. Save process water quality data per variable
        out_path = os.path.join(csv_dir, f"{variable_code}.csv")
        df.to_csv(out_path, index=False)
        
        # Final summary
        final_len = len(df)
        print(f"  Original rows: {original_len}")
        print(f"  Final rows:    {final_len}")
        print(f"  Total dropped: {original_len - final_len}")
        print(f"Cleaned and saved: {out_path}")


###-----------------------------------------------------###
###    Supplement water quality data with streamflow    ###
###-----------------------------------------------------###

def build_netcdf_file_map(base_path):
    """Build a mapping of gauge_id to NetCDF file paths"""
    
    netcdf_files = glob.glob(os.path.join(base_path, "**/*.nc"), recursive=True)
    
    gauge_to_path = {}
    for file_path in netcdf_files:
        gauge_id = os.path.splitext(os.path.basename(file_path))[0]
        gauge_to_path[gauge_id] = file_path
    
    print(f"Found {len(gauge_to_path)} NetCDF files in directory structure")
    return gauge_to_path


def process_csv_with_streamflow(param_tuple):
    """Add streamflow data to water quality .csv files"""
    
    param_name, df_param = param_tuple
    
    print(f" Processing {param_name}")
    
    df_to_process = df_param
    df_to_process = df_to_process.merge(df_sites[["wqms_id", "gauge_id", "area"]], on="wqms_id", how="left")
    
    gauge_sf_dict = {}
    skipped_gauges = []
    
    for gauge_id, group in df_to_process.groupby("gauge_id"):
        if pd.isna(gauge_id) or gauge_id.strip() == "":
            continue
        gauge_id = str(gauge_id).strip().lower()
        
        valid_sites_in_group = []
        for _, row in group.iterrows():
            if (row["wqms_id"], gauge_id) in valid_site_gauge_pairs:
                valid_sites_in_group.append(row["wqms_id"])
        
        if not valid_sites_in_group:
            continue
        
        #check area is avaliable (required for converting mm/day to m3/s) and that streamflow file exists
        area_val = group["area"].iloc[0]
        if pd.isna(area_val):
            skipped_gauges.append(gauge_id)
            print(f"  Skipping gauge {gauge_id} - missing area data")
            continue
        
        if gauge_id not in NETCDF_FILE_MAP:
            print(f"  Streamflow file not found for gauge_id: {gauge_id}")
            continue
        
        nc_file = NETCDF_FILE_MAP[gauge_id]
        
        try:
            with Dataset(nc_file, "r") as nc:
                if "streamflow" not in nc.variables or "date" not in nc.variables:
                    print(f"No required variables in {nc_file}")
                    continue
                
                date_var = nc.variables["date"]
                
                if not hasattr(date_var, 'units'):
                    raise ValueError(f"NetCDF file for gauge {gauge_id} has no 'units' attribute for date variable")
                
                units = date_var.units
                
                if 'since' not in units.lower():
                    raise ValueError(f"Unexpected time units format for gauge {gauge_id}: '{units}'. Expected format like 'days since YYYY-MM-DD'")
                
                ref_date_str = units.lower().split('since')[-1].strip()
                try:
                    ref_date = pd.to_datetime(ref_date_str)
                except Exception as e:
                    raise ValueError(f"Could not parse reference date from units '{units}' for gauge {gauge_id}: {e}")
                
                times = date_var[:]
                dates_nc = pd.to_datetime([ref_date + pd.Timedelta(days=int(t)) for t in times])
                
                #Check for plausible date range
                if dates_nc.min().year < 1900 or dates_nc.max().year > 2026:
                    print(f"Warning: Suspicious date range for {gauge_id}: {dates_nc.min()} to {dates_nc.max()}")
                
                sf_values = nc.variables["streamflow"][:].filled(np.nan)
                
                #convert mm/day to m3/s
                sf_values = (sf_values * area_val * 1000) / 86400
                
                gauge_sf_dict[gauge_id] = pd.Series(sf_values, index=dates_nc)
        
        except Exception as e:
            print(f"Error reading {nc_file}: {e}")
            raise
    
    def get_streamflow(row):
        gauge_id = row["gauge_id"]
        wqms_id = row["wqms_id"]
        if pd.isna(gauge_id) or (wqms_id, gauge_id) not in valid_site_gauge_pairs:
            return np.nan
        if gauge_id in gauge_sf_dict:
            return gauge_sf_dict[gauge_id].get(row["dates"], np.nan)
        return np.nan
    
    df_to_process["streamflow"] = df_to_process.apply(get_streamflow, axis=1)
    df_to_process.drop(columns=["gauge_id", "area"], inplace=True)
    
    out_csv_path = os.path.join(csv_dir, f"{param_name}.csv")
    df_to_process = df_to_process.sort_values(['wqms_id', 'dates'])
    df_to_process.to_csv(out_csv_path, index=False, na_rep="")


def init_worker(global_valid_site_gauge_pairs):
    """Initialise worker process with global variables"""
    global valid_site_gauge_pairs
    valid_site_gauge_pairs = global_valid_site_gauge_pairs


def add_streamflow_to_csvs():
    """Add streamflow data to cleaned water quality CSV files"""
    
    print(f" Adding streamflow data to .csv files, where there is a matching gauge_id within {distance_threshold_km} km")
    
    global df_sites, valid_site_gauge_pairs
    df_sites = pd.read_csv(site_info, dtype={"gauge_id": str})
    df_attrs = pd.read_csv(caravan_attributes, dtype={"gauge_id": str})
    df_sites = df_sites.merge(df_attrs[["gauge_id", "area"]], on="gauge_id", how="left")
    
    #apply filter by distance_threshold for gauge_ids
    valid_site_gauge_pairs = set()
    for _, row in df_sites.iterrows():
        if (pd.notna(row["gauge_id"]) and 
            pd.notna(row["gauge_distance_km"]) and 
            row["gauge_distance_km"] < distance_threshold_km):
            valid_site_gauge_pairs.add((row["wqms_id"], row["gauge_id"]))
    
    valid_gauge_ids = set(gauge_id for _, gauge_id in valid_site_gauge_pairs)
    
    total_sites = len(df_sites[df_sites["gauge_id"].notna()])
    valid_sites = len(valid_site_gauge_pairs)
    
    print(f"Total sites with gauge_id: {total_sites}")
    print(f"Site-gauge pairs within {distance_threshold_km} km threshold: {valid_sites}")
    print(f"Unique gauge_ids with at least one site within threshold: {len(valid_gauge_ids)}")
    
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
    
    with Pool(processes=nproc_csv, maxtasksperchild=1, initializer=init_worker,
              initargs=(valid_site_gauge_pairs,)) as pool:
        pool.map(process_csv_with_streamflow, param_tuples)


###---------------------------------------------------###
###                   Main                            ###
###---------------------------------------------------###

if __name__ == "__main__":
    
    clean_wq_data()
    print(f"Finished processing water quality data")
    
    if add_streamflow:
        NETCDF_FILE_MAP = build_netcdf_file_map(caravan_timeseries_path)
        NETCDF_FILE_MAP = {k.lower(): v for k, v in NETCDF_FILE_MAP.items()}
        add_streamflow_to_csvs()
        print(f"Finished adding streamflow data to .csv files")
    else:
        print(f"Streamflow not added to .csv files")