import xarray as xr
import pandas as pd
import numpy as np
import os
from pathlib import Path
import hashlib

def load_argo_file(file_path):
    """Load single Argo NetCDF file"""
    try:
        # Use scipy engine (most reliable)
        ds = xr.open_dataset(file_path, engine='scipy')
        return ds
    except:
        return None

def extract_argo_data(data):
    """Extract core Argo variables"""
    var_mapping = {
        'PRES': ['pres', 'pres_adjusted'],
        'TEMP': ['temp', 'temp_adjusted'], 
        'PSAL': ['psal', 'psal_adjusted'],
        'LATITUDE': ['latitude'],
        'LONGITUDE': ['longitude'],
        'JULD': ['juld']
    }
    
    extracted = {}
    available_vars = list(data.data_vars.keys()) + list(data.coords.keys())
    
    for standard_name, possible_names in var_mapping.items():
        for var_name in possible_names:
            if var_name in available_vars:
                extracted[standard_name] = data[var_name].values
                break
    
    return extracted

def create_profile_signature(lat, lon, time, precision_lat=3, precision_lon=3, precision_time=2):
    """Create a unique signature for a profile based on location and time"""
    # Round coordinates and time to specified precision
    lat_rounded = round(float(lat), precision_lat)
    lon_rounded = round(float(lon), precision_lon)
    
    try:
        time_rounded = round(float(time), precision_time)
    except:
        time_rounded = str(time)[:10]  # Take first 10 chars if string
    
    # Create signature string
    signature = f"LAT{lat_rounded}_LON{lon_rounded}_TIME{time_rounded}"
    return signature

def generate_profile_id(signature, profile_id_counter):
    """Generate a unique profile ID using a counter"""
    if signature not in profile_id_counter:
        profile_id_counter[signature] = len(profile_id_counter) + 1
    
    return profile_id_counter[signature]

def create_dataframe(data_dict, filename, profile_id_counter):
    """Create DataFrame from extracted data with dynamic profile IDs"""
    main_vars = ['PRES', 'TEMP', 'PSAL']
    available_vars = [v for v in main_vars if v in data_dict]
    
    if not available_vars:
        return None, profile_id_counter
    
    first_var = available_vars[0]
    data_shape = data_dict[first_var].shape
    rows = []
    
    if len(data_shape) == 2:  # Multiple profiles
        n_prof, n_levels = data_shape
        
        for prof in range(n_prof):
            # First, get the location and time for this profile
            profile_lat = None
            profile_lon = None
            profile_time = None
            
            for coord_var in ['LATITUDE', 'LONGITUDE', 'JULD']:
                if coord_var in data_dict:
                    try:
                        if data_dict[coord_var].ndim >= 1:
                            value = float(data_dict[coord_var][prof])
                        else:
                            value = float(data_dict[coord_var])
                        
                        if coord_var == 'LATITUDE':
                            profile_lat = value
                        elif coord_var == 'LONGITUDE':
                            profile_lon = value
                        elif coord_var == 'JULD':
                            profile_time = value
                    except:
                        pass
            
            # Skip if we don't have valid coordinates
            if profile_lat is None or profile_lon is None:
                continue
            
            # Create profile signature and get dynamic profile ID
            profile_signature = create_profile_signature(profile_lat, profile_lon, profile_time)
            dynamic_profile_id = generate_profile_id(profile_signature, profile_id_counter)
            
            # Now process all levels for this profile
            for level in range(n_levels):
                row = {
                    'filename': filename,
                    'profile_id': dynamic_profile_id,  # Use dynamic profile ID
                    'profile_signature': profile_signature,  # Keep signature for reference
                    'level': level
                }
                
                # Add measurement variables
                for var in available_vars:
                    try:
                        value = data_dict[var][prof, level]
                        if hasattr(value, 'mask') and value.mask:
                            row[var] = np.nan
                        elif np.ma.is_masked(value):
                            row[var] = np.nan
                        else:
                            row[var] = float(value)
                    except:
                        row[var] = np.nan
                
                # Add position/time data
                row['LATITUDE'] = profile_lat if profile_lat is not None else np.nan
                row['LONGITUDE'] = profile_lon if profile_lon is not None else np.nan
                row['JULD'] = profile_time if profile_time is not None else np.nan
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Remove invalid rows
    if len(df) > 0:
        valid_rows = pd.Series([False] * len(df))
        for var in available_vars:
            if var in df.columns:
                valid_mask = df[var].notna() & (df[var] != -99999) & (df[var] != 99999)
                valid_rows = valid_rows | valid_mask
        
        return df[valid_rows] if valid_rows.any() else None, profile_id_counter
    else:
        return None, profile_id_counter

def process_folder(input_folder="netcdf_data", output_folder="csv_data"):
    """Process all NetCDF files and group by unique profile (location+time)"""
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"❌ Input folder not found: {input_folder}")
        return
    
    # Create output folder if it doesn't exist
    output_path.mkdir(exist_ok=True)
    print(f"✓ Output folder ready: {output_folder}")
    
    # Find all NetCDF files
    nc_files = list(input_path.glob("*.nc"))
    
    if not nc_files:
        print(f"❌ No .nc files found in {input_folder}")
        return
    
    print(f"Found {len(nc_files)} NetCDF files")
    print("=" * 50)
    
    # Global profile ID counter
    profile_id_counter = {}
    
    # Dictionary to store data by unique profile signature
    profile_data = {}
    
    for nc_file in nc_files:
        print(f"Processing: {nc_file.name}")
        
        # Load file
        dataset = load_argo_file(nc_file)
        if dataset is None:
            print(f"  ❌ Failed to load {nc_file.name}")
            continue
        
        # Extract data
        argo_data = extract_argo_data(dataset)
        if not argo_data:
            print(f"  ❌ No data extracted from {nc_file.name}")
            dataset.close()
            continue
        
        # Create DataFrame with dynamic profile IDs
        df, profile_id_counter = create_dataframe(argo_data, nc_file.name, profile_id_counter)
        if df is not None and len(df) > 0:
            print(f"  ✓ Extracted {len(df)} data points")
            
            # Group by profile signature
            unique_profiles = df.groupby('profile_signature')
            
            for profile_signature, profile_df in unique_profiles:
                if profile_signature not in profile_data:
                    profile_data[profile_signature] = []
                
                profile_data[profile_signature].append(profile_df)
                
            print(f"    Found {len(unique_profiles)} unique profiles in this file")
            
            # Show profile IDs assigned
            profile_ids = df['profile_id'].unique()
            print(f"    Profile IDs assigned: {sorted(profile_ids)}")
            
        else:
            print(f"  ❌ No valid data in {nc_file.name}")
        
        dataset.close()
    
    # Save combined data for each unique profile
    if profile_data:
        print("=" * 50)
        print("Combining unique profiles and saving CSV files...")
        
        saved_count = 0
        for profile_signature, dataframes in profile_data.items():
            # Combine all dataframes for this unique profile
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Ensure all rows have the same profile_id (they should, but let's be safe)
            unique_profile_ids = combined_df['profile_id'].unique()
            if len(unique_profile_ids) > 1:
                print(f"  ⚠️  Warning: Multiple profile IDs found for signature {profile_signature}: {unique_profile_ids}")
            
            profile_id = combined_df['profile_id'].iloc[0]
            
            # Sort by depth (pressure) for better organization
            if 'PRES' in combined_df.columns:
                combined_df = combined_df.sort_values('PRES')
            
            # Create filename using profile ID
            csv_filename = f"profile_{profile_id:04d}.csv"
            csv_path = output_path / csv_filename
            
            # Save CSV
            combined_df.to_csv(csv_path, index=False)
            
            # Show profile info
            lat = combined_df['LATITUDE'].iloc[0] if 'LATITUDE' in combined_df.columns else 'N/A'
            lon = combined_df['LONGITUDE'].iloc[0] if 'LONGITUDE' in combined_df.columns else 'N/A'
            time = combined_df['JULD'].iloc[0] if 'JULD' in combined_df.columns else 'N/A'
            depth_range = f"{combined_df['PRES'].min():.1f}-{combined_df['PRES'].max():.1f}m" if 'PRES' in combined_df.columns else 'N/A'
            
            print(f"  ✓ Profile ID {profile_id}: LAT={lat}, LON={lon}, TIME={time}")
            print(f"    -> {len(combined_df)} measurements, Depth range: {depth_range} -> {csv_filename}")
            saved_count += 1
        
        # Print profile ID mapping summary
        print("=" * 50)
        print("PROFILE ID MAPPING SUMMARY:")
        for signature, profile_id in sorted(profile_id_counter.items(), key=lambda x: x[1]):
            print(f"  Profile ID {profile_id}: {signature}")
        
        print("=" * 50)
        print("✅ PROCESSING COMPLETE!")
        print(f"✅ Unique profiles created: {saved_count}")
        print(f"✅ CSV files saved in: {output_folder}/")
        print(f"✅ Each CSV contains one unique profile with consistent profile_id")
        
        return saved_count
    else:
        print("❌ No data could be extracted from any files")
        return 0

# Main execution
if __name__ == "__main__":
    print("Enhanced Argo NetCDF to CSV Converter")
    print("With Dynamic Profile ID Assignment")
    print("=" * 40)
    
    # Process all files: netcdf_data folder -> csv_data folder
    processed = process_folder("netcdf_data", "parquets")
    
    if processed and processed > 0:
        print(f"\n✅ SUCCESS: {processed} files converted to CSV!")
        print("📁 Check the parquets/ folder for your files")
        print("🔢 Each CSV has a unique profile_id for the same location+time")
    else:
        print("\n❌ No files were processed successfully")