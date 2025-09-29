import xarray as xr
import pandas as pd
import numpy as np
import os
from pathlib import Path

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

def create_dataframe(data_dict, filename):
    """Create DataFrame from extracted data"""
    main_vars = ['PRES', 'TEMP', 'PSAL']
    available_vars = [v for v in main_vars if v in data_dict]
    
    if not available_vars:
        return None
    
    first_var = available_vars[0]
    data_shape = data_dict[first_var].shape
    rows = []
    
    if len(data_shape) == 2:  # Multiple profiles
        n_prof, n_levels = data_shape
        
        for prof in range(n_prof):
            for level in range(n_levels):
                row = {
                    'filename': filename,
                    'profile': prof, 
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
                for coord_var in ['LATITUDE', 'LONGITUDE', 'JULD']:
                    if coord_var in data_dict:
                        try:
                            if data_dict[coord_var].ndim >= 1:
                                row[coord_var] = float(data_dict[coord_var][prof])
                            else:
                                row[coord_var] = float(data_dict[coord_var])
                        except:
                            row[coord_var] = np.nan
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Remove invalid rows
    valid_rows = pd.Series([False] * len(df))
    for var in available_vars:
        if var in df.columns:
            valid_mask = df[var].notna() & (df[var] != -99999) & (df[var] != 99999)
            valid_rows = valid_rows | valid_mask
    
    return df[valid_rows] if valid_rows.any() else None

def create_profile_id(lat, lon, time):
    """Create unique profile ID based on location and time"""
    # Round coordinates to avoid tiny differences
    lat_rounded = round(float(lat), 3)
    lon_rounded = round(float(lon), 3)
    
    # Handle time (could be julian day or other format)
    try:
        time_rounded = round(float(time), 2)
    except:
        time_rounded = str(time)[:10]  # Take first 10 chars if string
    
    return f"LAT{lat_rounded}_LON{lon_rounded}_TIME{time_rounded}"

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
    
    # Dictionary to store data by unique profile ID
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
        
        # Create DataFrame
        df = create_dataframe(argo_data, nc_file.name)
        if df is not None and len(df) > 0:
            print(f"  ✓ Extracted {len(df)} data points")
            
            # Create unique profile IDs for each profile in this file
            unique_profiles = {}
            
            for _, row in df.iterrows():
                if pd.notna(row.get('LATITUDE')) and pd.notna(row.get('LONGITUDE')):
                    lat = row['LATITUDE']
                    lon = row['LONGITUDE']
                    time = row.get('JULD', 'UNKNOWN')
                    
                    # Create unique profile ID
                    profile_id = create_profile_id(lat, lon, time)
                    
                    if profile_id not in unique_profiles:
                        unique_profiles[profile_id] = []
                    unique_profiles[profile_id].append(row)
            
            # Add to main profile data dictionary
            for profile_id, rows in unique_profiles.items():
                if profile_id not in profile_data:
                    profile_data[profile_id] = []
                
                # Convert rows to DataFrame
                profile_df = pd.DataFrame(rows)
                profile_data[profile_id].append(profile_df)
                
            print(f"    Found {len(unique_profiles)} unique profiles in this file")
        else:
            print(f"  ❌ No valid data in {nc_file.name}")
        
        dataset.close()
    
    # Save combined data for each unique profile
    if profile_data:
        print("=" * 50)
        print("Combining unique profiles and saving CSV files...")
        
        saved_count = 0
        for profile_id, dataframes in profile_data.items():
            # Combine all dataframes for this unique profile
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Create safe filename (replace problematic characters)
            safe_filename = profile_id.replace('.', 'p').replace('-', 'n')
            csv_filename = f"{safe_filename}.csv"
            csv_path = output_path / csv_filename
            
            # Save CSV
            combined_df.to_csv(csv_path, index=False)
            
            # Show profile info
            lat = combined_df['LATITUDE'].iloc[0] if 'LATITUDE' in combined_df.columns else 'N/A'
            lon = combined_df['LONGITUDE'].iloc[0] if 'LONGITUDE' in combined_df.columns else 'N/A'
            time = combined_df['JULD'].iloc[0] if 'JULD' in combined_df.columns else 'N/A'
            
            print(f"  ✓ Profile: LAT={lat}, LON={lon}, TIME={time}")
            print(f"    -> {len(combined_df)} data points from {len(dataframes)} sources -> {csv_filename}")
            saved_count += 1
        
        print("=" * 50)
        print("✅ PROCESSING COMPLETE!")
        print(f"✅ Unique profiles created: {saved_count}")
        print(f"✅ CSV files saved in: {output_folder}/")
        
        return saved_count
    else:
        print("❌ No data could be extracted from any files")
        return 0

# Main execution
if __name__ == "__main__":
    print("Argo NetCDF to CSV Converter")
    print("=" * 30)
    
    # Process all files: netcdf_data folder -> csv_data folder
    processed = process_folder("netcdf_data", "parquets")
    
    if processed and processed > 0:
        print(f"\n✅ SUCCESS: {processed} files converted to CSV!")
        print("📁 Check the csv_data/ folder for your files")
    else:
        print("\n❌ No files were processed successfully")