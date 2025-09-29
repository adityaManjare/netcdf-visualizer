import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class ArgoProfileSummarizer:
    def __init__(self, csv_folder="csv_data"):
        self.csv_folder = Path(csv_folder)
        self.summaries = []
        
        # Geographic regions for semantic understanding
        self.regions = {
            'equatorial': {'lat_range': (-5, 5), 'description': 'near the equator'},
            'tropical': {'lat_range': (-23.5, 23.5), 'description': 'in tropical waters'},
            'subtropical_north': {'lat_range': (23.5, 35), 'description': 'in northern subtropical waters'},
            'subtropical_south': {'lat_range': (-35, -23.5), 'description': 'in southern subtropical waters'},
            'temperate_north': {'lat_range': (35, 60), 'description': 'in northern temperate waters'},
            'temperate_south': {'lat_range': (-60, -35), 'description': 'in southern temperate waters'},
            'polar_north': {'lat_range': (60, 90), 'description': 'in northern polar waters'},
            'polar_south': {'lat_range': (-90, -60), 'description': 'in southern polar waters'},
            'arctic': {'lat_range': (66.5, 90), 'description': 'in Arctic waters'},
            'antarctic': {'lat_range': (-90, -66.5), 'description': 'in Antarctic waters'}
        }
        
        # Ocean basins (simplified longitude-based)
        self.oceans = {
            'atlantic': {'lon_range': [(-67.5, 20)], 'description': 'Atlantic Ocean'},
            'pacific': {'lon_range': [(-180, -67.5), (140, 180)], 'description': 'Pacific Ocean'},
            'indian': {'lon_range': [(20, 140)], 'description': 'Indian Ocean'},
            'southern': {'description': 'Southern Ocean'}  # Will be determined by latitude < -60
        }
        
        # Water mass characteristics
        self.water_characteristics = {
            'surface': {'depth_range': (0, 50), 'description': 'surface waters'},
            'subsurface': {'depth_range': (50, 200), 'description': 'subsurface waters'},
            'intermediate': {'depth_range': (200, 1000), 'description': 'intermediate depth waters'},
            'deep': {'depth_range': (1000, 3000), 'description': 'deep waters'},
            'abyssal': {'depth_range': (3000, 6000), 'description': 'abyssal waters'},
            'very_cold': {'temp_range': (-2, 4), 'description': 'very cold water'},
            'cold': {'temp_range': (4, 10), 'description': 'cold water'},
            'temperate': {'temp_range': (10, 20), 'description': 'temperate water'},
            'warm': {'temp_range': (20, 26), 'description': 'warm water'},
            'very_warm': {'temp_range': (26, 35), 'description': 'very warm water'},
            'fresh': {'salinity_range': (30, 34), 'description': 'relatively fresh water'},
            'normal_salinity': {'salinity_range': (34, 35), 'description': 'normal salinity water'},
            'saline': {'salinity_range': (35, 37), 'description': 'saline water'},
            'hypersaline': {'salinity_range': (37, 42), 'description': 'hypersaline water'}
        }
    
    def julian_to_datetime(self, julian_day):
        """Convert Julian day to readable datetime"""
        try:
            # Argo uses days since 1950-01-01
            base_date = datetime(1950, 1, 1)
            actual_date = base_date + pd.Timedelta(days=julian_day)
            return actual_date
        except:
            return None
    
    def classify_geographic_region(self, lat, lon):
        """Classify geographic region based on coordinates, separating ocean and climate zones."""
        climatic_zones = []
        ocean = "Unknown Ocean"
        
        # Check latitude-based regions
        for region, info in self.regions.items():
            lat_min, lat_max = info['lat_range']
            if lat_min <= lat <= lat_max:
                climatic_zones.append(info['description'])
        
        # Check ocean basins
        if lat < -60:
            ocean = self.oceans['southern']['description']
        else:
            for ocean_name, info in self.oceans.items():
                if 'lon_range' in info:
                    for lon_min, lon_max in info['lon_range']:
                        if lon_min <= lon <= lon_max:
                            ocean = info['description']
                            break
                if ocean != "Unknown Ocean":
                    break
        
        return {'ocean': ocean, 'climatic_zones': climatic_zones}
    
    def classify_water_characteristics(self, df):
        """Classify water mass characteristics"""
        characteristics = []
        
        # Depth characteristics
        if 'PRES' in df.columns:
            max_depth = df['PRES'].max()
            min_depth = df['PRES'].min()
            
            for char, info in self.water_characteristics.items():
                if 'depth_range' in info:
                    depth_min, depth_max = info['depth_range']
                    if min_depth <= depth_max and max_depth >= depth_min:
                        characteristics.append(info['description'])
        
        # Temperature characteristics
        if 'TEMP' in df.columns:
            temp_data = df['TEMP'].dropna()
            if len(temp_data) > 0:
                avg_temp = temp_data.mean()
                
                for char, info in self.water_characteristics.items():
                    if 'temp_range' in info:
                        temp_min, temp_max = info['temp_range']
                        if temp_min <= avg_temp <= temp_max:
                            characteristics.append(info['description'])
        
        # Salinity characteristics
        if 'PSAL' in df.columns:
            sal_data = df['PSAL'].dropna()
            if len(sal_data) > 0:
                avg_sal = sal_data.mean()
                
                for char, info in self.water_characteristics.items():
                    if 'salinity_range' in info:
                        sal_min, sal_max = info['salinity_range']
                        if sal_min <= avg_sal <= sal_max:
                            characteristics.append(info['description'])
        
        return list(set(characteristics))  # Remove duplicates
    
    def analyze_single_csv(self, csv_file):
        """Analyze a single CSV file and generate summary"""
        try:
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                return None
            
            # Basic information
            profile_id = df['profile_id'].iloc[0] if 'profile_id' in df.columns else 'Unknown'
            lat = df['LATITUDE'].iloc[0] if 'LATITUDE' in df.columns else None
            lon = df['LONGITUDE'].iloc[0] if 'LONGITUDE' in df.columns else None
            julian_time = df['JULD'].iloc[0] if 'JULD' in df.columns else None
            
            # Convert time
            measurement_date = None
            if julian_time and not pd.isna(julian_time):
                measurement_date = self.julian_to_datetime(julian_time)
            
            # Data statistics
            stats = {}
            if 'PRES' in df.columns:
                pres_data = df['PRES'].dropna()
                if len(pres_data) > 0:
                    stats['depth'] = {
                        'min': pres_data.min(),
                        'max': pres_data.max(),
                        'count': len(pres_data)
                    }
            
            if 'TEMP' in df.columns:
                temp_data = df['TEMP'].dropna()
                if len(temp_data) > 0:
                    stats['temperature'] = {
                        'min': temp_data.min(),
                        'max': temp_data.max(),
                        'mean': temp_data.mean(),
                        'count': len(temp_data)
                    }
            
            if 'PSAL' in df.columns:
                sal_data = df['PSAL'].dropna()
                if len(sal_data) > 0:
                    stats['salinity'] = {
                        'min': sal_data.min(),
                        'max': sal_data.max(),
                        'mean': sal_data.mean(),
                        'count': len(sal_data)
                    }
            
            # Geographic and water mass classification
            geographic_regions = []
            if lat is not None and lon is not None:
                geographic_regions = self.classify_geographic_region(lat, lon)
            
            water_characteristics = self.classify_water_characteristics(df)
            
            # Generate human-readable summary
            summary_text = self.generate_summary_text(
                profile_id, lat, lon, measurement_date, stats, 
                geographic_regions, water_characteristics, csv_file.name
            )
            
            return {
                'file': csv_file.name,
                'profile_id': profile_id,
                'latitude': lat,
                'longitude': lon,
                'measurement_date': measurement_date.isoformat() if measurement_date else None,
                'julian_day': julian_time,
                'statistics': stats,
                'geographic_regions': geographic_regions,
                'water_characteristics': water_characteristics,
                'summary_text': summary_text,
                'data_points': len(df)
            }
            
        except Exception as e:
            return None
    
    def generate_summary_text(self, profile_id, lat, lon, date, stats, regions, characteristics, filename):
        """Generate a human-readable summary paragraph with emphasis on extremes and location."""
        
        # Start with basic info and date
        summary = f"Profile {profile_id}"
        if date:
            summary += f", collected on {date.strftime('%B %d, %Y')},"
        else:
            summary += ","
            
        # Add detailed location information
        if lat is not None and lon is not None:
            lat_dir = "N" if lat >= 0 else "S"
            lon_dir = "E" if lon >= 0 else "W"
            ocean = regions.get('ocean', 'an unknown ocean')
            climatic_zones = regions.get('climatic_zones', [])
            
            summary += f" was recorded in the {ocean} at coordinates {abs(lat):.3f}°{lat_dir}, {abs(lon):.3f}°{lon_dir}"
            
            if climatic_zones:
                summary += f", within {' and '.join(climatic_zones)}"
        
        summary += ". "
        
        # Add depth information with emphasis on min/max
        if 'depth' in stats:
            depth_stats = stats['depth']
            summary += (f"The profile contains {depth_stats['count']} measurements, spanning from a minimum depth "
                        f"of {depth_stats['min']:.1f}m to a maximum of {depth_stats['max']:.1f}m. ")
        
        # Add water characteristics
        if characteristics:
            summary += f"The water column includes characteristics of {', '.join(characteristics[:3])}. "
        
        # Add temperature information with emphasis on min/max
        if 'temperature' in stats:
            temp_stats = stats['temperature']
            summary += (f"A significant temperature gradient was observed, with a maximum of {temp_stats['max']:.2f}°C "
                        f"and a minimum of {temp_stats['min']:.2f}°C (average: {temp_stats['mean']:.2f}°C). ")
        
        # Add salinity information with emphasis on min/max
        if 'salinity' in stats:
            sal_stats = stats['salinity']
            summary += (f"Salinity extremes were recorded between a minimum of {sal_stats['min']:.2f} PSU and a "
                        f"maximum of {sal_stats['max']:.2f} PSU (average: {sal_stats['mean']:.2f} PSU). ")
        
        summary += f"Data source: {filename}."
        
        return summary
    
    def process_all_csvs(self):
        """Process all CSV files in the folder"""
        if not self.csv_folder.exists():
            print(f"CSV folder not found: {self.csv_folder}")
            return 0
        
        csv_files = list(self.csv_folder.glob("*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.csv_folder}")
            return 0
        
        processed_count = 0
        for csv_file in csv_files:
            summary = self.analyze_single_csv(csv_file)
            
            if summary:
                self.summaries.append(summary)
                processed_count += 1
        
        return processed_count
    
    def save_summaries(self, output_file="argo_summaries.json"):
        """Save summaries to JSON file"""
        if not self.summaries:
            print("No summaries to save!")
            return
        
        with open(output_file, 'w') as f:
            json.dump(self.summaries, f, indent=2, default=str)
        
        print(f"Summaries saved to {output_file}")

# Main execution
if __name__ == "__main__":
    # Initialize summarizer
    summarizer = ArgoProfileSummarizer("parquets")
    
    # Process all CSV files
    processed = summarizer.process_all_csvs()
    
    if processed > 0:
        # Save summaries
        summarizer.save_summaries("argo_summaries.json")
        print(f"Successfully processed {processed} profiles")
    else:
        print("No profiles were processed successfully")