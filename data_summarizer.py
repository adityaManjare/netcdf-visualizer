import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class ArgoProfileSummarizer:
    def __init__(self, csv_folder="parquets"):
        self.csv_folder = Path(csv_folder)
        self.summaries = []
        
        # Geographic regions for semantic understanding
        self.regions = {
            'equatorial': {'lat_range': (-5, 5), 'keywords': ['equatorial', 'equator', 'tropical convergence']},
            'tropical': {'lat_range': (-23.5, 23.5), 'keywords': ['tropical', 'warm waters', 'low latitude']},
            'subtropical_north': {'lat_range': (23.5, 35), 'keywords': ['subtropical', 'northern subtropical', 'mid-latitude']},
            'subtropical_south': {'lat_range': (-35, -23.5), 'keywords': ['subtropical', 'southern subtropical', 'mid-latitude']},
            'temperate_north': {'lat_range': (35, 60), 'keywords': ['temperate', 'northern temperate', 'mid-latitude']},
            'temperate_south': {'lat_range': (-60, -35), 'keywords': ['temperate', 'southern temperate', 'mid-latitude']},
            'polar_north': {'lat_range': (60, 90), 'keywords': ['polar', 'arctic', 'high latitude', 'cold']},
            'polar_south': {'lat_range': (-90, -60), 'keywords': ['polar', 'antarctic', 'high latitude', 'cold']},
            'arctic': {'lat_range': (66.5, 90), 'keywords': ['arctic', 'polar', 'ice-covered', 'extremely cold']},
            'antarctic': {'lat_range': (-90, -66.5), 'keywords': ['antarctic', 'polar', 'ice-covered', 'extremely cold']}
        }
        
        # Ocean basins with keywords
        self.oceans = {
            'atlantic': {'lon_range': [(-67.5, 20)], 'keywords': ['Atlantic', 'Atlantic Ocean']},
            'pacific': {'lon_range': [(-180, -67.5), (140, 180)], 'keywords': ['Pacific', 'Pacific Ocean']},
            'indian': {'lon_range': [(20, 140)], 'keywords': ['Indian', 'Indian Ocean']},
            'southern': {'keywords': ['Southern Ocean', 'Antarctic Ocean']}
        }
        
        # Enhanced water characteristics with search-friendly terms
        self.water_characteristics = {
            'surface': {'depth_range': (0, 50), 'keywords': ['surface', 'shallow', 'epipelagic']},
            'subsurface': {'depth_range': (50, 200), 'keywords': ['subsurface', 'upper ocean']},
            'intermediate': {'depth_range': (200, 1000), 'keywords': ['intermediate', 'mesopelagic', 'twilight zone']},
            'deep': {'depth_range': (1000, 3000), 'keywords': ['deep', 'bathypelagic', 'deep water']},
            'abyssal': {'depth_range': (3000, 6000), 'keywords': ['abyssal', 'deep ocean', 'ocean floor']},
            'very_cold': {'temp_range': (-2, 4), 'keywords': ['very cold', 'freezing', 'ice-cold', 'polar water']},
            'cold': {'temp_range': (4, 10), 'keywords': ['cold', 'cool water']},
            'moderate': {'temp_range': (10, 20), 'keywords': ['moderate temperature', 'temperate water']},
            'warm': {'temp_range': (20, 26), 'keywords': ['warm', 'tropical water']},
            'very_warm': {'temp_range': (26, 35), 'keywords': ['very warm', 'hot', 'tropical surface water']},
            'low_salinity': {'salinity_range': (30, 34), 'keywords': ['low salinity', 'fresh', 'diluted']},
            'normal_salinity': {'salinity_range': (34, 35.5), 'keywords': ['normal salinity', 'typical salinity']},
            'high_salinity': {'salinity_range': (35.5, 37), 'keywords': ['high salinity', 'saline', 'salty']},
            'very_high_salinity': {'salinity_range': (37, 42), 'keywords': ['very high salinity', 'hypersaline', 'extremely salty']}
        }
    
    def julian_to_datetime(self, julian_day):
        """Convert Julian day to readable datetime"""
        try:
            base_date = datetime(1950, 1, 1)
            actual_date = base_date + pd.Timedelta(days=julian_day)
            return actual_date
        except:
            return None
    
    def classify_geographic_region(self, lat, lon):
        """Classify geographic region with enhanced keywords"""
        climatic_zones = []
        ocean_keywords = []
        
        # Check latitude-based regions
        for region, info in self.regions.items():
            lat_min, lat_max = info['lat_range']
            if lat_min <= lat <= lat_max:
                climatic_zones.extend(info['keywords'])
        
        # Check ocean basins
        if lat < -60:
            ocean_keywords = self.oceans['southern']['keywords']
        else:
            for ocean_name, info in self.oceans.items():
                if 'lon_range' in info:
                    for lon_min, lon_max in info['lon_range']:
                        if lon_min <= lon <= lon_max:
                            ocean_keywords = info['keywords']
                            break
                if ocean_keywords:
                    break
        
        return {
            'climatic_zones': climatic_zones,
            'ocean_keywords': ocean_keywords
        }
    
    def classify_water_characteristics(self, df):
        """Enhanced water mass classification with keywords"""
        characteristic_keywords = []
        
        # Depth characteristics
        if 'PRES' in df.columns:
            depth_data = df['PRES'].dropna()
            if len(depth_data) > 0:
                max_depth = depth_data.max()
                min_depth = depth_data.min()
                avg_depth = depth_data.mean()
                
                for char, info in self.water_characteristics.items():
                    if 'depth_range' in info:
                        depth_min, depth_max = info['depth_range']
                        if min_depth <= depth_max and max_depth >= depth_min:
                            characteristic_keywords.extend(info['keywords'])
        
        # Temperature characteristics
        if 'TEMP' in df.columns:
            temp_data = df['TEMP'].dropna()
            if len(temp_data) > 0:
                avg_temp = temp_data.mean()
                min_temp = temp_data.min()
                max_temp = temp_data.max()
                
                for char, info in self.water_characteristics.items():
                    if 'temp_range' in info:
                        temp_min, temp_max = info['temp_range']
                        if temp_min <= avg_temp <= temp_max:
                            characteristic_keywords.extend(info['keywords'])
        
        # Salinity characteristics
        if 'PSAL' in df.columns:
            sal_data = df['PSAL'].dropna()
            if len(sal_data) > 0:
                avg_sal = sal_data.mean()
                min_sal = sal_data.min()
                max_sal = sal_data.max()
                
                for char, info in self.water_characteristics.items():
                    if 'salinity_range' in info:
                        sal_min, sal_max = info['salinity_range']
                        if sal_min <= avg_sal <= sal_max:
                            characteristic_keywords.extend(info['keywords'])
        
        return list(set(characteristic_keywords))
    
    def generate_extreme_tags(self, stats):
        """Generate tags for extreme values to help with queries like 'highest salinity'"""
        tags = []
        
        if 'temperature' in stats:
            temp = stats['temperature']
            if temp['max'] > 28:
                tags.append("extremely high temperature")
            elif temp['max'] > 25:
                tags.append("high temperature")
            if temp['min'] < 2:
                tags.append("extremely low temperature")
            elif temp['min'] < 5:
                tags.append("low temperature")
            if temp['max'] - temp['min'] > 20:
                tags.append("large temperature gradient")
        
        if 'salinity' in stats:
            sal = stats['salinity']
            if sal['max'] > 36.5:
                tags.append("extremely high salinity")
            elif sal['max'] > 35.5:
                tags.append("high salinity")
            if sal['min'] < 33:
                tags.append("extremely low salinity")
            elif sal['min'] < 34:
                tags.append("low salinity")
            if sal['max'] - sal['min'] > 2:
                tags.append("large salinity gradient")
        
        if 'depth' in stats:
            depth = stats['depth']
            if depth['max'] > 4000:
                tags.append("very deep profile")
            elif depth['max'] > 2000:
                tags.append("deep profile")
            elif depth['max'] < 100:
                tags.append("shallow profile")
        
        return tags
    
    def generate_coordinate_tags(self, lat, lon):
        """Generate coordinate-based tags for geographic queries"""
        tags = []
        
        if lat is not None and lon is not None:
            # Round coordinates for better matching
            lat_rounded = round(lat, 1)
            lon_rounded = round(lon, 1)
            
            tags.extend([
                f"latitude {lat_rounded}",
                f"longitude {lon_rounded}",
                f"coordinates {lat_rounded} {lon_rounded}",
                f"position {abs(lat_rounded)}°{'N' if lat >= 0 else 'S'} {abs(lon_rounded)}°{'E' if lon >= 0 else 'W'}"
            ])
            
            # Add broader regional tags
            lat_band = int(lat // 10) * 10
            lon_band = int(lon // 10) * 10
            tags.extend([
                f"latitude band {lat_band} to {lat_band + 10}",
                f"longitude band {lon_band} to {lon_band + 10}"
            ])
        
        return tags
    
    def analyze_single_csv(self, csv_file):
        """Analyze a single CSV file and generate optimized summary"""
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
                        'mean': pres_data.mean(),
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
            
            # Enhanced classification
            geographic_info = {}
            if lat is not None and lon is not None:
                geographic_info = self.classify_geographic_region(lat, lon)
            
            water_keywords = self.classify_water_characteristics(df)
            extreme_tags = self.generate_extreme_tags(stats)
            coordinate_tags = self.generate_coordinate_tags(lat, lon)
            
            # Generate optimized summary
            summary_text = self.generate_optimized_summary_text(
                profile_id, lat, lon, measurement_date, stats, 
                geographic_info, water_keywords, extreme_tags, 
                coordinate_tags, csv_file.name
            )
            
            return {
                'file': csv_file.name,
                'profile_id': profile_id,
                'latitude': lat,
                'longitude': lon,
                'measurement_date': measurement_date.isoformat() if measurement_date else None,
                'julian_day': julian_time,
                'statistics': stats,
                'geographic_keywords': geographic_info,
                'water_characteristics': water_keywords,
                'extreme_tags': extreme_tags,
                'coordinate_tags': coordinate_tags,
                'summary_text': summary_text,
                'data_points': len(df)
            }
            
        except Exception as e:
            return None
    
    def generate_optimized_summary_text(self, profile_id, lat, lon, date, stats, 
                                       geographic_info, water_keywords, extreme_tags, 
                                       coordinate_tags, filename):
        """Generate search-optimized summary text"""
        
        # Start with searchable identifiers
        summary_parts = []
        
        # Profile and location information (highly searchable)
        if lat is not None and lon is not None:
            lat_str = f"{lat:.3f}"
            lon_str = f"{lon:.3f}"
            summary_parts.append(f"Profile {profile_id} at latitude {lat_str} longitude {lon_str}")
            summary_parts.append(f"Location {abs(lat):.1f}°{'N' if lat >= 0 else 'S'} {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}")
        else:
            summary_parts.append(f"Profile {profile_id}")
        
        # Date information
        if date:
            summary_parts.append(f"Measured {date.strftime('%B %Y')}")
            summary_parts.append(f"Date {date.strftime('%Y-%m-%d')}")
        
        # Geographic and oceanographic context
        if geographic_info:
            if 'ocean_keywords' in geographic_info:
                summary_parts.extend(geographic_info['ocean_keywords'])
            if 'climatic_zones' in geographic_info:
                summary_parts.extend(geographic_info['climatic_zones'])
        
        # Water characteristics
        summary_parts.extend(water_keywords)
        
        # Statistical information (key for range queries)
        if 'temperature' in stats:
            temp = stats['temperature']
            summary_parts.extend([
                f"Temperature range {temp['min']:.1f} to {temp['max']:.1f} degrees Celsius",
                f"Average temperature {temp['mean']:.1f}°C",
                f"Minimum temperature {temp['min']:.1f}°C",
                f"Maximum temperature {temp['max']:.1f}°C"
            ])
        
        if 'salinity' in stats:
            sal = stats['salinity']
            summary_parts.extend([
                f"Salinity range {sal['min']:.2f} to {sal['max']:.2f} PSU",
                f"Average salinity {sal['mean']:.2f} PSU",
                f"Minimum salinity {sal['min']:.2f} PSU",
                f"Maximum salinity {sal['max']:.2f} PSU"
            ])
        
        if 'depth' in stats:
            depth = stats['depth']
            summary_parts.extend([
                f"Depth range {depth['min']:.0f} to {depth['max']:.0f} meters",
                f"Maximum depth {depth['max']:.0f}m",
                f"Minimum depth {depth['min']:.0f}m",
                f"{depth['count']} depth measurements"
            ])
        
        # Extreme value tags for superlative queries
        summary_parts.extend(extreme_tags)
        
        # Coordinate tags for geographic queries
        summary_parts.extend(coordinate_tags)
        
        # Additional searchable metadata
        summary_parts.extend([
            f"Data file {filename}",
            f"Argo float profile",
            "Oceanographic data",
            "CTD measurements"
        ])
        
        # Join with periods and spaces for better tokenization
        return ". ".join(summary_parts) + "."
    
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
    
    def generate_search_index(self, output_file="argo_search_terms.json"):
        """Generate a search term index for debugging embedding performance"""
        search_terms = {
            'coordinate_patterns': [],
            'temperature_patterns': [],
            'salinity_patterns': [],
            'depth_patterns': [],
            'geographic_terms': [],
            'extreme_value_terms': []
        }
        
        for summary in self.summaries:
            if 'coordinate_tags' in summary:
                search_terms['coordinate_patterns'].extend(summary['coordinate_tags'])
            if 'extreme_tags' in summary:
                search_terms['extreme_value_terms'].extend(summary['extreme_tags'])
            if 'geographic_keywords' in summary:
                if 'ocean_keywords' in summary['geographic_keywords']:
                    search_terms['geographic_terms'].extend(summary['geographic_keywords']['ocean_keywords'])
                if 'climatic_zones' in summary['geographic_keywords']:
                    search_terms['geographic_terms'].extend(summary['geographic_keywords']['climatic_zones'])
        
        # Remove duplicates
        for key in search_terms:
            search_terms[key] = list(set(search_terms[key]))
        
        with open(output_file, 'w') as f:
            json.dump(search_terms, f, indent=2)
        
        print(f"Search term index saved to {output_file}")

# Main execution
if __name__ == "__main__":
    # Initialize summarizer
    summarizer = ArgoProfileSummarizer("parquets")
    
    # Process all CSV files
    processed = summarizer.process_all_csvs()
    
    if processed > 0:
        # Save summaries
        summarizer.save_summaries("argo_summaries.json")
        summarizer.generate_search_index("argo_search_terms.json")
        print(f"Successfully processed {processed} profiles")
        
        # Print example of optimized summary
        if summarizer.summaries:
            print("\nExample optimized summary:")
            print(summarizer.summaries[0]['summary_text'][:500] + "...")
    else:
        print("No profiles were processed successfully")