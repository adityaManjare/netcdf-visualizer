import sys
import torch
import json
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any, Tuple

# --- Master Configuration ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "argo_profiles"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Model Instance ---
model = None

def initialize_model():
    """Initialize the embedding model once globally."""
    global model
    if model is None:
        try:
            print(f"Loading embedding model '{MODEL_NAME}' on device '{DEVICE.upper()}'...", file=sys.stderr)
            model = SentenceTransformer(MODEL_NAME, device=DEVICE)
            print("Model loaded successfully.", file=sys.stderr)
            
            # Warm up the model with a test encoding
            _ = model.encode("test warmup", convert_to_numpy=True)
            print("Model warmed up and ready.", file=sys.stderr)
            
        except Exception as e:
            error_msg = f"Failed to load embedding model: {e}"
            print(f"FATAL: {error_msg}", file=sys.stderr)
            return {"success": False, "error": error_msg, "results": []}
    
    return {"success": True}

def extract_query_components(query_text: str) -> Dict[str, Any]:
    """
    Extract oceanographic components from query text to understand search intent.
    This helps in query expansion and result interpretation.
    """
    components = {
        "coordinates": [],
        "ocean_regions": [],
        "depth_mentions": [],
        "temperature_mentions": [],
        "salinity_mentions": [],
        "climatic_zones": [],
        "temporal_mentions": [],
        "extremes_mentioned": False
    }
    
    query_lower = query_text.lower()
    
    # Extract coordinates
    coord_patterns = [
        r'(\d+\.?\d*)\s*[°degrees]?\s*([ns])',  # Latitude
        r'(\d+\.?\d*)\s*[°degrees]?\s*([ew])'   # Longitude
    ]
    
    for pattern in coord_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            components["coordinates"].append(f"{match[0]}°{match[1].upper()}")
    
    # Extract ocean regions
    ocean_keywords = {
        'southern ocean': 'Southern Ocean',
        'pacific': 'Pacific Ocean', 
        'atlantic': 'Atlantic Ocean',
        'arctic': 'Arctic Ocean',
        'indian ocean': 'Indian Ocean'
    }
    
    for keyword, formal_name in ocean_keywords.items():
        if keyword in query_lower:
            components["ocean_regions"].append(formal_name)
    
    # Extract climatic zones
    climatic_keywords = {
        'polar': 'polar waters',
        'tropical': 'tropical waters',
        'subtropical': 'subtropical waters',
        'equatorial': 'equatorial waters',
        'antarctic': 'Antarctic waters',
        'arctic': 'Arctic waters'
    }
    
    for keyword, zone in climatic_keywords.items():
        if keyword in query_lower:
            components["climatic_zones"].append(zone)
    
    # Check for extremes mentions
    extreme_keywords = ['minimum', 'maximum', 'extreme', 'gradient', 'range', 'coldest', 'warmest', 'saltiest']
    components["extremes_mentioned"] = any(keyword in query_lower for keyword in extreme_keywords)
    
    # Extract depth mentions
    depth_patterns = [
        r'(\d+)\s*m(?:eter)?s?\s*depth',
        r'depth.*?(\d+)\s*m',
        r'(\d+)\s*to\s*(\d+)\s*m'
    ]
    
    for pattern in depth_patterns:
        matches = re.findall(pattern, query_lower)
        components["depth_mentions"].extend(matches)
    
    # Extract temperature mentions
    temp_patterns = [
        r'(-?\d+\.?\d*)\s*[°degrees]?\s*c',
        r'temperature.*?(-?\d+\.?\d*)',
        r'(-?\d+\.?\d*)\s*celsius'
    ]
    
    for pattern in temp_patterns:
        matches = re.findall(pattern, query_lower)
        components["temperature_mentions"].extend(matches)
    
    # Extract salinity mentions
    sal_patterns = [
        r'(\d+\.?\d*)\s*psu',
        r'salinity.*?(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*practical\s*salinity'
    ]
    
    for pattern in sal_patterns:
        matches = re.findall(pattern, query_lower)
        components["salinity_mentions"].extend(matches)
    
    return components

def expand_query_for_embedding(original_query: str, components: Dict[str, Any]) -> str:
    """
    Expand the query to better match the summary format stored in the database.
    This creates a query that's more likely to match the embedding space of stored summaries.
    """
    expanded_parts = [original_query]
    
    # Add oceanographic context terms
    if components["coordinates"]:
        expanded_parts.append(f"coordinates {' '.join(components['coordinates'])}")
    
    if components["ocean_regions"]:
        expanded_parts.append(f"recorded in {' '.join(components['ocean_regions'])}")
    
    if components["climatic_zones"]:
        expanded_parts.append(f"within {' '.join(components['climatic_zones'])}")
    
    # Add measurement context
    if components["extremes_mentioned"]:
        expanded_parts.extend([
            "temperature extremes minimum maximum gradient",
            "salinity extremes minimum maximum PSU",
            "significant temperature gradient observed",
            "salinity extremes recorded between minimum maximum"
        ])
    
    # Add depth context
    if components["depth_mentions"]:
        expanded_parts.append("depth measurements spanning minimum maximum profile contains")
    
    # Add general oceanographic terms to improve matching
    expanded_parts.extend([
        "oceanographic profile measurements",
        "water column characteristics",
        "collected data source",
        "temperature salinity depth profile"
    ])
    
    expanded_query = " ".join(expanded_parts)
    
    print(f"🔍 Query expansion: '{original_query}' expanded to include oceanographic context", file=sys.stderr)
    return expanded_query

def generate_embedding(text: str) -> List[float]:
    """Generate vector embedding for given text."""
    global model
    if model is None:
        init_result = initialize_model()
        if not init_result["success"]:
            return None
    
    try:
        print(f"Generating embedding for query of length {len(text)} characters...", file=sys.stderr)
        embedding = model.encode(text, convert_to_numpy=True).tolist()
        print("Embedding generated successfully.", file=sys.stderr)
        return embedding
    except Exception as e:
        print(f"Error during embedding generation: {e}", file=sys.stderr)
        return None

def calculate_relevance_score(profile_data: Dict, query_components: Dict) -> float:
    """
    Calculate additional relevance score based on query components.
    This helps in re-ranking results based on specific user intent.
    """
    relevance_score = 1.0
    metadata = profile_data
    document = profile_data.get('document', '').lower()
    
    # Boost for coordinate matches
    if query_components["coordinates"]:
        lat = metadata.get('latitude')
        lon = metadata.get('longitude')
        if lat is not None and lon is not None:
            # Check if coordinates are close (within reasonable range)
            for coord in query_components["coordinates"]:
                if any(str(round(abs(lat), 1)) in coord or str(round(abs(lon), 1)) in coord 
                       for coord in query_components["coordinates"]):
                    relevance_score += 0.2
    
    # Boost for ocean region matches
    for region in query_components["ocean_regions"]:
        if region.lower() in document:
            relevance_score += 0.15
    
    # Boost for climatic zone matches  
    for zone in query_components["climatic_zones"]:
        if zone.lower() in document:
            relevance_score += 0.1
    
    # Boost for extremes if mentioned
    if query_components["extremes_mentioned"]:
        extreme_terms = ['minimum', 'maximum', 'extreme', 'gradient']
        if any(term in document for term in extreme_terms):
            relevance_score += 0.1
    
    return min(relevance_score, 2.0)  # Cap the boost

def find_similar_profiles(query_vector: List[float], query_components: Dict, n_results: int = 5) -> Dict[str, Any]:
    """Search ChromaDB for similar profiles with enhanced relevance scoring."""
    print(f"\nSearching database for top {n_results} similar profiles...", file=sys.stderr)

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        total_profiles = collection.count()
        print(f"📚 Connected to '{COLLECTION_NAME}' with {total_profiles:,} oceanographic profiles.", file=sys.stderr)

    except Exception as e:
        error_msg = f"Could not connect to ChromaDB: {e}"
        print(f"❌ Critical Error: {error_msg}", file=sys.stderr)
        return {"success": False, "error": error_msg, "results": []}

    try:
        # Query more results than needed for re-ranking
        search_results = min(n_results * 3, 50)  # Get 3x results for re-ranking
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=search_results,
            include=['documents', 'metadatas', 'distances']
        )

        if results and results.get('ids') and results['ids'][0]:
            print(f"✅ Found {len(results['ids'][0])} initial results for re-ranking.", file=sys.stderr)

            result_docs = results['documents'][0]
            result_metadatas = results['metadatas'][0]  
            result_distances = results['distances'][0]
            
            # Calculate enhanced scores
            enhanced_results = []
            for i, (metadata, distance, doc) in enumerate(zip(result_metadatas, result_distances, result_docs)):
                similarity = 1 - distance
                
                # Create combined profile data for relevance scoring
                profile_data = dict(metadata)
                profile_data['document'] = doc
                
                # Calculate relevance boost
                relevance_score = calculate_relevance_score(profile_data, query_components)
                enhanced_score = similarity * relevance_score
                
                profile_result = {
                    "original_rank": i + 1,
                    "profile_id": metadata.get('profile_id', 'N/A'),
                    "similarity": round(similarity, 4),
                    "relevance_score": round(relevance_score, 2),
                    "enhanced_score": round(enhanced_score, 4),
                    "file": metadata.get('file', 'N/A'),
                    "latitude": metadata.get('latitude'),
                    "longitude": metadata.get('longitude'),
                    "document": doc,
                    "document_preview": doc[:300].replace('\n', ' ') + "..." if len(doc) > 300 else doc
                }
                enhanced_results.append(profile_result)
            
            # Re-rank by enhanced score
            enhanced_results.sort(key=lambda x: x['enhanced_score'], reverse=True)
            final_results = enhanced_results[:n_results]
            
            # Log the re-ranking results
            for i, profile in enumerate(final_results, 1):
                print(f"Rank {i}: Profile {profile['profile_id']} "
                      f"[Sim: {profile['similarity']:.3f}, Rel: {profile['relevance_score']:.2f}, "
                      f"Final: {profile['enhanced_score']:.3f}]", file=sys.stderr)
            
            # Update ranks after re-ranking
            for i, profile in enumerate(final_results, 1):
                profile['rank'] = i
                del profile['original_rank']  # Clean up
                del profile['enhanced_score']  # Clean up for final output
            
            return {
                "success": True,
                "query_components": query_components,
                "num_results": len(final_results),
                "total_profiles_searched": total_profiles,
                "results": final_results
            }
            
        else:
            print("❌ No results found for the given query.", file=sys.stderr)
            return {
                "success": True,
                "query_components": query_components,
                "num_results": 0,
                "results": [],
                "message": "No oceanographic profiles found matching the search criteria."
            }
            
    except Exception as e:
        error_msg = f"Error during search: {e}"
        print(f"❌ Error during search: {e}", file=sys.stderr)
        return {"success": False, "error": error_msg, "results": []}

def main():
    """Main execution with enhanced query processing."""
    if len(sys.argv) < 2:
        error_result = {
            "success": False,
            "error": "Usage: python find_similar.py \"<search query>\" [num_results]",
            "results": []
        }
        print(json.dumps(error_result))
        sys.exit(1)

    # Parse arguments
    input_text = sys.argv[1]
    num_results = 5
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        num_results = int(sys.argv[2])

    print(f"🌊 Starting oceanographic profile search for: '{input_text}'", file=sys.stderr)

    # Initialize model
    init_result = initialize_model()
    if not init_result["success"]:
        print(json.dumps(init_result))
        sys.exit(1)

    # Extract query components
    query_components = extract_query_components(input_text)
    print(f"🔍 Extracted components: {len(query_components['coordinates'])} coordinates, "
          f"{len(query_components['ocean_regions'])} regions, "
          f"extremes: {query_components['extremes_mentioned']}", file=sys.stderr)

    # Expand query for better embedding matching
    expanded_query = expand_query_for_embedding(input_text, query_components)

    # Generate embedding
    query_embedding = generate_embedding(expanded_query)

    if query_embedding:
        # Search with enhanced relevance scoring
        result = find_similar_profiles(query_embedding, query_components, n_results=num_results)
        result["original_query"] = input_text
        result["expanded_query"] = expanded_query
        
        # Output structured JSON
        print(json.dumps(result, indent=2))
    else:
        error_result = {
            "success": False,
            "error": "Failed to generate query embedding",
            "results": []
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()