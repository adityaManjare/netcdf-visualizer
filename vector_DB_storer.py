import json
import chromadb
from pathlib import Path
from typing import List, Dict, Any
import time
import os
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

class ArgoProfileEmbedder:
    def __init__(self, 
                 chroma_db_path: str = "./chroma_db",
                 collection_name: str = "argo_profiles"):
        """
        Initialize the Argo Profile Embedder with LOCAL GPU-based embeddings
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name

        # Load local embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

        print(f"🚀 Initializing Argo Profile Embedder (LOCAL GPU-based)")
        print(f"   ChromaDB: {chroma_db_path}")
        print(f"   Device: {device.upper()}")

        self.client = None
        self.collection = None
        self._init_chroma_db()
    
    def _init_chroma_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create directory
            self.chroma_db_path.mkdir(parents=True, exist_ok=True)
            
            # Create client
            self.client = chromadb.PersistentClient(path=str(self.chroma_db_path))
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                existing_count = self.collection.count()
                print(f"📚 Found existing collection '{self.collection_name}' with {existing_count} profiles")
            except:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Argo oceanographic profiles"}
                )
                print(f"🆕 Created new collection '{self.collection_name}'")
            
        except Exception as e:
            print(f"❌ Failed to initialize ChromaDB: {e}")
            raise
    
    def call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings locally on GPU"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
            return embeddings
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            return None
    
    def load_summaries(self, json_file: str = "argo_summaries.json") -> List[Dict[Any, Any]]:
        """Load profile summaries from JSON file"""
        json_path = Path(json_file)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Summary file not found: {json_file}")
        
        with open(json_path, 'r') as f:
            summaries = json.load(f)
        
        print(f"📄 Loaded {len(summaries)} profile summaries")
        return summaries
    
    def create_chunk_text(self, summary: Dict[Any, Any]) -> str:
        """Create text chunk for embedding"""
        chunk_text = summary.get('summary_text', '')
        
        # Add extra info
        additional_info = []
        
        if summary.get('geographic_regions'):
            regions = ', '.join(summary['geographic_regions'])
            additional_info.append(f"Regions: {regions}")
        
        if summary.get('water_characteristics'):
            characteristics = ', '.join(summary['water_characteristics'])
            additional_info.append(f"Water: {characteristics}")
        
        if summary.get('latitude') and summary.get('longitude'):
            lat, lon = summary['latitude'], summary['longitude']
            additional_info.append(f"Location: {lat:.3f}°, {lon:.3f}°")
        
        if additional_info:
            chunk_text += " " + " ".join(additional_info)
        
        return chunk_text
    
    def create_metadata(self, summary: Dict[Any, Any]) -> Dict[str, Any]:
        """Create metadata for ChromaDB"""
        metadata = {
            'profile_id': str(summary.get('profile_id', 'unknown')),
            'file': summary.get('file', ''),
            'data_points': summary.get('data_points', 0)
        }
        
        if summary.get('latitude') and summary.get('longitude') and summary.get('julian_day'):
            lat = round(float(summary['latitude']), 3)
            lon = round(float(summary['longitude']), 3) 
            time_val = round(float(summary['julian_day']), 2)
            metadata['profile_signature'] = f"LAT{lat}_LON{lon}_TIME{time_val}"
        
        if summary.get('latitude') is not None:
            metadata['latitude'] = float(summary['latitude'])
        if summary.get('longitude') is not None:
            metadata['longitude'] = float(summary['longitude'])
        if summary.get('measurement_date'):
            metadata['measurement_date'] = summary['measurement_date']
        
        if summary.get('geographic_regions'):
            metadata['geographic_regions'] = ', '.join(summary['geographic_regions'])
        if summary.get('water_characteristics'):
            metadata['water_characteristics'] = ', '.join(summary['water_characteristics'])
        
        stats = summary.get('statistics', {})
        if 'depth' in stats:
            depth_stats = stats['depth']
            metadata.update({
                'min_depth': float(depth_stats['min']),
                'max_depth': float(depth_stats['max'])
            })
        
        if 'temperature' in stats:
            temp_stats = stats['temperature']
            metadata.update({
                'avg_temperature': float(temp_stats['mean'])
            })
        
        if 'salinity' in stats:
            sal_stats = stats['salinity']
            metadata.update({
                'avg_salinity': float(sal_stats['mean'])
            })
        
        return metadata
    
    def embed_profiles(self, summaries: List[Dict[Any, Any]], overwrite_existing: bool = False):
        """Embed all profiles and store in ChromaDB"""
        
        print(f"\n🔄 Starting embedding process for {len(summaries)} profiles")
        
        existing_ids = set()
        if not overwrite_existing:
            try:
                existing_data = self.collection.get()
                if existing_data['metadatas']:
                    existing_ids = {meta.get('profile_id') for meta in existing_data['metadatas'] if meta.get('profile_id')}
                    print(f"📋 Found {len(existing_ids)} existing profiles (will skip)")
            except:
                pass
        
        profiles_to_embed = []
        for summary in summaries:
            profile_id = str(summary.get('profile_id', 'unknown'))
            
            if not overwrite_existing and profile_id in existing_ids:
                continue
            
            chunk_text = self.create_chunk_text(summary)
            metadata = self.create_metadata(summary)
            
            profiles_to_embed.append({
                'id': f"profile_{profile_id}",
                'text': chunk_text,
                'metadata': metadata
            })
        
        if not profiles_to_embed:
            print("✅ No new profiles to embed!")
            return
        
        print(f"📊 Processing {len(profiles_to_embed)} profiles...")
        
        successful_embeds = 0
        
        for i, profile in enumerate(profiles_to_embed, 1):
            print(f"🔄 Processing {i}/{len(profiles_to_embed)}: Profile {profile['metadata']['profile_id']}")
            
            try:
                embeddings = self.call_embedding_api([profile['text']])
                
                if embeddings and len(embeddings) > 0:
                    self.collection.add(
                        ids=[profile['id']],
                        embeddings=[embeddings[0]],
                        documents=[profile['text']],
                        metadatas=[profile['metadata']]
                    )
                    
                    successful_embeds += 1
                    print(f"   ✅ Embedded successfully")
                else:
                    print(f"   ❌ Failed to get embedding")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        print("=" * 60)
        print(f"✅ Embedding complete!")
        print(f"   Successfully embedded: {successful_embeds}/{len(profiles_to_embed)}")
        print(f"   Total collection size: {self.collection.count()}")
    
    def query_profiles(self, query_text: str, n_results: int = 5):
        """Query for similar profiles"""
        print(f"🔍 Searching for: '{query_text}'")
        
        query_embeddings = self.call_embedding_api([query_text])
        
        if not query_embeddings:
            print("❌ Failed to get query embedding")
            return {}
        
        results = self.collection.query(
            query_embeddings=[query_embeddings[0]],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        if results.get('ids') and results['ids'][0]:
            print(f"\n📋 Found {len(results['ids'][0])} results:")
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            ), 1):
                similarity = 1 - distance
                print(f"\n{i}. Profile {metadata.get('profile_id')} [Similarity: {similarity:.3f}]")
                print(f"   📄 File: {metadata.get('file')}")
                
                if metadata.get('latitude') and metadata.get('longitude'):
                    print(f"   🌍 Location: {metadata['latitude']:.3f}°, {metadata['longitude']:.3f}°")
                
                doc_preview = doc[:150] + "..." if len(doc) > 150 else doc
                print(f"   📝 {doc_preview}")
        else:
            print("❌ No results found!")
        
        return results

# Main execution
if __name__ == "__main__":
    print("🌊 Argo Profile Embedder (LOCAL GPU-based)")
    print("=" * 50)
    
    embedder = ArgoProfileEmbedder(
        chroma_db_path="./chroma_db",
        collection_name="argo_profiles"
    )
    
    try:
        summaries = embedder.load_summaries("argo_summaries.json")
        embedder.embed_profiles(summaries, overwrite_existing=False)
        
        # embedder.query_profiles("cold Arctic water", n_results=3)
        
        print(f"\n✅ Process completed!")
        
    except FileNotFoundError:
        print("❌ argo_summaries.json not found! Run argo_summarizer.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")
