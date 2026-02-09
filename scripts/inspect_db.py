import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.knowledge_base import vector_store

def inspect_db():
    print("--- CHROMA DB INSPECTION ---")
    try:
        # Get collection info
        collection = vector_store._collection
        count = collection.count()
        print(f"Total documents in collection: {count}")
        
        if count > 0:
            # Get some samples
            samples = collection.get(limit=5)
            print("\nSample Documents:")
            for i in range(len(samples['ids'])):
                print(f"- ID: {samples['ids'][i]}")
                print(f"  Content snippet: {samples['documents'][i][:100]}...")
                print(f"  Metadata: {samples['metadatas'][i]}")
        else:
            print("The database is currently empty.")
            
    except Exception as e:
        print(f"Error inspecting database: {e}")

if __name__ == "__main__":
    inspect_db()
