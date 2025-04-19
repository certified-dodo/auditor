import json
import lancedb
import pandas as pd
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import time

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_connection_params():
    """Get LanceDB connection parameters."""
    return {
        "uri": "db://hh-auditor-d5e90z",
        "api_key": "sk_DENVTWWWJO5EB5LVK5UABDICKBLC7LMX6LNCBYP4WVDVCF7EBMCJQ====",
        "region": "us-east-1"
    }

def connect_to_lancedb():
    """Connect to LanceDB."""
    print("Connecting to LanceDB...")
    params = get_connection_params()
    db = lancedb.connect(**params)
    print("Successfully connected to LanceDB!")
    return db

def create_embeddings(texts):
    """Create embeddings for text."""
    print(f"Creating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts).tolist()
    print("Embeddings created successfully!")
    return embeddings

def get_or_create_table(db, name="embeddings"):
    """Get or create a table."""
    print(f"Getting or creating table '{name}'...")
    
    # Define schema for the table - 384 is the dimension for all-MiniLM-L6-v2
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 384)),
        pa.field("text", pa.utf8()),
        pa.field("id", pa.int64()),
        pa.field("category", pa.utf8()),
        pa.field("timestamp", pa.timestamp('us'))
    ])
    
    tables = db.table_names()
    print(f"Existing tables: {tables}")
    
    if name in tables:
        print(f"Table '{name}' already exists. Opening...")
        return db.open_table(name)
    else:
        print(f"Table '{name}' does not exist. Creating...")
        return db.create_table(name, schema=schema)

def process_json_file(file_path):
    """Process a JSON file and store embeddings in LanceDB."""
    print(f"Processing JSON file: {file_path}")
    
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    if 'text' not in df.columns:
        print("Error: JSON must contain a 'text' field for embedding generation")
        return False
    
    # Create embeddings
    embeddings = create_embeddings(df['text'].tolist())
    
    # Connect to LanceDB
    db = connect_to_lancedb()
    
    # Get or create table
    table = get_or_create_table(db)
    
    # Prepare records
    records = []
    for i, row in df.iterrows():
        record = {
            "vector": embeddings[i],
            "text": row['text'],
            "id": int(i),
            "category": row.get('category', 'default'),
            "timestamp": pd.Timestamp.now()
        }
        records.append(record)
    
    print(f"Storing {len(records)} records in LanceDB...")
    
    # Store records
    batch = pa.RecordBatch.from_pylist(records)
    table.add(batch)
    
    print("Data successfully stored in LanceDB!")
    return True

def main():
    """Main function."""
    try:
        # Process the sample JSON file
        result = process_json_file('sample.json')
        
        if result:
            print("JSON processing completed successfully.")
            
            # Test a query
            db = connect_to_lancedb()
            table = get_or_create_table(db)
            
            print("Testing query...")
            query_text = "artificial intelligence"
            query_embedding = create_embeddings([query_text])[0]
            
            results = table.search(query_embedding).limit(3).to_pandas()
            print(f"Query results for '{query_text}':")
            print(results)
            
        else:
            print("JSON processing failed.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 