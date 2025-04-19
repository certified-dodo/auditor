import json
import lancedb
import pandas as pd
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import time

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Use the provided LanceDB connection parameters
def connect_to_lancedb():
    """Connect to LanceDB."""
    print("Connecting to LanceDB...")
    db = lancedb.connect(
        uri="db://hh-auditor-d5e90z",
        api_key="sk_NKURWX2FYFHN3FSYKM3YR56G5P6CUFWGFSHX2SLAZOGMXDSDTW5Q====",
        region="us-east-1"
    )
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
    
    tables = db.table_names()
    tables_list = list(tables)  # Convert generator to list for better display
    print(f"Existing tables: {tables_list}")
    
    if name in tables_list:
        print(f"Table '{name}' already exists. Opening...")
        return db.open_table(name)
    else:
        print(f"Table '{name}' does not exist. Creating...")
        # We'll create the table with data instead of schema
        return None

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
    table_name = "embeddings"
    table = get_or_create_table(db, table_name)
    
    # Create a DataFrame with all the necessary data
    records_df = pd.DataFrame({
        "vector": embeddings,
        "text": df['text'].tolist(),
        "id": df.get('id', list(range(len(df)))),
        "category": df.get('category', ['default'] * len(df)),
        "timestamp": [pd.Timestamp.now()] * len(df)
    })
    
    print(f"Storing {len(records_df)} records in LanceDB...")
    
    # Store records using the DataFrame
    if table is None:
        # Create table with data
        table = db.create_table(table_name, data=records_df)
        print(f"Created new table '{table_name}' with data")
    else:
        # Add to existing table
        table.add(records_df)
        print(f"Added data to existing table '{table_name}'")
    
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
            table_name = "embeddings"
            table = db.open_table(table_name)
            
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