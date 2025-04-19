from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import json
import lancedb
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from typing import List, Dict, Any, Optional
import pyarrow as pa
from functools import partial
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from tenacity import retry, stop_after_attempt, wait_exponential

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connection pool to reuse connections
_db_pool = None
MAX_RETRIES = 3
INITIAL_WAIT = 1  # seconds

def get_connection_params():
    """Get LanceDB connection parameters."""
    return {
        "uri": "db://hh-auditor-d5e90z",
        "api_key": "sk_DENVTWWWJO5EB5LVK5UABDICKBLC7LMX6LNCBYP4WVDVCF7EBMCJQ====",
        "region": "us-east-1",
        "timeout": 30,  # 30 seconds timeout
        "read_consistency_interval": 1000,  # 1 second consistency interval
        "max_connections": 10  # Maximum number of connections in the pool
    }

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=INITIAL_WAIT))
def get_db():
    """Initialize LanceDB connection with retry logic."""
    global _db_pool
    
    try:
        if _db_pool is not None:
            return _db_pool
            
        print("Attempting to connect to LanceDB...")
        params = get_connection_params()
        # Use synchronous connection instead of async
        _db_pool = lancedb.connect(**params)
        print("Successfully connected to hosted LanceDB")
        return _db_pool
    except Exception as e:
        print(f"Detailed error connecting to LanceDB: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")
        # Clear pool on error
        _db_pool = None
        raise

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    return model.encode(texts).tolist()

def get_or_create_table(name: str = "embeddings"):
    """Get existing table or create a new one with schema."""
    try:
        db = get_db()
        if not db:
            raise HTTPException(status_code=500, detail="LanceDB connection not available")
            
        # Define schema for the table - 384 is the dimension for all-MiniLM-L6-v2
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 384)),
            pa.field("text", pa.utf8()),
            pa.field("id", pa.int64()),
            pa.field("category", pa.utf8()),
            pa.field("timestamp", pa.timestamp('us'))  # Add timestamp for versioning
        ])
        
        tables = db.table_names()
        if name in tables:
            return db.open_table(name)
        else:
            return db.create_table(name, schema=schema)
    except Exception as e:
        print(f"Error in get_or_create_table: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error accessing LanceDB table: {str(e)}")

def process_and_store_data(records: List[Dict]):
    """Process and store data in LanceDB."""
    try:
        table = get_or_create_table()
        # Add timestamp to each record
        for record in records:
            record["timestamp"] = pd.Timestamp.now()
        # Convert records to RecordBatch for better performance
        batch = pa.RecordBatch.from_pylist(records)
        table.add(batch)
    except Exception as e:
        print(f"Error in data processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-json")
async def process_json(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        # Read the JSON file
        content = await file.read()
        data = json.loads(content)
        
        # Convert JSON to DataFrame
        df = pd.DataFrame(data)
        
        if 'text' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="JSON must contain a 'text' field for embedding generation"
            )
        
        # Create embeddings and prepare records
        embeddings = await asyncio.get_event_loop().run_in_executor(executor, create_embeddings, df['text'].tolist())
        records = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            record = {
                "vector": embeddings[i],
                "text": row['text'],
                "id": int(i),  # Use index as ID if not provided
                "category": row.get('category', 'default'),  # Use default if category not provided
                "timestamp": pd.Timestamp.now()  # Add timestamp
            }
            records.append(record)
        
        # Process and store data
        await asyncio.get_event_loop().run_in_executor(executor, process_and_store_data, records)
        
        return {
            "message": "JSON processing completed",
            "num_records": len(records),
            "embeddings_generated": True
        }
            
    except Exception as e:
        print(f"Error in process_json: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_similar(text: str, limit: int = 5):
    """Query for similar texts based on embedding."""
    try:
        # Generate embedding for query text
        query_vector = await asyncio.get_event_loop().run_in_executor(executor, create_embeddings, [text])
        query_vector = query_vector[0]
        
        # Get table and perform search
        def execute_search():
            table = get_or_create_table()
            return table.search(query_vector).limit(limit).to_pandas()
            
        results = await asyncio.get_event_loop().run_in_executor(executor, execute_search)
        
        return {
            "results": results.to_dict(orient='records'),
            "query": text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Check the status of LanceDB connection and available tables."""
    try:
        def check_connection():
            db = get_db()
            if not db:
                raise HTTPException(status_code=500, detail="LanceDB connection not available")
            
            # Test the connection by actually querying table names
            try:
                tables = db.table_names()
                return {
                    "status": "connected",
                    "tables": tables,
                    "connection_info": {
                        "region": get_connection_params()["region"],
                        "timeout": get_connection_params()["timeout"],
                        "read_consistency": get_connection_params()["read_consistency_interval"],
                        "max_connections": get_connection_params()["max_connections"]
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")
        
        return await asyncio.get_event_loop().run_in_executor(executor, check_connection)
            
    except Exception as e:
        # Clear connection pool on error
        global _db_pool
        _db_pool = None
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/test-lancedb")
async def test_lancedb():
    """Simple endpoint to test LanceDB connection."""
    try:
        def connect_and_check():
            db = get_db()
            tables = db.table_names()
            return {
                "status": "success",
                "connection": "active",
                "tables": tables
            }
        
        result = await asyncio.get_event_loop().run_in_executor(executor, connect_and_check)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error testing LanceDB: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
