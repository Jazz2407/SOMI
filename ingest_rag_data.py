import sys
import json
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import components
from agents.rag_agent import MedicalRAG
from config import Config
import argparse

# Initialize parser
parser = argparse.ArgumentParser(description="Medical Data Ingestion Script")
parser.add_argument("--file", type=str, required=False, help="Path to a single file")
parser.add_argument("--dir", type=str, required=False, help="Path to a directory of files")

args = parser.parse_args()

# Load configuration and RAG system
config = Config()
rag = MedicalRAG(config)

def data_ingestion():
    # Initialize result with a default failure state
    result = {"success": False, "message": "No input provided and default directory is empty."}
    
    try:
        if args.file:
            print(f"Processing single file: {args.file}")
            result = rag.ingest_file(args.file)
        elif args.dir:
            print(f"Processing directory: {args.dir}")
            result = rag.ingest_directory(args.dir)
        else:
            # DEFAULT BEHAVIOR: Use the path defined in your Config
            default_path = config.rag.doc_local_path
            print(f"No arguments provided. Checking default directory: {default_path}")
            if os.path.exists(default_path):
                result = rag.ingest_directory(default_path)
            else:
                result = {"success": False, "message": f"Default path {default_path} not found."}

    except Exception as e:
        result = {"success": False, "message": str(e)}
        print(f"Critical error during ingestion: {e}")

    # Safe printing using get() to avoid KeyErrors
    print("\nIngestion result:", json.dumps(result, indent=2))
    return result.get("success", False)

if __name__ == "__main__":
    import os
    print("\n--- Starting Medical Data Ingestion ---")
    
    success = data_ingestion()
    
    if success:
        print("\n✅ Successfully ingested the documents into Qdrant!")
    else:
        print("\n❌ Ingestion failed. Please check the error message above.")