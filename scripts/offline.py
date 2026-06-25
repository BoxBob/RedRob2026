import os
import sys
import time

# Setup stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding='utf-8')

# Resolve absolute path to workspace root (one level up from scripts/)
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Inject paths into python's search path
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "src"))

try:
    import src.preprocessing.installing_pytlex as installing_pytlex
    import src.preprocessing.loading_raw_data as loading_raw_data
    import src.preprocessing.embedder as embedder
except ImportError as e:
    print(f"Error importing pre-processing modules: {e}")
    sys.exit(1)

def run_offline_pipeline():
    print("==========================================================")
    print("🚀 STARTING OFFLINE PRE-PROCESSING PIPELINE")
    print("==========================================================")
    
    start_total = time.time()

    # Step 1: Install and setup pyTLEX
    print("\n--- [Step 1/3] Setting up pyTLEX ---")
    installing_pytlex.setup_pytlex()

    # Step 2: Extract candidates and copy Job Description to data/raw
    print("\n--- [Step 2/3] Extracting raw datasets ---")
    loading_raw_data.extract_candidates()
    loading_raw_data.copy_job_description()

    # Step 3: Generate vectors, binarize, and write memmap file
    print("\n--- [Step 3/3] Generating candidate embeddings & memory mapping ---")
    raw_json = os.path.join(workspace_root, "data", "raw", "candidates.jsonl")
    processed_memmap = os.path.join(workspace_root, "data", "processed", "candidate_embeddings.dat")
    index_json = os.path.join(workspace_root, "data", "processed", "candidate_index.json")
    
    # Process candidates
    embedder.process_candidates_to_memmap(raw_json, processed_memmap, index_json)

    end_total = time.time()
    elapsed = round(end_total - start_total, 2)
    
    print("\n==========================================================")
    print(f"🎉 OFFLINE PIPELINE COMPLETE! (Total time: {elapsed}s)")
    print(f"Processed memmap: {processed_memmap}")
    print(f"Candidate index : {index_json}")
    print("==========================================================")

if __name__ == "__main__":
    run_offline_pipeline()
