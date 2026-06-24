import os
import urllib.request
import zipfile
import sys
import gzip
import json

# Setup stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding='utf-8')

# Resolve absolute path to workspace root (parent of the 'scripts' directory)
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def setup_pytlex():
    url = "https://cognac.cs.fiu.edu/wp-content/uploads/sites/11/2023/11/edu.fiu_.pytlex.zip"
    zip_path = os.path.join(workspace_root, "fiu.edu.pytlex.zip")
    extract_dir = os.path.join(workspace_root, "fiu_pytlex")

    if not os.path.exists(extract_dir):
        print(f"Downloading pyTLEX from {url}...")
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        try:
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Download failed with error: {e}")
            return
        print("Extracting pyTLEX...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        os.remove(zip_path)
        print("pyTLEX setup complete.")
    else:
        print("pyTLEX is already installed locally. Skipping download.")

def extract_candidates():
    # Locate the challenge folder containing candidates
    challenge_dir = None
    for name in ["India_runs_data_and_ai_challenge", "india_runs"]:
        p = os.path.join(workspace_root, "PS", name)
        if os.path.exists(p):
            challenge_dir = p
            break
            
    if not challenge_dir:
        challenge_dir = os.path.join(workspace_root, "PS", "India_runs_data_and_ai_challenge")
        
    input_path = os.path.join(challenge_dir, "candidates.jsonl")
    input_gz_path = input_path + ".gz"
    output_dir = os.path.join(workspace_root, "data", "raw")
    output_path = os.path.join(output_dir, "candidates.jsonl")
    
    os.makedirs(output_dir, exist_ok=True)
    
    candidates = []
    
    # Check if .gz file exists, otherwise load normal jsonl
    if os.path.exists(input_gz_path):
        print(f"Reading candidates from compressed file: {input_gz_path}...")
        with gzip.open(input_gz_path, "rt", encoding="utf-8") as f:
            candidates = [json.loads(line) for line in f if line.strip()]
    elif os.path.exists(input_path):
        print(f"Reading candidates from: {input_path}...")
        with open(input_path, "r", encoding="utf-8") as f:
            candidates = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"Error: Candidate pool file not found at {input_path} or {input_gz_path}")
        return
        
    print(f"Loaded {len(candidates)} candidates.")
    
    # Write to data/raw/candidates.jsonl
    print(f"Saving candidates to: {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for candidate in candidates:
            f.write(json.dumps(candidate) + "\n")
            
    print("Candidates extraction to data/raw completed successfully.")

if __name__ == "__main__":
    setup_pytlex()
    extract_candidates()

# Inject the package root into python path for downstream tasks
project_root = os.path.join(workspace_root, 'fiu_pytlex', 'edu.fiu.pytlex')
sys.path.insert(0, project_root)

#import like this-->
# from pytlex_core.data import Graph
#from pytlex_core.algorithms.TLEX import TLEX