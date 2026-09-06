import os
import sys
import gzip
import json

# Setup stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding='utf-8')

# Resolve absolute path to workspace root
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def copy_job_description():
    # Locate the challenge folder containing the job description
    challenge_dir = None
    for name in ["India_runs_data_and_ai_challenge", "india_runs"]:
        p = os.path.join(workspace_root, "PS", name)
        if os.path.exists(p):
            challenge_dir = p
            break
            
    if not challenge_dir:
        challenge_dir = os.path.join(workspace_root, "PS", "India_runs_data_and_ai_challenge")
        
    input_path = os.path.join(challenge_dir, "job_description.docx")
    output_dir = os.path.join(workspace_root, "data", "raw")
    output_path = os.path.join(output_dir, "job_description.docx")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(input_path):
        import shutil
        print(f"Copying job description from: {input_path} to {output_path}...")
        shutil.copy(input_path, output_path)
        print("Job description copied successfully to data/raw.")
    else:
        print(f"Error: Job description file not found at {input_path}")

if __name__ == "__main__":
    extract_candidates()
    copy_job_description()
