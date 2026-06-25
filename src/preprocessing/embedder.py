import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import time

def build_candidate_text(cand):
    profile = cand.get('profile', {})
    headline = profile.get('headline', '')
    summary = profile.get('summary', '')
    
    skills = [s.get('name', '') for s in cand.get('skills', []) if s.get('name')]
    skills_str = ", ".join(skills)
    
    history = []
    for job in cand.get('career_history', []):
        title = job.get('title', '')
        company = job.get('company', '')
        desc = job.get('description', '')
        if title:
            history.append(f"{title} at {company}: {desc}")
            
    history_str = " | ".join(history)
    
    text_parts = [
        f"Headline: {headline}",
        f"Summary: {summary}",
        f"Skills: {skills_str}",
        f"Experience: {history_str}"
    ]
    
    return " ".join([p for p in text_parts if p.strip()])

def process_candidates_to_memmap(json_filepath, memmap_filepath, index_filepath, batch_size=1000):
    print("Loading embedding model (nomic-embed-text-v1.5 or all-MiniLM-L6-v2)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_dim = 384 

    print(f"Loading candidate data from {json_filepath}...")
    if json_filepath.endswith('.jsonl'):
        with open(json_filepath, 'r', encoding='utf-8') as f:
            candidates = [json.loads(line) for line in f if line.strip()]
    else:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
        
    num_candidates = len(candidates)
    print(f"Total candidates to process: {num_candidates}")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(memmap_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(index_filepath), exist_ok=True)

    # We use int8 because we are going to binarize the embeddings (0 or 1)
    print(f"Creating memory-mapped file at {memmap_filepath}...")
    fp = np.memmap(memmap_filepath, dtype='int8', mode='w+', shape=(num_candidates, embedding_dim))

    candidate_ids = []
    
    start_time = time.time()
    for i in range(0, num_candidates, batch_size):
        batch = candidates[i:i + batch_size]

        texts = [build_candidate_text(cand) for cand in batch]
        ids = [cand.get('candidate_id') for cand in batch]
        candidate_ids.extend(ids)
        
        embeddings_float = model.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=False)
        
        # Binarize the embeddings: If value > 0, it becomes 1. Else 0.
        embeddings_binary = (embeddings_float > 0).astype(np.int8)
        
        # Stream directly to the disk array
        fp[i:i + len(batch)] = embeddings_binary
        
        # Force writing to disk to prevent RAM buildup
        fp.flush()
        
        print(f"Processed batch {i} to {i + len(batch)}... Time elapsed: {round(time.time() - start_time, 2)}s")

    # 4. Save the ID index so we know which row in the memmap belongs to which candidate
    with open(index_filepath, 'w') as f:
        json.dump(candidate_ids, f)

    print("✅ Offline Embedding Complete! Data flushed to disk.")

if __name__ == "__main__":
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_JSON = os.path.join(workspace_root, "data", "raw", "candidates.jsonl")
    PROCESSED_MEMMAP = os.path.join(workspace_root, "data", "processed", "candidate_embeddings.dat")
    INDEX_JSON = os.path.join(workspace_root, "data", "processed", "candidate_index.json")
    
    process_candidates_to_memmap(RAW_JSON, PROCESSED_MEMMAP, INDEX_JSON)
