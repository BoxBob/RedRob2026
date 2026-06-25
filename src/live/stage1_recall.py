import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import gc

class Stage1Retriever:
    def __init__(self, memmap_path, index_path, embedding_dim=384):
        print("Initializing Stage 1: Loading resources...")
        
        # 1. Load the Identity Map (Fast, takes ~2MB RAM)
        with open(index_path, 'r') as f:
            self.candidate_ids = json.load(f)
            
        num_candidates = len(self.candidate_ids)
        print(f"Loaded {num_candidates} candidate IDs from index.")
            
        # 2. Map the Math Array (Zero RAM overhead, purely paged by the OS)
        self.candidate_matrix = np.memmap(
            memmap_path, 
            dtype='int8', 
            mode='r',  # Read-only mode for safety
            shape=(num_candidates, embedding_dim)
        )
        
        # 3. Load the identical bi-encoder used in the offline phase
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def search(self, job_description_text, top_k=2000):
        print(f"\n--- Starting Stage 1 Search (Target: Top {top_k}) ---")
        start_time = time.time()
        
        # Step A: Embed the JD live (Takes ~30-50ms)
        jd_float = self.model.encode([job_description_text], convert_to_numpy=True)[0]
        
        # Step B: Binarize the JD to match the memmap format
        jd_binary = (jd_float > 0).astype(np.int8)
        
        # Step C: The AVX Hamming Distance Calculation
        # bitwise_xor outputs 1 where bits differ, 0 where they match.
        # sum(axis=1) counts the differences for each candidate.
        print("Executing bitwise XOR over candidates...")
        distances = np.bitwise_xor(self.candidate_matrix, jd_binary).sum(axis=1)
        
        # Step D: Get the row indices of the lowest distances (fewest differences)
        # argpartition is faster than argsort because it doesn't sort the whole array.
        # It handles small arrays safely by capping k to len(self.candidate_ids).
        actual_top_k = min(top_k, len(self.candidate_ids))
        if actual_top_k == len(self.candidate_ids):
            top_k_sorted_indices = np.argsort(distances)
            top_k_indices = top_k_sorted_indices
        else:
            top_k_indices = np.argpartition(distances, actual_top_k)[:actual_top_k]
            top_k_sorted_indices = top_k_indices[np.argsort(distances[top_k_indices])]
        
        # Step E: Translate row numbers back to Candidate IDs
        top_candidate_ids = [self.candidate_ids[i] for i in top_k_sorted_indices]
        
        print(f"Stage 1 Complete! Time elapsed: {round(time.time() - start_time, 4)} seconds.")
        
        # Step F: Purge memory before handing off to Stage 2
        del distances
        del top_k_indices
        gc.collect()
        
        return top_candidate_ids

# --- Test the Script ---
if __name__ == "__main__":
    # Resolve absolute path to workspace root
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define absolute paths
    MEMMAP_PATH = os.path.join(workspace_root, "data", "processed", "candidate_embeddings.dat")
    INDEX_PATH = os.path.join(workspace_root, "data", "processed", "candidate_index.json")
    
    # Instantiate the retriever
    retriever = Stage1Retriever(
        memmap_path=MEMMAP_PATH, 
        index_path=INDEX_PATH
    )
    
    # A dummy Job Description for testing
    jd_text = """
    We are looking for a Senior Software Engineer with deep expertise in Python, 
    Next.js, and AWS. The ideal candidate has experience building scalable microservices 
    and optimizing databases. 5+ years of experience required.
    """
    
    # Run the search!
    top_2000 = retriever.search(jd_text, top_k=2000)
    
    print("\nTop 5 Matches (IDs):", top_2000[:5])