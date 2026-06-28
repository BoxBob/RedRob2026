import os
import json
import numpy as np
import pickle
import scipy.sparse
import sys

# Setup stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding='utf-8')

# Resolve absolute path to workspace root
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "src"))

from src.config import (
    RAW_JD,
    OUT_FRAUD_FLAGS_JSON,
    OUT_TFIDF_VECTORIZER_PKL,
    OUT_TFIDF_MATRIX_NPZ,
    OUT_ANOMALY_MODEL_PKL
)

class RedrobHoneypotDetector:
    def __init__(self):
        self.flags = {}
        self.vectorizer = None
        self.tfidf_matrix = None
        self.anomaly_model = None
        
        self.load_artifacts()

    def load_artifacts(self):
        print("Loading honeypot detection artifacts...")
        if os.path.exists(OUT_FRAUD_FLAGS_JSON):
            with open(OUT_FRAUD_FLAGS_JSON, "r", encoding="utf-8") as f:
                flags_list = json.load(f)
                for item in flags_list:
                    self.flags[item["candidate_id"]] = item
        
        if os.path.exists(OUT_TFIDF_VECTORIZER_PKL):
            with open(OUT_TFIDF_VECTORIZER_PKL, "rb") as f:
                self.vectorizer = pickle.load(f)
                
        if os.path.exists(OUT_TFIDF_MATRIX_NPZ):
            self.tfidf_matrix = scipy.sparse.load_npz(OUT_TFIDF_MATRIX_NPZ)
            
        if os.path.exists(OUT_ANOMALY_MODEL_PKL):
            with open(OUT_ANOMALY_MODEL_PKL, "rb") as f:
                self.anomaly_model = pickle.load(f)

    def extract_jd_text(self):
        if not os.path.exists(RAW_JD):
            # Fallback for text version if docx not handled or we parse txt
            jd_txt_path = os.path.join(workspace_root, "jd_text.txt")
            if os.path.exists(jd_txt_path):
                with open(jd_txt_path, "r", encoding="utf-8") as f:
                    return f.read().lower()
            return ""
        
        # In a real environment, we'd use docx to parse RAW_JD.
        # Assuming jd_text.txt is a plain text extraction available in the root for ease.
        jd_txt_path = os.path.join(workspace_root, "jd_text.txt")
        if os.path.exists(jd_txt_path):
            with open(jd_txt_path, "r", encoding="utf-8") as f:
                return f.read().lower()
        return ""

    def evaluate_candidates(self, candidate_ids):
        """
        Returns a dictionary mapping candidate_id to their final fraud features.
        """
        jd_text = self.extract_jd_text()
        if not jd_text or self.vectorizer is None or self.tfidf_matrix is None:
            return {cid: {} for cid in candidate_ids}
            
        jd_vector = self.vectorizer.transform([jd_text])
        
        # To get similarities quickly, we can do dot product since tfidf vectors are l2 normalized
        similarities = self.tfidf_matrix.dot(jd_vector.T).toarray().flatten()
        
        results = {}
        # We need to map candidate ID to matrix index. 
        # Wait, how do we know the index in tfidf_matrix?
        # The list in fraud_flags.json has the same order as candidates.jsonl and tfidf_matrix.
        # Let's map candidate_id to index from our flags list.
        
        # Create an index map if not already done
        cand_order = [item["candidate_id"] for item in self.flags.values()]
        cand_to_idx = {cid: idx for idx, cid in enumerate(self.flags.keys())}
        
        for cid in candidate_ids:
            cand_features = {}
            if cid in self.flags:
                base_flags = self.flags[cid]
                cand_features.update(base_flags)
                
                idx = cand_to_idx.get(cid)
                if idx is not None:
                    jd_sim = similarities[idx]
                    richness = base_flags.get("vocabulary_richness", 1.0)
                    mimicry_score = jd_sim * (1.0 - richness)
                    cand_features["jd_similarity"] = jd_sim
                    cand_features["mimicry_score"] = mimicry_score
            
            results[cid] = cand_features
            
        return results

if __name__ == "__main__":
    # Quick test
    detector = RedrobHoneypotDetector()
    if detector.flags:
        sample_ids = list(detector.flags.keys())[:5]
        res = detector.evaluate_candidates(sample_ids)
        print(json.dumps(res, indent=2))
