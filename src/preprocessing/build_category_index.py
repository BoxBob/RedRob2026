import os
import sys
import json
import pickle
import numpy as np
from collections import defaultdict
import time

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)

from src.config import RAW_CANDIDATES, OUT_CATEGORY_INDEX_PKL

CATEGORY_RULES = {
    "HR/People":          ["hr", "human resources", "people", "talent", "recruiter", "hrbp"],
    "Sales/BD":           ["sales", "business development", "account executive", "bdm"],
    "Marketing":          ["marketing", "growth", "brand", "content", "seo", "campaign"],
    "Mechanical/Mfg":     ["mechanical", "manufacturing", "production", "quality", "tooling", "civil"],
    "Finance/Accounting": ["finance", "accounting", "cfo", "controller", "auditor"],
    "Legal/Compliance":   ["legal", "compliance", "counsel", "attorney", "paralegal"],
    "Engineering/Tech":   ["engineer", "developer", "architect", "data", "ml", "ai", 
                           "software", "backend", "frontend", "devops", "sre"],
    "Product/Design":     ["product", "ux", "ui", "designer", "researcher"],
    "Operations":         ["operations", "ops", "supply chain", "logistics", "procurement"],
}

def assign_category(title: str) -> str:
    if not title:
        return "Other"
    title_lower = title.lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(kw in title_lower for kw in keywords):
            return category
    return "Other"

def main():
    print(f"Loading candidates from {RAW_CANDIDATES}...")
    t0 = time.time()
    with open(RAW_CANDIDATES, "r", encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(candidates)} candidates in {time.time() - t0:.2f}s")

    category_index = defaultdict(list)
    
    t0 = time.time()
    for idx, candidate in enumerate(candidates):
        history = candidate.get("career_history", [])
        title = history[0].get("title", "") if history else ""
        cat = assign_category(title)
        category_index[cat].append(idx)
    
    print(f"Categorized {len(candidates)} candidates into {len(category_index)} categories in {time.time() - t0:.2f}s")
    for cat, indices in category_index.items():
        print(f"  - {cat}: {len(indices)} profiles")
        
    print(f"Saving category index to {OUT_CATEGORY_INDEX_PKL}...")
    with open(OUT_CATEGORY_INDEX_PKL, "wb") as f:
        pickle.dump({k: np.array(v, dtype=np.int32) for k, v in category_index.items()}, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
