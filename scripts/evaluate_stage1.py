import sys
import os
import time

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(workspace_root, ".pycache")
os.environ["HF_HOME"] = os.path.join(workspace_root, ".hf_cache")
sys.path.insert(0, workspace_root)

from src.live.stage1_recall import Stage1Retriever, read_docx_text
from src.config import RAW_JD
from src.test.preview_profiles import preview_profiles

def main():
    print("=" * 80)
    print("  STAGE 1 RECALL — EFFECTIVENESS EVALUATION (LLM v1.04)")
    print("=" * 80)

    if not os.path.exists(RAW_JD):
        print(f"JD file not found at {RAW_JD}")
        return

    jd_text = read_docx_text(RAW_JD)
    print(f"Loaded Job Description: {len(jd_text)} characters.")
    
    # Initialize the retriever
    retriever = Stage1Retriever(use_llm=True, use_leace=True)

    print("\n⏳ Warming up Numba JIT (compiling kernel)...")
    _ = retriever.search("test string for compilation", top_k=2)
    
    # ---------------------------------------------------------
    # RUN 1: BASELINE (Pure Vector Search, No LLM/LEACE)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("  RUN 1: BASELINE (No LLM Parsing, No LEACE Repulsion)")
    print("=" * 80)
    # Temporarily disable LLM
    temp_llm = retriever.use_llm
    retriever.use_llm = False
    
    base_results = retriever.search(jd_text, top_k=10)
    
    # ---------------------------------------------------------
    # RUN 2: FULL PIPELINE
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("  RUN 2: FULL PIPELINE (LLM Bitmask + Multi-Vector HyDE)")
    print("=" * 80)
    # Restore LLM
    retriever.use_llm = temp_llm
    
    full_results = retriever.search(jd_text, top_k=10)
    
    # ---------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("  COMPARISON: BASELINE vs FULL PIPELINE")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("  BASELINE (Top 10 Candidates)")
    print("-" * 80)
    preview_profiles(base_results)
    
    print("\n" + "-" * 80)
    print("  FULL PIPELINE (Top 10 Candidates)")
    print("-" * 80)
    preview_profiles(full_results)

if __name__ == "__main__":
    main()
