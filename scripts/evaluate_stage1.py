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
    print("  STAGE 1 RECALL — EFFECTIVENESS EVALUATION")
    print("=" * 80)

    if not os.path.exists(RAW_JD):
        print(f"JD file not found at {RAW_JD}")
        return

    jd_text = read_docx_text(RAW_JD)
    print(f"Loaded Job Description: {len(jd_text)} characters.")
    
    # Initialize the retriever (we will mock GLiNER to avoid massive model downloads on Windows)
    retriever = Stage1Retriever(use_gliner=False, use_leace=True)

    # Mock the GLiNER parsing function so we can simulate the extraction pipeline safely
    def mock_parse_jd(jd_text):
        if not retriever.use_gliner:
            return {
                'excluded_companies': [], 'excluded_industries': [],
                'excluded_titles': [], 'required_locations': [],
                'required_skills': [], 'negative_concepts': [],
            }
        print("  [MOCK GLiNER] Simulating ONNX extraction...")
        return {
            'excluded_companies': ['Accenture'],
            'excluded_industries': ['Consulting'],
            'excluded_titles': ['Intern'],
            'required_locations': [],
            'required_skills': [],
            'negative_concepts': ['Junior Developer', 'Intern', 'Entry Level']
        }
    
    retriever._parse_jd = mock_parse_jd

    print("\n⏳ Warming up Numba JIT (compiling kernel)...")
    _ = retriever.search("test string for compilation", top_k=2)
    
    # ---------------------------------------------------------
    # RUN 1: BASELINE (Pure Vector Search, No Exclusions/LEACE)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("  RUN 1: BASELINE (No GLiNER Parsing, No LEACE Repulsion)")
    print("=" * 80)
    # Disable GLiNER and temporarily hide LEACE matrices
    retriever.use_gliner = False
    temp_leace = retriever.leace_matrices
    retriever.leace_matrices = {}
    
    base_results = retriever.search(jd_text, top_k=3)
    
    # ---------------------------------------------------------
    # RUN 2: FULL PIPELINE
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("  RUN 2: FULL PIPELINE (GLiNER Bitmask + LEACE Repulsion)")
    print("=" * 80)
    # Enable GLiNER and restore LEACE matrices
    retriever.use_gliner = True
    retriever.leace_matrices = temp_leace
    
    full_results = retriever.search(jd_text, top_k=3)
    
    # ---------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("  COMPARISON: BASELINE vs FULL PIPELINE")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("  BASELINE (Top 3 Candidates)")
    print("-" * 80)
    preview_profiles(base_results)
    
    print("\n" + "-" * 80)
    print("  FULL PIPELINE (Top 3 Candidates)")
    print("-" * 80)
    preview_profiles(full_results)

if __name__ == "__main__":
    main()
