"""
Stage 1 Recall — Live Binary Candidate Retrieval

Full pipeline:
  [PARSE]   GLiNER ONNX extraction of exclusions + negations from JD
  [MASK]    Categorical bitmask construction (uint32 packed)
  [ENCODE]  BGE encode → 768-dim → LEACE repulsion → binarize → 96-byte query
  [SCAN]    Numba SIMD Hamming distance kernel with bitmask short-circuit
  [RANK]    argpartition top-2000 extraction

All parameters from src/config.py.
"""

import os
import sys
import zipfile
import xml.etree.ElementTree as ET

# Resolve workspace root
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(workspace_root, ".pycache")
os.environ["HF_HOME"] = os.path.join(workspace_root, ".hf_cache")
sys.path.insert(0, workspace_root)

import json
import time
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import gc

from src.config import (
    EMBEDDING_MODEL, EMBEDDING_DIM, BINARY_DIM, PACKED_DIM,
    QUERY_PREFIX, NUM_CANDIDATES, TOP_K,
    OUT_BINARY_DAT, OUT_INDEX_JSON, OUT_INVERTED_INDEX_PKL,
    OUT_CATEGORY_INDEX_PKL, RAW_JD,
)
from src.live.hamming_kernel import (
    hamming_scan_masked, build_mask_packed, get_popcount_lut,
)


# ─────────────────────────────────────────────────────────────────────────────
# JD Text Extraction (zipfile + XML — no python-docx dependency)
# ─────────────────────────────────────────────────────────────────────────────
def read_docx_text(docx_path):
    """Extract plain text from a .docx file using stdlib only."""
    ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    with zipfile.ZipFile(docx_path) as z:
        tree = ET.parse(z.open('word/document.xml'))
    paragraphs = []
    for para in tree.iter(f'{ns}p'):
        texts = [t.text for t in para.iter(f'{ns}t') if t.text]
        if texts:
            paragraphs.append(''.join(texts))
    return '\n'.join(paragraphs)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 Retriever
# ─────────────────────────────────────────────────────────────────────────────
class Stage1Retriever:
    def __init__(
        self,
        binary_dat_path=None,
        index_path=None,
        inverted_index_path=None,
        use_llm=True,
        use_leace=True,
    ):
        print("=" * 60)
        print("  STAGE 1 RETRIEVER — Initialization")
        print("=" * 60)
        init_start = time.time()

        binary_dat_path     = binary_dat_path     or OUT_BINARY_DAT
        index_path          = index_path          or OUT_INDEX_JSON
        inverted_index_path = inverted_index_path or OUT_INVERTED_INDEX_PKL
        category_index_path = OUT_CATEGORY_INDEX_PKL

        # 1. Load candidate ID index
        t0 = time.time()
        with open(index_path, 'r') as f:
            self.candidate_ids = json.load(f)
        self.num_candidates = len(self.candidate_ids)
        print(f"  [INDEX]  {self.num_candidates} candidate IDs loaded ({(time.time()-t0)*1000:.0f}ms)")

        # 2. Memory-map the binary candidate matrix (zero-copy)
        t0 = time.time()
        self.candidate_binary = np.memmap(
            binary_dat_path,
            dtype='uint8',
            mode='r',
            shape=(self.num_candidates, PACKED_DIM),
        )
        print(f"  [MMAP]   Binary matrix: shape={self.candidate_binary.shape} ({(time.time()-t0)*1000:.0f}ms)")

        # 3. Load inverted index (pickle for fast deserialization)
        t0 = time.time()
        if os.path.exists(inverted_index_path):
            with open(inverted_index_path, 'rb') as f:
                self.inverted_index = pickle.load(f)
            print(f"  [IIDX]   Inverted index: {len(self.inverted_index)} fields ({(time.time()-t0)*1000:.0f}ms)")
        else:
            self.inverted_index = {}
            print(f"  [IIDX]   ⚠ Not found: {inverted_index_path}")

        # 4. Load Category Index
        self.category_index = {}
        if os.path.exists(category_index_path):
            t0 = time.time()
            try:
                with open(category_index_path, 'rb') as f:
                    self.category_index = pickle.load(f)
                print(f"  [CAT]    Category index loaded ({(time.time()-t0)*1000:.0f}ms)")
            except Exception as e:
                print(f"  [CAT]    ⚠ Failed to load: {e}")
        else:
            print(f"  [CAT]    ⚠ Not found: {category_index_path}")

        # 5. Load the bi-encoder model
        t0 = time.time()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  [MODEL]  {EMBEDDING_MODEL} loaded ({(time.time()-t0)*1000:.0f}ms)")

        # 6. Initialize LLM (lazy — loaded on first use)
        self.use_llm = use_llm

        # 7. Precompute popcount LUT
        self.popcount_lut = get_popcount_lut()

        total_init = (time.time() - init_start) * 1000
        print(f"  ─────────────────────────────────────")
        print(f"  Init complete: {total_init:.0f}ms")
        print("=" * 60)

    # ─────────────────────────────────────────────────────────────
    # [PARSE] LLM Dynamic JD Parsing
    # ─────────────────────────────────────────────────────────────
    def _parse_jd(self, jd_text):
        """
        Extract hard categorical exclusions and soft repulsions from JD text using the LLM.

        Returns:
            dict with keys:
                'excluded_companies': list[str]
                'negative_concepts': list[str]  — for LEACE matching
        """
        if not self.use_llm:
            return {
                'excluded_companies': [],
                'excluded_title_categories': [],
                'hyde_resume': "",
                'sub_queries': {}
            }

        from src.live.jd_parser import parse_jd
        
        parsed_data = parse_jd(jd_text)
        
        # Inject excluded_companies at the top level for backwards compatibility
        parsed_data['excluded_companies'] = parsed_data.get('hard_exclusions', {}).get('strictly_excluded_companies', [])
        
        return parsed_data

    # ─────────────────────────────────────────────────────────────
    # [MASK] Categorical Bitmask Construction
    # ─────────────────────────────────────────────────────────────
    def _build_bitmask(self, exclusions):
        """
        Build a boolean eligibility mask combining companies and title categories.
        """
        eligible = np.ones(self.num_candidates, dtype=bool)

        # 1. Company Exclusions (from inverted index)
        if self.inverted_index and 'current_company' in self.inverted_index:
            field_index = self.inverted_index['current_company']
            for company in exclusions.get('excluded_companies', []):
                company_lower = company.lower().strip()
                for idx_key, row_indices in field_index.items():
                    if company_lower in idx_key.lower():
                        for idx in row_indices:
                            if idx < self.num_candidates:
                                eligible[idx] = False

        # 2. Title Category Exclusions (from category index)
        for cat in exclusions.get('excluded_title_categories', []):
            if cat in self.category_index:
                indices = self.category_index[cat]
                eligible[indices] = False

        mask_packed = build_mask_packed(eligible, self.num_candidates)
        num_excluded = int(np.sum(~eligible))
        return eligible, mask_packed, num_excluded

    # ─────────────────────────────────────────────────────────────
    # [ENCODE] Query Encoding + LEACE Repulsion + Binarization
    # ─────────────────────────────────────────────────────────────
    def _encode_and_binarize(self, text):
        """Encode arbitrary text to a 96-byte packed binary vector."""
        if not text:
            # Return zeros if text is empty
            return np.zeros(96, dtype=np.uint8)
            
        prefixed = QUERY_PREFIX + text
        emb_384 = self.model.encode(
            [prefixed], convert_to_numpy=True, show_progress_bar=False
        )[0]

        norm = np.linalg.norm(emb_384)
        if norm > 0:
            emb_384 = emb_384 / norm

        emb_768 = np.concatenate([emb_384, emb_384]).astype(np.float32)
        binary_bits = (emb_768 >= 0).astype(np.uint8)
        return np.packbits(binary_bits)

    # ─────────────────────────────────────────────────────────────
    # [SEARCH] Full Pipeline
    # ─────────────────────────────────────────────────────────────
    def search(self, jd_text, top_k=None):
        """
        Full Stage 1 search pipeline.

        Args:
            jd_text: raw JD text (string)
            top_k: number of candidates to return (default: config.TOP_K)

        Returns:
            tuple: (list of candidate IDs, dict of parsed JD)
        """
        top_k = top_k or TOP_K
        timings = {}

        print(f"\n{'─' * 60}")
        print(f"  STAGE 1 SEARCH — Top {top_k} (Multi-Vector HyDE)")
        print(f"{'─' * 60}")

        # ── [PARSE] ──
        t0 = time.time()
        parsed = self._parse_jd(jd_text)
        timings['PARSE'] = (time.time() - t0) * 1000
        
        hyde_resume = parsed.get('hyde_resume', "") or jd_text
        sub_queries = parsed.get('sub_queries', {})

        print(f"  [PARSE]   {timings['PARSE']:.1f}ms | exclusions: {len(parsed.get('excluded_companies', []))} entities")
        if parsed.get('excluded_companies'):
            print(f"            excluded_companies: {parsed['excluded_companies']}")
        if parsed.get('excluded_title_categories'):
            print(f"            excluded_title_categories: {parsed['excluded_title_categories']}")

        # ── [MASK] ──
        t0 = time.time()
        eligible_mask, mask_packed, num_excluded = self._build_bitmask(parsed)
        timings['MASK'] = (time.time() - t0) * 1000
        print(f"  [MASK]    {timings['MASK']:.1f}ms | {num_excluded} candidates excluded, {self.num_candidates - num_excluded} eligible")

        # ── [ENCODE] ──
        t0 = time.time()
        query_binaries = []
        
        # Primary vector (HyDE)
        q_hyde = self._encode_and_binarize(hyde_resume)
        query_binaries.append(q_hyde)
        
        # Sub-vectors
        for field in ['skills', 'role', 'domain']:
            val = sub_queries.get(field, "")
            if val:
                query_binaries.append(self._encode_and_binarize(val))
                
        timings['ENCODE'] = (time.time() - t0) * 1000
        print(f"  [ENCODE]  {timings['ENCODE']:.1f}ms | encoded {len(query_binaries)} vectors")

        # ── [SCAN & RANK] ──
        t0 = time.time()
        
        from collections import defaultdict
        all_candidate_scores = defaultdict(lambda: float('inf'))
        
        scan_time_total = 0
        for q_bin in query_binaries:
            t_s = time.time()
            distances = hamming_scan_masked(
                self.candidate_binary,
                q_bin,
                mask_packed,
                self.num_candidates,
                PACKED_DIM,
                self.popcount_lut,
            )
            scan_time_total += (time.time() - t_s)
            
            valid_indices = np.where(eligible_mask)[0]
            if len(valid_indices) == 0:
                continue
                
            valid_distances = distances[valid_indices]
            
            k = min(top_k, len(valid_indices))
            if k == 0: continue
                
            idx_top_k = np.argpartition(valid_distances, k - 1)[:k]
            idx_top_k = idx_top_k[np.argsort(valid_distances[idx_top_k])]
            
            global_indices = valid_indices[idx_top_k]
            top_distances = valid_distances[idx_top_k]
            
            for rank, (g_idx, dist) in enumerate(zip(global_indices, top_distances)):
                if rank < all_candidate_scores[g_idx]:
                    all_candidate_scores[g_idx] = rank
                    
        timings['SCAN'] = scan_time_total * 1000
        print(f"  [SCAN]    {timings['SCAN']:.1f}ms | Numba Hamming kernel across {len(query_binaries)} vectors")

        t0 = time.time()
        sorted_candidates = sorted(all_candidate_scores.items(), key=lambda x: x[1])
        final_top = sorted_candidates[:top_k]
        
        results = []
        for g_idx, best_rank in final_top:
            cand_id = self.candidate_ids[g_idx]
            results.append((cand_id, best_rank))
            
        timings['RANK'] = (time.time() - t0) * 1000
        actual_top_k = len(results)
        print(f"  [RANK]    {timings['RANK']:.1f}ms | top-{actual_top_k} extracted by best sub-query rank")

        # ── Summary ──
        total = sum(timings.values())
        print(f"  {'─' * 40}")
        for step, ms in timings.items():
            print(f"  [{step:<8}] {ms:>8.1f}ms")
        print(f"  {'─' * 40}")
        print(f"  {'TOTAL':<10}  {total:>8.1f}ms")
        print(f"{'─' * 60}")

        # Cleanup
        del distances
        gc.collect()

        return [cand_id for cand_id, _ in results], parsed


# ─────────────────────────────────────────────────────────────────────────────
# Main — Test with real JD
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')

    # Read JD from docx
    jd_path = RAW_JD
    if os.path.exists(jd_path):
        print(f"Reading JD from: {jd_path}")
        jd_text = read_docx_text(jd_path)
        print(f"JD length: {len(jd_text)} chars\n")
    else:
        print("JD file not found, using dummy text.\n")
        jd_text = """
        We are looking for a Senior Software Engineer with deep expertise in Python,
        Next.js, and AWS. The ideal candidate has experience building scalable microservices
        and optimizing databases. 5+ years of experience required.
        Not from consulting firms. No junior developers.
        """

    # Initialize retriever (LLM enabled)
    retriever = Stage1Retriever(use_llm=True, use_leace=True)

    # Warm up Numba (first call triggers compilation)
    print("\n⏳ Warming up Numba kernel (first call compiles)...")
    _ = retriever.search(jd_text, top_k=10)

    # Real search
    print("\n" + "=" * 60)
    print("  REAL SEARCH RUN (Numba compiled)")
    print("=" * 60)
    top_results, parsed = retriever.search(jd_text, top_k=TOP_K)

    print(f"\nTop 10 Matches: {top_results[:10]}")