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
    GLINER_LABELS, GLINER_CONFIDENCE_THRESHOLD,
    OUT_BINARY_DAT, OUT_INDEX_JSON, OUT_INVERTED_INDEX_PKL,
    OUT_LEACE_MATRICES_NPZ, OUT_LEACE_CONFIG, RAW_JD,
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
        leace_matrices_path=None,
        use_gliner=True,
        use_leace=True,
    ):
        print("=" * 60)
        print("  STAGE 1 RETRIEVER — Initialization")
        print("=" * 60)
        init_start = time.time()

        binary_dat_path     = binary_dat_path     or OUT_BINARY_DAT
        index_path          = index_path          or OUT_INDEX_JSON
        inverted_index_path = inverted_index_path or OUT_INVERTED_INDEX_PKL
        leace_matrices_path = leace_matrices_path or OUT_LEACE_MATRICES_NPZ

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

        # 4. Load per-concept LEACE matrices (.npz)
        self.leace_matrices = {}
        if use_leace and os.path.exists(leace_matrices_path):
            t0 = time.time()
            npz = np.load(leace_matrices_path)
            for storage_key in npz.files:
                # Convert storage key (underscores) back to concept key (spaces)
                concept_key = storage_key.replace("_", " ")
                self.leace_matrices[concept_key] = npz[storage_key]
            print(f"  [LEACE]  {len(self.leace_matrices)} concept matrices loaded ({(time.time()-t0)*1000:.0f}ms)")
        elif use_leace:
            print(f"  [LEACE]  ⚠ Not found: {leace_matrices_path}")

        # 5. Load the bi-encoder model
        t0 = time.time()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  [MODEL]  {EMBEDDING_MODEL} loaded ({(time.time()-t0)*1000:.0f}ms)")

        # 6. Initialize GLiNER (lazy — loaded on first use)
        self.use_gliner = use_gliner
        self.gliner = None

        # 7. Precompute popcount LUT
        self.popcount_lut = get_popcount_lut()

        total_init = (time.time() - init_start) * 1000
        print(f"  ─────────────────────────────────────")
        print(f"  Init complete: {total_init:.0f}ms")
        print("=" * 60)

    # ─────────────────────────────────────────────────────────────
    # [PARSE] GLiNER Dynamic JD Parsing
    # ─────────────────────────────────────────────────────────────
    def _parse_jd(self, jd_text):
        """
        Extract hard categorical exclusions and semantic negations from JD text.

        Returns:
            dict with keys:
                'excluded_companies': list[str]
                'excluded_industries': list[str]
                'excluded_titles': list[str]
                'required_locations': list[str]
                'required_skills': list[str]
                'negative_concepts': list[str]  — for LEACE matching
        """
        if not self.use_gliner:
            return {
                'excluded_companies': [], 'excluded_industries': [],
                'excluded_titles': [], 'required_locations': [],
                'required_skills': [], 'negative_concepts': [],
            }

        # Lazy-init GLiNER
        if self.gliner is None:
            from src.preprocessing.GLiNER_setup import initialize_gliner_session
            self.gliner = initialize_gliner_session()

        # Split by newline to avoid BERT 512-token truncation
        chunks = [c.strip() for c in jd_text.split('\n') if c.strip()]
        entities = []
        for chunk in chunks:
            chunk_entities = self.gliner.predict_entities(
                chunk, GLINER_LABELS, threshold=GLINER_CONFIDENCE_THRESHOLD
            )
            entities.extend(chunk_entities)

        parsed = {
            'excluded_companies': [],
            'excluded_industries': [],
            'excluded_titles': [],
            'required_locations': [],
            'required_skills': [],
            'negative_concepts': [],
        }

        for ent in entities:
            label = ent['label']
            text = ent['text'].strip().lower()
            score = ent['score']

            # We safely ignore the "Good" contrastive labels (they are just buckets to prevent false positives)
            if label in ("hiring company", "preferred company", "target job title"):
                continue
                
            if label == "company the candidate must not have worked for":
                parsed['excluded_companies'].append(text)
            elif label == "mandatory job location":
                parsed['required_locations'].append(text)
            elif label == "mandatory technical skill":
                parsed['required_skills'].append(text)
            elif label == "disqualified skill or unwanted experience":
                parsed['negative_concepts'].append(text)

        return parsed

    # ─────────────────────────────────────────────────────────────
    # [MASK] Categorical Bitmask Construction
    # ─────────────────────────────────────────────────────────────
    def _build_bitmask(self, exclusions):
        """
        Build a boolean eligibility mask from categorical exclusions.

        Looks up exclusions in the inverted index and marks matching
        candidates as ineligible.

        Args:
            exclusions: dict from _parse_jd()

        Returns:
            eligible_mask: (N,) bool — True = eligible
            mask_packed: (ceil(N/32),) uint32 — packed for kernel
        """
        eligible = np.ones(self.num_candidates, dtype=bool)

        if not self.inverted_index:
            return eligible, build_mask_packed(eligible, self.num_candidates)

        # Map exclusion types to inverted index field names
        exclusion_map = {
            'excluded_companies': 'current_company',
            'excluded_industries': 'current_industry',
            'excluded_titles': 'current_title',
        }

        # For each excluded company, look for candidates at that company
        for exc_type, field_name in exclusion_map.items():
            if field_name not in self.inverted_index:
                continue
            field_index = self.inverted_index[field_name]
            for value in exclusions.get(exc_type, []):
                # Case-insensitive match
                value_lower = value.lower().strip()
                for idx_key, row_indices in field_index.items():
                    if value_lower in idx_key.lower():
                        for idx in row_indices:
                            if idx < self.num_candidates:
                                eligible[idx] = False

        mask_packed = build_mask_packed(eligible, self.num_candidates)
        num_excluded = int(np.sum(~eligible))
        return eligible, mask_packed, num_excluded

    # ─────────────────────────────────────────────────────────────
    # [ENCODE] Query Encoding + LEACE Repulsion + Binarization
    # ─────────────────────────────────────────────────────────────
    def _encode_and_binarize(self, jd_text, negations):
        """
        Encode JD → 384-dim → duplicate → 768-dim → LEACE repulsion → binarize.

        Args:
            jd_text: raw JD text
            negations: list of negative concept strings from GLiNER

        Returns:
            query_binary: (96,) uint8 — bit-packed query vector
        """
        # BGE encode with query prefix
        prefixed = QUERY_PREFIX + jd_text
        emb_384 = self.model.encode(
            [prefixed], convert_to_numpy=True, show_progress_bar=False
        )[0]  # (384,)

        # Unit-normalize
        norm = np.linalg.norm(emb_384)
        if norm > 0:
            emb_384 = emb_384 / norm

        # Duplicate to 768-dim
        emb_768 = np.concatenate([emb_384, emb_384]).astype(np.float32)  # (768,)

        # Apply per-concept LEACE repulsion
        if negations and self.leace_matrices:
            for concept_text in negations:
                concept_key = concept_text.lower().strip()
                # Try exact match first, then substring match
                matched_key = None
                if concept_key in self.leace_matrices:
                    matched_key = concept_key
                else:
                    for k in self.leace_matrices:
                        if k in concept_key or concept_key in k:
                            matched_key = k
                            break

                if matched_key is not None:
                    M = self.leace_matrices[matched_key]
                    emb_768 = (M @ emb_768).astype(np.float32)
                    print(f"    LEACE applied: '{concept_text}' → matrix '{matched_key}'")

        # Sign threshold → binarize → packbits
        binary_bits = (emb_768 >= 0).astype(np.uint8)  # (768,)
        query_binary = np.packbits(binary_bits)          # (96,)

        return query_binary

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
            list of candidate IDs (strings), ranked by Hamming distance
        """
        top_k = top_k or TOP_K
        timings = {}

        print(f"\n{'─' * 60}")
        print(f"  STAGE 1 SEARCH — Top {top_k}")
        print(f"{'─' * 60}")

        # ── [PARSE] ──
        t0 = time.time()
        parsed = self._parse_jd(jd_text)
        timings['PARSE'] = (time.time() - t0) * 1000
        negations = parsed['negative_concepts']
        print(f"  [PARSE]   {timings['PARSE']:.1f}ms | exclusions: {sum(len(v) for v in parsed.values())} entities")
        for key, vals in parsed.items():
            if vals:
                print(f"            {key}: {vals}")

        # ── [MASK] ──
        t0 = time.time()
        eligible, mask_packed, num_excluded = self._build_bitmask(parsed)
        timings['MASK'] = (time.time() - t0) * 1000
        print(f"  [MASK]    {timings['MASK']:.1f}ms | {num_excluded} candidates excluded, {self.num_candidates - num_excluded} eligible")

        # ── [ENCODE] ──
        t0 = time.time()
        query_binary = self._encode_and_binarize(jd_text, negations)
        timings['ENCODE'] = (time.time() - t0) * 1000
        print(f"  [ENCODE]  {timings['ENCODE']:.1f}ms | query shape: {query_binary.shape}")

        # ── [SCAN] ──
        t0 = time.time()
        distances = hamming_scan_masked(
            self.candidate_binary,
            query_binary,
            mask_packed,
            self.num_candidates,
            PACKED_DIM,
            self.popcount_lut,
        )
        timings['SCAN'] = (time.time() - t0) * 1000
        print(f"  [SCAN]    {timings['SCAN']:.1f}ms | Numba Hamming kernel complete")

        # ── [RANK] ──
        t0 = time.time()
        actual_top_k = min(top_k, self.num_candidates)
        if actual_top_k == self.num_candidates:
            top_indices = np.argsort(distances)
        else:
            top_indices_unsorted = np.argpartition(distances, actual_top_k)[:actual_top_k]
            top_indices = top_indices_unsorted[np.argsort(distances[top_indices_unsorted])]

        top_candidate_ids = [self.candidate_ids[i] for i in top_indices]
        timings['RANK'] = (time.time() - t0) * 1000
        print(f"  [RANK]    {timings['RANK']:.1f}ms | top-{actual_top_k} extracted")

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

        return top_candidate_ids


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

    # Initialize retriever (GLiNER disabled by default for fast testing)
    retriever = Stage1Retriever(use_gliner=True, use_leace=True)

    # Warm up Numba (first call triggers compilation)
    print("\n⏳ Warming up Numba kernel (first call compiles)...")
    _ = retriever.search(jd_text, top_k=10)

    # Real search
    print("\n" + "=" * 60)
    print("  REAL SEARCH RUN (Numba compiled)")
    print("=" * 60)
    top_results = retriever.search(jd_text, top_k=TOP_K)

    print(f"\nTop 10 Matches: {top_results[:10]}")