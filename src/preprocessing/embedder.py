"""
Embedder Module — Dense Embedding Generation with Triple Export

Generates candidate embeddings using the BGE model and exports three formats:
  1. candidate_embeddings_float32.npy  — (N, 384) float32 for cosine search
  2. candidate_embeddings_768.dat      — (N, 768) float32 memmap (384 duplicated, for LEACE)
  3. candidate_index_binary.dat        — (N, 96)  uint8 memmap (768 → packbits, for Hamming)
  4. candidate_index.json              — row-to-candidate-ID mapping
"""

import os
import sys

# Resolve workspace root and setup pycache
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(workspace_root, ".pycache")
sys.path.insert(0, workspace_root)

import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL, EMBEDDING_DIM, BINARY_DIM, PACKED_DIM,
    ENCODE_BATCH_SIZE, OUT_FLOAT32_NPY, OUT_FLOAT768_DAT,
    OUT_BINARY_DAT, OUT_INDEX_JSON, RAW_CANDIDATES,
)


def build_candidate_text(cand):
    """Flatten a candidate dict into a single text string for embedding."""
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
        f"Experience: {history_str}",
    ]

    return " ".join([p for p in text_parts if p.strip()])


def process_candidates(
    json_filepath=None,
    float32_npy_path=None,
    float768_dat_path=None,
    binary_dat_path=None,
    index_json_path=None,
    batch_size=None,
):
    """
    Main entry point: encode candidates → write 3 embedding files + 1 index.

    All arguments default to config.py values if not provided.
    """
    json_filepath    = json_filepath    or RAW_CANDIDATES
    float32_npy_path = float32_npy_path or OUT_FLOAT32_NPY
    float768_dat_path = float768_dat_path or OUT_FLOAT768_DAT
    binary_dat_path  = binary_dat_path  or OUT_BINARY_DAT
    index_json_path  = index_json_path  or OUT_INDEX_JSON
    batch_size       = batch_size       or ENCODE_BATCH_SIZE

    # ── Load model ──
    print(f"  Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Load candidate data ──
    print(f"  Loading candidate data from {json_filepath}...")
    if json_filepath.endswith('.jsonl'):
        with open(json_filepath, 'r', encoding='utf-8') as f:
            candidates = [json.loads(line) for line in f if line.strip()]
    else:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            candidates = json.load(f)

    num_candidates = len(candidates)
    print(f"  Total candidates: {num_candidates}")

    # ── Ensure output directories exist ──
    for path in [float32_npy_path, float768_dat_path, binary_dat_path, index_json_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # ── Create memory-mapped output files ──
    print(f"  Creating float768 memmap: {float768_dat_path}  shape=({num_candidates}, {BINARY_DIM})")
    fp_768 = np.memmap(float768_dat_path, dtype='float32', mode='w+', shape=(num_candidates, BINARY_DIM))

    print(f"  Creating binary memmap:   {binary_dat_path}  shape=({num_candidates}, {PACKED_DIM})")
    fp_bin = np.memmap(binary_dat_path, dtype='uint8', mode='w+', shape=(num_candidates, PACKED_DIM))

    # We also accumulate the 384-dim float32 vectors in memory for .npy save
    # For 100K × 384 × 4 bytes ≈ 146 MB — fits comfortably in RAM
    all_float32 = np.empty((num_candidates, EMBEDDING_DIM), dtype=np.float32)

    candidate_ids = []

    # ── Batch encode and stream to all outputs ──
    start_time = time.time()
    for i in range(0, num_candidates, batch_size):
        batch = candidates[i:i + batch_size]
        batch_end = i + len(batch)

        # Build text representations
        texts = [build_candidate_text(cand) for cand in batch]
        ids = [cand.get('candidate_id') for cand in batch]
        candidate_ids.extend(ids)

        # Encode to float32 (384-dim) — no instruction prefix for documents
        emb_384 = model.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=False)

        # Unit-normalize for cosine similarity via dot product
        norms = np.linalg.norm(emb_384, axis=1, keepdims=True)
        emb_384 = np.divide(emb_384, norms, out=np.zeros_like(emb_384), where=norms != 0)

        # Store 384-dim float32
        all_float32[i:batch_end] = emb_384

        # Duplicate 384 → 768 (placeholder for future second encoder)
        emb_768 = np.concatenate([emb_384, emb_384], axis=1)

        # Stream 768-dim float32 to memmap
        fp_768[i:batch_end] = emb_768
        fp_768.flush()

        # Binarize: sign threshold → packbits → 96 bytes uint8
        binary_bits = (emb_768 >= 0).astype(np.uint8)
        packed = np.packbits(binary_bits, axis=1)  # (batch, 96)
        fp_bin[i:batch_end] = packed
        fp_bin.flush()

        elapsed = round(time.time() - start_time, 2)
        print(f"  Batch {i:>6} → {batch_end:>6} / {num_candidates}  ({elapsed}s)")

    # ── Save 384-dim float32 as .npy ──
    print(f"  Saving float32 npy: {float32_npy_path}")
    np.save(float32_npy_path, all_float32)

    # ── Save candidate ID index ──
    print(f"  Saving candidate index: {index_json_path}")
    with open(index_json_path, 'w') as f:
        json.dump(candidate_ids, f)

    # ── Cleanup ──
    del fp_768, fp_bin, all_float32
    elapsed_total = round(time.time() - start_time, 2)

    print(f"  ✅ Embedding complete! ({elapsed_total}s)")
    return num_candidates


if __name__ == "__main__":
    process_candidates()
