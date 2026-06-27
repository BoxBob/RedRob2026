"""
LEACE-Switch — Concept Erasure for Geometric Repulsion (768-dim)

Computes:
  1. Combined projection matrix P = I - V_k^T @ V_k  (backward compat)
  2. Per-concept matrices saved as .npz:
     { "junior developer": M_768x768, "intern": M_768x768, ... }

At live time: for each matched negation, v' = M_concept @ v
"""

import os
import sys

# Resolve workspace root and setup pycache
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(workspace_root, ".pycache")
sys.path.insert(0, workspace_root)

import json
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL, BINARY_DIM, LEACE_NEGATIVE_CONCEPTS, LEACE_RANK,
    OUT_LEACE_DIRECTIONS, OUT_LEACE_PROJECTION, OUT_LEACE_CONFIG,
    OUT_LEACE_MATRICES_NPZ,
)


def compute_leace_projection(model=None):
    """
    Compute LEACE concept erasure artifacts in 768-dim space.

    Produces:
        - Combined projection matrix (.npy) for backward compat
        - Per-concept matrices (.npz) for live-phase per-negation application

    Returns:
        P: combined projection matrix (768, 768)
        per_concept: dict mapping lowercased concept → (768, 768) matrix
    """
    concepts = LEACE_NEGATIVE_CONCEPTS

    if not concepts:
        print("  ⚠ No negative concepts configured. Skipping LEACE.")
        return None, None

    print(f"  Negative concepts ({len(concepts)}): {concepts}")

    # ── Load model (reuse if passed in) ──
    if model is None:
        print(f"  Loading embedding model ({EMBEDDING_MODEL})...")
        model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Encode each concept phrase → 384-dim ──
    print(f"  Encoding {len(concepts)} concept phrases...")
    concept_emb_384 = model.encode(concepts, convert_to_numpy=True, show_progress_bar=False)

    # Unit-normalize
    norms = np.linalg.norm(concept_emb_384, axis=1, keepdims=True)
    concept_emb_384 = np.divide(concept_emb_384, norms, out=np.zeros_like(concept_emb_384), where=norms != 0)

    # ── Duplicate to 768-dim ──
    concept_emb_768 = np.concatenate([concept_emb_384, concept_emb_384], axis=1)  # (K, 768)
    print(f"  Concept matrix shape: {concept_emb_768.shape}")

    I = np.eye(BINARY_DIM, dtype=np.float32)  # (768, 768)

    # ── Per-concept matrices ──
    # Each concept gets its own rank-1 projection: M_i = I - v_i @ v_i^T
    per_concept = {}
    npz_data = {}
    for idx, concept in enumerate(concepts):
        v = concept_emb_768[idx]          # (768,)
        v = v / (np.linalg.norm(v) + 1e-10)  # re-normalize
        v = v.reshape(1, -1)              # (1, 768)
        M = I - (v.T @ v)                 # (768, 768) rank-1 projection
        M = M.astype(np.float32)

        key = concept.lower().strip()
        per_concept[key] = M
        # np.savez keys can't have spaces, use underscores for storage
        storage_key = key.replace(" ", "_")
        npz_data[storage_key] = M
        print(f"    [{key}] matrix norm: {np.linalg.norm(M):.4f}")

    # ── Combined projection (backward compat) ──
    # Center and SVD across all concepts
    centroid = concept_emb_768.mean(axis=0, keepdims=True)
    C_centered = concept_emb_768 - centroid
    U, S, Vt = np.linalg.svd(C_centered, full_matrices=False)
    actual_rank = min(LEACE_RANK, len(S))
    V_k = Vt[:actual_rank]
    P = I - V_k.T @ V_k
    P = P.astype(np.float32)

    sym_error = np.max(np.abs(P - P.T))
    idem_error = np.max(np.abs(P @ P - P))
    print(f"  Combined projection — rank: {actual_rank}, sym_err: {sym_error:.2e}, idem_err: {idem_error:.2e}")

    # ── Save all artifacts ──
    for path in [OUT_LEACE_DIRECTIONS, OUT_LEACE_PROJECTION, OUT_LEACE_MATRICES_NPZ, OUT_LEACE_CONFIG]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    np.save(OUT_LEACE_DIRECTIONS, V_k.astype(np.float32))
    print(f"  Saved: {OUT_LEACE_DIRECTIONS}")

    np.save(OUT_LEACE_PROJECTION, P)
    print(f"  Saved: {OUT_LEACE_PROJECTION}")

    # Per-concept .npz — keys are underscore-separated concept names
    np.savez(OUT_LEACE_MATRICES_NPZ, **npz_data)
    print(f"  Saved: {OUT_LEACE_MATRICES_NPZ} ({len(npz_data)} concept matrices)")

    config = {
        "model": EMBEDDING_MODEL,
        "embedding_dim": BINARY_DIM,
        "combined_rank": actual_rank,
        "negative_concepts": concepts,
        "concept_keys": list(per_concept.keys()),
        "storage_keys": list(npz_data.keys()),
        "singular_values": S[:actual_rank].tolist(),
    }
    with open(OUT_LEACE_CONFIG, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {OUT_LEACE_CONFIG}")

    print(f"  [OK] LEACE-Switch complete!")
    return P, per_concept


if __name__ == "__main__":
    compute_leace_projection()
