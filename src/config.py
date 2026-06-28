"""
Central Configuration — Single source of truth for the Redrob pipeline.

All modules import from src.config. To change any parameter (model, fields,
concepts, paths), edit ONLY this file — every module auto-adapts.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Workspace Root (resolved once, used everywhere)
# ─────────────────────────────────────────────────────────────────────────────
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
BINARY_DIM = 768              # 384 × 2 (second half = placeholder duplication for future fine-tuned encoder)
PACKED_DIM = 96               # 768 / 8 bits per byte
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ENCODE_BATCH_SIZE = 1000      # candidates per encoding batch

# ─────────────────────────────────────────────────────────────────────────────
# Categorical Index Fields
# ─────────────────────────────────────────────────────────────────────────────
# key_name → (top_level_key, nested_key)  or  None for special extraction logic
CATEGORICAL_FIELDS = {
    "location":             ("profile", "location"),
    "country":              ("profile", "country"),
    "current_industry":     ("profile", "current_industry"),
    "current_company_size": ("profile", "current_company_size"),
    "current_title":        ("profile", "current_title"),
    "current_company":      ("profile", "current_company"),
    "preferred_work_mode":  ("redrob_signals", "preferred_work_mode"),
    "willing_to_relocate":  ("redrob_signals", "willing_to_relocate"),
    "education_tier":       None,   # special: education[0].tier
    "skills":               None,   # special: all skills[].name, lowercased + stripped
    "notice_period_bucket": None,   # special: bucketed from redrob_signals.notice_period_days
}

# Buckets for notice period (inclusive bounds). Anything above last upper → "90+"
NOTICE_PERIOD_BUCKETS = [
    (0,  30,  "0-30"),
    (31, 60,  "31-60"),
    (61, 90,  "61-90"),
    # Above 90 → "90+"
]

# ─────────────────────────────────────────────────────────────────────────────
# LEACE-Switch — Negative Concept Erasure
# ─────────────────────────────────────────────────────────────────────────────
LEACE_NEGATIVE_CONCEPTS = [
    "academia",
    "consulting",
    "junior",
    "management",
    "non-technical"
]
LEACE_RANK = 2  # Number of concept directions to erase (top-k SVD components)

# ─────────────────────────────────────────────────────────────────────────────
# Live Phase — LLM JD Parsing (onnxruntime-genai)
# ─────────────────────────────────────────────────────────────────────────────
LLM_MODEL_REPO = "microsoft/Phi-4-mini-instruct-onnx"
LLM_MODEL_FOLDER = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096

# ─────────────────────────────────────────────────────────────────────────────
# Live Phase — Hamming Search
# ─────────────────────────────────────────────────────────────────────────────
NUM_CANDIDATES = 100_000
TOP_K = 2000
HAMMING_STRIDE = 32           # candidates per uint32 bitmask chunk

# ─────────────────────────────────────────────────────────────────────────────
# File Paths (relative to WORKSPACE_ROOT)
# ─────────────────────────────────────────────────────────────────────────────
# Raw inputs
RAW_CANDIDATES       = os.path.join(WORKSPACE_ROOT, "data", "raw", "candidates.jsonl")
RAW_JD               = os.path.join(WORKSPACE_ROOT, "data", "raw", "job_description.docx")

# Processed outputs — Embeddings
OUT_FLOAT32_NPY      = os.path.join(WORKSPACE_ROOT, "data", "processed", "candidate_embeddings_float32.npy")
OUT_FLOAT768_DAT     = os.path.join(WORKSPACE_ROOT, "data", "processed", "candidate_embeddings_768.dat")
OUT_BINARY_DAT       = os.path.join(WORKSPACE_ROOT, "data", "processed", "candidate_index_binary.dat")
OUT_INDEX_JSON       = os.path.join(WORKSPACE_ROOT, "data", "processed", "candidate_index.json")

# Processed outputs — Categorical Index
OUT_CATEGORICAL_JSON = os.path.join(WORKSPACE_ROOT, "data", "processed", "categorical_index.json")
OUT_INVERTED_INDEX_PKL = os.path.join(WORKSPACE_ROOT, "data", "processed", "inverted_index.pkl")
OUT_CATEGORY_INDEX_PKL = os.path.join(WORKSPACE_ROOT, "data", "processed", "category_index.pkl")

# Processed outputs — LEACE
OUT_LEACE_DIRECTIONS = os.path.join(WORKSPACE_ROOT, "data", "processed", "leace_concept_directions.npy")
OUT_LEACE_PROJECTION = os.path.join(WORKSPACE_ROOT, "data", "processed", "leace_projection_matrix.npy")
OUT_LEACE_CONFIG     = os.path.join(WORKSPACE_ROOT, "data", "processed", "leace_config.json")

# Processed outputs — Fraud Detection
OUT_FRAUD_FLAGS_JSON = os.path.join(WORKSPACE_ROOT, "data", "processed", "fraud_flags.json")
OUT_TFIDF_VECTORIZER_PKL = os.path.join(WORKSPACE_ROOT, "data", "processed", "tfidf_vectorizer.pkl")
OUT_TFIDF_MATRIX_NPZ = os.path.join(WORKSPACE_ROOT, "data", "processed", "tfidf_matrix.npz")
OUT_ANOMALY_MODEL_PKL = os.path.join(WORKSPACE_ROOT, "data", "processed", "anomaly_model.pkl")
