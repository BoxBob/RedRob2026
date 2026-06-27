"""
Categorical Inverted Index Builder

Single-pass over candidates.jsonl → categorical_index.json + inverted_index.pkl
Maps each categorical field → each value → list of row indices (ints).
"""

import os
import sys

# Resolve workspace root and setup pycache
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(workspace_root, ".pycache")
sys.path.insert(0, workspace_root)

import json
import pickle
from collections import defaultdict

from src.config import (
    CATEGORICAL_FIELDS, NOTICE_PERIOD_BUCKETS,
    RAW_CANDIDATES, OUT_CATEGORICAL_JSON, OUT_INVERTED_INDEX_PKL,
)


def _bucket_notice_period(days):
    """Convert notice_period_days integer to a bucket string."""
    if days is None:
        return None
    try:
        days = int(days)
    except (ValueError, TypeError):
        return None
    for low, high, label in NOTICE_PERIOD_BUCKETS:
        if low <= days <= high:
            return label
    return "90+"


def _extract_field(cand, field_name, field_spec, row_idx):
    """
    Extract one or more (field_value) entries for a candidate.
    Returns a list of string values (most fields return a single-element list).
    """
    # ── Special extraction logic ──
    if field_spec is None:
        if field_name == "education_tier":
            education = cand.get("education", [])
            if education and isinstance(education, list):
                tier = education[0].get("tier")
                return [str(tier)] if tier else []
            return []

        elif field_name == "skills":
            skills = cand.get("skills", [])
            values = []
            for s in skills:
                name = s.get("name")
                if name and isinstance(name, str):
                    cleaned = name.strip().lower()
                    if cleaned:
                        values.append(cleaned)
            return values

        elif field_name == "notice_period_bucket":
            days = cand.get("redrob_signals", {}).get("notice_period_days")
            bucket = _bucket_notice_period(days)
            return [bucket] if bucket else []

        return []

    # ── Standard two-level extraction ──
    top_key, nested_key = field_spec
    parent = cand.get(top_key, {})
    if not isinstance(parent, dict):
        return []

    value = parent.get(nested_key)

    # Handle booleans → string
    if isinstance(value, bool):
        return [str(value).lower()]
    if value is not None and str(value).strip():
        return [str(value).strip()]
    return []


def build_categorical_index(json_filepath=None, output_path=None):
    """
    Main entry point: read candidates, build inverted index, write JSON.
    """
    json_filepath = json_filepath or RAW_CANDIDATES
    output_path   = output_path   or OUT_CATEGORICAL_JSON

    print(f"  Loading candidates from {json_filepath}...")
    if json_filepath.endswith('.jsonl'):
        with open(json_filepath, 'r', encoding='utf-8') as f:
            candidates = [json.loads(line) for line in f if line.strip()]
    else:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            candidates = json.load(f)

    num_candidates = len(candidates)
    print(f"  Total candidates: {num_candidates}")
    print(f"  Indexing {len(CATEGORICAL_FIELDS)} fields: {list(CATEGORICAL_FIELDS.keys())}")

    # ── Build the index ──
    # Structure: { field_name: { value: [row_indices] } }
    index = {field_name: defaultdict(list) for field_name in CATEGORICAL_FIELDS}

    for row_idx, cand in enumerate(candidates):
        for field_name, field_spec in CATEGORICAL_FIELDS.items():
            values = _extract_field(cand, field_name, field_spec, row_idx)
            for v in values:
                index[field_name][v].append(row_idx)

    # ── Convert defaultdicts to regular dicts for JSON serialization ──
    index_serializable = {}
    for field_name, value_map in index.items():
        index_serializable[field_name] = dict(value_map)

    # ── Print stats ──
    print(f"  Index statistics:")
    for field_name, value_map in index_serializable.items():
        num_values = len(value_map)
        total_entries = sum(len(v) for v in value_map.values())
        print(f"    {field_name:<25} {num_values:>6} unique values, {total_entries:>8} entries")

    # ── Write JSON output ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"  Saving categorical index (JSON): {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index_serializable, f, ensure_ascii=False)
    json_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # ── Write PKL output (fast deserialization for live phase) ──
    pkl_path = OUT_INVERTED_INDEX_PKL
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    print(f"  Saving inverted index (PKL):  {pkl_path}...")
    with open(pkl_path, 'wb') as f:
        pickle.dump(index_serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)

    print(f"  [OK] Categorical index complete! (JSON: {json_size_mb:.1f} MB, PKL: {pkl_size_mb:.1f} MB)")
    return index_serializable


if __name__ == "__main__":
    build_categorical_index()
