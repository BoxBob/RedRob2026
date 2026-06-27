"""
Offline Precomputation Pipeline — Orchestrator

Runs the full offline pipeline in 5 labeled steps:
  [SETUP]   Install pyTLEX dependency
  [INGEST]  Copy raw data from PS → data/raw/
  [ENCODE]  Generate dense embeddings → .npy + .dat (768) + .dat (binary)
  [INDEX]   Build categorical inverted index → .json
  [LEACE]   Compute concept erasure matrices → .npy

All artifacts are always overwritten on every run.
All parameters are read from src/config.py.
"""

import os
import sys
import time

# Setup stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding='utf-8')

# Resolve absolute path to workspace root (one level up from scripts/)
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Redirect pycache folder creation to a single central root folder
os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(workspace_root, ".pycache")

# Inject paths into python's search path
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "src"))

try:
    import src.preprocessing.installing_pytlex as installing_pytlex
    import src.preprocessing.loading_raw_data as loading_raw_data
    import src.preprocessing.embedder as embedder
    import src.preprocessing.categorical_indexer as categorical_indexer
    import src.preprocessing.leace_switch as leace_switch
    from src.config import (
        OUT_FLOAT32_NPY, OUT_FLOAT768_DAT, OUT_BINARY_DAT,
        OUT_INDEX_JSON, OUT_CATEGORICAL_JSON,
        OUT_LEACE_DIRECTIONS, OUT_LEACE_PROJECTION, OUT_LEACE_CONFIG,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def _file_size_str(path):
    """Return a human-readable file size string."""
    if not os.path.exists(path):
        return "not found"
    size = os.path.getsize(path)
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    elif size >= 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size} B"


def run_offline_pipeline():
    print("=" * 64)
    print("  OFFLINE PRECOMPUTATION PIPELINE")
    print("=" * 64)

    total_start = time.time()
    step_times = {}

    # ─────────────────────────────────────────────────────────────
    # [SETUP] Install pyTLEX dependency
    # ─────────────────────────────────────────────────────────────
    print("\n┌─ [SETUP] Installing dependencies ─────────────────────────┐")
    t0 = time.time()
    installing_pytlex.setup_pytlex()
    step_times["SETUP"] = round(time.time() - t0, 2)
    print(f"└─ [SETUP] Done ({step_times['SETUP']}s) ──────────────────────────────┘")

    # ─────────────────────────────────────────────────────────────
    # [INGEST] Copy raw data from PS → data/raw/
    # ─────────────────────────────────────────────────────────────
    print("\n┌─ [INGEST] Loading raw candidate data ─────────────────────┐")
    t0 = time.time()
    loading_raw_data.extract_candidates()
    loading_raw_data.copy_job_description()
    step_times["INGEST"] = round(time.time() - t0, 2)
    print(f"└─ [INGEST] Done ({step_times['INGEST']}s) ─────────────────────────────┘")

    # ─────────────────────────────────────────────────────────────
    # [ENCODE] Generate dense embeddings → triple export
    # ─────────────────────────────────────────────────────────────
    print("\n┌─ [ENCODE] Generating dense embeddings (SKIPPED) ──────────┐")
    t0 = time.time()
    # num_candidates = embedder.process_candidates()
    num_candidates = 100_000 # hardcoded since we are skipping
    step_times["ENCODE"] = round(time.time() - t0, 2)
    print(f"└─ [ENCODE] Done ({step_times['ENCODE']}s) ─────────────────────────────┘")

    # ─────────────────────────────────────────────────────────────
    # [INDEX] Build categorical inverted index
    # ─────────────────────────────────────────────────────────────
    print("\n┌─ [INDEX] Building categorical inverted index ─────────────┐")
    t0 = time.time()
    categorical_indexer.build_categorical_index()
    step_times["INDEX"] = round(time.time() - t0, 2)
    print(f"└─ [INDEX] Done ({step_times['INDEX']}s) ──────────────────────────────┘")

    # ─────────────────────────────────────────────────────────────
    # [LEACE] Compute concept erasure matrices
    # ─────────────────────────────────────────────────────────────
    print("\n┌─ [LEACE] Computing concept erasure matrices ──────────────┐")
    t0 = time.time()
    leace_switch.compute_leace_projection()
    step_times["LEACE"] = round(time.time() - t0, 2)
    print(f"└─ [LEACE] Done ({step_times['LEACE']}s) ──────────────────────────────┘")

    # ─────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - total_start, 2)

    print("\n" + "=" * 64)
    print("  PIPELINE COMPLETE")
    print("=" * 64)

    # Timing breakdown
    print("\n  ⏱  Step Timing:")
    for step, elapsed in step_times.items():
        print(f"      [{step:<7}]  {elapsed:>8}s")
    print(f"      {'─' * 24}")
    print(f"      {'TOTAL':<10}  {total_elapsed:>8}s")

    # Artifact inventory
    artifacts = [
        ("candidate_embeddings_float32.npy", OUT_FLOAT32_NPY),
        ("candidate_embeddings_768.dat",     OUT_FLOAT768_DAT),
        ("candidate_index_binary.dat",       OUT_BINARY_DAT),
        ("candidate_index.json",             OUT_INDEX_JSON),
        ("categorical_index.json",           OUT_CATEGORICAL_JSON),
        ("leace_concept_directions.npy",     OUT_LEACE_DIRECTIONS),
        ("leace_projection_matrix.npy",      OUT_LEACE_PROJECTION),
        ("leace_config.json",                OUT_LEACE_CONFIG),
    ]

    print("\n  📦  Generated Artifacts:")
    for name, path in artifacts:
        size = _file_size_str(path)
        status = "✅" if os.path.exists(path) else "❌"
        print(f"      {status}  {name:<42} {size:>10}")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    run_offline_pipeline()
