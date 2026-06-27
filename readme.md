# Redrob Intelligent Candidate Discovery & Ranking Challenge

This repository contains the code and resources for the Intelligent Candidate Discovery & Ranking Challenge. The goal is to identify and rank the top 100 best-fit candidate profiles for a given Job Description (JD) from a dataset of 100,000 candidates while satisfying all compute and system constraints.

---

## 📂 Repository Structure

```text
Redrob_AIML_Challenge/
├── PS/
│   └── India_runs_data_and_ai_challenge/  # Challenge materials (data, schemas, validation)
│       ├── README.docx
│       ├── candidate_schema.json
│       ├── candidates.jsonl               # 100,000-candidate pool (source)
│       ├── job_description.docx
│       ├── redrob_signals_doc.docx
│       ├── sample_candidates.json
│       ├── sample_submission.csv
│       ├── submission_metadata_template.yaml
│       ├── submission_spec.docx
│       └── validate_submission.py
│
├── data/                                  # Pipeline data (gitignored)
│   ├── raw/                               # Ingested copies of candidates + JD
│   │   ├── candidates.jsonl
│   │   └── job_description.docx
│   └── processed/                         # Offline-generated artifacts
│       ├── candidate_embeddings_float32.npy   # (146.5 MB) 384-dim float32
│       ├── candidate_embeddings_768.dat       # (293.0 MB) 768-dim float32
│       ├── candidate_index_binary.dat         # (9.2 MB) packed uint8 binary
│       ├── candidate_index.json               # (1.5 MB) candidate ID list
│       ├── categorical_index.json             # (12.9 MB) field → value → row IDs
│       ├── inverted_index.pkl                 # pickle mirror of categorical index
│       ├── leace_concept_directions.npy       # concept embedding matrix
│       ├── leace_projection_matrix.npy        # combined projection P
│       ├── leace_matrices.npz                 # per-concept LEACE matrices
│       └── leace_config.json                  # concept metadata
│
├── src/
│   ├── config.py                          # Central configuration (single source of truth)
│   ├── preprocessing/                     # Offline phase modules
│   │   ├── GLiNER_setup.py                # GLiNER ONNX session init (singleton)
│   │   ├── categorical_indexer.py         # Inverted index builder (11 fields)
│   │   ├── embedder.py                    # BGE embedding + binarization
│   │   ├── installing_pytlex.py           # pyTLEX dependency setup
│   │   ├── leace_switch.py                # LEACE concept erasure matrices
│   │   └── loading_raw_data.py            # Raw data ingestion
│   ├── live/                              # Live phase modules
│   │   ├── hamming_kernel.py              # Numba SIMD Hamming scan + bitmask
│   │   └── stage1_recall.py               # Full Stage 1 pipeline (parse → mask → encode → scan → rank)
│   └── test/
│       └── preview_profiles.py            # Candidate profile viewer
│
├── scripts/
│   ├── offline.py                         # Master offline pipeline (ingest → encode → index → LEACE)
│   └── evaluate_stage1.py                 # Stage 1 evaluation harness
│
├── findings/                              # Experiment logs & research notes
│   ├── findings.md                        # Master findings document (v1.01 → v1.03)
│   ├── v1.01.md                           # Experiment v1.01 detailed log
│   ├── v1.02.md                           # Experiment v1.02 detailed log
│   ├── v.1.03.md                          # Experiment v1.03 detailed log
│   └── Best_Profiles_found.md             # Top candidate tracking
│
├── env/                                   # Python virtual environment (gitignored)
├── .hf_cache/                             # HuggingFace model cache (gitignored)
├── .pycache/                              # Centralized __pycache__ (gitignored)
├── .gitignore
├── requirements.txt                       # Core dependencies
└── readme.md                              # Project documentation (this file)
```

---

## ⚙️ Environment Setup

Follow these commands to configure the Python 3.13 virtual environment and install the required modules.

### 1. Python 3.13 Virtual Environment Setup
To initialize a virtual environment named `env` using Python 3.13:
```powershell
python -m venv env
```

### 2. Activate Virtual Environment
Activate the environment in your shell (Windows):
```powershell
.\env\Scripts\activate
```

### 3. Install Dependencies
Install all required modules from `requirement.txt`:
```powershell
pip install -r requirement.txt
```

---

## 🛠️ Phase 1: Data Pre-processing & Environment Setup (Offline)

Processing 100K text embeddings live on a CPU during inference is not feasible. The primary goal of Phase 1 is to shift heavy computation to the offline phase to keep live runs fast and lightweight.

### 1. Tooling Setup
*   **Libraries**: Install `onnxruntime`, `optimum` (HuggingFace), `lightgbm`, `numpy`, and `pyTLEX`.
*   **Progress**:
    *   [x] **Python 3.13 Virtual Environment**: Created successfully under `env/`.
    *   [x] **Core Libraries**: `onnxruntime`, `optimum`, `lightgbm`, and `numpy` are installed.
    *   [x] **pyTLEX**: installed.

### 2. Embedding Generation
*   **Task**: Pass all 100,000 candidate profiles through a binarization-optimized bi-encoder (e.g., `nomic-embed-text-v1.5`).
*   **Progress**: [ ] *Planned (Offline)*

### 3. Dimensionality Reduction
*   **Task**: Cast these float32 embeddings to binary hashes or INT8 to optimize memory footprint and computation speed.
*   **Progress**: [ ] *Planned (Offline)*

### 4. Memory Mapping
*   **Task**: Save the resulting embeddings array to disk. Load this using `numpy.memmap` during the live run so you only page chunks into RAM on demand, safely staying under the strict **16GB RAM limit**.
*   **Progress**: [ ] *Planned (Offline)*

### 5. Knowledge Graph Construction
*   **Task**: Build a static dictionary mapping Tier-2/3 Indian institutions, regional companies, and local job titles to standardized categorical weights to prevent localized false negatives.
*   **Progress**: [ ] *Planned (Offline)*
