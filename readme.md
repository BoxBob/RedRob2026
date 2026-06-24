# Redrob Intelligent Candidate Discovery & Ranking Challenge

This repository contains the code and resources for the Intelligent Candidate Discovery & Ranking Challenge. The goal is to identify and rank the top 100 best-fit candidate profiles for a given Job Description (JD) from a dataset of 100,000 candidates while satisfying all compute and system constraints.

---

## 📂 Repository Structure

```text
Redrob_AIML_Challenge/
├── PS/
│   └── India_runs_data_and_ai_challenge/  # Challenge materials (data, schemas, validation)
│       ├── README.docx                   # Participant bundle overview
│       ├── candidate_schema.json         # JSON schema describing candidates dataset
│       ├── candidates.jsonl              # 100,000-candidate pool dataset (or candidates.jsonl.gz)
│       ├── job_description.docx          # Target Job Description details
│       ├── redrob_signals_doc.docx       # Reference for the 23 behavioral/trap signals
│       ├── sample_candidates.json        # Reference schema subset (first 50 candidates)
│       ├── sample_submission.csv         # Submission format template
│       ├── submission_metadata_template.yaml # Metadata details to supply with submission
│       ├── submission_spec.docx          # Rules, constraints, and evaluation pipeline details
│       └── validate_submission.py        # Local submission validation script
├── env/                                  # Python 3.13 Virtual Environment (ignored)
├── .gitignore                            # Ignored paths (PS, env)
├── requirement.txt                       # Core dependencies list
└── readme.md                             # Project documentation (this file)
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
    *   [ ] **pyTLEX**: Pending setup and configuration.

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
