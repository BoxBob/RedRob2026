# 📊 Research & Search Findings

This document tracks the results of our candidate retrieval experiments and highlights the differences between our initial test implementations and our **True Architectural Intent** for the discovery pipeline.

---

## 🔍 Experiment v1.01: Simple Vector Binarization

In this initial run, we tested a baseline binarization approach on the candidate vectors.
*   **Model**: `all-MiniLM-L6-v2` (384 float32 dimensions)
*   **Method**: Basic thresholding to `int8` (0 or 1):
    $$\text{weight} > 0 \rightarrow 1 \quad \text{and} \quad \text{weight} \le 0 \rightarrow 0$$
*   **Metric**: Hamming distance (bitwise XOR) over the 100,000 candidate pool.

### 📈 Disk & Performance Metrics
*   💾 **Binarized Embedding File**: **36.8 MB** (extremely low footprint)
*   🗂️ **Candidate Index File**: **1.52 MB**
*   ⏱️ **Offline Embedding Creation**: **~1 hour**
*   🚀 **Stage 1 Recall Retrieval (Top 2000)**: **< 1 second**
    > Retrieval is **extremely fast** (sub-second). This speed suggests we can afford to use more computationally expensive matching or retrieval algorithms during Stage 1 if it improves recall accuracy.

### 📝 Experiment Conclusion

> [!CAUTION]
> After manually comparing these top results against the target Job Description:
> **These are exactly the profiles the JD told us to outrightly reject or rank low.** Looking at their job titles and summaries, these candidates have no relevant backend ML experience, yet they are ranked in the top results. 
> The semantic quality suffered due to naive threshold-based binarization on a standard bi-encoder (`all-MiniLM-L6-v2`) which was not designed to preserve similarity under binary projection.

---

### 💡 Takeaways

1.  🛠️ **Filter/Remove Skills During Stage 1 Recall**: Listing generic or unrelated skills might dilute the retrieval focus. We should consider filtering or excluding them from the recall embedding payload.
2.  📉 **Naive Binarization Issues**: Simple threshold binarization is not detailed enough to perform high-quality semantic recall on standard models.
3.  ⚖️ **Strict Constraint Importance**: We need to pay special attention to the hard JD constraints, including **work experience**, **job title**, and **years of experience** during the first pass.

---

### 🚀 Future Steps

*   **Reconsider Binarization**: We perhaps need to avoid binarization entirely, or switch to a binarization-optimized model (like `nomic-embed-text-v1.5`) that supports binarization natively without heavy information loss.
*   **Utilize Retrieval Speed**: Since retrieval currently takes less than a second, we have ample headroom to introduce more semantically rich models or hybrid scoring.

## 🔍 Experiment v1.02: Trying full float32 embeddings and vector cosine similarity

In this initial run, we tested a baseline binarization approach on the candidate vectors.
*   **Model**: `all-MiniLM-L6-v2` (384 float32 dimensions)

### 📈 Disk & Performance Metrics
*   💾 **Binarized Embedding File**: **36.8 MB** (extremely low footprint)
*   🗂️ **Candidate Index File**: **1.52 MB**
*   ⏱️ **Offline Embedding Creation**: **~2 hour**
*   🚀 **Stage 1 Recall Retrieval (Top 2000)**: **< 1 second**
    > Retrieval is still **extremely fast** (sub-second), with very little memory consumed thanks to memmap.

---



### 📝 Experiment Conclusion

> [!CAUTION]
>The quality of the retrieved resumes are still pretty terrible, even when the JD explicitly asks not to pick them. So we might need to venture into a new direction

---

### 💡 Takeaways

1.  🛠️ **Filter/Remove Skills During Stage 1 Recall**: The retrieval quality is very low, and stage 1 might not be recall at all.

2.  ⚖️ **Strict Constraint Importance**: So the issue that we might be having is that semantic seraching might not really be considering the not factor, if it sees infosys in the resume, it might think its similar to JD even tough the JD says dont have it. It struggles to understand what you don't want.

---

### 🚀 Future Steps

*   **Dynamic Querry passing from the JD**: Before the system can filter candidates, it needs to understand the JD's requirements and exclusions. 
*   **Hard Categorical Filtering**: We should implement a more robust way to handle hard constraints. 

---

## 🔍 Experiment v1.03: GLiNER + LEACE-Switch + Inverted Index Bitmask

Addressed the two core failures from v1.01/v1.02 by adding three new subsystems on top of the binary Hamming search:

1. **GLiNER (Zero-Shot NER)**: An ONNX bi-encoder model (`gliner_small-v2.1`) that extracts structured entities from the JD at runtime — excluded companies, required locations, required skills, and disqualified concepts.
2. **Inverted Index + Bitmask**: A precomputed categorical index over 11 fields (`current_company`, `current_title`, `current_industry`, `skills`, `location`, etc.) used to build a `uint32`-packed boolean mask. Candidates matching any `hard_exclusion` are permanently zeroed out before the Hamming scan even begins.
3. **LEACE-Switch (Geometric Repulsion)**: Per-concept rank-1 projection matrices that geometrically push the query vector *away* from undesirable concept directions (e.g., "Junior Developer", "Consulting") in 768-dim space before binarization.

*   **Embedding Model**: `BAAI/bge-small-en-v1.5` (384-dim → duplicated to 768-dim → binarized to 96 bytes)
*   **GLiNER Model**: `onnx-community/gliner_small-v2.1` (ONNX, CPU, 12 threads)
*   **GLiNER Labels**: Contrastive label design — "good" labels (`hiring company`, `preferred company`, `target job title`) act as semantic sinks to prevent false positives, while "bad" labels (`company the candidate must not have worked for`, `disqualified skill or unwanted experience`) capture the actual exclusions.
*   **LEACE Concepts**: 5 static concepts — `Junior Developer`, `Intern`, `Entry Level`, `Consulting`, `Staffing Agency`

### 📈 Disk & Performance Metrics

| Artifact | Size |
|---|---|
| 💾 Binary Candidate Matrix (`candidate_index_binary.dat`) | **9.2 MB** |
| 📊 Float32 Embeddings (`candidate_embeddings_float32.npy`) | **146.5 MB** |
| 📊 768-dim Embeddings (`candidate_embeddings_768.dat`) | **293.0 MB** |
| 🗂️ Candidate Index (`candidate_index.json`) | **1.5 MB** |
| 🗃️ Categorical Index (`categorical_index.json`) | **12.9 MB** |
| 🧠 LEACE Matrices (5 concepts) | **2.3 MB** |

**Offline Pipeline Timing** (100,000 candidates):
| Step | Time |
|---|---|
| [INGEST] Load raw data | 31.8s |
| [ENCODE] Embedding generation | ~2 hours (skipped on re-runs) |
| [INDEX] Categorical inverted index | 17.9s |
| [LEACE] Concept erasure matrices | 7.2s |
| **TOTAL** (without ENCODE) | **~57s** |

**Live Pipeline Timing** (after Numba warmup):
| Step | Time | Notes |
|---|---|---|
| [PARSE] GLiNER entity extraction | **5,612ms** | 67 JD lines × 8 labels each |
| [MASK] Bitmask construction | **7ms** | 26,372 candidates excluded |
| [ENCODE] BGE + LEACE + binarize | **307ms** | |
| [SCAN] Numba Hamming kernel | **2ms** | 73,628 eligible candidates |
| [RANK] argpartition top-2000 | **2ms** | |
| **TOTAL** | **~5,930ms** | |

**Memory Profile**:
- Binary matrix: **9.2 MB** (memory-mapped, zero-copy)
- Inverted index (pickle): **~7 MB** in RAM
- LEACE matrices: **~12 MB** in RAM (5 × 768×768 float32)
- GLiNER ONNX model: **~80 MB** in RAM
- BGE model: **~130 MB** in RAM

### 📝 Experiment Conclusion

> [!TIP]
> **Massive improvement over v1.01/v1.02.** The bitmask successfully identified and excluded **26,372 candidates** (26.4%) from companies like TCS, Infosys, Wipro, Accenture, Cognizant, and Capgemini — exactly as the JD instructed. Previously, these candidates were ranked in the top results because the bi-encoder saw "Infosys" in both the JD and the resume and assumed they were similar.

**What worked well:**
- ✅ Company exclusion via inverted index is **instant** (7ms) and **100% precise** — zero false negatives.
- ✅ The contrastive label design for GLiNER prevented the model from accidentally extracting the hiring company's own name.
- ✅ LEACE repulsion gently pushed consulting/junior profiles down without hard-killing them.
- ✅ The Numba SIMD kernel with bitmask short-circuiting runs in **2ms** for 100K candidates.

> [!WARNING]
> **What still failed:**
> - ❌ **Irrelevant titles in Top 10**: `Mechanical Engineer` (Paper Products industry) and `Customer Support` (Manufacturing) still appeared in the top results. These candidates have no AI/ML experience but happened to have a few overlapping skill keywords (like "Python" or "Reinforcement Learning") that inflated their Hamming similarity.
> - ❌ **GLiNER cannot reason**: GLiNER is a pattern-matching NER model. It can extract "TCS" from "No TCS candidates" because the text explicitly says it. But it **cannot deduce** that "Mechanical Engineering" is irrelevant for an AI role — that requires reasoning, which GLiNER fundamentally lacks.
> - ❌ **GLiNER label sensitivity**: The label `"job title the candidate must not hold"` caused GLiNER to extract `"Senior AI Engineer"` as a disqualified title — the exact role the JD was hiring for. This killed all the best candidates until we removed the label entirely. Zero-shot NER is extremely fragile to label phrasing.
> - ❌ **GLiNER speed**: At ~5.6 seconds per parse (67 chunks × 8 labels), GLiNER is the bottleneck of the entire pipeline. The actual search (mask + encode + scan + rank) takes only ~320ms.

---

### 💡 Takeaways

1. 🎯 **Hard categorical filtering is solved.** The inverted index + bitmask architecture works flawlessly for explicit exclusions. The problem is no longer "how do we filter" — it's "how do we extract what to filter."

2. 🧠 **GLiNER is the wrong tool for JD parsing.** It only works when the JD explicitly names companies to exclude. It cannot perform the implicit reasoning needed to say "this JD is for an AI Engineer, therefore Mechanical Engineers are irrelevant." We need a model that can *reason*, not just *extract*.

3. ⚖️ **The hard/soft split is architecturally correct.** Routing explicit exclusions to the bitmask (permanent kill) and implicit repulsions to LEACE (gentle push) is the right design. We just need a smarter upstream parser to feed both channels.

4. 🏗️ **LEACE needs to be dynamic.** The 5 precomputed concepts (`Junior Developer`, `Intern`, etc.) are too static. Every JD has different "soft" exclusions. We need to compute LEACE matrices on-the-fly for arbitrary concepts at runtime.

---

### 🚀 Next Step: Replace GLiNER with Local LLM (Planned)

We are replacing GLiNER with a **local quantized LLM** (`Phi-4-mini Q4_K_M`, ~2.5 GB) running via `llama-cpp-python` on CPU. The LLM will:

1. **Reason about implicit exclusions**: Given an AI Engineer JD, it will deduce that `Mechanical Engineering`, `Customer Support`, and `Sales` are irrelevant — something GLiNER could never do.
2. **Output structured JSON** with two keys:
   - `hard_exclusions` → routed to the inverted index bitmask (permanent kill)
   - `soft_repulsions` → routed to dynamic LEACE (geometric push)
3. **Compute LEACE on-the-fly**: For every novel concept in `soft_repulsions`, we encode it with BGE, build a rank-1 projection matrix (`M = I - v·vᵀ`), and apply it to the query vector before binarization. Cost: ~50ms per concept.

**Expected improvements:**
- 🧠 Implicit reasoning will eliminate Mechanical Engineers / Customer Support from the top results
- 🎯 Hard exclusions will remain precise (same bitmask, better upstream parser)
- ⏱️ Parse time may increase from ~6s to ~10-15s, but well within the 5-minute pipeline SLA

---

## 🔍 Experiment v1.04: LLM Parser + Static LEACE (2-Step Pivot)

To fix the vector destruction caused by dynamic LEACE in our initial tests, we implemented a 2-step pivot:
1. **Restricted LLM Output**: The LLM was restricted to selecting from a predefined list of 5 atomic concepts (`['academia', 'consulting', 'junior', 'management', 'non-technical']`) that were precomputed completely offline.
2. **Global Projection Matrix**: We fixed a mathematical bug where applying `M = I - v^T v` on the raw BGE-Small vector destroyed the embedding space (due to vector anisotropy). We applied the correct global projection matrix `P` which properly centered the subspace.

### 📈 Disk & Performance Metrics
*   🤖 **LLM Parsing Time**: **~142 seconds** (Running onnxruntime-genai on CPU for a long JD)
*   🛡️ **Bitmasking Time**: **13.6ms** (Excluded 22,607 candidates from banned companies Infosys, TCS, Wipro)
*   🚀 **Stage 1 Recall Retrieval (Top 2000)**: **~5ms** (Hamming scan remains incredibly fast)

### 📝 Experiment Conclusion

> [!CAUTION]
> Despite the mathematically sound application of LEACE and successful bitmasking of companies, the final candidate list was still highly irrelevant.
> 
> Because the synthetic dataset randomly generated AI skills for unrelated roles (e.g., HR Managers with 'Vector Search' or Sales Executives with 'RAG'), the BGE-Small vector search strongly retrieved them based on skill matching, completely ignoring their titles. 
> LEACE mathematically pushed away "academia" and "junior", but it fundamentally cannot prevent a "Mechanical Engineer" from being retrieved if their generated skills align with the prompt.

---

### 💡 Takeaways

1. ❌ **LEACE is the wrong tool for categorical parsing.** Job Descriptions contain categorical constraints ("We don't want HR managers"), but vector spaces are soft. If an HR manager has strong AI skills, the vector search will retrieve them regardless of how much we geometrically repel "non-technical" concepts.
2. 🛡️ **Bitmasking is the only reliable way to enforce JD constraints.** The bitmask successfully eliminated 22,000 candidates from banned companies instantly. We need to expand this mechanism.

---

### 🚀 Future Steps

*   **Pivot to LLM-Driven Boolean Bitmasking**: Abandon LEACE completely. Instruct the LLM to deduce *hard categorical exclusions* (Titles, Industries) based on the JD's requirements, and feed those directly into the bitmask to physically block irrelevant roles before vector ranking.

---

## 🔍 Experiment v1.05: Multi-Vector HyDE + Categorical Bitmasking

We completely replaced the LEACE geometric repulsion approach with a combination of offline Categorical Bitmasking and Multi-Vector (HyDE) semantic search.

1. **Title-Category Bitmasking**: We built an offline script to map candidate job titles into functional buckets (e.g., `HR/People`, `Sales/BD`, `Mechanical/Mfg`). The LLM reads the JD and explicitly excludes entire categories. These candidates are instantly dropped using a bitmask before vector search even happens.
2. **Multi-Vector HyDE Decomposition**: The LLM generates a 150-word ideal candidate resume (HyDE) and extracts three sub-queries (`skills`, `role`, `domain`).
3. **Multi-Vector Hamming Scan**: We run the fast binary Hamming search on all four queries separately, union the candidates, and score them by their single best rank across all queries.

### 📈 Disk & Performance Metrics
*   🤖 **LLM Parsing Time**: **~62 seconds** (Running onnxruntime-genai on CPU for a long JD)
*   🛡️ **Bitmasking Time**: **15.2ms**
*   🚀 **Stage 1 Multi-Vector Search (Top 10)**: **~216ms** (Executing 4 Hamming kernels over 100k candidates)

### 📝 Experiment Conclusion

> [!TIP]
> The top 10 results are now completely clean of the contamination we saw in previous runs. We successfully retrieved **AI Specialists**, **Search Engineers**, and **Machine Learning Engineers** who built actual recommendation systems or semantic search features.
>
> The Title-Category bitmask successfully eradicated the "HR Managers" and "Mechanical Engineers", allowing the multi-vector semantic scan (HyDE) to cleanly retrieve relevant profiles.

---

### 💡 Takeaways

1. 🎯 **Hard Categorical Exclusions via Bitmasking work flawlessly.** When we allow the LLM to output broad buckets (e.g., `HR/People`, `Mechanical/Mfg`) and apply them strictly via bitwise AND, we eliminate 100% of title-based contamination. 
2. 🚀 **Multi-Vector HyDE search increases relevant recall.** By searching across the ideal resume, domain, role, and skills separately and taking the best rank, we avoid vector dilution.

---

### 🏁 Final Architecture Decision
The v1.05 pipeline (LLM Parser -> Categorical + Company Bitmask -> Multi-Vector Hamming Search) is our **true architectural intent** for Stage 1. We will proceed with this implementation.

## 🔍 Experiment v1.06: Final Validation

We ran the live pipeline with the newly built category_index.pkl and multi-vector HyDE decomposition. 
The top 10 results are completely clean of the contamination we saw in previous runs. We successfully retrieved **AI Specialists**, **Search Engineers**, and **Machine Learning Engineers** who built actual recommendation systems or semantic search features.

The Title-Category bitmask successfully eradicated the non-technical roles, allowing the multi-vector semantic scan to cleanly retrieve relevant profiles.
(See 1.06.md for the full top 10 profiles).
