import os
import sys
import json
import time
import math
import numpy as np

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)

from src.config import OUT_RRF_FEATURES_JSON, OUT_FRAUD_FLAGS_JSON, FINAL_TOP_K

class Stage2RRF:
    def __init__(self, rrf_features_path=None, fraud_flags_path=None, sentence_transformer_model=None):
        print("=" * 60)
        print("  STAGE 2 RRF — Initialization")
        print("=" * 60)
        t0 = time.time()
        
        rrf_features_path = rrf_features_path or OUT_RRF_FEATURES_JSON
        fraud_flags_path = fraud_flags_path or OUT_FRAUD_FLAGS_JSON
        
        # Load offline features
        with open(rrf_features_path, 'r', encoding='utf-8') as f:
            self.features = json.load(f)
            
        with open(fraud_flags_path, 'r', encoding='utf-8') as f:
            raw_flags = json.load(f)
            self.flags = {item["candidate_id"]: item for item in raw_flags}
            
        self.st_model = sentence_transformer_model
        
        print(f"  [LOAD] Loaded {len(self.features)} candidate profiles ({(time.time()-t0)*1000:.0f}ms)")
        
    def _safe_jaccard(self, set1, set2):
        if not set1 and not set2: return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def rank(self, candidate_ids, parsed_jd, top_k=None):
        """
        Takes candidates from Stage 1 and fuses them using 12-dimensional Weighted RRF.
        """
        top_k = top_k or FINAL_TOP_K
        t_start = time.time()
        
        # Extract dynamic targets from JD
        jd_req_min_exp = parsed_jd.get('required_min_experience_months', 0)
        jd_req_max_exp = parsed_jd.get('required_max_experience_months', 999)
        jd_req_skills = set([s.lower() for s in parsed_jd.get('required_skills', [])])
        jd_nice_skills = set([s.lower() for s in parsed_jd.get('nice_to_have_skills', [])])
        jd_leadership = parsed_jd.get('leadership_required', False)
        jd_category = parsed_jd.get('required_role_category', 'Other')
        jd_seniority = parsed_jd.get('target_seniority', 'senior')
        jd_work_mode = parsed_jd.get('preferred_work_mode', 'flexible').lower()
        jd_notice_max = parsed_jd.get('max_notice_period_days', 90)
        
        # We need the JD semantic vector for role-embedding match
        jd_role_query = f"{jd_seniority} {jd_category}"
        if self.st_model:
            jd_role_emb = self.st_model.encode([jd_role_query], convert_to_numpy=True)[0]
            jd_role_emb = jd_role_emb / max(np.linalg.norm(jd_role_emb), 1e-10)
        else:
            jd_role_emb = None
            
        # Precompute candidate embeddings in a batch to avoid O(N) model overhead
        surviving_cids = []
        valid_role_texts = []
        for stage1_rank, cid in enumerate(candidate_ids):
            flag_info = self.flags.get(cid, {})
            if flag_info.get("hard_fraud_flag", False) or flag_info.get("critical_overlap_flag", False) or flag_info.get("pre_grad_seniority_flag", False):
                continue
            feat = self.features.get(cid)
            if not feat:
                continue
            
            cand_skills = set(feat.get("skills_set", []))
            cand_cat = feat.get("dominant_role_category", "")
            cand_role_text = cand_cat + " " + " ".join([k for k in cand_skills][:3])
            
            surviving_cids.append((stage1_rank, cid, feat, flag_info, cand_skills))
            valid_role_texts.append(cand_role_text)
            
        cand_embs = None
        if self.st_model and valid_role_texts:
            print(f"  [RRF] Batch encoding {len(valid_role_texts)} candidate roles...")
            cand_embs = self.st_model.encode(valid_role_texts, batch_size=256, convert_to_numpy=True)
            norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
            cand_embs = cand_embs / np.maximum(norms, 1e-10)
            
        # Filter and score candidates
        surviving = []
        scores_matrix = [] # Stores raw scores for each dimension
        
        for idx, (stage1_rank, cid, feat, flag_info, cand_skills) in enumerate(surviving_cids):
            # --- Dimensional Raw Scoring ---
            # Dim 1: Stage 1 Rank (Higher is better for negative stage1_rank)
            raw_stage1 = -stage1_rank 
            
            # Dim 2: TF-IDF Adj
            raw_tfidf = 0.0 
            
            # Dim 3: Skills Match
            req_sim = self._safe_jaccard(cand_skills, jd_req_skills)
            nice_sim = self._safe_jaccard(cand_skills, jd_nice_skills)
            raw_skills = req_sim + (0.3 * nice_sim)
            
            # Dim 4: Experience Proximity
            cand_exp = feat.get("total_experience_months", 0)
            if jd_req_min_exp <= cand_exp <= jd_req_max_exp:
                raw_exp = 1.0
            elif cand_exp < jd_req_min_exp:
                delta = jd_req_min_exp - cand_exp
                raw_exp = math.exp(-((delta/12.0)**2))
            else:
                delta = cand_exp - jd_req_max_exp
                raw_exp = math.exp(-((delta/24.0)**2))
            
            # Dim 5: Role Category Alignment
            cand_cat = feat.get("dominant_role_category", "")
            consist = feat.get("career_consistency_score", 0.0)
            raw_category = consist if cand_cat == jd_category else min(consist, 0.1)
            
            # Dim 6: Role Embedding Match
            raw_role_embed = 0.0
            if jd_role_emb is not None and cand_embs is not None:
                cand_emb = cand_embs[idx]
                raw_role_embed = float(np.dot(jd_role_emb, cand_emb))
                
            # Dim 7, 8, 9, 10, 11
            raw_tenure = feat.get("avg_tenure_months", 0)
            raw_promo = feat.get("promotion_velocity", 0)
            raw_product = feat.get("product_vs_service_ratio", 0)
            raw_anomaly = -flag_info.get("anomaly_score", 0) # Higher means less anomalous (since anomaly score higher is worse)
            raw_availability = feat.get("redrob_availability", 0)
            
            # Dim 12: Leadership
            raw_leadership = 0.0
            if jd_leadership:
                raw_leadership = float(feat.get("leadership_roles_count", 0))
                
            # Bonus
            bonus = 0.0
            if feat.get("is_open_to_work"): bonus += 0.005
            if feat.get("is_verified"): bonus += 0.003
            if feat.get("has_certifications"): bonus += 0.002
            if feat.get("notice_period_days", 90) <= jd_notice_max: bonus += 0.005
            if feat.get("preferred_work_mode") == jd_work_mode: bonus += 0.004
            
            scores = [
                raw_stage1, raw_tfidf, raw_skills, raw_exp, raw_category, 
                raw_role_embed, raw_tenure, raw_promo, raw_product, 
                raw_anomaly, raw_availability, raw_leadership
            ]
            
            surviving.append((cid, bonus))
            scores_matrix.append(scores)
            
        if not surviving:
            return []
            
        # Convert to numpy for fast column-wise sorting (ranking)
        scores_matrix = np.array(scores_matrix)
        N, D = scores_matrix.shape
        ranks_matrix = np.zeros_like(scores_matrix)
        
        # For each dimension, calculate rank (1 to N, where 1 is the highest raw score)
        for d in range(D):
            # argsort sorts ascending. We want descending (highest raw score = rank 1).
            # So we argsort the negative of the scores.
            col = scores_matrix[:, d]
            order = np.argsort(-col)
            # Create ranks array
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, N + 1)
            ranks_matrix[:, d] = ranks
            
        # Apply RRF Weights
        weights = np.array([4.0, 2.0, 3.5, 3.0, 3.0, 2.5, 1.5, 1.0, 2.0, 1.0, 1.5, 1.5])
        # If leadership not required, zero out its weight
        if not jd_leadership:
            weights[11] = 0.0
            
        K = 60.0
        final_scores = np.zeros(N)
        
        for d in range(D):
            final_scores += weights[d] / (K + ranks_matrix[:, d])
            
        # Add bonuses
        for i in range(N):
            final_scores[i] += surviving[i][1]
            
        # Sort final results
        result_order = np.argsort(-final_scores)
        
        results = []
        for rank_idx, idx in enumerate(result_order[:top_k]):
            cid = surviving[idx][0]
            score = float(final_scores[idx])
            
            # Observability: find the dimensions where this candidate ranked in the top 10%
            c_ranks = ranks_matrix[idx, :]
            top_decile = N * 0.1
            
            dim_names = [
                "Stage 1 Semantic", "Lexical Fit", "Skills Match", "Experience Proximity", 
                "Role Category Match", "Role Embedding", "Tenure Stability", "Promotion Velocity",
                "Product/Service Ratio", "Trust/Anomaly", "Availability", "Leadership"
            ]
            
            strengths = []
            for d in range(D):
                if weights[d] > 0 and c_ranks[d] <= max(3, top_decile):
                    strengths.append(f"{dim_names[d]} (Rank {int(c_ranks[d])})")
                    
            rationale = ", ".join(strengths) if strengths else "Balanced performance across dimensions."
            
            results.append({
                "candidate_id": cid,
                "rrf_score": score,
                "ranking_rationale": rationale
            })
            
        print(f"  [RRF]     Fused {len(surviving)} candidates in {(time.time() - t_start)*1000:.1f}ms")
        return results
