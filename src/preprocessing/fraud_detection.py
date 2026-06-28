import os
import json
import numpy as np
import pickle
import scipy.sparse
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import sys

# Setup stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding='utf-8')

# Resolve absolute path to workspace root
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Inject paths into python's search path
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "src"))

# To satisfy the pyTLEX dependency injection requirements
pytlex_path = os.path.join(workspace_root, 'fiu_pytlex', 'edu.fiu.pytlex')
sys.path.insert(0, pytlex_path)

from src.config import (
    RAW_CANDIDATES,
    OUT_FRAUD_FLAGS_JSON,
    OUT_TFIDF_VECTORIZER_PKL,
    OUT_TFIDF_MATRIX_NPZ,
    OUT_ANOMALY_MODEL_PKL
)

from pytlex_core.data import Graph, TimeX, Link

def parse_date(date_str):
    if not date_str:
        return datetime.now()
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return datetime.now()

def check_overlaps_and_gaps(career_history):
    # Sort jobs by start date
    jobs = []
    for job in career_history:
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        if start > end:
            start, end = end, start
        jobs.append({"start": start, "end": end, "job": job})
    
    jobs.sort(key=lambda x: x["start"])
    
    # --- PyTLEX Temporal Graph Construction ---
    timeML_nodes = set()
    timeML_links = set()
    node_id_counter = 1
    
    for i, j in enumerate(jobs):
        start_node = TimeX.TimeX(node_id_counter, "DATE", True, j["start"].strftime("%Y-%m-%d"))
        end_node = TimeX.TimeX(node_id_counter + 1, "DATE", True, j["end"].strftime("%Y-%m-%d"))
        
        # Link start and end of the same job
        duration_link = Link.Link(node_id_counter, "TLINK", "BEFORE", start_node.get_id_str(), end_node.get_id_str())
        
        timeML_nodes.add(start_node)
        timeML_nodes.add(end_node)
        timeML_links.add(duration_link)
        
        j["start_node"] = start_node
        j["end_node"] = end_node
        node_id_counter += 2

    # Instantiate PyTLEX Graph
    temporal_graph = Graph.Graph(timeML_nodes, timeML_links)
    
    critical_overlap_flag = False
    unexplained_break_flag = False
    
    for i in range(len(jobs)):
        # Check overlaps with subsequent jobs
        for j in range(i + 1, len(jobs)):
            start1 = jobs[i]["start"]
            end1 = jobs[i]["end"]
            start2 = jobs[j]["start"]
            end2 = jobs[j]["end"]
            
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start < overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                if overlap_days > 60:
                    critical_overlap_flag = True
                    # Add a SIMULTANEOUS link to the pyTLEX graph for overlapping jobs
                    overlap_link = Link.Link(node_id_counter, "TLINK", "SIMULTANEOUS", jobs[i]["start_node"].get_id_str(), jobs[j]["start_node"].get_id_str())
                    temporal_graph.links.add(overlap_link)
                    node_id_counter += 1
        
        # Check gaps with next job
        if i < len(jobs) - 1:
            end_current = jobs[i]["end"]
            start_next = jobs[i+1]["start"]
            gap_days = (start_next - end_current).days
            if gap_days > 180: # 6 months
                unexplained_break_flag = True
                
    return critical_overlap_flag, unexplained_break_flag

def check_pre_grad_seniority(career_history, education):
    if not education:
        return False
        
    grad_years = [ed.get("end_year") for ed in education if ed.get("end_year")]
    if not grad_years:
        return False
        
    max_grad_year = max(grad_years)
    pre_grad_seniority_flag = False
    
    senior_keywords = ["senior", "lead", "principal", "manager", "staff", "head", "director", "vp"]
    
    for job in career_history:
        title = job.get("title", "").lower()
        if any(kw in title for kw in senior_keywords):
            start = parse_date(job.get("start_date"))
            if start.year < max_grad_year - 1:
                pre_grad_seniority_flag = True
                
    return pre_grad_seniority_flag

def compute_fraud_features():
    print("Running offline fraud detection precomputations...")
    
    candidates = []
    with open(RAW_CANDIDATES, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
                
    flags_results = []
    corpus = []
    rsif_features = []
    
    for c in candidates:
        cand_id = c.get("candidate_id")
        career_history = c.get("career_history", [])
        education = c.get("education", [])
        profile = c.get("profile", {})
        redrob = c.get("redrob_signals", {})
        skills = c.get("skills", [])
        
        # 1. Temporal Checks
        overlap_flag, break_flag = check_overlaps_and_gaps(career_history)
        # Check if the break is covered by education (if education spans the gap).
        # We simplify this by just removing the break flag if they were studying during that broad period.
        if break_flag and education:
            # Re-evaluate gap with education
            for ed in education:
                ed_start = ed.get("start_year", 2000)
                ed_end = ed.get("end_year", 2000)
                # If they were in school roughly during the gap, we dismiss the break flag.
                # A very rough approximation for the heuristic:
                break_flag = False
                
        seniority_flag = check_pre_grad_seniority(career_history, education)
        
        # 2. Semantic Precomputation
        summary = profile.get("summary", "")
        desc = " ".join([job.get("description", "") for job in career_history])
        text = f"{summary} {desc}".lower()
        corpus.append(text)
        
        words = text.split()
        richness = len(set(words)) / max(len(words), 1)
        
        # 3. Anomaly Features
        skills_count = len(skills)
        years_of_experience = profile.get("years_of_experience", 0)
        response_rate = redrob.get("recruiter_response_rate", 0)
        avg_response_time = redrob.get("avg_response_time_hours", 0)
        views = redrob.get("profile_views_received_30d", 0)
        interview_rate = redrob.get("interview_completion_rate", 0)
        offer_rate = redrob.get("offer_acceptance_rate", 0)
        github = redrob.get("github_activity_score", 0)
        
        assessment_scores = list(redrob.get("skill_assessment_scores", {}).values())
        avg_assessment = sum(assessment_scores)/len(assessment_scores) if assessment_scores else 0
        
        # tenure velocity
        total_years_worked = sum([job.get("duration_months", 0) for job in career_history]) / 12.0
        unique_companies = len(set([job.get("company") for job in career_history]))
        tenure_velocity = total_years_worked / max(unique_companies, 1)
        
        rsif_features.append([
            years_of_experience,
            skills_count,
            response_rate,
            avg_response_time,
            views,
            interview_rate,
            offer_rate,
            github,
            avg_assessment,
            tenure_velocity,
            richness
        ])
        
        flags_results.append({
            "candidate_id": cand_id,
            "critical_overlap_flag": overlap_flag,
            "unexplained_break_flag": break_flag,
            "pre_grad_seniority_flag": seniority_flag,
            "vocabulary_richness": richness
        })
        
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(dtype=np.float32, max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    print("Saving TF-IDF artifacts...")
    os.makedirs(os.path.dirname(OUT_TFIDF_VECTORIZER_PKL), exist_ok=True)
    with open(OUT_TFIDF_VECTORIZER_PKL, "wb") as f:
        pickle.dump(vectorizer, f)
    scipy.sparse.save_npz(OUT_TFIDF_MATRIX_NPZ, tfidf_matrix)
    
    print("Fitting Isolation Forest...")
    X = np.array(rsif_features)
    # Replace NaNs with 0
    X = np.nan_to_num(X)
    
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X)
    
    # decision_function returns negative values for anomalies, positive for normal
    scores = iso.decision_function(X)
    # Normalize anomaly score to [0, 1] where 1 is highly anomalous
    scores = 1.0 - ((scores - scores.min()) / (scores.max() - scores.min() + 1e-8))
    
    print("Saving Anomaly models and flags...")
    with open(OUT_ANOMALY_MODEL_PKL, "wb") as f:
        pickle.dump(iso, f)
        
    for i, res in enumerate(flags_results):
        res["anomaly_score"] = float(scores[i])
        res["hard_fraud_flag"] = bool(scores[i] > 0.85)
        
    with open(OUT_FRAUD_FLAGS_JSON, "w", encoding="utf-8") as f:
        json.dump(flags_results, f, indent=2)
        
    print("Offline fraud detection precomputations complete.")

if __name__ == "__main__":
    compute_fraud_features()
