import os
import sys
import json
import time

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_root)

from src.config import RAW_JD, TOP_K, RAW_CANDIDATES
from src.live.stage1_recall import Stage1Retriever, read_docx_text
from src.live.honeypot_detection import RedrobHoneypotDetector

def main():
    jd_path = RAW_JD
    if os.path.exists(jd_path):
        jd_text = read_docx_text(jd_path)
    else:
        txt_path = os.path.join(workspace_root, "jd_text.txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

    print("Running Stage 1 Retriever...")
    t0 = time.time()
    retriever = Stage1Retriever(use_llm=True, use_leace=False)
    # We retrieve top 2000 to have enough non-fraud candidates
    top_2000_stage1, parsed_jd = retriever.search(jd_text, top_k=2000)
    print(f"Stage 1 retrieved {len(top_2000_stage1)} candidates in {time.time()-t0:.2f}s")
    
    print("Evaluating with Honeypot Detector...")
    detector = RedrobHoneypotDetector()
    fraud_features = detector.evaluate_candidates(top_2000_stage1)
    
    survivors = []
    for cid in top_2000_stage1:
        feats = fraud_features.get(cid, {})
        # Hard filters exactly as defined in Stage 2
        if feats.get("critical_overlap_flag", False):
            continue
        if feats.get("hard_fraud", False):
            continue
        if feats.get("is_temporal_fraud", False):
            continue
            
        survivors.append(cid)
        
    top_100 = survivors[:100]
    print(f"Filtered {len(top_2000_stage1)} -> {len(survivors)} survivors. Taking Top 100.")
    
    print("Fetching full profiles...")
    target_set = set(top_100)
    profiles = {}
    with open(RAW_CANDIDATES, "r", encoding="utf-8") as f:
        for line in f:
            cid = line.split('"', 4)[3]
            if cid in target_set:
                profiles[cid] = json.loads(line)
                if len(profiles) == len(target_set):
                    break
                    
    findings_dir = os.path.join(workspace_root, "findings")
    os.makedirs(findings_dir, exist_ok=True)
    out_path = os.path.join(findings_dir, "stage1_top_100_nonfraud.md")
    
    with open(out_path, "w", encoding="utf-8") as md:
        md.write("# Top 100 Non-Fraud Resumes from Stage 1\n\n")
        
        for rank, cid in enumerate(top_100, 1):
            profile_data = profiles.get(cid)
            if not profile_data:
                continue
                
            personal = profile_data.get("profile", {})
            title = personal.get("current_title", "N/A")
            company = personal.get("current_company", "N/A")
            exp = personal.get("years_of_experience", "N/A")
            industry = personal.get("current_industry", "N/A")
            summary = personal.get("summary", "")
            
            skills_list = profile_data.get("skills", [])
            skills_str = ", ".join([s.get("name") for s in skills_list if "name" in s])
            
            md.write(f"## Rank {rank} | {title} at {company}\n\n")
            md.write(f"**Candidate ID:** {cid}\n")
            md.write(f"**Experience:** {exp} years\n")
            md.write(f"**Industry:** {industry}\n\n")
            
            md.write("### Skills\n")
            md.write(f"{skills_str}\n\n")
            
            if summary:
                md.write("### Summary\n")
                md.write(f"{summary}\n\n")
                
            md.write("### Career History\n")
            for job in profile_data.get("career_history", []):
                j_title = job.get("title", "N/A")
                j_comp = job.get("company", "N/A")
                j_start = job.get("start_date", "N/A")
                j_end = job.get("end_date", "Present")
                j_desc = job.get("description", "")
                md.write(f"#### {j_title} @ {j_comp}\n")
                md.write(f"*{j_start} - {j_end}*\n\n")
                if j_desc:
                    md.write(f"{j_desc}\n\n")
                    
            md.write("### Education\n")
            for edu in profile_data.get("education_history", []):
                e_deg = edu.get("degree", "N/A")
                e_inst = edu.get("institution", "N/A")
                e_grad = edu.get("graduation_date", "N/A")
                md.write(f"- **{e_deg}** from {e_inst} ({e_grad})\n")
                
            md.write("\n---\n\n")
            
    print(f"Successfully exported to {out_path}")

if __name__ == "__main__":
    main()
