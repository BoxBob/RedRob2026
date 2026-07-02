import os
import sys
import json
import re

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_root)

from src.config import RAW_CANDIDATES

def main():
    artifact_path = r"C:\Users\kevin\.gemini\antigravity-ide\brain\3df7ca35-f210-4036-a574-b17fced29aca\top_100_evaluation.md"
    findings_dir = os.path.join(workspace_root, "findings")
    os.makedirs(findings_dir, exist_ok=True)
    
    print(f"Reading {artifact_path}...")
    if not os.path.exists(artifact_path):
        print("Error: Eval artifact not found.")
        return
        
    # Extract IDs and Ranks
    candidates_to_export = []
    with open(artifact_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"### Rank (\d+) \| ID: (CAND_\d+) \|", line)
            if match:
                rank = int(match.group(1))
                cid = match.group(2)
                candidates_to_export.append((rank, cid))
                
    if not candidates_to_export:
        print("No candidates found in artifact.")
        return
        
    print(f"Found {len(candidates_to_export)} candidates. Extracting full profiles...")
    
    # We'll map cid to its rank for fast lookup
    target_set = {cid: rank for rank, cid in candidates_to_export}
    
    # Fetch profiles
    extracted = {}
    with open(RAW_CANDIDATES, "r", encoding="utf-8") as f:
        for line in f:
            # Quick check to avoid slow json.loads
            cid = line.split('"', 4)[3]
            if cid in target_set:
                extracted[cid] = json.loads(line)
                if len(extracted) == len(target_set):
                    break
                    
    single_md_path = os.path.join(findings_dir, "top_100_full_resumes.md")
    
    with open(single_md_path, "w", encoding="utf-8") as md:
        md.write("# Top 100 Full Resumes\n\n")
        
        for rank, cid in candidates_to_export:
            profile_data = extracted.get(cid)
            if not profile_data:
                print(f"Warning: Data for {cid} not found.")
                continue
                
            personal = profile_data.get("profile", {})
            title = personal.get("current_title", "N/A")
            company = personal.get("current_company", "N/A")
            exp = personal.get("years_of_experience", "N/A")
            industry = personal.get("current_industry", "N/A")
            summary = personal.get("summary", "")
            
            # Format skills
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
            
    print(f"Successfully exported {len(candidates_to_export)} resumes to {single_md_path}")

if __name__ == "__main__":
    main()
