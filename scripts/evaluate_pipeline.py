import os
import sys
import json
import time

sys.stdout.reconfigure(encoding='utf-8')

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_root)

from src.config import RAW_JD, TOP_K, FINAL_TOP_K, RAW_CANDIDATES
from src.live.stage1_recall import Stage1Retriever, read_docx_text
from src.live.stage2_rrf import Stage2RRF

def get_candidate_profile(cid):
    with open(RAW_CANDIDATES, "r", encoding="utf-8") as f:
        for line in f:
            if cid in line:
                data = json.loads(line)
                if data["candidate_id"] == cid:
                    return data
    return None

def main():
    print("="*60)
    print(" END-TO-END PIPELINE EVALUATION (STAGE 1 -> STAGE 2 RRF) ")
    print("="*60)
    
    jd_path = RAW_JD
    jd_text = ""
    if os.path.exists(jd_path):
        jd_text = read_docx_text(jd_path)
    else:
        txt_path = os.path.join(workspace_root, "jd_text.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                jd_text = f.read()

    print(f"\n[INFO] Loaded Job Description ({len(jd_text)} characters).")
    
    # STAGE 1 
    # ---------------------------------------------------------
    t0 = time.time()
    retriever = Stage1Retriever(use_llm=True, use_leace=False)
    
    print(f"\n⏳ Running Stage 1 Retriever (Target {TOP_K} candidates)...")
    top_stage1_ids, parsed_jd = retriever.search(jd_text, top_k=TOP_K)
    print(f"✅ Stage 1 retrieved {len(top_stage1_ids)} candidates in {time.time()-t0:.2f}s")
    
    # STAGE 2
    # ---------------------------------------------------------
    t1 = time.time()
    # Pass the SentenceTransformer model from Stage 1 into Stage 2 so we don't load it twice
    ranker = Stage2RRF(sentence_transformer_model=retriever.model)
    
    print("\n⏳ Running Stage 2 Weighted RRF...")
    final_results = ranker.rank(top_stage1_ids, parsed_jd, top_k=FINAL_TOP_K)
    print(f"✅ Stage 2 ranked candidates in {time.time()-t1:.2f}s")
    
    # OUTPUT ARTIFACT
    # ---------------------------------------------------------
    # Get current artifact path based on conversation ID or use a standard path
    app_data_dir = os.environ.get("APPDATA", "") or os.path.expanduser("~")
    # For now, write to findings directory instead of hardcoded brain path
    artifact_path = os.path.join(workspace_root, "findings", "top_100_rrf_evaluation.md")
    
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write("# Top 100 Candidates Evaluation (RRF)\n\n")
        f.write(f"**JD Requirements Parsed**:\n")
        
        min_exp = parsed_jd.get('required_min_experience_months', 0) // 12
        max_exp = parsed_jd.get('required_max_experience_months', 999) // 12
        max_str = f"{max_exp}" if max_exp < 50 else "No Limit"
        f.write(f"- Experience: {min_exp} to {max_str} years\n")
        f.write(f"- Category: {parsed_jd.get('required_role_category')}\n")
        f.write(f"- Leadership: {parsed_jd.get('leadership_required')}\n\n")
        
        f.write("### Extracted Search Criteria & HyDE\n")
        f.write(f"**Excluded Companies:** {', '.join(parsed_jd.get('excluded_companies', [])) or 'None'}\n\n")
        
        sq = parsed_jd.get('sub_queries', {})
        f.write(f"**Skills Query:** {sq.get('skills', 'N/A')}\n\n")
        f.write(f"**Role Query:** {sq.get('role', 'N/A')}\n\n")
        
        hyde = parsed_jd.get('hyde_resume', 'N/A')
        f.write(f"**HyDE Resume:**\n> {hyde}\n\n")
        f.write("---\n\n")
        
        for rank, res in enumerate(final_results, 1):
            cid = res["candidate_id"]
            score = res["rrf_score"]
            rationale = res["ranking_rationale"]
            
            profile = get_candidate_profile(cid)
            if profile:
                personal = profile.get("profile", {})
                title = personal.get("current_title", "N/A")
                company = personal.get("current_company", "N/A")
                exp = personal.get("years_of_experience", "N/A")
                summary = personal.get("summary", "No summary provided.")
                # Basic string formatting to avoid breaking markdown if there are newlines
                summary = summary.replace('\n', ' ').strip()
                if len(summary) > 300: summary = summary[:297] + "..."
                
                skills_list = profile.get("skills", [])
                skill_names = [s.get("name") for s in skills_list if "name" in s][:8]
                skills_str = ', '.join(skill_names) if skill_names else "N/A"
                
                f.write(f"### Rank {rank} | ID: {cid} | Score: {score:.4f}\n")
                f.write(f"> **{title} at {company} ({exp} years exp)**  \n")
                f.write(f"> **Skills**: {skills_str}  \n")
                f.write(f"> **Summary**: {summary}  \n")
                f.write(f"> **Why highly ranked?** {rationale} \n\n")
            else:
                f.write(f"### Rank {rank} | ID: {cid} | Score: {score:.4f}\n")
                f.write("> Profile not found.\n\n")
                
        # ---------------------------------------------------------
        # FINAL TIMING SUMMARY
        # ---------------------------------------------------------
        t_end = time.time()
        total_time = t_end - t0
        stage2_time = t_end - t1
        stage1_time = total_time - stage2_time

        f.write("\n---\n")
        f.write("### ⏱️ END-TO-END PIPELINE TIMING SUMMARY\n")
        f.write(f"- **Stage 1 (Parsing & Search)**: {stage1_time:.2f}s\n")
        f.write(f"- **Stage 2 (RRF Scoring)**: {stage2_time:.2f}s\n")
        f.write(f"- **Total Pipeline Execution Time**: {total_time:.2f}s\n")

    print(f"✅ Saved Top {FINAL_TOP_K} evaluation to {artifact_path}")
    
    print("\n" + "="*60)
    print(" ⏱️  END-TO-END PIPELINE TIMING SUMMARY ")
    print("="*60)
    print(f"  Stage 1 (Parsing & Search):   {stage1_time:.2f}s")
    print(f"  Stage 2 (RRF Scoring):        {stage2_time:.2f}s")
    print("-" * 60)
    print(f"  Total Pipeline Execution Time: {total_time:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
