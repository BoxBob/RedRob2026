import os
import sys
import json
import time

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)

from src.config import RAW_CANDIDATES, OUT_RRF_FEATURES_JSON

SERVICE_KEYWORDS = [
    "services", "consulting", "outsourcing", "staffing", "advisory",
    "bpo", "it services", "professional services"
]

SERVICE_COMPANIES = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "hexaware",
    "deloitte", "kpmg", "ey", "pwc", "mckinsey", "bcg", "bain",
    "ibm", "larsen & toubro", "l&t", "genpact", "williams lea",
    "sutherland", "concentrix", "conduent", "teleperformance",
    "exl", "wms", "syntel"
]

CATEGORY_RULES = {
    "HR/People":          ["hr", "human resources", "people", "talent", "recruiter", "hrbp"],
    "Sales/BD":           ["sales", "business development", "account executive", "bdm"],
    "Marketing":          ["marketing", "growth", "brand", "content", "seo", "campaign"],
    "Mechanical/Mfg":     ["mechanical", "manufacturing", "production", "quality", "tooling"],
    "Finance/Accounting": ["finance", "accounting", "cfo", "controller", "auditor"],
    "Legal/Compliance":   ["legal", "compliance", "counsel", "attorney", "paralegal"],
    "Engineering/Tech":   ["engineer", "developer", "architect", "data", "ml", "ai", 
                           "software", "backend", "frontend", "devops", "sre"],
    "Product/Design":     ["product", "ux", "ui", "designer", "researcher"],
    "Operations":         ["operations", "ops", "supply chain", "logistics", "procurement"],
}

SENIORITY_LEVELS = {
    "intern": 0, "trainee": 0, "associate": 1, "junior": 1,
    "engineer": 2, "developer": 2, "analyst": 2, "scientist": 2,
    "senior": 3, "lead": 4, "staff": 4, "principal": 5,
    "manager": 4, "director": 5, "vp": 6, "head": 6, "cto": 7, "ceo": 7
}

LEADERSHIP_KEYWORDS = ["manager", "lead", "director", "vp", "head", "principal", "staff"]

def assign_category(title: str) -> str:
    if not title: return "Other"
    title_lower = title.lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(kw in title_lower for kw in keywords):
            return category
    return "Other"

def get_seniority(title: str) -> int:
    if not title: return -1
    title_lower = title.lower()
    highest = -1
    for kw, lvl in SENIORITY_LEVELS.items():
        if kw in title_lower and lvl > highest:
            highest = lvl
    return highest

def clamp(val, min_val=0.0, max_val=1.0):
    return max(min_val, min(val, max_val))

def compute_rrf_features():
    print("Extracting offline RRF features...")
    t0 = time.time()
    
    rrf_features = {}
    
    with open(RAW_CANDIDATES, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip(): continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            if not cid: continue
            
            profile = cand.get("profile", {})
            career_history = cand.get("career_history", [])
            education = cand.get("education", [])
            skills = cand.get("skills", [])
            redrob = cand.get("redrob_signals", {})
            
            # --- Continuous Features ---
            total_experience_months = int(profile.get("years_of_experience", 0) * 12)
            
            total_duration = sum([j.get("duration_months", 0) for j in career_history])
            unique_companies = len(set(j.get("company") for j in career_history if j.get("company")))
            avg_tenure_months = total_duration / max(unique_companies, 1)
            
            # Promotion Velocity
            company_roles = {}
            for job in career_history:
                comp = job.get("company", "").strip().lower()
                if not comp: continue
                if comp not in company_roles:
                    company_roles[comp] = []
                company_roles[comp].append(job)
            
            promotions = 0
            for comp, roles in company_roles.items():
                if len(roles) > 1:
                    # In candidates.jsonl, jobs are usually newest first. Let's sort oldest first.
                    def get_start(r):
                        sd = r.get("start_date")
                        return sd if sd else ""
                    roles.sort(key=get_start)
                    
                    last_sen = get_seniority(roles[0].get("title", ""))
                    for r in roles[1:]:
                        sen = get_seniority(r.get("title", ""))
                        if sen > last_sen and sen != -1 and last_sen != -1:
                            promotions += 1
                        last_sen = sen
            promotion_velocity = promotions / max(profile.get("years_of_experience", 1), 1)
            
            # Product vs Service
            service_months = 0
            for job in career_history:
                ind = job.get("industry", "").lower()
                comp = job.get("company", "").lower()
                is_service = False
                if any(k in ind for k in SERVICE_KEYWORDS): is_service = True
                if any(k == comp or k in comp for k in SERVICE_COMPANIES): is_service = True
                if is_service:
                    service_months += job.get("duration_months", 0)
            product_vs_service_ratio = (total_duration - service_months) / max(total_duration, 1)
            
            # Skills Depth Score
            skills_depth_score = 0
            for s in skills:
                prof = s.get("proficiency", "beginner")
                w = 3 if prof == "advanced" else 2 if prof == "intermediate" else 1
                dur = s.get("duration_months", 0)
                end = s.get("endorsements", 0)
                import math
                skills_depth_score += w * dur * math.log(1 + end)
                
            # Career Consistency Score
            cats = [assign_category(j.get("title", "")) for j in career_history]
            if cats:
                from collections import Counter
                most_common = Counter(cats).most_common(1)[0]
                dominant_role_category = most_common[0]
                career_consistency_score = most_common[1] / len(cats)
            else:
                dominant_role_category = "Other"
                career_consistency_score = 0.0
                
            # Redrob Trust Score & Availability
            prof_comp = redrob.get("profile_completeness_score", 0) / 100.0
            ver_email = 1.0 if redrob.get("verified_email") else 0.0
            ver_phone = 1.0 if redrob.get("verified_phone") else 0.0
            lin_conn = 1.0 if redrob.get("linkedin_connected") else 0.0
            gh_act = clamp(redrob.get("github_activity_score", 0) / 10.0)
            int_rate = clamp(redrob.get("interview_completion_rate", 0))
            
            redrob_trust_score = (0.3 * prof_comp + 0.15 * ver_email + 0.15 * ver_phone + 
                                  0.1 * lin_conn + 0.15 * gh_act + 0.15 * int_rate)
                                  
            otw = 1.0 if redrob.get("open_to_work_flag") else 0.0
            resp_rate = clamp(redrob.get("recruiter_response_rate", 0))
            resp_time = clamp(redrob.get("avg_response_time_hours", 0) / 336.0) 
            
            last_active = redrob.get("last_active_date", "2020-01-01")
            try:
                import datetime
                act_date = datetime.datetime.strptime(last_active, "%Y-%m-%d")
                curr_date = datetime.datetime(2026, 7, 2)
                days_since = (curr_date - act_date).days
                recency_score = max(0.0, 1.0 - (days_since / 180.0))
            except:
                recency_score = 0.0
                
            redrob_availability = 0.25 * otw + 0.25 * resp_rate + 0.25 * (1.0 - resp_time) + 0.25 * recency_score
            
            # --- Binary & Categorical ---
            leadership_roles = 0
            for job in career_history:
                t = job.get("title", "").lower()
                if any(kw in t for kw in LEADERSHIP_KEYWORDS):
                    leadership_roles += 1
                    
            has_certifications = len(cand.get("certifications", [])) > 0
            
            education_tier = "tier_4"
            education_field = "Unknown"
            if education:
                education_tier = education[0].get("tier", "tier_4")
                education_field = education[0].get("field_of_study", "Unknown")
            tier_map = {"tier_1": 3, "tier_2": 2, "tier_3": 1, "tier_4": 0}
            education_tier_score = tier_map.get(education_tier, 0)
            
            skills_set = list(set([s.get("name", "").lower() for s in skills]))
            
            # Assemble feature payload
            feat = {
                "total_experience_months": total_experience_months,
                "avg_tenure_months": avg_tenure_months,
                "promotion_velocity": promotion_velocity,
                "product_vs_service_ratio": product_vs_service_ratio,
                "skills_depth_score": skills_depth_score,
                "career_consistency_score": career_consistency_score,
                "redrob_trust_score": redrob_trust_score,
                "redrob_availability": redrob_availability,
                "leadership_roles_count": leadership_roles,
                "has_leadership_experience": leadership_roles > 0,
                "has_certifications": has_certifications,
                "is_open_to_work": otw > 0,
                "is_verified": (ver_email + ver_phone) == 2.0,
                "dominant_role_category": dominant_role_category,
                "education_tier": education_tier,
                "education_tier_score": education_tier_score,
                "skills_set": skills_set,
                "current_industry": profile.get("current_industry", ""),
                "preferred_work_mode": redrob.get("preferred_work_mode", ""),
                "notice_period_days": redrob.get("notice_period_days", 90)
            }
            
            rrf_features[cid] = feat
            
    # Save
    with open(OUT_RRF_FEATURES_JSON, "w", encoding="utf-8") as f:
        json.dump(rrf_features, f)
        
    print(f"Extracted features for {len(rrf_features)} candidates in {(time.time() - t0):.1f}s")
    print(f"Saved to {OUT_RRF_FEATURES_JSON}")

if __name__ == "__main__":
    compute_rrf_features()
