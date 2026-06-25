import os
import sys
import json

# Setup stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding='utf-8')

# Resolve absolute path to workspace root (three levels up from src/test/preview_profiles.py)
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import colorama
    colorama.init()
except ImportError:
    pass

# Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

def preview_profiles(candidate_ids):
    if not candidate_ids:
        print("No candidate IDs provided.")
        return

    raw_data_path = os.path.join(workspace_root, "data", "raw", "candidates.jsonl")
    if not os.path.exists(raw_data_path):
        print(f"{RED}Error: Raw candidates file not found at {raw_data_path}{RESET}")
        return

    # Convert search list to set for O(1) lookups
    search_ids = set(candidate_ids)
    found_profiles = {}

    print(f"Searching for {len(search_ids)} profiles in raw candidate pool...")
    
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            # High-performance filter: skip parsing JSON unless the candidate ID is in the line
            if any(cid in line for cid in search_ids):
                try:
                    cand = json.loads(line)
                    cid = cand.get('candidate_id')
                    if cid in search_ids:
                        found_profiles[cid] = cand
                        if len(found_profiles) == len(search_ids):
                            break  # All profiles found, stop scanning
                except Exception:
                    continue

    # Print profiles in the original input order
    for cid in candidate_ids:
        if cid not in found_profiles:
            print(f"\n{RED}{BOLD}⚠️ Candidate {cid} not found in dataset.{RESET}")
            continue

        cand = found_profiles[cid]
        profile = cand.get('profile', {})
        signals = cand.get('redrob_signals', {})
        
        name = profile.get('anonymized_name', 'Anonymized')
        headline = profile.get('headline', 'N/A')
        summary = profile.get('summary', 'No summary provided.')
        location = profile.get('location', 'N/A')
        country = profile.get('country', 'N/A')
        exp_years = profile.get('years_of_experience', 0)
        curr_company = profile.get('current_company', 'N/A')
        curr_industry = profile.get('current_industry', 'N/A')
        
        print("\n" + "="*80)
        print(f"{BLUE}{BOLD}👤 [{cid}] {name}{RESET}")
        print("="*80)
        print(f"{BOLD}Headline :{RESET} {CYAN}{headline}{RESET}")
        print(f"{BOLD}Location :{RESET} {location}, {country} ({exp_years} Years Experience)")
        print(f"{BOLD}Company  :{RESET} {curr_company} ({curr_industry})")
        
        print(f"\n{YELLOW}{BOLD}📝 SUMMARY{RESET}")
        print(f"  {summary}")
        
        # Skills
        print(f"\n{YELLOW}{BOLD}🛠️ SKILLS{RESET}")
        skills = cand.get('skills', [])
        if skills:
            skill_tags = []
            for s in skills:
                sname = s.get('name', '')
                prof = s.get('proficiency', '')
                skill_tags.append(f"{GREEN}[{sname} ({prof})]{RESET}")
            print("  " + "  ".join(skill_tags))
        else:
            print("  No skills listed.")
            
        # Career History
        print(f"\n{YELLOW}{BOLD}💼 EXPERIENCE{RESET}")
        history = cand.get('career_history', [])
        if history:
            for job in history:
                title = job.get('title', 'N/A')
                company = job.get('company', 'N/A')
                dur = job.get('duration_months', 0)
                desc = job.get('description', '')
                is_curr = " (Current)" if job.get('is_current') else ""
                print(f"  • {BOLD}{title}{RESET} at {CYAN}{company}{RESET} - {dur} months{is_curr}")
                if desc:
                    print(f"    {desc}")
        else:
            print("  No work history listed.")
            
        # Education
        print(f"\n{YELLOW}{BOLD}🎓 EDUCATION{RESET}")
        education = cand.get('education', [])
        if education:
            for edu in education:
                inst = edu.get('institution', 'N/A')
                deg = edu.get('degree', 'N/A')
                field = edu.get('field_of_study', 'N/A')
                tier = edu.get('tier', 'N/A')
                grade = edu.get('grade', 'N/A')
                print(f"  • {BOLD}{inst}{RESET} ({CYAN}{tier}{RESET})")
                print(f"    {deg} in {field} | Grade: {grade}")
        else:
            print("  No education listed.")
            
        # Signals
        print(f"\n{YELLOW}{BOLD}📊 REDROB SIGNALS{RESET}")
        expect_sal = signals.get('expected_salary_range_inr_lpa', {})
        sal_min = expect_sal.get('min', 'N/A')
        sal_max = expect_sal.get('max', 'N/A')
        notice = signals.get('notice_period_days', 'N/A')
        mode = signals.get('preferred_work_mode', 'N/A')
        completeness = signals.get('profile_completeness_score', 'N/A')
        git_score = signals.get('github_activity_score', 'N/A')
        
        print(f"  - {BOLD}Expected Salary Range   :{RESET} {sal_min} - {sal_max} LPA (INR)")
        print(f"  - {BOLD}Notice Period (Days)    :{RESET} {notice} days")
        print(f"  - {BOLD}Preferred Work Mode     :{RESET} {mode}")
        print(f"  - {BOLD}Profile Completeness    :{RESET} {completeness}%")
        print(f"  - {BOLD}GitHub Activity Score   :{RESET} {git_score}")
        print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Accept IDs passed via command line
        preview_profiles(sys.argv[1:])
    else:
        # Fallback to previewing first 3 candidates as a demo/test
        print("No candidate IDs provided. Running preview on sample candidates...")
        sample_ids = ["CAND_0000001", "CAND_0000002", "CAND_0000003"]
        preview_profiles(sample_ids)
