"""
jd_parser.py - Replaces GLiNER

Takes the raw JD text, feeds it to the local quantized LLM via onnxruntime-genai,
and extracts structured facts and generative search queries in a single LLM call to save time.
"""

import os
import sys
import json
import re
import time

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)

from src.preprocessing.setup_LLM import initialize_llm_session
import onnxruntime_genai as og
from src.config import LLM_TEMPERATURE, LLM_MAX_TOKENS

SYSTEM_PROMPT = """Your task is to extract criteria from the Job Description and generate search queries.
Do not output JSON. Instead, strictly use the exact headers below followed by your generated text.
Example Output:
[HARD_EXCLUSIONS]
TCS, Infosys, Wipro

[EXCLUDED_TITLE_CATEGORIES]
Marketing, Sales/BD, HR/People

[MIN_EXPERIENCE_MONTHS]
60

[MAX_EXPERIENCE_MONTHS]
108

[TARGET_SENIORITY]
senior

[LEADERSHIP_REQUIRED]
False

[ROLE_CATEGORY]
Engineering/Tech

[HYDE_RESUME]
I am a senior AI engineer...

[SKILLS_QUERY]
PyTorch, Python...

[ROLE_QUERY]
AI Engineer

---

[HARD_EXCLUSIONS]
(Extract ONLY the exact names of companies or types of consulting firms explicitly forbidden by the JD (e.g. TCS, Infosys, Wipro). Do NOT include job titles, roles, or phrases like "marketing manager" or "framework enthusiasts" here. Separate by commas. If none, write NONE)

[EXCLUDED_TITLE_CATEGORIES]
(Choose any categories from this list that are explicitly undesirable for this role: HR/People, Sales/BD, Marketing, Mechanical/Mfg, Finance/Accounting, Legal/Compliance, Product/Design, Operations, Other. Separate by commas. If none, write NONE)

[MIN_EXPERIENCE_MONTHS]
(Extract the minimum required experience in months. e.g. 5 years = 60. Write only the number.)

[MAX_EXPERIENCE_MONTHS]
(Extract the maximum experience in months if stated. e.g. 9 years = 108. If none stated, write 999. Write only the number.)

[TARGET_SENIORITY]
(Determine the seniority: intern, junior, engineer, senior, lead, manager, director. Write only the word.)

[LEADERSHIP_REQUIRED]
(True if the role is a people manager or team lead; False if Individual Contributor. Write only True or False.)

[ROLE_CATEGORY]
(Choose exactly one category that best fits this role: HR/People, Sales/BD, Marketing, Mechanical/Mfg, Finance/Accounting, Legal/Compliance, Engineering/Tech, Product/Design, Operations, Other)

[HYDE_RESUME]
(Write a 150-word resume summary for an ideal candidate for this role. Use first-person achievement language.)

[SKILLS_QUERY]
(Write a search query focusing on the key technical skills REQUIRED. Do NOT include skills or domains that the JD explicitly says they do NOT want or want to avoid (such as Computer Vision, Speech Recognition, or Robotics). For example, if the JD says "no CV or robotics", do not include those keywords. Write only the technical skills to search for.)

[ROLE_QUERY]
(Write a search query focusing on the title and domain.)"""

def parse_llm_output(text: str, jd_text_raw: str = "") -> dict:
    data = {
        "hard_exclusions": {"hiring_companies": [], "target_companies": [], "strictly_excluded_companies": []}, 
        "excluded_title_categories": [],
        "required_min_experience_months": 0,
        "required_max_experience_months": 999,
        "target_seniority": "senior",
        "leadership_required": False,
        "required_skills": [],
        "nice_to_have_skills": [],
        "required_role_category": "Engineering/Tech",
        "preferred_work_mode": "flexible",
        "max_notice_period_days": 90,
        "hyde_resume": "",
        "sub_queries": {
            "skills": "",
            "role": "",
            "domain": ""
        }
    }
    
    # Parse HARD_EXCLUSIONS
    exc_match = re.search(r'\[HARD_EXCLUSIONS\](.*?)(?:\[|$)', text, re.DOTALL | re.IGNORECASE)
    if exc_match:
        val = exc_match.group(1).strip()
        if val.upper() != "NONE" and val:
            companies = [c.strip() for c in val.split(',') if c.strip()]
            data["hard_exclusions"]["strictly_excluded_companies"] = companies
            
    # Parse EXCLUDED_TITLE_CATEGORIES
    exc_cat_match = re.search(r'\[EXCLUDED_TITLE_CATEGORIES\](.*?)(?:\[|$)', text, re.DOTALL | re.IGNORECASE)
    if exc_cat_match:
        val = exc_cat_match.group(1).strip()
        if val.upper() != "NONE" and val:
            categories = [c.strip() for c in val.split(',') if c.strip()]
            data["excluded_title_categories"] = categories
            
    # Parse MIN_EXPERIENCE_MONTHS
    min_exp_match = re.search(r'\[MIN_EXPERIENCE_MONTHS\](.*?)(?:\[|$)', text, re.DOTALL | re.IGNORECASE)
    if min_exp_match:
        val = min_exp_match.group(1).strip()
        nums = re.findall(r'\d+', val)
        if nums:
            data["required_min_experience_months"] = int(nums[0])
            if len(nums) > 1:
                # If LLM put a range like "60-108" in the MIN block
                data["required_max_experience_months"] = int(nums[1])
            
    # Parse MAX_EXPERIENCE_MONTHS
    max_exp_match = re.search(r'\[MAX_EXPERIENCE_MONTHS\](.*?)(?:\[|$)', text, re.DOTALL | re.IGNORECASE)
    if max_exp_match:
        val = max_exp_match.group(1).strip()
        nums = re.findall(r'\d+', val)
        if nums:
            data["required_max_experience_months"] = int(nums[0])
            
    # Fallback if LLM used old [EXPERIENCE_MONTHS] tag
    fallback_exp_match = re.search(r'\[EXPERIENCE_MONTHS\](.*?)(?:\[|$)', text, re.DOTALL | re.IGNORECASE)
    if fallback_exp_match and data["required_min_experience_months"] == 0:
        val = fallback_exp_match.group(1).strip()
        nums = re.findall(r'\d+', val)
        if nums:
            data["required_min_experience_months"] = int(nums[0])
            if len(nums) > 1:
                data["required_max_experience_months"] = int(nums[1])
                
    # Ultimate Fallback: Scrape the raw JD text if max is still 999 but min was found
    if data["required_max_experience_months"] == 999:
        # Look for "X-Y years" or "X to Y years" in the raw JD text
        raw_range_match = re.search(r'(\d+)\s*(?:-|to)\s*(\d+)\s*years?', jd_text_raw, re.IGNORECASE)
        if raw_range_match:
            min_y = int(raw_range_match.group(1))
            max_y = int(raw_range_match.group(2))
            
            # Verify if this matches the LLM's min finding, or if LLM failed min too
            if data["required_min_experience_months"] in (0, min_y * 12):
                data["required_min_experience_months"] = min_y * 12
                data["required_max_experience_months"] = max_y * 12
            
    # Parse TARGET_SENIORITY
    sen_match = re.search(r'\[TARGET_SENIORITY\](.*?)(?:\[|$)', text, re.DOTALL | re.IGNORECASE)
    if sen_match:
        val = sen_match.group(1).strip().lower()
        if val:
            data["target_seniority"] = val
            
    # Parse LEADERSHIP_REQUIRED
    lead_match = re.search(r'\[LEADERSHIP_REQUIRED\](.*?)(?:\[|$)', text, re.DOTALL | re.IGNORECASE)
    if lead_match:
        val = lead_match.group(1).strip().lower()
        if "true" in val:
            data["leadership_required"] = True
            
    # Parse ROLE_CATEGORY
    cat_match = re.search(r'\[ROLE_CATEGORY\](.*?)(?:\[|$)', text, re.DOTALL | re.IGNORECASE)
    if cat_match:
        val = cat_match.group(1).strip()
        if val:
            data["required_role_category"] = val
            
    # Use case-insensitive and loose regex to catch variations like [Skills Query]:
    parts = re.split(r'\[(HYDE_RESUME|SKILLS_QUERY|ROLE_QUERY|Skills Query|Role Query|Hyde Resume)\]:?', text, flags=re.IGNORECASE)
    
    hyde = ""
    skills = ""
    role = ""
    
    current_key = None
    for p in parts:
        p = p.strip()
        p_upper = p.upper().replace(" ", "_")
        if p_upper in ["HYDE_RESUME", "SKILLS_QUERY", "ROLE_QUERY"]:
            current_key = p_upper
        elif current_key:
            if current_key == "HYDE_RESUME":
                hyde += " " + p
            elif current_key == "SKILLS_QUERY":
                skills += " " + p
            elif current_key == "ROLE_QUERY":
                role += " " + p
                
    data["hyde_resume"] = hyde.strip()[:1000]
    data["sub_queries"]["skills"] = skills.strip()[:300]
    data["sub_queries"]["role"] = role.strip()[:300]
            
    return data

def generate_with_model(model, tokenizer, prompt, max_new_tokens, temperature):
    input_tokens = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    total_length = len(input_tokens) + max_new_tokens
    params.set_search_options(max_length=total_length, temperature=temperature, repetition_penalty=1.05)
    
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    
    while not generator.is_done():
        generator.generate_next_token()
        
    new_tokens = generator.get_sequence(0)[len(input_tokens):]
    return tokenizer.decode(new_tokens).strip()

def parse_jd(jd_text: str) -> dict:
    t0 = time.perf_counter()
    model, tokenizer = initialize_llm_session()
    
    print(f"  [LLM] Call 1: Extracting criteria and generating HyDE...")
    prompt = f"""<|system|>You are an expert technical recruiter analyzing a Job Description (JD).<|end|>
<|user|>
Job Description:

{jd_text}

{SYSTEM_PROMPT}<|end|>
<|assistant|>"""
    
    t_call_0 = time.perf_counter()
    # Increased token limit since we are doing both tasks at once
    response = generate_with_model(model, tokenizer, prompt, 700, LLM_TEMPERATURE)
    t_call_1 = time.perf_counter()
    
    print("\n" + "="*60)
    print(" LLM RAW GENERATED OUTPUT")
    print("="*60)
    print(response)
    print("="*60 + "\n")
    
    parsed_data = parse_llm_output(response, jd_text)
    
    print(f"  [LLM] JD Parsing finished in {(t_call_1 - t_call_0) * 1000:.1f}ms")
    
    t1 = time.perf_counter()
    print(f"  [LLM] Total JD parsing time: {(t1 - t0) * 1000:.1f}ms")
    
    return parsed_data

if __name__ == "__main__":
    test_jd = "Looking for a Senior AI Engineer. Must have PyTorch experience. We do not want anyone from Infosys, TCS, or Wipro. We also don't want generic IT support staff."
    res = parse_jd(test_jd)
    print("Final Return:")
    print(json.dumps(res, indent=2))
