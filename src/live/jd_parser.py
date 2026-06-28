"""
jd_parser.py - Replaces GLiNER

Takes the raw JD text, feeds it to the local quantized LLM via onnxruntime-genai,
and extracts a JSON structure. 
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

SYSTEM_PROMPT = """Your task is to identify categorical exclusions and generate search queries for the role.
You MUST output ONLY valid JSON matching this schema:
{
  "hard_exclusions": {
    "hiring_companies": [],
    "target_companies": [],
    "strictly_excluded_companies": []
  },
  "excluded_title_categories": [],
  "hyde_resume": "",
  "sub_queries": {
    "skills": "",
    "role": "",
    "domain": ""
  }
}

Rules:
1. `hard_exclusions`: Extract 'hiring_companies', 'target_companies', and 'strictly_excluded_companies'. Only place a specific, single company name string in 'strictly_excluded_companies' (e.g. ["Infosys", "TCS"]).
2. `excluded_title_categories`: Which functional categories from the list below are clearly incompatible with this role? Only list categories where hiring someone from that function would be a categorical mismatch, not a skills gap.
   Categories: HR/People, Sales/BD, Marketing, Mechanical/Mfg, Finance/Accounting, Legal/Compliance, Engineering/Tech, Product/Design, Operations, Other
3. `hyde_resume`: Write a 150-word resume summary for an ideal candidate for this role. Use first-person achievement language as it would appear on a real resume. Include specific technologies, outcomes, and seniority signals. Do not include any exclusion language.
4. `sub_queries`: Extract sub-queries for 'skills', 'role', and 'domain'.
5. OUTPUT ONLY JSON. Do not include markdown code blocks or explanations."""

def extract_json_from_text(text: str) -> str:
    """Attempt to extract a JSON block from the model's text output."""
    # First, just try to parse the whole string
    try:
        json.loads(text)
        return text
    except:
        pass
    
    # Try to find a JSON block (with or without markdown)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    
    return "{}"

def parse_jd(jd_text: str) -> dict:
    """
    Given the JD text, run it through the LLM and return the parsed structured exclusions.
    """
    t0 = time.perf_counter()
    model, tokenizer = initialize_llm_session()
    
    prompt = f"<|system|>You are an expert technical recruiter analyzing a Job Description (JD).<|end|>\n<|user|>\nJob Description:\n\n{jd_text}\n\n{SYSTEM_PROMPT}<|end|>\n<|assistant|>"
    
    print(f"  [LLM] Running inference on JD...")
    
    input_tokens = tokenizer.encode(prompt)
    
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE)
    
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    
    while not generator.is_done():
        generator.generate_next_token()
        
    # Slice out the prompt tokens so we only decode the newly generated text
    new_tokens = generator.get_sequence(0)[len(input_tokens):]
    response_text = tokenizer.decode(new_tokens).strip()
    
    json_str = extract_json_from_text(response_text)
    
    try:
        parsed_data = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"  [LLM] Error: Failed to parse LLM output as JSON.")
        print(f"  [LLM] Raw output: {response_text}")
        parsed_data = {
            "hard_exclusions": {"hiring_companies": [], "target_companies": [], "strictly_excluded_companies": []}, 
            "excluded_title_categories": [],
            "hyde_resume": "",
            "sub_queries": {"skills": "", "role": "", "domain": ""}
        }
        
    t1 = time.perf_counter()
    print(f"  [LLM] Parsed JD in {(t1 - t0) * 1000:.1f}ms")
    
    return parsed_data

if __name__ == "__main__":
    test_jd = "Looking for a Senior AI Engineer. Must have PyTorch experience. We do not want anyone from Infosys, TCS, or Wipro. We also don't want generic IT support staff."
    res = parse_jd(test_jd)
    print(json.dumps(res, indent=2))
