import os
import sys
import json

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_root)

from src.live.jd_parser import parse_jd

def main():
    jd_path = os.path.join(workspace_root, "jd_text.txt")
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print("Parsing JD...")
    parsed = parse_jd(jd_text)
    
    print("\n================ HYDE RESUME ================")
    print(parsed.get('hyde_resume', ''))
    
    print("\n================ FULL PARSED ================")
    print(json.dumps(parsed, indent=2))

if __name__ == "__main__":
    main()
