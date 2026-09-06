"""
LLM ONNX Session — Isolated initialization for structured JD parsing.

Downloads the ONNX model from Hugging Face if not present, and initializes
the onnxruntime-genai model and tokenizer.
"""

import os
import sys

# Resolve workspace root
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(workspace_root, ".pycache")
os.environ["HF_HOME"] = os.path.join(workspace_root, ".hf_cache")
sys.path.insert(0, workspace_root)

from src.config import LLM_MODEL_REPO, LLM_MODEL_FOLDER
from huggingface_hub import snapshot_download
import onnxruntime_genai as og

# Module-level singletons
_model = None
_tokenizer = None

def initialize_llm_session():
    """
    Download (if needed) and initialize the ONNX model for CPU inference.

    Returns:
        tuple: (model, tokenizer)
    """
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    print(f"  [LLM] Setting up ONNX Model: {LLM_MODEL_REPO} / {LLM_MODEL_FOLDER}")
    
    # Download the specific ONNX folder from Hugging Face Hub
    model_path = snapshot_download(
        repo_id=LLM_MODEL_REPO,
        allow_patterns=[f"{LLM_MODEL_FOLDER}/*"],
    )
    
    actual_model_path = os.path.join(model_path, LLM_MODEL_FOLDER)
    print(f"  [LLM] Model downloaded/found at: {actual_model_path}")
    print(f"  [LLM] Initializing onnxruntime-genai session...")

    _model = og.Model(actual_model_path)
    _tokenizer = og.Tokenizer(_model)

    print(f"  [LLM] Session ready.")
    return _model, _tokenizer


if __name__ == "__main__":
    # Quick smoke test
    model, tokenizer = initialize_llm_session()
    
    prompt = "<|system|>You output JSON only.<|end|><|user|>Give me a mock user object with name and age.<|end|><|assistant|>"
    input_tokens = tokenizer.encode(prompt)
    
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=1024, temperature=0.0)
    
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    
    while not generator.is_done():
        generator.generate_next_token()
        
    print("\nTest JSON output:")
    new_tokens = generator.get_sequence(0)[len(input_tokens):]
    text = tokenizer.decode(new_tokens)
    print(text)
