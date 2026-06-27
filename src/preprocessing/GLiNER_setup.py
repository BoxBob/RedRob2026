"""
GLiNER ONNX Session — Isolated initialization for zero-shot NER.

Manages the GLiNER ONNX runtime session with optimized thread limits
for CPU-only inference. The session is lazily initialized and cached
as a module-level singleton.
"""

import os
import sys

# Resolve workspace root
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(workspace_root, ".pycache")
os.environ["HF_HOME"] = os.path.join(workspace_root, ".hf_cache")
sys.path.insert(0, workspace_root)

from src.config import GLINER_MODEL

# Module-level singleton
_gliner_instance = None


def initialize_gliner_session(model_name=None):
    """
    Load and return a GLiNER ONNX model with optimized CPU thread settings.

    The model is cached as a module-level singleton — subsequent calls
    return the same instance without reloading.

    Args:
        model_name: HuggingFace model name/path. Defaults to config.GLINER_MODEL.

    Returns:
        A GLiNER instance ready for predict_entities().
    """
    global _gliner_instance

    if _gliner_instance is not None:
        return _gliner_instance

    model_name = model_name or GLINER_MODEL

    print(f"  [GLiNER] Initializing ONNX session ({model_name})...")

    # Configure ONNX Runtime session options for CPU optimization
    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = os.cpu_count()
    sess_opts.inter_op_num_threads = 1
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Set environment variable before GLiNER import to influence ONNX session
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

    from gliner import GLiNER

    _gliner_instance = GLiNER.from_pretrained(
        model_name,
        load_onnx_model=True,
        load_tokenizer=True,
        onnx_model_file="onnx/model.onnx"
    )

    print(f"  [GLiNER] Session ready (threads: {os.cpu_count()})")
    return _gliner_instance


if __name__ == "__main__":
    # Quick smoke test
    model = initialize_gliner_session()
    test_text = "We need a Senior Python Engineer, not from consulting firms like Accenture."
    labels = ["required skill", "excluded company", "unwanted trait"]
    entities = model.predict_entities(test_text, labels)
    print(f"\nTest entities from: '{test_text}'")
    for e in entities:
        print(f"  [{e['label']}] '{e['text']}' (score: {e['score']:.3f})")
