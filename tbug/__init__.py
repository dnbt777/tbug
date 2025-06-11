# ========================
# file: tdiff/__init__.py
# ========================
"""tdiff – minimal tensor diffing toolkit.

Functions
---------
capture(arr, name, project, root='tdiff_captures')
    Persist *arr* under *root*/*project*/*name*.npy in a framework-agnostic way
    (supports NumPy, PyTorch, TensorFlow, JAX).

Notes
-----
*  Data are kept in NumPy's .npy format which already records shape & dtype.
*  Multiple projects can be compared with `tdiff.checklist` and explored
   visually with `tdiff.compare`.
"""

from pathlib import Path
import sys
import numpy as np
import json

__all__ = ["capture"]

def _to_numpy(arr):
    # PyTorch
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            orig_dtype = str(arr.dtype)
            if arr.dtype == torch.bfloat16:
                arr = arr.to(dtype=torch.float32)
            return arr.detach().cpu().numpy(), orig_dtype
    except ModuleNotFoundError:
        pass

    # TensorFlow
    try:
        import tensorflow as tf
        if isinstance(arr, tf.Tensor):
            return arr.numpy(), str(arr.dtype)
    except ModuleNotFoundError:
        pass

    # NumPy or JAX or unknown
    try:
        arr_np = np.asarray(arr)
        orig_dtype = str(arr_np.dtype)
        if "bfloat16" in str(arr_np.dtype) or "bf16" in str(arr_np.dtype) or orig_dtype in ["bfloat16", "bf16"]:
            arr_np = arr_np.astype(np.float32)
        elif arr_np.dtype.kind == 'V':
            arr_np = arr_np.astype(np.float32)
            orig_dtype = "bfloat16_raw"
        return arr_np, orig_dtype
    except Exception as e:
        raise TypeError(f"Unsupported array type or content: {type(arr)}") from e

def capture(arr, name: str, project: str, root: str = "tdiff_captures") -> Path:
    """Save array to .npy and metadata (.json)."""
    np_arr, orig_dtype = _to_numpy(arr)
    proj_dir = Path(root).expanduser().resolve() / project
    proj_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = proj_dir / f"{name}.npy"
    meta_path = proj_dir / f"{name}.json"

    with tensor_path.open("wb") as f:
        np.save(f, np_arr, allow_pickle=False)

    with meta_path.open("w") as f:
        json.dump({"original_dtype": orig_dtype, "shape": list(np_arr.shape)}, f)

    return tensor_path

# Re-export submodules for convenience
from importlib import import_module as _imp

def __getattr__(item):
    if item in {"checklist", "compare"}:
        return _imp(f"tdiff.{item}")
    raise AttributeError(item)
