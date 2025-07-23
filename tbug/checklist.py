"""Utilities to create a consistency checklist across projects.

The “dtype” comparison is now done with the *string stored in the JSON
side‑car* for each tensor capture – no NumPy dtype objects are involved.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Sequence
import json
import numpy as np

CHECK = "✅"
CROSS = "❌"
NA    = "N/A"

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

EQUIVALENT_DTYPES = [
    ("torch.bfloat16", "bfloat16", "np.bfloat16"),
    ("torch.float16",  "float16",  "np.float16"),
    ("torch.float32",  "float32",  "np.float32"),
    ("torch.float64",  "float64",  "np.float64"),
    ("torch.complex64", "complex64", "jax.complex64"),
]

# build a fast alias-to-canonical lookup
_ALIAS_TO_CANON = {alias: group[0] for group in EQUIVALENT_DTYPES for alias in group}

def _canon_dtype(dtype_str: str) -> str:
    """
    Map *dtype_str* to its canonical representative so that all strings in the
    same equivalence tuple compare equal.
    """
    return _ALIAS_TO_CANON.get(dtype_str, dtype_str)



def _collect_projects(root: Path) -> Dict[str, List[Path]]:
    """
    Return a mapping {tensor-stem: [<proj-A/file.npy>, <proj-B/file.npy>, …]}.
    """
    mapping: Dict[str, List[Path]] = {}
    for proj in root.iterdir():
        if not proj.is_dir():
            continue
        for npy in proj.glob("*.npy"):
            mapping.setdefault(npy.stem, []).append(npy)
    return mapping


def _load_meta(json_path: Path) -> Tuple[str, Tuple[int, ...]]:
    """
    Read the original *dtype* **string** and *shape* tuple from ``json_path``.

    For older captures without a JSON file we fall back to inspecting the `.npy`
    payload itself so that the tool still works.
    """
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta["original_dtype"], tuple(meta["shape"])

    # -------- legacy fallback (no JSON) ------------------------------------
    npy_path = json_path.with_suffix(".npy")     # convert “…/tensor.json” → “…/tensor.npy”
    arr      = np.load(npy_path, allow_pickle=False)
    return str(arr.dtype), arr.shape


# ---------------------------------------------------------------------------
# 2 )  Full drop-in replacement for _compare()
# ---------------------------------------------------------------------------
from typing import Sequence, Tuple, List
import numpy as np


def _compare(                            # pylint: disable=too-many-branches
    tensors: Sequence[np.ndarray],
    dtype_strs: Sequence[str],
    shapes: Sequence[Tuple[int, ...]],
) -> Tuple[bool, List[str]]:
    """
    Compare every tensor to the first one (reference).

    * dtype equivalence is resolved via `_canon_dtype` so any alias pair listed
      in `EQUIVALENT_DTYPES` counts as identical.
    * Returns (identical?, reasons) where `reasons` may include
      'dtype', 'shape', or 'numeric@[coord]'.
    """
    # ------------ single-project edge-case
    if len(tensors) < 2:
        return False, ["only one project"]

    # Canonicalise dtype strings *before* comparison
    canon_dtypes = [_canon_dtype(s) for s in dtype_strs]

    ref_dtype   = canon_dtypes[0]
    ref_shape   = shapes[0]
    ref_tensor  = tensors[0]

    reasons: List[str] = []

    # ------------ dtype / shape meta checks
    if any(dt != ref_dtype for dt in canon_dtypes[1:]):
        reasons.append("dtype")
    if any(sh != ref_shape for sh in shapes[1:]):
        reasons.append("shape")
        return False, reasons           # shape mismatch ⇒ numeric diff pointless

    # ------------ numeric byte-wise diff (only if meta passes)
    max_abs = -1.0
    max_loc: Tuple[int, ...] | None = None

    for t in tensors[1:]:
        if np.array_equal(ref_tensor, t):
            continue                    # this one matches exactly
        diff     = t.astype(np.float64) - ref_tensor.astype(np.float64)
        abs_diff = np.abs(diff)
        local_max = abs_diff.max()
        if local_max > max_abs:
            max_abs = float(local_max)
            max_loc = np.unravel_index(abs_diff.argmax(), diff.shape)

    if max_loc is not None:
        reasons.append(f"numeric@{[int(x) for x in max_loc]}")

    identical = len(reasons) == 0
    return identical, reasons



# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def make_checklist(
    root: str = "tbug_captures",
) -> Dict[str, Tuple[Union[bool, None], List[str]]]:
    """
    Build ``{tensor-name: (status, reasons)}``.

    status  = True   → identical across all projects  
            = False  → mismatch  
            = None   → captured in only one project
    """
    root_path = Path(root).expanduser().resolve()
    tensors_per_name = _collect_projects(root_path)

    results: Dict[str, Tuple[Union[bool, None], List[str]]] = {}
    for stem, npy_paths in tensors_per_name.items():
        tensors:     List[np.ndarray]       = []
        dtype_strs:  List[str]              = []
        shapes:      List[Tuple[int, ...]]  = []

        for npy in npy_paths:
            tensors.append(np.load(npy, allow_pickle=False))
            dt_str, shp = _load_meta(npy.with_suffix(".json"))
            dtype_strs.append(dt_str)
            shapes.append(shp)

        ok, reasons = _compare(tensors, dtype_strs, shapes)
        results[stem] = (None if reasons == ["only one project"] else ok, reasons)

    return results


def print_checklist(
    root: str = "tbug_captures",
    *,
    section_order: Union[str, List[str], None] = None,
) -> None:
    """
    Human‑friendly report: table + mismatch details (dtype/shape from JSON).
    If a tensor appears in only one project, the project name is shown in the
    reason column, e.g. ``only one project (projA)``.
    """
    # ---------------------------- gather
    results = make_checklist(root)
    if not results:
        print("No tensors found.")
        return

    root_path     = Path(root).expanduser().resolve()
    project_files = _collect_projects(root_path)

    # ---------------------------- optional ordering
    if isinstance(section_order, str):
        section_order = section_order.split()
    section_order = section_order or []
    sect_pos      = {sec: i for i, sec in enumerate(section_order)}
    default_pos   = len(section_order)

    def _sort_key(tname: str) -> Tuple[int, str]:
        if "." in tname:
            head = tname.split(".", 1)[0]
            return (sect_pos.get(head, default_pos), tname)
        return (default_pos, tname)

    # ---------------------------- table header
    col_w = max(len(n) for n in results) + 2
    print(f"{'Tensor Name'.ljust(col_w)}Status  Reason(s)")
    print("-" * (col_w + 18))

    detail_lines: List[str] = []

    for name in sorted(results, key=_sort_key):
        ok, reasons = results[name]

        if ok is True:
            icon, reason_text = CHECK, "—"
        elif ok is False:
            icon, reason_text = CROSS, ",".join(reasons)
        else:  # ok is None  → only one project
            proj_name = project_files[name][0].parent.name if project_files[name] else "unknown"
            icon, reason_text = NA, f"only one project ({proj_name})"

        print(f"{name.ljust(col_w)}{icon}     {reason_text}")

        # detailed mismatch section
        if ok is False:
            detail_lines.append(f"• {name}: {', '.join(reasons)}")
            if {"dtype", "shape"} & set(reasons):
                for npy in project_files[name]:
                    proj   = npy.parent.name
                    dt, sh = _load_meta(npy.with_suffix(".json"))
                    detail_lines.append(f"    {proj}: shape={sh}, dtype={dt}")

    # ---------------------------- verbose block
    if detail_lines:
        print("\nDetailed mismatch report:")
        for ln in detail_lines:
            print(ln)
