# ==========================
# file: tbug/compare.py
# ==========================
"""
Visual comparison of tensors captured from multiple projects.

Highlights
----------
* 1-D and 2-D modes:  'diff' | 'abs' | 'sign' | 'ratio'
* Binary masks:
      diff_red=True   – red ≠ 0, black = 0
      zero_black=True – heat-map but exact zeros → black
* 3-D voxel scatter:  render_mode='3d'
* On-the-fly slice:   index="2, :, 1, ..."  or index=(2, slice(None), 1, ...)
* Adjustable canvas size:  canvas_size=(W, H) in inches
* Interactive read-out (cursor hover)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Cursor
from mpl_toolkits.mplot3d import Axes3D       # noqa: F401 – registers 3-D

# --------------------------------------------------------------------------- #
# New helper: apply an arbitrary view-transform to each tensor                #
# --------------------------------------------------------------------------- #
from collections.abc import Callable
from typing import Sequence

def _apply_view_function(
    tensors: list[np.ndarray],
    view_function: Callable[[np.ndarray], np.ndarray] | Sequence[Callable[[np.ndarray], np.ndarray]] | None,
) -> list[np.ndarray]:
    """
    Apply *view_function* to each tensor.

    Parameters
    ----------
    tensors
        List of tensors loaded from disk.
    view_function
        • None .......................... → tensors are returned unchanged.  
        • Single callable ............... → callable is applied to every tensor.  
        • Tuple / list of callables ..... → ``view_function[i]`` is applied to
                                            ``tensors[i]`` (lengths must match).

    Returns
    -------
    List[np.ndarray]
        The transformed tensors.

    Raises
    ------
    ValueError
        If ``len(view_function)`` does not match ``len(tensors)``.
    TypeError
        If *view_function* is neither *None*, a callable, nor a
        sequence of callables.
    """
    if view_function is None:
        return tensors

    if callable(view_function):
        return [view_function(t) for t in tensors]

    if isinstance(view_function, (list, tuple)):
        if len(view_function) != len(tensors):
            raise ValueError(
                "Tuple of view_functions must have the same length as the "
                f"number of tensors ({len(tensors)})."
            )
        transformed: list[np.ndarray] = []
        for fn, t in zip(view_function, tensors):
            if fn is None:
                transformed.append(t)
            elif callable(fn):
                transformed.append(fn(t))
            else:
                raise TypeError("Each item in view_function must be callable or None.")
        return transformed

    raise TypeError("view_function must be a callable, a tuple/list of callables, or None.")





# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_by_name(
    name: str,
    root: Path,
    projects: list[str] | None = None,
) -> Tuple[
    List[np.ndarray],          # tensors (after safe coercion)
    List[str],                 # project names
    List[np.dtype],            # original dtypes on disk
    List[Tuple[int, ...]],     # original shapes on disk
]:
    """
    Load *name*.npy from each project in *root*.

    Any non-numeric or structured dtype is safely coerced to float32 for
    arithmetic, **but** the pristine dtype/shape is preserved and returned.
    """
    tensors, pnames, odtypes, oshapes = [], [], [], []
    root = root.expanduser().resolve()

    for proj in sorted(root.iterdir()):
        if not proj.is_dir():
            continue
        if projects and proj.name not in projects:
            continue

        f = proj / f"{name}.npy"
        if not f.exists():
            continue

        arr = np.load(f, allow_pickle=False)
        odtypes.append(arr.dtype)
        oshapes.append(arr.shape)

        if (
            arr.dtype.kind == "V"
            or not (np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_)
        ):
            print(f"[tbug] Converting {f} from {arr.dtype} → float32")
            arr = arr.view(np.uint16).astype(np.float32)

        tensors.append(arr)
        pnames.append(proj.name)

    if not tensors:
        raise FileNotFoundError(f"No '{name}.npy' found under {root}")

    return tensors, pnames, odtypes, oshapes


# --------------------------------------------------------------------------- #
# Public helper: show_values  (1-D raw plots)                                 #
# --------------------------------------------------------------------------- #
def show_values(
    name: str,
    *,
    root: str = "tbug_captures",
    projects: list[str] | None = None,
    view_function: Callable[[np.ndarray], np.ndarray] | Sequence[Callable[[np.ndarray], np.ndarray]] | None = None,
    canvas_size: Tuple[float, float] = (8, 4),
    cmap: str = "tab10",
    alpha: float = 0.9,
    just_print: bool = False,
) -> None:
    """
    Plot the *raw values* of one or more tensors (1-D only).

    The optional *view_function* lets you transform each tensor before plotting,
    e.g. ``lambda x: x[0, ...]`` for slicing or any other custom operation.
    A single callable is applied to every tensor; a tuple/list allows one
    callable per project.
    """
    root_path = Path(root).expanduser().resolve()
    tensors, pnames, odtypes, oshapes = _load_by_name(name, root_path, projects)

    tensors = _apply_view_function(tensors, view_function)

    if just_print:
        for pname, tensor in zip(pnames, tensors):
            print(f"\nProject: {pname}")
            print(tensor)
        return

    fig, ax = plt.subplots(figsize=canvas_size)
    ax.set_title(
        f"Tensor '{name}' – orig shape: {oshapes[0]}  dtype: {odtypes[0]}"
        + ("  view_function applied" if view_function else "")
    )
    ax.set_ylabel("value")

    for i, (tensor, lbl) in enumerate(zip(tensors, pnames)):
        data = tensor.ravel() if tensor.ndim > 1 else tensor
        ax.plot(
            data,
            label=lbl if len(pnames) > 1 else None,
            alpha=alpha,
            color=plt.get_cmap(cmap)(i % 10),
        )

    if len(pnames) > 1:
        ax.legend()
    ax.grid(True, linewidth=.3)
    plt.show()


# --------------------------------------------------------------------------- #
# Enhanced show()                                                             #
# --------------------------------------------------------------------------- #
def show(
    name: str,
    *,
    root: str = "tbug_captures",
    cmap: str = "coolwarm",
    alpha: float = 0.8,
    mode: str = "diff",             # 'diff' | 'abs' | 'sign' | 'ratio'
    render_mode: str | None = None, # None | '3d'
    projects: list[str] | None = None,
    zero_black: bool = False,
    diff_red: bool = False,
    view_function: Callable[[np.ndarray], np.ndarray] | Sequence[Callable[[np.ndarray], np.ndarray]] | None = None,
    canvas_size: Tuple[float, float] = (6, 6),
    just_print: bool = False,
) -> None:
    """
    Visualise a tensor *or the difference between two tensors*.
    """
    root_path = Path(root).expanduser().resolve()
    tensors, pnames, odtypes, oshapes = _load_by_name(name, root_path, projects)

    tensors = _apply_view_function(tensors, view_function)

    if just_print:
        for pname, tensor in zip(pnames, tensors):
            print(f"\nProject: {pname}")
            print(tensor)
        return

    # Check shapes after view_function for comparison
    effective_shapes = [t.shape for t in tensors]
    if len(tensors) > 1 and any(s != effective_shapes[0] for s in effective_shapes):
        msg = "Tensors have differing shapes after view_function; cannot compare:\n"
        for p, os, es in zip(pnames, oshapes, effective_shapes):
            msg += f"• {p}: {os} → {es}\n"
        raise ValueError(msg)

    view_note = "  view_function applied" if view_function else ""

    # ------------------------------------------------------------------ #
    # 1-D branch                                                          #
    # ------------------------------------------------------------------ #
    if tensors[0].ndim == 1:
        if len(tensors) == 1:
            show_values(
                name,
                root=root,
                projects=projects,
                view_function=view_function,
                canvas_size=canvas_size,
                just_print=just_print,
            )
            return

        if len(tensors) != 2:
            raise ValueError("For 1-D diff you must supply exactly two projects.")

        ref, other = tensors
        diff = other - ref

        if mode == "abs":
            diff = np.abs(diff)
        elif mode == "sign":
            diff = np.sign(diff)
        elif mode == "ratio":
            denom = np.where(np.abs(ref) < 1e-6, np.nan, ref)
            diff = np.abs(diff) / np.abs(denom) * 100
        elif mode != "diff":
            raise ValueError(f"Unknown mode '{mode}' for 1-D diff.")

        shape_note = (
            f"shapes differ: {oshapes[0]} ≠ {oshapes[1]}" 
            if oshapes[0] != oshapes[1] 
            else f"shape: {oshapes[0]}"
        )
        dtype_note = (
            f"dtypes differ: {odtypes[0]} ≠ {odtypes[1]}" 
            if odtypes[0] != odtypes[1] 
            else f"dtype: {odtypes[0]}"
        )

        x = np.arange(diff.shape[0])
        fig, ax = plt.subplots(figsize=canvas_size)
        ax.set_title(
            f"1-D {mode} for '{name}' – orig {shape_note}  {dtype_note}"
            + view_note
        )
        ax.set_ylabel(mode)

        if diff_red:
            nz = diff != 0
            ax.scatter(x[nz], diff[nz], c="red", marker="s", alpha=alpha, label="≠0")
            ax.scatter(
                x[~nz], diff[~nz], c="black", marker=".", alpha=alpha * 0.6, label="0"
            )
            ax.legend()
        else:
            ax.plot(x, diff, alpha=alpha)

        ax.grid(True, linewidth=.3)
        plt.show()
        return

    # ------------------------------------------------------------------ #
    # 3-D branch (unchanged except for title tweak)                       #
    # ------------------------------------------------------------------ #
    if render_mode == "3d":
        if len(tensors) != 2:
            raise ValueError("render_mode='3d' requires exactly two tensors.")
        ref, other = tensors
        if ref.ndim < 3:
            raise ValueError("render_mode='3d' requires tensors with ≥3 dims.")

        diff = other - ref
        if diff.ndim > 3:
            lead = tuple(range(diff.ndim - 3))
            diff3d = (
                np.any(diff != 0, axis=lead) if diff_red else diff.mean(axis=lead)
            )
        else:
            diff3d = np.any(diff != 0, axis=0) if diff_red else diff

        if diff_red:
            coords = np.argwhere(diff3d)
            colours = "red"
            vals_for_cb = None
        else:
            coords = np.argwhere(~np.isnan(diff3d))
            vals_for_cb = diff3d[tuple(coords.T)]
            colours = vals_for_cb

        shape_note = (
            f"shapes differ: {oshapes[0]} ≠ {oshapes[1]}" 
            if oshapes[0] != oshapes[1] 
            else f"shape: {oshapes[0]}"
        )
        dtype_note = (
            f"dtypes differ: {odtypes[0]} ≠ {odtypes[1]}" 
            if odtypes[0] != odtypes[1] 
            else f"dtype: {odtypes[0]}"
        )

        fig = plt.figure(figsize=canvas_size)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect(diff3d.shape[::-1])

        sc = ax.scatter(
            coords[:, 2],
            coords[:, 1],
            coords[:, 0],
            c=colours,
            cmap=cmap,
            alpha=alpha,
            s=4,
            marker="s",
            linewidths=0,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"3-D {mode} for '{name}'\n"
            f"orig {shape_note}  {dtype_note}"
            + view_note
        )
        if vals_for_cb is not None:
            fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.05)
        plt.show()
        return

    # ------------------------------------------------------------------ #
    # 2-D branch                                                          #
    # ------------------------------------------------------------------ #
    if len(tensors) == 1:
        im_data = tensors[0]
        if im_data.ndim != 2:
            h, w = im_data.shape[-2:]
            im_data = im_data.reshape(-1, h, w).mean(0)
        title = (
            f"Tensor '{name}'\n"
            f"orig shape: {oshapes[0]}  dtype: {odtypes[0]}"
            + view_note
        )
    else:
        ref, other = tensors
        diff = other - ref
        diff_mask = ref != other
        diff_count = int(np.count_nonzero(diff_mask))

        if diff.ndim != 2:
            h, w = diff.shape[-2:]
            diff = diff.reshape(-1, h, w).mean(0)

        if mode == "diff":
            im_data = diff
        elif mode == "abs":
            im_data = np.abs(diff)
        elif mode == "sign":
            im_data = np.sign(diff)
        elif mode == "ratio":
            denom = np.where(np.abs(ref) < 1e-6, np.nan, ref)
            im_data = np.abs(diff) / np.abs(denom) * 100
        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        mn, mx = np.nanmin(im_data), np.nanmax(im_data)

        shape_note = (
            f"shapes differ: {oshapes[0]} ≠ {oshapes[1]}" 
            if oshapes[0] != oshapes[1] 
            else f"shape: {oshapes[0]}"
        )
        dtype_note = (
            f"dtypes differ: {odtypes[0]} ≠ {odtypes[1]}" 
            if odtypes[0] != odtypes[1] 
            else f"dtype: {odtypes[0]}"
        )
        title = (
            f"{mode} for '{name}'\n"
            f"orig {shape_note}  {dtype_note}"
            + view_note + "\n"
            f"min: {mn:.4g}, max: {mx:.4g}, diff-count: {diff_count:,}"
        )

    # ------------------------------------------------------------------ #
    # Render 2-D figure                                                  #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=canvas_size)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")
    ax.set_title(title)

    if diff_red:
        mask = (im_data != 0).astype(np.uint8)
        im = ax.imshow(
            mask,
            cmap=ListedColormap(["black", "red"]),
            alpha=alpha,
            interpolation="nearest",
        )
    elif zero_black:
        base_cmap = plt.get_cmap(cmap)
        colors = base_cmap(np.linspace(0, 1, 256))
        masked = np.ma.masked_where(im_data == 0, im_data)
        cmap_mod = ListedColormap(colors)
        cmap_mod.set_bad(color="black")
        im = ax.imshow(masked, cmap=cmap_mod, alpha=alpha, interpolation="nearest")
    else:
        im = ax.imshow(im_data, cmap=cmap, alpha=alpha, interpolation="nearest")

    # -------- interactive read-out ------------------------------------- #
    Cursor(ax, useblit=True, color="black", linewidth=1)
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    def _on_move(evt):
        if evt.inaxes is not ax or evt.xdata is None or evt.ydata is None:
            return
        x, y = int(evt.xdata + 0.5), int(evt.ydata + 0.5)
        arr = im.get_array()
        if 0 <= y < arr.shape[0] and 0 <= x < arr.shape[1]:
            val = arr[y, x]
            if isinstance(val, np.ma.core.MaskedConstant):
                val = 0.0
            annot.xy = (x, y)
            annot.set_text(f"({y}, {x}) : {val:.4g}")
            annot.set_visible(True)
        else:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)
    plt.colorbar(im, ax=ax)
    plt.show()


# --------------------------------------------------------------------------- #
# Convenience loader                                                          #
# --------------------------------------------------------------------------- #
def load(
    name: str,
    *,
    root: str = "tbug_captures",
    projects: list[str] | None = None,
    view_function: Callable[[np.ndarray], np.ndarray] | Sequence[Callable[[np.ndarray], np.ndarray]] | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    """
    Load a tensor (optionally transformed) from one or more projects.

    Returns a single tensor when only one project matches, otherwise a
    ``{project_name: tensor}`` dictionary.
    """
    root_path = Path(root).expanduser().resolve()
    tensors, pnames, _, _ = _load_by_name(name, root_path, projects)

    tensors = _apply_view_function(tensors, view_function)

    return tensors[0] if len(tensors) == 1 else dict(zip(pnames, tensors))
