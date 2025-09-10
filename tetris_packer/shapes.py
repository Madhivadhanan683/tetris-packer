# tetris_packer/shapes.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import json
import os

# Public API
__all__ = [
    "Piece",
    "normalize",
    "generate_variants",
    "load_shapes_from_json",
    "expand_shapes_from_input",
]

@dataclass
class Piece:
    """
    Represents a single copy of a block (one physical tile to place).
    - id: unique id for this copy (useful to mark grid cells)
    - shape_idx: index of the base shape (for grouping / legend)
    - name: optional human readable name
    - base: normalized base numpy array for the shape
    - variants: list of numpy arrays (unique rotations/reflections) usable for placement
    - area: number of filled cells
    """
    id: int
    shape_idx: int
    name: str
    base: np.ndarray
    variants: List[np.ndarray]
    area: int

def _to_numpy_matrix(matrix_like: Any) -> np.ndarray:
    """
    Convert an input list-of-lists (or numpy) to a normalized uint8 numpy array.
    Accepts 0/1 or truthy values.
    """
    arr = np.array(matrix_like, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("Shape matrix must be a 2D list/array.")
    return arr

def normalize(shape: np.ndarray) -> np.ndarray:
    """
    Trim zero-only rows/columns from the edges of the binary shape.
    Returns a new numpy array (dtype=uint8).
    """
    if not isinstance(shape, np.ndarray):
        shape = _to_numpy_matrix(shape)
    else:
        shape = shape.astype(np.uint8)

    # find rows/cols that contain at least one 1
    rows = np.any(shape != 0, axis=1)
    cols = np.any(shape != 0, axis=0)
    if not rows.any() or not cols.any():
        # empty shape -> return a (0,0) array
        return np.zeros((0, 0), dtype=np.uint8)
    rmin, rmax = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    cmin, cmax = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return shape[rmin:rmax, cmin:cmax].astype(np.uint8)

def _array_key(a: np.ndarray) -> bytes:
    """
    Unique key for deduping variants: shape bytes + shape dims
    """
    return (str(a.shape) + ":" + a.tobytes().hex()).encode("utf-8")

def generate_variants(shape: np.ndarray, allow_reflect: bool = False) -> List[np.ndarray]:
    """
    Generate unique rotated variants of a normalized binary shape.
    If allow_reflect is True, also include mirrored variants (flip left-right).
    Returns a list of numpy arrays (each normalized).
    """
    base = normalize(shape)
    if base.size == 0:
        return [base]
    variants: List[np.ndarray] = []
    seen = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for candidate in ([rot] if not allow_reflect else [rot, np.fliplr(rot)]):
            cand = normalize(candidate)
            key = _array_key(cand)
            if key not in seen:
                seen.add(key)
                variants.append(cand.copy())
    return variants

def load_shapes_from_json(path: str) -> Dict[str, Any]:
    """
    Load input JSON for Option 1. Expected structure:
    {
      "B": <int>,
      "blocks": [
        {"copies": <int>, "matrix": [[0/1,...], ...], "name": "optional"},
        ...
      ]
    }
    Returns parsed dict. Raises ValueError for malformed input.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "B" not in data or "blocks" not in data:
        raise ValueError("JSON must include 'B' and 'blocks' keys.")
    if not isinstance(data["blocks"], list):
        raise ValueError("'blocks' must be a list of block entries.")

    # Basic validation and normalization of matrices
    blocks = []
    for idx, b in enumerate(data["blocks"]):
        if "copies" not in b or "matrix" not in b:
            raise ValueError(f"Each block entry must contain 'copies' and 'matrix' (entry {idx}).")
        copies = int(b["copies"])
        matrix = _to_numpy_matrix(b["matrix"])
        name = b.get("name", f"shape_{idx}")
        blocks.append({"copies": copies, "matrix": matrix.tolist(), "name": name})
    return {"B": int(data["B"]), "blocks": blocks}

def expand_shapes_from_input(input_data: Dict[str, Any], allow_reflect: bool = False) -> List[Piece]:
    """
    Given a parsed input dict (the output of load_shapes_from_json or an equivalent dict),
    expand each shape into individual Piece copies with unique ids and precomputed variants.

    Returns a list of Piece objects (one per copy). IDs start at 1 and increase.
    """
    if "blocks" not in input_data:
        raise ValueError("Input must contain 'blocks' key (list).")
    pieces: List[Piece] = []
    pid = 1
    for shape_idx, b in enumerate(input_data["blocks"]):
        copies = int(b["copies"])
        matrix = _to_numpy_matrix(b["matrix"])
        name = b.get("name", f"shape_{shape_idx}")
        base = normalize(matrix)
        area = int(base.sum())
        if area <= 0:
            raise ValueError(f"Shape {shape_idx} ('{name}') has zero area after normalization.")
        variants = generate_variants(base, allow_reflect=allow_reflect)
        for _ in range(copies):
            pieces.append(Piece(
                id=pid,
                shape_idx=shape_idx,
                name=name,
                base=base.copy(),
                variants=[v.copy() for v in variants],
                area=area
            ))
            pid += 1
    return pieces

# Example helper: build a minimal input dict from a list of (matrix, copies, name)
def build_input_from_list(shapes_list: List[Tuple[Any, int, Optional[str]]]) -> Dict[str, Any]:
    """
    Helper to construct an input dict programmatically.
    shapes_list: list of tuples (matrix, copies, name)
    """
    blocks = []
    for (mat, copies, name) in shapes_list:
        blocks.append({"copies": int(copies), "matrix": _to_numpy_matrix(mat).tolist(), "name": name or ""})
    return {"B": len(blocks), "blocks": blocks}
