from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

def ensure_dir(path: PathLike) -> Path:
    """
    Ensure that a directory exists. Create it if it does not exist (and parents), and return the Path.

    Args:
        path (PathLike): The directory path to ensure.
    
    Returns:
        Path: The ensured directory path as a Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: PathLike, obj: Dict[str, Any], indent: int = 2) -> None:
    """
    Save a dictionary as a JSON file.
    
    Args:
        path (PathLike): Output file path.
        obj (Dict[str, Any]): The dictionary to save.
        indent (int, optional): The indentation level for pretty-printing.

    Returns:
        None
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent)


def load_json(path: PathLike) -> Dict[str, Any]:
    """
    Load a JSON file into a Python dict.

    Args:
        path: JSON file path to read.

    Returns:
        Dictionary parsed from JSON.
    """
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)
    

def save_csv(path: PathLike, df: pd.DataFrame, index: bool = False) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Parent directories are created automatically.

    Args:
        path: Output CSV file path.
        df: DataFrame to write.
        index: Whether to write row index into CSV.

    Returns:
        None
    """
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index)

def read_csv(path: PathLike) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Args:
        path: CSV file path.

    Returns:
        DataFrame containing CSV contents.
    """
    return pd.read_csv(Path(path))

def save_npz(path: PathLike, **arrays: np.ndarray) -> None:
    """
    Save one or more numpy arrays to a compressed .npz archive.

    Parent directories are created automatically.

    Args:
        path: Output path ending in ".npz".
        **arrays: Named numpy arrays to store (e.g., X=..., X_hat=...).

    Returns:
        None
    """
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(p, **arrays)

def load_npz(path: PathLike) -> Dict[str, np.ndarray]:
    """
    Load a compressed .npz archive into a dict of numpy arrays.

    Args:
        path: Path to a ".npz" file.

    Returns:
        Dict mapping stored array names to numpy arrays.
    """
    p = Path(path)
    data = np.load(p, allow_pickle=False)
    return {key: data[key] for key in data.files}