import os
import json
import pickle
import pandas as pd
from IPython.display import display


def ensure_dir(path: str):
    """
    Create directory if it does not exist.
    
    Parameters
    ----------
    path : str
        Directory path.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame.
    
    Parameters
    ----------
    path : str
        CSV file path.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str, index: bool = False):
    """
    Save DataFrame to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
    path : str
    index : bool
    """
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)
    print(f"Saved CSV -> {path}")



def save_pickle(obj, path: str):
    """
    Save Python object using pickle.
    
    Parameters
    ----------
    obj : any
    path : str
    """
    ensure_dir(os.path.dirname(path))
    
    with open(path, "wb") as f:
        pickle.dump(obj, f)
        
    print(f"Saved Pickle -> {path}")


def load_pickle(path: str):
    """
    Load Python object from pickle.
    
    Parameters
    ----------
    path : str
    
    Returns
    -------
    any
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data, path: str, indent: int = 4):
    """
    Save dictionary/list to JSON file.
    
    Parameters
    ----------
    data : dict | list
    path : str
    indent : int
    """
    ensure_dir(os.path.dirname(path))
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        
    print(f"Saved JSON -> {path}")


def load_json(path: str):
    """
    Load JSON file.
    
    Parameters
    ----------
    path : str
    
    Returns
    -------
    dict | list
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preview_df(df: pd.DataFrame, n: int = 5):
    """
    Print quick preview of dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
    n : int
    """
    print("=" * 60)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("=" * 60)
    display(df.head(n))
    print("=" * 60)
