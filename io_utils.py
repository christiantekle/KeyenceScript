"""
io_utils.py

File I/O utilities for saving and loading depth maps and CSV data.
Note: For loading PCD/STL point clouds, use clean_depth_utils.py
"""

import os
import numpy as np


def save_depth_map(filename: str, data: np.ndarray) -> None:
    """Save depth map to .npy file."""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    np.save(filename, data)


def load_depth_map(filename: str) -> np.ndarray:
    """Load depth map from .npy file."""
    return np.load(filename)


def save_to_csv(filename: str, data: np.ndarray, delimiter: str = ',', header: str = '') -> None:
    """
    Save numpy array to CSV file.
    
    Args:
        filename: Output CSV file path
        data: Numpy array to save
        delimiter: Delimiter character (default: ',')
        header: Optional header string
    """
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    if header:
        np.savetxt(filename, data, delimiter=delimiter, header=header, comments='')
    else:
        np.savetxt(filename, data, delimiter=delimiter)


def read_columns(file_path: str, first_col: int = 1, second_col: int = 2) -> np.ndarray:
    """
    Read two columns from a whitespace-delimited file.
    Skips the first line (header).
    
    Args:
        file_path: Path to input file
        first_col: Index of first column to extract (0-based)
        second_col: Index of second column to extract (0-based)
    
    Returns:
        2xN array where first row is first column, second row is second column
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    first = []
    second = []
    
    for line in lines[1:]:  # Skip header
        parts = line.strip().split()
        if len(parts) > max(first_col, second_col):
            try:
                first.append(float(parts[first_col]))
                second.append(float(parts[second_col]))
            except (ValueError, IndexError):
                continue  # Skip malformed lines
    
    return np.array([first, second])


def load_columns_from_csv(filename: str, cols: list = None, delimiter: str = ',', 
                          skip_header: int = 1) -> np.ndarray:
    """
    Load specific columns from a CSV file.
    
    Args:
        filename: Path to CSV file
        cols: List of column indices to load (None = all columns)
        delimiter: Delimiter character
        skip_header: Number of header rows to skip
    
    Returns:
        Numpy array with requested columns
    """
    return np.loadtxt(filename, delimiter=delimiter, skiprows=skip_header, 
                     usecols=cols, ndmin=2)