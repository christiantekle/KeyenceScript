"""
io_utils.py

File I/O utilities for loading and saving point clouds and depth maps.
"""

import os
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


def load_stl_points(stl_filename: str, sample_points: int = None) -> np.ndarray:
    """
    Load STL file and return vertices as Nx3 array.
    If sample_points is specified, uniformly sample the mesh.
    """
    if o3d is None:
        raise RuntimeError("open3d required for load_stl_points")
    
    mesh = o3d.io.read_triangle_mesh(stl_filename)
    
    if sample_points:
        pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
        pts = np.asarray(pcd.points)
    else:
        pts = np.asarray(mesh.vertices)
    
    return pts


def load_pcd_points(pcd_filename: str) -> np.ndarray:
    """Load PCD file and return points as Nx3 array."""
    if o3d is None:
        raise RuntimeError("open3d required for load_pcd_points")
    
    pcd = o3d.io.read_point_cloud(pcd_filename)
    return np.asarray(pcd.points)


def save_depth_map(filename: str, data: np.ndarray) -> None:
    """Save depth map to .npy file."""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    np.save(filename, data)


def load_depth_map(filename: str) -> np.ndarray:
    """Load depth map from .npy file."""
    return np.load(filename)


def save_to_csv(filename: str, data: np.ndarray) -> None:
    """Save numpy array to CSV file."""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    np.savetxt(filename, data, delimiter=',')


def read_columns(file_path: str, first_col: int = 1, second_col: int = 2) -> np.ndarray:
    """
    Read two columns from a whitespace-delimited file.
    Skips the first line (header).
    
    Returns: 2xN array where first row is first column, second row is second column
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    first = []
    second = []
    
    for line in lines[1:]:  # Skip header
        parts = line.strip().split()
        if len(parts) > max(first_col, second_col):
            first.append(float(parts[first_col]))
            second.append(float(parts[second_col]))
    
    return np.array([first, second])
