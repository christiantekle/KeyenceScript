"""
processing_pipeline.py

High-level processing pipeline for converting point clouds to depth maps.
"""

import math
import numpy as np
from transforms import ransac_level_plane


DEPTH_MAP_RESOLUTION_XY = 2  # microns


def interpolate_nans_1d(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs in 1D array using linear interpolation."""
    try:
        bad = np.isnan(arr)
        if not np.any(bad):
            return arr
        
        good = ~bad
        if np.sum(good) < 2:
            return arr
        
        x = np.arange(arr.size)
        arr[bad] = np.interp(x[bad], x[good], arr[good])
        return arr
    except Exception:
        return arr


def points_to_depth_map(points: np.ndarray,
                        resolution: float = DEPTH_MAP_RESOLUTION_XY,
                        auto_level_plane: bool = True,
                        auto_level_tol: float = 450.0,
                        crop_z_min: float = None,
                        crop_z_max: float = None) -> np.ndarray:
    """
    Convert Nx3 point cloud to 2D depth map.
    
    Args:
        points: Nx3 array of (x, y, z) coordinates
        resolution: Grid cell size in microns
        auto_level_plane: Whether to level the top plane
        auto_level_tol: Tolerance for plane leveling (microns)
        crop_z_min: Minimum z value to keep
        crop_z_max: Maximum z value to keep
    
    Returns:
        2D depth map array (rows=Y, cols=X) with NaN for empty cells
    """
    if points is None or points.size == 0:
        raise ValueError("Empty point cloud")
    
    pts = points.copy().astype(float)
    
    # Convert to microns and scale xy coordinates to grid units
    scale_xy = 1000.0 / resolution
    pts[:, :2] *= scale_xy
    pts[:, 2] *= 1000.0
    
    # Remove non-finite points
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        raise ValueError("No finite points after cleanup")
    
    # Plane leveling
    if auto_level_plane:
        pts = ransac_level_plane(pts, tolerance=auto_level_tol)
    
    # Z cropping
    if crop_z_min is not None:
        pts = pts[pts[:, 2] >= crop_z_min]
    if crop_z_max is not None:
        pts = pts[pts[:, 2] <= crop_z_max]
    
    if pts.size == 0:
        raise ValueError("No points remaining after cropping")
    
    # Determine grid size
    x_min, x_max = np.nanmin(pts[:, 0]), np.nanmax(pts[:, 0])
    y_min, y_max = np.nanmin(pts[:, 1]), np.nanmax(pts[:, 1])
    
    nx = max(1, int(math.ceil(x_max - x_min)))
    ny = max(1, int(math.ceil(y_max - y_min)))
    
    # Shift to origin
    pts[:, 0] -= x_min
    pts[:, 1] -= y_min
    
    # Convert to grid indices
    x_idx = pts[:, 0].astype(int)
    y_idx = pts[:, 1].astype(int)
    
    # Initialize depth map
    depth_map = np.full((ny + 1, nx + 1), np.nan, dtype=np.float32)
    
    # Fill with maximum z per cell
    for xi, yi, zi in zip(x_idx, y_idx, pts[:, 2]):
        if np.isnan(zi):
            continue
        cur = depth_map[yi, xi]
        if np.isnan(cur) or zi > cur:
            depth_map[yi, xi] = zi
    
    # Interpolate missing values
    for r in range(depth_map.shape[0]):
        depth_map[r, :] = interpolate_nans_1d(depth_map[r, :])
    
    for c in range(depth_map.shape[1]):
        depth_map[:, c] = interpolate_nans_1d(depth_map[:, c])
    
    return depth_map


def load_and_process_pcd(pcd_filename: str,
                         resolution: float = DEPTH_MAP_RESOLUTION_XY,
                         auto_level: bool = True,
                         auto_level_tol: float = 450.0,
                         crop_z_min: float = None,
                         crop_z_max: float = None,
                         initial_crop_z_max: float = None,
                         manual_z_offset: float = None) -> np.ndarray:
    """
    High-level function to load PCD and convert to depth map.
    """
    from io_utils import load_pcd_points
    
    points = load_pcd_points(pcd_filename)
    
    # Initial z cropping before main processing
    if initial_crop_z_max is not None:
        points = points[points[:, 2] <= initial_crop_z_max / 1000.0]
    
    # Remove NaN points
    points = points[~np.isnan(points).any(axis=1)]
    
    depth_map = points_to_depth_map(points, resolution, auto_level, 
                                    auto_level_tol, crop_z_min, crop_z_max)
    
    # Apply manual offset
    if manual_z_offset is not None:
        depth_map += manual_z_offset
    
    return depth_map


def load_and_process_stl(stl_filename: str,
                         sample_points: int = None,
                         resolution: float = DEPTH_MAP_RESOLUTION_XY,
                         auto_level: bool = True,
                         auto_level_tol: float = 450.0,
                         crop_z_min: float = None,
                         crop_z_max: float = None) -> np.ndarray:
    """
    High-level function to load STL and convert to depth map.
    """
    from io_utils import load_stl_points
    
    points = load_stl_points(stl_filename, sample_points)
    points = points[~np.isnan(points).any(axis=1)]
    
    depth_map = points_to_depth_map(points, resolution, auto_level,
                                    auto_level_tol, crop_z_min, crop_z_max)
    
    return depth_map
