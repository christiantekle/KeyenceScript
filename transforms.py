"""
transforms.py

Geometric transformations, plane leveling, and compensation utilities.
"""

import numpy as np

try:
    from scipy.stats import linregress
except Exception:
    linregress = None

try:
    from sklearn import linear_model
except Exception:
    linear_model = None


def ransac_level_plane(points: np.ndarray, tolerance: float = 450.0) -> np.ndarray:
    """
    Estimate and subtract top plane using RANSAC.
    Falls back to median subtraction if RANSAC unavailable.
    """
    if points is None or points.size == 0:
        return points
    
    z_max = np.nanmax(points[:, 2])
    selector = points[:, 2] >= z_max - tolerance
    top_pts = points[selector]
    
    if top_pts.shape[0] < 3 or linear_model is None:
        median_z = np.median(top_pts[:, 2]) if top_pts.shape[0] > 0 else 0.0
        points[:, 2] -= median_z
        return points
    
    try:
        XY = top_pts[:, :2]
        Z = top_pts[:, 2]
        ransac = linear_model.RANSACRegressor(
            linear_model.LinearRegression(), 
            residual_threshold=0.1
        )
        ransac.fit(XY, Z)
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_
        points[:, 2] -= (a * points[:, 0] + b * points[:, 1] + c)
    except Exception:
        median_z = np.median(top_pts[:, 2]) if top_pts.shape[0] > 0 else 0.0
        points[:, 2] -= median_z
    
    return points


def calc_compensation_model(point_data: np.ndarray,
                            index_for_fit: int,
                            use_ransac: bool = True,
                            axis: int = 0,
                            mode: str = 'OffsetAndSlope') -> tuple:
    """
    Fit linear model (slope, intercept) along a row or column.
    
    Args:
        axis: 0 for x-direction (columns), 1 for y-direction (rows)
        mode: 'OffsetAndSlope', 'OffsetOnly', or 'SlopeOnly'
    
    Returns: (slope, intercept)
    """
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")
    
    if axis == 0:
        z = point_data[index_for_fit, :]
    else:
        z = point_data[:, index_for_fit]
    
    x = np.arange(len(z))
    valid = np.isfinite(z)
    
    if np.sum(valid) < 2:
        return 0.0, 0.0
    
    x_valid = x[valid].reshape(-1, 1)
    z_valid = z[valid]
    
    if use_ransac and linear_model is not None:
        ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        ransac.fit(x_valid, z_valid)
        slope = float(ransac.estimator_.coef_[0])
        intercept = float(ransac.estimator_.intercept_)
    else:
        if linregress is None:
            slope, intercept = 0.0, float(np.nanmean(z_valid))
        else:
            s, inter, _, _, _ = linregress(x[valid], z_valid)
            slope, intercept = float(s), float(inter)
    
    if mode == 'OffsetOnly':
        slope = 0.0
    elif mode == 'SlopeOnly':
        intercept = 0.0
    
    return slope, intercept


def get_lin_model(point_data: np.ndarray, slope: float, intercept: float, axis: int = 0) -> np.ndarray:
    """Generate 1D linear array for compensation."""
    length = point_data.shape[1] if axis == 0 else point_data.shape[0]
    x = np.arange(length)
    return slope * x + intercept


def apply_compensation_model(point_data: np.ndarray, lin_model: np.ndarray, axis: int = 0) -> np.ndarray:
    """Subtract linear model from each row/column."""
    out = point_data.copy()
    if axis == 0:
        out[:] = out - lin_model[np.newaxis, :]
    else:
        out[:] = out - lin_model[:, np.newaxis]
    return out


def compensate_with_model(point_data: np.ndarray,
                         index_for_fit: int,
                         use_ransac: bool = True,
                         axis: int = 0,
                         mode: str = 'OffsetAndSlope') -> np.ndarray:
    """Complete compensation pipeline: fit, generate model, apply."""
    slope, intercept = calc_compensation_model(point_data, index_for_fit, use_ransac, axis, mode)
    lin_model = get_lin_model(point_data, slope, intercept, axis)
    return apply_compensation_model(point_data, lin_model, axis)


def shift_array(arr: np.ndarray, shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    """Shift 2D array by offset. Fill empty cells with NaN."""
    rows, cols = arr.shape
    shifted = np.full_like(arr, np.nan)
    
    shift_x = abs(shift_x)
    shift_y = abs(shift_y)
    
    if shift_x > 0:
        shifted = np.hstack([np.full((rows, shift_x), np.nan), arr])
    else:
        shifted = arr
    
    if shift_y > 0:
        shifted = np.vstack([np.full((shift_y, shifted.shape[1]), np.nan), shifted])
    
    return shifted


def mirror_vertical(arr: np.ndarray) -> np.ndarray:
    """Flip array along vertical axis (columns)."""
    return np.flip(arr, axis=1)


def crop_nan_borders(arr: np.ndarray) -> np.ndarray:
    """Remove rows/columns that are entirely NaN."""
    if np.all(~np.isnan(arr)):
        return arr
    
    row_mask = ~np.all(np.isnan(arr), axis=1)
    col_mask = ~np.all(np.isnan(arr), axis=0)
    
    return arr[row_mask][:, col_mask]
