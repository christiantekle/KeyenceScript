"""
Utility functions for handling point clouds, generating depth maps, 
basic filtering, and registration/alignment.
"""

from __future__ import annotations

import os
import math
import time
from time import strftime
from typing import Optional, Tuple, List

import numpy as np

# Optional imports (only used if available)
try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from scipy.ndimage import median_filter, sobel, gaussian_filter
    from scipy.stats import linregress
except Exception:
    median_filter = None
    sobel = None
    gaussian_filter = None
    linregress = None

try:
    from sklearn import linear_model
except Exception:
    linear_model = None


TIME_FORMAT = "%H:%M:%S"
DEPTH_MAP_RESOLUTION_XY = 2  # microns


# -------------------------- IO helpers ---------------------------------
def load_stl_points(stl_filename: str, sample_points: Optional[int] = None) -> np.ndarray:
    """
    Load an STL file and return the vertices as an Nx3 array.
    If `sample_points` is set, the mesh is uniformly sampled to that size.
    """
    if o3d is None:
        raise RuntimeError("open3d is required for load_stl_points")
    mesh = o3d.io.read_triangle_mesh(stl_filename)
    if sample_points:
        pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
        pts = np.asarray(pcd.points)
    else:
        pts = np.asarray(mesh.vertices)
    return pts


def load_pcd_points(pcd_filename: str) -> np.ndarray:
    """Load a PCD (or any Open3D-supported point cloud) into an Nx3 array."""
    if o3d is None:
        raise RuntimeError("open3d is required for load_pcd_points")
    pcd = o3d.io.read_point_cloud(pcd_filename)
    return np.asarray(pcd.points)


# -------------------------- Point cloud to depth map --------------------
def interpolate_nans_1d(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 1D array using linear interpolation."""
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
                        crop_z_min: Optional[float] = None,
                        crop_z_max: Optional[float] = None) -> np.ndarray:
    """
    Convert an Nx3 cloud (x,y,z) into a 2D depth map. Values are scaled 
    to microns to match downstream processing.

    Returns a 2D array (rows = Y, cols = X) with NaNs for empty cells.
    """
    if points is None or points.size == 0:
        raise ValueError("Empty point cloud")

    pts = points.copy().astype(float)

    # Convert coordinates to microns
    scale_xy = 1000.0 / DEPTH_MAP_RESOLUTION_XY
    pts[:, :2] *= scale_xy
    pts[:, 2] *= 1000.0

    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        raise ValueError("No finite points after cleanup")

    # Optional plane leveling
    if auto_level_plane:
        pts = ransac_level_top_plane(pts, AutoLevelTolerance=auto_level_tol)

    # Optional z clipping
    if crop_z_min is not None:
        pts = pts[pts[:, 2] >= crop_z_min]
    if crop_z_max is not None:
        pts = pts[pts[:, 2] <= crop_z_max]

    xmin, xmax = np.nanmin(pts[:, 0]), np.nanmax(pts[:, 0])
    ymin, ymax = np.nanmin(pts[:, 1]), np.nanmax(pts[:, 1])

    nx = int(math.ceil(xmax - xmin)) or 1
    ny = int(math.ceil(ymax - ymin)) or 1

    pts[:, 0] -= xmin
    pts[:, 1] -= ymin

    x_idx = pts[:, 0].astype(int)
    y_idx = pts[:, 1].astype(int)

    depth_map = np.full((ny + 1, nx + 1), np.nan, dtype=np.float32)

    # Keep the highest z per cell
    for xi, yi, zi in zip(x_idx, y_idx, pts[:, 2]):
        if np.isnan(zi):
            continue
        cur = depth_map[yi, xi]
        if np.isnan(cur) or zi > cur:
            depth_map[yi, xi] = zi

    # Fill small gaps
    for r in range(depth_map.shape[0]):
        depth_map[r, :] = interpolate_nans_1d(depth_map[r, :])
    for c in range(depth_map.shape[1]):
        depth_map[:, c] = interpolate_nans_1d(depth_map[:, c])

    return depth_map


# -------------------------- Plane leveling --------------------------------
def ransac_level_top_plane(points: np.ndarray, AutoLevelTolerance: float = 450.0) -> np.ndarray:
    """
    Estimate and subtract the top plane of the cloud based on points near 
    the maximum z height. Falls back to median subtraction if RANSAC isn't available.
    """
    if points is None or points.size == 0:
        return points

    zmax = np.nanmax(points[:, 2])
    selector = points[:, 2] >= zmax - AutoLevelTolerance
    top_pts = points[selector]
    if top_pts.shape[0] < 3 or linear_model is None:
        median_z = np.median(top_pts[:, 2]) if top_pts.shape[0] > 0 else 0.0
        points[:, 2] -= median_z
        return points

    try:
        XY = top_pts[:, :2]
        Z = top_pts[:, 2]
        ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=0.1)
        ransac.fit(XY, Z)
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_
        points[:, 2] -= (a * points[:, 0] + b * points[:, 1] + c)
    except Exception:
        median_z = np.median(top_pts[:, 2]) if top_pts.shape[0] > 0 else 0.0
        points[:, 2] -= median_z

    return points


# -------------------------- Simple filters / outlier removal ----------------


def detect_outliers(data: np.ndarray, outlier_bottom_perceptile: float, outlier_top_perceptile: float, remove_rare_depths_histothres: float, use_gradient: bool) -> np.ndarray:
    """
    Detect local outliers in a 2D depth map using percentile thresholds and/or gradient deviation.
    Returns a boolean mask of the same shape as `data`.

    Parameters:
        data: 2D numpy array (depth map)
        outlier_bottom_percentile: bottom percentile threshold (e.g., 1.0 for 1st percentile)
        outlier_top_percentile: top percentile threshold (e.g., 1.0 for 99th percentile)
        remove_rare_depths_thres: minimum relative frequency in % threshold for rare depth removal (0 or None disables)
        use_gradient: if True, also detect outliers based on high gradient magnitude
    """
    if data is None:
       raise ValueError("data is None")
        
    #data = data.astype(float)
    mask = np.zeros(data.shape, dtype=bool)  
    
    finite_data = data[np.isfinite(data)]

    if (outlier_bottom_perceptile is not None) and (outlier_bottom_perceptile > 0.0):
        q1 = np.percentile(finite_data, outlier_bottom_perceptile)
        mask |= (data < q1)
        
    if (outlier_top_perceptile is not None) and (outlier_top_perceptile > 0.0):
        q3 = np.percentile(finite_data, 100 - outlier_top_perceptile)
        mask |= (data > q3) 

    if (remove_rare_depths_histothres is not None) and (remove_rare_depths_histothres > 0):
        hist, bin_edges = np.histogram(finite_data, bins=50)
        hist_rel = hist / hist.sum()  # relative frequencies
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # find rare bins
        rare_bins = hist_rel < remove_rare_depths_histothres/100
        if np.any(rare_bins):
            # mask values falling into rare bins
            for center, is_rare in zip(bin_centers, rare_bins):
                if is_rare:
                   mask |= (np.isclose(data, center, atol=(bin_edges[1]-bin_edges[0])/2))

    if use_gradient and sobel is not None:
       gx = sobel(finite_data, axis=0)
       gy = sobel(finite_data, axis=1)
       grad = np.abs(np.hypot(gx, gy))
       
       q3 = np.percentile(grad[~np.isnan(grad)], 100 - outlier_top_perceptile)
       mask |= (grad > q3)

    return mask


def remove_outliers(data: np.ndarray, outlier_top_perceptile: float = None, outlier_bottom_perceptile: float = None, remove_rare_depths_histothres = 2.5, replace_with_nan: bool = False, kernel_size: int = 9) -> np.ndarray:
    """Replace detected outliers with local median or NaN."""
    arr = data.copy().astype(float)
    mask = detect_outliers(data=arr, outlier_top_perceptile = outlier_top_perceptile, outlier_bottom_perceptile = outlier_bottom_perceptile, remove_rare_depths_histothres=remove_rare_depths_histothres, use_gradient=False)
    arr[mask] = np.nan
    
    # Fill small gaps
    if not(replace_with_nan):
       if median_filter is not None:
          med = median_filter(arr, size=(kernel_size, kernel_size))
          arr[mask] = med[mask]
        
       for c in range(arr.shape[1]):
          interpolated_value = interpolate_nans_1d(arr[:, c])
          col =  arr[:, c]
          nan_mask = np.isnan(col)
          arr[:, c][nan_mask] = interpolated_value[nan_mask] 

    return arr

# -------------------------- Array utilities -------------------------------
def shift_array_by_offset(arr: np.ndarray, shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    """Shift a 2D array in x/y by integer offsets. Fill new areas with NaN."""
    rows, cols = arr.shape
    out = np.full_like(arr, np.nan)
    x0 = max(0, shift_x)
    x1 = min(cols, cols + shift_x) if shift_x < 0 else min(cols, cols)
    sx0 = max(0, -shift_x)
    sx1 = sx0 + (x1 - x0)
    y0 = max(0, shift_y)
    y1 = min(rows, rows + shift_y) if shift_y < 0 else min(rows, rows)
    sy0 = max(0, -shift_y)
    sy1 = sy0 + (y1 - y0)

    out[y0:y1, x0:x1] = arr[sy0:sy1, sx0:sx1]
    return out


def crop_nan_borders(arr: np.ndarray) -> np.ndarray:
    """Remove rows/columns that are entirely NaN."""
    if np.all(~np.isnan(arr)):
        return arr
    row_mask = ~np.all(np.isnan(arr), axis=1)
    col_mask = ~np.all(np.isnan(arr), axis=0)
    return arr[row_mask][:, col_mask]


# -------------------------- Compensation modeling ------------------------
def calc_compensation_model(point_data: np.ndarray, index_for_fit: int, use_ransac: bool = True, axis: int = 0, mode: str = 'OffsetAndSlope') -> Tuple[float, float]:
    """Fit a simple linear model (slope, intercept) along a row or column."""
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if axis == 0:
        z = point_data[index_for_fit, :]
    else:
        z = point_data[:, index_for_fit]
    x = np.arange(z.size)
    valid = np.isfinite(z)
    if np.sum(valid) < 2:
        return 0.0, 0.0
    xv = x[valid].reshape(-1, 1)
    zv = z[valid]
    if use_ransac and linear_model is not None:
        r = linear_model.RANSACRegressor(linear_model.LinearRegression())
        r.fit(xv, zv)
        slope = float(r.estimator_.coef_[0])
        intercept = float(r.estimator_.intercept_)
    else:
        if linregress is None:
            slope, intercept = 0.0, float(np.nanmean(zv))
        else:
            s, inter, rval, p, se = linregress(x[valid], zv)
            slope, intercept = float(s), float(inter)
    if mode == 'OffsetOnly':
        slope = 0.0
    elif mode == 'SlopeOnly':
        intercept = 0.0
    return slope, intercept


def get_lin_model_for_compensation(point_data: np.ndarray, slope: float, intercept: float, axis: int = 0) -> np.ndarray:
    """Generate a 1D linear array for compensation."""
    length = point_data.shape[1] if axis == 0 else point_data.shape[0]
    x = np.arange(length)
    return slope * x + intercept


def apply_compensation_model(point_data: np.ndarray, lin_model: np.ndarray, axis: int = 0) -> np.ndarray:
    """Subtract the linear model from each row/column."""
    out = point_data.copy()
    if axis == 0:
        out[:] = out - lin_model[np.newaxis, :]
    else:
        out[:] = out - lin_model[:, np.newaxis]
    return out


def compensate_with_model(point_data: np.ndarray, index_for_fit: int, use_ransac: bool = True, axis: int = 0, mode: str = 'OffsetAndSlope') -> np.ndarray:
    slope, intercept = calc_compensation_model(point_data, index_for_fit, use_ransac, axis, mode)
    lin_model = get_lin_model_for_compensation(point_data, slope, intercept, axis)
    return apply_compensation_model(point_data, lin_model, axis)


# -------------------------- Alignment utilities -------------------------
def find_phase_shift(y1: np.ndarray, y2: np.ndarray, max_offset: int = -1) -> int:
    """Estimate integer shift between two 1D sequences by minimizing absolute difference."""
    minlen = min(y1.shape[0], y2.shape[0])
    a1 = np.array(y1[:minlen])
    a2 = np.array(y2[:minlen])
    if max_offset < 0:
        max_offset = minlen // 4
    min_diff = float('inf')
    best = 0
    for offset in range(-max_offset, max_offset + 1):
        if offset >= 0:
            s2 = a2[offset:]
            d = np.nansum(np.abs(a1[:len(s2)] - s2)) / max(1, len(s2))
        else:
            s1 = a1[-offset:]
            d = np.nansum(np.abs(s1 - a2[:len(s1)])) / max(1, len(s1))
        if d < min_diff:
            min_diff = d
            best = offset
    return int(best)


def combine_arrays(arr1: np.ndarray, arr2: np.ndarray, method: str = 'min', fill_value: float = np.nan) -> np.ndarray:
    """Combine two arrays of differing shapes into their common overlapping region."""
    rows = min(arr1.shape[0], arr2.shape[0])
    cols = min(arr1.shape[1], arr2.shape[1])
    a1 = arr1[:rows, :cols]
    a2 = arr2[:rows, :cols]
    if method == 'min':
        return np.fmin(a1, a2)
    if method == 'mean':
        return np.nanmean(np.stack([a1, a2], axis=0), axis=0)
    if method == 'diff':
        return a1 - a2
    raise ValueError('unsupported method')


def combine_arrays_with_offsets(arrays: List[np.ndarray], column_offsets: List[int], combine_method: str = 'min', fill_value: float = np.nan) -> np.ndarray:
    """Merge multiple arrays into a shared canvas using per-array column offsets."""
    max_rows = max(a.shape[0] for a in arrays)
    max_cols = max(offset + a.shape[1] for a, offset in zip(arrays, column_offsets))
    res = np.full((max_rows, max_cols), fill_value, dtype=float)
    if combine_method == 'mean':
        count = np.zeros_like(res)
    for arr, off in zip(arrays, column_offsets):
        r, c = arr.shape
        target = (slice(0, r), slice(off, off + c))
        if combine_method == 'min':
            existing = res[target]
            res[target] = np.minimum(existing, arr)
        elif combine_method == 'mean':
            mask = ~np.isnan(arr)
            res[target][mask] = np.nan_to_num(res[target][mask]) + np.nan_to_num(arr[mask])
            count[target][mask] += 1
    if combine_method == 'mean':
        with np.errstate(invalid='ignore'):
            res[count > 0] = res[count > 0] / count[count > 0]
            res[count == 0] = np.nan
    return res


# -------------------------- Registration using OpenCV ECC ----------------
def register_point_cloud_to_stl(stl_depth_map: np.ndarray, point_cloud_depth_map: np.ndarray,
                                 motion_type=int(0), scaling_kernel_size: int = 4,
                                 warp_matrix: Optional[np.ndarray] = None, suppress_scaling: bool = True,
                                 term_max_count: int = 200, term_epsilon: float = -1.0) -> np.ndarray:
    """
    Estimate an affine transform between two depth maps using OpenCV's ECC algorithm.
    Returns a 2×3 warp matrix.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required for registration")

    if warp_matrix is None:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Optional block pooling (reduces image size but preserves local maxima)
    if scaling_kernel_size > 1:
        def pooling(mat, k):
            out_h = (mat.shape[0] - k) // k + 1
            out_w = (mat.shape[1] - k) // k + 1
            res = np.full((out_h, out_w), np.nan)
            for i in range(out_h):
                for j in range(out_w):
                    patch = mat[i*k:i*k+k, j*k:j*k+k]
                    res[i, j] = np.nanmax(patch)
            return res
        stl_depth_map = pooling(stl_depth_map, scaling_kernel_size)
        point_cloud_depth_map = pooling(point_cloud_depth_map, scaling_kernel_size)
        scaling_pyramid_factor = 1.0 / scaling_kernel_size
    else:
        scaling_pyramid_factor = 1.0

    def normalize_to_uint8(mat):
        matf = mat.astype(np.float32)
        mn, mx = np.nanmin(matf), np.nanmax(matf)
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            return ((matf - mn) / (mx - mn) * 255.0).astype(np.uint8)
        else:
            return np.zeros(matf.shape, dtype=np.uint8)

    timg = normalize_to_uint8(stl_depth_map)
    iimg = normalize_to_uint8(point_cloud_depth_map)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, term_max_count, term_epsilon)
    warp = cv2.findTransformECC(templateImage=timg, inputImage=iimg, warpMatrix=warp_matrix, motionType=motion_type, criteria=criteria)[1]

    warp[0, 2] /= scaling_pyramid_factor
    warp[1, 2] /= scaling_pyramid_factor

    if suppress_scaling:
        calc_scaling = math.hypot(warp[0, 0], warp[1, 0])
        if calc_scaling != 0 and not math.isclose(calc_scaling, 1.0):
            warp[0, 0] /= calc_scaling
            warp[1, 1] /= calc_scaling
            warp[0, 1] /= calc_scaling
            warp[1, 0] /= calc_scaling

    return warp


# -------------------------- Visualization helpers -----------------------
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    plt = None


def plot_depth_map(depth_map: np.ndarray, title: str = "depth_map", z_min: Optional[float] = None, z_max: Optional[float] = None,
                   lateral_resolution: float = DEPTH_MAP_RESOLUTION_XY, output_folder: str = ".") -> None:
    """Plot a depth map and save it as PNG."""
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    x = np.arange(depth_map.shape[1]) * lateral_resolution * 1e-3
    y = np.arange(depth_map.shape[0]) * lateral_resolution * 1e-3
    plt.figure()
    im = plt.imshow(depth_map, extent=[x.min(), x.max(), y.min(), y.max()],
                    vmin=z_min, vmax=z_max, origin='lower', cmap='jet', aspect='equal')
    plt.colorbar(im, label='Z [µm]')
    plt.title(title)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{title}.png"))
    plt.show()
    plt.close()


def plot_3d_depth_maps(depth_maps: List[np.ndarray], lateral_resolution: float = DEPTH_MAP_RESOLUTION_XY, alphas: Optional[List[float]] = None, colormap: str = 'jet') -> None:
    """3D surface plot of one or more depth maps."""
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    if not isinstance(depth_maps, list):
        depth_maps = [depth_maps]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ny, nx = depth_maps[0].shape
    x = np.arange(nx) * lateral_resolution * 1e-3
    y = np.arange(ny) * lateral_resolution * 1e-3
    X, Y = np.meshgrid(x, y)
    for Z in depth_maps:
        ax.plot_surface(X, Y, Z * 1e-3, cmap=colormap, edgecolor='none')
    plt.show()


# -------------------------- Export / util --------------------------------
def save_depth_map_to_npy(filename: str, data: np.ndarray) -> None:
    """Save a depth map to a .npy file."""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    np.save(filename, data)


def load_depth_map_from_npy(filename: str) -> np.ndarray:
    """Load a depth map from a .npy file."""
    return np.load(filename)


# -------------------------- Parallel apply utilities --------------------
def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Apply a function along an axis using multiprocessing (simple split on axis 0)."""
    import multiprocessing as mp
    from functools import partial
    splitted = np.array_split(arr, mp.cpu_count(), axis=0)
    with mp.Pool() as pool:
        results = pool.map(partial(np.apply_along_axis, func1d, axis), splitted)
    return np.concatenate(results, axis=0)


# -------------------------- File column reader ---------------------------
def read_columns(file_path: str, first_col: int = 1, second_col: int = 2) -> np.ndarray:
    """Read two numeric columns from a whitespace file, skipping the first line."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    first = []
    second = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) <= max(first_col, second_col):
            continue
        first.append(float(parts[first_col]))
        second.append(float(parts[second_col]))
    return np.vstack([first, second])


# -------------------------- Script entry point --------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python to_be_cleaned.py <pcd_or_stl_path>")
        sys.exit(0)
    path = sys.argv[1]
    kind = 'pcd' if path.lower().endswith('.pcd') else 'stl'
    if kind == 'pcd':
        pts = load_pcd_points(path)
    else:
        pts = load_stl_points(path, sample_points=50000)

    dm = points_to_depth_map(pts, resolution=DEPTH_MAP_RESOLUTION_XY)
    print('Depth map shape:', dm.shape)
    if plt is not None:
        plot_depth_map(dm, title='depth_map_demo')
