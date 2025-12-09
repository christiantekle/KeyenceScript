"""
registration.py

Registration and alignment utilities for depth maps and point clouds.
"""

import math
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def find_phase_shift(y1: np.ndarray, y2: np.ndarray, max_offset: int = -1) -> int:
    """
    Estimate integer shift between two 1D arrays by minimizing difference.
    """
    min_len = min(y1.shape[0], y2.shape[0])
    y1 = np.array(y1)[:min_len]
    y2 = np.array(y2)[:min_len]
    
    if max_offset < 0:
        max_offset = min_len // 4
    
    min_diff = float('inf')
    best_offset = 0
    
    for offset in range(-max_offset, max_offset + 1):
        if offset >= 0:
            shifted_y2 = y2[offset:]
            diffs = np.abs(y1[:len(shifted_y2)] - shifted_y2)
        else:
            shifted_y1 = y1[-offset:]
            diffs = np.abs(y2[:len(shifted_y1)] - shifted_y1)
        
        diff = np.sum(diffs) / min_len
        if diff < min_diff:
            min_diff = diff
            best_offset = offset
    
    return best_offset


def combine_arrays(arr1: np.ndarray, arr2: np.ndarray, 
                   method: str = 'min', fill_value: float = np.nan) -> np.ndarray:
    """
    Combine two arrays using specified method.
    Arrays are cropped to minimum overlapping size.
    """
    rows = min(arr1.shape[0], arr2.shape[0])
    cols = min(arr1.shape[1], arr2.shape[1])
    
    a1 = arr1[:rows, :cols]
    a2 = arr2[:rows, :cols]
    
    if method == 'min':
        return np.minimum(a1, a2)
    elif method == 'mean':
        return np.nanmean(np.stack([a1, a2], axis=0), axis=0)
    elif method == 'diff':
        return a1 - a2
    else:
        raise ValueError(f"Unsupported method: {method}")


def combine_arrays_with_offsets(arrays: list, column_offsets: list,
                                combine_method: str = 'min',
                                fill_value: float = np.nan) -> np.ndarray:
    """
    Merge multiple arrays into shared canvas using column offsets.
    """
    max_rows = max(arr.shape[0] for arr in arrays)
    max_cols = max(offset + arr.shape[1] for arr, offset in zip(arrays, column_offsets))
    
    result = np.full((max_rows, max_cols), fill_value, dtype=float)
    
    if combine_method == 'mean':
        count = np.zeros((max_rows, max_cols), dtype=np.int32)
    
    for arr, offset in zip(arrays, column_offsets):
        rows, cols = arr.shape
        target_slice = (slice(0, rows), slice(offset, offset + cols))
        
        if combine_method == 'min':
            existing = result[target_slice]
            result[target_slice] = np.fmin(existing, arr)
        elif combine_method == 'mean':
            mask = ~np.isnan(arr)
            result[target_slice][mask] = np.nan_to_num(result[target_slice][mask]) + np.nan_to_num(arr[mask])
            count[target_slice][mask] += 1
    
    if combine_method == 'mean':
        with np.errstate(divide='ignore', invalid='ignore'):
            result[count > 0] = result[count > 0] / count[count > 0]
            result[count == 0] = fill_value
    
    return result


def pooling_overlap(mat: np.ndarray, ksize: tuple, stride: tuple = None, 
                   method: str = 'max', pad: bool = False) -> np.ndarray:
    """
    Overlapping pooling on 2D array using max or mean.
    """
    m, n = mat.shape[:2]
    ky, kx = ksize
    
    if stride is None:
        stride = (ky, kx)
    sy, sx = stride
    
    if pad:
        ny = int(np.ceil(m / sy))
        nx = int(np.ceil(n / sx))
        size = ((ny - 1) * sy + ky, (nx - 1) * sx + kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m - ky) // sy * sy + ky, :(n - kx) // sx * sx + kx, ...]
    
    out_h = (mat_pad.shape[0] - ky) // sy + 1
    out_w = (mat_pad.shape[1] - kx) // sx + 1
    result = np.full((out_h, out_w), np.nan)
    
    for i in range(out_h):
        for j in range(out_w):
            patch = mat_pad[i * sy:i * sy + ky, j * sx:j * sx + kx]
            if method == 'max':
                result[i, j] = np.nanmax(patch)
            else:
                result[i, j] = np.nanmean(patch)
    
    return result


def register_depth_maps(template_map: np.ndarray, input_map: np.ndarray,
                       motion_type: int = 0,
                       scaling_kernel: int = 4,
                       warp_matrix: np.ndarray = None,
                       suppress_scaling: bool = True,
                       term_max_count: int = 200,
                       term_epsilon: float = -1.0) -> np.ndarray:
    """
    Estimate affine transform between two depth maps using OpenCV ECC.
    
    Returns: 2x3 warp matrix
    """
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) required for registration")
    
    if warp_matrix is None:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Pooling for speed
    if scaling_kernel > 1:
        template_map = pooling_overlap(template_map, (scaling_kernel, scaling_kernel))
        input_map = pooling_overlap(input_map, (scaling_kernel, scaling_kernel))
        scale_factor = 1.0 / scaling_kernel
    else:
        scale_factor = 1.0
    
    # Normalize to uint8
    def normalize_uint8(m):
        mf = m.astype(np.float32)
        mn, mx = np.nanmin(mf), np.nanmax(mf)
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            return ((mf - mn) / (mx - mn) * 255.0).astype(np.uint8)
        return np.zeros(mf.shape, dtype=np.uint8)
    
    timg = normalize_uint8(template_map)
    iimg = normalize_uint8(input_map)
    
    # ECC registration
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, term_max_count, term_epsilon)
    warp = cv2.findTransformECC(templateImage=timg, inputImage=iimg, 
                                warpMatrix=warp_matrix, motionType=motion_type,
                                criteria=criteria)[1]
    
    # Correct scaling
    warp[0, 2] /= scale_factor
    warp[1, 2] /= scale_factor
    
    if suppress_scaling:
        calc_scale = math.hypot(warp[0, 0], warp[1, 0])
        if calc_scale != 0 and not math.isclose(calc_scale, 1.0):
            warp[0, 0] /= calc_scale
            warp[1, 1] /= calc_scale
            warp[0, 1] /= calc_scale
            warp[1, 0] /= calc_scale
    
    return warp
