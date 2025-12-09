"""
filters.py

Filtering and outlier removal utilities for depth maps.
"""

import numpy as np

try:
    from scipy.ndimage import median_filter, sobel, gaussian_filter
    from scipy.signal import convolve2d
except Exception:
    median_filter = None
    sobel = None
    gaussian_filter = None
    convolve2d = None


def detect_outliers(data: np.ndarray, 
                   threshold_median: float = 0.5,
                   threshold_gradient: float = 0.5,
                   kernel_size: int = 9,
                   use_iqr: bool = True,
                   use_gradient: bool = True) -> np.ndarray:
    """
    Detect local outliers using IQR analysis and/or gradient method.
    
    Returns: Boolean mask with True for detected outliers
    """
    if data.dtype not in [np.float32, np.float64]:
        data = data.astype(np.float32)
    
    kernel_size = max(3, int(kernel_size))
    mask = np.zeros_like(data, dtype=bool)
    
    # IQR-based detection
    if use_iqr and median_filter is not None:
        med = median_filter(data, size=(kernel_size, kernel_size))
        diff = np.abs(data - med)
        q1 = np.percentile(diff[~np.isnan(diff)], 25)
        q3 = np.percentile(diff[~np.isnan(diff)], 75)
        iqr = q3 - q1
        mask |= diff > (threshold_median * max(iqr, 1e-9))
    
    # Gradient-based detection
    if use_gradient and sobel is not None:
        gx = sobel(data, axis=0)
        gy = sobel(data, axis=1)
        grad = np.hypot(gx, gy)
        
        med_grad = median_filter(grad, size=(kernel_size, kernel_size))
        diff_grad = np.abs(grad - med_grad)
        q3 = np.percentile(diff_grad[~np.isnan(diff_grad)], 75)
        mask |= diff_grad > (threshold_gradient * max(q3, 1e-9))
    
    return mask


def remove_outliers(data: np.ndarray,
                   threshold_median: float = 0.5,
                   threshold_gradient: float = 0.5,
                   kernel_size: int = 9,
                   replace_with_nan: bool = False,
                   use_iqr: bool = True,
                   use_gradient: bool = True,
                   crop_z_max: float = None) -> np.ndarray:
    """
    Remove outliers from depth map. Replace with median or NaN.
    """
    arr = data.copy().astype(float)
    outliers = detect_outliers(arr, threshold_median, threshold_gradient, 
                               kernel_size, use_iqr, use_gradient)
    
    if replace_with_nan:
        arr[outliers] = np.nan
    else:
        if median_filter is not None:
            med = median_filter(arr, size=(kernel_size, kernel_size))
            arr[outliers] = med[outliers]
        else:
            arr[outliers] = np.nan
    
    if crop_z_max is not None:
        arr[arr > crop_z_max] = crop_z_max
    
    return arr


def gaussian_filter_2d(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian filter to 2D array."""
    if gaussian_filter is None:
        raise RuntimeError("scipy required for gaussian_filter_2d")
    return gaussian_filter(data, sigma=sigma)


def mean_filter_2d(data: np.ndarray, size: int) -> np.ndarray:
    """Apply mean filter to 2D array."""
    if convolve2d is None:
        raise RuntimeError("scipy required for mean_filter_2d")
    kernel = np.ones((size, size)) / (size * size)
    return convolve2d(data, kernel, mode='same', boundary='wrap')


def median_filter_2d(data: np.ndarray, size: int) -> np.ndarray:
    """Apply median filter to 2D array."""
    if median_filter is None:
        raise RuntimeError("scipy required for median_filter_2d")
    return median_filter(data, size=size)


def sliding_average(z: np.ndarray, n: int = 4) -> np.ndarray:
    """
    Compute sliding average for each row of depth map.
    Averages n nearby rows for smoothing.
    """
    rows, cols = z.shape
    smoothed_map = np.empty((rows, cols))
    
    for i in range(0, rows, n // 2):
        start = max(0, i - n // 2)
        end = min(rows, i + n // 2 + 1)
        smoothed_map[i:i + n // 2, :] = np.mean(z[start:end, :], axis=0)
    
    return smoothed_map
