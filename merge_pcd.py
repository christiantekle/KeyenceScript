import numpy as np
import matplotlib.pyplot as plt
from clean_depth_utils import pcd_to_points, points_to_depth_map

def find_phase_shift(arr1, arr2, max_offset=None):
    """
    Estimate the offset between two 1D arrays by minimizing their difference.
    Returns the offset (in indices) that best aligns arr2 to arr1.
    """
    min_len = min(len(arr1), len(arr2))
    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]
    if max_offset is None:
        max_offset = min_len // 4
    min_diff = float('inf')
    best_offset = 0
    for offset in range(-max_offset, max_offset + 1):
        if offset >= 0:
            shifted_arr2 = arr2[offset:]
            diffs = np.abs(arr1[:len(shifted_arr2)] - shifted_arr2)
        else:
            shifted_arr1 = arr1[-offset:]
            diffs = np.abs(shifted_arr1 - arr2[:len(shifted_arr1)])
        diff = np.nansum(diffs) / len(diffs)
        if diff < min_diff:
            min_diff = diff
            best_offset = offset
    return best_offset

def shift_depth_map(depth_map, shift_x, shift_y):
    """
    Shift a 2D depth map by shift_x and shift_y pixels, filling gaps with NaN.
    """
    rows, cols = depth_map.shape
    shifted_map = np.full_like(depth_map, np.nan)
    x_start = max(0, shift_x)
    x_end = cols - max(0, -shift_x)
    y_start = max(0, shift_y)
    y_end = rows - max(0, -shift_y)
    shifted_map[y_start:y_end, x_start:x_end] = depth_map[max(0, -shift_y):rows - max(0, shift_y), max(0, -shift_x):cols - max(0, shift_x)]
    return shifted_map

def merge_depth_maps(depth_map1, depth_map2, method='max'):
    """
    Merge two depth maps of the same shape using the specified method ('max', 'mean').
    """
    if depth_map1.shape != depth_map2.shape:
        raise ValueError("Depth maps must have the same shape")
    if method == 'max':
        return np.nanmax(np.stack([depth_map1, depth_map2], axis=0), axis=0)
    elif method == 'mean':
        return np.nanmean(np.stack([depth_map1, depth_map2], axis=0), axis=0)
    else:
        raise ValueError("Method must be 'max' or 'mean'")

# File paths
pcd1_file = "Messung1_Scanner1.pcd"
pcd2_file = "Messung1_Scanner2.pcd"
resolution = 0.02  # 20 microns per pixel

# Step 1: Load PCD files into point clouds
pcd1 = pcd_to_points(pcd1_file)
pcd2 = pcd_to_points(pcd2_file)

# Step 2: Convert to depth maps with the same grid

xmin = min(np.nanmin(pcd1[:, 0]), np.nanmin(pcd2[:, 0]))
xmax = max(np.nanmax(pcd1[:, 0]), np.nanmax(pcd2[:, 0]))
ymin = min(np.nanmin(pcd1[:, 1]), np.nanmin(pcd2[:, 1]))
ymax = max(np.nanmax(pcd1[:, 1]), np.nanmax(pcd2[:, 1]))
bounds = ((xmin, xmax), (ymin, ymax))

depth_map1, x_edges, y_edges = points_to_depth_map(
    pcd1,
    resolution=resolution,
    agg="max",
    bounds=bounds
)

depth_map2, _, _ = points_to_depth_map(
    pcd2,
    resolution=resolution,
    agg="max",
    bounds=bounds
)

# Step 3: Estimate x/y offsets by comparing middle rows/columns

middle_row1 = depth_map1[depth_map1.shape[0] // 2, :]
middle_row2 = depth_map2[depth_map2.shape[0] // 2, :]
offset_x = find_phase_shift(middle_row1, middle_row2)

middle_col1 = depth_map1[:, depth_map1.shape[1] // 2]
middle_col2 = depth_map2[:, depth_map2.shape[1] // 2]
offset_y = find_phase_shift(middle_col1, middle_col2)

print(f"Estimated offsets: x={offset_x} pixels, y={offset_y} pixels")

# Step 4: Shift the second depth mapö
depth_map2_shifted = shift_depth_map(depth_map2, offset_x, offset_y)

# Step 5: Merge the depth maps
merged_depth_map = merge_depth_maps(depth_map1, depth_map2_shifted, method='max')

# Step 6: Visualize the merged depth map
plt.imshow(merged_depth_map, cmap='viridis', origin='lower',
           extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
plt.colorbar(label='Depth (Z)')
plt.title('Merged Depth Map from Two PCD Files')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.show()