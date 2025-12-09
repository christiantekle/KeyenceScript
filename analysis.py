"""
analysis.py

Analysis utilities for depth maps: cuts, statistics, and quality metrics.
"""

import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt as None

try:
    from scipy.stats import linregress
except Exception:
    linregress = None


DEPTH_MAP_RESOLUTION_XY = 2  # microns


def plot_depth_analysis(z: np.ndarray, lateral_resolution: float = DEPTH_MAP_RESOLUTION_XY,
                       output_folder: str = ".", y_limits_mm: tuple = None) -> None:
    """
    Create and save plots for horizontal cut, standard deviation, and max-min difference.
    """
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    
    os.makedirs(output_folder, exist_ok=True)
    
    rows, cols = z.shape
    
    # Limit analysis to specified y range
    if y_limits_mm is not None:
        y_min, y_max = y_limits_mm
        y_min_idx = max(0, int(y_min / (lateral_resolution * 1e-3)))
        y_max_idx = min(rows, int(y_max / (lateral_resolution * 1e-3)))
        z_limited = z[y_min_idx:y_max_idx, :]
    else:
        z_limited = z
    
    limited_rows = z_limited.shape[0]
    
    if limited_rows == 0:
        return
    
    # Horizontal cut at middle
    middle_y = limited_rows // 2
    horizontal_cut = z_limited[middle_y, :]
    
    # Statistics
    std_dev = np.std(z_limited, axis=0)
    max_min_diff = np.max(z_limited, axis=0) - np.min(z_limited, axis=0)
    
    x_positions = np.arange(cols) * lateral_resolution * 1e-3  # mm
    
    # Plotting
    plt.figure(figsize=(14, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(x_positions, horizontal_cut, label=f'Horizontal Cut at Y={middle_y * lateral_resolution * 1e-3:.2f} mm', color='blue')
    plt.title('Horizontal Cut')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Depth (Z values)')
    plt.grid()
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(x_positions, std_dev, label='Standard Deviation', color='orange')
    title_suffix = f' (Y limits: {y_limits_mm[0]} mm to {y_limits_mm[1]} mm)' if y_limits_mm else ''
    plt.title('Standard Deviation for Each X Position' + title_suffix)
    plt.xlabel('X Position (mm)')
    plt.ylabel('Standard Deviation (Z values)')
    plt.ylim(0.0, 15.0)
    plt.grid()
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(x_positions, max_min_diff, label='Max-Min Difference', color='green')
    plt.title('Max-Min Difference for Each X Position' + title_suffix)
    plt.xlabel('X Position (mm)')
    plt.ylabel('Max-Min Difference (Z values)')
    plt.ylim(0.0, 15.0)
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'depth_analysis.png'))
    plt.show()
    plt.close()
    
    # Save to CSV
    analysis_data = np.column_stack((x_positions, horizontal_cut, std_dev, max_min_diff))
    np.savetxt(os.path.join(output_folder, 'depth_analysis.csv'),
              analysis_data, delimiter=',',
              header='X Position (mm),Z Profile (µm),Standard Deviation (µm),Max-Min Difference (µm)',
              comments='')


def save_cuts(z: np.ndarray, output_folder: str = ".",
             lateral_resolution: float = DEPTH_MAP_RESOLUTION_XY,
             y_positions_mm: list = None, x_positions_mm: list = None) -> None:
    """
    Save horizontal and vertical cuts at specified positions.
    """
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    
    os.makedirs(output_folder, exist_ok=True)
    
    if y_positions_mm is None:
        y_positions_mm = []
    if x_positions_mm is None:
        x_positions_mm = []
    
    # Convert mm to pixel indices
    y_positions_idx = [int(y / (lateral_resolution * 1e-3)) for y in y_positions_mm]
    x_positions_idx = [int(x / (lateral_resolution * 1e-3)) for x in x_positions_mm]
    
    # Horizontal cuts
    for y_idx in y_positions_idx:
        if 0 <= y_idx < z.shape[0]:
            horizontal_cut = z[y_idx, :]
            x_pos = np.arange(z.shape[1]) * lateral_resolution * 1e-3
            
            plt.figure(figsize=(10, 5))
            plt.plot(x_pos, horizontal_cut, 
                    label=f'Horizontal Cut at Y={y_idx * lateral_resolution * 1e-3:.2f} mm', 
                    color='blue')
            plt.title('Horizontal Cut')
            plt.xlabel('X Position (mm)')
            plt.ylabel('Depth (Z values)')
            plt.grid()
            plt.legend()
            
            y_mm = y_idx * lateral_resolution * 1e-3
            plt.savefig(os.path.join(output_folder, f'horizontal_cut_y{y_mm:.2f}mm.png'))
            plt.close()
            
            np.savetxt(os.path.join(output_folder, f'horizontal_cut_y{y_mm:.2f}mm.csv'),
                      horizontal_cut, delimiter=',', header='Depth (Z values)', comments='')
    
    # Vertical cuts
    for x_idx in x_positions_idx:
        if 0 <= x_idx < z.shape[1]:
            vertical_cut = z[:, x_idx]
            y_pos = np.arange(z.shape[0]) * lateral_resolution * 1e-3
            
            plt.figure(figsize=(10, 5))
            plt.plot(y_pos, vertical_cut,
                    label=f'Vertical Cut at X={x_idx * lateral_resolution * 1e-3:.2f} mm',
                    color='green')
            plt.title('Vertical Cut')
            plt.xlabel('Y Position (mm)')
            plt.ylabel('Depth (Z values)')
            plt.grid()
            plt.legend()
            
            x_mm = x_idx * lateral_resolution * 1e-3
            plt.savefig(os.path.join(output_folder, f'vertical_cut_x{x_mm:.2f}mm.png'))
            plt.close()
            
            np.savetxt(os.path.join(output_folder, f'vertical_cut_x{x_mm:.2f}mm.csv'),
                      vertical_cut, delimiter=',', header='Depth (Z values)', comments='')


def find_best_worst_linearity(data: np.ndarray) -> dict:
    """
    Find rows and columns with best and worst linearity.
    Linearity measured using R-squared from linear regression.
    """
    if linregress is None:
        raise RuntimeError("scipy required for linearity analysis")
    
    rows, cols = data.shape
    
    best_row_idx = -1
    worst_row_idx = -1
    best_col_idx = -1
    worst_col_idx = -1
    
    best_row_r2 = -np.inf
    worst_row_r2 = np.inf
    best_col_r2 = -np.inf
    worst_col_r2 = np.inf
    
    # Check rows
    for row_idx in range(rows):
        x = np.arange(cols)
        y = data[row_idx, :]
        valid = np.isfinite(y)
        
        if np.sum(valid) < 2:
            continue
        
        _, _, r_value, _, _ = linregress(x[valid], y[valid])
        r_squared = r_value ** 2
        
        if r_squared > best_row_r2:
            best_row_r2 = r_squared
            best_row_idx = row_idx
        
        if r_squared < worst_row_r2:
            worst_row_r2 = r_squared
            worst_row_idx = row_idx
    
    # Check columns
    for col_idx in range(cols):
        x = np.arange(rows)
        y = data[:, col_idx]
        valid = np.isfinite(y)
        
        if np.sum(valid) < 2:
            continue
        
        _, _, r_value, _, _ = linregress(x[valid], y[valid])
        r_squared = r_value ** 2
        
        if r_squared > best_col_r2:
            best_col_r2 = r_squared
            best_col_idx = col_idx
        
        if r_squared < worst_col_r2:
            worst_col_r2 = r_squared
            worst_col_idx = col_idx
    
    return {
        'best_row_idx': best_row_idx, 'best_row_r2': best_row_r2,
        'worst_row_idx': worst_row_idx, 'worst_row_r2': worst_row_r2,
        'best_col_idx': best_col_idx, 'best_col_r2': best_col_r2,
        'worst_col_idx': worst_col_idx, 'worst_col_r2': worst_col_r2
    }
