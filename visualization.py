"""
visualization.py

Plotting and visualization utilities for depth maps.
"""

import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except Exception:
    plt = None


DEPTH_MAP_RESOLUTION_XY = 2  # microns


def plot_depth_map(depth_map: np.ndarray, title: str = "depth_map",
                  z_min: float = None, z_max: float = None,
                  lateral_resolution: float = DEPTH_MAP_RESOLUTION_XY,
                  output_folder: str = ".") -> None:
    """Plot depth map and save as PNG."""
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    
    x = np.arange(depth_map.shape[1]) * lateral_resolution * 1e-3  # to mm
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


def plot_3d_depth_maps(depth_maps: list, lateral_resolution: float = DEPTH_MAP_RESOLUTION_XY,
                      alphas: list = None, colormap: str = 'jet') -> None:
    """Interactive 3D surface plot of one or more depth maps."""
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    
    if not isinstance(depth_maps, list):
        depth_maps = [depth_maps]
    
    if alphas is None:
        alphas = [1.0] * len(depth_maps)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ny, nx = depth_maps[0].shape
    x = np.arange(nx) * lateral_resolution * 1e-3  # mm
    y = np.arange(ny) * lateral_resolution * 1e-3
    X, Y = np.meshgrid(x, y)
    
    for i, Z in enumerate(depth_maps):
        alpha = alphas[i] if i < len(alphas) else 1.0
        surf = ax.plot_surface(X, Y, Z * 1e-3, cmap=colormap, alpha=alpha,
                              edgecolor='none', antialiased=True)
    
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title('3D Depth Map')
    
    plt.tight_layout()
    plt.show()


def plot_depth_map_with_cuts(z: np.ndarray, lateral_resolution: float = DEPTH_MAP_RESOLUTION_XY) -> None:
    """
    Interactive depth map viewer. Click to show X/Y cuts.
    """
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    
    x = np.arange(z.shape[1]) * lateral_resolution
    y = np.arange(z.shape[0]) * lateral_resolution
    
    fig, (ax_main, ax_xcut, ax_ycut) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Main heatmap
    im = ax_main.imshow(z, extent=[x.min() * 1e-3, x.max() * 1e-3, 
                                   y.min() * 1e-3, y.max() * 1e-3],
                       origin='lower', cmap='jet', aspect='equal')
    fig.colorbar(im, ax=ax_main, label='Height (µm)')
    ax_main.set_xlabel('X (mm)')
    ax_main.set_ylabel('Y (mm)')
    ax_main.set_title('Click on map to view X/Y cuts')
    
    ax_xcut.set_title('Horizontal Cut (Y constant)')
    ax_xcut.set_xlabel('X (mm)')
    ax_xcut.set_ylabel('Height (mm)')
    
    ax_ycut.set_title('Vertical Cut (X constant)')
    ax_ycut.set_xlabel('Y (mm)')
    ax_ycut.set_ylabel('Height (mm)')
    
    def update_cuts(ix, iy):
        x_mm = lateral_resolution * ix * 1e-3
        y_mm = lateral_resolution * iy * 1e-3
        
        ax_xcut.clear()
        ax_xcut.plot(x * 1e-3, z[iy, :] * 1e-3, color='red')
        ax_xcut.set_title(f'Horizontal Cut at Y = {y_mm:.2f} mm')
        ax_xcut.set_xlabel('X (mm)')
        ax_xcut.set_ylabel('Height (mm)')
        
        ax_ycut.clear()
        ax_ycut.plot(y * 1e-3, z[:, ix] * 1e-3, color='blue')
        ax_ycut.set_title(f'Vertical Cut at X = {x_mm:.2f} mm')
        ax_ycut.set_xlabel('Y (mm)')
        ax_ycut.set_ylabel('Height (mm)')
        
        fig.canvas.draw()
    
    def onclick(event):
        if event.inaxes != ax_main or event.xdata is None or event.ydata is None:
            return
        
        x_click = event.xdata * 1e3
        y_click = event.ydata * 1e3
        ix = int(round(x_click / lateral_resolution))
        iy = int(round(y_click / lateral_resolution))
        
        if 0 <= ix < z.shape[1] and 0 <= iy < z.shape[0]:
            update_cuts(ix, iy)
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show(block=True)


def plot_histogram_cdf(dm_part, bins=50, title='Histogram and Cumulative Distribution Function'):
    """
    Plots the histogram and cumulative distribution function (CDF) of a depth map segment.

    Args:
        dm_part (np.ndarray): 2D array or segment of a depth map.
        bins (int): Number of histogram bins.
        title (str): Plot title.
    """
    # Only valid (finite) values
    vals = dm_part[np.isfinite(dm_part)]

    # Compute histogram
    hist, bin_edges = np.histogram(vals, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute CDF
    cdf = np.cumsum(hist) / np.sum(hist)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8,5))

    # Histogram (bars) on left Y-axis
    ax1.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0],
            alpha=0.6, color='skyblue', label='Histogram')
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # CDF on right Y-axis
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, cdf, color='red', marker='o', label='CDF')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_ylim(0, 1)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(title)
    plt.tight_layout()
    plt.show()    
