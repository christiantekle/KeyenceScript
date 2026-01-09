"""
visualization.py

Plotting and visualization utilities for depth maps.
"""

import os
import numpy as np
import datetime


try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except Exception:
    plt = None



def plot_depth_map(depth_map: np.ndarray, title: str = "depth_map",
                  z_min: float = None, z_max: float = None,
                  lateral_resolution: float = 0.01,
                  output_folder: str = ".") -> None:
    """Plot depth map and save as PNG."""
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    
    x = np.arange(depth_map.shape[1]) * lateral_resolution  # to mm
    y = np.arange(depth_map.shape[0]) * lateral_resolution
    
    plt.figure()
    im = plt.imshow(depth_map, extent=[x.min(), x.max(), y.min(), y.max()],
                   vmin=z_min, vmax=z_max, origin='lower', cmap='jet', aspect='equal')
    plt.colorbar(im, label='Z [mm]')
    plt.title(title)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{title}.png"))
    plt.show()
    plt.close()


def plot_3d_depth_maps(depth_maps: list, lateral_resolution: float = 0.01,
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
    x = np.arange(nx) * lateral_resolution   # mm
    y = np.arange(ny) * lateral_resolution 
    X, Y = np.meshgrid(x, y)
    
    for i, Z in enumerate(depth_maps):
        alpha = alphas[i] if i < len(alphas) else 1.0
        surf = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=alpha,
                              edgecolor='none', antialiased=True)
    
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title('3D Depth Map')
    
    plt.tight_layout()
    plt.show()


def plot_depth_map_with_cuts(z: np.ndarray, lateral_resolution: float) -> None:
    """
    Interactive depth map viewer. Click to show X/Y cuts.
    """
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    
    x = np.arange(z.shape[1]) * lateral_resolution
    y = np.arange(z.shape[0]) * lateral_resolution
    
    fig, (ax_main, ax_xcut, ax_ycut) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Main heatmap
    im = ax_main.imshow(z, extent=[x.min() , x.max() , 
                                   y.min() , y.max() ],
                        origin='lower', cmap='jet', aspect='equal')
    fig.colorbar(im, ax=ax_main, label='Height (mm)')
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
        x_mm = lateral_resolution * ix 
        y_mm = lateral_resolution * iy 
        
        ax_xcut.clear()
        ax_xcut.plot(x , z[iy, :],  color='red')
        ax_xcut.set_title(f'Horizontal Cut at Y = {y_mm:.2f} mm')
        ax_xcut.set_xlabel('X (mm)')
        ax_xcut.set_ylabel('Height (mm)')
        
        ax_ycut.clear()
        ax_ycut.plot(y, z[:, ix], color='blue')
        ax_ycut.set_title(f'Vertical Cut at X = {x_mm:.2f} mm')
        ax_ycut.set_xlabel('Y (mm)')
        ax_ycut.set_ylabel('Height (mm)')
        
        fig.canvas.draw()
    
    def onclick(event):
        if event.inaxes != ax_main or event.xdata is None or event.ydata is None:
            return
        
        x_click = event.xdata 
        y_click = event.ydata 
        ix = int(round(x_click / lateral_resolution))
        iy = int(round(y_click / lateral_resolution))
        
        if 0 <= ix < z.shape[1] and 0 <= iy < z.shape[0]:
            update_cuts(ix, iy)
            
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show(block=True)
    
    



def export_current_cuts(z_maps, x, y, ix, iy, lateral_resolution, out_dir=None):
    """Export horizontal/vertical cuts at ix, iy for all maps."""
    if out_dir is None:
        out_dir = f"cuts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    x_mm = ix * lateral_resolution
    y_mm = iy * lateral_resolution

    for i, z in enumerate(z_maps):
        # Horizontal cut
        if 0 <= iy < z.shape[0]:
            values = z[iy, :]
            coord = x[:len(values)]
            mask = np.isfinite(values)
            data = np.column_stack((coord[mask], values[mask]))
            fname = f"map{i}_horizontal_Y{y_mm:.3f}mm.csv"
            np.savetxt(
                os.path.join(out_dir, fname),
                data,
                delimiter=",",
                header="x_mm,height_mm",
                comments=""
            )

        # Vertical cut
        if 0 <= ix < z.shape[1]:
            values = z[:, ix]
            coord = y[:len(values)]
            mask = np.isfinite(values)
            data = np.column_stack((coord[mask], values[mask]))
            fname = f"map{i}_vertical_X{x_mm:.3f}mm.csv"
            np.savetxt(
                os.path.join(out_dir, fname),
                data,
                delimiter=",",
                header="y_mm,height_mm",
                comments=""
            )

def plot_mult_depth_map_with_cuts(z_maps, names=None, lateral_resolution=0.01):
    """Interactive depth map viewer with cuts and export."""
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    if len(z_maps) == 0:
        raise ValueError("z_maps must not be empty")
    
    z_maps = [np.asarray(z) for z in z_maps]
    if names is None:
        names = [f"Depth Map {i}" for i in range(len(z_maps))]
    elif len(names) != len(z_maps):
        raise ValueError("names must have same length as z_maps")

    plt.ion()
    
    # Dimensions for main map (use largest map for consistent axes)
    max_H = max(z.shape[0] for z in z_maps)
    max_W = max(z.shape[1] for z in z_maps)
    x_full = np.arange(max_W) * lateral_resolution
    y_full = np.arange(max_H) * lateral_resolution

    fig, (ax_main, ax_xcut, ax_ycut) = plt.subplots(1, 3, figsize=(18, 6))
    active_idx = 0
    colors = plt.cm.tab10.colors
    last_ix, last_iy = None, None

    # --- Helper: NaN-safe plot ---
    def nan_safe_plot(ax, coord, values, **kwargs):
        values = np.asarray(values)
        coord = np.asarray(coord)
        min_len = min(len(coord), len(values))
        coord = coord[:min_len]
        values = values[:min_len]
        mask = np.isfinite(values)
        if np.any(mask):
            ax.plot(coord[mask], values[mask], **kwargs)

    # --- Update cuts ---
    def update_cuts(ix, iy):
        nonlocal last_ix, last_iy
        last_ix, last_iy = ix, iy
        x_mm = ix * lateral_resolution
        y_mm = iy * lateral_resolution

        ax_xcut.clear()
        ax_ycut.clear()

        for i, z in enumerate(z_maps):
            c = colors[i % len(colors)]
            label = names[i]
            x = np.arange(z.shape[1]) * lateral_resolution
            y = np.arange(z.shape[0]) * lateral_resolution

            if 0 <= iy < z.shape[0]:
                nan_safe_plot(ax_xcut, x, z[iy, :], color=c, label=label)
            if 0 <= ix < z.shape[1]:
                nan_safe_plot(ax_ycut, y, z[:, ix], color=c, label=label)

        ax_xcut.set_title(f"Horizontal Cut @ Y = {y_mm:.2f} mm")
        ax_xcut.set_xlabel("X (mm)")
        ax_xcut.set_ylabel("Height (mm)")
        ax_xcut.legend()

        ax_ycut.set_title(f"Vertical Cut @ X = {x_mm:.2f} mm")
        ax_ycut.set_xlabel("Y (mm)")
        ax_ycut.set_ylabel("Height (mm)")
        ax_ycut.legend()

        # Update crosshair
        vline.set_xdata([x_mm, x_mm])
        hline.set_ydata([y_mm, y_mm])
        fig.canvas.draw_idle()

    # --- Update main image ---
    def update_main_image():
        im.set_data(z_maps[active_idx])
        ax_main.set_title(f"{names[active_idx]} (scroll to change, press 'E' for CSV export) – click to cut")
        im.autoscale()
        fig.canvas.draw_idle()

    # --- Main image ---
    im = ax_main.imshow(
        z_maps[active_idx],
        extent=[0, max_W*lateral_resolution, 0, max_H*lateral_resolution],
        origin="lower",
        cmap="jet",
        aspect="equal"
    )
    fig.colorbar(im, ax=ax_main, label="Height (mm)")

    ax_main.set_xlabel("X (mm)")
    ax_main.set_ylabel("Y (mm)")
    ax_main.set_title(f"{names[active_idx]} (scroll to change, press 'E' for CSV export) – click to cut")

    vline = ax_main.axvline(np.nan, color="white", lw=1)
    hline = ax_main.axhline(np.nan, color="white", lw=1)

    # --- Cut axes ---
    ax_xcut.set_xlabel("X (mm)")
    ax_xcut.set_ylabel("Height (mm)")
    ax_ycut.set_xlabel("Y (mm)")
    ax_ycut.set_ylabel("Height (mm)")

    # --- Event callbacks ---
    def onclick(event):
        if event.inaxes != ax_main or event.xdata is None or event.ydata is None:
            return
        ix = int(round(event.xdata / lateral_resolution))
        iy = int(round(event.ydata / lateral_resolution))
        if 0 <= ix < max_W and 0 <= iy < max_H:
            update_cuts(ix, iy)

    def onscroll(event):
        nonlocal active_idx
        if event.button == "up":
            active_idx = (active_idx + 1) % len(z_maps)
        elif event.button == "down":
            active_idx = (active_idx - 1) % len(z_maps)
        update_main_image()
        if last_ix is not None and last_iy is not None:
            update_cuts(last_ix, last_iy)

    def onkey(event):
        if event.key == "e":
            if last_ix is None or last_iy is None:
                print("No cut selected yet – click on the map first.")
                return
            export_current_cuts(
                z_maps=z_maps,
                x=x_full,
                y=y_full,
                ix=last_ix,
                iy=last_iy,
                lateral_resolution=lateral_resolution
            )
            print(f"Exported cuts at ix={last_ix}, iy={last_iy}")

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("scroll_event", onscroll)
    fig.canvas.mpl_connect("key_press_event", onkey)

    plt.tight_layout()
    plt.show(block=True)
    plt.ioff()


        
    
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
