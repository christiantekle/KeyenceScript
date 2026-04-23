"""
clean_depth_utils.py

Minimal utilities to:
 - load STL and PCD into Nx3 numpy arrays
 - convert Nx3 point clouds into a 2D depth map (gridded height image)
 - optional plane leveling (RANSAC) and interpolation of empty pixels


"""

import numpy as np

# Optional imports (used only when functions that rely on them are called)
try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    from scipy import stats
    from scipy.interpolate import griddata, RegularGridInterpolator
except Exception:
    stats = None
    griddata = None
    RegularGridInterpolator = None

try:
    from sklearn.linear_model import RANSACRegressor, LinearRegression
except Exception:
    RANSACRegressor = None
    LinearRegression = None

def stl_to_points(stl_path: str, sample_points: int = None) -> np.ndarray:
    """
    Load an STL file and return a Nx3 numpy array of 3D points.
    If sample_points is provided, the mesh will be uniformly sampled to that many points.
    Requires open3d.
    """
    if o3d is None:
        raise RuntimeError("open3d is required for stl_to_points. Install with `pip install open3d`.")
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if sample_points:
        pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
        pts = np.asarray(pcd.points, dtype=float)
    else:
        # Return triangle vertices (may contain duplicates)
        verts = np.asarray(mesh.vertices, dtype=float)
        pts = verts
    return pts

def pcd_to_points(pcd_path: str) -> np.ndarray:
    """
    Load a PCD (or other point cloud format supported by Open3D) and return Nx3 numpy array.
    Requires open3d.
    """
    if o3d is None:
        raise RuntimeError("open3d is required for pcd_to_points. Install with `pip install open3d`.")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=float)
    return pts

def fit_plane_ransac(points: np.ndarray, residual_threshold: float = 0.001, min_samples: int = 3):
    """
    Fit plane z = a*x + b*y + c using RANSAC on the given Nx3 points.
    Returns (a, b, c, inlier_mask)
    Requires scikit-learn.
    """
    if RANSACRegressor is None or LinearRegression is None:
        raise RuntimeError("scikit-learn is required for fit_plane_ransac. Install with `pip install scikit-learn`.")
    X = points[:, :2]
    y = points[:, 2]
    base = LinearRegression()
    ransac = RANSACRegressor(base_estimator=base, residual_threshold=residual_threshold, min_samples=min_samples)
    ransac.fit(X, y)
    coef = ransac.estimator_.coef_
    intercept = float(ransac.estimator_.intercept_)
    a, b = float(coef[0]), float(coef[1])
    inlier_mask = ransac.inlier_mask_ if hasattr(ransac, "inlier_mask_") else None
    return a, b, intercept, inlier_mask

def remove_plane(points: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Subtract plane z = a*x + b*y + c from points (in-place safe copy).
    Returns new Nx3 array with z' = z - (a*x + b*y + c)
    """
    pts = points.copy()
    pts[:, 2] = pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)
    return pts

def points_to_depth_map(points: np.ndarray,
                        resolution: float = 0.001,
                        agg: str = "max",
                        fill_method: str = None,
                        bounds: tuple = None):
    """
    Convert Nx3 points (x,y,z) into a 2D depth map (numpy array) by binning points into a grid.
    - resolution: grid cell size in same units as points (default 0.001)
    - agg: aggregation method for multiple points in same cell: "max". "min", "mean", "median"
    - fill_method: if 'griddata' will try to interpolate NaNs using scipy.interpolate.griddata (slow but smooth)
    - bounds: optional ((xmin, xmax), (ymin, ymax)). If None, computed from points.

    Returns:
      depth_map: 2D numpy array of shape (rows, cols) with dtype float64 (NaN for empty cells)
      x_edges: 1D array of x bin edges (length cols+1)
      y_edges: 1D array of y bin edges (length rows+1)
    """
    if points.size == 0:
        raise ValueError("Empty points array")

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    if bounds is None:
        xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    else:
        (xmin, xmax), (ymin, ymax) = bounds

    # number of bins along each axis
    nx = max(1, int(np.ceil((xmax - xmin) / resolution)))
    ny = max(1, int(np.ceil((ymax - ymin) / resolution)))

    # Use scipy.stats.binned_statistic_2d if available (fast, vectorized)
    if stats is not None:
        statistic = agg if agg in ("mean", "median", "count") else "max"
        res = stats.binned_statistic_2d(xs, ys, zs, statistic=statistic, bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]])
        # res.statistic shape is (nx, ny); we want (ny, nx) so transpose and flip y
        stat = res.statistic.T  # rows = y bins, cols = x bins
        x_edges = res.x_edge
        y_edges = res.y_edge
        depth_map = stat.astype(float)
        # binned_statistic_2d returns NaN for empty bins for 'max'/'mean' etc.
    else:
        # Fallback naive implementation (may be slower for large clouds)
        x_idx = np.clip(((xs - xmin) / resolution).astype(int), 0, nx - 1)
        y_idx = np.clip(((ys - ymin) / resolution).astype(int), 0, ny - 1)
        depth_map = np.full((ny, nx), np.nan, dtype=float)
        if agg == "max":
            for x_i, y_i, z in zip(x_idx, y_idx, zs):
                if np.isnan(z):
                    continue
                cur = depth_map[y_i, x_i]
                if np.isnan(cur) or z > cur:
                    depth_map[y_i, x_i] = z
        elif agg == "min":
              for x_i, y_i, z in zip(x_idx, y_idx, zs):
                 if np.isnan(z):
                     continue
                 cur = depth_map[y_i, x_i]
                 if np.isnan(cur) or z < cur:
                    depth_map[y_i, x_i] = z
        elif agg == "mean" or agg == "median":
            from collections import defaultdict
            cells = defaultdict(list)
            for x_i, y_i, z in zip(x_idx, y_idx, zs):
                if np.isnan(z): continue
                cells[(y_i, x_i)].append(z)
            for (y_i, x_i), vals in cells.items():
                if agg == "mean":
                    depth_map[y_i, x_i] = float(np.mean(vals))
                else:
                    depth_map[y_i, x_i] = float(np.median(vals))
        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)

    # Optionally fill NaNs using griddata
    if fill_method == "griddata":
        if griddata is None:
            raise RuntimeError("scipy.interpolate.griddata required for fill_method='griddata' (install scipy).")
        # Create coordinate mesh for valid points in depth_map
        Xc = (x_edges[:-1] + x_edges[1:]) / 2.0
        Yc = (y_edges[:-1] + y_edges[1:]) / 2.0
        XX, YY = np.meshgrid(Xc, Yc)
        mask = ~np.isnan(depth_map)
        if np.any(mask):
            known_points = np.column_stack((XX[mask], YY[mask]))
            known_values = depth_map[mask]
            missing_mask = np.isnan(depth_map)
            missing_points = np.column_stack((XX[missing_mask], YY[missing_mask]))
            if missing_points.size > 0:
                filled = griddata(known_points, known_values, missing_points, method="nearest")
                depth_map[missing_mask] = filled

    return depth_map, x_edges, y_edges


def scale_depth_map_xy(
    depth_map: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0,
    method: str = "linear",
    fill_value: float = np.nan):
    
    """
    Scale a 2D depth map in X and/or Y direction (geometric scaling).

    Parameters
    ----------
    depth_map : (H, W) array
        Input depth map
    x_edges, y_edges : bin edges as returned by points_to_depth_map
    scale_x, scale_y : scaling factors (>1 stretches, <1 shrinks)
    method : 'linear' or 'nearest'
    fill_value : value outside original domain

    Returns
    -------
    depth_map_scaled : 2D array
    x_edges_scaled : 1D array
    y_edges_scaled : 1D array
    """

    if depth_map.ndim != 2:
        raise ValueError("depth_map must be 2D")

    # Cell centers
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Interpolator (NaNs allowed)
    interp = RegularGridInterpolator(
        (y_centers, x_centers),
        depth_map,
        method=method,
        bounds_error=False,
        fill_value=fill_value,
    )

    # New scaled grid
    x_centers_new = x_centers * scale_x
    y_centers_new = y_centers * scale_y

    XXn, YYn = np.meshgrid(x_centers_new, y_centers_new)
    query_points = np.column_stack([YYn.ravel(), XXn.ravel()])

    depth_map_scaled = interp(query_points).reshape(
        len(y_centers_new), len(x_centers_new)
    ) 
    
    
    if (scale_z != 1.0):
      depth_map_scaled = depth_map_scaled * scale_z

    # Scale edges accordingly
    x_edges_scaled = x_edges * scale_x
    y_edges_scaled = y_edges * scale_y

    return depth_map_scaled, x_edges_scaled, y_edges_scaled

# Example utility: high-level pipeline
def load_and_grid(input_path: str, kind: str = "pcd", resolution: float = 0.001, sample_points: int = None):
    """
    High-level convenience function:
      kind: 'pcd' or 'stl'
      resolution: grid cell size
    """
    if kind.lower() == "pcd":
        pts = pcd_to_points(input_path)
    elif kind.lower() == "stl":
        pts = stl_to_points(input_path, sample_points=sample_points)
    else:
        raise ValueError("kind must be 'pcd' or 'stl'")
    depth_map, x_edges, y_edges = points_to_depth_map(pts, resolution=resolution, agg="max")
    return depth_map, x_edges, y_edges

""" if __name__ == "__main__":
    # Demo: print 2D array for a fixed 2mm x 2mm square
    import sys
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    if len(sys.argv) < 2:
        print("Usage: python clean_depth_utils.py <pcd_or_stl_path>")
        sys.exit(0)

    path = sys.argv[1]
    kind = "pcd" if path.lower().endswith(".pcd") else "stl"

    # Load points
    if kind == "pcd":
        pts = pcd_to_points(path)
    else:
        pts = stl_to_points(path, sample_points=50000)

    # Print bounding box info to help adjust bounds
    print("X range:", pts[:,0].min(), "to", pts[:,0].max())
    print("Y range:", pts[:,1].min(), "to", pts[:,1].max())
    print("Z range:", pts[:,2].min(), "to", pts[:,2].max())

    # Adjust bounds depending on units:
    # If data is in meters, 0.002 = 2 mm.
    # If data is in millimeters, use (0,2) instead of (0,0.002).

    # Generate depth map for 2mm x 2mm square
    dm, xe, ye = points_to_depth_map(
        pts,
        resolution=0.00001,  # 0.01 mm per pixel
        bounds=((0, 0.002), (0, 0.002)),
        agg="max"
    )

    print("Depth map shape:", dm.shape)
    print("Displaying 2D Depth Array (2mm x 2mm region) in a popup window...")

    plt.imshow(dm, cmap="viridis", origin="lower",
               extent=[xe[0], xe[-1], ye[0], ye[-1]])
    plt.colorbar(label="Depth (Z)")
    plt.title("2D Depth Array (2mm x 2mm region)")
    plt.show()

    # print(np.array2string(dm, precision=3, suppress_small=True, max_line_width=120)) """

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    if len(sys.argv) < 2:
        print("Usage: python clean_depth_utils.py <pcd_or_stl_path>")
        sys.exit(0)

    path = sys.argv[1]
    kind = "pcd" if path.lower().endswith(".pcd") else "stl"

    # Load points
    if kind == "pcd":
        pts = pcd_to_points(path)
    else:
        pts = stl_to_points(path, sample_points=50000)

    # Print bounding box info
    print("X range:", pts[:,0].min(), "to", pts[:,0].max())
    print("Y range:", pts[:,1].min(), "to", pts[:,1].max())
    print("Z range:", pts[:,2].min(), "to", pts[:,2].max())

    # Generate depth map for the full cloud (no 2mm box)
    dm, xe, ye = points_to_depth_map(
        pts,
        resolution=0.02,  # coarser resolution for large cloud
        agg="max"
    )

    print("Depth map shape:", dm.shape)
    print("Displaying full 2D Depth Array in a popup window...")

    plt.imshow(dm, cmap="viridis", origin="lower",
               extent=[xe[0], xe[-1], ye[0], ye[-1]])
    plt.colorbar(label="Depth (Z)")
    plt.title("2D Depth Array (full cloud)")
    plt.show()