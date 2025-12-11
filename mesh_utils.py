"""
mesh_utils.py

Mesh generation and parallel processing utilities.
"""

import numpy as np
import multiprocessing

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except Exception:
    plt = None
    cm = None


DEPTH_MAP_RESOLUTION_XY = 2  # microns


def depth_map_to_mesh(depth_map: np.ndarray,
                      poisson_depth: int = 9,
                      poisson_width: int = 0,
                      save_filename: str = '',
                      resolution_xy: float = DEPTH_MAP_RESOLUTION_XY) -> object:
    """
    Convert a depth map to a 3D triangle mesh using Poisson surface reconstruction.
    
    Args:
        depth_map: 2D array of depth values (in microns)
        poisson_depth: Depth parameter for Poisson reconstruction (default: 9)
        poisson_width: Width parameter for Poisson reconstruction (default: 0)
        save_filename: Optional filename to save the mesh (e.g., 'output.stl')
        resolution_xy: Lateral resolution in microns (default: 2)
    
    Returns:
        open3d.geometry.TriangleMesh object
    """
    if o3d is None:
        raise RuntimeError("open3d required for depth_map_to_mesh")
    
    print("Creating point cloud from depth map...")
    
    # Scale factors
    scale_xy = 1000.0 / resolution_xy
    dim = np.shape(depth_map)
    
    # Create coordinate grid
    x_grid, y_grid = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))
    
    # Build point cloud (convert back to meters)
    points = np.column_stack([
        x_grid.flatten() / scale_xy,
        y_grid.flatten() / scale_xy,
        depth_map.flatten() / 1000.0
    ])
    
    # Remove NaN points
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    print("Estimating normals...")
    pcd.estimate_normals()
    
    # Calculate colors based on z-height
    if cm is not None and plt is not None:
        norm = plt.Normalize(points[:, 2].min(), points[:, 2].max())
        cmap = cm.jet
        colors = cmap(norm(points[:, 2]))
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    print("Performing Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=poisson_width
    )
    
    print("Cleaning mesh...")
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # Crop to original bounding box (remove Poisson extrapolation)
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    
    # Save if filename provided
    if save_filename:
        print(f"Saving mesh to {save_filename}...")
        o3d.io.write_triangle_mesh(mesh, save_filename, print_progress=False)
        print("Mesh saved successfully")
    
    print("Mesh creation complete")
    return mesh


def parallel_apply_along_axis(func1d, axis: int, arr: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Like numpy.apply_along_axis(), but uses multiple CPU cores for parallel processing.
    
    Args:
        func1d: Function to apply (must take 1D array as input)
        axis: Axis along which to apply the function
        arr: Input array
        *args, **kwargs: Additional arguments passed to func1d
    
    Returns:
        Array with func1d applied along the specified axis
    
    Example:
        # Apply interpolation along rows in parallel
        result = parallel_apply_along_axis(interpolate_nans_1d, 1, depth_map)
    """
    # Effective axis for processing
    effective_axis = 1 if axis == 0 else axis
    
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)
    
    # Split array into chunks for each CPU core
    num_cores = multiprocessing.cpu_count()
    chunks = [
        (func1d, effective_axis, sub_arr, args, kwargs)
        for sub_arr in np.array_split(arr, num_cores)
    ]
    
    # Process chunks in parallel
    with multiprocessing.Pool() as pool:
        individual_results = pool.map(_unpacking_apply_along_axis, chunks)
    
    return np.concatenate(individual_results)


def _unpacking_apply_along_axis(all_args):
    """
    Helper function for parallel_apply_along_axis.
    Unpacks arguments and applies function.
    """
    func1d, axis, arr, args, kwargs = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def visualize_mesh(mesh, window_name: str = "3D Mesh") -> None:
    """
    Visualize a mesh using Open3D viewer.
    
    Args:
        mesh: open3d.geometry.TriangleMesh object
        window_name: Title for the visualization window
    """
    if o3d is None:
        raise RuntimeError("open3d required for mesh visualization")
    
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=window_name,
        width=1024,
        height=768
    )


# Example usage
if __name__ == "__main__":
    # Example: Create mesh from depth map
    # depth_map = np.random.rand(500, 500) * 100  # Random depth map
    # mesh = depth_map_to_mesh(depth_map, save_filename="output.stl")
    # visualize_mesh(mesh)
    
    print("mesh_utils.py - Run from another script to use these functions")
