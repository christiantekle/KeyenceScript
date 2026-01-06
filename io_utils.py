"""
io_utils.py

File I/O utilities for loading and saving point clouds and depth maps.
"""

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

try:
    import open3d as o3d
except Exception:
    o3d = None


def load_stl_points(stl_filename: str, sample_points: int = None) -> np.ndarray:
    """
    Load STL file and return vertices as Nx3 array.
    If sample_points is specified, uniformly sample the mesh.
    """
    if o3d is None:
        raise RuntimeError("open3d required for load_stl_points")
    
    mesh = o3d.io.read_triangle_mesh(stl_filename)
    
    if sample_points:
        pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
        pts = np.asarray(pcd.points)
    else:
        pts = np.asarray(mesh.vertices)
        pts = mesh.sample_points_uniformly(number_of_points=pts.size ) #* 30)
        pts = np.asarray(pts)
    
    return pts



def load_pcd_points(pcd_filename: str) -> np.ndarray:
    """Load PCD file and return points as Nx3 array."""
    if o3d is None:
        raise RuntimeError("open3d required for load_pcd_points")
    
    pcd = o3d.io.read_point_cloud(pcd_filename)
    return np.asarray(pcd.points)


def load_depth_map(filename: str) -> np.ndarray:
    """Load depth map from .npy file."""
    return np.load(filename)
    

def read_columns(file_path: str, first_col: int = 1, second_col: int = 2) -> np.ndarray:
    """
    Read two columns from a whitespace-delimited file.
    Skips the first line (header).
    
    Returns: 2xN array where first row is first column, second row is second column
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    first = []
    second = []
    
    for line in lines[1:]:  # Skip header
        parts = line.strip().split()
        if len(parts) > max(first_col, second_col):
            first.append(float(parts[first_col]))
            second.append(float(parts[second_col]))
    
    return np.array([first, second])


def save_to_csv(filename: str, data: np.ndarray, formatting = '%s') -> None:
    """Save numpy array to CSV file."""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    np.savetxt(fname=filename, X=data, delimiter=';', fmt=formatting)
    
    

def save_depth_map(filename: str, data: np.ndarray) -> None:
    """Save depth map to .npy file."""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    np.save(filename, data)

def save_to_stl(depthmap, resolution=0.01, poisson_depth=9, poisson_width=0, 
                fileName='', diameter=None, theta_max=None):
    """
    Convert a depth map to a TriangleMesh using Poisson surface reconstruction.
    Supports flat or cylindrical projection. 
    If diameter is set, the Depthmap is mapped onto a cylinder only over its actual width.

    Args:
        depthmap (np.ndarray): 2D array representing z-values of a surface.
        resolution (float): Scale factor for x/y coordinates (default: 0.01).
        poisson_depth (int): Depth for Poisson reconstruction (default: 9).
        poisson_width (int): Width parameter for Poisson reconstruction (default: 0).
        is_closed (bool): If True, attempts to close the mesh (currently placeholder).
        fileName (str): If non-empty, saves mesh to this STL file.
        diameter (float or None): If set, projects the depthmap onto a cylinder.
        theta_max (float or None): Max angle in radians for cylinder coverage. If None, 
                                   calculates automatically from depthmap width and resolution.

    Returns:
        o3d.geometry.TriangleMesh: The reconstructed mesh.
    """

    depthmap = depthmap.astype(np.float32)
    H, W = depthmap.shape

    if (diameter is not None) and (diameter > 0):
        H, W = depthmap.shape
        radius = diameter / 2.0
        
        # Winkel über H (wir wickeln die Höhe um den Umfang)
        theta_max = H * resolution / radius
        theta = np.linspace(0, theta_max, H, endpoint=False)
        
        theta_grid, x_grid = np.meshgrid(theta, np.arange(W) * resolution)
        
        # depthmap transponieren, damit H ↔ Umfang passt
        dm = depthmap.T
        
        X = x_grid   # Zylinderachse
        Y = (radius + dm) * np.cos(theta_grid)
        Z = (radius + dm) * np.sin(theta_grid)
        
        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])       
    else:
        # Flache Projektion
        y_idx, x_idx = np.indices((H, W))
        points = np.column_stack([
            x_idx.flatten() * resolution,
            y_idx.flatten() * resolution,
            depthmap.flatten()
        ])

    # Open3D PointCloud erstellen
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Normals schätzen
    pcd.estimate_normals()

    # Farben basierend auf z-Höhe
    norm = plt.Normalize(points[:, 2].min(), points[:, 2].max())
    cmap = cm.jet
    colors = cmap(norm(points[:, 2]))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Poisson-Rekonstruktion
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=poisson_width
    )

    # Mesh bereinigen
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    # Crop auf Bounding Box der Punktwolke
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Speichern, falls gewünscht
    if fileName:
        o3d.io.write_triangle_mesh(fileName, mesh, print_progress=False)

    return mesh


# def save_to_stl(depthmap, resolution=0.01, poisson_depth=9, poisson_width=0, is_closed=False, fileName=''):
#     """
#     Convert a PointCloud to a TriangleMesh using Poisson surface reconstruction.

#     Args:
#         point_cloud (o3d.geometry.PointCloud): The input point cloud.
#         poisson_depth (int): Depth parameter for Poisson surface reconstruction (default: 9).
#         poisson_width (int): Width parameter for Poisson surface reconstruction (default: 0).

#     Returns:
#         o3d.geometry.TriangleMesh: The converted mesh.
#     """
    
#     dim = np.shape(depthmap)

#     # Generate grid for x- & y-coordinats
#     x_grid, y_grid = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))
    
#     # Calculate point cloud
#     p = np.column_stack([x_grid.flatten() * resolution, y_grid.flatten() * resolution, depthmap.flatten()])
#     pcd = o3d.geometry.PointCloud()
    
#     # Sweep memory
#     del x_grid, y_grid
    
#     # Befülle die Punktwolke
#     pcd.points = o3d.utility.Vector3dVector(p)
    
#     # Perform Poisson surface reconstruction
#     pcd.estimate_normals()
    
#     # Calculate colors based on z-height
#     norm = plt.Normalize(p[:, 2].min(), p[:, 2].max())
#     cmap = cm.jet
#     colors = cmap(norm(p[:, 2]))
#     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
   
#     # Sweep memory
#     del p, norm
   
   
#     # Create a blank mesh
#     mesh_dat = o3d.geometry.TriangleMesh()

#     # Perform Poisson surface reconstruction
#     mesh_dat, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         pcd, depth=poisson_depth, width=poisson_width)
    
#     # Extract the triangles from the mesh
#     mesh_dat.compute_vertex_normals()
#     mesh_dat.remove_duplicated_triangles()
#     mesh_dat.remove_duplicated_vertices()
#     mesh_dat.remove_non_manifold_edges()
    
#     # Remove extrapolations of Poisson algorithm
#     bbox = pcd.get_axis_aligned_bounding_box()
#     mesh_dat = mesh_dat.crop(bbox)
    
       
#     if fileName!='':
#       o3d.io.write_triangle_mesh(mesh = mesh_dat,filename = fileName, print_progress = False)

    
#     return mesh_dat
