"""
processing_pipeline.py

High-level processing pipelines for converting point clouds to depth maps.
Uses clean_depth_utils.py for core conversion functionality.
"""

import numpy as np

# Import from clean_depth_utils for point cloud operations
try:
    from clean_depth_utils import pcd_to_points, stl_to_points, points_to_depth_map
except ImportError:
    # Fallback error message if clean_depth_utils is not available
    def _missing_module_error():
        raise ImportError("clean_depth_utils.py is required for processing_pipeline. "
                         "Make sure it's in your Python path.")
    
    pcd_to_points = _missing_module_error
    stl_to_points = _missing_module_error
    points_to_depth_map = _missing_module_error


DEPTH_MAP_RESOLUTION_XY = 2  # microns


def load_and_process_pcd(pcd_filename: str,
                         resolution: float = DEPTH_MAP_RESOLUTION_XY,
                         agg: str = "max",
                         fill_method: str = None,
                         bounds: tuple = None,
                         crop_z_min: float = None,
                         crop_z_max: float = None,
                         initial_crop_z_max: float = None,
                         manual_z_offset: float = None) -> np.ndarray:
    """
    High-level function to load PCD file and convert to depth map.
    
    Args:
        pcd_filename: Path to PCD file
        resolution: Grid cell size in same units as points (default: 0.001)
        agg: Aggregation method: "max", "mean", "median"
        fill_method: Optional 'griddata' for interpolation
        bounds: Optional ((xmin, xmax), (ymin, ymax))
        crop_z_min: Minimum z value to keep (after conversion)
        crop_z_max: Maximum z value to keep (after conversion)
        initial_crop_z_max: Maximum z value for initial filtering (meters)
        manual_z_offset: Manual z offset to apply (after conversion)
    
    Returns:
        2D depth map array
    """
    # Load point cloud
    points = pcd_to_points(pcd_filename)
    
    # Initial z cropping (in original units - typically meters)
    if initial_crop_z_max is not None:
        points = points[points[:, 2] <= initial_crop_z_max]
    
    # Remove NaN points
    points = points[~np.isnan(points).any(axis=1)]
    
    if points.size == 0:
        raise ValueError("No valid points remaining after filtering")
    
    # Convert to depth map
    depth_map, _, _ = points_to_depth_map(
        points,
        resolution=resolution,
        agg=agg,
        fill_method=fill_method,
        bounds=bounds
    )
    
    # Post-processing: z cropping (in depth map units - typically microns)
    if crop_z_min is not None:
        depth_map[depth_map < crop_z_min] = np.nan
    
    if crop_z_max is not None:
        depth_map[depth_map > crop_z_max] = np.nan
    
    # Apply manual offset
    if manual_z_offset is not None:
        depth_map = depth_map + manual_z_offset
    
    return depth_map


def load_and_process_stl(stl_filename: str,
                         sample_points: int = None,
                         resolution: float = DEPTH_MAP_RESOLUTION_XY,
                         agg: str = "max",
                         fill_method: str = None,
                         bounds: tuple = None,
                         crop_z_min: float = None,
                         crop_z_max: float = None) -> np.ndarray:
    """
    High-level function to load STL file and convert to depth map.
    
    Args:
        stl_filename: Path to STL file
        sample_points: Number of points to uniformly sample from mesh
        resolution: Grid cell size in same units as points
        agg: Aggregation method: "max", "mean", "median"
        fill_method: Optional 'griddata' for interpolation
        bounds: Optional ((xmin, xmax), (ymin, ymax))
        crop_z_min: Minimum z value to keep
        crop_z_max: Maximum z value to keep
    
    Returns:
        2D depth map array
    """
    # Load STL
    points = stl_to_points(stl_filename, sample_points=sample_points)
    
    # Remove NaN points
    points = points[~np.isnan(points).any(axis=1)]
    
    if points.size == 0:
        raise ValueError("No valid points in STL file")
    
    # Convert to depth map
    depth_map, _, _ = points_to_depth_map(
        points,
        resolution=resolution,
        agg=agg,
        fill_method=fill_method,
        bounds=bounds
    )
    
    # Post-processing: z cropping
    if crop_z_min is not None:
        depth_map[depth_map < crop_z_min] = np.nan
    
    if crop_z_max is not None:
        depth_map[depth_map > crop_z_max] = np.nan
    
    return depth_map


def batch_process_pcds(pcd_files: list, 
                       output_folder: str = None,
                       **processing_kwargs) -> list:
    """
    Process multiple PCD files with the same parameters.
    
    Args:
        pcd_files: List of PCD file paths
        output_folder: Optional folder to save depth maps as .npy files
        **processing_kwargs: Arguments passed to load_and_process_pcd
    
    Returns:
        List of depth map arrays
    """
    from io_utils import save_depth_map
    import os
    
    depth_maps = []
    
    for i, pcd_file in enumerate(pcd_files):
        print(f"Processing {i+1}/{len(pcd_files)}: {pcd_file}")
        
        depth_map = load_and_process_pcd(pcd_file, **processing_kwargs)
        depth_maps.append(depth_map)
        
        # Save if output folder specified
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            basename = os.path.basename(pcd_file).replace('.pcd', '')
            output_file = os.path.join(output_folder, f"{basename}_depth.npy")
            save_depth_map(output_file, depth_map)
    
    return depth_maps


def create_region_of_interest(depth_map: np.ndarray, 
                              x_range: tuple = None,
                              y_range: tuple = None,
                              resolution: float = DEPTH_MAP_RESOLUTION_XY) -> np.ndarray:
    """
    Extract a region of interest from a depth map.
    
    Args:
        depth_map: Input depth map
        x_range: (x_min, x_max) in mm (None = full range)
        y_range: (y_min, y_max) in mm (None = full range)
        resolution: Lateral resolution in microns
    
    Returns:
        Cropped depth map
    """
    rows, cols = depth_map.shape
    
    # Convert mm to pixel indices
    if x_range is not None:
        x_min_idx = max(0, int(x_range[0] * 1000 / resolution))
        x_max_idx = min(cols, int(x_range[1] * 1000 / resolution))
    else:
        x_min_idx, x_max_idx = 0, cols
    
    if y_range is not None:
        y_min_idx = max(0, int(y_range[0] * 1000 / resolution))
        y_max_idx = min(rows, int(y_range[1] * 1000 / resolution))
    else:
        y_min_idx, y_max_idx = 0, rows
    
    return depth_map[y_min_idx:y_max_idx, x_min_idx:x_max_idx]


# Example usage
if __name__ == "__main__":
    # Example: Process a single PCD file
    # depth_map = load_and_process_pcd("scan.pcd", resolution=0.002)
    
    # Example: Batch process multiple files
    # files = ["scan1.pcd", "scan2.pcd", "scan3.pcd"]
    # depth_maps = batch_process_pcds(files, output_folder="./output")
    
    print("processing_pipeline.py - Import and use these functions in your scripts")