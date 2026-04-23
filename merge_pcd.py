import numpy as np
import os
from pathlib import Path
from scipy.spatial import cKDTree


from clean_depth_utils import pcd_to_points, points_to_depth_map
from io_utils import save_depth_map, load_depth_map, load_pcd_points, load_stl_points, save_to_csv, save_to_stl
from analysis import find_best_worst_linearity
from registration import find_phase_shift
from transforms import shift_array, mirror_vertical
from cleaned_script import remove_outliers
from visualization import plot_depth_map, plot_histogram_cdf, plot_mult_depth_map_with_cuts

def calc_depthmap_KDTree_distance(dm1: np.ndarray, dm2: np.ndarray, resolution=0.01):
    """
    Calculate nearest-point distances between two depth maps using KDTree.

    Returns:
        signed_distances_2d : 2D array of signed distances (same shape as dm1)
        difference_vectors : Nx3 array of vectors (dm2 - dm1) for valid dm1 pixels
    """
    H1, W1 = dm1.shape
    H2, W2 = dm2.shape

    # Grid coordinates
    y1, x1 = np.indices((H1, W1))
    points1 = np.column_stack([
        x1.ravel() * resolution,
        y1.ravel() * resolution,
        dm1.ravel()
    ])

    y2, x2 = np.indices((H2, W2))
    points2 = np.column_stack([
        x2.ravel() * resolution,
        y2.ravel() * resolution,
        dm2.ravel()
    ])

    # Valid points
    mask1 = np.isfinite(points1[:, 2])
    mask2 = np.isfinite(points2[:, 2])

    points1_valid = points1[mask1]
    points2_valid = points2[mask2]

    # KDTree
    tree2 = cKDTree(points2_valid)
    distances, indices = tree2.query(points1_valid)

    nearest_dm2_z = points2_valid[indices][:, 2]

    signed_distances_valid = (
        np.sign(points1_valid[:, 2] - nearest_dm2_z) * distances
    )

    # Back to 2D
    signed_distances_2d = np.full(dm1.size, np.nan)
    signed_distances_2d[mask1] = signed_distances_valid
    signed_distances_2d = signed_distances_2d.reshape(H1, W1)

    # Difference vectors (only for valid dm1 pixels)
    difference_vectors = points2_valid[indices] - points1_valid

    return signed_distances_2d, difference_vectors


def merge_depth_maps(depth_map1: np.ndarray, depth_map2: np.ndarray, method: str = 'min', shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    """
    Merge two depth maps of the same shape using the specified method ('max', 'mean').
    """
    #if depth_map1.shape != depth_map2.shape:
    #    raise ValueError("Depth maps must have the same shape")
        
    if (shift_x != 0) or (shift_y != 0):
       # depth_map2 = shift_array(depth_map2, shift_x, shift_y) 
       if shift_x >= 0:
          depth_map2 = shift_array(depth_map2, shift_x, 0) 
       else:
          depth_map1 = shift_array(depth_map1, -shift_x, 0)
       
       if shift_y >= 0:
          depth_map2 = shift_array(depth_map2, 0, shift_y) 
       else:
          depth_map1 = shift_array(depth_map1, 0, -shift_y) 
       
    max_rows = min(depth_map1.shape[0], depth_map2.shape[0]) 
    max_cols = min(depth_map1.shape[1], depth_map2.shape[1])
    depth_map1 = depth_map1[:max_rows, :max_cols]
    depth_map2 = depth_map2[:max_rows, :max_cols]
        
    if method == 'max':
        return np.nanmax(np.stack([depth_map1, depth_map2], axis=0), axis=0)
    elif method == 'min':
        return np.nanmin(np.stack([depth_map1, depth_map2], axis=0), axis=0)
    elif method == 'mean':
        return np.nanmean(np.stack([depth_map1, depth_map2], axis=0), axis=0)
    elif method == 'diff':
        return (depth_map1 - depth_map2) # Element-wise minimum of both arrays
    elif method == 'nearest_point_distance':
        dist, _ = calc_depthmap_KDTree_distance(depth_map1, depth_map2, 1)
        return dist
    else:
        raise ValueError(f"merge_depth_maps - Unsupported combine method: {method}")
       
        
def merge_scanning_positions(arrays: list, column_offsets: list, combine_method: str = 'min', fill_value: float = np.nan) -> np.ndarray:
    """
    Combine multiple 2D arrays with specified column offsets into a larger 2D array.

    Parameters:
    arrays (list of np.ndarray): List of 2D arrays to combine.
    column_offsets (list of int): Column offsets for each array.
    combine_method (str): 'min', 'max', 'mean', or 'diff'. (diff is meaningless for >2 arrays, so usually use 'min' or 'mean')
    fill_value: Value to fill empty spaces (default is np.nan).

    Returns:
    np.ndarray: Combined array.
    """

    # Find overall dimensions
    max_rows = max(arr.shape[0] for arr in arrays)
    max_cols = max(offset + arr.shape[1] for arr, offset in zip(arrays, column_offsets))

    # Initialize result array
    result = np.full((max_rows, max_cols), fill_value, dtype=arrays[0].dtype)

    # For 'mean', we need to track how many values contributed to each cell
    if combine_method == 'mean':
        count = np.zeros((max_rows, max_cols), dtype=np.int32)

    # Place each array into result
    for arr, offset in zip(arrays, column_offsets):
        rows, cols = arr.shape
        target_slice = (slice(0, rows), slice(offset, offset + cols))

        if combine_method == 'min':
            existing = result[target_slice]
            result[target_slice] = np.fmin(existing, arr)
        elif combine_method == 'max':
            existing = result[target_slice]
            result[target_slice] = np.fmax(existing, arr)
        elif combine_method == 'mean':
            existing = result[target_slice]
            # Replace fill_value with 0 temporarily
            existing_filled = np.where(np.isnan(existing), 0, existing)
            arr_filled = np.where(np.isnan(arr), 0, arr)
            result[target_slice] = existing_filled + arr_filled
            count[target_slice] += ~np.isnan(arr)
        elif combine_method == 'diff':   
            existing = result[target_slice]
            result[target_slice] = (existing - arr)
        else:
            raise ValueError(f"merge_arrays_with_offsets - Unsupported combine method: {combine_method}")

    # For mean, divide by counts
    if combine_method == 'mean':
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(result, count)
            result[count == 0] = fill_value

    return result
                
        
# Main programme


# ---------------------------------- Settings ------------------------------------------------------------

resolution = 0.01                  # lateral resolution (XY) of depth map in [mm] -- 10microns
convertFilesToNpy = False           # Load PCD-files and STL-File and store to*.npy files with given resolution
loadFromNpy = True                 # Load PCD-files and STL-File from *.npy
mergeSensor12 = True               # Merge sensor 1 with sensor 2
autoAlignSensors = False;          # Auto calculate offset between sensor 1 and 2
offsetSensor12MM = 0.0             # Offset between sensor 1 and sensor 2 in Y-direction in mm
mergeScanningPositions = True;     # Merge scanning positions
invertMergingDirection = True;     # Invert merging direction of scanned regions (left-right vs. right-left)
scanningPositionOffsetMM =  5.0    # Offset between scans in X-direction in mm
part_diameter = 80.21              # Cylinder diameter of scanned tool in mm
mirrorStlVertical = True           # Flip STL horizonal (switch between positive form (part) and negative shape (die)) 
ref_point_part_scan = (0.0, 0.0)   # Reference point of part scan in stl-file (X in mm, Y in mm) -- todo
calcDeviationMaps = True           # Calculate profile deviation

# File paths:
dir_sensor1 = Path(r"D:\Projekte\11-13788_SAB-Fertigungstechnik\251117\Sensor vorn_ 08.04.527")
dir_sensor2 = Path(r"D:\Projekte\11-13788_SAB-Fertigungstechnik\251117\Sensor hinten_08.04.528")
stl_file = Path(r"I:\2_Projekte\11-13788_SAB-Fertigungstechnik\Messungen\251117\H2Go_AF2_Surface.stl")
dir_result = Path(r"D:\Projekte\11-13788_SAB-Fertigungstechnik\251117")

# --------------------------------------------------------------------------------------------------------



pcd_files_sensor1 = sorted([str(p) for p in dir_sensor1.glob("*.pcd")])     
pcd_files_sensor2 = sorted([str(p) for p in dir_sensor2.glob("*.pcd")]) 
pcd_files = pcd_files_sensor1 + pcd_files_sensor2 

dm_files_sensor1 = sorted([str(p) for p in dir_sensor1.glob("*_"+str(resolution)+"mm.npy")])     
dm_files_sensor2 = sorted([str(p) for p in dir_sensor2.glob("*_"+str(resolution)+"mm.npy")])     
dm_files = dm_files_sensor1 + dm_files_sensor2    

dm_sensor1 = []
dm_sensor2 = []
dm_fusion12 = []
dm_stl = None
dm_zdiff = None
dm_profiledev = None

colOffsetsScanPos = []

 
if convertFilesToNpy:
 pts_stl = load_stl_points(stl_file)
 dm, _, _ = points_to_depth_map(points = pts_stl,
                                resolution = resolution,
                                agg = "max",
                                fill_method = "griddata", #None,
                                bounds = None)
 save_depth_map(str(stl_file).replace(".stl",f"_{resolution}mm.npy"),dm)

 for f in pcd_files:      
    pts = pcd_to_points(f)
    dm, _, _ = points_to_depth_map(points = pts,
                                   resolution = resolution,
                                   agg = "max",
                                   fill_method = "griddata", #None,
                                   bounds = None)
    save_depth_map(f.replace(".pcd",f"_{resolution}mm.npy"),dm)

if loadFromNpy:
 dm_stl = load_depth_map(str(stl_file).replace(".stl",f"_{resolution}mm.npy"))
 if mirrorStlVertical:
     dm_stl = mirror_vertical(dm_stl)

 for f in dm_files_sensor1:
    dm = load_depth_map(f)
    dm_sensor1.append(dm)
    #plot_depth_map(dm, title = "sensor 1: " + os.path.basename(f))  

 for f in dm_files_sensor2:
    dm = load_depth_map(f)
    dm = mirror_vertical(dm)  # sensor 2 is turned 180° in relation sensor 1     
    dm_sensor2.append(dm)
    #plot_depth_map(dm, title = "sensor 2: " + os.path.basename(f))

if mergeSensor12:
  n = min(len(dm_sensor1), len(dm_sensor2))  
  offset_x = [0 for i in range(n)] 
  offset_y = [0 for i in range(n)] 
  
  glob_offset_y = int(offsetSensor12MM / resolution)
  if autoAlignSensors:  
     for i in range(n):
         linAnaRes1 = find_best_worst_linearity(dm_sensor1[i])
         linAnaRes2 = find_best_worst_linearity(dm_sensor2[i])
         col1 = dm_sensor1[i][:, linAnaRes1['worst_col_idx']]
         col2 = dm_sensor2[i][:, linAnaRes2['worst_col_idx']]
         row1 = dm_sensor1[i][linAnaRes1['worst_row_idx'], :]
         row2 = dm_sensor2[i][linAnaRes2['worst_row_idx'], :]
         offset_y[i] = find_phase_shift(col1, col2)
         offset_x[i] = find_phase_shift(row1, row2)
     
     glob_offset_y = int(np.median(offset_y)) 
     glob_offset_x = int(np.median(offset_x)) 
     offsetSensor12MM = glob_offset_y * resolution
   
  for i in range(n):
      dm_sensor1[i] = remove_outliers(dm_sensor1[i])
      dm_sensor2[i] = remove_outliers(dm_sensor2[i])
      dm = merge_depth_maps(dm_sensor1[i], dm_sensor2[i], method = 'min', shift_x = 0, shift_y = glob_offset_y ) 
      dm_fusion12.append(dm)
      #plot_depthmap(dm, title = f"sensor 1 & 2: {i+1}.pcd") 

else:
  dm_fusion12 = dm_sensor1
           
if mergeScanningPositions:
  if invertMergingDirection:
     dm_fusion12 = dm_fusion12[::-1]
      
  colOffsetsScanPos = [ i * int(scanningPositionOffsetMM / resolution) for i in range(len(dm_sensor1))] 
  dm_part = merge_scanning_positions(dm_fusion12, colOffsetsScanPos) 
  
  # Export
  save_to_csv(filename=os.path.join(dir_result,"Result_Part.csv"), data=dm_part * 1000, formatting='%.0f')
  save_depth_map(filename=os.path.join(dir_result,"Result_Part.npy"), data=dm_part)
  mesh_flatten  = save_to_stl(depthmap=dm_part, resolution=resolution, poisson_depth=9, poisson_width=0, diameter=None, fileName=os.path.join(dir_result,"Result.stl"))
  mesh_cylinder = save_to_stl(depthmap=dm_part, resolution=resolution, poisson_depth=9, poisson_width=0, diameter=part_diameter, fileName=os.path.join(dir_result,"ResultCylinder.stl"))

if calcDeviationMaps:
    dm_zdiff      = merge_depth_maps(depth_map1 = dm_part, depth_map2 = dm_stl, method = 'diff', shift_x = ref_point_part_scan[0] / resolution, shift_y = ref_point_part_scan[1] / resolution)
    dm_profiledev = merge_depth_maps(depth_map1 = dm_part, depth_map2 = dm_stl, method = 'nearest_point_distance', shift_x = ref_point_part_scan[0] / resolution, shift_y = ref_point_part_scan[1] / resolution)
    
    # Export
    save_to_csv(filename=os.path.join(dir_result,"Result_ZDiff.csv"), data=dm_zdiff * 1000, formatting='%.0f')
    save_depth_map(filename=os.path.join(dir_result,"Result_ZDiff.npy"), data=dm_zdiff)
    
    save_to_csv(filename=os.path.join(dir_result,"Result_ProfileDev.csv"), data=dm_profiledev * 1000, formatting='%.0f')
    save_depth_map(filename=os.path.join(dir_result,"Result_ProfileDev.npy"), data=dm_profiledev)
    
# Plot results
if (dm_part is not None): 
    plot_depth_map(dm_part, title = "Part", lateral_resolution = resolution)
    # plot_histogram_cdf(dm_part)
if (dm_stl is not None): 
    plot_depth_map(dm_stl, title = "STL: " + os.path.basename(stl_file), lateral_resolution = resolution) 
if (dm_zdiff is not None): 
    plot_depth_map(dm_zdiff, title = "Z-Difference: " + os.path.basename(stl_file), lateral_resolution = resolution)
if (dm_profiledev is not None): 
    plot_depth_map(dm_profiledev, title = "Profile Deviation: " + os.path.basename(stl_file), lateral_resolution = resolution)

# Interactive plot - Spyder -> Tools -> Preferences -> IPython Console -> Graphics -> Backend = Automatic (Default was Inline) 
plot_mult_depth_map_with_cuts(z_maps = [dm_stl,dm_part,dm_zdiff,dm_profiledev],names = ["STL","Scan-Data","Z-Diff","Profile Deviation"],lateral_resolution = resolution)