#globals().clear()  # Purge global variables

import numpy as np
import os
from pathlib import Path
from scipy.spatial import cKDTree
import gc

from clean_depth_utils import pcd_to_points, points_to_depth_map, autolevel_depth_map, scale_depth_map_xy
from io_utils import save_depth_map, load_depth_map, load_pcd_points, load_stl_points, save_to_csv, save_to_stl
from analysis import find_best_worst_linearity
from registration import find_phase_shift
from transforms import shift_array, mirror_vertical
from cleaned_script import remove_outliers
from visualization import plot_depth_map, plot_histogram_cdf, plot_mult_depth_map_with_cuts
from global_defs import depthmapVarType




def calc_depthmap_KDTree_distance(dm1: np.ndarray, dm2: np.ndarray, resolution=0.01):
    """
    Calculate nearest-point distances between two depth maps using KDTree.

    Returns:
        signed_distances_2d : 2D array of signed distances (same shape as dm1)
        difference_vectors : Nx3 array of vectors (dm2 - dm1) for valid dm1 pixels
    """
    if (dm1 is None) or (dm2 is None): 
       return None, None
    
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


def shift_maps_relative(dm1: np.ndarray, dm2: np.ndarray, shift_x: int = 0, shift_y: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift two depth maps relative to each other.

    Positive shifts move dm2 (by adding NaN rows/cols).
    Negative shifts move dm1.

    After shifting, both maps are cropped to the same shape.

    Parameters
    ----------
    dm1, dm2 : (H, W) ndarray
        Input depth maps
    shift_x : int
        Column shift (positive → dm2 right, negative → dm1 right)
    shift_y : int
        Row shift (positive → dm2 down, negative → dm1 down)

    Returns
    -------
    dm1_shifted, dm2_shifted : ndarray
        Shifted depth maps with identical shape
    """
   
    depth_map1 = dm1.copy().astype(depthmapVarType, copy=False)
    depth_map2 = dm2.copy().astype(depthmapVarType, copy=False)
        
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
      
    return depth_map1, depth_map2


def shift_array_grow_shrink(
    arr: np.ndarray,
    shift_rows: int = 0,
    shift_cols: int = 0,
    fill_value=np.nan
) -> np.ndarray:
    """
    Shift a 2D array by adding/removing rows and columns.

    Positive shifts add NaN rows/columns.
    Negative shifts remove rows/columns.

    Parameters
    ----------
    arr : (H, W) ndarray
        Input array
    shift_rows : int
        +N → add N rows at top
        -N → remove N rows from top
    shift_cols : int
        +N → add N columns on left
        -N → remove N columns from left
    fill_value : scalar
        Value used for newly created cells

    Returns
    -------
    shifted : ndarray
        Shifted array (shape may change)
    """

    if arr.ndim != 2:
        raise ValueError("arr must be a 2D array")

    out = arr.copy().astype(depthmapVarType, copy=False)

    # --- rows ---
    if shift_rows > 0:
        pad = np.full((shift_rows, out.shape[1]), fill_value, dtype=out.dtype)
        out = np.vstack((pad, out))
    elif shift_rows < 0:
        out = out[-shift_rows:, :]

    # --- columns ---
    if shift_cols > 0:
        pad = np.full((out.shape[0], shift_cols), fill_value, dtype=out.dtype)
        out = np.hstack((pad, out))
    elif shift_cols < 0:
        out = out[:, -shift_cols:]

    return out


def merge_depth_maps(depth_map1: np.ndarray, depth_map2: np.ndarray, method: str = 'min', shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    """
    Merge two depth maps of the same shape using the specified method ('max', 'mean').
    """
    #if depth_map1.shape != depth_map2.shape:
    #    raise ValueError("Depth maps must have the same shape")
    gc.collect(); # purge unused variables from garbage collector
        
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
    depth_map1 = depth_map1[:max_rows, :max_cols].astype(depthmapVarType, copy=False)
    depth_map2 = depth_map2[:max_rows, :max_cols].astype(depthmapVarType, copy=False)
        
    if method == 'max':
        return np.nanmax(np.stack([depth_map1, depth_map2], axis=0), axis=0).astype(depthmapVarType, copy=False)
    elif method == 'min':
        return np.nanmin(np.stack([depth_map1, depth_map2], axis=0), axis=0).astype(depthmapVarType, copy=False)
    elif method == 'mean':
        return np.nanmean(np.stack([depth_map1, depth_map2], axis=0), axis=0).astype(depthmapVarType, copy=False)
    elif method == 'diff':
        return (depth_map1 - depth_map2).astype(depthmapVarType) # Element-wise minimum of both arrays
    elif method == 'nearest_point_distance':
        dist, _ = calc_depthmap_KDTree_distance(depth_map1, depth_map2, 1)
        return dist.astype(depthmapVarType, copy=False)
    else:
        raise ValueError(f"merge_depth_maps - Unsupported combine method: {method}")
       
        
def merge_scanning_positions(arrays: list, column_offsets: list, invertMergingDirection: bool = False, combine_method: str = 'min', fill_value: float = np.nan) -> np.ndarray:
    """
    Combine multiple 2D arrays with specified column offsets into a larger 2D array.

    Parameters:
    arrays (list of np.ndarray): List of 2D arrays to combine.
    column_offsets (list of int): Column offsets for each array.
    invertMergingDirection (bool): invert merging direction of arrays (False: left-to-right or True: right-to-left)
    combine_method (str): 'min', 'max', 'mean', or 'diff'. (diff is meaningless for >2 arrays, so usually use 'min' or 'mean')
    fill_value: Value to fill empty spaces (default is np.nan).

    Returns:
    np.ndarray: Combined array.
    """
    #gc.collect(); # purge unused variables from garbage collector
    
    
    # process array from left-to-right or from right-to-left
    if invertMergingDirection:
       arrays = arrays[::-1]

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
        arr = arr.astype(depthmapVarType, copy=False)
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

    return result.astype(depthmapVarType, copy=False)
                
        
# Main programme

# ---------------------------------- Settings ------------------------------------------------------------

resolution = 0.01                    # lateral resolution (XY) of depth map in [mm] -- 10microns
convertFilesToNpy = False            # Load PCD-files and STL-File and store to*.npy binary files with given resolution
loadFromNpy = True                   # Load PCD-files and STL-File from *.npy
mergeSensor12 = True                 # Merge sensor 1 with sensor 2
outlierHistoThres = 2.5              # Histogramme threshold for outlier purging [%], default = 2.5%
autoAlignSensors = False             # Auto calculate offset between sensor 1 and 2
offsetYSensor12MM = 0.0              # Offset between sensor 1 and sensor 2 in Y-direction in mm
offsetXSensor12MM = 0.0              # Offset between sensor 1 and sensor 2 in Y-direction in mm
mergeScanningPositions = True        # Merge scanning positions
invertMergingDirection = True        # Invert merging direction of scanned regions (left-right vs. right-left)
scanningPositionOffsetMM =  5.0      # Offset between scans in X-direction in mm
part_diameter = 150.0                # Cylinder diameter of scanned tool in mm.   #80.21
mirrorStlVertical = True             # Flip STL horizonal (switch between positive form (part) and negative shape (die)) 
flipSTLinZdirection = True           # Flip Z-levels of STL (switch between positive form (part) and negative shape (die)) 
calcDeviationMaps = True             # Calculate profile deviation
ref_point_part_scan = (-1.00, -1.34) # Reference point of part scan in stl-file (X in mm, Y in mm)
sensorScaleY = 1.83                  # Scaling of sensor data in Y
sensorScaleX = 1.0                   # Scaling of sensor data in X
sensorScaleZ = 1.0                   # Scaling of sensor data in Z
plotResults = True                   # Plot results as diagramme
interactivePlot = True               # Interactive plot; Note: Adjust Spyder graphic settings: Spyder main menu -> Tools -> Preferences -> IPython Console -> Graphics -> Backend = Automatic (Default was Inline) 
exportResultsCsv = False              # Export results to CSV-files in dir_result
exportResultsStl = True              # Export results to STL-files in dir_result

# File paths:
dir_sensor1 = Path(r"D:\Projekte\11-13788_SAB-Fertigungstechnik\251117\Sensor vorn_ 08.04.527")
dir_sensor2 = Path(r"D:\Projekte\11-13788_SAB-Fertigungstechnik\251117\Sensor hinten_08.04.528")
stl_file = Path(r"I:\2_Projekte\11-13788_SAB-Fertigungstechnik\Messungen\251117\STL\STL_BPP_30x100_UT.stl")
dir_result = Path(r"D:\Projekte\11-13788_SAB-Fertigungstechnik\251117\PythonResults")

# --------------------------------------------------------------------------------------------------------


# Initialize gloabl variables
pcd_files_sensor1 = sorted([str(p) for p in dir_sensor1.glob("*.pcd")])
pcd_files_sensor2 = sorted([str(p) for p in dir_sensor2.glob("*.pcd")]) 
pcd_files = pcd_files_sensor1 + pcd_files_sensor2 

dm_files_sensor1 = sorted([str(p) for p in dir_sensor1.glob("*_"+str(resolution)+"mm.npy")])     
dm_files_sensor2 = sorted([str(p) for p in dir_sensor2.glob("*_"+str(resolution)+"mm.npy")])     
dm_files = dm_files_sensor1 + dm_files_sensor2

# initialize depth maps
dm_sensor1 = []
dm_sensor2 = []
dm_fusion12 = []
dm_stl = None
dm_zdiff = None
dm_profiledev = None

colOffsetsScanPos = []

# Convert PCD and STL files to Python's internal *.npy-format and store to files 
if convertFilesToNpy:
 # load STL
 pts = load_stl_points(stl_file)
 dm, _, _ = points_to_depth_map(points = pts,
                                resolution = resolution,
                                agg = "max",
                                fill_method = "griddata", #None,
                                bounds = None)
 save_depth_map(str(stl_file).replace(".stl",f"_{resolution}mm.npy"),dm)
 
 # purge variables not needed anymore  
 del pts
 gc.collect()
 
 # load *.pcd-files of sensors 1 & 2
 for f in pcd_files:      
    pts = pcd_to_points(f)
    dm, _, _ = points_to_depth_map(points = pts,
                                   resolution = resolution,
                                   agg = "max",
                                   fill_method = "griddata", #None,
                                   bounds = None)
    save_depth_map(f.replace(".pcd",f"_{resolution}mm.npy"),dm)
    
    # purge variables not needed anymore  
    del pts
    #gc.collect()

# Load binary raw data from *.npy-files
if loadFromNpy:
 dm_stl = load_depth_map(str(stl_file).replace(".stl",f"_{resolution}mm.npy"))
 dm_stl = autolevel_depth_map(depth_map = dm_stl, flipZdirection=flipSTLinZdirection) # set top level of STL to surface of tool (0mm)
 
 # flip STL (if required)
 if mirrorStlVertical:                
     dm_stl = mirror_vertical(dm_stl)
 
 # load data of sensor 1
 for f in dm_files_sensor1:
    dm = load_depth_map(f)
    dm_sensor1.append(dm)
    #plot_depth_map(dm, title = "sensor 1: " + os.path.basename(f))  

 # load data of sensor 2
 for f in dm_files_sensor2:
    dm = load_depth_map(f)
    dm = mirror_vertical(dm)  # sensor 2 is turned 180° in relation sensor 1     
    dm_sensor2.append(dm)
    #plot_depth_map(dm, title = "sensor 2: " + os.path.basename(f))
    
 # purge variables not needed anymore  
 del dm
 #gc.collect()

# Merging of sensors 1 & 2
if mergeSensor12:
  n = min(len(dm_sensor1), len(dm_sensor2))  
  
  #Offset matrix-rows between sensors
  glob_offset_y = int(offsetYSensor12MM / resolution)
  glob_offset_x = int(offsetXSensor12MM / resolution)
  
  # automatically determine offset sensor 1-2 in x- and y-direction
  if autoAlignSensors: 
     offset_x = [0 for i in range(n)] # column offset sensor 1-2
     offset_y = [0 for i in range(n)] # row offset sensor 1-2
      
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
     
  # Purge outliers from sensor 1 & 2 and merge both sensors
  for i in range(n):
      dm_sensor1[i] = remove_outliers(data=dm_sensor1[i], remove_rare_depths_histothres = outlierHistoThres)
      dm_sensor2[i] = remove_outliers(data=dm_sensor2[i], remove_rare_depths_histothres = outlierHistoThres)
      dm = merge_depth_maps(dm_sensor1[i], dm_sensor2[i], method = 'min', shift_x = glob_offset_x, shift_y = glob_offset_y) 
      dm_fusion12.append(dm)
      #plot_depthmap(dm, title = f"sensor 1 & 2: {i+1}.pcd") 

else:
  dm_fusion12 = dm_sensor1 # only use sensor 1
  
# purge variables not needed anymore  
del dm_sensor1, dm_sensor2
#gc.collect()

# Merge scanning slices to overall part scan
if mergeScanningPositions:
  # Merge scanning position to a common scan of part
  colOffsetsScanPos = [ i * int(scanningPositionOffsetMM / resolution) for i in range(len(dm_fusion12))] 
  dm_part = merge_scanning_positions(arrays = dm_fusion12, column_offsets = colOffsetsScanPos, invertMergingDirection = invertMergingDirection) 
  
  # Adjust scaling of scan data (only in y-direction)
  dm_part = scale_depth_map_xy(depth_map = dm_part,scale_x = 1.0, scale_y = sensorScaleY, scale_z = 1.0)

# purge variables not needed anymore  
del dm_fusion12
#gc.collect()

# Caluclate deviations (deltaZ and profile deviation) between scan and STL
if calcDeviationMaps:    
    # Shift scanned data of part to reference point of STL
    dm_part_shifted = shift_array_grow_shrink(arr=dm_part, shift_cols = int(ref_point_part_scan[0] / resolution), shift_rows = int(ref_point_part_scan[1] / resolution))

    # Deviation maps deltaZ and profile deviation (KDTree)
    dm_zdiff      = merge_depth_maps(depth_map1 = dm_part_shifted, depth_map2 = dm_stl, method = 'diff', shift_x = 0, shift_y = 0)
    dm_profiledev = merge_depth_maps(depth_map1 = dm_part_shifted, depth_map2 = dm_stl, method = 'nearest_point_distance', shift_x = 0, shift_y = 0)
    #dm_zdiff      = merge_depth_maps(depth_map1 = dm_part_shifted, depth_map2 = dm_stl_shifted, method = 'diff', shift_x = int(ref_point_part_scan[0] / resolution), shift_y = int(ref_point_part_scan[1] / resolution))
    #dm_profiledev = merge_depth_maps(depth_map1 = dm_part_shifted, depth_map2 = dm_stl_shifted, method = 'nearest_point_distance', shift_x = int(ref_point_part_scan[0] / resolution), shift_y = int(ref_point_part_scan[1] / resolution))
    
if exportResultsCsv:
   save_to_csv(filename=os.path.join(dir_result,"Result_Part.csv"), data=dm_part * 1000, formatting='%.0f')
   save_depth_map(filename=os.path.join(dir_result,"Result_Part.npy"), data=dm_part) 
        
   save_to_csv(filename=os.path.join(dir_result,"Result_ZDiffDeviation.csv"), data=dm_zdiff * 1000, formatting='%.0f')
   save_depth_map(filename=os.path.join(dir_result,"Result_ZDiffDeviation.npy"), data=dm_zdiff)
    
   save_to_csv(filename=os.path.join(dir_result,"Result_ProfileDeviation.csv"), data=dm_profiledev * 1000, formatting='%.0f')
   save_depth_map(filename=os.path.join(dir_result,"Result_ProfileDeviation.npy"), data=dm_profiledev)
       
   save_to_csv(filename=os.path.join(dir_result,"Result_PartShifted.csv"), data=dm_part_shifted * 1000, formatting='%.0f')
   save_depth_map(filename=os.path.join(dir_result,"Result_PartShifted.npy"), data=dm_part_shifted)    
    
# Export to files
if exportResultsStl:  
   #save_to_stl(data=dm_part, resolution=resolution, poisson_depth=9, poisson_width=0, diameter=None, fileName=os.path.join(dir_result,"Result_Part.stl"))
   #save_to_stl(data=dm_part, resolution=resolution, poisson_depth=9, poisson_width=0, diameter=part_diameter, fileName=os.path.join(dir_result,"Result_Part_Cylinder.stl"))

   save_to_stl(data=dm_zdiff, resolution=resolution, poisson_depth=9, poisson_width=0, diameter=None, fileName=os.path.join(dir_result,"Result_ZDiffDeviation.stl"))
   save_to_stl(data=dm_zdiff, resolution=resolution, poisson_depth=9, poisson_width=0, diameter=part_diameter, fileName=os.path.join(dir_result,"Result_ZDiffDeviation_Cylinder.stl"))
    
   save_to_stl(data=dm_profiledev, resolution=resolution, poisson_depth=9, poisson_width=0, diameter=None, fileName=os.path.join(dir_result,"Result_ProfileDeviation.stl"))
   save_to_stl(data=dm_profiledev, resolution=resolution, poisson_depth=9, poisson_width=0, diameter=part_diameter, fileName=os.path.join(dir_result,"Result_ProfileDeviation_Cylinder.stl"))
      
   save_to_stl(data=dm_part_shifted, resolution=resolution, poisson_depth=9, poisson_width=0, diameter=None, fileName=os.path.join(dir_result,"Result_PartShifted.stl"))
   save_to_stl(data=dm_part_shifted, resolution=resolution, poisson_depth=9, poisson_width=0, diameter=part_diameter, fileName=os.path.join(dir_result,"Result_PartShifted_Cylinder.stl"))
      
    
# Plot results
if plotResults:
   if (dm_part is not None): 
      plot_depth_map(dm_part_shifted, title = "Part scan", lateral_resolution = resolution, output_folder=dir_result)
      plot_histogram_cdf(dm_part_shifted)
   if (dm_stl is not None): 
      plot_depth_map(dm_stl, title = "STL - " + os.path.basename(stl_file), lateral_resolution = resolution, output_folder=dir_result) 
   if (dm_zdiff is not None): 
      plot_depth_map(dm_zdiff, title = "Z-Difference - " + os.path.basename(stl_file), lateral_resolution = resolution, output_folder=dir_result)
   if (dm_profiledev is not None): 
      plot_depth_map(dm_profiledev, title = "Profile Deviation - " + os.path.basename(stl_file), lateral_resolution = resolution, output_folder=dir_result)


# Interactive plot; Note: Adjust Spyder graphic settings: Spyder main menu -> Tools -> Preferences -> IPython Console -> Graphics -> Backend = Automatic (Default was Inline) 
if interactivePlot:
   #plot_mult_depth_map_with_cuts(z_maps = [dm_part_shifted,dm_stl], names = ["Scan-Data","STL"], lateral_resolution = resolution, out_dir=dir_result)
   plot_mult_depth_map_with_cuts(z_maps = [dm_part_shifted,dm_stl,dm_zdiff,dm_profiledev], names = ["Scan-Data","STL","Z-Diff","Profile Deviation"], lateral_resolution = resolution, out_dir=dir_result)