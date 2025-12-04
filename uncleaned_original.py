import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
from scipy.ndimage import median_filter, sobel, generic_filter, generic_filter, gaussian_filter
from scipy.signal import convolve2d
import open3d as o3d
import cv2
import math
from time import strftime
import multiprocessing
from sklearn import linear_model
import os


import builtins

plt.close('all')

# For inline (static) plots:
#%matplotlib inline

# For interactive windowed plots:
#%matplotlib qt


compensateExcentricity = True # compensate excentricity of roll in y-direction

DEPTH_MAP_RESOLUTION_XY = 2; # microns  # 10, 5, 2

TIME_FORMAT = "%H:%M:%S"
UNIT_STR_MICRON = '['+u"\u03bc"+'m]'
if DEPTH_MAP_RESOLUTION_XY == 1:
  UNIT_STR = ' ['+u"\u03bc"+'m]'
else:
  UNIT_STR = ' ['+str(DEPTH_MAP_RESOLUTION_XY)+u"\u03bc"+'m]'



def LoadDepthMapFromSTLFile(stl_filename, AutoLevelPlane=True, AutoLevelTolerance=450, crop_z_min=None, crop_z_max=None):
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": LoadDepthMapFromSTLFile: " + stl_filename)
    # Determine the file extension
    file_extension = stl_filename.split(".")[-1].lower()

    # Case switch based on the file extension
    if file_extension == "stl":
      mesh_data = o3d.io.read_triangle_mesh(stl_filename)

      # Extract vertices
      print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": LoadDepthMapFromSTLFile - Extract vertices")
      vertices = np.array(mesh_data.vertices)
      vertices = mesh_data.sample_points_uniformly(number_of_points=vertices.size*30) #3360440)

      #sweep memory
      del mesh_data

      #o3d.visualization.draw_geometries([vertices], window_name="PCD")
      points = np.asarray(vertices.points)[:]

      #sweep memory
      del vertices
    else:
      raise ValueError("Unsupported file type. Supported types: STL")

    dp = HandlePoints(points, AutoLevelPlane, AutoLevelTolerance, crop_z_min, crop_z_max)

    # Sweep memory
    del points

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": LoadDepthMapFromSTLFile - DONE")

    return dp



# convert point cloud to depth map
def LoadDepthMapFromPCDFile(pcd_filename, AutoLevelPlane=True, AutoLevelTolerance=450, scaleY=1.0, crop_z_min=None, crop_z_max=None, initial_crop_z_max=-10, manual_z_offset=0):
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": LoadDepthMapFromPCDFile - load data from: " + pcd_filename)

    # Determine the file extension
    file_extension = pcd_filename.split(".")[-1].lower()

    # Case switch based on the file extension
    if file_extension == "pcd":
      # Read the PCD file
      pcd_data = o3d.io.read_point_cloud(pcd_filename)

      # Convert to numpy array
      points = np.asarray(pcd_data.points)

      #sweep memory
      del pcd_data

      if initial_crop_z_max is not None:
        print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": LoadDepthMapFromPCDFile - Initial crop z_max")
        points = points[(points[:,2] <= initial_crop_z_max / 1000), :]
        points = points[points[:,2] !=  np.nan, :]

      maxy = np.max(points[:,1])
      if scaleY!=1.0:
        points[:,1] *= scaleY

    else:
      raise ValueError("Unsupported file type. Supported types: PCD")

    dp = HandlePoints(points, AutoLevelPlane, AutoLevelTolerance, crop_z_min, crop_z_max)

    if manual_z_offset is not None:
      dp[:] += manual_z_offset

    #sweep memory
    del points

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": LoadDepthMapFromPCDFile - DONE")

    return dp


def HandlePoints(points, AutoLevelPlane, AutoLevelTolerance, crop_z_min, crop_z_max):
    # Scale x/y/z-coordinates to microns and transform to (0,0,0)
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": HandlePoints - Scale and transform x/y/z-ccordinates")

    # remove points with nan-coordinates
    points = points[points[:,2] !=  np.nan, :]
    points = points[points[:,0] !=  np.nan, :]
    points = points[points[:,1] !=  np.nan, :]

    points[:,:2] *= 1000.0/DEPTH_MAP_RESOLUTION_XY
    points[:,2] *= 1000.0
    zmax_orig = np.nanmax(points[:, 2])
    ymin_orig = np.nanmin(points[:, 1])
    xmin_orig = np.nanmin(points[:, 0])
    points[:,0] -= xmin_orig
    points[:,1] -= ymin_orig
    #points[:,2] -= zmax_orig

    # Find top plane of point cloud and level top plane as z=0
    if AutoLevelPlane:
      points = RANSACLevelPointsToTopPlane(points, AutoLevelTolerance)

    # Create a 2D depth map
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": HandlePoints - Allocate memory for 2d map")
    y_dim = math.ceil(np.nanmax(points[:, 1]) - np.nanmin(points[:, 1]))
    x_dim = math.ceil(np.nanmax(points[:, 0]) - np.nanmin(points[:, 0]))
    depth_map = np.full((y_dim+1, x_dim+1), np.nan, dtype=np.float32)
    #depth_map = np.empty((y_dim+1, x_dim+1), dtype=np.float16) #depth_map[:] = np.nan
    #depth_map.fill(np.nan)

    # Populate the depth map with z-coordinates in a x/y-1 micron-grid
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": HandlePoints - Prepare pixel coordinates")
    y = points[:,1].astype(np.int32)
    x = points[:,0].astype(np.int32)
    z = points[:,2].astype(np.float32)


    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": HandlePoints - Populate 2d map")
    unique_coordinates, unique_indices = np.unique(np.column_stack((y, x)), axis=0, return_index=True)  # Find unique coordinates
    depth_map[unique_coordinates[:, 0], unique_coordinates[:, 1]] = z[unique_indices] # Set unique values at unique coordinates
    #depth_map[y, x] = z

    #sweep memory
    del x, y, z, unique_coordinates, unique_indices
    del points

    # Clip the values if z_min and z_max are provided
    if crop_z_min is not None:
      print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": HandlePoints - Crop z_min")
      depth_map[depth_map<crop_z_min] = np.nan

    if crop_z_max is not None:
      print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": HandlePoints - Crop z_max")
      depth_map[depth_map>crop_z_max] = np.nan

    # Interpolate caps and missing/clipped points
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": HandlePoints - Interpolate missing points")
    if np.any(a=depth_map, where=np.nan).size > 0:
      #interpolate rows
      depth_map_interpol = np.apply_along_axis(interpolateNaNs, 1, depth_map) #parallelApplyAlongAxis(interpolateNaNs, 1, depth_map)

      #sweep memory
      del depth_map

      #interpolate cols
      if np.any(a=depth_map_interpol, where=np.nan):
        depth_map_interpol = np.apply_along_axis(interpolateNaNs, 0, depth_map_interpol) #parallelApplyAlongAxis(interpolateNaNs, 0, depth_map)
    else:
      depth_map_interpol = depth_map

    # crop remaining NaN-rows and cols, see: https://stackoverflow.com/questions/25831023/numpy-crop-2d-array-to-non-nan-values
    # nans = np.isnan(depth_map_interpol) #Find position of all the NaNs
    # nancols = np.all(nans, axis=0) # Find all the columns that have only NaNs
    # nanrows = np.all(nans, axis=1) # Find all the columns that have only NaNs
    # depth_map_interpol = image[:,~nancols][~nanrows] #remove all the rows and columns that are all NaNs

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": HandlePoints - Done")
    return depth_map_interpol.astype(np.float16)


def MirrorArrayVertical(arr):
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": MirrorArrayVertical")
    # Flip the array along the vertical axis (columns)
    return np.flip(arr, axis=1)


# def ShiftArrayByOffset(arr, shift_x=0, shift_y=0):
#     """
#     Shifts a 2D numpy array by the specified offsets in x and y directions and
#     fills empty cells with zeroes instead of wrapping around.

#     Parameters:
#     arr (np.ndarray): The 2D array to be shifted.
#     shift_sensor12_x (int): The number of elements to shift in the x-direction (columns).
#     shift_sensor12_y (int): The number of elements to shift in the y-direction (rows).

#     Returns:
#     np.ndarray: The shifted 2D array with zero-padding.
#     """
#     # Get the shape of the array
#     rows, cols = arr.shape

#     # Create an array of zeros with the same shape
#     shifted_arr = np.empty_like(arr)
#     shifted_arr.fill(np.nan)

#     # Apply the shifts with np.roll
#     shifted_arr[max(shift_y, 0):rows, max(shift_x, 0):cols] = arr[:rows-max(shift_y, 0), :cols-max(shift_x, 0)]
#     shifted_arr[:max(shift_y, 0), :] = 0
#     shifted_arr[:, :max(shift_x, 0)] = 0

#     return shifted_arr

def ShiftArrayByOffset(arr, shift_x=0, shift_y=0):
    """
    Shifts a 2D numpy array by the specified offsets in x and y directions and
    fills empty cells with zeroes instead of wrapping around.

    Parameters:
    arr (np.ndarray): The 2D array to be shifted.
    shift_x (int): The number of elements to shift in the x-direction (columns).
    shift_y (int): The number of elements to shift in the y-direction (rows).

    Returns:
    np.ndarray: The shifted 2D array with zero-padding.
    """

    # How to use example:
    #
    # if shift_sensor12_x>0:
    #      point_cloud_depth_map_scanner1 = ShiftArrayByOffset(point_cloud_depth_map_scanner1,abs(shift_sensor12_x),0)
    # elif shift_sensor12_x<0:
    #      point_cloud_depth_map_scanner2 = ShiftArrayByOffset(point_cloud_depth_map_scanner2,abs(shift_sensor12_x),0)

    # if shift_sensor12_y>0:
    #     point_cloud_depth_map_scanner1 = ShiftArrayByOffset(point_cloud_depth_map_scanner1,0, abs(shift_sensor12_y))
    # elif shift_sensor12_y<0:
    #     point_cloud_depth_map_scanner2 = ShiftArrayByOffset(point_cloud_depth_map_scanner2,0, abs(shift_sensor12_y))

    # Get the shape of the array
    rows, cols = arr.shape

    shift_x = abs(shift_x);
    shift_y = abs(shift_y);

    if shift_x > 0:
     shifted_arr = np.hstack([np.zeros((rows, shift_x)), arr])
    else:
     shifted_arr = arr

     if shift_y > 0:
       shifted_arr = np.vstack([np.zeros((shift_y, shifted_arr.shape[1])), shifted_arr])
     else:
       shifted_arr = shifted_arr

    return shifted_arr



def CropNanBorders(arr):
    """
    Crops the rows and columns at the borders of the 2D array that only contain NaNs.

    Parameters:
    arr (np.ndarray): The 2D array to be cropped.

    Returns:
    np.ndarray: The cropped 2D array with NaN-only rows and columns removed.
    """
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": CropNanBorders")

    # Check for rows that are all NaN
    row_mask = np.all(np.isnan(arr), axis=1)

    # Check for columns that are all NaN
    col_mask = np.all(np.isnan(arr), axis=0)

    # Crop the array by removing NaN-only rows and columns
    cropped_arr = arr[~row_mask, :][:, ~col_mask]

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": CropNanBorders - Done")

    return cropped_arr


def CalcCompensationModel(point_data, indexForModelFit, use_ransac=True, axis=0, mode = 'OffsetAndSlope'):
    """
    Compensates each row or column of a 2D array using either linear regression (linregress)
    or RANSAC (Random Sample Consensus) based on the choice. The compensation is done
    along either the x (columns) or y (rows) direction.

    Parameters:
    point_data (np.ndarray): The 2D array of depth map data.
    indexForModelFit (int): The row or column (depending on axis parameter) index to use for fitting the model.
    use_ransac (bool): If True, uses RANSAC instead of linregress. Default is False (use linregress).
    axis (int): Direction of compensation (0 for x-direction/columns, 1 for y-direction/rows).
    mode (String): Type of linear compensation model 'OffsetAndSlope', 'OffsetOnly', 'SlopeOnly'

    Returns:
    np.ndarray: The depth map array with the applied compensation.
    """

    # Check if axis is valid
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 (for columns) or 1 (for rows)")

    # Extract the relevant row or column based on the axis
    if axis == 0:  # Compensation in x-direction (columns)
        z = point_data[indexForModelFit, :]
        x = np.arange(len(z))  # Create x values corresponding to the length of the row
    else:  # Compensation in y-direction (rows)
        z = point_data[:, indexForModelFit]
        x = np.arange(len(z))  # Create x values corresponding to the length of the column

    if use_ransac:
        # Perform RANSAC regression
        x_reshaped = x.reshape(-1, 1)  # Reshape x to fit RANSAC
        ransac = linear_model.RANSACRegressor()
        ransac.fit(x_reshaped, z)
        slope, intercept = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
    else:
        # Perform regular linear regression using linregress
        slope, intercept, r_value, p_value, std_err = linregress(x, z)

    # Compute the linear model (slope * x + intercept)
    if (mode == 'OffsetOnly'):
      slope = 0
    elif (mode == 'SlopeOnly'):
      intercept = 0

    return slope, intercept


def GetLinModelForCompensation(point_data, slope, intercept, axis=0):
    # Extract the relevant row or column based on the axis
    if axis == 0:  # Compensation in x-direction (columns)
        z = point_data[0, :]
        x = np.arange(len(z))  # Create x values corresponding to the length of the row
    else:  # Compensation in y-direction (rows)
        z = point_data[:, 0]
        x = np.arange(len(z))  # Create x values corresponding to the length of the column

    # # Compute the linear model (slope * x + intercept)
    # if (mode == 'OffsetOnly'):
    #      lin_model = intercept
    # elif (mode == 'SlopeOnly'):
    #      lin_model = slope * x
    # else:  #'OffsetAndSlope'
    lin_model = slope * x + intercept

    return lin_model

def ApplyCompensationModel(point_data, lin_model, axis=0):
    # Subtract the linear model from the appropriate axis (x or y direction)
    if axis == 0:
        for row_idx in range(point_data.shape[0]):
            point_data[row_idx, :] -= lin_model  # Compensation in x-direction (columns)
    else:
        for col_idx in range(point_data.shape[1]):
            point_data[:, col_idx] -= lin_model  # Compensation in y-direction (rows)
    return point_data


def CompensateWithModel(point_data, indexForModelFit, use_ransac=True, axis=0, mode = 'OffsetAndSlope'):
    """
    Compensates each row or column of a 2D array using either linear regression (linregress)
    or RANSAC (Random Sample Consensus) based on the choice. The compensation is done
    along either the x (columns) or y (rows) direction.

    Parameters:
    point_data (np.ndarray): The 2D array of depth map data.
    indexForModelFit (int): The row or column (depending on axis parameter) index to use for fitting the model.
    use_ransac (bool): If True, uses RANSAC instead of linregress. Default is False (use linregress).
    axis (int): Direction of compensation (0 for x-direction/columns, 1 for y-direction/rows).
    mode (String): Type of linear compensation model 'OffsetAndSlope', 'OffsetOnly', 'SlopeOnly'

    Returns:
    np.ndarray: The depth map array with the applied compensation.
    """

    # # Extract the relevant row or column based on the axis
    # if axis == 0:  # Compensation in x-direction (columns)
    #     z = point_data[indexForModelFit, :]
    #     x = np.arange(len(z))  # Create x values corresponding to the length of the row
    # else:  # Compensation in y-direction (rows)
    #     z = point_data[:, indexForModelFit]
    #     x = np.arange(len(z))  # Create x values corresponding to the length of the column

    slope, intercept = CalcCompensationModel(point_data, indexForModelFit, use_ransac, axis, mode)

    lin_model = GetLinModelForCompensation(point_data, slope, intercept, axis)

    # # Compute the linear model (slope * x + intercept)
    # if (mode == 'OffsetOnly'):
    #      lin_model = intercept
    # elif (mode == 'SlopeOnly'):
    #      lin_model = slope * x
    # else:  #'OffsetAndSlope'
    #      lin_model = slope * x + intercept

    # Subtract the linear model from the appropriate axis (x or y direction)
    # if axis == 0:
    #     for row_idx in range(point_data.shape[0]):
    #         point_data[row_idx, :] -= lin_model  # Compensation in x-direction (columns)
    # else:
    #     for col_idx in range(point_data.shape[1]):
    #         point_data[:, col_idx] -= lin_model  # Compensation in y-direction (rows)

    point_data = ApplyCompensationModel(point_data, lin_model, axis)

    return point_data


def FindPhaseShift(y1, y2, max_offset = -1):
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": FindPhaseShift")

    # Get the shape of the arrays
    minlen = min(y1.shape[0], y2.shape[0])

    # Ensure y1 and y2 are 1D arrays
    y1 = np.array(y1)[0:minlen-1]
    y2 = np.array(y2)[0:minlen-1]

    if max_offset < 0:
        max_offset = minlen // 4

    min_diff = float('inf')  # Initialize the minimum difference to infinity
    best_offset = 0          # Initialize the best offset

    # Loop over possible offsets (within the max_offset range)
    for offset in range(-max_offset, max_offset + 1):
        # If offset is positive, shift y2 to the right
        if (offset >= 0):
            shifted_y2 = y2[offset:]
            diffs = np.abs(y1[:len(shifted_y2)] - shifted_y2)
        else:
            shifted_y1 = y1[abs(offset):]
            diffs = np.abs(y2[:len(shifted_y1)] - shifted_y1)

        # If the sum of differences is less than the current minimum, update the best offset
        diffs /= minlen
        diff = np.sum(diffs)
        if diff < min_diff:
            min_diff = diff
            best_offset = offset

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": FindPhaseShift - Done")

    return best_offset



def CombineArrays(arr1, arr2, combine_method='min', fill_value=np.nan):
    """
    Combine two 2D arrays of different sizes into one 2D array of the same size,
    with empty spaces filled by a given fill_value (e.g., NaN). The method for combining
    the values from the arrays can be specified, and it uses vectorized operations for efficiency.

    Parameters:
    arr1 (np.ndarray): First 2D array.
    arr2 (np.ndarray): Second 2D array.
    combine_method (str): The method to combine values ('min','mean', 'diff').
    fill_value: Value to fill the extra spaces (default is np.nan).

    Returns:
    np.ndarray: A new array with the same size as the largest array, with the original arrays'
                values placed in the corresponding positions and the rest filled with fill_value.
    """

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": CombineArrays")

    # Get the shape of the arrays
    rows1, cols1 = arr1.shape
    rows2, cols2 = arr2.shape

    # Determine the maximum number of rows and columns
    min_rows = min(rows1, rows2)
    min_cols = min(cols1, cols2)

    # Create a new array with the max size and fill it with the fill_value
    res_arr = np.full((min_rows, min_cols), fill_value)

    new_arr1 = arr1[:min_rows, :min_cols]
    new_arr2 = arr2[:min_rows, :min_cols]

    # Use vectorized operations to combine the arrays
    if combine_method == 'min':
        # Min operation: Replace NaN values with the valid values
        res_arr = np.minimum(new_arr1, new_arr2)  # Element-wise minimum of both arrays
    elif combine_method == 'mean':
        # Mean operation: Calculate the mean ignoring NaNs
        res_arr = np.nanmean([new_arr1, new_arr1], axis=0)  # Calculate mean ignoring NaNs
    elif combine_method == 'diff':
        # diff operation: Replace NaN values with the valid values
        res_arr = (new_arr1 - new_arr2)  # Element-wise minimum of both arrays


    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": CombineArrays - Done")

    return res_arr



def CombineArraysWithOffsets(arrays, column_offsets, combine_method='min', fill_value=np.nan):
    """
    Combine multiple 2D arrays with specified column offsets into a larger 2D array.

    Parameters:
    arrays (list of np.ndarray): List of 2D arrays to combine.
    column_offsets (list of int): Column offsets for each array.
    combine_method (str): 'min', 'mean', or 'diff'. (diff is meaningless for >2 arrays, so usually use 'min' or 'mean')
    fill_value: Value to fill empty spaces (default is np.nan).

    Returns:
    np.ndarray: Combined array.
    """

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": CombineArraysWithOffsets")

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
        elif combine_method == 'mean':
            existing = result[target_slice]
            # Replace fill_value with 0 temporarily
            existing_filled = np.where(np.isnan(existing), 0, existing)
            arr_filled = np.where(np.isnan(arr), 0, arr)
            result[target_slice] = existing_filled + arr_filled
            count[target_slice] += ~np.isnan(arr)
        else:
            raise ValueError(f"Unsupported combine method: {combine_method}")

    # For mean, divide by counts
    if combine_method == 'mean':
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(result, count)
            result[count == 0] = fill_value

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": CombineArraysWithOffsets - Done")

    return result




def FindBestAndWorstLinearity(data):
    """
    Find the rows and columns with the best and worst linearity in a 2D array.
    Linearity is measured using the R-squared value from linear regression.

    Parameters:
    data (np.ndarray): The 2D array of data.

    Returns:
    dict: A dictionary containing the best and worst rows and columns based on linearity.
    """
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": FindBestAndWorstLinearity")

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
        x = np.arange(cols)  # x-values (column indices)
        y = data[row_idx, :]  # y-values (data in the row)

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value**2  # R² value

        if r_squared > best_row_r2:
            best_row_r2 = r_squared
            best_row_idx = row_idx

        if r_squared < worst_row_r2:
            worst_row_r2 = r_squared
            worst_row_idx = row_idx

    # Check columns
    for col_idx in range(cols):
        x = np.arange(rows)  # x-values (row indices)
        y = data[:, col_idx]  # y-values (data in the column)

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value**2  # R² value

        if r_squared > best_col_r2:
            best_col_r2 = r_squared
            best_col_idx = col_idx

        if r_squared < worst_col_r2:
            worst_col_r2 = r_squared
            worst_col_idx = col_idx

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": FindBestAndWorstLinearity - Done")

    return {
        'best_row_idx': best_row_idx, 'best_row_r2': best_row_r2,
        'worst_row_idx': worst_row_idx, 'worst_row_r2': worst_row_r2,
        'best_col_idx': best_col_idx, 'best_col_r2': best_col_r2,
        'worst_col_idx': worst_col_idx, 'worst_col_r2': worst_col_r2
    }


# Find top plane of point cloud and level top plane as z=0
def RANSACLevelPointsToTopPlane(points, AutoLevelTolerance=450):
   print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RANSACLevelPointsToTopPlane - Start")

   # top plane model Z= aX + bY + c, see: https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
   z_offset = np.nanmax(points[:,2])
   plane_coeff_a = 0
   plane_coeff_b = 0
   plane_coeff_c = z_offset

   try:
     rows=np.where((points[:,2]>=z_offset-AutoLevelTolerance))
     top_plane_datapoints=points[rows]
     XY=top_plane_datapoints[:,:2]
     Z=top_plane_datapoints[:,2]
     median_z = np.median(Z)
     plane_coeff_c = median_z

     if np.size(Z)>4:
       print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RANSACLevelPointsToTopPlane - Calculate Z-fitting parameters")
       ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),residual_threshold=0.1)
       ransac.fit(XY,Z)
       ransac_inlier_mask = ransac.inlier_mask_
       ransac_intercept = ransac.estimator_.intercept_
       ransac_coeff = ransac.estimator_.coef_
       plane_coeff_a = ransac_coeff[0]
       plane_coeff_b = ransac_coeff[1]
       plane_coeff_c = ransac_intercept
   except Exception:
       pass

   # remove points with nan-coordinates
   points = points[points[:,2] !=  np.nan, :]
   points = points[points[:,0] !=  np.nan, :]
   points = points[points[:,1] !=  np.nan, :]

   points[:,2] -= plane_coeff_c  # correct offset in Z-direction, further improvements use all coeff to correct plane or use inlier-points for a least-minimum-square best-fit of plane

   print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RANSACLevelPointsToTopPlane - Done")
   return points


# interpolate missing points (NaNs), see: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def interpolateNaNs(data):
    #print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": interpolateNaNs - start")
    try:
      bad_indexes = np.isnan(data)
      good_indexes = np.logical_not(bad_indexes)

      if (good_indexes.size > 2):
        good_data = data[good_indexes]
        interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
        data[bad_indexes] = interpolated
    except:
      #print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": interpolateNaNs - Done")
      return data
    finally:
      #print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": interpolateNaNs - Done")
      return data


def asStride(arr,sub_shape,stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1=arr.strides[:2]
    m1,n1=arr.shape[:2]
    m2,n2=sub_shape
    view_shape=(1+(m1-m2)//stride[0],1+(n1-n2)//stride[1],m2,n2)+arr.shape[2:]
    strides=(stride[0]*s0,stride[1]*s1,s0,s1)+arr.strides[2:]
    subs=np.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    return subs


def poolingOverlap(mat,ksize,stride=None,method='max',pad=False):
    '''Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize
    if stride is None:
        stride=(ky,kx)
    sy,sx=stride

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,sy)
        nx=_ceil(n,sx)
        size=((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        mat_pad=mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

    view=asStride(mat_pad,ksize,stride)

    if method=='max':
        result=np.nanmax(view,axis=(2,3))
    else:
        result=np.nanmean(view,axis=(2,3))

    return result



# find 2D-transformation between two depth maps, scaling may speed-up registration
def RegisterPointCloudToSTL(stl_depth_map, point_cloud_depth_map, ApplyMotionType=cv2.MOTION_TRANSLATION, scaling_kernel_size=4, warp_matrix=np.eye(2, 3, dtype=np.float32), supressScaling=True, term_max_count=200, term_epsilon=-1): # default term=10000,1e-5
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RegisterPointCloudToSTL - Start")

    # Pooling of input data (data reduction)
    if scaling_kernel_size>1:
      print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RegisterPointCloudToSTL - Pooling of input data (data reduction) with factor: 1/"+str(scaling_kernel_size))
      stl_depth_map=poolingOverlap(stl_depth_map,[scaling_kernel_size, scaling_kernel_size])
      point_cloud_depth_map=poolingOverlap(point_cloud_depth_map,[scaling_kernel_size, scaling_kernel_size])
      scaling_pyramid_factor = 1 / scaling_kernel_size
    else:
      scaling_pyramid_factor = 1 #0.2


    # Convert depth maps to uint8 for ECC registration
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RegisterPointCloudToSTL - Normalize images")

    #tl_depth_map_uint8 = stl_depth_map.astype('float32')
    #point_cloud_depth_map_uint8 = point_cloud_depth_map.astype('float32')
    if (isinstance(stl_depth_map[0,0],float)):
      stl_depth_map_uint8 = cv2.normalize(stl_depth_map , None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)#.astype(np.uint8)
    else:
      stl_depth_map_uint8 = cv2.normalize(stl_depth_map.astype('float32') , None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)#.astype(np.uint8)

    if isinstance(point_cloud_depth_map[0,0],float):
      point_cloud_depth_map_uint8 = cv2.normalize(point_cloud_depth_map , None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)#.astype(np.uint8)
    else:
      point_cloud_depth_map_uint8 = cv2.normalize(point_cloud_depth_map.astype('float32') , None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)#.astype(np.uint8)

    # cv2.imwrite("D:/PROJEKTE/11-13788_SAB-Fertigungstechnik/keyence/pcd/stl.png", stl_depth_map_uint8)
    # cv2.imwrite("D:/PROJEKTE/11-13788_SAB-Fertigungstechnik/keyence/pcd/messung.png", point_cloud_depth_map_uint8)
    # cv2.namedWindow("Display 1", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("Display 2", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("Display 1", stl_depth_map_uint8)
    # cv2.imshow("Display 2", point_cloud_depth_map_uint8)
    # cv2.waitKey(0)

    # if (scaling_pyramid_factor < 1.0) and (scaling_pyramid_factor > 0.0):
    #   print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RegisterPointCloudToSTL - Scale pyramid images")
    #   point_cloud_depth_map_uint8_orig = point_cloud_depth_map_uint8.copy # copy.deepcopy(point_cloud_depth_map_uint8)
    #   stl_depth_map_uint8 = cv2.resize(stl_depth_map_uint8, (0,0), fx=scaling_pyramid_factor, fy=scaling_pyramid_factor)
    #   point_cloud_depth_map_uint8 = cv2.resize(point_cloud_depth_map_uint8, (0,0), fx=scaling_pyramid_factor, fy=scaling_pyramid_factor)

    # Perform ECC registration
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RegisterPointCloudToSTL - findTransformECC")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, term_max_count, term_epsilon)
    warp_matrix = cv2.findTransformECC(templateImage=stl_depth_map_uint8, inputImage=point_cloud_depth_map_uint8, warpMatrix=warp_matrix, motionType=ApplyMotionType, criteria=criteria)[1]

    # correct scaling (see: appendix dissertation Thomas Wiener)
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RegisterPointCloudToSTL - Retrieve and post-process warp_matrix")
    warp_matrix[0,2] /= scaling_pyramid_factor
    warp_matrix[1,2] /= scaling_pyramid_factor
    if (np.shape(warp_matrix)[0]>2):
      warp_matrix[2,0] = scaling_pyramid_factor
      warp_matrix[2,1] = scaling_pyramid_factor

    calc_scaling_factor = math.sqrt(warp_matrix[0,0]*warp_matrix[0,0]+warp_matrix[1,0]*warp_matrix[1,0])
    if (calc_scaling_factor != 1.0) and supressScaling:
      warp_matrix[0,0] /= calc_scaling_factor
      warp_matrix[1,1] /= calc_scaling_factor
      warp_matrix[0,1] /= calc_scaling_factor
      warp_matrix[1,0] /= calc_scaling_factor

    # print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RegisterPointCloudToSTL - Prepare results")
    # restore unscaled depth map of point cloud
    # if (scaling_pyramid_factor < 1.0) and (scaling_pyramid_factor > 0.0):
    #   point_cloud_depth_map_uint8 = point_cloud_depth_map_uint8_orig

    # Apply the warp matrix to the point cloud depth map
    #registered_point_cloud_depth_map = cv2.warpAffine(point_cloud_depth_map, warp_matrix, (point_cloud_depth_map.shape[1], point_cloud_depth_map.shape[0]))

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RegisterPointCloudToSTL - Done")
    print("  warp_matrix=" + str(warp_matrix))
    return warp_matrix



# Filter outliers
def DetectOutliers(data, threshold_median=0.5, threshold_gradient=0.5, kernel_size=9, use_iqr=True, use_gradient=True):
    """
    Erkennt lokale Ausreißer basierend auf einer adaptiven Medianfilterung, IQR-Analyse und/oder Gradientenmethode.

    :param data: 2D-Array mit Z-Werten der Punktwolke.
    :param threshold: Faktor für die IQR-basierte Erkennung von Ausreißern.
    :param kernel_size: Größe des Kernels für den Medianfilter (muss ungerade sein).
    :param use_iqr: Boolescher Wert, ob die IQR-Analyse genutzt werden soll.
    :param use_gradient: Boolescher Wert, ob die Gradientenmethode genutzt werden soll.
    :param crop_z_max: Punkte die in z-Richtung größer sind abschneiden.
    :return: Maske mit True für erkannte Ausreißer.
    """

    if data.dtype not in [np.float32, np.float64]:
        data = np.asarray(data, dtype=np.float32)  # Sicherstellen, dass es ein Float-Array ist #float64

    kernel_size = max(3, int(kernel_size))  # Mindestens 3, keine negativen Werte

    mask = np.zeros_like(data, dtype=bool)

    if use_iqr:
        median = median_filter(data, size=(kernel_size, kernel_size))
        diff = np.abs(data - median)
        q1 = np.percentile(diff, 25)
        q3 = np.percentile(diff, 75)
        iqr = q3 - q1
        mask |= diff > (threshold_median * iqr)

    if use_gradient:
        gradient_x = sobel(data, axis=0)
        gradient_y = sobel(data, axis=1)
        gradient_magnitude = np.hypot(gradient_x, gradient_y)

        median_gradient = median_filter(gradient_magnitude, size=(kernel_size, kernel_size))
        diff_gradient = np.abs(gradient_magnitude - median_gradient)
        q1 = 0 #np.percentile(diff_gradient, 25)
        q3 = np.percentile(diff_gradient, 75) # nur die zu steilen Anstiege sollen raus
        iqr = q3 - q1
        mask |= diff_gradient > (threshold_gradient * iqr)

    return mask

def RemoveOutliers(data, threshold_median=0.5, threshold_gradient=0.5, kernel_size=9, replace_with_nan=False, use_iqr=True, use_gradient=True, crop_z_max = None):
    """
    Entfernt lokale Ausreißer aus einer Punktwolke basierend auf adaptiver Analyse mit IQR- und/oder Gradientenmethode.

    :param data: 2D-Array mit Z-Werten der Punktwolke.
    :param threshold: Faktor für die IQR-basierte Erkennung von Ausreißern.
    :param kernel_size: Größe des Kernels für den Medianfilter.
    :param replace_with_nan: Wenn True, werden Ausreißer mit NaN ersetzt, sonst mit Median.
    :param use_iqr: Boolescher Wert, ob die IQR-Analyse genutzt werden soll.
    :param use_gradient: Boolescher Wert, ob die Gradientenmethode genutzt werden soll.
    :param crop_z_max: Werte oberhalb abschneiden [micrometer].
    :return: Bereinigtes 2D-Array.
    """
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RemoveOutliers")

    if data.dtype not in [np.float32, np.float64]:
        data = np.asarray(data, dtype=np.float32)  # Float-Array erzwingen #float64

    data_clean = data.copy()
    outliers = DetectOutliers(data, threshold_median, threshold_gradient, kernel_size, use_iqr, use_gradient)

    if replace_with_nan:
        data_clean[outliers] = np.nan
    else:
        data_clean[outliers] = median_filter(data, size=(kernel_size, kernel_size))[outliers]

    if not(crop_z_max is None):
      data_clean[data_clean > crop_z_max] = crop_z_max

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": RemoveOutliers - DONE")

    return data_clean


def SlidingAverage(z, n = 4):
    """
    Computes a sliding average for each row of the depth map.

    Args:
        depth_map (np.ndarray): The input depth map.
        n (int): The number of nearby rows to average.

    Returns:
        np.ndarray: The smoothed depth map.
    """
    rows, cols = z.shape
    smoothed_map = np.empty((rows, cols))


   #h = np.nanmean(z[:, :], axis=0)

    # Loop through each row in the depth map
    for i in range(0,rows, n//2):
        # Define the range for averaging
        start = max(0, i - n // 2)
        end = min(rows, i + n // 2 + 1)

        # Compute the average for the current row
        smoothed_map[i:i+n//2, :] = np.mean(z[start:end, :], axis=0)

    return smoothed_map

## Glättungsfilter
def gaussian_filter_2d(data, sigma):
    """
    Applies a Gaussian filter to a 2D array.

    Parameters:
    data (np.ndarray): The 2D array to be filtered.
    sigma (float or tuple): Standard deviation for the Gaussian kernel.
                            If a single float is provided, it is used for both dimensions.

    Returns:
    np.ndarray: The filtered 2D array.
    """
    filtered_data = gaussian_filter(data, sigma=sigma)
    return filtered_data

def mean_filter_2d(data, size):
    """
    Applies a mean filter to a 2D array.

    Parameters:
    data (np.ndarray): The 2D array to be filtered.
    size (int or tuple): The size of the filtering window.

    Returns:
    np.ndarray: The filtered 2D array.
    """
    kernel = np.ones((size, size)) / (size * size)
    filtered_data = convolve2d(data, kernel, mode='same', boundary='wrap')
    return filtered_data


def median_filter_2d(data, size):
    """
    Applies a median filter to a 2D array.

    Parameters:
    data (np.ndarray): The 2D array to be filtered.
    size (int or tuple): The size of the filtering window.

    Returns:
    np.ndarray: The filtered 2D array.
    """
    filtered_data = median_filter(data, size=size)
    return filtered_data

## Filter ende

# plot depth map to screen
def PlotDepthMap(depth_map, title, z_min=None, z_max=None, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, output_folder=r".\output"):
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": PlotDepthMap - Start")
    #plt.figure(dpi=400)

    # Create coordinate axes
    x = np.arange(depth_map.shape[1]) * lateral_resolution * 1e-3
    y = np.arange(depth_map.shape[0]) * lateral_resolution * 1e-3

    # Interaktive Ansicht aktivieren
    #plt.ion()

    plt.imshow(depth_map, extent=[x.min(), x.max(), y.min(), y.max()], vmin=z_min, vmax=z_max, origin='lower', cmap='jet', aspect='equal')
    plt.colorbar(label='Z [µm]')
    plt.rc('font', **{'size': 12})
    plt.title(title)
    plt.rc('font', **{'size': 10})
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')

    plt.savefig(os.path.join(output_folder, title+'.png'))

    plt.show()

    plt.close()

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": PlotDepthMap - Done")



def Plot3DDepthMaps(depth_maps, lateral_resolution=2.0, alphas=None, colormap='jet'):
    """
    Interaktive 3D-Visualisierung von mehreren Tiefenkarten als glatte Heatmap-Flächen.
    """
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": Plot3DDepthMaps - Start")


    if not isinstance(depth_maps, list):
        depth_maps = [depth_maps]

    n_maps = len(depth_maps)
    if alphas is None:
        alphas = [1.0] * n_maps

    # Interaktive Ansicht aktivieren
    plt.ion()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ny, nx = depth_maps[0].shape
    x = np.arange(nx) * lateral_resolution * 1e-3  # mm
    y = np.arange(ny) * lateral_resolution * 1e-3
    X, Y = np.meshgrid(x, y)

    mappable = None
    for i, Z in enumerate(depth_maps):
        alpha = alphas[i] if i < len(alphas) else 1.0
        #Z_scaled = Z * 1e-3  # µm → mm
        surf = ax.plot_surface(
            X, Y, Z * 1e-3,
            cmap=colormap,
            alpha=alpha,
            edgecolor='none',
            antialiased=True
        )
        mappable = surf  # für die Farbleiste

    z_min = min([np.min(dm) for dm in depth_maps])
    z_max = max([np.max(dm) for dm in depth_maps])

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title('3D')
    ax.set_box_aspect([np.ptp(x), np.ptp(y), (z_max - z_min) * 1e-3])  # Optional: schöneres Seitenverhältnis

    if mappable is not None:
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Z [mm]')

    plt.tight_layout()
    plt.show()

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": Plot3DDepthMaps - Done")


def PlotDepthMapWithCuts(z, lateral_resolution = DEPTH_MAP_RESOLUTION_XY):
    """
    Zeigt eine interaktive Höhenkarte. Bei Mausklick werden Querschnitte entlang X und Y im selben Fenster angezeigt.

    Parameter:
    - z: 2D numpy Array der Höhenwerte
    - lateral_resolution: Abstand zwischen den Pixeln in µm (Standard: 2.0)
    """
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": PlotDepthMapWithCuts - Start")


    # Generierung der X- und Y-Koordinaten basierend auf der Auflösung
    x = np.arange(z.shape[1]) * lateral_resolution
    y = np.arange(z.shape[0]) * lateral_resolution

    # Erstellen der Hauptgrafik (Heatmap) und Subplots für die Schnitte
    fig, (ax_main, ax_xcut, ax_ycut) = plt.subplots(1, 3, figsize=(18, 6))

    # Haupt-Heatmap anzeigen
    im = ax_main.imshow(
        z,
        extent=[x.min() * 1e-3, x.max() * 1e-3, y.min() * 1e-3, y.max() * 1e-3],
        origin='lower',
        cmap='jet',
        aspect='equal'
    )
    fig.colorbar(im, ax=ax_main, label='Height (µm)')
    ax_main.set_xlabel('X (mm)')
    ax_main.set_ylabel('Y (mm)')
    ax_main.set_title('Click on map to view X/Y cuts')

    # Initialisierung der Schnitt-Subplots
    ax_xcut.set_title('Horizontal Cut (Y constant)')
    ax_xcut.set_xlabel('X (µm)')
    ax_xcut.set_ylabel('Height (µm)')

    ax_ycut.set_title('Vertical Cut (X constant)')
    ax_ycut.set_xlabel('Y (µm)')
    ax_ycut.set_ylabel('Height (µm)')

    # Update-Funktion für die Schnitte
    def update_cuts(ix, iy):
        x_mm = lateral_resolution * ix * 1e-3
        y_mm = lateral_resolution * iy * 1e-3

        # Update des horizontalen Schnitts (rote Linie)
        ax_xcut.clear()
        ax_xcut.plot(x * 1e-3, z[iy, :] * 1e-3, color='red')
        ax_xcut.set_title(f'Horizontal Cut at Y = {y_mm:.2f} mm')
        ax_xcut.set_xlabel('X (mm)')
        ax_xcut.set_ylabel('Height (mm)')

        # Update des vertikalen Schnitts (blaue Linie)
        ax_ycut.clear()
        ax_ycut.plot(y * 1e-3, z[:, ix] * 1e-3, color='blue')
        ax_ycut.set_title(f'Vertical Cut at X = {x_mm:.2f} mm')
        ax_ycut.set_xlabel('Y (mm)')
        ax_ycut.set_ylabel('Height (mm)')

        # Zeige die aktualisierten Schnitte
        fig.canvas.draw()

    def onclick(event):
        if event.inaxes != ax_main:
            return

        if event.inaxes != ax_main or event.xdata is None or event.ydata is None:
            return

        # Klickposition in µm (von mm zu µm konvertieren)
        x_click, y_click = event.xdata * 1e3, event.ydata * 1e3
        ix = builtins.int(builtins.round(x_click / lateral_resolution))
        iy = builtins.int(builtins.round(y_click / lateral_resolution))

        if 0 <= ix < z.shape[1] and 0 <= iy < z.shape[0]:
            # Update der Schnitte
            update_cuts(ix, iy)



    # Interaktive Event-Handler verbinden
    fig.canvas.mpl_connect('button_press_event', onclick)

    # Bessere Layout-Anpassung, um den leeren Raum zu reduzieren
    plt.tight_layout()  # Optimiert das Layout und reduziert den leeren Raum

    #update_cuts(len(x)//2, len(y)//2)

    # Zeige das Hauptfenster mit der Heatmap und den Schnitten
    plt.show(block=True)  # Blockiert das Fenster, bis es manuell geschlossen wird

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": PlotDepthMapWithCuts - Done")




def PlotDepthAnalysis(z, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, output_folder=r".\output", y_limits_mm = None):
    """
    Create and save plots for horizontal cut, standard deviation, and max-min difference.

    Parameters:
    z (np.ndarray): 2D array of the fused depth map.
    lateral_resolution (float): Lateral resolution in micrometers (or the unit used).
    output_folder (str): Directory to save the output plots and CSV files.
    y_limits_mm (tuple): Tuple containing (y_min, y_max) limits in mm for analysis. If None, analyze the entire depth map.
    """
    print(strftime("%H:%M:%S") + ": PlotDepthAnalysis - Start")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Determine the rows to analyze based on y_limits_mm
    rows, cols = z.shape
    if y_limits_mm is not None:
        y_min, y_max = y_limits_mm
        y_min_idx = int(y_min / (lateral_resolution * 1e-3))
        y_max_idx = int(y_max / (lateral_resolution * 1e-3))

        # Ensure indices are within bounds
        y_min_idx = max(0, y_min_idx)
        y_max_idx = min(rows, y_max_idx)

        # Limit the depth map to the specified y range
        z_limited = z[y_min_idx:y_max_idx, :]
    else:
        z_limited = z

    # Step 2: Horizontal cut at the middle y-position of the limited data
    limited_rows = z_limited.shape[0]
    middle_y = limited_rows // 2
    horizontal_cut = z_limited[middle_y, :] if limited_rows > 0 else np.array([])

    # Step 3: Standard deviation for each x-position in the limited data
    std_dev = np.std(z_limited, axis=0) if limited_rows > 0 else np.array([])

    # Step 4: Maximum max-min difference for each x-position in the limited data
    max_min_diff = np.max(z_limited, axis=0) - np.min(z_limited, axis=0) if limited_rows > 0 else np.array([])

    # Step 5: Prepare x positions based on lateral resolution
    x_positions = np.arange(cols) * lateral_resolution * 1e-3  # Convert to mm

    # Step 6: Plotting
    plt.figure(figsize=(14, 8))

    # Plot horizontal cut
    plt.subplot(3, 1, 1)
    plt.plot(x_positions, horizontal_cut, label='Horizontal Cut at Y=' + f"{middle_y * lateral_resolution * 1e-3:.2f} mm", color='blue')
    plt.title('Horizontal Cut')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Depth (Z values)')
    plt.grid()
    plt.legend()

    # Plot standard deviation
    plt.subplot(3, 1, 2)
    plt.plot(x_positions, std_dev, label='Standard Deviation', color='orange')
    plt.title('Standard Deviation for Each X Position' + (f' (Y limits: {y_limits_mm[0]} mm to {y_limits_mm[1]} mm)' if y_limits_mm else ''))
    plt.xlabel('X Position (mm)')
    plt.ylabel('Standard Deviation (Z values)')
    plt.ylim(0.0,15.0)
    plt.grid()
    plt.legend()

    # Plot max-min difference
    plt.subplot(3, 1, 3)
    plt.plot(x_positions, max_min_diff, label='Max-Min Difference', color='green')
    plt.title('Max-Min Difference for Each X Position' + (f' (Y limits: {y_limits_mm[0]} mm to {y_limits_mm[1]} mm)' if y_limits_mm else ''))
    plt.xlabel('X Position (mm)')
    plt.ylabel('Max-Min Difference (Z values)')
    plt.ylim(0.0,15.0)
    plt.grid()
    plt.legend()

    plt.tight_layout()


    # Save the plots to PNG files
    plt.savefig(os.path.join(output_folder, 'depth_analysis.png'))

    plt.show()
    plt.close()  # Close the figure to free memory

    # Save standard deviation and max-min difference to CSV
    analysis_data = np.column_stack((x_positions, horizontal_cut, std_dev, max_min_diff))
    np.savetxt(os.path.join(output_folder, 'depth_analysis.csv'),
               analysis_data,
               delimiter=',',
               header='X Position (mm),Z Profile (µm),Standard Deviation (µm),Max-Min Difference (µm)',
               comments='')

    print(strftime("%H:%M:%S") + ": PlotDepthAnalysis - Done")



def SaveCuts(z, output_folder = r".\output", lateral_resolution = DEPTH_MAP_RESOLUTION_XY, y_positions_mm = [], x_positions_mm = []):
    """
    Save horizontal and vertical cuts of the depth map at specified positions.

    Parameters:
    fusioned_cloud (np.ndarray): 2D array of the fused depth map.
    output_folder (str): Directory to save the output images and CSV files.
    lateral_resolution (float): Lateral resolution in micrometers (or the unit used).
    y_positions_mm (list): List of y-positions in mm for horizontal cuts.
    x_positions_mm (list): List of x-positions in mm for vertical cuts.
    """

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": SaveCuts - Start")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert positions from mm to pixel indices
    y_positions_idx = [int(y / (lateral_resolution * 1e-3)) for y in y_positions_mm]
    x_positions_idx = [int(x / (lateral_resolution * 1e-3)) for x in x_positions_mm]

    # Step 1: Horizontal Cuts
    for y in y_positions_idx:
        if 0 <= y < z.shape[0]:  # Check bounds
            horizontal_cut = z[y, :]
            x_positions = np.arange(z.shape[1]) * lateral_resolution * 1e-3  # Convert to mm

            # Save the horizontal cut plot
            plt.figure(figsize=(10, 5))
            plt.plot(x_positions, horizontal_cut, label=f'Horizontal Cut at Y={y * lateral_resolution * 1e-3:.2f} mm', color='blue')
            plt.title('Horizontal Cut')
            plt.xlabel('X Position (mm)')
            plt.ylabel('Depth (Z values)')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(output_folder, f'horizontal_cut_y{y * lateral_resolution * 1e-3:.2f}mm.png'))

            plt.close()  # Close the figure to free memory

            # Save to CSV
            np.savetxt(os.path.join(output_folder, f'horizontal_cut_y{y * lateral_resolution * 1e-3:.2f}mm.csv'),
                       horizontal_cut,
                       delimiter=',',
                       header='Depth (Z values)',
                       comments='')

    # Step 2: Vertical Cuts
    for x in x_positions_idx:
        if 0 <= x < z.shape[1]:  # Check bounds
            vertical_cut = z[:, x]
            y_positions = np.arange(z.shape[0]) * lateral_resolution * 1e-3  # Convert to mm

            # Save the vertical cut plot
            plt.figure(figsize=(10, 5))
            plt.plot(y_positions, vertical_cut, label=f'Vertical Cut at X={x * lateral_resolution * 1e-3:.2f} mm', color='green')
            plt.title('Vertical Cut')
            plt.xlabel('Y Position (mm)')
            plt.ylabel('Depth (Z values)')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(output_folder, f'vertical_cut_x{x * lateral_resolution * 1e-3:.2f}mm.png'))
            plt.close()  # Close the figure to free memory

            # Save to CSV
            np.savetxt(os.path.join(output_folder, f'vertical_cut_x{x * lateral_resolution * 1e-3:.2f}mm.csv'),
                       vertical_cut,
                       delimiter=',',
                       header='Depth (Z values)',
                       comments='')

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": SaveCuts - Done")



def DepthMapToMesh(depthmap, poisson_depth=9, poisson_width=0, is_closed=False, fileName=''):
    """
    Convert a PointCloud to a TriangleMesh using Poisson surface reconstruction.

    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        poisson_depth (int): Depth parameter for Poisson surface reconstruction (default: 9).
        poisson_width (int): Width parameter for Poisson surface reconstruction (default: 0).

    Returns:
        o3d.geometry.TriangleMesh: The converted mesh.
    """

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Rescale point cloud and prepare point cloud data")
    scaleXY = 1000 / DEPTH_MAP_RESOLUTION_XY
    dim = np.shape(depthmap)

    # Erzeuge das Grid für x- und y-Koordinaten
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Create grid")
    x_grid, y_grid = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))

    # Berechne die Punktwolke
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Create point cloud")
    p = np.column_stack([x_grid.flatten() / scaleXY, y_grid.flatten() / scaleXY, depthmap.flatten()/1000])
    pcd = o3d.geometry.PointCloud()

    # Sweep memory
    del x_grid, y_grid

    # Befülle die Punktwolke
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Populate point cloud")
    pcd.points = o3d.utility.Vector3dVector(p)

    # Perform Poisson surface reconstruction
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Estimate Normals")
    pcd.estimate_normals()

    # Calculate colors based on z-height
    norm = plt.Normalize(p[:, 2].min(), p[:, 2].max())
    cmap = cm.jet
    colors = cmap(norm(p[:, 2]))
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # print(pcd)
    # o3d.visualization.draw_geometries([pcd],
    #                               window_name="DepthMapToMesh - PCD",
    #                               zoom=0.664,
    #                               front=[-0.4761, -0.4698, -0.7434],
    #                               lookat=[1.8900, 3.2596, 0.9284],
    #                               up=[0.2304, -0.8825, 0.4101])

    # Sweep memory
    del p, norm

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Create blank TriangleMesh")

    # Create a blank mesh
    mesh_dat = o3d.geometry.TriangleMesh()

    # Perform Poisson surface reconstruction
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Perform Poisson surface reconstruction")
    mesh_dat, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=poisson_width)

    # Extract the triangles from the mesh
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Extract the triangles from the mesh")
    mesh_dat.compute_vertex_normals()
    mesh_dat.remove_duplicated_triangles()
    mesh_dat.remove_duplicated_vertices()
    mesh_dat.remove_non_manifold_edges()

    # Remove extrapolations of Poisson algorithm
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Remove extrapolations of Poisson algorithm")
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_dat = mesh_dat.crop(bbox)

    # print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Mesh created " + str(mesh_dat))
    # o3d.visualization.draw_geometries([mesh_dat], window_name="Mesh")

    if fileName!='':
      print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Store to file " + fileName)
      o3d.io.write_triangle_mesh(mesh = mesh_dat,filename = fileName, print_progress =False)
      print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Store to file DONE")

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": DepthToMesh - Done")

    return mesh_dat


def SaveDepthMapToBinFile(filename, data):
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": SaveDepthMapToBinFile - " + filename)
    np.save(filename,data)
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": SaveDepthMapToBinFile - Done")


def LoadDepthMapFromBinFile(filename):
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": LoadDepthMapFromBinFile - " + filename)
    data = np.load(filename)
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": LoadDepthMapFromBinFile - Done")
    return data


# Multi-core parallel processing
# see: https://stackoverflow.com/questions/45526700/easy-parallelization-of-numpy-apply-along-axis
def parallelApplyAlongAxis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": multi-core parallelApplyAlongAxis")

    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": multi-core parallelApplyAlongAxis - creating chunks")
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": multi-core parallelApplyAlongAxis - multiprocessing")
    pool = multiprocessing.Pool()
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": multi-core parallelApplyAlongAxis - retrieving results")
    individual_results = pool.map(unpackingApplyAlongAxis, chunks)

    # Freeing the workers:
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": multi-core parallelApplyAlongAxis - close multi-processing and join results")
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


# Multi-core parallel processing
# https://stackoverflow.com/questions/45526700/easy-parallelization-of-numpy-apply-along-axis
def unpackingApplyAlongAxis(all_args):
    """…"""

    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)



# Convert numpy array to pandas DataFrame
def SaveToCSV(filename, data):
    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": Save data to CSV - " + filename)

    #df = pd.DataFrame(data)
    #df.to_csv('data.csv', index=False)
    #data.tofile(filename, sep=',', format='%s')
    np.savetxt(fname=filename, X=data, fmt='%s')

    print(strftime(TIME_FORMAT, time.localtime(time.time())) + ": Save data to CSV - DONE")




def read_columns(file_path, firstCol=1, secondCol=2):
    """
    Reads a file and extracts two specified columns.

    Args:
        file_path (str): Path to the input file.
        firstCol (int): Index of the first column to extract (default is 1, the 2nd column).
        secondCol (int): Index of the second column to extract (default is 2, the 3rd column).

    Returns:
        list: A list containing two lists: [first_column_data, second_column_data].

    Raises:
        FileNotFoundError: If the file does not exist.
        IndexError: If the specified column indices are out of range.
    """
    try:
        # Open the file and read lines
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Initialize lists to store column data
        first_column = []
        second_column = []

        # Skip the header line and extract the specified columns
        for line in lines[1:]:  # Skip the first line (header)
            columns = line.split()  # Split by whitespace
            # Ensure the specified columns exist in the line
            if len(columns) > max(firstCol, secondCol):
                first_column.append(float(columns[firstCol]))  # Extract first column
                second_column.append(float(columns[secondCol]))  # Extract second column
            else:
                raise IndexError(f"Column index out of range in line: {line.strip()}")

        return np.array([first_column, second_column])

    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    except ValueError:
        raise ValueError("Non-numeric data found in the specified columns.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


# Main programme
files = [r"D:\Temp\250514_Messungen\1.pcd",
         r"D:\Temp\250514_Messungen\2.pcd",
         r"D:\Temp\250514_Messungen\3.pcd",
         r"D:\Temp\250514_Messungen\4.pcd",
         r"D:\Temp\250514_Messungen\5.pcd",
         r"D:\Temp\250514_Messungen\6.pcd",
         r"D:\Temp\250514_Messungen\7.pcd",
         r"D:\Temp\250514_Messungen\8.pcd",
         r"D:\Temp\250514_Messungen\9.pcd",
         r"D:\Temp\250514_Messungen\10.pcd",
         r"D:\Temp\250514_Messungen\11.pcd",
         r"D:\Temp\250514_Messungen\12.pcd",
         r"D:\Temp\250514_Messungen\13.pcd",
         r"D:\Temp\250514_Messungen\13_2.pcd",
         r"D:\Temp\250514_Messungen\13_3.pcd",
         r"D:\Temp\250514_Messungen\14.pcd",
         r"D:\Temp\250514_Messungen\15.pcd",
         r"D:\Temp\250514_Messungen\16.pcd",
         r"D:\Temp\250514_Messungen\17.pcd",
         r"D:\Temp\250514_Messungen\18.pcd",
         r"D:\Temp\250514_Messungen\19.pcd",
         r"D:\Temp\250514_Messungen\20_1.pcd",
         r"D:\Temp\250514_Messungen\20_2.pcd",
         r"D:\Temp\250514_Messungen\21_1.pcd",
         r"D:\Temp\250514_Messungen\21_2.pcd",
         r"D:\Temp\250514_Messungen\21_3.pcd",
         r"D:\Temp\250514_Messungen\21_4.pcd",
         r"D:\Temp\250514_Messungen\22_1.pcd",
         r"D:\Temp\250514_Messungen\22_2.pcd",
         r"D:\Temp\250514_Messungen\22_3.pcd",
         r"D:\Temp\250514_Messungen\22_4.pcd",
         r"D:\Temp\250514_Messungen\Nut5.pcd",
         r"D:\Temp\250514_Messungen\Nut5_1.pcd",
         r"D:\Temp\250514_Messungen\Nut5_2.pcd",
         r"D:\Temp\250514_Messungen\Nut5_Wdh1.pcd",
         r"D:\Temp\250514_Messungen\Nut5_Wdh2.pcd",
         r"D:\Temp\250514_Messungen\Nut10.pcd",
         r"D:\Temp\250514_Messungen\Nut15.pcd",
         r"D:\Temp\250514_Messungen\Nut20.pcd",
         r"D:\Temp\250514_Messungen\Nut25.pcd"]


#y_offset_mm = [0,1,5,10,15,20,25,30,35]
#y_offset_idx = [int(y * 1000 / DEPTH_MAP_RESOLUTION_XY) for y in y_offset_mm]

# Optional: replace backslashes with forward slashes for all files
files = [f.replace("\\", r"/") for f in files]

folder = os.path.dirname(files[0])


point_clouds = []

lin_model_Y = None
lin_model_X = None
index = 1
for f in files:
    basefilename = os.path.basename(f)

    npy_file = f.replace(".pcd",f"_{DEPTH_MAP_RESOLUTION_XY}miron.npy")
    #point_cloud = LoadDepthMapFromPCDFile(f, AutoLevelPlane=False, AutoLevelTolerance=200.0, scaleY = 224.0/0.087, crop_z_min=None, crop_z_max=None, initial_crop_z_max=None, manual_z_offset=None) # F120mm/min, 1kHz, Keyence-Triggerabst=1micro
    #SaveDepthMapToBinFile(npy_file , point_cloud)
    point_cloud = LoadDepthMapFromBinFile(npy_file)

    smooth_point_cloud = SlidingAverage(point_cloud, n = 10)

    enableCompensate = False
    if enableCompensate:
      if index == 1:
        #compensations model wird hier nur für den ersten Scan ermittelt und auf alle weiteren angewendet
        #linearityResult = FindBestAndWorstLinearity(point_cloud);

        # fitIndex_xPos_idx = linearityResult['best_col_idx']
        # slopeY, interceptY = CalcCompensationModel(point_cloud, fitIndex_xPos_idx, True, 1, 'SlopeOnly')
        # lin_model_Y = GetLinModelForCompensation(point_cloud, slopeY, interceptY, 1)

        fitIndex_yPos_idx =  int(20.0e3 / DEPTH_MAP_RESOLUTION_XY)  #linearityResult['best_row_idx']
        slopeX, interceptX = CalcCompensationModel(point_cloud, fitIndex_yPos_idx, True, 0, 'OffsetAndSlope')
        lin_model_X = GetLinModelForCompensation(point_cloud, slopeX, interceptX, 0)

      # Compesation of Z-offset and tilt of sensors (calculated by linear regression along y-axis)
      point_cloud = ApplyCompensationModel(point_cloud, lin_model_X, 0)
      # if compensateExcentricity == True:
      #   point_cloud = ApplyCompensationModel(point_cloud, lin_model_Y, 0)

    rows, cols = point_cloud.shape
    max_y = DEPTH_MAP_RESOLUTION_XY * rows * 0.001


    #point_cloud = RemoveOutliers(data = point_cloud, threshold_median=0.5, threshold_gradient=0.5, kernel_size=9, replace_with_nan=False, use_iqr=True, use_gradient=False, crop_z_max = 10.0);
    #point_clouds.append(point_cloud)

    #PlotDepthMap(point_cloud, basefilename, output_folder = r'D:\Temp\250514_Messungen\output_ohneAusreisserkontrolle\\")
    #Plot3DDepthMaps([point_cloud])

    PlotDepthAnalysis(z = point_cloud, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, output_folder = r'D:\Temp\250514_Messungen\output_Mittelung10Messlinien\\' + basefilename, y_limits_mm=(5, 15))
    SaveCuts(z = point_cloud, output_folder = r'D:\Temp\250514_Messungen\output_Mittelung10Messlinien\\' + basefilename, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, y_positions_mm = np.linspace(0, max_y, 21), x_positions_mm = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 7.5])


    index = index + 1




lin_model_Y = None
lin_model_X = None
index = 1
for f in files:
    basefilename = os.path.basename(f)

    npy_file = f.replace(".pcd",f"_{DEPTH_MAP_RESOLUTION_XY}miron.npy")
    #point_cloud = LoadDepthMapFromPCDFile(f, AutoLevelPlane=False, AutoLevelTolerance=200.0, scaleY = 224.0/0.087, crop_z_min=None, crop_z_max=None, initial_crop_z_max=None, manual_z_offset=None) # F120mm/min, 1kHz, Keyence-Triggerabst=1micro
    #SaveDepthMapToBinFile(npy_file , point_cloud)
    point_cloud = LoadDepthMapFromBinFile(npy_file)

    enableCompensate = False
    if enableCompensate:
      if index == 1:
        #compensations model wird hier nur für den ersten Scan ermittelt und auf alle weiteren angewendet
        #linearityResult = FindBestAndWorstLinearity(point_cloud);

        # fitIndex_xPos_idx = linearityResult['best_col_idx']
        # slopeY, interceptY = CalcCompensationModel(point_cloud, fitIndex_xPos_idx, True, 1, 'SlopeOnly')
        # lin_model_Y = GetLinModelForCompensation(point_cloud, slopeY, interceptY, 1)

        fitIndex_yPos_idx =  int(20.0e3 / DEPTH_MAP_RESOLUTION_XY)  #linearityResult['best_row_idx']
        slopeX, interceptX = CalcCompensationModel(point_cloud, fitIndex_yPos_idx, True, 0, 'OffsetAndSlope')
        lin_model_X = GetLinModelForCompensation(point_cloud, slopeX, interceptX, 0)

      # Compesation of Z-offset and tilt of sensors (calculated by linear regression along y-axis)
      point_cloud = ApplyCompensationModel(point_cloud, lin_model_X, 0)
      # if compensateExcentricity == True:
      #   point_cloud = ApplyCompensationModel(point_cloud, lin_model_Y, 0)

    rows, cols = point_cloud.shape
    max_y = DEPTH_MAP_RESOLUTION_XY * rows * 0.001

    point_cloud = RemoveOutliers(data = point_cloud, threshold_median=0.25, threshold_gradient=0.5, kernel_size=9, replace_with_nan=False, use_iqr=True, use_gradient=True, crop_z_max = 10000.0);
    #point_clouds.append(point_cloud)

    PlotDepthMap(point_cloud, basefilename, output_folder = r'D:\Temp\250514_Messungen\\')
    #Plot3DDepthMaps([point_cloud])

    PlotDepthAnalysis(z = point_cloud, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, output_folder = r'D:\Temp\250514_Messungen\output_mitAusreisserkontrolle\\' + basefilename, y_limits_mm=(5, 15))
    SaveCuts(z = point_cloud, output_folder = r'D:\Temp\250514_Messungen\output_mitAusreisserkontrolle\\' + basefilename, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, y_positions_mm = np.linspace(0, max_y, 21), x_positions_mm = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 7.5])

    index = index + 1


lin_model_Y = None
lin_model_X = None
index = 1
for f in files:
    basefilename = os.path.basename(f)

    npy_file = f.replace(".pcd",f"_{DEPTH_MAP_RESOLUTION_XY}miron.npy")
    #point_cloud = LoadDepthMapFromPCDFile(f, AutoLevelPlane=False, AutoLevelTolerance=200.0, scaleY = 224.0/0.087, crop_z_min=None, crop_z_max=None, initial_crop_z_max=None, manual_z_offset=None) # F120mm/min, 1kHz, Keyence-Triggerabst=1micro
    #SaveDepthMapToBinFile(npy_file , point_cloud)
    point_cloud = LoadDepthMapFromBinFile(npy_file)

    smooth_point_cloud = gaussian_filter_2d(point_cloud, sigma = 2)

    enableCompensate = False
    if enableCompensate:
      if index == 1:
        #compensations model wird hier nur für den ersten Scan ermittelt und auf alle weiteren angewendet
        #linearityResult = FindBestAndWorstLinearity(point_cloud);

        # fitIndex_xPos_idx = linearityResult['best_col_idx']
        # slopeY, interceptY = CalcCompensationModel(point_cloud, fitIndex_xPos_idx, True, 1, 'SlopeOnly')
        # lin_model_Y = GetLinModelForCompensation(point_cloud, slopeY, interceptY, 1)

        fitIndex_yPos_idx =  int(20.0e3 / DEPTH_MAP_RESOLUTION_XY)  #linearityResult['best_row_idx']
        slopeX, interceptX = CalcCompensationModel(point_cloud, fitIndex_yPos_idx, True, 0, 'OffsetAndSlope')
        lin_model_X = GetLinModelForCompensation(point_cloud, slopeX, interceptX, 0)

      # Compesation of Z-offset and tilt of sensors (calculated by linear regression along y-axis)
      point_cloud = ApplyCompensationModel(point_cloud, lin_model_X, 0)
      # if compensateExcentricity == True:
      #   point_cloud = ApplyCompensationModel(point_cloud, lin_model_Y, 0)

    rows, cols = point_cloud.shape
    max_y = DEPTH_MAP_RESOLUTION_XY * rows * 0.001


    #point_cloud = RemoveOutliers(data = point_cloud, threshold_median=0.5, threshold_gradient=0.5, kernel_size=9, replace_with_nan=False, use_iqr=True, use_gradient=False, crop_z_max = 10.0);
    #point_clouds.append(point_cloud)

    #PlotDepthMap(point_cloud, basefilename, output_folder = r'D:\Temp\250514_Messungen\output_ohneAusreisserkontrolle\\")
    #Plot3DDepthMaps([point_cloud])

    PlotDepthAnalysis(z = point_cloud, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, output_folder = r'D:\Temp\250514_Messungen\output_GaussSigma2\\' + basefilename, y_limits_mm=(5, 15))
    SaveCuts(z = point_cloud, output_folder = r'D:\Temp\250514_Messungen\output_output_GaussSigma2\\' + basefilename, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, y_positions_mm = np.linspace(0, max_y, 21), x_positions_mm = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 7.5])


    index = index + 1


lin_model_Y = None
lin_model_X = None
index = 1
for f in files:
    basefilename = os.path.basename(f)

    npy_file = f.replace(".pcd",f"_{DEPTH_MAP_RESOLUTION_XY}miron.npy")
    #point_cloud = LoadDepthMapFromPCDFile(f, AutoLevelPlane=False, AutoLevelTolerance=200.0, scaleY = 224.0/0.087, crop_z_min=None, crop_z_max=None, initial_crop_z_max=None, manual_z_offset=None) # F120mm/min, 1kHz, Keyence-Triggerabst=1micro
    #SaveDepthMapToBinFile(npy_file , point_cloud)
    point_cloud = LoadDepthMapFromBinFile(npy_file)

    smooth_point_cloud = median_filter_2d(point_cloud, size = 4)

    enableCompensate = False
    if enableCompensate:
      if index == 1:
        #compensations model wird hier nur für den ersten Scan ermittelt und auf alle weiteren angewendet
        #linearityResult = FindBestAndWorstLinearity(point_cloud);

        # fitIndex_xPos_idx = linearityResult['best_col_idx']
        # slopeY, interceptY = CalcCompensationModel(point_cloud, fitIndex_xPos_idx, True, 1, 'SlopeOnly')
        # lin_model_Y = GetLinModelForCompensation(point_cloud, slopeY, interceptY, 1)

        fitIndex_yPos_idx =  int(20.0e3 / DEPTH_MAP_RESOLUTION_XY)  #linearityResult['best_row_idx']
        slopeX, interceptX = CalcCompensationModel(point_cloud, fitIndex_yPos_idx, True, 0, 'OffsetAndSlope')
        lin_model_X = GetLinModelForCompensation(point_cloud, slopeX, interceptX, 0)

      # Compesation of Z-offset and tilt of sensors (calculated by linear regression along y-axis)
      point_cloud = ApplyCompensationModel(point_cloud, lin_model_X, 0)
      # if compensateExcentricity == True:
      #   point_cloud = ApplyCompensationModel(point_cloud, lin_model_Y, 0)

    rows, cols = point_cloud.shape
    max_y = DEPTH_MAP_RESOLUTION_XY * rows * 0.001


    #point_cloud = RemoveOutliers(data = point_cloud, threshold_median=0.5, threshold_gradient=0.5, kernel_size=9, replace_with_nan=False, use_iqr=True, use_gradient=False, crop_z_max = 10.0);
    #point_clouds.append(point_cloud)

    #PlotDepthMap(point_cloud, basefilename, output_folder = r'D:\Temp\250514_Messungen\output_ohneAusreisserkontrolle\\")
    #Plot3DDepthMaps([point_cloud])

    PlotDepthAnalysis(z = point_cloud, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, output_folder = r'D:\Temp\250514_Messungen\output_MedianFilterSize4\\' + basefilename, y_limits_mm=(5, 15))
    SaveCuts(z = point_cloud, output_folder = r'D:\Temp\250514_Messungen\output_MedianFilterSize4\\' + basefilename, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, y_positions_mm = np.linspace(0, max_y, 21), x_positions_mm = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 7.5])


    index = index + 1


lin_model_Y = None
lin_model_X = None
index = 1
for f in files:
    basefilename = os.path.basename(f)

    npy_file = f.replace(".pcd",f"_{DEPTH_MAP_RESOLUTION_XY}miron.npy")
    #point_cloud = LoadDepthMapFromPCDFile(f, AutoLevelPlane=False, AutoLevelTolerance=200.0, scaleY = 224.0/0.087, crop_z_min=None, crop_z_max=None, initial_crop_z_max=None, manual_z_offset=None) # F120mm/min, 1kHz, Keyence-Triggerabst=1micro
    #SaveDepthMapToBinFile(npy_file , point_cloud)
    point_cloud = LoadDepthMapFromBinFile(npy_file)

    smooth_point_cloud = mean_filter_2d(point_cloud, size = 4)

    enableCompensate = False
    if enableCompensate:
      if index == 1:
        #compensations model wird hier nur für den ersten Scan ermittelt und auf alle weiteren angewendet
        #linearityResult = FindBestAndWorstLinearity(point_cloud);

        # fitIndex_xPos_idx = linearityResult['best_col_idx']
        # slopeY, interceptY = CalcCompensationModel(point_cloud, fitIndex_xPos_idx, True, 1, 'SlopeOnly')
        # lin_model_Y = GetLinModelForCompensation(point_cloud, slopeY, interceptY, 1)

        fitIndex_yPos_idx =  int(20.0e3 / DEPTH_MAP_RESOLUTION_XY)  #linearityResult['best_row_idx']
        slopeX, interceptX = CalcCompensationModel(point_cloud, fitIndex_yPos_idx, True, 0, 'OffsetAndSlope')
        lin_model_X = GetLinModelForCompensation(point_cloud, slopeX, interceptX, 0)

      # Compesation of Z-offset and tilt of sensors (calculated by linear regression along y-axis)
      point_cloud = ApplyCompensationModel(point_cloud, lin_model_X, 0)
      # if compensateExcentricity == True:
      #   point_cloud = ApplyCompensationModel(point_cloud, lin_model_Y, 0)

    rows, cols = point_cloud.shape
    max_y = DEPTH_MAP_RESOLUTION_XY * rows * 0.001


    #point_cloud = RemoveOutliers(data = point_cloud, threshold_median=0.5, threshold_gradient=0.5, kernel_size=9, replace_with_nan=False, use_iqr=True, use_gradient=False, crop_z_max = 10.0);
    #point_clouds.append(point_cloud)

    #PlotDepthMap(point_cloud, basefilename, output_folder = r'D:\Temp\250514_Messungen\output_ohneAusreisserkontrolle\\")
    #Plot3DDepthMaps([point_cloud])

    PlotDepthAnalysis(z = point_cloud, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, output_folder = r'D:\Temp\250514_Messungen\output_MeanFilterSize4\\' + basefilename, y_limits_mm=(5, 15))
    SaveCuts(z = point_cloud, output_folder = r'D:\Temp\250514_Messungen\output_MeanFilterSize4\\' + basefilename, lateral_resolution = DEPTH_MAP_RESOLUTION_XY, y_positions_mm = np.linspace(0, max_y, 21), x_positions_mm = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 7.5])


    index = index + 1



# fusioned_cloud = CombineArraysWithOffsets(arrays=point_clouds, column_offsets=y_offset_idx, combine_method='min', fill_value=np.nan)
# result_file = os.path.join(folder, f"fusioned_{DEPTH_MAP_RESOLUTION_XY}miron.npy")
# SaveDepthMapToBinFile(result_file , point_cloud)
# PlotDepthMapWithCuts(fusioned_cloud)
# result_fileCSV = os.path.join(folder, f"fusioned_{DEPTH_MAP_RESOLUTION_XY}miron.csv")
# SaveToCSV(result_fileCSV, fusioned_cloud)

#result_file_stl = os.path.join(folder, f"fusioned_{DEPTH_MAP_RESOLUTION_XY}miron.stl")
#stl_mesh=DepthMapToMesh(depthmap=fusioned_cloud,poisson_depth=9,poisson_width=0,fileName=result_file_stl)
#o3d.visualization.draw_geometries([stl_mesh], window_name="Mesh of fusioned sensors")