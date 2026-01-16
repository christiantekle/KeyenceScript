"""
test_pcd_processing.py

Step-by-step testing script for PCD file processing.
Tests each component individually before running full pipeline.
"""

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import your modules
try:
    from clean_depth_utils import pcd_to_points, points_to_depth_map
    from visualization import plot_depth_map, plot_histogram_cdf
    from io_utils import save_depth_map, load_depth_map
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)


def test_single_pcd_file(pcd_path, resolution=0.01):
    """Test loading and processing a single PCD file."""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(pcd_path)}")
    print(f"{'='*60}")
    
    # Step 1: Load PCD file
    print("\n[1/5] Loading PCD file...")
    try:
        points = pcd_to_points(pcd_path)
        print(f"  ✓ Loaded {points.shape[0]} points")
        print(f"  - X range: {points[:,0].min():.6f} to {points[:,0].max():.6f}")
        print(f"  - Y range: {points[:,1].min():.6f} to {points[:,1].max():.6f}")
        print(f"  - Z range: {points[:,2].min():.6f} to {points[:,2].max():.6f}")
    except Exception as e:
        print(f"  ✗ Error loading PCD: {e}")
        return None
    
    # Step 2: Check for NaN or invalid values
    print("\n[2/5] Checking data quality...")
    nan_count = np.sum(~np.isfinite(points))
    if nan_count > 0:
        print(f"  ⚠ Found {nan_count} NaN/Inf values ({nan_count/points.size*100:.2f}%)")
        points = points[np.isfinite(points).all(axis=1)]
        print(f"  → Cleaned to {points.shape[0]} points")
    else:
        print(f"  ✓ No NaN/Inf values found")
    
    # Step 3: Convert to depth map
    print(f"\n[3/5] Converting to depth map (resolution={resolution} mm)...")
    try:
        depth_map, x_edges, y_edges = points_to_depth_map(
            points, 
            resolution=resolution, 
            agg="max"
        )
        print(f"  ✓ Depth map shape: {depth_map.shape}")
        print(f"  - Size: {depth_map.shape[1]*resolution:.1f} x {depth_map.shape[0]*resolution:.1f} mm")
        print(f"  - Z range: {np.nanmin(depth_map):.6f} to {np.nanmax(depth_map):.6f}")
        
        # Check for empty cells
        nan_pixels = np.sum(np.isnan(depth_map))
        total_pixels = depth_map.size
        print(f"  - Empty cells: {nan_pixels}/{total_pixels} ({nan_pixels/total_pixels*100:.1f}%)")
        
    except Exception as e:
        print(f"  ✗ Error creating depth map: {e}")
        return None
    
    # Step 4: Basic statistics
    print("\n[4/5] Computing statistics...")
    valid_data = depth_map[np.isfinite(depth_map)]
    if valid_data.size > 0:
        print(f"  - Mean depth: {np.mean(valid_data):.6f}")
        print(f"  - Std dev: {np.std(valid_data):.6f}")
        print(f"  - Min: {np.min(valid_data):.6f}")
        print(f"  - Max: {np.max(valid_data):.6f}")
    else:
        print(f"  ✗ No valid depth data!")
        return None
    
    # Step 5: Visualization
    print("\n[5/5] Creating visualization...")
    try:
        # Plot depth map
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        im = plt.imshow(depth_map, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(im, label='Depth (mm)')
        plt.title(f'Depth Map: {os.path.basename(pcd_path)}')
        plt.xlabel('X pixels')
        plt.ylabel('Y pixels')
        
        # Plot histogram
        plt.subplot(1, 2, 2)
        plt.hist(valid_data.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Depth (mm)')
        plt.ylabel('Frequency')
        plt.title('Depth Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("  ✓ Visualization complete")
        
    except Exception as e:
        print(f"  ⚠ Visualization error: {e}")
    
    return depth_map


def test_two_pcd_alignment(pcd_path1, pcd_path2, resolution=0.01):
    """Test loading and basic alignment of two PCD files."""
    print(f"\n{'='*60}")
    print("Testing Two-File Alignment")
    print(f"{'='*60}")
    
    # Load both files
    print("\n[1/4] Loading both PCD files...")
    try:
        points1 = pcd_to_points(pcd_path1)
        points2 = pcd_to_points(pcd_path2)
        print(f"  ✓ File 1: {points1.shape[0]} points")
        print(f"  ✓ File 2: {points2.shape[0]} points")
    except Exception as e:
        print(f"  ✗ Error loading files: {e}")
        return
    
    # Convert to depth maps with common bounds
    print("\n[2/4] Converting to depth maps with common bounds...")
    try:
        # Find common bounds
        xmin = min(np.min(points1[:, 0]), np.min(points2[:, 0]))
        xmax = max(np.max(points1[:, 0]), np.max(points2[:, 0]))
        ymin = min(np.min(points1[:, 1]), np.min(points2[:, 1]))
        ymax = max(np.max(points1[:, 1]), np.max(points2[:, 1]))
        
        bounds = ((xmin, xmax), (ymin, ymax))
        print(f"  - Common X bounds: {xmin:.6f} to {xmax:.6f}")
        print(f"  - Common Y bounds: {ymin:.6f} to {ymax:.6f}")
        
        dm1, _, _ = points_to_depth_map(points1, resolution=resolution, agg="max", bounds=bounds)
        dm2, _, _ = points_to_depth_map(points2, resolution=resolution, agg="max", bounds=bounds)
        
        print(f"  ✓ Depth map 1: {dm1.shape}")
        print(f"  ✓ Depth map 2: {dm2.shape}")
        
    except Exception as e:
        print(f"  ✗ Error creating depth maps: {e}")
        return
    
    # Compare Z ranges
    print("\n[3/4] Comparing Z ranges...")
    z1_valid = dm1[np.isfinite(dm1)]
    z2_valid = dm2[np.isfinite(dm2)]
    
    if z1_valid.size > 0 and z2_valid.size > 0:
        print(f"  File 1: Z = {z1_valid.min():.6f} to {z1_valid.max():.6f}")
        print(f"  File 2: Z = {z2_valid.min():.6f} to {z2_valid.max():.6f}")
        print(f"  Z offset: {np.mean(z1_valid) - np.mean(z2_valid):.6f}")
    
    # Visualize both
    print("\n[4/4] Visualizing comparison...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Depth map 1
        im1 = axes[0, 0].imshow(dm1, cmap='viridis', origin='lower')
        plt.colorbar(im1, ax=axes[0, 0], label='Depth (mm)')
        axes[0, 0].set_title(f'File 1: {os.path.basename(pcd_path1)}')
        
        # Depth map 2
        im2 = axes[0, 1].imshow(dm2, cmap='viridis', origin='lower')
        plt.colorbar(im2, ax=axes[0, 1], label='Depth (mm)')
        axes[0, 1].set_title(f'File 2: {os.path.basename(pcd_path2)}')
        
        # Difference (where both have data)
        diff = dm1 - dm2
        im3 = axes[1, 0].imshow(diff, cmap='RdBu_r', origin='lower')
        plt.colorbar(im3, ax=axes[1, 0], label='Difference (mm)')
        axes[1, 0].set_title('Difference (File 1 - File 2)')
        
        # Histograms
        if z1_valid.size > 0:
            axes[1, 1].hist(z1_valid, bins=30, alpha=0.5, label='File 1', color='blue')
        if z2_valid.size > 0:
            axes[1, 1].hist(z2_valid, bins=30, alpha=0.5, label='File 2', color='red')
        axes[1, 1].set_xlabel('Depth (mm)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Depth Distributions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"  ⚠ Visualization error: {e}")


def main():
    """Main testing function."""
    print("\n" + "="*60)
    print("PCD File Testing Script")
    print("="*60)
    
    # Configuration
    print("\nPlease provide the path to your PCD files:")
    print("Example: E:\\YourFlashDrive\\data\\file.pcd")
    
    # Test single file
    pcd_path = input("\nEnter path to first PCD file: ").strip().strip('"')
    
    if not os.path.exists(pcd_path):
        print(f"\n✗ File not found: {pcd_path}")
        return
    
    resolution = 0.01  # 10 microns = 0.01 mm
    
    # Run single file test
    depth_map = test_single_pcd_file(pcd_path, resolution)
    
    # Ask if user wants to test alignment
    if depth_map is not None:
        test_alignment = input("\nTest alignment with another file? (y/n): ").strip().lower()
        
        if test_alignment == 'y':
            pcd_path2 = input("Enter path to second PCD file: ").strip().strip('"')
            
            if os.path.exists(pcd_path2):
                test_two_pcd_alignment(pcd_path, pcd_path2, resolution)
            else:
                print(f"✗ File not found: {pcd_path2}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()