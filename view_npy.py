"""
view_npy.py - inspect and DISPLAY a Keyence .npy capture in a window.

Usage:
    python view_npy.py scanner1.npy
    python view_npy.py scanner1.npy scanner2.npy      (compare two side by side)

This version opens an interactive matplotlib window (does NOT just save a PNG).
If a window still doesn't appear, it falls back to saving a PNG and tells you
where it is.
"""

import sys
import numpy as np


def load_and_report(path):
    arr = np.load(path)
    print(f"\nFile: {path}")
    print(f"  shape: {arr.shape}  (rows = profiles/Y, cols = X points)")
    print(f"  dtype: {arr.dtype}")
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        print("  WARNING: all NaN (no valid measurements)")
    else:
        print(f"  valid: {valid.size:,}/{arr.size:,} "
              f"({100*valid.size/arr.size:.1f}%)")
        print(f"  Z: min={valid.min():.3f}  max={valid.max():.3f}  "
              f"mean={valid.mean():.3f} mm")
        # Show a numeric corner so you see real numbers even without graphics
        print("  Top-left 4x6 corner (mm):")
        with np.printoptions(precision=3, suppress=True, nanstr="  nan "):
            print(arr[:4, :6])
    return arr


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_npy.py <file.npy> [file2.npy]")
        sys.exit(1)

    paths = sys.argv[1:]
    arrays = [load_and_report(p) for p in paths]

    try:
        import matplotlib
        # IMPORTANT: do NOT force 'Agg'. Let matplotlib pick an interactive
        # backend so a window actually opens. TkAgg ships with standard Python.
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\nmatplotlib not available: {e}")
        print("Install it with:  pip install matplotlib")
        sys.exit(0)

    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)
    for ax, arr, path in zip(axes[0], arrays, paths):
        masked = np.ma.masked_invalid(arr)
        im = ax.imshow(masked, aspect="auto", cmap="viridis",
                       interpolation="nearest")
        ax.set_xlabel("X point index (0..3199)")
        ax.set_ylabel("Profile index (Y / encoder ticks)")
        ax.set_title(f"{path}\n{arr.shape[0]} profiles x {arr.shape[1]} points")
        fig.colorbar(im, ax=ax, label="Height (mm)")
    plt.tight_layout()

    # Try to show a window. If the backend is non-interactive, save a PNG.
    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        out = paths[0].replace(".npy", "_preview.png")
        plt.savefig(out, dpi=120)
        print(f"\n(No interactive display available; saved image to {out})")
    else:
        print("\nOpening window... close it to exit.")
        plt.show()


if __name__ == "__main__":
    main()