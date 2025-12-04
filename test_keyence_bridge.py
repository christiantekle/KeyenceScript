import clr
import numpy as np
import os
from System.Collections.Generic import List  

# Path to the Keyence DLL
dll_path = r"C:\Projects\KeyenceScript\LJXAlignmentSupport.dll"

# Check if the DLL exists
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"Could not find the DLL at {dll_path}")

# Load the DLLs
try:
    # clr.AddReference("WindowsBase")  # For System.Windows.Point
    clr.AddReference(dll_path)
    print("DLLs loaded successfully!")
except Exception as e:
    print(f"Failed to load DLL: {e}")
    exit()

# Import the namespaces
try:
    from System.Windows import Point  # For Point class
    from LJX_AlignmentSupportTool.Utility import ProfileCalculator
    from LJX_AlignmentSupportTool.Data import AlignmentParameter
except Exception as e:
    print(f"Failed to import namespace: {e}")
    exit()

# Create an instance of ProfileCalculator
try:
    calculator = ProfileCalculator()
    print("ProfileCalculator initialized successfully!")

    # Create an AlignmentParameter object
    param = AlignmentParameter(0)  # DeviceId = 0, adjust as needed
    param.X1Start = 0.0  # Example values (adjust based on scanner data)
    param.X1End = 10.0
    param.X2Start = 20.0
    param.X2End = 30.0
    param.VertexX = 15.0
    param.VertexZ = 5.0
    param.AngleForSurface = 0.0

    # Create a list of points (simulating profile data)
    points = List[Point]()
    points.Add(Point(0.0, 0.0))  # Example point (x, z)
    points.Add(Point(10.0, 5.0))

    # Call a method (e.g., CalcSurfaceAlignment)
    result = calculator.CalcSurfaceAlignment(points, param)
    print("Transformation result:", result.Angle, result.OffsetX, result.OffsetZ)
except Exception as e:
    print(f"Error initializing or calling method: {e}")