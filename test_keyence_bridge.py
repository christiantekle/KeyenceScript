import clr #from pythonnet package
import numpy as np
import os

dll_path = "C:\Projects\LJ-X8000A_AlignmentSupportTool_1_0_0-01\Sample_AlignmentSupportTool\exe\x64\LJXAlignmentSupport.dll"

if not os.path.exists(dll_path):
    raise FileNotFoundError(f"Could not find the DLL at {dll_path}")    

#Load the dll
clr.AddReference(dll_path)

#import the namespace (Keyence SDK)
from Keyence.LJX8XX import LJX8XXAlignmentSupport

#create an instance of the LJX8XXAlignmentSupport class
alignment_support = LJX8XXAlignmentSupport()

#call methods from the DLL class
print("LJX8XXAlignmentSupport version:", alignment_support.GetScannerName())
