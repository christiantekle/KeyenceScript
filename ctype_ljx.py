import os
import sys
import ctypes
from ctypes import c_uint8, c_uint16, c_uint32, Structure, POINTER, byref

# --- Paths ---
SDK_DIR = r"C:\Projects\KeyenceScript"
DLL_PATH = os.path.join(SDK_DIR, "LJX8_IF.dll")

# Needed for Python 3.8+ to find DLLs
if sys.version_info >= (3, 8):
    os.add_dll_directory(SDK_DIR)

# Load DLL
ljx = ctypes.WinDLL(DLL_PATH, use_last_error=True)
print("DLL loaded successfully.")

# --- Structs ---
class LJX8IF_ETHERNET_CONFIG(Structure):
    _pack_ = 1
    _fields_ = [
        ("abyIpAddress", c_uint8 * 4),
        ("wPortNo", c_uint16),
        ("reserve", c_uint8 * 2),
    ]

# --- Function signatures ---
ljx.LJX8IF_Initialize.argtypes = []
ljx.LJX8IF_Initialize.restype  = ctypes.c_int32

ljx.LJX8IF_Finalize.argtypes = []
ljx.LJX8IF_Finalize.restype  = ctypes.c_int32

ljx.LJX8IF_EthernetOpen.argtypes = [POINTER(LJX8IF_ETHERNET_CONFIG), POINTER(c_uint32)]
ljx.LJX8IF_EthernetOpen.restype  = ctypes.c_int32

ljx.LJX8IF_CommunicationClose.argtypes = [c_uint32]
ljx.LJX8IF_CommunicationClose.restype  = ctypes.c_int32

# --- Helper ---
def rc_check(name, rc):
    if rc != 0:
        raise RuntimeError(f"{name} failed with rc=0x{rc:08X}")
    else:
        print(f"{name} OK (rc=0x{rc:08X})")

# --- MAIN ---
if __name__ == "__main__":
    # 1) Initialize
    rc = ljx.LJX8IF_Initialize()
    rc_check("Initialize", rc)

    # 2) Configure Ethernet
    cfg = LJX8IF_ETHERNET_CONFIG()
    cfg.abyIpAddress[:] = (192, 168, 0, 1)  #
    cfg.wPortNo = 24691
    cfg.reserve[:] = (0, 0)

    ip_str = ".".join(map(str, cfg.abyIpAddress))
    print(f"Attempting to connect to: {ip_str}:{cfg.wPortNo}")

    # 3) Open Ethernet connection
    hDevice = c_uint32()
    rc = ljx.LJX8IF_EthernetOpen(byref(cfg), byref(hDevice))
    rc_check("EthernetOpen", rc)
    print("Device handle:", hDevice.value)

    rc = ljx.LJX8IF_CommunicationClose(hDevice)
    rc_check("CommunicationClose", rc)

    # 5) Finalize
    rc = ljx.LJX8IF_Finalize()
    rc_check("Finalize", rc)