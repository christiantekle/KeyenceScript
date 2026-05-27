"""
keyence_acquire.py --- simultaneous dual-sensor capture
================================================================

Captures from BOTH LJ-X8020 sensors at the same time, in encoder-trigger
mode, via the high-speed data channel. Saves two 2D .npy matrices.

Built on Keyence's official LJXAwrap.py.

HOW SIMULTANEOUS WORKS
----------------------
Each controller is opened with its OWN device id:
    192.168.0.1 -> deviceId 0
    192.168.0.2 -> deviceId 1
Each gets its own callback (so data lands in the right buffer).
Both are armed (Init -> PreStart -> Start), then BOTH StartMeasure calls
are issued back-to-back. Since both controllers share the same physical
encoder, their profiles fire on the same triggers and stay aligned.

We then wait until BOTH have filled their image (or timeout), stop/finalize
both, and save scanner1.npy + scanner2.npy.

Run:
    python keyence_acquire_both.py --ip1 192.168.0.1 --ip2 192.168.0.2 \
        --out1 scanner1.npy --out2 scanner2.npy --ylines 1000 --timeout 30

The encoder must be rotating during capture.
"""

import sys
import time
import argparse
import ctypes
import numpy as np

import LJXAwrap


# Per-device state. Index by deviceId (0 or 1).
_dev = {
    0: {"image_available": False, "ysize_acquired": 0, "z_val": None,
        "xsize": 0, "ysize": 0, "callback": None},
    1: {"image_available": False, "ysize_acquired": 0, "z_val": None,
        "xsize": 0, "ysize": 0, "callback": None},
}


def _make_callback(device_id: int):
    """Create a callback bound to a specific device id."""
    st = _dev[device_id]

    def callback_s_a(p_header, p_height, p_lumi,
                     luminance_enable, xpointnum, profnum, notify, user):
        if (notify == 0) or (notify == 0x10000):
            if profnum != 0 and not st["image_available"]:
                n = int(xpointnum) * int(profnum)
                st["z_val"] = [p_height[i] for i in range(n)]
                st["ysize_acquired"] = int(profnum)
                st["image_available"] = True
        return 0

    cb = LJXAwrap.LJX8IF_CALLBACK_SIMPLE_ARRAY(callback_s_a)
    st["callback"] = cb   # keep a strong reference so it isn't GC'd
    return cb


def make_cfg(ip: str):
    cfg = LJXAwrap.LJX8IF_ETHERNET_CONFIG()
    octets = [int(o) for o in ip.split(".")]
    cfg.abyIpAddress[0] = octets[0]
    cfg.abyIpAddress[1] = octets[1]
    cfg.abyIpAddress[2] = octets[2]
    cfg.abyIpAddress[3] = octets[3]
    cfg.wPortNo = 24691
    return cfg


def hx(res):
    return hex(res & 0xFFFFFFFF)


def arm_device(device_id: int, ip: str, ylines: int, high_speed_port: int):
    """Open + Init + PreStart + Start one device. Returns profinfo or None on error."""
    st = _dev[device_id]
    st["image_available"] = False
    st["ysize_acquired"]  = 0
    st["z_val"]           = None
    st["ysize"]           = ylines

    cfg = make_cfg(ip)

    res = LJXAwrap.LJX8IF_EthernetOpen(device_id, cfg)
    print(f"[dev{device_id} {ip}] EthernetOpen: {hx(res)}")
    if res != 0:
        return None

    cb = _make_callback(device_id)
    res = LJXAwrap.LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray(
        device_id, cfg, high_speed_port, cb, ylines, 0)
    print(f"[dev{device_id} {ip}] InitializeHighSpeed: {hx(res)}")
    if res != 0:
        LJXAwrap.LJX8IF_CommunicationClose(device_id)
        return None

    req = LJXAwrap.LJX8IF_HIGH_SPEED_PRE_START_REQ()
    req.bySendPosition = 2
    profinfo = LJXAwrap.LJX8IF_PROFILE_INFO()
    res = LJXAwrap.LJX8IF_PreStartHighSpeedDataCommunication(device_id, req, profinfo)
    print(f"[dev{device_id} {ip}] PreStart: {hx(res)}")
    if res != 0:
        LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(device_id)
        LJXAwrap.LJX8IF_CommunicationClose(device_id)
        return None

    st["xsize"] = profinfo.wProfileDataCount
    st["z_val"] = [0] * profinfo.wProfileDataCount * ylines
    print(f"[dev{device_id} {ip}] x_points={profinfo.wProfileDataCount}, "
          f"x_pitch={profinfo.lXPitch/100.0} um, luminance={profinfo.byLuminanceOutput}")

    res = LJXAwrap.LJX8IF_StartHighSpeedDataCommunication(device_id)
    print(f"[dev{device_id} {ip}] StartHighSpeed: {hx(res)}")
    if res != 0:
        LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(device_id)
        LJXAwrap.LJX8IF_CommunicationClose(device_id)
        return None

    return profinfo


def teardown_device(device_id: int):
    LJXAwrap.LJX8IF_StopHighSpeedDataCommunication(device_id)
    LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(device_id)
    LJXAwrap.LJX8IF_CommunicationClose(device_id)


def save_device(device_id: int, output_path: str):
    """Convert one device's captured data to a 2D mm matrix and save."""
    st = _dev[device_id]
    if not st["image_available"]:
        print(f"[dev{device_id}] no image (timeout) — not saving {output_path}")
        return False

    z_unit = ctypes.c_ushort()
    LJXAwrap.LJX8IF_GetZUnitSimpleArray(device_id, z_unit)
    z_unit_val = z_unit.value

    xsize = st["xsize"]
    ysize = st["ysize_acquired"]
    flat = np.array(st["z_val"][:xsize * ysize], dtype=np.float64)
    img = flat.reshape(ysize, xsize)

    out = np.full_like(img, np.nan, dtype=np.float32)
    valid = img != 0
    decoded = (img - 32768.0) * (z_unit_val / 100.0) / 1000.0  # -> mm
    out[valid] = decoded[valid].astype(np.float32)

    np.save(output_path, out)
    vmask = ~np.isnan(out)
    pct = 100 * vmask.sum() / out.size if out.size else 0
    print(f"[dev{device_id}] saved {output_path}  shape={out.shape}  "
          f"valid={pct:.1f}%"
          + (f"  Z={out[vmask].min():.3f}..{out[vmask].max():.3f}mm" if vmask.any() else ""))
    return True


def main():
    ap = argparse.ArgumentParser(description="Simultaneous dual-sensor capture -> two .npy")
    ap.add_argument("--ip1", default="192.168.0.1")
    ap.add_argument("--ip2", default="192.168.0.2")
    ap.add_argument("--out1", default="scanner1.npy")
    ap.add_argument("--out2", default="scanner2.npy")
    ap.add_argument("--ylines", type=int, default=1000)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--port", type=int, default=24692)
    args = ap.parse_args()

    print("=== Arming both devices ===")
    p1 = arm_device(0, args.ip1, args.ylines, args.port)
    if p1 is None:
        print("Failed to arm device 0. Aborting.")
        sys.exit(1)
    p2 = arm_device(1, args.ip2, args.ylines, args.port)
    if p2 is None:
        print("Failed to arm device 1. Cleaning up device 0.")
        teardown_device(0)
        sys.exit(1)

    print("\n=== Starting measurement on BOTH (simultaneous) ===")
    # Issue both StartMeasure calls back-to-back. The shared encoder keeps
    # the two streams aligned tick-for-tick.
    r0 = LJXAwrap.LJX8IF_StartMeasure(0)
    r1 = LJXAwrap.LJX8IF_StartMeasure(1)
    print(f"StartMeasure dev0: {hx(r0)}   dev1: {hx(r1)}")

    print(f"\nWaiting up to {args.timeout:.0f} s for BOTH images "
          f"({args.ylines} profiles each). Rotate the encoder...")
    start = time.time()
    while True:
        if _dev[0]["image_available"] and _dev[1]["image_available"]:
            print("  Both images complete.")
            break
        if time.time() - start > args.timeout:
            print("  Timeout reached.")
            break
        time.sleep(0.05)

    # Report partial progress
    print(f"  dev0 acquired: {_dev[0]['ysize_acquired']}/{args.ylines}  "
          f"dev1 acquired: {_dev[1]['ysize_acquired']}/{args.ylines}")

    print("\n=== Tearing down ===")
    teardown_device(0)
    teardown_device(1)

    print("\n=== Saving ===")
    ok1 = save_device(0, args.out1)
    ok2 = save_device(1, args.out2)

    if ok1 and ok2:
        print("\n[SUCCESS] Both sensors captured simultaneously.")
    else:
        print("\n[PARTIAL] Not both sensors produced an image — check encoder motion / targets.")


if __name__ == "__main__":
    main()