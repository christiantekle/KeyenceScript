"""
keyence_gui.py - Qt control panel for the Keyence dual-sensor rig
=================================================================

Maps 1:1 to the supervisor's minimum requirements:

  1. CAPTURE        Simultaneous acquisition from both LJ-X8020 sensors in
                    encoder-trigger mode (uses keyence_acquire.py internals).
  2. TARE           Per-sensor zero-line offset. Subtracts a robust reference
                    height (median of valid pixels) so the surface reads ~0.
  3. MERGE          Combines sensor1 + sensor2 with the "minimum" method and
                    histogram-based outlier filtering. Uses the EXISTING
                    merge_depth_maps() from merge_pcd.py and remove_outliers()
                    from filters.py.
  4. PLOT           1D (single laser line), 2D heatmap, optional 3D surface,
                    for sensor1, sensor2, or the merged field. Plots open in
                    separate matplotlib windows with their own zoom/save
                    toolbar.

Design notes
------------
- PyQt5. One main window. State-driven button enables (you can't merge
  before capturing, etc).
- Acquisition runs on a QThread worker so the GUI never freezes.
- File saves use a native Save As dialog every time (no silent auto-save).
- All operations log to a scrolling text box at the bottom.

Required files in the same folder
---------------------------------
- LJXAwrap.py and LJX8_IF.dll       (Keyence)
- keyence_acquire.py                (our acquisition module)
- filters.py                        (provides remove_outliers)
- merge_pcd.py                      (provides merge_depth_maps)

Install
-------
    pip install PyQt5 matplotlib numpy

Run
---
    python keyence_gui.py
"""

import sys
import time
import traceback
import datetime

import numpy as np

# -- Qt --------------------------------------------------------------------
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# -- Matplotlib (separate windows, not embedded) ---------------------------
import matplotlib
# Use Qt5 backend so plot windows integrate cleanly with the GUI
try:
    matplotlib.use("Qt5Agg")
except Exception:
    pass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection (used by plot)
_ = Axes3D  # silence "unused" linters


# -- Existing project modules (defensive imports) --------------------------
_IMPORT_ERRORS = []

try:
    import keyence_acquire as kacq
except Exception as e:
    kacq = None
    _IMPORT_ERRORS.append(f"keyence_acquire.py: {e}")

try:
    from filters import remove_outliers
except Exception as e:
    remove_outliers = None
    _IMPORT_ERRORS.append(f"filters.py / remove_outliers: {e}")

try:
    from merge_pcd import merge_depth_maps
except Exception as e:
    merge_depth_maps = None
    _IMPORT_ERRORS.append(f"merge_pcd.py / merge_depth_maps: {e}")


# =========================================================================
# Background capture worker (QThread + signals)
# =========================================================================

class CaptureWorker(QtCore.QObject):
    """
    Runs keyence_acquire's arm/start/wait/teardown sequence on a Qt worker
    thread. Communicates back to the GUI via Qt signals (thread-safe by
    construction).
    """

    log         = pyqtSignal(str)
    status      = pyqtSignal(int, str)  # sensor index (0 or 1), text
    error       = pyqtSignal(str)
    finished    = pyqtSignal(object, object)  # arr1, arr2 (np.ndarray or None)

    def __init__(self, ip1, ip2, ylines, timeout, port=24692):
        super().__init__()
        self.ip1 = ip1
        self.ip2 = ip2
        self.ylines = ylines
        self.timeout = timeout
        self.port = port
        self._stop = False

    def cancel(self):
        self._stop = True

    @QtCore.pyqtSlot()
    def run(self):
        if kacq is None:
            self.error.emit("keyence_acquire module not available")
            self.finished.emit(None, None)
            return
        try:
            self.log.emit("=== Arming both devices ===")
            self.status.emit(0, "arming...")
            self.status.emit(1, "arming...")

            p1 = kacq.arm_device(0, self.ip1, self.ylines, self.port)
            if p1 is None:
                self.log.emit("[ERROR] Failed to arm sensor 1.")
                self.finished.emit(None, None)
                return
            self.status.emit(0, f"armed (x={p1.wProfileDataCount} pts)")

            p2 = kacq.arm_device(1, self.ip2, self.ylines, self.port)
            if p2 is None:
                self.log.emit("[ERROR] Failed to arm sensor 2. Cleaning up sensor 1.")
                kacq.teardown_device(0)
                self.finished.emit(None, None)
                return
            self.status.emit(1, f"armed (x={p2.wProfileDataCount} pts)")

            self.log.emit("=== Starting measurement on BOTH (simultaneous) ===")
            r0 = kacq.LJXAwrap.LJX8IF_StartMeasure(0)
            r1 = kacq.LJXAwrap.LJX8IF_StartMeasure(1)
            self.log.emit(f"StartMeasure dev0: {kacq.hx(r0)}   dev1: {kacq.hx(r1)}")

            self.log.emit(f"Waiting up to {self.timeout:.0f} s for "
                          f"{self.ylines} profiles each...")
            self.status.emit(0, "waiting for data...")
            self.status.emit(1, "waiting for data...")

            start = time.time()
            while True:
                if self._stop:
                    self.log.emit("[CANCEL] Stop requested.")
                    break
                if (kacq._dev[0]["image_available"]
                        and kacq._dev[1]["image_available"]):
                    self.log.emit("Both images complete.")
                    break
                if time.time() - start > self.timeout:
                    self.log.emit("[TIMEOUT] reached.")
                    break
                time.sleep(0.05)

            a0 = kacq._dev[0]["ysize_acquired"]
            a1 = kacq._dev[1]["ysize_acquired"]
            self.log.emit(f"  dev0 acquired: {a0}/{self.ylines}   "
                          f"dev1 acquired: {a1}/{self.ylines}")
            self.status.emit(0, f"{a0}/{self.ylines} profiles")
            self.status.emit(1, f"{a1}/{self.ylines} profiles")

            self.log.emit("=== Tearing down ===")
            kacq.teardown_device(0)
            kacq.teardown_device(1)

            def to_mm(dev_id):
                st = kacq._dev[dev_id]
                if not st["image_available"]:
                    return None
                z_unit = kacq.ctypes.c_ushort()
                kacq.LJXAwrap.LJX8IF_GetZUnitSimpleArray(dev_id, z_unit)
                z_unit_val = z_unit.value
                xsize = st["xsize"]
                ysize = st["ysize_acquired"]
                flat = np.array(st["z_val"][:xsize * ysize], dtype=np.float64)
                img = flat.reshape(ysize, xsize)
                out = np.full_like(img, np.nan, dtype=np.float32)
                valid = img != 0
                decoded = (img - 32768.0) * (z_unit_val / 100.0) / 1000.0
                out[valid] = decoded[valid].astype(np.float32)
                return out

            arr1 = to_mm(0)
            arr2 = to_mm(1)

            for label, arr in [("Sensor 1", arr1), ("Sensor 2", arr2)]:
                if arr is None:
                    self.log.emit(f"  {label}: no image acquired.")
                else:
                    v = arr[~np.isnan(arr)]
                    if v.size:
                        self.log.emit(
                            f"  {label}: shape={arr.shape}  "
                            f"valid={100*v.size/arr.size:.1f}%  "
                            f"Z={v.min():.3f}..{v.max():.3f}mm")
                    else:
                        self.log.emit(f"  {label}: shape={arr.shape} (all NaN)")

            self.finished.emit(arr1, arr2)

        except Exception:
            self.error.emit(traceback.format_exc())
            self.finished.emit(None, None)


# =========================================================================
# Main window
# =========================================================================

class KeyenceGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keyence dual-sensor control")
        self.resize(820, 820)

        # ----- State -----
        self.arr1 = None
        self.arr2 = None
        self.tare1 = 0.0
        self.tare2 = 0.0
        self.merged = None
        self.worker = None
        self.worker_thread = None
        self._last_fig = None

        self._build_ui()
        self._refresh_state()

        if _IMPORT_ERRORS:
            self._log("[STARTUP] Some imports failed:")
            for e in _IMPORT_ERRORS:
                self._log("  - " + e)
            self._log("Some features may not work until those files are fixed.")
        else:
            self._log("Ready. Click CAPTURE BOTH to begin.")

    # --------------------------------------------------------------
    # Widget construction
    # --------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # === ACQUISITION ===
        gb = QtWidgets.QGroupBox("Acquisition")
        outer.addWidget(gb)
        v = QtWidgets.QVBoxLayout(gb)

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        h.addWidget(QtWidgets.QLabel("Sensor 1 IP:"))
        self.ed_ip1 = QtWidgets.QLineEdit("192.168.0.1"); self.ed_ip1.setFixedWidth(140)
        h.addWidget(self.ed_ip1)
        h.addSpacing(20)
        h.addWidget(QtWidgets.QLabel("Sensor 2 IP:"))
        self.ed_ip2 = QtWidgets.QLineEdit("192.168.0.2"); self.ed_ip2.setFixedWidth(140)
        h.addWidget(self.ed_ip2)
        h.addStretch()

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        h.addWidget(QtWidgets.QLabel("Y-lines (profiles per capture):"))
        self.sp_ylines = QtWidgets.QSpinBox(); self.sp_ylines.setRange(1, 200000)
        self.sp_ylines.setValue(1000); self.sp_ylines.setFixedWidth(100)
        h.addWidget(self.sp_ylines)
        h.addSpacing(20)
        h.addWidget(QtWidgets.QLabel("Timeout (s):"))
        self.sp_timeout = QtWidgets.QDoubleSpinBox()
        self.sp_timeout.setRange(1.0, 3600.0); self.sp_timeout.setValue(30.0)
        self.sp_timeout.setDecimals(1); self.sp_timeout.setFixedWidth(100)
        h.addWidget(self.sp_timeout)
        h.addStretch()

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        self.btn_capture = QtWidgets.QPushButton("CAPTURE BOTH")
        self.btn_capture.setMinimumHeight(34)
        self.btn_capture.clicked.connect(self._on_capture)
        h.addWidget(self.btn_capture)
        self.btn_stop = QtWidgets.QPushButton("STOP")
        self.btn_stop.setMinimumHeight(34)
        self.btn_stop.clicked.connect(self._on_stop)
        h.addWidget(self.btn_stop)
        h.addStretch()

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        h.addWidget(QtWidgets.QLabel("Sensor 1:"))
        self.lbl_status1 = QtWidgets.QLabel("-- no data --")
        self.lbl_status1.setStyleSheet("color: navy;"); self.lbl_status1.setMinimumWidth(240)
        h.addWidget(self.lbl_status1)
        h.addSpacing(20)
        h.addWidget(QtWidgets.QLabel("Sensor 2:"))
        self.lbl_status2 = QtWidgets.QLabel("-- no data --")
        self.lbl_status2.setStyleSheet("color: navy;"); self.lbl_status2.setMinimumWidth(240)
        h.addWidget(self.lbl_status2)
        h.addStretch()

        # === TARE ===
        gb = QtWidgets.QGroupBox("Tare (zero-line)")
        outer.addWidget(gb)
        v = QtWidgets.QVBoxLayout(gb)

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        self.btn_tare1 = QtWidgets.QPushButton("Tare Sensor 1")
        self.btn_tare1.clicked.connect(lambda: self._on_tare(1))
        h.addWidget(self.btn_tare1)
        self.btn_tare2 = QtWidgets.QPushButton("Tare Sensor 2")
        self.btn_tare2.clicked.connect(lambda: self._on_tare(2))
        h.addWidget(self.btn_tare2)
        self.btn_tare_clear = QtWidgets.QPushButton("Clear Tare")
        self.btn_tare_clear.clicked.connect(self._on_clear_tare)
        h.addWidget(self.btn_tare_clear)
        h.addStretch()

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        self.lbl_tare1 = QtWidgets.QLabel("S1 offset: 0.000 mm")
        h.addWidget(self.lbl_tare1)
        h.addSpacing(30)
        self.lbl_tare2 = QtWidgets.QLabel("S2 offset: 0.000 mm")
        h.addWidget(self.lbl_tare2)
        h.addStretch()

        # === MERGE ===
        gb = QtWidgets.QGroupBox("Merge")
        outer.addWidget(gb)
        v = QtWidgets.QVBoxLayout(gb)

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        h.addWidget(QtWidgets.QLabel("Method:"))
        self.cb_method = QtWidgets.QComboBox()
        self.cb_method.addItems(["min", "max", "mean"])
        self.cb_method.setCurrentText("min")
        self.cb_method.setFixedWidth(90)
        h.addWidget(self.cb_method)
        h.addSpacing(20)
        self.chk_filter = QtWidgets.QCheckBox("Outlier filter (histogram/IQR)")
        self.chk_filter.setChecked(True)
        h.addWidget(self.chk_filter)
        h.addStretch()

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        h.addWidget(QtWidgets.QLabel("Threshold:"))
        self.sp_thresh = QtWidgets.QDoubleSpinBox()
        self.sp_thresh.setRange(0.0, 1000.0); self.sp_thresh.setDecimals(3)
        self.sp_thresh.setValue(0.5); self.sp_thresh.setSingleStep(0.05)
        self.sp_thresh.setFixedWidth(90)
        h.addWidget(self.sp_thresh)
        h.addSpacing(20)
        h.addWidget(QtWidgets.QLabel("Kernel size:"))
        self.sp_kernel = QtWidgets.QSpinBox()
        self.sp_kernel.setRange(1, 99); self.sp_kernel.setValue(9)
        self.sp_kernel.setFixedWidth(70)
        h.addWidget(self.sp_kernel)
        h.addStretch()

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        self.btn_merge = QtWidgets.QPushButton("MERGE")
        self.btn_merge.setMinimumHeight(30)
        self.btn_merge.clicked.connect(self._on_merge)
        h.addWidget(self.btn_merge)
        h.addStretch()

        # === VIEW ===
        gb = QtWidgets.QGroupBox("View")
        outer.addWidget(gb)
        v = QtWidgets.QVBoxLayout(gb)

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        h.addWidget(QtWidgets.QLabel("Source:"))
        self.rb_src_group = QtWidgets.QButtonGroup(self)
        self.rb_s1 = QtWidgets.QRadioButton("Sensor 1")
        self.rb_s2 = QtWidgets.QRadioButton("Sensor 2")
        self.rb_merged = QtWidgets.QRadioButton("Merged")
        self.rb_merged.setChecked(True)
        for rb in (self.rb_s1, self.rb_s2, self.rb_merged):
            self.rb_src_group.addButton(rb)
            h.addWidget(rb)
        h.addStretch()

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        h.addWidget(QtWidgets.QLabel("Mode:"))
        self.rb_mode_group = QtWidgets.QButtonGroup(self)
        self.rb_2d = QtWidgets.QRadioButton("2D heatmap"); self.rb_2d.setChecked(True)
        self.rb_1d = QtWidgets.QRadioButton("1D line")
        self.rb_3d = QtWidgets.QRadioButton("3D surface (optional)")
        for rb in (self.rb_2d, self.rb_1d, self.rb_3d):
            self.rb_mode_group.addButton(rb)
            h.addWidget(rb)
        h.addStretch()

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        h.addWidget(QtWidgets.QLabel("1D row index:"))
        self.sp_row = QtWidgets.QSpinBox()
        self.sp_row.setRange(0, 9999999); self.sp_row.setValue(0)
        self.sp_row.setFixedWidth(90)
        h.addWidget(self.sp_row)
        h.addWidget(QtWidgets.QLabel("(used only for 1D mode; 0 = first profile)"))
        h.addStretch()

        h = QtWidgets.QHBoxLayout(); v.addLayout(h)
        self.btn_plot = QtWidgets.QPushButton("SHOW PLOT")
        self.btn_plot.setMinimumHeight(30)
        self.btn_plot.clicked.connect(self._on_plot)
        h.addWidget(self.btn_plot)
        self.btn_save_npy = QtWidgets.QPushButton("Save .npy...")
        self.btn_save_npy.clicked.connect(self._on_save_npy)
        h.addWidget(self.btn_save_npy)
        self.btn_save_png = QtWidgets.QPushButton("Save current plot as PNG...")
        self.btn_save_png.clicked.connect(self._on_save_png)
        h.addWidget(self.btn_save_png)
        h.addStretch()

        # === LOG ===
        gb = QtWidgets.QGroupBox("Log")
        outer.addWidget(gb, 1)  # stretch=1 so log expands to fill window
        v = QtWidgets.QVBoxLayout(gb)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        mono = QtGui.QFont("Consolas"); mono.setStyleHint(QtGui.QFont.Monospace); mono.setPointSize(9)
        self.log.setFont(mono)
        v.addWidget(self.log)

    # --------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------
    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")
        # auto-scroll
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    # --------------------------------------------------------------
    # Enable/disable logic based on current state
    # --------------------------------------------------------------
    def _refresh_state(self):
        capturing = self.worker_thread is not None and self.worker_thread.isRunning()

        self.btn_capture.setEnabled(not capturing)
        self.btn_stop.setEnabled(capturing)

        has1 = self.arr1 is not None
        has2 = self.arr2 is not None

        self.btn_tare1.setEnabled(has1 and not capturing)
        self.btn_tare2.setEnabled(has2 and not capturing)
        self.btn_tare_clear.setEnabled((self.tare1 != 0 or self.tare2 != 0) and not capturing)

        self.btn_merge.setEnabled(
            has1 and has2 and not capturing and merge_depth_maps is not None)

        any_data = has1 or has2 or self.merged is not None
        self.btn_plot.setEnabled(any_data and not capturing)
        self.btn_save_npy.setEnabled(any_data and not capturing)
        self.btn_save_png.setEnabled(self._last_fig is not None and not capturing)

    # --------------------------------------------------------------
    # Tare helpers
    # --------------------------------------------------------------
    def _tared(self, arr, tare):
        if arr is None:
            return None
        if tare == 0:
            return arr
        return arr - tare

    def _selected_array(self):
        if self.rb_s1.isChecked():
            label = "Sensor 1 (tared)" if self.tare1 else "Sensor 1"
            return self._tared(self.arr1, self.tare1), label
        if self.rb_s2.isChecked():
            label = "Sensor 2 (tared)" if self.tare2 else "Sensor 2"
            return self._tared(self.arr2, self.tare2), label
        return self.merged, "Merged"

    # --------------------------------------------------------------
    # Capture button
    # --------------------------------------------------------------
    def _on_capture(self):
        if kacq is None:
            QtWidgets.QMessageBox.critical(
                self, "Missing module",
                "keyence_acquire.py could not be imported.\nSee the log for details.")
            return

        ip1 = self.ed_ip1.text().strip()
        ip2 = self.ed_ip2.text().strip()
        ylines = self.sp_ylines.value()
        timeout = self.sp_timeout.value()

        # Build worker and thread
        self.worker = CaptureWorker(ip1, ip2, ylines, timeout, 24692)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        # Wire signals (these are thread-safe by Qt design)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self._log)
        self.worker.status.connect(self._on_status)
        self.worker.error.connect(self._on_worker_error)
        self.worker.finished.connect(self._on_worker_finished)
        # Make sure the thread shuts down when work completes
        self.worker.finished.connect(self.worker_thread.quit)

        self.lbl_status1.setText("starting...")
        self.lbl_status2.setText("starting...")
        self.worker_thread.start()
        self._refresh_state()

    def _on_stop(self):
        if self.worker is not None:
            self.worker.cancel()
            self._log("Cancel requested.")

    @QtCore.pyqtSlot(int, str)
    def _on_status(self, sensor_idx, text):
        (self.lbl_status1 if sensor_idx == 0 else self.lbl_status2).setText(text)

    @QtCore.pyqtSlot(str)
    def _on_worker_error(self, tb_text):
        self._log("[ERROR]\n" + tb_text)

    @QtCore.pyqtSlot(object, object)
    def _on_worker_finished(self, arr1, arr2):
        if arr1 is not None:
            self.arr1 = arr1
        if arr2 is not None:
            self.arr2 = arr2
        # New capture invalidates the previous merge
        self.merged = None
        # Wait for the QThread to fully stop before clearing references
        if self.worker_thread is not None:
            self.worker_thread.wait(2000)
        self.worker = None
        self.worker_thread = None
        self._refresh_state()

    # --------------------------------------------------------------
    # Tare buttons
    # --------------------------------------------------------------
    def _on_tare(self, which):
        arr = self.arr1 if which == 1 else self.arr2
        if arr is None:
            return
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            self._log(f"Sensor {which}: no valid pixels - cannot tare.")
            return
        ref = float(np.median(valid))  # robust center: ignores edges/outliers
        if which == 1:
            self.tare1 = ref
            self.lbl_tare1.setText(f"S1 offset: {ref:+.3f} mm")
        else:
            self.tare2 = ref
            self.lbl_tare2.setText(f"S2 offset: {ref:+.3f} mm")
        self._log(f"Sensor {which} tared to {ref:+.3f} mm (median of valid).")
        self.merged = None  # any previous merge is now stale
        self._refresh_state()

    def _on_clear_tare(self):
        self.tare1 = 0.0
        self.tare2 = 0.0
        self.lbl_tare1.setText("S1 offset: 0.000 mm")
        self.lbl_tare2.setText("S2 offset: 0.000 mm")
        self.merged = None
        self._log("Tare cleared on both sensors.")
        self._refresh_state()

    # --------------------------------------------------------------
    # Merge button
    # --------------------------------------------------------------
    def _on_merge(self):
        if merge_depth_maps is None:
            QtWidgets.QMessageBox.critical(
                self, "Missing module",
                "merge_pcd.merge_depth_maps could not be imported.")
            return
        if self.arr1 is None or self.arr2 is None:
            return

        a1 = self._tared(self.arr1, self.tare1)
        a2 = self._tared(self.arr2, self.tare2)

        if self.chk_filter.isChecked():
            if remove_outliers is None:
                self._log("[WARN] outlier filter requested but filters.py "
                          "not available - skipping filter.")
            else:
                try:
                    thr = self.sp_thresh.value()
                    ks = self.sp_kernel.value()
                    a1 = remove_outliers(a1, threshold_median=thr, kernel_size=ks)
                    a2 = remove_outliers(a2, threshold_median=thr, kernel_size=ks)
                    self._log(f"Outlier filter applied "
                              f"(threshold={thr}, kernel={ks}).")
                except Exception as e:
                    self._log(f"[WARN] outlier filter failed: {e}")

        try:
            method = self.cb_method.currentText()
            merged = merge_depth_maps(a1, a2, method=method)
            self.merged = merged.astype(np.float32)
            v = self.merged[~np.isnan(self.merged)]
            if v.size:
                self._log(f"Merged ({method}): shape={self.merged.shape}  "
                          f"valid={100*v.size/self.merged.size:.1f}%  "
                          f"Z={v.min():.3f}..{v.max():.3f}mm")
            else:
                self._log(f"Merged ({method}): shape={self.merged.shape}  (all NaN)")
        except Exception as e:
            self._log(f"[ERROR] merge failed: {e}")
            traceback.print_exc()
            return

        self._refresh_state()

    # --------------------------------------------------------------
    # Plot button
    # --------------------------------------------------------------
    def _on_plot(self):
        arr, label = self._selected_array()
        if arr is None:
            QtWidgets.QMessageBox.information(
                self, "Nothing to plot",
                "There is no data for the selected source yet.")
            return

        try:
            if self.rb_2d.isChecked():
                fig, ax = plt.subplots(figsize=(11, 5))
                masked = np.ma.masked_invalid(arr)
                im = ax.imshow(masked, aspect="auto", cmap="viridis",
                               interpolation="nearest")
                ax.set_xlabel("X point index")
                ax.set_ylabel("Profile (Y / encoder tick)")
                ax.set_title(f"{label} - shape {arr.shape}")
                fig.colorbar(im, ax=ax, label="Height (mm)")
                fig.tight_layout()
                fig.show()

            elif self.rb_1d.isChecked():
                idx = self.sp_row.value()
                if idx < 0 or idx >= arr.shape[0]:
                    QtWidgets.QMessageBox.critical(
                        self, "Out of range",
                        f"Row {idx} is outside 0..{arr.shape[0]-1}.")
                    return
                fig, ax = plt.subplots(figsize=(11, 4))
                row = arr[idx, :]
                ax.plot(np.arange(arr.shape[1]), row, lw=1)
                ax.set_xlabel("X point index")
                ax.set_ylabel("Height (mm)")
                ax.set_title(f"{label} - profile row {idx}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.show()

            elif self.rb_3d.isChecked():
                # Downsample so 3D doesn't choke on (1000, 3200)
                step_y = max(1, arr.shape[0] // 200)
                step_x = max(1, arr.shape[1] // 200)
                z = arr[::step_y, ::step_x]
                y = np.arange(0, arr.shape[0], step_y)
                x = np.arange(0, arr.shape[1], step_x)
                X, Y = np.meshgrid(x, y)
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(X, Y, z, cmap="viridis",
                                linewidth=0, antialiased=False)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Height (mm)")
                ax.set_title(f"{label} (downsampled {step_y}x{step_x})")
                fig.tight_layout()
                fig.show()

            self._last_fig = fig
            self._refresh_state()

        except Exception as e:
            self._log(f"[ERROR] plot failed: {e}")
            traceback.print_exc()

    # --------------------------------------------------------------
    # Save buttons
    # --------------------------------------------------------------
    def _on_save_npy(self):
        arr, label = self._selected_array()
        if arr is None:
            QtWidgets.QMessageBox.information(
                self, "Nothing to save",
                "There is no data for the selected source yet.")
            return
        default = {"Sensor 1 (tared)": "scanner1.npy",
                   "Sensor 1": "scanner1.npy",
                   "Sensor 2 (tared)": "scanner2.npy",
                   "Sensor 2": "scanner2.npy",
                   "Merged": "merged.npy"}.get(label, "data.npy")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save .npy", default,
            "NumPy array (*.npy);;All files (*.*)")
        if not path:
            return
        np.save(path, arr.astype(np.float32))
        self._log(f"Saved {path}  shape={arr.shape}")

    def _on_save_png(self):
        if self._last_fig is None:
            QtWidgets.QMessageBox.information(
                self, "No plot",
                "Click SHOW PLOT first, then save the PNG.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save plot as PNG", "plot.png",
            "PNG image (*.png);;All files (*.*)")
        if not path:
            return
        self._last_fig.savefig(path, dpi=140, bbox_inches="tight")
        self._log(f"Saved plot to {path}")


# =========================================================================
# Entry point
# =========================================================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    # Subtle modern look
    app.setStyle("Fusion")
    w = KeyenceGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()