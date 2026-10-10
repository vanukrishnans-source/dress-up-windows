"""Dress Up for Windows - one fixed screen, no scrolling. Try-on runs in a separate worker process
(DressUp_cli.exe --tryon), so a GPU driver crash or Cancel never takes the window down."""
import os, sys, json, time, tempfile
from pathlib import Path
from PySide6.QtCore import Qt, QProcess, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
                               QFileDialog, QProgressBar, QButtonGroup, QMessageBox, QSizePolicy)
from . import __version__

IMG_FILTER = "Images (*.jpg *.jpeg *.png *.webp *.bmp)"
CSS = """
QWidget{background:#16181d;color:#e8e8ea;font-size:15px}
QPushButton{background:#2a2e37;border:1px solid #3a3f4b;border-radius:8px;padding:8px 14px;min-height:30px}
QPushButton:hover{background:#343945} QPushButton:checked{background:#c2185b;border-color:#e91e63}
QPushButton:disabled{color:#666}
QPushButton#go{background:#e91e63;font-weight:bold;font-size:17px} QPushButton#go:disabled{background:#4a2a36}
QLabel#pane{background:#0e0f12;border:2px dashed #3a3f4b;border-radius:10px;color:#888}
QProgressBar{border:1px solid #3a3f4b;border-radius:6px;text-align:center;height:22px}
QProgressBar::chunk{background:#e91e63;border-radius:6px}
"""

def _worker_cmd():
    if getattr(sys, "frozen", False):
        cli = Path(sys.executable).with_name("DressUp_cli.exe")
        return str(cli if cli.is_file() else sys.executable), []
    return sys.executable, ["-m", "dressup"]

def pictures_dir() -> Path:
    d = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Pictures" / "DressUp"
    d.mkdir(parents=True, exist_ok=True); return d

class Pane(QLabel):
    def __init__(self, hint, on_click=None):
        super().__init__(hint); self.setObjectName("pane"); self.hint = hint; self.on_click = on_click
        self.setAlignment(Qt.AlignCenter); self.setMinimumSize(200, 200); self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored); self.img = None; self.setAcceptDrops(on_click is not None)
    def set_image(self, path_or_qimage):
        self.img = QImage(path_or_qimage) if isinstance(path_or_qimage, str) else path_or_qimage
        self._fit()
    def clear(self): self.img = None; self.setPixmap(QPixmap()); self.setText(self.hint)
    def _fit(self):
        if self.img is not None and not self.img.isNull():
            self.setPixmap(QPixmap.fromImage(self.img).scaled(self.size() * 0.98, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def resizeEvent(self, e): self._fit(); super().resizeEvent(e)
    def mousePressEvent(self, e):
        if self.on_click: self.on_click()
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        u = e.mimeData().urls()
        if u and self.on_click: self.on_click(u[0].toLocalFile())

class Main(QWidget):
    def __init__(self, args):
        super().__init__(); self.args = args
        self.setWindowTitle(f"Dress Up {__version__}"); self.setStyleSheet(CSS)
        self.person = self.garment = None; self.result = None; self.proc = None; self.showing_after = True
        self.tmp = Path(tempfile.gettempdir()) / "DressUp"; self.tmp.mkdir(exist_ok=True)
        # panes
        self.p_person = Pane("YOUR PHOTO\n\nClick or drop a photo\n(front-facing, full body for dresses)", self.pick_person)
        self.p_result = Pane("RESULT\n\nPick your photo and a garment,\nthen press TRY ON", None)
        self.p_garment = Pane("GARMENT\n\nClick or drop a photo of the\ndress / top / bottom\n(product or worn photo)", self.pick_garment)
        self.p_result.mousePressEvent = lambda e: self.toggle_ba()
        # controls
        def seg(names, default):
            g = QButtonGroup(self); g.setExclusive(True); row = QHBoxLayout(); row.setSpacing(4)
            for n in names:
                b = QPushButton(n); b.setCheckable(True); b.setChecked(n == default); g.addButton(b); row.addWidget(b)
            return g, row
        self.g_kind, kind_row = seg(["Upper", "Lower", "Full dress"], "Upper")
        self.g_q, q_row = seg(["Fast", "Best"], "Best")
        self.b_go = QPushButton("TRY ON"); self.b_go.setObjectName("go"); self.b_go.clicked.connect(self.start)
        self.b_cancel = QPushButton("Cancel"); self.b_cancel.clicked.connect(self.cancel); self.b_cancel.setEnabled(False)
        self.b_ba = QPushButton("Before / After"); self.b_ba.clicked.connect(self.toggle_ba); self.b_ba.setEnabled(False)
        self.b_png = QPushButton("Save PNG"); self.b_png.clicked.connect(lambda: self.save("png"))
        self.b_jpg = QPushButton("Save JPEG"); self.b_jpg.clicked.connect(lambda: self.save("jpg"))
        for b in (self.b_png, self.b_jpg): b.setEnabled(False)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0)
        self.status = QLabel("Ready. Safety filter: on (Relaxed)."); self.status.setWordWrap(False)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        # layout (fixed grid, no scroll areas anywhere)
        top = QHBoxLayout(); t = QLabel("Dress Up"); f = QFont(); f.setPointSize(16); f.setBold(True); t.setFont(f)
        self.dev_lbl = QLabel(""); top.addWidget(t); top.addStretch(1); top.addWidget(self.dev_lbl)
        panes = QHBoxLayout(); panes.setSpacing(10)
        panes.addWidget(self.p_person, 3); panes.addWidget(self.p_result, 4); panes.addWidget(self.p_garment, 3)
        ctl = QGridLayout(); ctl.setHorizontalSpacing(10); ctl.setVerticalSpacing(6)
        ctl.addWidget(QLabel("Garment:"), 0, 0); ctl.addLayout(kind_row, 0, 1)
        ctl.addWidget(QLabel("Quality:"), 0, 2); ctl.addLayout(q_row, 0, 3)
        ctl.addWidget(self.b_go, 0, 4); ctl.addWidget(self.b_cancel, 0, 5)
        ctl.addWidget(self.b_ba, 0, 6); ctl.addWidget(self.b_png, 0, 7); ctl.addWidget(self.b_jpg, 0, 8)
        ctl.addWidget(self.bar, 1, 0, 1, 5); ctl.addWidget(self.status, 1, 5, 1, 4)
        root = QVBoxLayout(self); root.setContentsMargins(12, 8, 12, 10); root.setSpacing(8)
        root.addLayout(top); root.addLayout(panes, 1); root.addLayout(ctl)
        self.setMinimumSize(1100, 600); self.resize(1240, 680)
        if args.get("person"): self.pick_person(args["person"])
        if args.get("garment"): self.pick_garment(args["garment"])

    # --- inputs
    def pick_person(self, path=None):
        path = path or QFileDialog.getOpenFileName(self, "Your photo", str(Path.home() / "Pictures"), IMG_FILTER)[0]
        if path: self.person = path; self.p_person.set_image(path); self.reset_result()
    def pick_garment(self, path=None):
        path = path or QFileDialog.getOpenFileName(self, "Garment photo", str(Path.home() / "Pictures"), IMG_FILTER)[0]
        if path: self.garment = path; self.p_garment.set_image(path); self.reset_result()
    def reset_result(self):
        self.result = None; self.p_result.clear()
        for b in (self.b_ba, self.b_png, self.b_jpg): b.setEnabled(False)

    # --- run (separate process)
    def start(self):
        if not (self.person and self.garment):
            self.status.setText("Pick your photo (left) and a garment (right) first."); return
        kind = self.g_kind.checkedButton().text(); q = self.g_q.checkedButton().text()
        self.out = str(self.tmp / f"result_{int(time.time())}.png")
        exe, pre = _worker_cmd()
        argv = pre + ["--tryon", self.person, self.garment, "--kind", kind, "--quality", q, "--out", self.out,
                      "--device", self.args.get("device", "auto")]
        if self.args.get("models"): argv += ["--models", self.args["models"]]
        self.proc = QProcess(self); self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_out); self.proc.finished.connect(self.on_done)
        self.buf = ""; self.blocked = None; self.t0 = time.time()
        self.busy(True); self.bar.setValue(0); self.status.setText("Starting...")
        self.proc.start(exe, argv)
    def busy(self, b):
        self.b_go.setEnabled(not b); self.b_cancel.setEnabled(b)
        for x in self.g_kind.buttons() + self.g_q.buttons(): x.setEnabled(not b)
    def on_out(self):
        self.buf += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self.buf = self.buf.split("\n")
        for ln in lines:
            ln = ln.strip(); el = int(time.time() - self.t0)
            if ln.startswith("P "):
                i, n = map(int, ln.split()[1:3]); self.bar.setValue(int(100 * i / n))
                self.status.setText(f"Dressing... step {i}/{n} ({el}s)")
            elif ln.startswith("DL "):
                _, d, t, *lab = ln.split(" "); self.bar.setValue(int(100 * int(d) / max(1, int(t))))
                self.status.setText(f"First run: downloading models {int(d) >> 20}/{int(t) >> 20} MB")
            elif ln.startswith("STATUS "): self.status.setText(ln[7:])
            elif ln.startswith("DEVICE "): self.dev_lbl.setText(ln[7:60])
            elif ln.startswith("BLOCKED "): self.blocked = ln[8:]
    def on_done(self, code, _st):
        self.on_out(); self.busy(False); self.proc = None
        if code == 0 and Path(self.out).is_file():
            self.result = self.out; self.showing_after = True; self.p_result.set_image(self.out)
            for b in (self.b_ba, self.b_png, self.b_jpg): b.setEnabled(True)
            self.bar.setValue(100); self.status.setText(f"Done in {int(time.time() - self.t0)}s. Tap the result for before/after.")
            if self.args.get("auto_save"): self.save_to(self.args["auto_save"]); QTimer.singleShot(500, self.autorun_finish)
        elif self.blocked: self.status.setText(self.blocked); self.bar.setValue(0)
        elif getattr(self, "cancelled", False): self.status.setText("Cancelled."); self.bar.setValue(0); self.cancelled = False
        else: self.status.setText(f"Try-on failed (code {code}). {self.buf[-120:]}"); self.bar.setValue(0)
        if self.args.get("auto_save") and code != 0: self.autorun_finish(code or 1)
    def cancel(self):
        if self.proc: self.cancelled = True; self.proc.kill()
    def toggle_ba(self):
        if not self.result: return
        self.showing_after = not self.showing_after
        self.p_result.set_image(self.result if self.showing_after else self.person)
        self.status.setText("Showing: AFTER" if self.showing_after else "Showing: BEFORE")
    def save(self, ext):
        if not self.result: return
        name = f"DressUp_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"
        self.save_to(str(pictures_dir() / name))
    def save_to(self, path):
        from PIL import Image
        im = Image.open(self.result)
        if path.lower().endswith((".jpg", ".jpeg")): im.convert("RGB").save(path, quality=97, subsampling=0)
        else: im.save(path)
        self.status.setText(f"Saved {path} ({im.width}x{im.height})")
    # --- CI autorun: layout check + screenshot
    def autorun_finish(self, code=0):
        rep = {"code": code, "window": [self.width(), self.height()], "result": self.result,
               "scroll_areas": len(self.findChildren(__import__("PySide6.QtWidgets", fromlist=["QAbstractScrollArea"]).QAbstractScrollArea)),
               "saved": self.args.get("auto_save"), "saved_exists": bool(self.args.get("auto_save") and Path(self.args["auto_save"]).is_file())}
        sd = self.args.get("shots_dir")
        if sd:
            Path(sd).mkdir(parents=True, exist_ok=True); self.grab().save(str(Path(sd) / "window.png"))
            (Path(sd) / "gui_autorun_report.json").write_text(json.dumps(rep, indent=2))
        QApplication.instance().exit(0 if code == 0 and rep["scroll_areas"] == 0 and rep["saved_exists"] else 1)

def run_gui(argv):
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--person"); p.add_argument("--garment"); p.add_argument("--models")
    p.add_argument("--device", default="auto"); p.add_argument("--auto-tryon", action="store_true")
    p.add_argument("--kind"); p.add_argument("--quality"); p.add_argument("--auto-save"); p.add_argument("--shots-dir")
    p.add_argument("--size", default="1280x720")
    a, _ = p.parse_known_args(argv[1:])
    app = QApplication(argv[:1]); w = Main(vars(a))
    if a.size: w.resize(*map(int, a.size.split("x")))
    if a.kind: [b.setChecked(True) for b in w.g_kind.buttons() if b.text() == a.kind]
    if a.quality: [b.setChecked(True) for b in w.g_q.buttons() if b.text() == a.quality]
    w.show()
    if a.auto_tryon: QTimer.singleShot(800, w.start)
    return app.exec()
