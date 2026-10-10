# -*- mode: python ; coding: utf-8 -*-
# PyInstaller onedir build for Dress Up (Windows x64): windowed DressUp.exe + console DressUp_cli.exe (worker / selftest).
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata
root = Path(SPECPATH)
datas = [(str(root / "packaging" / "icon.ico"), "."), (str(root / "THIRD_PARTY.md"), "."), (str(root / "LICENSE"), "."),
         (str(root / "testdata"), "testdata")]
binaries = []; hidden = ["PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets", "cv2", "onnxruntime"]
for pkg in ("onnxruntime", "torch_directml", "diffusers"):
    d, b, h = collect_all(pkg); datas += d; binaries += b; hidden += h
for m in ("requests", "filelock", "huggingface_hub", "numpy", "safetensors", "regex", "Pillow", "importlib_metadata", "tqdm", "packaging", "torch", "diffusers", "pyyaml"):
    try: datas += copy_metadata(m)
    except Exception: pass
hidden += ["requests"]
hidden += collect_submodules("diffusers.models") + collect_submodules("diffusers.schedulers")
a = Analysis([str(root / "dressup" / "__main__.py")], pathex=[str(root)], binaries=binaries, datas=datas, hiddenimports=hidden,
             hookspath=[], runtime_hooks=[], noarchive=False,
             excludes=["tkinter", "matplotlib", "IPython", "notebook", "pytest", "transformers", "tensorflow", "jax", "flax",
                       "scipy", "pandas", "torchvision", "torchaudio", "triton",
                       "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtNetwork", "PySide6.QtOpenGL", "PySide6.QtPdf",
                       "PySide6.QtSvg", "PySide6.QtDBus", "PySide6.QtMultimedia", "PySide6.QtWebEngineCore"])
pyz = PYZ(a.pure, a.zipped_data)
icon = str(root / "packaging" / "icon.ico")
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DressUp", console=False, upx=False, icon=icon)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DressUp_cli", console=True, upx=False, icon=icon)
coll = COLLECT(exe, cli, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="DressUp")
