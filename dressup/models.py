"""Model manifest + resumable, SHA-256 verified downloader.
Files live in %LOCALAPPDATA%\\DressUp\\models. Primary source: vanukrishnans-source/dress-up-windows models-v1."""
from __future__ import annotations
import hashlib, os, time, urllib.request
from pathlib import Path

REL = "https://github.com/vanukrishnans-source/dress-up-windows/releases/download/models-v1/"
DRESSUP_ANDROID = "https://github.com/vanukrishnans-source/dress-up/releases/download/models-v1/"

# (file, label, bytes, sha256, extra mirrors)
MANIFEST = [
    ("catvton_unet_fp16.safetensors", "Try-on model (CatVTON on SD1.5-inpainting, fp16)", 1631217792,
     "bd58719be9fcb65bd34edb2b59cca209e0a75d3ab051f31b9d01ab01bd7ba09b", []),
    ("vae_fp16.safetensors", "Image decoder (sd-vae-ft-mse)", 167335342,
     "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a", []),
    ("segformer_b2_clothes.onnx", "Clothes finder (SegFormer B2 clothes)", 109725326,
     "1d3f5d24a6527f2eae44c6f06845b07f1a5b884fc79493d8f896caee37954841", []),
    ("nsfw_vit_int8.onnx", "Safety filter (Falconsai NSFW ViT int8)", 88639960,
     "b6734b6fdc6be4edac08ffd91b07c7b2bc800c6a93c79b029c865bde2fb43627", [DRESSUP_ANDROID]),
    ("unet_config.json", "config", 748, "669f6581087fac5b16c44e42c58b359a35eadf2af5691d7945480bd7ceac8e6d", []),
    ("vae_config.json", "config", 758, "d9189c713f41cca1206754ca7e08daaba5d271b5716ad8de9f4480b4e6485414", []),
    ("scheduler_config.json", "config", 313, "8b457848b74deeeab2a967b75048dd2cb6f393cfc17751eca1b5ec655cff0183", []),
]
TOTAL_BYTES = sum(m[2] for m in MANIFEST)

def default_dir() -> Path:
    if os.environ.get("DRESSUP_MODELS"): return Path(os.environ["DRESSUP_MODELS"])
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "DressUp" / "models"

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def is_ok(d: Path, entry) -> bool:
    f, _, n, sha, _ = entry; p = d / f; mk = d / (f + ".ok")
    try:
        if not p.is_file() or p.stat().st_size != n: return False
        if mk.is_file() and mk.read_text().strip() == sha: return True
        if _sha(p) == sha: mk.write_text(sha); return True
    except OSError: pass
    return False

def missing(d: Path | None = None):
    d = Path(d or default_dir()); d.mkdir(parents=True, exist_ok=True)
    return [e for e in MANIFEST if not is_ok(d, e)]

def ensure(d: Path | None = None, progress=None, cancel=None) -> Path:
    """Download whatever is missing. progress(done_bytes, total_bytes, label). Resumes .part files."""
    d = Path(d or default_dir()); d.mkdir(parents=True, exist_ok=True)
    todo = missing(d); total = sum(e[2] for e in todo); done = 0
    for f, label, n, sha, extra in todo:
        part = d / (f + ".part"); err = None
        for url in [REL + f] + [m + f for m in extra]:
            for attempt in range(3):
                try:
                    have = part.stat().st_size if part.exists() else 0
                    if have > n: part.unlink(); have = 0
                    req = urllib.request.Request(url, headers={"User-Agent": "DressUp/1.0", **({"Range": f"bytes={have}-"} if have else {})})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        if have and r.status != 206: have = 0; part.unlink(missing_ok=True)
                        with open(part, "ab") as out:
                            while True:
                                if cancel and cancel(): raise InterruptedError("cancelled")
                                b = r.read(1 << 20)
                                if not b: break
                                out.write(b); have += len(b)
                                if progress: progress(done + have, total, label)
                    if part.stat().st_size != n: raise IOError(f"size {part.stat().st_size} != {n}")
                    if _sha(part) != sha: part.unlink(); raise IOError("SHA-256 mismatch")
                    os.replace(part, d / f); (d / (f + ".ok")).write_text(sha); err = None; break
                except InterruptedError: raise
                except Exception as e:  # noqa: BLE001
                    err = e; time.sleep(2)
            if err is None: break
        if err is not None: raise RuntimeError(f"Download failed for {f}: {err}")
        done += n
    return d
