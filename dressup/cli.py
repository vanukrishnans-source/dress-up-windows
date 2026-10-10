"""Console entry: --version, --dml-probe, --download-models, --tryon (also used by the GUI as a crash-isolated worker), --selftest."""
import argparse, json, os, sys, time
from pathlib import Path

def _say(*a): print(*a, flush=True)

def resolve_device(want: str):
    if want == "cpu": return "cpu", "CPU (forced)"
    from .dml_probe import probe
    ok, info = probe()
    if ok: return "dml", "GPU (DirectML): " + info
    if want == "dml": raise SystemExit(f"DirectML not usable: {info}")
    return "cpu", "CPU (GPU probe failed: " + info + ")"

def before_after(images, labels, h=900):
    from PIL import Image, ImageDraw
    ims = [i.convert("RGB").resize((max(1, int(i.width * h / i.height)), h)) for i in images]
    W = sum(i.width for i in ims) + 10 * (len(ims) - 1)
    c = Image.new("RGB", (W, h + 40), "white"); d = ImageDraw.Draw(c); x = 0
    for im, lb in zip(ims, labels):
        c.paste(im, (x, 40)); d.text((x + 8, 12), lb, fill="black"); x += im.width + 10
    return c

def tryon(a):
    from PIL import Image
    from .engine import Engine, Blocked, Cancelled
    from . import models
    mdir = Path(a.models) if a.models else models.default_dir()
    if models.missing(mdir):
        _say("STATUS Downloading models")
        models.ensure(mdir, progress=lambda d, t, l: _say(f"DL {d} {t} {l}"))
    dev, info = resolve_device(a.device); _say("DEVICE", info)
    eng = Engine(str(mdir), dev)
    t0 = time.time()
    try:
        _say("STATUS Loading try-on model"); eng.load()
        _say("STATUS Generating")
        person, garment = Image.open(a.tryon[0]), Image.open(a.tryon[1])
        from PIL import ImageOps
        person, garment = ImageOps.exif_transpose(person), ImageOps.exif_transpose(garment)
        res = eng.try_on(person, garment, a.kind, a.quality, progress=lambda i, n: _say(f"P {i} {n}"))
    except Blocked as e:
        _say("BLOCKED", e); return 3
    except Cancelled:
        return 4
    except Exception as e:  # noqa: BLE001  (GPU op failure inside python -> retry on CPU)
        if dev == "dml" and not a.no_fallback:
            _say("STATUS GPU failed, retrying on CPU:", f"{type(e).__name__}: {e}"[:200])
            a.device = "cpu"; return tryon(a)
        raise
    res.save(a.out) if not a.out.lower().endswith((".jpg", ".jpeg")) else res.save(a.out, quality=97, subsampling=0)
    if a.before_after:
        before_after([person, garment, res], ["Person", "Garment", f"Result ({a.kind})"]).save(a.before_after, quality=92)
    _say(f"DONE {a.out} secs={time.time() - t0:.1f} device={dev}")
    return 0

def selftest(a):
    """Packaged-exe check: real pipeline (Fast), unmasked pixels bit-exact, masked region changed, safety filter live."""
    import numpy as np
    from PIL import Image
    from .engine import Engine, garment_mask, Blocked
    from . import models
    mdir = Path(a.models) if a.models else models.default_dir()
    models.ensure(mdir)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    person = Image.open(root / "testdata" / "person.jpg").convert("RGB"); garment = Image.open(root / "testdata" / "tshirt.jpg")
    dev, info = resolve_device(a.device)
    eng = Engine(str(mdir), dev); t0 = time.time(); d = {}
    res = eng.try_on(person, garment, "Upper", "Fast", debug=d)
    secs = time.time() - t0
    res.save(out / "selftest_result.png")
    before_after([person, garment, res], ["Person", "Garment", "Result"]).save(out / "selftest_before_after.jpg")
    m = np.asarray(d["mask"]) > 0
    po, ro = np.asarray(person), np.asarray(res)
    import cv2
    grow = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    exact = bool((po[~grow] == ro[~grow]).all())
    changed = float(np.abs(po[m].astype(int) - ro[m].astype(int)).mean())
    s_res = eng.safety.score(res)
    ok = exact and changed > 8 and s_res <= 0.85
    rep = dict(ok=ok, device=info, secs=round(secs, 1), outside_mask_exact=exact, mean_change_in_mask=round(changed, 1),
               nsfw_result=round(s_res, 4), mask_fraction=round(float(m.mean()), 3), size=person.size)
    (out / "selftest_report.json").write_text(json.dumps(rep, indent=2)); _say(json.dumps(rep))
    return 0 if ok else 1

def main(argv=None):
    from . import __version__
    p = argparse.ArgumentParser(prog="DressUp")
    p.add_argument("--version", action="store_true"); p.add_argument("--dml-probe", action="store_true")
    p.add_argument("--download-models", action="store_true"); p.add_argument("--models")
    p.add_argument("--tryon", nargs=2, metavar=("PERSON", "GARMENT"))
    p.add_argument("--kind", default="Upper", choices=["Upper", "Lower", "Full dress"])
    p.add_argument("--quality", default="Best", choices=["Fast", "Best"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "dml"])
    p.add_argument("--no-fallback", action="store_true")
    p.add_argument("--out", default="dressup_result.png"); p.add_argument("--before-after")
    p.add_argument("--selftest", action="store_true"); p.add_argument("--out-dir", default="selftest-out")
    a, _ = p.parse_known_args(argv)
    if a.version: _say(f"DressUp {__version__}"); return 0
    if a.dml_probe:
        from .dml_probe import child; return child()
    if a.download_models:
        from . import models; models.ensure(Path(a.models) if a.models else None, progress=lambda d, t, l: _say(f"DL {d} {t} {l}")); return 0
    if a.selftest: return selftest(a)
    if a.tryon: return tryon(a)
    return None
