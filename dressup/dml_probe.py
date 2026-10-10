"""Out-of-process DirectML probe: a native crash in the GPU driver kills only the child; the app falls back to CPU."""
import json, os, subprocess, sys, time
from pathlib import Path

def _status() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    d = Path(base) / "DressUp"; d.mkdir(parents=True, exist_ok=True); return d / "dml_status.json"

def child() -> int:
    """`--dml-probe`: tiny conv + attention-sized matmul on torch-directml. 0 = OK."""
    if os.environ.get("DRESSUP_FORCE_DML_FAIL") == "1": print("DML_PROBE fail: forced"); return 2
    try:
        import torch, torch_directml
        dev = torch_directml.device()
        x = torch.randn(2, 4, 64, 48).to(dev)
        y = torch.nn.functional.conv2d(x, torch.randn(8, 4, 3, 3).to(dev), padding=1)
        a = torch.randn(2, 1024, 64).to(dev); s = (a @ a.transpose(1, 2)).softmax(-1) @ a
        v = float(y.mean().cpu()) + float(s.mean().cpu())
        print("DML_PROBE ok", torch_directml.device_name(0), v, flush=True); return 0
    except Exception as e:  # noqa: BLE001
        print(f"DML_PROBE fail: {type(e).__name__}: {e}", flush=True); return 2

def probe(timeout=120, reprobe=False):
    """Returns (ok, info). Cached in %LOCALAPPDATA%\\DressUp\\dml_status.json."""
    if os.environ.get("DRESSUP_FORCE_DML_FAIL") == "1":
        return False, "DRESSUP_FORCE_DML_FAIL=1 (simulated GPU failure)"
    p = _status()
    if not reprobe and p.is_file():
        try:
            st = json.loads(p.read_text()); return bool(st["ok"]), st.get("info", "")
        except Exception: pass  # noqa: BLE001
    cmd = [sys.executable, "--dml-probe"] if getattr(sys, "frozen", False) else [sys.executable, "-m", "dressup", "--dml-probe"]
    if getattr(sys, "frozen", False):
        cli = Path(sys.executable).with_name("DressUp_cli.exe")
        if cli.is_file(): cmd[0] = str(cli)
    try:
        cp = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=0x08000000 if os.name == "nt" else 0)
        out = (cp.stdout + cp.stderr).decode("utf-8", "replace")
        ok = cp.returncode == 0 and "DML_PROBE ok" in out
        info = out.strip().splitlines()[-1][:300] if out.strip() else f"exit {cp.returncode} (native crash?)"
    except Exception as e:  # noqa: BLE001
        ok, info = False, f"{type(e).__name__}: {e}"
    try: p.write_text(json.dumps({"ok": ok, "info": info, "at": time.strftime("%Y-%m-%d %H:%M:%S")}))
    except OSError: pass
    return ok, info
