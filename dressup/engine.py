"""Dress Up engine: SegFormer clothes parsing -> garment-area mask -> CatVTON diffusion try-on -> full-res paste-back.
Pixels outside the mask are copied back exactly from the original photo."""
import os, json, time
import numpy as np, cv2
from PIL import Image
import onnxruntime as ort

NSFW_THRESHOLD = 0.85  # Relaxed; always on, no off switch
QUALITY = {"Fast": dict(steps=20, size=(384, 512)), "Best": dict(steps=30, size=(576, 768))}
GARMENTS = ("Upper", "Lower", "Full dress")

class Blocked(Exception): pass
class Cancelled(Exception): pass

def _ort(path):
    so = ort.SessionOptions(); so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

class Safety:
    def __init__(self, mdir): self.s = _ort(os.path.join(mdir, "nsfw_vit_int8.onnx"))
    def score(self, img):
        x = np.asarray(img.convert("RGB").resize((224, 224), Image.BOX), np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        o = self.s.run(None, {self.s.get_inputs()[0].name: x})[0][0].astype(np.float64)
        if not (abs(o.sum() - 1) < 1e-3 and o.min() >= 0): o = np.exp(o - o.max()); o /= o.sum()
        return float(o[1])
    def check(self, img, what):
        p = self.score(img)
        if p > NSFW_THRESHOLD: raise Blocked(f"Safety filter blocked the {what} (score {p:.2f}).")
        return p

class Parser:
    def __init__(self, mdir): self.s = _ort(os.path.join(mdir, "segformer_b2_clothes.onnx"))
    def labels(self, img):
        w, h = img.size
        x = np.asarray(img.convert("RGB").resize((512, 512), Image.BILINEAR), np.float32) / 255.0
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
        lg = self.s.run(None, {"pixel_values": x})[0][0]
        lg = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in lg])
        return lg.argmax(0).astype(np.uint8)

def waist_y(lab):
    """Waist row: belt, else top/bottom-garment boundary, else ~40% from shoulders to feet."""
    h = lab.shape[0]
    rows = lambda c: np.where(np.isin(lab, c).any(1))[0]
    b = rows([8])
    if len(b) > 2: return int(np.median(b))
    up, lo = rows([4]), rows([5, 6])
    if len(up) and len(lo): return int((up.max() + lo.min()) / 2)
    body = rows([4, 5, 6, 7, 17, 12, 13, 9, 10])
    if not len(body): return None
    return int(body.min() + 0.40 * (body.max() - body.min()))

def hands_mask(lab):
    """Hands = the far end of each arm segment (from the shoulder side), radius ~ face height. Kept bit-exact so the
    pose of the hands is preserved and they anchor the arms."""
    h, w = lab.shape; out = np.zeros((h, w), bool)
    fy = np.where((lab == 11).any(1))[0]
    r = (fy.max() - fy.min()) * 0.85 if len(fy) > 3 else 0.06 * h
    cy, cx = [np.mean(v) for v in np.where(np.isin(lab, [4, 7]))] if np.isin(lab, [4, 7]).any() else (h / 2, w / 2)
    for c in (14, 15):
        n, cc, st, _ = cv2.connectedComponentsWithStats((lab == c).astype(np.uint8))
        for i in range(1, n):
            if st[i, 4] < 30: continue
            ys, xs = np.where(cc == i)
            d0 = (ys - cy) ** 2 + (xs - cx) ** 2
            sy, sx = ys[d0.argmin()], xs[d0.argmin()]          # shoulder-side end
            d1 = (ys - sy) ** 2 + (xs - sx) ** 2
            ey, ex = ys[d1.argmax()], xs[d1.argmax()]          # hand end
            sel = (ys - ey) ** 2 + (xs - ex) ** 2 <= r * r
            out[ys[sel], xs[sel]] = True
    return out

def garment_mask(lab, kind):
    """Area the new garment may occupy: current clothes of that type (closed + dilated a little so the shape can change),
    face, hair, bare arms/hands, legs, shoes and bag are always protected (kept bit-exact)."""
    h, w = lab.shape
    upper, lower = {4}, {5, 6}
    cls = {"Upper": upper | {7, 8, 14, 15}, "Lower": lower | {7, 8, 12, 13}, "Full dress": upper | lower | {7, 8, 14, 15, 12, 13}}[kind]
    m = np.isin(lab, list(cls)).astype(np.uint8)
    core = np.isin(lab, list({"Upper": upper | {7}, "Lower": lower | {7}, "Full dress": upper | lower | {7}}[kind]))
    if kind == "Upper" and core.sum() == 0: core = np.isin(lab, [7])
    if core.sum() < 0.002 * h * w: return None
    ys = np.where(core.any(1))[0]; top, bot = ys.min(), ys.max()
    if kind != "Full dress":
        wy = waist_y(lab)
        if kind == "Upper":   # top half only (also when the person wears a one-piece dress)
            lo = bot if wy is None else min(bot, wy)
            m[: max(0, top - h // 40)] = 0; m[lo + h // 60:] = 0
        else:
            hi = top if wy is None else max(top, wy)
            m[: max(0, hi - h // 60)] = 0
        if m.sum() < 0.002 * h * w: return None
    k = max(3, int(0.02 * h)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 3, k * 3)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # fill holes
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = cv2.drawContours(np.zeros_like(m), cnts, -1, 1, -1)
    prot = (np.isin(lab, [1, 2, 3, 9, 10, 11, 16, 17]) | hands_mask(lab)).astype(np.uint8)  # head, scarf/hijab, hands, shoes, bag
    prot = cv2.dilate(prot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, h // 150) | 1,) * 2))
    m[prot > 0] = 0
    return m

def _crop_box(mask, W, H, aspect=0.75):
    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.12 * max(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    cw, ch = x1 - x0, y1 - y0
    if cw / ch < aspect: cw = ch * aspect
    else: ch = cw / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2   # box may extend past the photo; outside is padded and discarded
    if cw <= W: cx = np.clip(cx, cw / 2, W - cw / 2)
    if ch <= H: cy = np.clip(cy, ch / 2, H - ch / 2)
    return int(round(cx - cw / 2)), int(round(cy - ch / 2)), int(round(cw)), int(round(ch))

def _crop_pad(a, x0, y0, cw, ch):
    """Crop numpy image a (H,W[,C]) to box, edge-replicating outside the photo."""
    H, W = a.shape[:2]
    px0, py0 = max(0, -x0), max(0, -y0); px1, py1 = max(0, x0 + cw - W), max(0, y0 + ch - H)
    c = a[max(0, y0):min(H, y0 + ch), max(0, x0):min(W, x0 + cw)]
    if px0 or py0 or px1 or py1:
        pad = ((py0, py1), (px0, px1)) + (((0, 0),) if a.ndim == 3 else ())
        c = np.pad(c, pad, mode="edge" if a.ndim == 3 else "constant")
    return c

def _fit_garment(g, size):
    W, H = size; g = g.convert("RGB"); r = min(W / g.width, H / g.height)
    g = g.resize((max(1, int(g.width * r)), max(1, int(g.height * r))), Image.LANCZOS)
    c = Image.new("RGB", size, (255, 255, 255)); c.paste(g, ((W - g.width) // 2, (H - g.height) // 2)); return c

class CatVTON:
    def __init__(self, mdir, device="cpu"):
        import torch
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
        from diffusers.models.attention_processor import AttnProcessor2_0, AttnProcessor
        from safetensors.torch import load_file
        self.torch = torch
        if device == "dml":
            import torch_directml; self.dev = torch_directml.device()
        else:
            self.dev = torch.device("cpu"); torch.set_num_threads(max(1, (os.cpu_count() or 4)))
        self.dtype = torch.float32
        cfg = json.load(open(os.path.join(mdir, "unet_config.json")))
        self.unet = UNet2DConditionModel.from_config(cfg)
        sd = load_file(os.path.join(mdir, "catvton_unet_fp16.safetensors"))
        miss = self.unet.load_state_dict(sd, strict=False); del sd
        assert all(".attn2." in k for k in miss.missing_keys), miss.missing_keys[:5]
        class Skip(torch.nn.Module):  # CatVTON: cross-attention removed (no text prompt at all)
            def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw): return hidden_states
        procs = {n: (Skip() if "attn2" in n else (AttnProcessor() if device == "dml" else AttnProcessor2_0())) for n in self.unet.attn_processors}
        self.unet.set_attn_processor(procs)
        self.unet.to(self.dev, self.dtype).eval()
        self.vae = AutoencoderKL.from_config(json.load(open(os.path.join(mdir, "vae_config.json"))))
        self.vae.load_state_dict(load_file(os.path.join(mdir, "vae_fp16.safetensors"))); self.vae.to(self.dev, self.dtype).eval()
        self.sched = DDIMScheduler.from_config(json.load(open(os.path.join(mdir, "scheduler_config.json"))))

    def _enc(self, im, keep=None):
        t = self.torch
        a = np.asarray(im.convert("RGB"), np.float32) / 127.5 - 1
        if keep is not None: a = a * keep[..., None]
        x = t.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].to(self.dev, self.dtype)
        return self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor

    def __call__(self, person, garment, mask, steps=40, guidance=2.5, seed=42, progress=None, cancel=None):
        t = self.torch
        with t.no_grad():
            m = t.from_numpy((np.asarray(mask, np.float32) / 255.0 > 0.5).astype(np.float32))[None, None].to(self.dev, self.dtype)
            masked = self._enc(person, keep=(np.asarray(mask) <= 127))  # masked pixels -> 0 (mid-grey), as in CatVTON
            cond = self._enc(garment)
            ml = t.nn.functional.interpolate(m, size=masked.shape[-2:], mode="nearest")
            mcat = t.cat([masked, cond], -2); mlcat = t.cat([ml, t.zeros_like(ml)], -2)
            mcat = t.cat([t.cat([masked, t.zeros_like(cond)], -2), mcat]); mlcat = t.cat([mlcat] * 2)
            g = t.Generator("cpu").manual_seed(seed)
            lat = t.randn(masked.shape[0], 4, masked.shape[2] * 2, masked.shape[3], generator=g).to(self.dev, self.dtype)
            self.sched.set_timesteps(steps); lat = lat * self.sched.init_noise_sigma
            for i, ts in enumerate(self.sched.timesteps):
                if cancel and cancel(): raise Cancelled()
                inp = self.sched.scale_model_input(t.cat([lat] * 2), ts)
                inp = t.cat([inp, mlcat, mcat], 1)
                n = self.unet(inp, ts.to(self.dev), encoder_hidden_states=None, return_dict=False)[0]
                nu, nc = n.chunk(2); n = nu + guidance * (nc - nu)
                lat = self.sched.step(n.cpu(), ts, lat.cpu(), eta=1.0, generator=g).prev_sample.to(self.dev, self.dtype)
                if progress: progress(i + 1, steps)
            lat = lat.split(lat.shape[-2] // 2, -2)[0] / self.vae.config.scaling_factor
            out = (self.vae.decode(lat).sample / 2 + 0.5).clamp(0, 1)
            return Image.fromarray((out[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8))

class Engine:
    def __init__(self, mdir, device="cpu"):
        self.mdir = mdir; self.device = device
        self.safety = Safety(mdir); self.parser = Parser(mdir); self.vton = None
    def load(self):
        if self.vton is None: self.vton = CatVTON(self.mdir, self.device)
    def try_on(self, person, garment, kind="Upper", quality="Best", progress=None, cancel=None, debug=None):
        t0 = time.time(); person = person.convert("RGB"); garment = garment.convert("RGB")
        self.safety.check(person, "person photo"); self.safety.check(garment, "garment photo")
        W, H = person.size
        # parse on a <=1024px copy for speed
        s = min(1.0, 1024 / max(W, H)); small = person.resize((int(W * s), int(H * s)), Image.BILINEAR) if s < 1 else person
        lab = self.parser.labels(small); mk = garment_mask(lab, kind)
        if mk is None: raise Blocked("Couldn't find the clothing to replace. Use a clear, front-facing photo of a dressed person.")
        mk = cv2.resize(mk * 255, (W, H), interpolation=cv2.INTER_NEAREST)
        x0, y0, cw, ch = _crop_box(mk, W, H)
        q = QUALITY[quality]; size = q["size"]
        pa = np.asarray(person)
        pc = Image.fromarray(_crop_pad(pa, x0, y0, cw, ch)).resize(size, Image.LANCZOS)
        mcrop = _crop_pad(mk, x0, y0, cw, ch)
        mc = Image.fromarray(mcrop).resize(size, Image.NEAREST)
        gc = _fit_garment(garment, size)
        self.load()
        gen = self.vton(pc, gc, mc, steps=q["steps"], progress=progress, cancel=cancel)
        # paste back: only inside a feathered mask; everything else stays bit-exact
        up = np.asarray(gen.resize((cw, ch), Image.LANCZOS), np.float32)
        a = mcrop.astype(np.float32) / 255
        f = max(3, int(0.004 * max(cw, ch))) | 1
        a = cv2.GaussianBlur(cv2.erode(a, np.ones((f, f), np.uint8)), (f * 2 + 1, f * 2 + 1), 0)
        a = np.maximum(a, cv2.erode(mcrop, np.ones((f * 3, f * 3), np.uint8)).astype(np.float32) / 255)
        a = (a * (mcrop > 0))[..., None]   # hard guarantee: zero outside the mask -> original pixels bit-exact
        out = pa.astype(np.float32).copy()
        H, W = out.shape[:2]
        ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(W, x0 + cw), min(H, y0 + ch)
        sl = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
        reg = out[iy0:iy1, ix0:ix1]
        out[iy0:iy1, ix0:ix1] = reg * (1 - a[sl]) + up[sl] * a[sl]
        res = Image.fromarray(out.round().clip(0, 255).astype(np.uint8))
        self.safety.check(res, "result")
        if debug is not None: debug.update(mask=Image.fromarray(mk), crop=(x0, y0, cw, ch), gen=gen, garment_in=gc, secs=time.time() - t0)
        return res
