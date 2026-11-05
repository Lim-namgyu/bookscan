#!/usr/bin/env python3
"""Run deblur.onnx on real phone photos and write side-by-side crops.

For every image: gray -> float 0..1 -> ONNX model in 512-row tiles with 8-px
overlap -> uint8.  Writes eval_<name>.png (input | output per crop, 3x nearest
zoom), eval_<name>_full.jpg (whole deblurred image, for a look at global
behaviour) and prints timing / metrics.

Usage:  python3 eval.py [--norm] [--model=_v2]
    --norm   divide each tile by its local paper level (98th percentile) before the
             model and multiply back afterwards (domain-gap experiment; off by default)
"""
import os, sys, time
import numpy as np
import cv2
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = "/tmp/claude-0/-home-claude/d3460888-10d6-50e2-87cb-0d3a11dad1bf/scratchpad/real"
NORM = "--norm" in sys.argv
MODEL = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "")   # e.g. --model=_v2
TAG = MODEL + ("_norm" if NORM else "")
ZOOM = 3
TILE, OVER = 512, 8

# (image, [ (x, y, w, h, label), ... ])  full-resolution crop windows on text
IMAGES = {
    "closeup_p2.jpg": [
        (120, 60, 240, 100, "body text"),
        (130, 1330, 240, 100, "body text 2"),
        (1010, 1300, 240, 100, "small answer text"),
    ],
    "closeup_p2_blur1.0.jpg": [
        (120, 60, 240, 100, "body text"),
        (130, 1330, 240, 100, "body text 2"),
        (1010, 1300, 240, 100, "small answer text"),
    ],
    "spread3_up.jpg": [
        (560, 700, 240, 100, "left page body"),
        (1400, 1090, 240, 100, "middle column"),
        (2300, 800, 240, 100, "right page body"),
        (3100, 400, 240, 100, "far-right small table"),
    ],
}

def load_session():
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    return ort.InferenceSession(os.path.join(HERE, f"deblur{MODEL}.onnx"), so, providers=["CPUExecutionProvider"])

def paper_level(gray, k=51):
    """Local paper (background) brightness: max-filter wider than a glyph, then smooth."""
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), k / 2)
    return np.maximum(bg, 40.0) / 255.0

def run_model(sess, gray, timing=None):
    """gray: uint8 HxW -> float32 HxW in 0..1 (model output).
    With --norm the image is divided by its local paper level first (flat-field) and
    multiplied back afterwards, so the model sees paper near 1.0 as in training."""
    H, W = gray.shape
    x = gray.astype(np.float32) / 255.0
    if NORM:
        bg = paper_level(gray)
        x = np.clip(x / bg, 0, 1)
    out = np.zeros_like(x)
    t0 = time.time()
    y0 = 0
    while y0 < H:
        y1 = min(H, y0 + TILE)
        a, b = max(0, y0 - OVER), min(H, y1 + OVER)
        o = sess.run(None, {"img": x[a:b][None, None]})[0][0, 0]
        out[y0:y1] = o[y0 - a:y0 - a + (y1 - y0)]
        y0 = y1
    if timing is not None:
        timing.append(time.time() - t0)
    if NORM:
        out = out * bg
    return np.clip(out, 0, 1)

def ink_f1(img, ref_bin):
    """Agreement of Otsu-binarised ink pixels with the binarised sharp reference (F1)."""
    _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b = b > 0
    tp = (b & ref_bin).sum(); fp = (b & ~ref_bin).sum(); fn = (~b & ref_bin).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn)

def zoom(img):
    return cv2.resize(img, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)

def lap_var(g):
    return float(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).var())

def val_sheet(sess, n=20):
    """Synthetic validation patches: input | model | target, 3x zoom, 4 per row."""
    try:
        X = np.load(os.path.join(HERE, "val_x.npy")); Y = np.load(os.path.join(HERE, "val_y.npy"))
    except FileNotFoundError:
        return
    cells = []
    for i in range(n):
        o = sess.run(None, {"img": X[i][None, None].astype(np.float32) / 255.0})[0][0, 0]
        o8 = np.clip(np.round(o * 255), 0, 255).astype(np.uint8)
        sep = np.full((64, 2), 128, np.uint8)
        cells.append(zoom(np.hstack([X[i], sep, o8, sep, Y[i]])))
    pad = np.full((cells[0].shape[0], 12), 255, np.uint8)
    rows = [np.hstack(sum([[c, pad] for c in cells[r:r + 4]], [])[:-1]) for r in range(0, n, 4)]
    cv2.imwrite(os.path.join(HERE, f"eval_val_patches{TAG}.png"), np.vstack(rows))

def main():
    sess = load_session()
    val_sheet(sess)
    report = []
    ref_sharp = None
    for name, crops in IMAGES.items():
        path = os.path.join(REAL, name)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        H, W = gray.shape
        # warm-up once on a small tile so session init is excluded from timing
        sess.run(None, {"img": np.zeros((1, 1, 64, 64), np.float32)})
        timing = []
        t0 = time.time(); out = run_model(sess, gray, timing); dt_total = time.time() - t0; dt = timing[0]
        out8 = np.clip(np.round(out * 255), 0, 255).astype(np.uint8)
        mpix = H * W / 1e6
        stem = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(HERE, f"eval_{stem}{TAG}_full.jpg"), out8, [cv2.IMWRITE_JPEG_QUALITY, 90])
        # ---- metrics
        diff = np.abs(out8.astype(np.float32) - gray.astype(np.float32))
        line = (f"{name}: {W}x{H} ({mpix:.2f} MP)  model {dt:.2f}s = {dt / mpix:.3f} s/MP"
                + (f" (+{dt_total - dt:.2f}s flat-field)" if NORM else "") +
                f"  mean|out-in| {diff.mean():.2f}/255  LapVar in {lap_var(gray):.0f} -> out {lap_var(out8):.0f}")
        if name == "closeup_p2.jpg":
            ref_sharp = gray
            _, rb = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU); ref_bin = rb > 0
            line += f"\n    sharp image: ink-F1 of model output vs binarised input = {ink_f1(out8, ref_bin):.4f} (1.0 = strokes unchanged)"
        if name == "closeup_p2_blur1.0.jpg" and ref_sharp is not None:
            l_in = np.abs(gray.astype(np.float32) - ref_sharp).mean()
            l_out = np.abs(out8.astype(np.float32) - ref_sharp).mean()
            ps = lambda l2: 10 * np.log10(255 ** 2 / l2)
            p_in = ps(((gray.astype(np.float32) - ref_sharp) ** 2).mean())
            p_out = ps(((out8.astype(np.float32) - ref_sharp) ** 2).mean())
            line += (f"\n    vs sharp closeup_p2: L1 blurred {l_in:.2f} -> deblurred {l_out:.2f} (/255); PSNR {p_in:.2f} -> {p_out:.2f} dB"
                     f"\n    ink-F1 vs binarised sharp: blurred {ink_f1(gray, ref_bin):.4f} -> deblurred {ink_f1(out8, ref_bin):.4f}")
        print(line); report.append(line)
        # ---- side-by-side crops
        rows = []
        for (x, y, w, h, label) in crops:
            x = min(x, W - w); y = min(y, H - h)
            a = zoom(gray[y:y + h, x:x + w]); b = zoom(out8[y:y + h, x:x + w])
            sep = np.full((a.shape[0], 6), 128, np.uint8)
            row = np.hstack([a, sep, b])
            # label strip
            strip = np.full((22, row.shape[1]), 255, np.uint8)
            cv2.putText(strip, f"{label}  (x={x},y={y})   input | model", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
            rows.append(np.vstack([strip, row]))
        sheet = np.vstack(rows)
        cv2.imwrite(os.path.join(HERE, f"eval_{stem}{TAG}.png"), sheet)
    with open(os.path.join(HERE, f"eval_metrics{TAG}.txt"), "w") as f:
        f.write("\n".join(report) + "\n")

if __name__ == "__main__":
    main()
