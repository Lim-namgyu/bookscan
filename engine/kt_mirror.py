"""
BookScanEngine.kt 의 처리 순서·OpenCV 호출을 1:1로 옮긴 파이썬 미러.
(글줄 반듯하게 단계는 Straightener.kt 와 같은 알고리즘인 straighten.py 를 그대로 사용)
안드로이드에서 새로 바뀐 부분(warpAffine 업샘플, 8비트 나눗셈, LUT, ROI 처리)이
파이썬 기준 구현(bookscan.py)과 같은 결과를 내는지 확인하는 용도.
"""
import math
import sys

import cv2
import numpy as np
import onnxruntime as ort

from straighten import straighten_maps

MODEL_W, MODEL_H, GRID_W, GRID_H = 488, 712, 31, 45
ANALYSIS_SIDE = 1600.0
GRID_EXT = 1


def ell(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def pct_hist(arr, p):
    hist = np.bincount(arr.ravel(), minlength=256)
    target = (arr.size - 1) * p / 100.0
    acc = np.cumsum(hist)
    return float(np.argmax(acc > target))


def expand(r, pad, size):
    x, y, w, h = r
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(size[0], x + w + pad), min(size[1], y + h + pad)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def to_full(r, s, w, h):
    x, y, rw, rh = r
    x0 = min(max(math.floor(x / s), 0), w - 1)
    y0 = min(max(math.floor(y / s), 0), h - 1)
    x1 = min(max(math.ceil((x + rw) / s), x0 + 1), w)
    y1 = min(max(math.ceil((y + rh) / s), y0 + 1), h)
    return (x0, y0, x1 - x0, y1 - y0)


def sub(m, r):
    x, y, w, h = r
    return m[y:y + h, x:x + w]


class Engine:
    def __init__(self, model):
        self.sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    def rectify(self, img):
        """BookScanEngine.rectify 미러: 곡면 펴기 ∘ 테두리 자르기 ∘ 글줄 반듯하게 → 한 번만 재샘플링"""
        h, w = img.shape[:2]
        inp = cv2.resize(img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        grid = self.sess.run(None, {"image": inp.transpose(2, 0, 1)[None]})[0][0]
        gx = ((grid[0] + 1) / 2 * (w - 1)).astype(np.float32)
        gy = ((grid[1] + 1) / 2 * (h - 1)).astype(np.float32)
        row = np.hypot(np.diff(gx, axis=1), np.diff(gy, axis=1)).sum(1)
        col = np.hypot(np.diff(gx, axis=0), np.diff(gy, axis=0)).sum(0)
        ow, oh = np.median(row), np.median(col)
        scale = min(1.15, math.sqrt(w * h / max(ow * oh, 1)))
        for _ in range(GRID_EXT):  # 위아래로 격자 연장
            gx = np.vstack([2 * gx[:1] - gx[1:2], gx, 2 * gx[-1:] - gx[-2:-1]]).astype(np.float32)
            gy = np.vstack([2 * gy[:1] - gy[1:2], gy, 2 * gy[-1:] - gy[-2:-1]]).astype(np.float32)
        oh = oh * (GRID_H - 1 + 2 * GRID_EXT) / (GRID_H - 1)
        outW, outH = max(16, round(ow * scale)), max(16, round(oh * scale))
        a = min(1.0, ANALYSIS_SIDE / max(outW, outH))
        aw, ah = max(16, round(outW * a)), max(16, round(outH * a))
        ax = self._up(gx, aw, ah); ay = self._up(gy, aw, ah)
        small = cv2.resize(img, None, fx=a, fy=a, interpolation=cv2.INTER_AREA) if a < 1 else img.copy()
        flatA = cv2.remap(small, ax * a, ay * a, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        t, bt, l, r = self.trim_rect(flatA)
        fa = flatA[t:ah - bt, l:aw - r].copy()
        st = straighten_maps(cv2.cvtColor(fa, cv2.COLOR_RGB2GRAY))
        if st is not None:
            sx, sy = st[0], st[1]
        else:
            sx, sy = np.meshgrid(np.arange(fa.shape[1], dtype=np.float32), np.arange(fa.shape[0], dtype=np.float32))
        fh, fw = sx.shape
        sx = sx + l; sy = sy + t
        fxa = cv2.remap(ax, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fya = cv2.remap(ay, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        FW, FH = max(16, round(fw / a)), max(16, round(fh / a))
        FX = cv2.resize(fxa, (FW, FH), interpolation=cv2.INTER_LINEAR)
        FY = cv2.resize(fya, (FW, FH), interpolation=cv2.INTER_LINEAR)
        out = cv2.remap(img, FX, FY, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        t2, b2, l2, r2 = self.trim_rect(out)
        if t2 + b2 + l2 + r2 == 0:
            return out
        return out[t2:out.shape[0] - b2, l2:out.shape[1] - r2].copy()

    @staticmethod
    def _up(g, W, H):
        M = np.array([[(g.shape[1] - 1) / (W - 1), 0, 0], [0, (g.shape[0] - 1) / (H - 1), 0]], np.float64)
        return cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    def trim_rect(self, img, maxFrac=0.06):
        h, w = img.shape[:2]
        s = min(1.0, 1000.0 / max(h, w))
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        sh, sw = g.shape
        thr = pct_hist(g, 70.0) * 0.72

        def scan(lines, limit):
            n = 0
            for v in lines[:limit]:
                if np.percentile(v.astype(np.float64), 50) < thr:
                    n += 1
                else:
                    break
            return n

        t = scan(g, int(sh * maxFrac)); bt = scan(g[::-1], int(sh * maxFrac))
        l = scan(g.T, int(sw * maxFrac)); r = scan(g.T[::-1], int(sw * maxFrac))
        return round(t / s), round(bt / s), round(l / s), round(r / s)

    def finger_mask(self, img, skip=None):
        h, w = img.shape[:2]
        s = 800.0 / max(h, w)
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
        sh, sw = lab.shape[:2]
        L = lab[..., 0].astype(np.float32) * 100 / 255
        A = lab[..., 1].astype(np.float32) - 128
        B = lab[..., 2].astype(np.float32) - 128
        p60 = np.percentile(L, 60)
        pa = A[L > p60]
        a0 = np.percentile(pa, 50) if pa.size else 0
        hue = np.degrees(np.arctan2(B, A)); ch = np.hypot(A, B)
        skin = (((A - a0) > 8) & (ch > 14) & (hue > 25) & (hue < 60) & (L > 20) & (L < 90)).astype(np.uint8) * 255
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, ell(3))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, ell(9))
        k = max(5, int(min(sh, sw) * 0.025)) | 1
        core = cv2.erode(skin, ell(k))
        n, lbl, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8, ltype=cv2.CV_32S)
        mask = np.zeros((sh, sw), np.uint8)
        edge = max(3, int(min(sh, sw) * 0.02)) + k // 2
        kern = ell(k + 2)
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if area < sh * sw * 0.0006:
                continue
            sides = []
            if x <= edge and skip != "left": sides.append((bh / sh, bw / sw))
            if x + bw >= sw - edge and skip != "right": sides.append((bh / sh, bw / sw))
            if y <= edge: sides.append((bw / sw, bh / sh))
            if y + bh >= sh - edge: sides.append((bw / sw, bh / sh))
            if not sides or any(a > 0.4 for a, _ in sides) or max(d for _, d in sides) < 0.02:
                continue
            # 가장자리를 따라 가늘고 긴 띠(책 옆면) 제외: 변 방향 길이 px > 2 × 깊이 px
            px_sides = []
            if x <= edge and skip != "left": px_sides.append((bh, bw))
            if x + bw >= sw - edge and skip != "right": px_sides.append((bh, bw))
            if y <= edge: px_sides.append((bw, bh))
            if y + bh >= sh - edge: px_sides.append((bw, bh))
            if all(al > 2.0 * dp for al, dp in px_sides):
                continue
            comp = cv2.compare(lbl, i, cv2.CMP_EQ)
            comp = cv2.dilate(comp, kern)
            comp = cv2.bitwise_and(comp, skin)
            mask = cv2.bitwise_or(mask, comp)
        if cv2.countNonZero(mask):
            d = max(5, int(min(sh, sw) * 0.03)) | 1
            mask = cv2.dilate(mask, ell(d))
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    def fill_paper(self, img, mask):
        h, w = img.shape[:2]
        s = 600.0 / max(h, w)
        ss = (round(w * s), round(h * s))
        small = cv2.resize(img, ss, interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask, ss, interpolation=cv2.INTER_NEAREST)
        longSide = max(ss)
        k = max(9, int(longSide * 0.03)) | 1
        paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE, ell(k))
        dil = cv2.dilate(m, ell(5))
        valid = (cv2.compare(dil, 0, cv2.CMP_EQ)).astype(np.float32) / 255
        sig = longSide * 0.05
        num = cv2.GaussianBlur(paper.astype(np.float32) * cv2.merge([valid] * 3), (0, 0), sig)
        den = np.maximum(cv2.GaussianBlur(valid, (0, 0), sig), 1e-3)
        ref = num / cv2.merge([den] * 3)
        paperS = cv2.morphologyEx(small, cv2.MORPH_CLOSE, ell(7)).astype(np.float32)
        lumRef = ref.mean(axis=2); lumP = np.maximum(paperS.mean(axis=2), 1)
        gain = np.clip(lumRef / lumP, 1, 2.5)
        gain = cv2.GaussianBlur(gain, (0, 0), 1.5)
        d = max(5, int(longSide * 0.06)) | 1
        ring8 = cv2.dilate(m, ell(d))
        ringF = cv2.GaussianBlur(ring8, (0, 0), 3).astype(np.float32) / 255
        g = 1 + (gain - 1) * ringF
        roiS = expand(cv2.boundingRect(ring8), 10, ss)
        roiF = to_full(roiS, s, w, h)
        out = img.copy()
        imgRoi = sub(img, roiF).astype(np.float32)
        size = (roiF[2], roiF[3])
        refUp = cv2.resize(sub(ref, roiS), size, interpolation=cv2.INTER_CUBIC)
        gUp = cv2.resize(sub(g, roiS), size, interpolation=cv2.INTER_LINEAR)
        soft = cv2.GaussianBlur(sub(mask, roiF), (0, 0), 4).astype(np.float32) / 255
        imgRoi = (imgRoi + (refUp - imgRoi) * cv2.merge([soft] * 3)) * cv2.merge([gUp] * 3)
        x, y, rw, rh = roiF
        out[y:y + rh, x:x + rw] = np.clip(imgRoi, 0, 255).astype(np.uint8)  # convertTo 는 반올림+포화
        return out

    def flatten(self, img):
        h, w = img.shape[:2]
        s = 1000.0 / max(h, w)
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        k = max(15, int(max(small.shape[:2]) * 0.035)) | 1
        bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, ell(k))
        bg = cv2.medianBlur(bg, 21)
        bg = cv2.GaussianBlur(bg, (0, 0), 5)
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)
        bg = np.maximum(bg, 1)
        return cv2.divide(img, bg, scale=255.0)

    def finish(self, norm, mode):
        if mode == "bw":
            g = cv2.GaussianBlur(cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY), (0, 0), 0.8)
            return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
        x = norm.copy() if mode == "color" else cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY)
        s = min(1.0, 1000.0 / max(x.shape[:2]))
        small = cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        lo = pct_hist(small, 1.0); hi = 238.0
        lut = np.array([round(min(max((v - lo) / max(hi - lo, 1), 0), 1) ** 1.25 * 255) for v in range(256)], np.uint8)
        y = cv2.LUT(x, lut)
        blur = cv2.GaussianBlur(y, (0, 0), 1.2)
        return cv2.addWeighted(y, 1.6, blur, -0.6, 0)

    def erase(self, out, mask):
        h, w = mask.shape
        s = 600.0 / max(h, w)
        ss = (round(w * s), round(h * s))
        mS = cv2.resize(mask, ss, interpolation=cv2.INTER_NEAREST)
        d = max(5, int(min(ss) * 0.04)) | 1
        ringS = cv2.dilate(mS, ell(d))
        roiS = expand(cv2.boundingRect(ringS), 6, ss)
        ringS = cv2.GaussianBlur(ringS, (0, 0), max(1.0, 6.0 * s))
        roiF = to_full(roiS, s, w, h)
        size = (roiF[2], roiF[3])
        ring = cv2.resize(sub(ringS, roiS), size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        soft = cv2.GaussianBlur(sub(mask, roiF), (0, 0), 3).astype(np.float32) / 255
        x, y, rw, rh = roiF
        outRoi = out[y:y + rh, x:x + rw]
        lum = (cv2.cvtColor(outRoi, cv2.COLOR_RGB2GRAY) if out.ndim == 3 else outRoi).astype(np.float32)
        light = np.clip((lum - 150) / 60, 0, 1) * ring
        wgt = np.maximum(soft, light)
        f = outRoi.astype(np.float32)
        wv = cv2.merge([wgt] * 3) if out.ndim == 3 else wgt
        f = f + (255 - f) * wv
        out[y:y + rh, x:x + rw] = np.clip(np.round(f), 0, 255).astype(np.uint8)

    def clean(self, flat, mode, fingers, skip):
        mask = self.finger_mask(flat, skip) if fingers else None
        has = mask is not None and cv2.countNonZero(mask) > 0
        filled = self.fill_paper(flat, mask) if has else flat
        out = self.finish(self.flatten(filled), mode)
        if has:
            self.erase(out, mask)
        return out


if __name__ == "__main__":
    sys.path.insert(0, ".")
    from bookscan import find_gutter, split_spread  # 양면 분리는 같은 로직(이미 검증)

    eng = Engine(sys.argv[1])
    outdir = sys.argv[2]
    for path in sys.argv[3:]:
        bgr = cv2.imread(path)
        g = find_gutter(bgr)
        parts = split_spread(bgr, g) if g is not None else [bgr]
        name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        for i, p in enumerate(parts):
            rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            flat = eng.rectify(rgb)
            skip = None if len(parts) == 1 else ("right" if i == 0 else "left")
            out = eng.clean(flat, "color", True, skip)
            cv2.imwrite(f"{outdir}/{name}_p{i + 1}.jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(name, "ok")
