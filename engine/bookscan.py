"""
BookScan 보정 엔진 (프로토타입)
  1) 양면 펼침 감지 → 가운데 접힌 선 기준 좌/우 페이지 분리
  2) UVDoc(ONNX)으로 곡면 펴기 + 실제 페이지 비율 복원
  3) 손가락 감지(종이색 대비 피부색 + 가장자리 연결 + 두께 조건) → 제거
  4) 조명/그림자 평탄화, 대비·선명도 보정 (컬러 / 회색 / 흑백)

안드로이드 앱(Kotlin + OpenCV + ONNX Runtime)에서 같은 순서·같은 수치로 구현한다.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from straighten import straighten_maps

MODEL_W, MODEL_H = 488, 712  # UVDoc 입력 크기 (가로, 세로)


@dataclass
class Page:
    image: np.ndarray        # 최종 결과 (BGR 또는 GRAY)
    dewarped: np.ndarray     # 곡면만 편 결과 (디버그용)
    finger_mask: np.ndarray  # 제거한 손가락 영역 (디버그용)


# ====================================================================== 1. 양면 분리
def find_gutter(img, force=False):
    """양면 펼침이면 가운데 접힌 선을 x = a*y + b 직선으로 반환, 아니면 None."""
    h, w = img.shape[:2]
    s = 600.0 / w
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), 3).astype(np.float32)
    sh, sw = gray.shape
    if not force and sw < sh * 1.05:  # 세로로 긴 사진은 한 페이지로 본다
        return None

    x0, x1 = int(sw * 0.30), int(sw * 0.70)
    y0, y1 = int(sh * 0.15), int(sh * 0.85)
    prof = gray[y0:y1, x0:x1].mean(axis=0)
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 2).ravel()
    xv = int(np.argmin(prof))
    side = np.concatenate([prof[: max(1, xv - 15)], prof[xv + 15:]])
    depth = (np.median(side) - prof[xv]) / max(np.median(side), 1)
    if not force and depth < 0.06:  # 접힌 선 그림자가 약하면 한 페이지
        return None

    # 행 띠마다 골짜기 위치를 찾아 직선 근사 (기울어진 책 대응)
    xc = x0 + xv
    pts = []
    for yy in range(y0, y1, max(4, (y1 - y0) // 30)):
        band = gray[yy:yy + 8, max(0, xc - 40):min(sw, xc + 40)].mean(axis=0)
        pts.append((yy + 4, max(0, xc - 40) + int(np.argmin(band))))
    pts = np.array(pts, np.float32)
    a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
    resid = np.abs(pts[:, 1] - (a * pts[:, 0] + b))
    keep = resid < max(3.0, np.percentile(resid, 70))
    a, b = np.polyfit(pts[keep, 0], pts[keep, 1], 1)
    return a, b / s  # 원본 좌표계 (기울기는 비율이라 그대로)


def split_spread(img, gutter):
    """접힌 선 기준으로 좌/우 페이지 이미지를 만든다. 반대쪽은 배경색으로 가린다."""
    h, w = img.shape[:2]
    a, b = gutter
    ys = np.arange(h, dtype=np.float32)
    xg = a * ys + b
    border = np.concatenate([img[:8].reshape(-1, 3), img[-8:].reshape(-1, 3)])
    bg = np.median(border, axis=0).astype(np.uint8)
    pages = []
    xx = np.arange(w, dtype=np.float32)[None, :]
    for left in (True, False):
        out = img.copy()
        cut = (xx > xg[:, None]) if left else (xx < xg[:, None])
        out[cut] = bg
        lo = 0 if left else max(0, int(xg.min()) - 10)
        hi = min(w, int(xg.max()) + 10) if left else w
        pages.append(out[:, lo:hi])
    return pages


# ====================================================================== 2. 곡면 펴기 + 글줄 반듯하게
ANALYSIS_SIDE = 1600  # 글줄 분석 해상도(긴 변)


class Dewarper:
    def __init__(self, model_path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    def grid(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = cv2.resize(rgb, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
        g = self.sess.run(None, {"image": inp.transpose(2, 0, 1)[None]})[0][0]
        return g  # (2, 45, 31) -1~1

    @staticmethod
    def _upsample(g, w, h):
        """align_corners 방식: 출력 (0..w-1) ↔ 격자 (0..gw-1)"""
        xs = np.linspace(0, g.shape[1] - 1, w, dtype=np.float32)
        ys = np.linspace(0, g.shape[0] - 1, h, dtype=np.float32)
        ux, uy = np.meshgrid(xs, ys)
        return cv2.remap(g, ux, uy, cv2.INTER_LINEAR)

    def rectify(self, img, straighten=True):
        """
        사진 한 쪽 → 반듯한 페이지.
          곡면 펴기 지도(모델) ∘ 테두리 자르기 ∘ 글줄 반듯하게 지도 를 합성해서
          원본 사진에서 단 한 번만 재샘플링한다.
        """
        h, w = img.shape[:2]
        g = self.grid(img)
        px = ((g[0] + 1) / 2 * (w - 1)).astype(np.float32)
        py = ((g[1] + 1) / 2 * (h - 1)).astype(np.float32)
        # 격자 곡선 길이로 실제 페이지 가로·세로 추정 → 비율 복원, 해상도 유지
        row_len = np.hypot(np.diff(px, axis=1), np.diff(py, axis=1)).sum(axis=1)
        col_len = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)).sum(axis=0)
        ow, oh = float(np.median(row_len)), float(np.median(col_len))
        scale = min(1.15, (w * h / max(ow * oh, 1)) ** 0.5)
        # 모델이 잡은 페이지 경계가 머리글·쪽번호에 바짝 붙는 경우가 있어, 격자를 위아래로 한 칸씩 연장.
        # 좌우는 연장하지 않는다(책 옆면 띠가 들어오면 그 위의 손가락과 엉겨 손가락 찾기가 어려워짐).
        px, py = extend_grid_v(px, GRID_EXT), extend_grid_v(py, GRID_EXT)
        oh = oh * (GRID_H - 1 + 2 * GRID_EXT) / (GRID_H - 1)
        ow, oh = int(round(ow * scale)), int(round(oh * scale))

        # 분석용 작은 페이지
        a = min(1.0, ANALYSIS_SIDE / max(ow, oh))
        aw, ah = max(16, int(round(ow * a))), max(16, int(round(oh * a)))
        ax = self._upsample(px, aw, ah)  # 분석 픽셀 → 원본 사진 좌표
        ay = self._upsample(py, aw, ah)
        small = cv2.resize(img, None, fx=a, fy=a, interpolation=cv2.INTER_AREA) if a < 1 else img
        flat_a = cv2.remap(small, ax * a, ay * a, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        t, bt, l, r = trim_rect(flat_a)
        fa = flat_a[t:ah - bt, l:aw - r]
        info = {}
        st = straighten_maps(fa) if straighten else None
        if st is not None:
            sx, sy, info = st
        else:
            sx, sy = np.meshgrid(np.arange(fa.shape[1], dtype=np.float32), np.arange(fa.shape[0], dtype=np.float32))
        fh, fw = sx.shape
        # 합성: 최종 분석 픽셀 → fa 좌표 → flat_a 좌표 → 원본 사진 좌표
        fxa = cv2.remap(ax, sx + l, sy + t, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fya = cv2.remap(ay, sx + l, sy + t, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        FW, FH = int(round(fw / a)), int(round(fh / a))
        FX = cv2.resize(fxa, (FW, FH), interpolation=cv2.INTER_LINEAR)
        FY = cv2.resize(fya, (FW, FH), interpolation=cv2.INTER_LINEAR)
        out = cv2.remap(img, FX, FY, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return trim_dark_border(out), info

    def dewarp(self, img):
        """(이전 방식) 곡면 펴기만"""
        return self.rectify(img, straighten=False)[0]


GRID_W, GRID_H, GRID_EXT = 31, 45, 1


def extend_grid_v(g, n):
    """격자를 위아래로 n 칸 직선 연장 (가장자리 두 행의 차이만큼 계속)"""
    for _ in range(n):
        g = np.vstack([2 * g[:1] - g[1:2], g, 2 * g[-1:] - g[-2:-1]])
    return g.astype(np.float32)


def trim_rect(img, max_frac=0.06):
    """가장자리에 남은 책상·책 옆면의 어두운 띠 두께 (위, 아래, 왼쪽, 오른쪽)"""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(1.0, 1000.0 / max(h, w))
    g = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA).astype(np.float32)
    sh, sw = g.shape
    thr = np.percentile(g, 70) * 0.72

    def scan(lines, limit):
        n = 0
        for v in lines[:limit]:
            if np.median(v) < thr:
                n += 1
            else:
                break
        return n

    t = scan(g, int(sh * max_frac))
    bt = scan(g[::-1], int(sh * max_frac))
    l = scan(g.T, int(sw * max_frac))
    r = scan(g.T[::-1], int(sw * max_frac))
    return int(round(t / s)), int(round(bt / s)), int(round(l / s)), int(round(r / s))


def trim_dark_border(img, max_frac=0.06):
    t, bt, l, r = trim_rect(img, max_frac)
    h, w = img.shape[:2]
    return img[t:h - bt, l:w - r]


# ====================================================================== 3. 손가락 제거
def finger_mask(img, skip_side=None):
    """종이보다 붉은(피부색) 두꺼운 덩어리 중 페이지 가장자리에 닿은 것을 손가락으로 본다.
    skip_side: 'left' / 'right' — 양면에서 접힌 선 쪽(손가락이 오지 않는 쪽)은 검사 제외"""
    h, w = img.shape[:2]
    s = 800.0 / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    paper = L > np.percentile(L, 60)
    a0, b0 = np.median(A[paper]), np.median(B[paper])

    # 피부: 종이보다 붉은기(a*)가 확실히 높고, 색상각이 피부 범위(25~60°), 채도가 있음.
    # 누렇게 바랜 종이·책 옆면은 색상각 60° 이상(노란 쪽)이라 여기서 대부분 걸러진다.
    hue = np.degrees(np.arctan2(B, A))
    chroma = np.hypot(A, B)
    skin = ((A - a0) > 8) & (chroma > 14) & (hue > 25) & (hue < 60) & (L > 20) & (L < 90)
    skin = skin.astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # 두께 조건: 글자(붉은 인쇄)처럼 얇은 것은 침식으로 사라진다
    k = max(5, int(min(sh, sw) * 0.025)) | 1
    core = cv2.erode(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    # 두꺼운 '심지'끼리 연결 요소를 나눈다 → 손가락이 책 옆면 띠와 붙어 있어도 분리됨
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    mask = np.zeros_like(skin)
    edge = max(3, int(min(sh, sw) * 0.02)) + k // 2
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 2, k + 2))
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < sh * sw * 0.0006:
            continue
        # 심지를 원래 두께만큼 되살리되 피부색 영역 안에서만
        comp = cv2.dilate((lbl == i).astype(np.uint8), kern).astype(bool) & (skin > 0)
        # 가장자리에 닿아 있어야 하고(손가락은 바깥에서 들어온다),
        # 한 변을 따라 길게 이어진 띠(책 옆면, 누렇게 변한 가장자리)는 제외
        # (변을 따라 길이 비율, 안쪽 깊이 비율, 변을 따라 길이 px, 깊이 px)
        sides = []
        if x <= edge and skip_side != "left":
            sides.append((bh / sh, bw / sw, bh, bw))
        if x + bw >= sw - edge and skip_side != "right":
            sides.append((bh / sh, bw / sw, bh, bw))
        if y <= edge:
            sides.append((bw / sw, bh / sh, bw, bh))
        if y + bh >= sh - edge:
            sides.append((bw / sw, bh / sh, bw, bh))
        if not sides:
            continue
        if any(s_[0] > 0.40 for s_ in sides):
            continue
        if max(s_[1] for s_ in sides) < 0.02:
            continue
        # 손가락은 안쪽으로 파고든다. 가장자리를 따라 가늘고 길게 붙은 띠(책 옆면·바랜 가장자리)는 제외
        if all(s_[2] > 2.0 * s_[3] for s_ in sides):
            continue
        mask[comp] = 255

    if mask.any():
        # 손가락 가장자리(어두운 테두리)와 맞닿은 그림자까지 덮도록 확장
        d = max(5, int(min(sh, sw) * 0.03)) | 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


def fill_paper(img, mask):
    """손가락 자리를 주변 종이색으로 메운다(조명 보정 전에 해야 테두리 얼룩이 안 남음)."""
    if not mask.any():
        return img
    h, w = img.shape[:2]
    s = 600.0 / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 글자를 지운 '종이 배경'을 만들고, 손가락 밖의 종이만으로 가중 평균(정규화 컨볼루션)해
    # 손가락 자리의 종이색을 추정한다. 경계의 그림자 한 줄에 끌려가지 않는다.
    k = max(9, int(max(small.shape[:2]) * 0.03)) | 1
    paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    valid = (cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) == 0).astype(np.float32)
    sig = max(small.shape[:2]) * 0.05
    num = cv2.GaussianBlur(paper.astype(np.float32) * valid[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(valid, (0, 0), sig)[..., None]
    ref = num / np.maximum(den, 1e-3)
    up = cv2.resize(ref, (w, h), interpolation=cv2.INTER_CUBIC)
    soft = cv2.GaussianBlur(mask, (0, 0), 4).astype(np.float32)[..., None] / 255.0
    out = img.astype(np.float32) * (1 - soft) + up * soft

    # 손가락 옆 그림자 밝히기: 손가락 주변 띠에서 '글자만 지운 종이 밝기'를 기준 종이 밝기로 끌어올림.
    # 글자도 같은 비율로 밝아지므로 그림자 아래 글자는 보존된다.
    paper_s = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    lum = lambda a: a.astype(np.float32).mean(axis=2)
    gain = np.clip(lum(ref) / np.maximum(lum(paper_s), 1), 1.0, 2.5)
    gain = cv2.GaussianBlur(gain, (0, 0), 1.5)
    d = max(5, int(max(small.shape[:2]) * 0.06)) | 1
    ring = cv2.GaussianBlur(cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))), (0, 0), 3)
    g = 1 + (gain - 1) * ring.astype(np.float32) / 255.0
    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    return np.clip(out * g, 0, 255).astype(np.uint8)


# ====================================================================== 4. 화질 보정
def flatten_illumination(img):
    """종이 배경(조명·그림자)을 추정해 나눠서 배경을 균일한 흰색으로 만든다."""
    h, w = img.shape[:2]
    s = 1000.0 / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    k = max(15, int(max(small.shape[:2]) * 0.035)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se)  # 글자(어두운 것) 제거
    bg = cv2.medianBlur(bg, 21)
    bg = cv2.GaussianBlur(bg, (0, 0), 5)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    out = img.astype(np.float32) / np.maximum(bg, 1) * 255.0
    return np.clip(out, 0, 255)


def finish(norm, mode="color"):
    """norm: 조명 평탄화된 float 이미지. mode: color / gray / bw"""
    if mode == "bw":
        g = cv2.cvtColor(norm.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (0, 0), 0.8)
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
        return bw
    x = norm if mode == "color" else cv2.cvtColor(norm.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 레벨: 종이(상위)는 완전 흰색, 글자는 더 진하게
    lo = np.percentile(x, 1.0)
    hi = 238.0
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    x = x ** 1.25 * 255.0
    # 언샤프 마스크
    blur = cv2.GaussianBlur(x, (0, 0), 1.2)
    x = cv2.addWeighted(x, 1.6, blur, -0.6, 0)
    return np.clip(x, 0, 255).astype(np.uint8)


def erase(out, mask):
    """손가락 영역을 종이색(흰색)으로 채우되 경계는 부드럽게."""
    if not mask.any():
        return out
    h, w = mask.shape
    soft = cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32) / 255.0
    # 손가락 주변 띠: 남은 옅은 그림자·테두리는 흰색으로 밀고, 진한 글자는 그대로 둔다
    d = max(5, int(min(h, w) * 0.04)) | 1
    ring = cv2.GaussianBlur(cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))), (0, 0), 6)
    ring = ring.astype(np.float32) / 255.0
    lum = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    light = np.clip((lum.astype(np.float32) - 150) / 60, 0, 1) * ring
    wgt = np.maximum(soft, light)
    if out.ndim == 3:
        wgt = wgt[..., None]
    return (out * (1 - wgt) + 255 * wgt).astype(np.uint8)


# ====================================================================== 전체 파이프라인
class BookScanner:
    def __init__(self, model_path, straighten=True):
        self.dw = Dewarper(model_path)
        self.straighten = straighten

    def process(self, img, layout="auto", mode="color"):
        """layout: auto / single / double"""
        gutter = None
        if layout != "single":
            gutter = find_gutter(img, force=(layout == "double"))
        parts = split_spread(img, gutter) if gutter is not None else [img]
        pages = []
        for idx, p in enumerate(parts):
            flat, _ = self.dw.rectify(p, straighten=self.straighten)
            # 양면이면 왼쪽 페이지의 오른쪽, 오른쪽 페이지의 왼쪽이 접힌 선
            skip = None if len(parts) == 1 else ("right" if idx == 0 else "left")
            m = finger_mask(flat, skip)
            out = finish(flatten_illumination(fill_paper(flat, m)), mode)
            pages.append(Page(erase(out, m), flat, m))
        return pages


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--model", default=os.path.join(here, "..", "android", "BookScan", "app", "src", "main", "assets", "uvdoc.onnx"))
    ap.add_argument("--out", default="out")
    ap.add_argument("--layout", default="auto")
    ap.add_argument("--mode", default="color")
    ap.add_argument("--no-straighten", action="store_true", help="글줄 반듯하게 펴기 끄기(비교용)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sc = BookScanner(a.model, straighten=not a.no_straighten)
    for path in a.images:
        name = os.path.splitext(os.path.basename(path))[0]
        for i, pg in enumerate(sc.process(cv2.imread(path), a.layout, a.mode)):
            cv2.imwrite(os.path.join(a.out, f"{name}_p{i + 1}.jpg"), pg.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if pg.finger_mask.any():
                cv2.imwrite(os.path.join(a.out, f"{name}_p{i + 1}_mask.png"), pg.finger_mask)
        print(name, "done")
