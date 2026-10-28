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

from orient import detect_rotation, rotate_image
from region import BookRegion, crop_box, hull_mask
from straighten import straighten_maps

MODEL_W, MODEL_H = 488, 712  # UVDoc 입력 크기 (가로, 세로)


@dataclass
class Page:
    image: np.ndarray        # 최종 결과 (BGR 또는 GRAY)
    dewarped: np.ndarray     # 곡면만 편 결과 (디버그용)
    finger_mask: np.ndarray  # 제거한 손가락 영역 (디버그용)


# ====================================================================== 1. 양면 분리
def _text_mask_small(gray):
    """글자 픽셀(True). 그림·상자 테두리 같은 큰 덩어리는 제외"""
    sh, sw = gray.shape
    block = max(15, int(min(sh, sw) * 0.03)) | 1
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 12)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    hs, ws = st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_WIDTH]
    keep = (hs < sh * 0.04) & (ws < sw * 0.04) & (st[:, cv2.CC_STAT_AREA] >= 3)
    keep[0] = False
    return keep[lbl]


def _clearance_1d(occ, cap):
    """빈 열에서 가장 가까운 글자 열까지 거리(최대 cap). 글자 열은 0"""
    n = len(occ)
    d = np.full(n, 1e9)
    last = -1e9
    for x in range(n):
        if occ[x]:
            last = x
        d[x] = x - last
    last = 1e9
    for x in range(n - 1, -1, -1):
        if occ[x]:
            last = x
        d[x] = min(d[x], last - x)
    return np.minimum(d, cap)


def _robust_line(Y, X):
    """x = a*y + b, 튀는 점(손가락·책갈피로 패인 곳) 제외하고 두 번 맞춤"""
    Y, X = np.asarray(Y, np.float64), np.asarray(X, np.float64)
    a, b = np.polyfit(Y, X, 1)
    for _ in range(2):
        r = X - (a * Y + b)
        mad = np.median(np.abs(r - np.median(r))) + 1e-9
        keep = np.abs(r) < max(2.0, 3.0 * mad)
        if keep.sum() < 5:
            break
        a, b = np.polyfit(Y[keep], X[keep], 1)
    return float(a), float(b)


def book_axis(mask_small):
    """
    책 마스크(작은 크기)에서 양면의 가운데 선과 가로세로비.
      가운데 선: 가운데 높이(30~70%) 행마다 책의 왼쪽·오른쪽 바깥 변의 중점 → 직선.
                 두 페이지 폭이 같으므로 접힌 선은 거의 이 선 위에 있다(책이 기울어져도).
      가로세로비: 책을 감싸는 가장 작은 회전 사각형의 (가로에 가까운 변) / (세로에 가까운 변)
    반환 (a, b, aspect) 또는 책이 사진 밖으로 잘려 믿을 수 없으면 None
    """
    m = mask_small > 127
    sh, sw = m.shape
    rows_on = np.nonzero(m.any(axis=1))[0]
    if len(rows_on) < 20:
        return None
    y0, y1 = rows_on[0], rows_on[-1]
    Y, L, R = [], [], []
    for y in range(int(y0 + 0.3 * (y1 - y0)), int(y0 + 0.7 * (y1 - y0)) + 1):
        c = np.nonzero(m[y])[0]
        if len(c):
            Y.append(y)
            L.append(c[0])
            R.append(c[-1])
    if len(Y) < 10:
        return None
    L, R = np.array(L), np.array(R)
    if (L <= 1).mean() > 0.3 or (R >= sw - 2).mean() > 0.3:  # 책 바깥 변이 사진 밖
        return None
    al, bl = _robust_line(Y, L)
    ar, br = _robust_line(Y, R)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = cv2.boxPoints(cv2.minAreaRect(max(cnts, key=cv2.contourArea)))
    v1, v2 = box[1] - box[0], box[2] - box[1]
    n1, n2 = float(np.hypot(*v1)), float(np.hypot(*v2))
    if n1 < 1 or n2 < 1:
        return None
    # 가로에 가까운 변
    if abs(v1[0]) / n1 >= abs(v2[0]) / n2:
        aspect = n1 / n2
    else:
        aspect = n2 / n1
    return (al + ar) / 2.0, (bl + br) / 2.0, aspect


def find_gutter(img, force=False, mask=None):
    """
    양면 펼침이면 가운데 접힌 선을 x = a*y + b 직선(원본 좌표)으로 반환, 아니면 None.
    접힌 선 = 두 페이지 글 사이의 '글 없는 세로 띠'를 위에서 아래까지 가장 여유 있게 지나는 직선.
    (밝기만 보면 조명 때문에 한쪽 페이지가 통째로 어두울 때 엉뚱한 곳을 잡음)
    mask(책 영역)가 있으면: 책이 가로로 긴지(양면)로 판단하고, 책 바깥 변의 가운데 선 근처에서만 찾는다.
    """
    h, w = img.shape[:2]
    s = 800.0 / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape
    axis = None
    if mask is not None:
        axis = book_axis(cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_LINEAR))
    if axis is not None:
        if not force and axis[2] < SPREAD_ASPECT:  # 책이 세로로 길면 한 쪽
            return None
    elif not force and sw < sh * 1.05:  # 세로로 긴 사진은 한 페이지로 본다
        return None
    txt = _text_mask_small(gray)
    cx = (sw - 1) / 2.0
    nb = 14
    ys = np.linspace(sh * 0.06, sh * 0.94, nb + 1).astype(int)
    yc = (ys[:-1] + ys[1:]) / 2.0
    cap = max(3.0, sw * 0.03)
    lo = int(max(0, cx - sw * 0.22))
    hi = int(min(sw - 1, cx + sw * 0.22))
    clear = np.zeros((nb, sw), np.float64)
    informative = 0
    for k in range(nb):
        dens = txt[ys[k]:ys[k + 1], :].sum(axis=0)
        occ = np.convolve(dens, np.ones(3), mode="same") > 0
        clear[k] = _clearance_1d(occ, cap)
        if clear[k][lo:hi + 1].max() > 0 and occ[lo:hi + 1].any():
            informative += 1  # 글과 빈 칸이 다 있는 띠만 판단에 씀(바닥·그림만 있는 띠 제외)
    xs = np.arange(lo, hi + 1)
    span = yc[-1] - yc[0]
    max_dx = np.tan(np.radians(15)) * span
    XT, XB = np.meshgrid(xs, xs, indexing="ij")
    score = np.zeros(XT.shape)
    hits = np.zeros(XT.shape, np.int32)
    for k in range(nb):
        t = (yc[k] - yc[0]) / span
        xk = np.clip(np.round(XT + (XB - XT) * t).astype(int), 0, sw - 1)
        ck = clear[k][xk]
        score += ck
        hits += ck > 0
    if axis is not None:
        # 책 가운데 선 근처만, 가까울수록 우선
        pt, pb = axis[0] * yc[0] + axis[1], axis[0] * yc[-1] + axis[1]
        dev = (np.abs(XT - pt) + np.abs(XB - pb)) / 2.0
        score -= AXIS_PULL * cap * dev / sw * nb
        score[(np.abs(XT - pt) > sw * AXIS_WIN) | (np.abs(XB - pb) > sw * AXIS_WIN)] = -1e18
        cxx = (pt + pb) / 2.0
    else:
        xm_all = (XT + XB) / 2.0
        score -= 0.5 * cap * np.abs(xm_all - cx) / sw * nb  # 같은 점수면 가운데 쪽
        cxx = cx
    score[np.abs(XB - XT) > max_dx] = -1e18
    if score.max() < -1e17:
        return None
    # 빈 띠 점수가 최고점 근처인 후보들을 점수 순으로 보며, '접힌 그림자'가 있는 첫 직선을 고른다.
    # (빈 띠만 보면 한쪽 페이지 여백 → 반대쪽 페이지 여백을 비스듬히 잇는 직선이 이기는 경우가 있음)
    paper = _fold_paper(gray)
    best = float(score.max())
    tol = GUTTER_TOL * nb * cap
    order = np.argsort(score.ravel())[::-1]
    cands = []
    for idx in order[:20000]:
        if score.flat[idx] < best - tol:
            break
        i, j = divmod(int(idx), score.shape[1])
        if any(abs(i - ci) <= 2 and abs(j - cj) <= 2 for ci, cj in cands):
            continue
        cands.append((i, j))
        if len(cands) >= GUTTER_CANDS:
            break
    k, fold = 0, 0.0
    for n_, (i, j) in enumerate(cands):
        xt_, xb_ = float(xs[i]), float(xs[j])
        a_ = (xb_ - xt_) / span
        f_ = _fold_shadow_p(paper, a_, xt_ - a_ * yc[0])
        if n_ == 0:
            fold = f_
        if f_ >= FOLD_MIN:
            k, fold = n_, f_
            break
    i, j = cands[k]
    xt, xb = float(xs[i]), float(xs[j])
    if DEBUG is not None:
        DEBUG.update(fold=round(fold, 4), k=k, ncand=len(cands), frac=round(hits[i, j] / max(1, informative), 2), inf=informative)
    a = (xb - xt) / span
    b = xt - a * yc[0]
    xm = (xt + xb) / 2.0
    if not force:
        frac = hits[i, j] / max(1, informative)
        left_txt = txt[:, :max(1, int(xm))].sum()
        right_txt = txt[:, int(xm):].sum()
        balance = min(left_txt, right_txt) / max(1, max(left_txt, right_txt))
        if informative < 6 or frac < 0.7 or abs(xm - cxx) > sw * 0.15 or balance < 0.25:
            return None
        # 진짜 접힌 선에는 종이가 꺾이며 생기는 옅은 그림자 골짜기가 있다(표의 칸 사이 빈 줄과 구분).
        # 책 모양이 확실히 양면(가로로 김)이면 그림자가 옅어도 받아들인다.
        if fold < FOLD_MIN:
            return None
    return a, b / s


DEBUG = None
SPREAD_ASPECT = 1.1        # 책(마스크)이 이보다 가로로 길어야 양면 후보
AXIS_PULL = 2.0           # 책 가운데 선에서 벗어난 만큼 깎는 점수(빈 띠 점수 단위)
AXIS_WIN = 0.07            # 책 가운데 선에서 이 폭(사진 폭 비율) 안에서만 접힌 선을 찾음
GUTTER_TOL = 0.2     # 최고점에서 (띠 수 x 최대 여유)의 이 비율 안쪽이면 후보
GUTTER_CANDS = 24
FOLD_MIN = 0.012


def _fold_paper(gray):
    """글자·가는 선을 지운 종이 밝기"""
    sw = gray.shape[1]
    k = max(7, int(sw * 0.025)) | 1
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)


def _fold_shadow(gray, a, b):
    return _fold_shadow_p(_fold_paper(gray), a, b)


def _fold_shadow_p(paper, a, b):
    """직선 x = a*y + b (작은 이미지 좌표) 를 따라 잰 그림자 골짜기 깊이(중앙값, 종이 밝기 대비 비율)"""
    sh, sw = paper.shape
    W = max(6, int(sw * 0.05))
    rels = []
    for yy in np.linspace(sh * 0.1, sh * 0.9, 12):
        y0 = int(yy)
        band = np.median(paper[max(0, y0 - 8):y0 + 8], axis=0)
        xl = int(np.round(a * yy + b))
        if xl - W < 0 or xl + W >= sw:
            continue
        c = band[xl - 3:xl + 4].min()
        side = min(band[xl - W:xl - 3].max(), band[xl + 4:xl + W].max())
        rels.append((side - c) / max(side, 1.0))
    return float(np.median(rels)) if rels else 0.0


def split_spread(img, gutter, mask=None):
    """접힌 선 기준으로 좌/우 페이지 이미지를 만든다. 반대쪽은 배경색으로 가린다.
    mask(책 영역)가 있으면 같은 방식으로 나눈 마스크도 함께 돌려준다: [(page, page_mask), ...]"""
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
        pm = None
        if mask is not None:
            pm = mask.copy()
            pm[cut] = 0
            pm = pm[:, lo:hi]
        pages.append((out[:, lo:hi], pm))
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

    def rectify(self, img, straighten=True, mask=None):
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
        # 가로세로 비율 보정: 모델의 격자 길이로 잰 페이지 폭은 휘어 기울어진 부분이 사진에서 짧게 찍혀
        # 실제보다 좁게 나온다(글자가 세로로 길쭉해짐). 지도의 국소 배율(가로/세로)이 평평한 곳에서 1이 되도록 폭을 맞춘다.
        r_as = aspect_ratio_fix(fxa, fya)
        info["aspect_fix"] = r_as
        FW, FH = int(round(fw / a * r_as)), int(round(fh / a))
        FX = cv2.resize(fxa, (FW, FH), interpolation=cv2.INTER_LINEAR)
        FY = cv2.resize(fya, (FW, FH), interpolation=cv2.INTER_LINEAR)
        out = cv2.remap(img, FX, FY, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        t, bt, l, r = trim_rect(out)
        H_, W_ = out.shape[:2]
        out = out[t:H_ - bt, l:W_ - r]
        if mask is not None:
            pm = cv2.remap(mask, FX, FY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            out = apply_page_mask(out, pm[t:H_ - bt, l:W_ - r])
        return out, info

    def dewarp(self, img):
        """(이전 방식) 곡면 펴기만"""
        return self.rectify(img, straighten=False)[0]


def apply_page_mask(flat, pm):
    """
    곡면 편 페이지에서 책 바깥(바닥·책상 무늬)을 지운다.
      1) 마스크가 절반도 안 덮는 가장자리 행·열은 잘라내고
      2) 남은 부분 중 마스크 바깥은 주변 종이색으로 메운다(손가락 메우기와 같은 방법) → 보정 후 흰색.
    마스크 경계가 거칠어(320 격자) 조금 안쪽으로 줄여서 쓴다. 페이지 여백은 비어 있어 손실 없음.
    """
    if pm is None:
        return flat
    h, w = pm.shape
    inside = pm > 127
    if inside.mean() < 0.3:  # 마스크가 이상하면 손대지 않음
        return flat
    e = max(3, int(min(h, w) * PAGE_MASK_ERODE)) | 1
    inside = cv2.erode(inside.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e))) > 0
    colc, rowc = inside.mean(axis=0), inside.mean(axis=1)
    cols, rows = np.where(colc >= 0.5)[0], np.where(rowc >= 0.5)[0]
    if len(cols) < w * 0.5 or len(rows) < h * 0.5:
        return flat
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    res = flat[y0:y1, x0:x1]
    outside = np.where(inside[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    if outside.any():
        res = fill_paper(res, outside)
    return np.ascontiguousarray(res)


PAGE_MASK_ERODE = 0.012
ISOLATE_MARGIN = 0.03   # 페이지 상자 바깥 여유
ISOLATE_DILATE = 0.02   # 마스크를 이만큼 넓힌 바깥만 단색으로
ISOLATE_FILL = (30, 30, 30)


def isolate_page(p, pm):
    """
    곡면 펴기 모델 입력 정리: 페이지 상자(+여유)로 자르고, 페이지(조금 넓힌 마스크) 바깥은 단색으로 칠한다.
    바닥 무늬·옆 페이지가 사라져 모델이 페이지 경계를 정확히 잡는다.
    """
    h, w = p.shape[:2]
    x, y, bw, bh = cv2.boundingRect(pm)
    if bw * bh < 0.1 * w * h:
        return p, pm
    mx, my = int(bw * ISOLATE_MARGIN), int(bh * ISOLATE_MARGIN)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    p, pm = p[y0:y1, x0:x1].copy(), pm[y0:y1, x0:x1]
    d = max(3, int(max(bw, bh) * ISOLATE_DILATE)) | 1
    keep = cv2.dilate(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    p[keep == 0] = ISOLATE_FILL
    return p, pm


GRID_W, GRID_H, GRID_EXT = 31, 45, 1
ASPECT_PCT = 75          # 국소 배율비의 이 백분위를 '평평한 곳'의 값으로 봄 (휜 곳은 가로가 짧게 찍혀 비가 작다)
ASPECT_MIN, ASPECT_MAX = 0.85, 1.25


def aspect_ratio_fix(mx, my):
    """
    출력 폭에 곱할 비율. mx, my: 출력(분석 해상도) 픽셀 → 원본 사진 좌표 지도.
    가로 한 칸이 사진에서 차지하는 길이 / 세로 한 칸이 차지하는 길이 를 페이지 안쪽에서 재고,
    평평한 곳(비가 큰 쪽)의 값을 쓴다. 그 값이 1보다 작으면 출력이 가로로 눌린 것이므로 폭을 그만큼 늘린다.
    """
    h, w = mx.shape
    st = max(4, min(h, w) // 40)
    ys = np.arange(int(h * 0.1), int(h * 0.9) - st, st)
    xs = np.arange(int(w * 0.1), int(w * 0.9) - st, st)
    if len(ys) < 3 or len(xs) < 3:
        return 1.0
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    ju = np.hypot(mx[Y, X + st] - mx[Y, X], my[Y, X + st] - my[Y, X])
    jv = np.hypot(mx[Y + st, X] - mx[Y, X], my[Y + st, X] - my[Y, X])
    ratio = ju / np.maximum(jv, 1e-6)
    r = float(np.percentile(ratio, ASPECT_PCT))
    return float(np.clip(r, ASPECT_MIN, ASPECT_MAX))


def extend_grid_v(g, n):
    """격자를 위아래로 n 칸 직선 연장 (가장자리 두 행의 차이만큼 계속)"""
    for _ in range(n):
        g = np.vstack([2 * g[:1] - g[1:2], g, 2 * g[-1:] - g[-2:-1]])
    return g.astype(np.float32)


def trim_rect(img, max_frac=0.06):
    """
    가장자리에서 잘라낼 두께 (위, 아래, 왼쪽, 오른쪽).
      1) 어두운 띠(책상, 그림자)
      2) 무늬 띠(바닥 카펫, 책 옆면의 종이 단면 줄무늬): 바깥에서 안쪽으로 훑다가
         빈 종이 여백이 시작되는 곳까지. 글·그림이 있는 곳은 자르지 않음.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(1.0, 1000.0 / max(h, w))
    g8 = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    g = g8.astype(np.float32)
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

    # 무늬 띠
    gx = cv2.Sobel(g8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g8, cv2.CV_32F, 0, 1, ksize=3)
    busy = cv2.dilate((gx * gx + gy * gy > 40.0 * 40.0).astype(np.uint8), np.ones((5, 5), np.uint8))
    blank = 1.0 - busy.astype(np.float32)
    r0, r1, c0, c1 = int(sh * 0.1), int(sh * 0.9), int(sw * 0.1), int(sw * 0.9)
    colb = blank[r0:r1].mean(axis=0)
    rowb = blank[:, c0:c1].mean(axis=1)
    # 결 방향: 책 옆면(종이 단면)은 변과 나란한 줄무늬라 한 방향 기울기만 강하다. 글자는 두 방향 다 있음
    ax_, ay_ = np.abs(gx), np.abs(gy)
    col_gx, col_gy = ax_[r0:r1].sum(axis=0), ay_[r0:r1].sum(axis=0)
    row_gx, row_gy = ax_[:, c0:c1].sum(axis=1), ay_[:, c0:c1].sum(axis=1)

    def tex(vals, along, across, limit):
        """vals: 빈 비율, along/across: 변과 수직/나란한 방향 기울기 합 (바깥→안 순서).
        바깥에서 안으로 훑다가 빈 여백이 시작되는 곳(i)까지가 잘라낼 후보 띠.
          - 띠 전체가 무늬(카펫 등): 한도 10%
          - 띠가 변과 나란한 줄무늬(책 옆면 종이 단면, 두꺼운 책은 폭이 넓음): 한도 15%"""
        for i in range(min(int(limit * 1.5), len(vals) - 3)):
            if vals[i] >= 0.75 and vals[i + 1] >= 0.75 and vals[i + 2] >= 0.75:
                if i == 0:
                    return 0
                if i <= limit and vals[:i].mean() < 0.3:  # 잘라낼 띠가 '무늬'일 때만
                    return i
                if along[:i].sum() > STREAK_RATIO * across[:i].sum():  # 변과 나란한 줄무늬(책 옆면)
                    return i
                return 0
        return 0

    lim_h, lim_w = int(sh * 0.10), int(sw * 0.10)
    t = max(t, tex(rowb, row_gy, row_gx, lim_h))
    bt = max(bt, tex(rowb[::-1], row_gy[::-1], row_gx[::-1], lim_h))
    l = max(l, tex(colb, col_gx, col_gy, lim_w))
    r = max(r, tex(colb[::-1], col_gx[::-1], col_gy[::-1], lim_w))
    return int(round(t / s)), int(round(bt / s)), int(round(l / s)), int(round(r / s))


STREAK_RATIO = 2.2


def trim_dark_border(img, max_frac=0.06):
    t, bt, l, r = trim_rect(img, max_frac)
    h, w = img.shape[:2]
    return img[t:h - bt, l:w - r]


# ====================================================================== 3. 손가락 제거
FINGER_MIN_DEPTH = 0.02   # 손가락이 변에서 안쪽으로 들어온 깊이(변과 수직인 길이 대비) 최소
FINGER_MAX_ALONG = 0.40   # 손가락 몸통이 변을 따라 차지하는 길이 최대(변 길이 대비)
FINGER_ALONG_DEPTH = 2.0  # 손가락 몸통의 (변을 따라 길이) / (깊이) 최대 — 이보다 길쭉하면 책 옆면 띠


def _finger_bulge(comp, side):
    """
    변에 닿은 피부색 덩어리에서 안쪽으로 볼록 튀어나온 손가락 부분(bool 마스크) 또는 None.
    변을 따라 행(또는 열)마다 '변에서 얼마나 깊이 들어왔나'를 재고, 가장 깊은 곳의 절반 이상 들어온
    구간(손가락 몸통)이 짧고 깊으면 손가락. 몸통 구간 앞뒤로 조금 넓혀(손가락 끝 둥근 부분) 그 부분만 돌려준다.
    """
    sh, sw = comp.shape
    c = comp if side in ("left", "right") else comp.T
    n_along, n_depth = c.shape
    if side in ("right", "bottom"):
        c = c[:, ::-1]
    any_ = c.any(axis=1)
    if not any_.any():
        return None
    # 변(열 0)에서 이어진 부분의 깊이: 행마다 가장 안쪽 픽셀 + 1
    depth = np.where(any_, n_depth - np.argmax(c[:, ::-1], axis=1), 0)
    dmax = int(depth.max())
    if dmax < FINGER_MIN_DEPTH * n_depth:
        return None
    core = depth >= 0.5 * dmax
    # 가장 긴 연속 구간
    best, cur, b0, c0 = 0, 0, 0, 0
    for i, v in enumerate(core):
        if v:
            if cur == 0:
                c0 = i
            cur += 1
            if cur > best:
                best, b0 = cur, c0
        else:
            cur = 0
    if best > FINGER_MAX_ALONG * n_along or best > FINGER_ALONG_DEPTH * dmax:
        return None
    pad = int(0.6 * dmax)
    r0, r1 = max(0, b0 - pad), min(n_along, b0 + best + pad)
    out = np.zeros_like(c)
    out[r0:r1] = c[r0:r1]
    if side in ("right", "bottom"):
        out = out[:, ::-1]
    return out if side in ("left", "right") else out.T


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
        # 가장자리에 닿아 있어야 하고(손가락은 바깥에서 들어온다), 안쪽으로 볼록 튀어나온 부분만 손가락.
        # 손가락이 책 옆면 띠(변을 따라 길고 얕은 피부색 띠)와 붙어 있으면 띠 부분은 떼어낸다.
        part = np.zeros_like(comp)
        for side in ("left", "right", "top", "bottom"):
            if side == skip_side:
                continue
            touch = {"left": x <= edge, "right": x + bw >= sw - edge,
                     "top": y <= edge, "bottom": y + bh >= sh - edge}[side]
            if not touch:
                continue
            got = _finger_bulge(comp, side)
            if got is not None:
                part |= got
        if part.any():
            mask[part] = 255

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
FLAT_PREBLUR = 1.5  # 배경 추정 전 흐림(긴 변 1000px 기준)


def flatten_illumination(img):
    """종이 배경(조명·그림자)을 추정해 나눠서 배경을 균일한 흰색으로 만든다."""
    h, w = img.shape[:2]
    s = 1000.0 / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # 종이 결·잡티의 밝은 점이 배경을 실제보다 밝게 끌어올리지 않도록 살짝 흐린 뒤 배경을 잰다
    # (안 그러면 그림자 진 곳·어두운 곳의 종이가 회색으로 남아 뒷면 비침과 구분이 안 됨)
    small = cv2.GaussianBlur(small, (0, 0), FLAT_PREBLUR)
    k = max(15, int(max(small.shape[:2]) * 0.035)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se)  # 글자(어두운 것) 제거
    bg = cv2.medianBlur(bg, 21)
    bg = cv2.GaussianBlur(bg, (0, 0), 5)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    out = img.astype(np.float32) / np.maximum(bg, 1) * 255.0
    # 어두운 곳은 밝히면서 노이즈도 같이 커진다(배율 = 255/배경 밝기) → 배율이 큰 곳만 살짝 흐려 잡음을 누른다
    gain = 255.0 / np.maximum(bg.mean(axis=2), 1.0)
    wn = np.clip((gain - DENOISE_GAIN0) / DENOISE_GAIN_SPAN, 0, 1)
    if wn.max() > 0:
        bl = cv2.GaussianBlur(out, (0, 0), DENOISE_SIGMA)
        out = out + (bl - out) * wn[..., None]
    return np.clip(out, 0, 255)


DENOISE_GAIN0 = 1.5      # 배율이 이보다 크면 흐리기 시작
DENOISE_GAIN_SPAN = 2.5  # 배율 1.5+2.5=4(배경 밝기 64) 에서 완전히 흐림
DENOISE_SIGMA = 1.0


PAPER_PCT = 60        # 종이 밝기 = 조명 평탄화 후 밝기의 이 백분위
WHITE_TEXT = 0.95     # 글자 근처의 흰색 기준(종이 밝기 대비): 글자 가장자리의 부드러운 회색을 살림
WHITE_BG = 0.86       # 글자에서 떨어진 곳의 흰색 기준: 뒷면 비침·종이 결을 지움
INK_DARK = 0.62       # 종이 밝기의 이 배보다 어두우면 확실한 글자(인쇄 잉크)
LINE_FAINT = 0.95     # 상자 테두리·밑줄처럼 긴 선은 옅어도 살림
LINE_DARK = 0.80      # 긴 선 조각 중 이 비율보다 어두운 픽셀이
LINE_DARK_FRAC = 0.25 # 이만큼은 있어야 인쇄된 선(접힌 자국·그림자 경계는 더 옅고 고름)
INK_NEAR = 0.006      # 글자 근처로 보는 거리(페이지 폭 비율)
CHROMA_KEEP = 8.0     # 종이 색을 뺀 채도가 이보다 진하면(색 상자·그림·색 글씨) 살림
COLOR_SAT = 1.3       # 색을 이만큼 진하게(1 = 그대로)
BLACK_INK_PCT = 50.0  # 글자 픽셀 밝기의 이 백분위를 완전한 검정으로
BLACK_OUT = 12.0      # 검정이 놓일 값 (0으로 몰면 글자 가장자리가 뭉개짐)
TONE_GAMMA = 1.5      # 1보다 크면 글자가 더 진하고 굵게
S_MIX = 0.6           # S자 곡선 섞는 비율(0~1): 클수록 글자 둘레 얼룩이 깨끗해지지만 옅은 인쇄도 옅어짐
SHARPEN = 0.3         # 언샤프 마스크 세기 (너무 세면 글자가 얼룩덜룩해짐)
SHARPEN_SIGMA = 1.0
BW_T = 140            # 흑백 모드: 톤 보정 후 이보다 어두우면 검정
BW_UPSCALE_BELOW = 1600  # 흑백 모드: 페이지 폭이 이보다 작으면 2배로 키워서 자름


def tone_map(norm):
    """
    조명 평탄화된 float 이미지 → 종이는 흰색, 글자는 원래 모양(가장자리의 부드러운 회색 포함) 그대로 진하게.
    글자를 이진화하듯 깎지 않는다(획이 끊기거나 얇아지거나 얼룩덜룩해지지 않게).
      - 흰색 기준을 자리마다 다르게: 진한 글자·긴 선·색 있는 곳 근처는 높게(가장자리 보존),
        그 밖은 낮게(뒷면에서 비치는 글씨와 종이 얼룩은 흰색으로)
      - 밝기만 바꾸고 색은 같은 비율로 유지(색 상자·그림 보존)
    """
    color = norm.ndim == 3
    n8 = np.clip(norm, 0, 255).astype(np.uint8)
    g = cv2.cvtColor(n8, cv2.COLOR_BGR2GRAY).astype(np.float32) if color else n8.astype(np.float32)
    h, w = g.shape
    s = min(1.0, 1000.0 / max(h, w))
    gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    paper = float(np.percentile(gs, PAPER_PCT))
    core = gs[gs < paper * INK_DARK]
    black = float(np.percentile(core, BLACK_INK_PCT)) if core.size > 100 else float(np.percentile(gs, 1.0))
    black = min(black, paper * WHITE_BG - 40.0)
    # 글자 근처 지도
    ink = (g < paper * INK_DARK).astype(np.uint8)
    faint = (g < paper * LINE_FAINT).astype(np.uint8)
    L = max(9, int(w * 0.03))
    lines = cv2.morphologyEx(faint, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))) | \
        cv2.morphologyEx(faint, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    # 종이 접힌 자국·그림자 경계도 긴 '옅은 선'으로 잡히므로, 선 조각마다 충분히 진한 픽셀이 있어야 인쇄된 선으로 본다
    nl, ll, _, _ = cv2.connectedComponentsWithStats(lines, 8)
    if nl > 1:
        darkpx = (g < paper * LINE_DARK).astype(np.int64)
        cnt = np.bincount(ll.ravel(), minlength=nl)
        dk = np.bincount(ll.ravel(), weights=darkpx.ravel(), minlength=nl)
        keep_l = dk >= LINE_DARK_FRAC * np.maximum(cnt, 1)
        keep_l[0] = False
        lines = keep_l[ll].astype(np.uint8)
    r = max(2, int(w * INK_NEAR))
    keep = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) | \
        cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if color:
        # 색: 종이 색(누런 기)을 뺀 채도. 색 상자·그림·색 글씨 근처는 지우지 않는다
        lab = cv2.cvtColor(n8, cv2.COLOR_BGR2LAB).astype(np.float32)
        labs = cv2.resize(lab, (gs.shape[1], gs.shape[0]), interpolation=cv2.INTER_AREA) if s < 1 else lab
        pp = gs >= paper * 0.97
        a_p = float(np.median(labs[..., 1][pp])) if pp.any() else 128.0
        b_p = float(np.median(labs[..., 2][pp])) if pp.any() else 128.0
        chroma = cv2.GaussianBlur(np.hypot(lab[..., 1] - a_p, lab[..., 2] - b_p), (0, 0), 2)
        keep |= (chroma > CHROMA_KEEP).astype(np.uint8)
    near = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), max(1.0, r / 2.0))
    white = paper * (WHITE_BG + (WHITE_TEXT - WHITE_BG) * near)
    t = np.clip((g - black) / np.maximum(white - black, 1.0), 0, 1)
    # 곡선: 감마(글자 진하게)와 S자 곡선(글자 둘레의 옅은 얼룩·비침을 흰색 쪽으로)을 섞는다.
    # 상자 테두리·밑줄 같은 옅은 긴 선 위에서는 S자를 쓰지 않는다(옅은 선이 사라지지 않게).
    lw = cv2.GaussianBlur(cv2.dilate(lines, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.0)
    sm = S_MIX * (1.0 - lw)
    u = (1.0 - sm) * t + sm * (t * t * (3.0 - 2.0 * t))
    y = BLACK_OUT + (255.0 - BLACK_OUT) * u ** TONE_GAMMA
    if color:
        # 밝기는 톤 곡선대로, 색(a, b)은 종이 색을 빼고 조금 진하게. 종이 부분은 순수한 흰색.
        # (RGB 에 같은 비율을 곱하면 밝은 색이 255 에서 잘려 색이 바래므로 Lab 에서 처리)
        # 색이 있는 곳만 색을 남긴다. 검은 글자의 가장자리 색 번짐(카메라 색수차)은 무채색으로
        colored = np.clip((chroma - CHROMA_KEEP * 0.5) / CHROMA_KEEP, 0, 1)
        Lp = lab[..., 0] + (y - g)
        ap = 128.0 + (lab[..., 1] - a_p) * (COLOR_SAT * colored)
        bp = 128.0 + (lab[..., 2] - b_p) * (COLOR_SAT * colored)
        out = cv2.cvtColor(np.clip(np.dstack([Lp, ap, bp]), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        out[(t >= 1) & (colored <= 0)] = 255.0
    else:
        out = y
    if SHARPEN > 0:
        bl = cv2.GaussianBlur(out, (0, 0), SHARPEN_SIGMA)
        out = out * (1 + SHARPEN) - bl * SHARPEN
    return np.clip(out, 0, 255).astype(np.uint8)


def finish(norm, mode="color"):
    """norm: 조명 평탄화된 float 이미지. mode: color / gray / bw"""
    if mode == "color":
        return tone_map(norm)
    g = cv2.cvtColor(np.clip(norm, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    y = tone_map(g)
    if mode == "gray":
        return y
    # 흑백: 조명이 이미 고르게 펴졌으므로 한 기준으로 자른다(적응형 이진화는 흐린 획을 끊어 먹음).
    # 해상도가 낮으면 2배로 키워 자른 뒤 되돌려 글자 가장자리가 계단지지 않게.
    yf = y.astype(np.float32)
    h, w = yf.shape
    if w < BW_UPSCALE_BELOW:
        up = cv2.resize(yf, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        up = cv2.GaussianBlur(up, (0, 0), 1.0)
        bw = np.where(up < BW_T, 0, 255).astype(np.uint8)
        return cv2.resize(bw, (w, h), interpolation=cv2.INTER_AREA)
    yf = cv2.GaussianBlur(yf, (0, 0), 0.7)
    return np.where(yf < BW_T, 0, 255).astype(np.uint8)


def clean_edges(out, gutter_side=None):
    """
    가장자리 띠(각 변의 4%, 접힌 선 쪽은 6%) 안의 선 찌꺼기(페이지 경계선, 접힌 선 그림자, 책 옆면 줄무늬)를 흰색으로.
      - 긴 세로선·가로선: 글자는 짧은 획들이라 긴 선 모양 걸러내기(열림 연산)를 통과하지 못한다.
      - 띠 안에 완전히 들어 있는 가늘고 긴 조각(변과 나란한 방향으로 2.5배 이상 김)
    gutter_side: 'left' / 'right'
    """
    gray = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(1.0, 1200.0 / max(h, w))
    g = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    sh, sw = g.shape
    dark = (g < 200).astype(np.uint8) * 255
    bl = max(2, int(sw * (EDGE_GUTTER if gutter_side == "left" else EDGE_BAND)))
    br = max(2, int(sw * (EDGE_GUTTER if gutter_side == "right" else EDGE_BAND)))
    by = max(2, int(sh * EDGE_BAND))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(sh * 0.04))))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(sw * 0.04)), 1))
    m = np.zeros_like(dark)
    # 띠 안의 긴 선 조각마다: 상자 테두리(가로로 넓게 이어진 덩어리의 일부이고 페이지 위아래 끝에 닿지 않음)는 남기고,
    # 페이지 경계선·접힌 곳 그림자(홀로 떨어진 선이거나 페이지 끝까지 이어진 선)는 지운다.
    # 접힌 선 쪽 띠에서는 옅은 그림자 선(밝기 235 미만)까지 본다.
    cache = {}

    def prep(thr):
        if thr not in cache:
            d = (g < thr).astype(np.uint8) * 255
            n_, lbl_, st_, _ = cv2.connectedComponentsWithStats(d, 8)
            cache[thr] = (cv2.morphologyEx(d, cv2.MORPH_OPEN, kv), cv2.morphologyEx(d, cv2.MORPH_OPEN, kh),
                          (st_[:, cv2.CC_STAT_WIDTH] > sw * EDGE_BOX)[lbl_], (st_[:, cv2.CC_STAT_HEIGHT] > sh * EDGE_BOX)[lbl_])
        return cache[thr]

    bands = [("left", (slice(None), slice(0, bl)), True), ("right", (slice(None), slice(sw - br, sw)), True),
             ("top", (slice(0, by), slice(None)), False), ("bottom", (slice(sh - by, sh), slice(None)), False)]
    for name, sl_, vertical in bands:
        vert, horz, wide, tall = prep(EDGE_FAINT if name == gutter_side else 200)
        seg_mask = np.zeros_like(dark)
        seg_mask[sl_] = (vert if vertical else horz)[sl_]
        owner_big = wide if vertical else tall
        along_lo, along_hi = (by, sh - by) if vertical else (bl, sw - br)
        ns, sl, ss, _ = cv2.connectedComponentsWithStats(seg_mask, 8)
        for j in range(1, ns):
            x, y, bw, bh, _ = ss[j]
            pix = sl[y:y + bh, x:x + bw] == j
            boxy = owner_big[y:y + bh, x:x + bw][pix].mean() > 0.5
            a0, a1 = (y, y + bh) if vertical else (x, x + bw)
            reaches_end = a0 <= along_lo or a1 >= along_hi
            if boxy and not reaches_end:
                continue
            m[y:y + bh, x:x + bw][pix] = 255
    n, lbl, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    # 띠 안의 가늘고 긴 조각: 좌우 띠는 세로로 긴 조각(책 옆면 줄), 위아래 띠는 아주 긴 가로 조각(페이지 경계선)만.
    # (글자 획 하나 — 예: 'ㅣ', '그'의 'ㅡ' — 를 지우지 않도록 글자 높이보다 훨씬 긴 것만)
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        side_lr = (x + bw <= bl or x >= sw - br) and bh >= 2.5 * bw and bh >= sh * EDGE_BOX
        side_tb = (y + bh <= by or y >= sh - by) and bw >= 8 * bh and bw >= sw * 0.05
        if side_lr or side_tb:
            m[y:y + bh, x:x + bw][lbl[y:y + bh, x:x + bw] == i] = 255
    # 좌우 변에 붙은 세로 잔무늬 띠(바닥 카펫·책 옆면 찌꺼기): 어두운 점이 빽빽한 곳이 변에 닿아 있고
    # 안쪽으로 얕게(8% 이내)만 들어왔으며 세로로 긴 것. 글줄(가로로 긴 덩어리)은 건드리지 않는다.
    k = max(5, int(min(sh, sw) * 0.015)) | 1
    dens = cv2.boxFilter((dark > 0).astype(np.float32), -1, (k, k))
    tex = (dens > 0.10).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(tex, 8)
    tx = sw * 0.01
    dx = sw * EDGE_TEXTURE
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        hit = ((x <= tx and x + bw <= dx) or (x + bw >= sw - tx and x >= sw - dx)) and bh >= 3 * bw
        if hit:
            m[y:y + bh, x:x + bw][lbl[y:y + bh, x:x + bw] == i] = 255
    if not m.any():
        return out
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    if s < 1:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    res = out.copy()
    res[m > 0] = 255
    return res


EDGE_TEXTURE = 0.08
EDGE_FAINT = 245      # 접힌 선 쪽 띠에서 그림자 선으로 볼 밝기 기준(이보다 어두우면)
EDGE_BOX = 0.03       # 가장자리 선이 이보다 넓게(세로선) / 높게(가로선) 이어진 덩어리에 속하면 상자 테두리로 보고 남김
EDGE_BAND = 0.04
EDGE_GUTTER = 0.06


ERASE_LIGHT = 200  # 손가락 주변 띠에서 이보다 밝은 것(남은 옅은 그림자)만 흰색으로. 글자 가장자리 회색은 보존


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
    light = np.clip((lum.astype(np.float32) - ERASE_LIGHT) / 40, 0, 1) * ring
    wgt = np.maximum(soft, light)
    if out.ndim == 3:
        wgt = wgt[..., None]
    return (out * (1 - wgt) + 255 * wgt).astype(np.uint8)


# ====================================================================== 전체 파이프라인
LANDSCAPE_RATIO = 1.2  # 한 쪽 페이지가 이보다 가로로 길면 세로로 눕혀서 곡면 펴기


class BookScanner:
    def __init__(self, model_path, straighten=True, region_model=None):
        self.dw = Dewarper(model_path)
        self.region = BookRegion(region_model) if region_model else None
        self.straighten = straighten

    def process(self, img, layout="auto", mode="color"):
        """layout: auto / single / double"""
        # 0) 책 영역: 책만 잘라내고(멀리서 찍어도 책이 화면을 채우게), 마스크는 마지막 바닥 지우기에 사용
        bmask = None
        if self.region is not None:
            bmask, box = self.region.mask(img)
            if box is not None:
                cb = crop_box(box, img.shape[1], img.shape[0])
                if cb is not None:
                    x0, y0, x1, y1 = cb
                    img, bmask = img[y0:y1, x0:x1], bmask[y0:y1, x0:x1]
        # 폰을 세로로 든 채 펼친 책을 찍으면 글이 옆으로 누워 있음 → 글이 바로 서도록 먼저 돌림
        rot = detect_rotation(img)
        if rot:
            img = rotate_image(img, rot)
            if bmask is not None:
                bmask = rotate_image(bmask, rot)
        gutter = None
        if layout != "single":
            gutter = find_gutter(img, force=(layout == "double"), mask=bmask)
        parts = split_spread(img, gutter, bmask) if gutter is not None else [(img, bmask)]
        pages = []
        for idx, (p, pm) in enumerate(parts):
            pm = hull_mask(pm)
            if pm is not None:
                p, pm = isolate_page(p, pm)
            # 곡면 펴기 모델은 세로로 긴 페이지를 기대한다. 가로로 인쇄된 쪽(표 등)을 돌려 세운 경우처럼
            # 페이지가 가로로 길면 세로로 눕혀서 편 뒤 다시 돌린다.
            land = p.shape[1] > p.shape[0] * LANDSCAPE_RATIO
            if land:
                p = cv2.rotate(p, cv2.ROTATE_90_CLOCKWISE)
                pm = None if pm is None else cv2.rotate(pm, cv2.ROTATE_90_CLOCKWISE)
            flat, info = self.dw.rectify(p, straighten=self.straighten, mask=pm)
            if land:
                flat = cv2.rotate(flat, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # 양면이면 왼쪽 페이지의 오른쪽, 오른쪽 페이지의 왼쪽이 접힌 선
            skip = None if len(parts) == 1 else ("right" if idx == 0 else "left")
            m = finger_mask(flat, skip)
            out = finish(flatten_illumination(fill_paper(flat, m)), mode)
            gside = None if len(parts) == 1 else ("right" if idx == 0 else "left")
            pages.append(Page(clean_edges(erase(out, m), gside), flat, m))
        return pages


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--model", default=os.path.join(here, "..", "android", "BookScan", "app", "src", "main", "assets", "uvdoc.onnx"))
    ap.add_argument("--region-model", default=os.path.join(here, "..", "android", "BookScan", "app", "src", "main", "assets", "u2netp.onnx"))
    ap.add_argument("--out", default="out")
    ap.add_argument("--layout", default="auto")
    ap.add_argument("--mode", default="color")
    ap.add_argument("--no-straighten", action="store_true", help="글줄 반듯하게 펴기 끄기(비교용)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sc = BookScanner(a.model, straighten=not a.no_straighten, region_model=a.region_model or None)
    for path in a.images:
        name = os.path.splitext(os.path.basename(path))[0]
        for i, pg in enumerate(sc.process(cv2.imread(path), a.layout, a.mode)):
            cv2.imwrite(os.path.join(a.out, f"{name}_p{i + 1}.jpg"), pg.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if pg.finger_mask.any():
                cv2.imwrite(os.path.join(a.out, f"{name}_p{i + 1}_mask.png"), pg.finger_mask)
        print(name, "done")
