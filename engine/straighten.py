"""
글줄 기반 반듯하게 펴기 (곡면 펴기 모델이 남긴 기울기·휨·좌우 쏠림 제거)

  1) 기울기: 글자 덩어리 중심들을 여러 각도로 투영해 글줄이 가장 또렷하게 겹치는 각도를 찾는다
  2) 휨: 기울기를 바로잡은 이미지에서 글줄마다 중심선을 곡선으로 맞추고,
         각 글줄이 수평 직선이 되도록 세로 이동량 지도를 만든다
  3) 좌우 쏠림: 줄 시작(왼쪽 여백)·끝(오른쪽 여백)이 세로 직선이 되도록 가로 보정 (사다리꼴·기울어짐)

결과는 '출력 좌표 → 입력 좌표' 지도(mapx, mapy)로 돌려준다. 이미지를 직접 바꾸지 않으므로
곡면 펴기 지도와 합성해서 원본 사진에서 한 번만 재샘플링할 수 있다(화질 손실 최소).
"""
import cv2
import numpy as np

from textlines import char_height, clean_mask, detect_lines, detect_rules, text_mask


def _centroids(mask):
    n, _, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    return cen[1:], st[1:, cv2.CC_STAT_AREA].astype(np.float64)


def _skew_search(mask, ch, center=0.0, max_deg=30.0):
    """center±max_deg 범위에서 글줄이 가장 또렷하게 겹치는 각도와 그 점수"""
    cen, wt = _centroids(mask)
    if len(cen) < 30:
        return center, 0.0
    x, y = cen[:, 0], cen[:, 1]
    bin_w = max(1.0, ch * 0.5)

    def score(deg):
        t = np.radians(deg)
        p = -x * np.sin(t) + y * np.cos(t)
        idx = np.floor((p - p.min()) / bin_w).astype(np.int64)
        h = np.bincount(idx, weights=wt)
        return float((h * h).sum())

    coarse = np.arange(center - max_deg, center + max_deg + 1e-6, 0.5)
    cs = [score(a) for a in coarse]
    best = coarse[int(np.argmax(cs))]
    fine = np.arange(best - 0.6, best + 0.6 + 1e-6, 0.05)
    fs = [score(a) for a in fine]
    i = int(np.argmax(fs))
    return float(fine[i]), fs[i]


def find_skew(mask, ch, max_deg=30.0):
    """글줄 기울기(도). 양수 = 오른쪽이 아래로 내려간 글줄 (이미지 좌표 y 아래 방향)"""
    return _skew_search(mask, ch, 0.0, max_deg)[0]


def _rot(phi_deg, c):
    """출력(u,v) → 입력(x,y) = c + R(phi)·((u,v) − c) 의 2x3 행렬"""
    t = np.radians(phi_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    off = np.array(c) - R @ np.array(c)
    return np.hstack([R, off[:, None]])


def _fit_line(xs, ys, w, ch):
    span = xs.max() - xs.min()
    deg = 3 if span > w * 0.5 else (2 if span > w * 0.25 else 1)
    keep = np.ones(len(xs), bool)
    for _ in range(2):
        if keep.sum() < deg + 3:
            return None
        coef = np.polyfit(xs[keep], ys[keep], deg)
        res = ys - np.polyval(coef, xs)
        keep = np.abs(res) < max(0.35 * ch, 1.0)
    rms = float(np.sqrt(np.mean(res[keep] ** 2))) if keep.any() else 1e9
    if rms > 0.25 * ch:
        return None
    # 다항식은 이상치 제거용. 실제 곡선은 표본점을 살짝 다듬어 그대로 따라간다(페이지 끝의 급한 휨까지)
    kx, ky = xs[keep], ys[keep]
    if len(ky) >= 3:
        pad = np.pad(ky, 1, mode="edge")
        ky = (pad[:-2] + 2 * pad[1:-1] + pad[2:]) / 4.0
    return coef, kx, ky, float(kx.min()), float(kx.max())


def _line_y(L, x):
    """글줄 곡선의 x 위치 높이. 범위 안은 표본 보간, 바깥은 끝 기울기로 짧게 연장"""
    coef, kx, ky, x0, x1 = L[:5]
    if x < x0:
        slope = np.polyval(np.polyder(coef), x0)
        return ky[0] + slope * (x - x0)
    if x > x1:
        slope = np.polyval(np.polyder(coef), x1)
        return ky[-1] + slope * (x - x1)
    return float(np.interp(x, kx, ky))


def _surface_basis(xn, yn):
    """x 3차 × y 2차 다항식 기저 (12개)"""
    return [xn ** i * yn ** j for j in range(3) for i in range(4)]


def _fit_surface(X, Y, Dv, w, h, ch):
    """세로 이동량의 매끈한 전체 곡면. 범위 밖은 끝값 유지(외삽 폭주 방지)를 위해 맞춘 범위를 함께 돌려준다."""
    box = (X.min(), X.max(), Y.min(), Y.max())
    if len(X) < 24:
        return None
    xn = (X - w / 2) / (w / 2)
    yn = (Y - h / 2) / (h / 2)
    A = np.stack(_surface_basis(xn, yn), axis=1)
    keep = np.ones(len(X), bool)
    coef = None
    for _ in range(3):
        if keep.sum() < 24:
            break
        coef, *_ = np.linalg.lstsq(A[keep], Dv[keep], rcond=None)
        r = Dv - A @ coef
        keep = np.abs(r) < max(1.5 * np.median(np.abs(r[keep])) * 1.4826 * 2, 0.3 * ch)
    return coef, box


def _eval_surface(surf, x, y, w, h):
    if surf is None:
        return np.zeros(np.broadcast(x, y).shape, np.float32)
    coef, (x0, x1, y0, y1) = surf
    xc = np.clip(x, x0, x1)
    yc = np.clip(y, y0, y1)
    xn = (xc - w / 2) / (w / 2)
    yn = (yc - h / 2) / (h / 2)
    xn, yn = np.broadcast_arrays(xn, yn)
    return sum(c * b for c, b in zip(coef, _surface_basis(xn, yn))).astype(np.float32)


def _ransac_margin(T, X, ch, h):
    """
    여백선 x = m(T) 를 강건하게 맞춤. 들여쓰기·가운데 정렬 줄은 이상치로 제외.
    곡면 펴기 뒤의 여백선은 완전한 직선이 아니라 살짝 휜 경우가 많아,
    직선으로 이상치를 거른 뒤 곡선(2~3차)으로 다시 맞춘다.
    (글줄 수가 적어 모든 두 줄 조합을 다 시험한다 → 결과가 항상 같음)
    반환: (coef, Tmin, Tmax) 또는 None
    """
    n = len(T)
    if n < 5:
        return None
    best, best_cnt = None, -1
    tol = 1.2 * ch
    for i in range(n):
        for j in range(i + 1, n):
            if abs(T[i] - T[j]) < h * 0.1:
                continue
            a = (X[j] - X[i]) / (T[j] - T[i])
            b = X[i] - a * T[i]
            inl = np.abs(X - (a * T + b)) < tol
            c = int(inl.sum())
            if c > best_cnt:
                best, best_cnt = inl, c
    if best is None:
        return None
    inl = best
    coef = None
    for _ in range(2):
        k = int(inl.sum())
        if k < 5:
            return None
        deg = 3 if k >= 12 else (2 if k >= 8 else 1)
        coef = np.polyfit(T[inl], X[inl], deg)
        inl = np.abs(X - np.polyval(coef, T)) < 0.6 * ch
    if inl.sum() < max(5, 0.3 * n) or np.ptp(T[inl]) < 0.3 * h:
        return None
    return coef, float(T[inl].min()), float(T[inl].max())


def _margin_at(m, v):
    """여백선 위치. 글줄이 있는 범위 밖은 끝값 유지"""
    coef, t0, t1 = m
    return np.polyval(coef, np.clip(v, t0, t1))


def straighten_maps(img):
    """
    img: 곡면 펴기가 끝난 페이지(BGR/RGB/GRAY, 분석 해상도).
    반환: (mapx, mapy, info) — img 좌표계의 float32 지도. 보정으로 밀려나는 내용이 잘리지 않도록
          img 보다 조금 클 수 있다(info["pad"] = 위, 아래, 왼쪽, 오른쪽 여유).
          글자가 너무 적으면 None.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = text_mask(gray)
    ch = char_height(mask)
    if ch is None:
        return None
    clean = clean_mask(mask, ch)
    a_h, s_h = _skew_search(clean, ch, 0.0)
    a_v, s_v = _skew_search(clean, ch, 90.0, 30.0)
    if s_v > s_h * 1.15:
        # 옆으로 누운 글(가로로 인쇄된 표 등): 페이지는 그대로 두고, 전치한 상태에서 같은 처리
        r = _straighten_core(np.ascontiguousarray(gray.T), np.ascontiguousarray(clean.T), ch, 90.0 - a_v)
        if r is None:
            return None
        mxT, myT, info = r
        info["vertical"] = True
        return np.ascontiguousarray(myT.T), np.ascontiguousarray(mxT.T), info
    return _straighten_core(gray, clean, ch, a_h)


def stem_angle(gray, clean):
    """
    글자 세로 획의 기울기(도). 양수 = 획 위쪽이 왼쪽으로 누움(아래쪽이 오른쪽으로 밀림).
    글자 가장자리 기울기(gradient) 방향 중 '세로 획'에 해당하는 것들의 최빈값.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    near = cv2.dilate(clean, np.ones((3, 3), np.uint8)) > 0
    if near.sum() < 500:
        return 0.0
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag[near], 60)
    sel = near & (mag > thr) & (np.abs(gx) > 1e-3)
    if sel.sum() < 500:
        return 0.0
    ang = np.degrees(np.arctan(gy[sel] / gx[sel]))
    wt = mag[sel]
    est = 0.0
    for win in (15.0, 6.0):
        m = np.abs(ang - est) < win
        nb = int(round(2 * win / 0.1))
        idx = np.clip(((ang[m] - (est - win)) / 0.1).astype(np.int64), 0, nb - 1)
        hist = np.bincount(idx, weights=wt[m], minlength=nb).astype(np.float32)
        hist = cv2.GaussianBlur(hist.reshape(1, -1), (0, 0), sigmaX=4, sigmaY=0.01, borderType=cv2.BORDER_REPLICATE).ravel()
        i = int(np.argmax(hist))
        est = est - win + (i + 0.5) * 0.1
    # 가장자리 기울기 방향각 est 와 획 기울기는 부호가 반대
    return float(-est)


STEM_MAX = 6.0  # 곡면 펴기 뒤 남는 비틀림은 이보다 작다. 더 큰 세로획 기울기는 이탤릭 글꼴로 본다
MARGIN_STEM_TOL = 2.0  # 여백선 기울기와 글자 세로획 각도가 이보다(도) 다르면 덜 기울인 쪽을 믿음



SCALE_MIN_LINES = 8      # 세로 배율 보정에 필요한 최소 글줄 수
SCALE_MIN_GLYPHS = 8     # 글줄마다 재는 글자 수 최소
SCALE_CLAMP = (0.85, 1.18)
SCALE_INLIER_FRAC = 0.5  # 글줄의 이 비율 이상이 추세에 맞아야 (제목·재료 목록 같은 다른 크기 줄은 빠짐)
SCALE_RESID_MAX = 0.03   # 추세를 뺀 글자 높이 흩어짐(log) 최대
SCALE_SPREAD_MIN = 0.5   # 추세에 맞는 줄들이 페이지 높이의 이 비율 이상에 걸쳐 있어야
SCALE_QUART_MIN = 2      # 그 범위를 넷으로 나눴을 때 칸마다 최소 줄 수
DEBUG = None


def _band_height(src, L, ch, offs):
    """
    글줄 하나의 '글자 몸통 띠' 두께(픽셀, 소수). 글줄 중심선을 따라 세로 방향 잉크 분포를 모은 뒤
    최대값의 절반이 되는 위아래 지점 사이 거리를 보간해서 잰다 — 글자 하나하나의 정수 높이와 달리
    줄 전체 평균이라 1px 아래까지 매끈하게 변한다 (라틴 글자는 x-높이, 한글·한자는 글자 높이).
    """
    coef, kx, ky, x0, x1, t = L
    # 글줄 표본은 기울기를 되돌린 좌표라 그림 밖으로 조금 나갈 수 있다 → 그림 안의 열만
    xs = np.arange(max(0, int(np.ceil(x0))), min(src.shape[1] - 1, int(np.floor(x1))) + 1, dtype=np.float32)
    if len(xs) < ch * 4:
        return None
    yc = np.interp(xs, kx, ky)
    yy = yc[None, :] + offs[:, None].astype(np.float64)  # (offs, cols) 표본 행 위치
    y0 = np.floor(yy).astype(np.int64)
    fy = (yy - y0).astype(np.float32)
    hh = src.shape[0]
    xi = xs.astype(np.int64)[None, :]
    v0 = np.where((y0 >= 0) & (y0 < hh), src[np.clip(y0, 0, hh - 1), xi], 0.0)
    v1 = np.where((y0 + 1 >= 0) & (y0 + 1 < hh), src[np.clip(y0 + 1, 0, hh - 1), xi], 0.0)
    prof = (v0 * (1 - fy) + v1 * fy).mean(axis=1)  # 열 방향 평균 → 세로 잉크 분포 (행 사이는 선형 보간)
    pk = int(np.argmax(prof))
    half = prof[pk] * 0.5
    if prof[pk] <= 0:
        return None
    i = pk
    while i > 0 and prof[i - 1] >= half:
        i -= 1
    j = pk
    while j < len(prof) - 1 and prof[j + 1] >= half:
        j += 1
    if i == 0 or j == len(prof) - 1:
        return None
    step = float(offs[1] - offs[0])
    lo = offs[i] - step * (prof[i] - half) / max(prof[i] - prof[i - 1], 1e-6)
    hi = offs[j] + step * (prof[j] - half) / max(prof[j] - prof[j + 1], 1e-6)
    return float(hi - lo)


def _vertical_scale(gray_clean, lines, ch, h):
    """
    곡면 펴기가 덜 된 곳(페이지 끝이 카메라 쪽으로 말린 곳)은 글자가 커진다 → 글줄마다 글자 몸통 띠 두께를 재서
    세로 위치에 따른 배율 r(v) 를 매끈하게(1차, 줄이 많으면 2차) 맞춘다. 반환: r(v) 배열(길이 h) 또는 None.
    좌우 배율은 여백선 보정이 이미 맞추므로 세로만 다룬다.
    """
    n, lbl, st, cen = cv2.connectedComponentsWithStats(gray_clean, 8)
    if n < 2 or len(lines) < SCALE_MIN_LINES:
        return None
    cx, cy = cen[1:, 0], cen[1:, 1]
    hs = st[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    ws = st[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    ok = (hs >= ch * 0.5) & (hs <= ch * 2.0) & (ws <= ch * 3.0)
    src = gray_clean.astype(np.float32)
    offs = np.arange(-1.5 * ch, 1.5 * ch + 1e-3, 0.25)
    T, H = [], []
    for L in lines:
        coef, kx, ky, x0, x1, t = L
        sel = ok & (cx >= x0) & (cx <= x1)
        if sel.sum() < SCALE_MIN_GLYPHS:
            continue
        yl = np.interp(cx[sel], kx, ky)
        if (np.abs(cy[sel] - yl) < ch * 0.6).sum() < SCALE_MIN_GLYPHS:  # 글자가 있는 줄만 (괘선 제외)
            continue
        bh = _band_height(src, L, ch, offs)
        if bh is None or bh < ch * 0.3:
            continue
        T.append(t)
        H.append(bh)
    if len(T) < SCALE_MIN_LINES:
        return None
    T, H = np.array(T), np.array(H)
    y = T / h
    lr = np.log(H / np.median(H))
    deg = 2 if len(T) >= 15 else 1
    keep = np.ones(len(T), bool)
    coef = None
    for _ in range(3):
        if keep.sum() < deg + 4:
            return None
        coef = np.polyfit(y[keep], lr[keep], deg)
        res = lr - np.polyval(coef, y)
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        keep = np.abs(res) < max(2.5 * mad, 0.03)
    # 글자 크기가 섞인 쪽(제목·본문·보기가 다른 크기)에서는 '높이 변화 = 말림' 이라 볼 수 없다:
    # 본문처럼 글자 크기가 고른 쪽(추세를 뺀 흩어짐이 작고 대부분이 추세에 맞음)에서만 보정
    if DEBUG is not None:
        DEBUG.update(lines=len(T), inl=round(float(keep.mean()), 2), resid=round(float(np.std(res[keep])), 3),
                     rng=(round(float(np.exp(np.polyval(coef, y[keep].min()))), 3), round(float(np.exp(np.polyval(coef, y[keep].max()))), 3)),
                     data=[(round(float(a), 2), round(float(b), 2), bool(k)) for a, b, k in zip(y, H, keep)])
    if keep.mean() < SCALE_INLIER_FRAC or np.std(res[keep]) > SCALE_RESID_MAX:
        return None
    # 추세에 맞는 줄이 페이지 전체에 고르게 있어야 한다 (위쪽 큰 글자 묶음 + 아래쪽 작은 글자 묶음 같은
    # '크기가 다른 두 구역'을 말림으로 잘못 읽지 않도록)
    yk = y[keep]
    span = yk.max() - yk.min()
    if span < SCALE_SPREAD_MIN:
        return None
    q = np.minimum(((yk - yk.min()) / span * 4).astype(int), 3)
    if np.bincount(q, minlength=4).min() < SCALE_QUART_MIN:
        return None
    # 글줄이 있는 범위 밖으로는 끝값을 그대로 이어서 과한 외삽을 막는다
    v = np.arange(h, dtype=np.float64) / h
    vc = np.clip(v, y[keep].min(), y[keep].max())
    r = np.exp(np.polyval(coef, vc))
    r = r / np.exp(np.polyval(coef, np.median(y[keep])))  # 가운데(중앙값 위치)를 기준 1로
    if np.abs(np.log(r)).max() < 0.02:  # 2% 미만이면 손대지 않음
        return None
    return np.clip(r, *SCALE_CLAMP)


def _straighten_core(gray, clean, ch, phi):
    """
    페이지를 '회전'하지 않는다. 회전하면, 글줄만 기울고 글자는 똑바로 선 경우(곡면 펴기 잔여 비틀림)에
    글자가 이탤릭처럼 눕기 때문. 대신
      · 글줄 기울기·휨  → 세로 이동(글자 세로획 방향은 그대로)
      · 글자 세로획 기울기 → 가로 이동(여백선이 있으면 여백선 기준, 없으면 세로획 각도 기준)
    로 나눠 보정한다. 회전된 페이지는 두 보정이 합쳐져 결국 회전 보정이 된다.
    phi 는 글줄 추적을 쉽게 하려고 탐지용 이미지에만 쓴다.
    """
    h, w = gray.shape
    c = ((w - 1) / 2.0, (h - 1) / 2.0)
    M = _rot(phi, c)

    # 기울기를 바로잡은 이미지 D 에서 글줄을 찾고, 표본점을 원래 좌표로 되돌린다
    D = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    raw, ch2 = detect_lines(D, min_frac=0.05)
    ch = ch2 or ch
    raw = raw + detect_rules(D, ch)
    lines = []
    for xs, ys in raw:
        xo = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
        yo = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
        order = np.argsort(xo, kind="stable")
        f = _fit_line(xo[order], yo[order], w, ch)
        if f is None:
            continue
        coef, kx, ky, x0, x1 = f
        T = float(np.median(ky))
        lines.append((coef, kx, ky, x0, x1, T))
    lines.sort(key=lambda L: L[5])

    # ---- 세로 이동량 지도 (거친 격자, 칸 = 글자 높이)
    #  d(x, y) = 매끈한 전체 곡면 S(x, y)  +  글줄 근처에서만 살아 있는 잔차 R(x, y)
    #  S 는 글이 없는 여백·위아래까지 자연스럽게 이어 주고, R 은 페이지 끝의 급한 휨 같은 국소 변형을 맞춘다.
    gstep = max(4, int(ch))
    gw = w // gstep + 2
    gh = h // gstep + 2
    dfield = np.zeros((gh, gw), np.float32)
    if len(lines) >= 3:
        X, Y, Dv = [], [], []
        for L in lines:
            x0, x1, T = L[3], L[4], L[5]
            for ci in range(int(np.ceil(x0 / gstep)), int(np.floor(x1 / gstep)) + 1):
                x = ci * gstep
                X.append(x); Y.append(T); Dv.append(_line_y(L, x) - T)
        X, Y, Dv = np.array(X, np.float64), np.array(Y, np.float64), np.array(Dv, np.float64)
        surf = _fit_surface(X, Y, Dv, w, h, ch)
        gx = np.arange(gw) * gstep
        gy = np.arange(gh) * gstep
        S = _eval_surface(surf, gx[None, :], gy[:, None], w, h)
        # 잔차를 격자에 흩뿌린 뒤 정규화 컨볼루션(가중 평균)으로 매끈하게
        res = Dv - _eval_surface(surf, X, Y, w, h)
        num = np.zeros((gh, gw), np.float64)
        den = np.zeros((gh, gw), np.float64)
        cx = np.clip(np.round(X / gstep).astype(int), 0, gw - 1)
        fy = Y / gstep
        y0 = np.clip(np.floor(fy).astype(int), 0, gh - 1)
        y1 = np.clip(y0 + 1, 0, gh - 1)
        wy1 = fy - np.floor(fy)
        np.add.at(num, (y0, cx), res * (1 - wy1)); np.add.at(den, (y0, cx), 1 - wy1)
        np.add.at(num, (y1, cx), res * wy1); np.add.at(den, (y1, cx), wy1)
        num = cv2.GaussianBlur(num.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.0)
        den = cv2.GaussianBlur(den.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.0)
        R = num / (den + 0.02)
        dfield = (S + R).astype(np.float32)

    # ---- 좌우 여백선 보정
    a_l = a_r = None
    if len(lines) >= 5:
        T = np.array([L[5] for L in lines])
        X0 = np.array([L[3] for L in lines])
        X1 = np.array([L[4] for L in lines])
        a_l = _ransac_margin(T, X0, ch, h)
        a_r = _ransac_margin(T, X1, ch, h)

    # ---- 가로 보정: 여백선이 있으면 여백선으로, 없으면 글자 세로획 각도로
    # 여백선 기울기와 세로획 각도가 크게 다르면(표처럼 줄 시작이 들쭉날쭉해 여백선을 잘못 잡았거나,
    # 이탤릭체라 세로획이 원래 누운 경우) 둘 중 덜 기울인 쪽을 믿는다.
    stem = None
    shear = 0.0
    if a_l is not None:
        stem = stem_angle(gray, clean)
        v0, v1 = h * 0.1, h * 0.9
        m_deg = float(np.degrees(np.arctan((_margin_at(a_l, v1) - _margin_at(a_l, v0)) / (v1 - v0))))
        if abs(m_deg - stem) > MARGIN_STEM_TOL and abs(stem) < abs(m_deg):
            a_l = a_r = None
    if a_l is None:
        if stem is None:
            stem = stem_angle(gray, clean)
        # 측정 오차 수준의 작은 값은 무시. 너무 크면 원래 기울어진 글꼴(이탤릭)이라 바로잡지 않음
        if 0.5 <= abs(stem) <= STEM_MAX:
            shear = float(np.tan(np.radians(stem)))
    L0 = float(_margin_at(a_l, h / 2.0)) if a_l is not None else 0.0
    R0 = float(_margin_at(a_r, h / 2.0)) if a_r is not None else 0.0

    def xv_at(u, v):
        u = np.asarray(u, np.float64)
        v = np.asarray(v, np.float64)
        if a_l is not None:
            Lv = _margin_at(a_l, v)
            if a_r is not None:
                ratio = np.clip((_margin_at(a_r, v) - Lv) / max(R0 - L0, 1), 0.9, 1.1)
                return Lv + (u - L0) * ratio
            return u + (Lv - L0)
        return u + shear * (v - c[1])

    def d_at(xq, vq):
        xq = np.asarray(xq, np.float32).reshape(1, -1) / gstep
        vq = np.asarray(vq, np.float32).reshape(1, -1) / gstep
        return cv2.remap(dfield, xq, vq, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()

    # ---- 세로 배율: 글자 높이가 커지는 곳(덜 펴진 말림)은 줄여서 글자 크기를 고르게
    rv = _vertical_scale(clean, lines, ch, h)
    if rv is not None:
        # 입력 행 v 에서 배율 rv[v] → 출력에서는 1/rv 만큼 촘촘히 샘플. g(v) = ∫ 1/rv, 출력 행 → 입력 행은 g 의 역함수
        g_acc = np.concatenate([[0.0], np.cumsum(1.0 / rv)])  # 길이 h+1, g(v) at v=0..h
        h_out = int(round(g_acc[-1]))
        vin_of = lambda vq: np.interp(np.asarray(vq, np.float64), g_acc, np.arange(h + 1, dtype=np.float64))
    else:
        h_out = h
        vin_of = lambda vq: np.asarray(vq, np.float64)

    # ---- 보정으로 화면 밖으로 밀려나는 내용이 없도록 캔버스를 넓힘 (머리글·쪽번호 잘림 방지)
    ur = np.arange(w, dtype=np.float64)
    vr = vin_of(np.arange(h_out, dtype=np.float64))
    top = d_at(xv_at(ur, np.zeros(w)), np.zeros(w))
    bot = d_at(xv_at(ur, np.full(w, h - 1.0)), np.full(w, h - 1.0))
    pad_t = min(int(np.ceil(max(0.0, float(top.max())))), int(h * 0.1))
    pad_b = min(int(np.ceil(max(0.0, -float(bot.min())))), int(h * 0.1))
    pad_l = min(int(np.ceil(max(0.0, float(xv_at(np.zeros(len(vr)), vr).max())))), int(w * 0.1))
    pad_r = min(int(np.ceil(max(0.0, float(((w - 1.0) - xv_at(np.full(len(vr), w - 1.0), vr)).max())))), int(w * 0.1))

    uu, vo = np.meshgrid(np.arange(-pad_l, w + pad_r, dtype=np.float32), np.arange(-pad_t, h_out + pad_b, dtype=np.float32))
    # 출력 행 → 입력 행 (세로 배율 보정; 캔버스 밖 여유 부분은 그대로 이어짐)
    vv = np.where(vo < 0, vo, np.where(vo > h_out - 1, vo - h_out + h, vin_of(np.clip(vo, 0, h_out - 1)))).astype(np.float32)
    xv = xv_at(uu, vv).astype(np.float32)
    # ---- 세로 보정: d(x_v, v) 거친 격자 → 픽셀 단위 보간
    dmap = cv2.remap(dfield, xv / gstep, vv / gstep, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mapx = xv
    mapy = (vv + dmap).astype(np.float32)
    info = dict(skew=phi, lines=len(lines), left=a_l is not None, right=a_r is not None, ch=ch, stem=stem,
                pad=(pad_t, pad_b, pad_l, pad_r), vscale=(float(rv.min()), float(rv.max())) if rv is not None else None,
                dmax=float(np.nanmax(np.abs(dfield))) if dfield.size else 0.0)
    return mapx, mapy, info
