"""
글줄 찾기 + 반듯함 측정.
  detect_lines(gray) → 글줄마다 중심선 표본점(x, y)
  measure(img)       → 결과 이미지의 글줄 기울기(도)·휨(px, 폭 1000px 기준) 통계
"""
import cv2
import numpy as np


def text_mask(gray):
    """글자 픽셀=255. 조명 얼룩에 강하도록 적응형 이진화."""
    h, w = gray.shape
    block = max(15, int(min(h, w) * 0.03)) | 1
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 15)
    return m


def char_height(mask):
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    hs = st[1:, cv2.CC_STAT_HEIGHT]
    ws = st[1:, cv2.CC_STAT_WIDTH]
    ok = (hs >= 4) & (hs < mask.shape[0] * 0.05) & (ws < mask.shape[1] * 0.1)
    if ok.sum() < 30:
        return None
    return float(np.median(hs[ok]))


def clean_mask(mask, ch):
    """글자 크기와 동떨어진 덩어리(그림, 테두리, 얼룩, 점 잡음) 제거"""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    hs, ws, ar = st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_AREA]
    keep = (hs <= ch * 2.5) & (ws <= ch * 6) & (ar >= max(4, ch * ch * 0.04))
    keep[0] = False
    return (keep[lbl]).astype(np.uint8) * 255


def detect_lines(gray, min_frac=0.12):
    """
    gray: 글자가 대략 가로로 놓인 8비트 이미지.
    반환: (lines, ch). lines 의 원소는 (xs, ys) 표본 배열 — 글줄 중심선
    """
    mask = text_mask(gray)
    ch = char_height(mask)
    if ch is None:
        return [], None
    mask = clean_mask(mask, ch)
    h, w = mask.shape
    # 글자를 가로로 이어 붙여 글줄 덩어리로
    kx = max(3, int(ch * 2.2))
    lines_blob = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    lines_blob = cv2.morphologyEx(lines_blob, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(ch)), 1)))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(lines_blob, 8)
    out = []
    step = max(2, int(ch))
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if bw < w * min_frac or bw < ch * 8:
            continue
        if bh > ch * 2.6 + bw * 0.06:  # 여러 줄이 엉긴 덩어리 (기울어진 줄은 bh 가 커질 수 있어 폭 비례 여유)
            continue
        sub = (lbl[y:y + bh, x:x + bw] == i) & (mask[y:y + bh, x:x + bw] > 0)
        cols = sub.any(axis=0)
        if cols.sum() < bw * 0.5:
            continue
        yy = np.arange(bh, dtype=np.float32)[:, None]

        def centers(band=None):
            xs, ys = [], []
            for c0 in range(0, bw, step):
                blk = sub[:, c0:c0 + step]
                xc = c0 + blk.shape[1] / 2.0
                if band is not None:
                    # 글줄 띠 안의 픽셀만: g, p, y 꼬리나 위 첨자·쉼표가 중심을 끌어내리지 않게
                    yc = np.polyval(band, xc)
                    blk = blk & (np.abs(yy - yc) < ch * 0.55)
                cnt = blk.sum()
                if cnt < ch * 0.5:
                    continue
                xs.append(x + xc)
                ys.append(y + float((blk * yy).sum() / cnt))
            return np.array(xs), np.array(ys)

        xs, ys = centers()
        if len(xs) < 6:
            continue
        deg = min(3, max(1, len(xs) // 8))
        band = np.polyfit(xs - x, ys - y, deg)
        xs, ys = centers(band)
        if len(xs) < 6:
            continue
        # 이웃 중앙값으로 다듬기 (끝부분은 창을 줄여서, 끝 글자 하나가 지배하지 않게)
        sm = np.empty_like(ys)
        for j in range(len(ys)):
            r = min(2, j, len(ys) - 1 - j)
            sm[j] = np.median(ys[j - r:j + r + 1])
        out.append((xs, sm))
    return out, ch


def _centers_of(sub, x, y, bw, bh, step):
    """부분 마스크(sub)의 열 묶음마다 세로 중심 → (xs, ys)"""
    yy = np.arange(bh, dtype=np.float32)[:, None]
    xs, ys = [], []
    for c0 in range(0, bw, step):
        blk = sub[:, c0:c0 + step]
        cnt = blk.sum()
        if cnt < 1:
            continue
        xs.append(x + c0 + blk.shape[1] / 2.0)
        ys.append(y + float((blk * yy).sum() / cnt))
    return np.array(xs), np.array(ys)


def detect_rules(gray, ch, min_frac=0.25):
    """
    가로 괘선(밑줄·표 테두리·꼬리말 선·머리글 장식선)의 중심선. 긴 직선이라 글줄보다 더 믿을 만한 기준이 된다.
      1) 가늘고 긴 덩어리: 휘어 있어도(페이지 모서리의 곡면 잔재) 그대로 따라감
      2) 열림 연산으로 남는 곧은 선: 글자와 붙은 밑줄처럼 덩어리로는 두꺼워 보이는 경우
    반환: [(xs, ys)]
    """
    mask = text_mask(gray)
    h, w = mask.shape
    step = max(2, int(ch))
    out = []
    boxes = []
    # 1) 가늘고 긴 덩어리
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if bw < w * min_frac or bh > bw * 0.2:
            continue
        if area / max(bw, 1) > ch * 0.6:
            continue
        sub = lbl[y:y + bh, x:x + bw] == i
        # 열마다 두께가 얇아야(글자가 이어 붙은 줄은 제외)
        thick = sub.sum(axis=0)
        if np.median(thick[thick > 0]) > ch * 0.4:
            continue
        xs, ys = _centers_of(sub, x, y, bw, bh, step)
        if len(xs) >= 6:
            out.append((xs, ys))
            boxes.append((x, y, bw, bh))
    # 2) 곧은 선 (열림 연산)
    k = max(9, int(ch * 4)) | 1
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(horiz, 8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if bw < w * min_frac:
            continue
        thick = area / max(bw, 1)
        if thick > ch * 0.9 or bh > bw * 0.2:
            continue
        # 1)에서 이미 잡은 선과 겹치면 건너뜀
        dup = False
        for (bx, by, bbw, bbh) in boxes:
            ox = min(x + bw, bx + bbw) - max(x, bx)
            if ox > 0.8 * bw and y + bh >= by - ch and y <= by + bbh + ch:
                dup = True
                break
        if dup:
            continue
        sub = lbl[y:y + bh, x:x + bw] == i
        xs, ys = _centers_of(sub, x, y, bw, bh, step)
        if len(xs) >= 6:
            out.append((xs, ys))
    return out


def measure(img, draw=None):
    """결과 이미지의 반듯함. 반환 dict: n(글줄 수), angle(중앙값 |기울기| 도), bend(휨 중앙값 px@1000), bend90"""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 1600.0 / max(gray.shape)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lines, ch = detect_lines(g, min_frac=0.3)
    norm = 1000.0 / g.shape[1]
    angs, bends = [], []
    for xs, ys in lines:
        a, b = np.polyfit(xs, ys, 1)
        angs.append(abs(np.degrees(np.arctan(a))))
        res = ys - (a * xs + b)
        bends.append((np.percentile(res, 95) - np.percentile(res, 5)) * norm)
        if draw is not None:
            for x0, y0 in zip(xs, ys):
                cv2.circle(draw, (int(x0 / s), int(y0 / s)), 3, (0, 0, 255), -1)
    if not angs:
        return dict(n=0, angle=float("nan"), bend=float("nan"), bend90=float("nan"))
    return dict(n=len(angs), angle=float(np.median(angs)), angle90=float(np.percentile(angs, 90)),
                bend=float(np.median(bends)), bend90=float(np.percentile(bends, 90)))
