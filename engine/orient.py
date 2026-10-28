"""
사진 방향 자동 맞춤.
  폰을 세로로 든 채 펼친 책을 찍으면 사진 속 글이 옆으로 누워 있다(글줄이 세로).
  1) 글줄이 가로인지 세로인지: 글자 크기·모양의 덩어리 중심점만으로 판단
     (책 옆면 줄무늬처럼 가늘고 긴 조각은 제외)
  2) 위아래 (글줄을 수평으로 돌린 뒤):
     - 마침표·쉼표는 글줄의 아래쪽(밑줄 선)에 붙는다
     - 줄 시작(왼쪽)은 가지런하고 줄 끝(오른쪽)은 들쭉날쭉하다
"""
import cv2
import numpy as np

from straighten import find_skew
from textlines import char_height, clean_mask, detect_lines, text_mask


def _rotate(img, deg_cw):
    if deg_cw == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg_cw == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg_cw == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _char_mask(mask, ch):
    """글자 크기·모양(가늘고 긴 조각 제외)인 덩어리만, 사진 가운데 80% 안"""
    n, lbl, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape
    ws, hs, ar = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_AREA]
    big = np.maximum(ws, hs)
    small = np.maximum(np.minimum(ws, hs), 1)
    ok = (big >= ch * 0.5) & (big <= ch * 2.5) & (big / small <= 4.0) & (small >= ch * 0.25) & (ar >= 4)
    ok &= (cen[:, 0] > w * 0.1) & (cen[:, 0] < w * 0.9) & (cen[:, 1] > h * 0.1) & (cen[:, 1] < h * 0.9)
    ok[0] = False
    return ok[lbl].astype(np.uint8) * 255


def line_likeness(cm, ch):
    """
    글자를 가로로 이어 붙였을 때 생기는 '가로 글줄'의 총 길이(H)와
    세로로 이어 붙였을 때 생기는 '세로 글줄'의 총 길이(V). 페이지가 휘어도 영향이 적다.
    """
    k = max(3, int(ch * 1.2))
    out = []
    for horiz in (True, False):
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3) if horiz else (3, k))  # 기울어진 글줄도 이어지게
        b = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, ker)
        n, _, st, _ = cv2.connectedComponentsWithStats(b, 8)
        ws, hs = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]
        if horiz:
            sel = (ws >= ch * 5) & (hs <= ch * 2.2)
            out.append(float(ws[sel].sum()))
        else:
            sel = (hs >= ch * 5) & (ws <= ch * 2.2)
            out.append(float(hs[sel].sum()))
    return out[0], out[1]


def punct_score(gray):
    """
    글줄 띠 안에서 작은 점(마침표·쉼표)이 아래쪽에 있으면 +, 위쪽이면 −. 반환: (점수 −1~1, 점 개수)
    gray 는 글줄이 가로로 놓인 상태여야 함.
    """
    m = text_mask(gray)
    ch = char_height(m)
    if ch is None:
        return 0.0, 0
    clean = clean_mask(m, ch)
    kx = max(3, int(ch * 1.5))
    blob = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(blob, 8)
    # 글줄 덩어리: 가로로 길고 높이가 글자 높이 정도
    is_line = np.zeros(n, bool)
    for i in range(1, n):
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        is_line[i] = bw >= ch * 4 and ch * 0.6 <= bh <= ch * 1.8
    # 작은 점
    n2, _, st2, cen2 = cv2.connectedComponentsWithStats(m, 8)
    h, w = m.shape
    bottom = top = 0
    r = max(2, int(round(ch * 0.6)))
    for i in range(1, n2):
        bw, bh, ar = st2[i, cv2.CC_STAT_WIDTH], st2[i, cv2.CC_STAT_HEIGHT], st2[i, cv2.CC_STAT_AREA]
        if not (2 <= ar <= ch * ch * 0.12 and max(bw, bh) <= ch * 0.4):
            continue
        cx, cy = int(round(cen2[i, 0])), int(round(cen2[i, 1]))
        # 바로 위·아래에 글자 획이 있는 점(i, j 의 점 등)은 문장부호가 아님
        x0, x1 = st2[i, cv2.CC_STAT_LEFT], st2[i, cv2.CC_STAT_LEFT] + bw
        ya0, ya1 = max(0, int(cy - ch * 0.9)), max(0, int(cy - ch * 0.3))
        yb0, yb1 = min(h, int(cy + ch * 0.3)), min(h, int(cy + ch * 0.9))
        if m[ya0:ya1, x0:x1].any() and m[yb0:yb1, x0:x1].any():
            pass  # 위아래 모두 막힌 점(글자 사이 빽빽한 곳)은 아래 판정에서 걸러짐
        if m[ya0:ya1, x0:x1].any() or m[yb0:yb1, x0:x1].any():
            continue
        # 점 바로 왼쪽·오른쪽(글자 높이 이내)에 있는 글줄 덩어리 찾기
        best = None
        for dx in (-r, r, -2 * r, 2 * r):
            xx = cx + dx
            if 0 <= xx < w:
                j = lbl[cy, xx]
                if j > 0 and is_line[j]:
                    best = j
                    break
        if best is None:
            continue
        ly, lh = st[best, cv2.CC_STAT_TOP], st[best, cv2.CC_STAT_HEIGHT]
        rel = (cy - ly) / max(lh, 1)
        if rel > 0.62:
            bottom += 1
        elif rel < 0.38:
            top += 1
    tot = bottom + top
    return (bottom - top) / max(1, tot), tot


VERT_RATIO = 1.5
FLIP_MIN = 0.5   # 가로 글을 거꾸로(180°) 돌리려면 이만큼 뚜렷한 근거가 있어야 함 (거꾸로 찍는 일은 드묾)


def deskew(g):
    """글줄이 수평이 되도록 돌린 이미지와 기울기(도). 기울어진 채로는 점의 위아래 판정이 틀어진다."""
    m = text_mask(g)
    ch = char_height(m)
    if ch is None:
        return g, 0.0
    sk = find_skew(_char_mask(m, ch), ch, 20.0)
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), sk, 1.0)
    return cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), sk


def align_score(g):
    """
    줄 시작이 줄 끝보다 잘 맞춰져 있으면 +, 반대면 − (−1~1). 가로쓰기는 왼쪽 정렬이고
    번호(①②…)·문단 끝 짧은 줄 때문에 오른쪽 끝은 들쭉날쭉하다. 거꾸로 놓이면 반대가 된다.
    """
    lines, ch = detect_lines(g, 0.04)
    if ch is None or len(lines) < 6:
        return 0.0, 0
    st = np.array([xs.min() for xs, ys in lines])
    en = np.array([xs.max() for xs, ys in lines])
    yc = np.array([np.median(ys) for xs, ys in lines])
    tol = ch * 0.6
    s_ = e_ = 0
    for i in range(len(lines)):
        near = np.abs(yc - yc[i]) < ch * 12  # 같은 단·가까운 줄끼리만
        near[i] = False
        s_ += int((np.abs(st[near] - st[i]) < tol).sum())
        e_ += int((np.abs(en[near] - en[i]) < tol).sum())
    return (s_ - e_) / max(1, s_ + e_), len(lines)


def detect_rotation(img, debug=None):
    """글이 바로 서도록 시계 방향으로 돌릴 각도(0/90/180/270)"""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(1.0, 1600.0 / max(h, w))
    g = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    m = text_mask(g)
    ch = char_height(m)
    if ch is None:
        return 0
    s_h, s_v = line_likeness(_char_mask(m, ch), ch)
    vert = s_v > s_h * VERT_RATIO
    base = _rotate(g, 90) if vert else g
    d, sk = deskew(base)
    # 두 단서의 합: 문장부호가 글줄 아래쪽에 있는가 + 줄 시작이 정렬되어 있는가
    p0, n0 = punct_score(d)
    p1, n1 = punct_score(_rotate(d, 180))
    al, nl = align_score(d)
    vote = al * 2.0
    if n0 + n1 >= 6:
        vote += (p0 - p1) / 2.0
    if debug is not None:
        debug.update(ratio=s_v / max(s_h, 1e-9), vert=vert, skew=round(sk, 1),
                     p=(round(p0, 2), n0, round(p1, 2), n1), align=(round(al, 2), nl), vote=round(vote, 2))
    if vert:
        return 270 if vote < 0 else 90
    return 180 if vote < -FLIP_MIN else 0


def rotate_image(img, deg_cw):
    return _rotate(img, deg_cw)


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        im = cv2.imread(f)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        s = min(1.0, 1600.0 / max(g.shape))
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        d = {}
        r = detect_rotation(im, d)
        print("%-30s rot=%3d  %s" % (f.split("/")[-1], r, d))
