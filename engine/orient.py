"""
사진 방향 자동 맞춤.
  폰을 세로로 든 채 펼친 책을 찍으면 사진 속 글이 옆으로 누워 있다(글줄이 세로).
  1) 글줄이 가로인지 세로인지: 글자 크기·모양의 덩어리 중심점만으로 판단
     (책 옆면 줄무늬처럼 가늘고 긴 조각은 제외)
  2) 위아래: 마침표·쉼표는 글줄의 아래쪽(밑줄 선)에 붙는다 → 아래쪽에 많은 방향이 바로 선 방향
"""
import cv2
import numpy as np

from textlines import char_height, clean_mask, text_mask


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
    cands = (90, 270) if s_v > s_h * VERT_RATIO else (0, 180)
    p0, n0 = punct_score(_rotate(g, cands[0]))
    p1, n1 = punct_score(_rotate(g, cands[1]))
    if debug is not None:
        debug.update(ratio=s_v / max(s_h, 1e-9), cands=cands, p=(round(p0, 2), n0, round(p1, 2), n1))
    if cands[0] == 90:
        # 세로 글: 어느 쪽이든 돌려야 하므로 점 단서가 더 바로 선 쪽
        return 270 if p1 > p0 else 90
    # 가로 글: 거꾸로(180°)는 점 단서가 충분하고 뚜렷할 때만
    if n1 >= 8 and p1 > p0 + 0.3:
        return 180
    return 0


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
        print("%-30s rot=%3d  v/h=%.2f cands=%s punct=%s" % (f.split("/")[-1], r, d.get("ratio", 0), d.get("cands"), d.get("p")))
