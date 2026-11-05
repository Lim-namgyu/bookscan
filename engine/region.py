"""
책 영역 찾기 (U^2-Net-p 두드러진 물체 분할, Apache-2.0, 4.6MB)
  사진에서 책(펼친 페이지 전체)을 바닥·책상·이불 등 배경과 분리한다.
  - 책이 화면에 작게 찍혀도 책 부분만 잘라 다음 단계(방향·양면 분리·곡면 펴기)에 넘김
  - 곡면 편 결과에서 책 바깥(바닥 무늬)을 지우는 데도 쓴다
"""
import cv2
import numpy as np
import onnxruntime as ort

SAL_SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class BookRegion:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def saliency(self, img):
        """0~1 확률 지도 (SAL_SIZE x SAL_SIZE)"""
        # 8비트 그대로 줄인 뒤 실수로 (큰 사진 전체를 실수로 바꾸지 않음 — 앱과 같은 순서)
        small = cv2.resize(img, (SAL_SIZE, SAL_SIZE), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = x / max(float(x.max()), 1e-6)
        x = (x - MEAN) / STD
        o = self.sess.run(None, {self.inp: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        lo, hi = float(o.min()), float(o.max())
        return (o - lo) / max(hi - lo, 1e-6)

    def mask(self, img):
        """
        책 영역 마스크(원본 크기, 0/255)와 경계 상자 (x0, y0, x1, y1). 못 찾으면 (None, None).
        가장 큰 덩어리의 볼록 껍질 → 페이지 안쪽이 그림자·진한 그림 때문에 패여도 메워진다.
        """
        h, w = img.shape[:2]
        p = self.saliency(img)
        b = (p > 0.5).astype(np.uint8)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(b, 8)
        if n < 2:
            return None, None
        areas = st[1:, cv2.CC_STAT_AREA]
        big = float(areas.max())
        if big / float(SAL_SIZE * SAL_SIZE) < 0.08:
            return None, None
        # 가장 큰 덩어리 + 그만큼 큰 덩어리들(접힌 곳이 어두워 양쪽 페이지가 따로 잡히는 경우)
        keep = np.zeros(n, bool)
        keep[1:] = areas >= big * 0.2
        comp = keep[lbl].astype(np.uint8) * 255
        # 구멍 메우기 (진한 그림·그림자 때문에 생긴 안쪽 구멍)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        comp = np.zeros_like(comp)
        cv2.drawContours(comp, cnts, -1, 255, -1)
        if paper_outside(img, comp) > PAPER_OUTSIDE_MAX or text_outside(img, comp) > TEXT_OUTSIDE_MAX:
            # 책 바깥이어야 할 곳이 대부분 종이(밝고 매끈함)거나 글자가 많음 → 페이지가 화면을 꽉 채웠거나
            # 페이지의 일부(한 단)만 잡은 것. 이런 마스크로 자르면 페이지 일부가 잘려 나가므로 쓰지 않는다.
            return None, None
        m = cv2.resize(comp, (w, h), interpolation=cv2.INTER_LINEAR)
        m = np.where(m > 127, 255, 0).astype(np.uint8)
        x, y, bw, bh = cv2.boundingRect(m)
        return m, (x, y, x + bw, y + bh)


PAPER_OUTSIDE_MAX = 0.15  # 마스크 바깥 중 종이처럼 보이는 비율이 이보다 크면 마스크를 믿지 않음


def paper_outside(img, comp):
    """마스크(SAL_SIZE 격자, 0/255) 바깥 영역 중 종이처럼 보이는(밝고 매끈한) 비율. 긴 변 500px 에서 잰다."""
    h, w = img.shape[:2]
    s = 500.0 / max(h, w)
    sw, sh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    g = cv2.cvtColor(cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mm = cv2.resize(comp, (sw, sh), interpolation=cv2.INTER_LINEAR) > 127
    if not mm.any():
        return 0.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    smooth = cv2.erode((gx * gx + gy * gy < 1600.0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    paper = float(np.percentile(g[mm], 75))
    out = cv2.dilate(mm.astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    n_out = int(out.sum())
    if n_out == 0:
        return 0.0
    return float(((g >= 0.85 * paper) & smooth & out).sum()) / n_out


TEXT_OUTSIDE_MAX = 0.5   # 마스크 바깥의 글줄 글자 수가 안쪽의 이 비율보다 많으면 마스크를 믿지 않음 (바닥 천 무늬는 0.2 이하)


def text_outside(img, comp):
    """마스크 바깥에 있는 글자 크기 덩어리 수 / 안쪽 글자 덩어리 수 (긴 변 1000px 에서). 바깥에 글자가 없으면 0"""
    h, w = img.shape[:2]
    s = 1000.0 / max(h, w)
    sw, sh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    g = cv2.cvtColor(cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    mm = cv2.resize(comp, (sw, sh), interpolation=cv2.INTER_LINEAR) > 127
    if not mm.any():
        return 0.0
    paper = float(np.percentile(g[mm], 75))
    ink = (g < paper * 0.55).astype(np.uint8)
    n, lbl, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    if n < 2:
        return 0.0
    hs, ws = st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_WIDTH]
    glyph = (hs >= 4) & (hs <= sh * 0.03) & (ws <= sw * 0.06) & (ws >= 2)
    if glyph.sum() < 30:
        return 0.0
    cxf, cyf = cen[1:, 0][glyph], cen[1:, 1][glyph]
    hg = hs[glyph].astype(np.float64)
    cx = np.clip(cxf.astype(int), 0, sw - 1)
    cy = np.clip(cyf.astype(int), 0, sh - 1)
    inside = cv2.dilate(mm.astype(np.uint8), np.ones((15, 15), np.uint8))[cy, cx] > 0
    # 글자는 줄을 이룬다: 양옆(같은 높이, 글자 높이의 4배 안)에 이웃 덩어리가 있는 것만 글자로 센다
    # (바닥 천의 무늬·잡티는 흩어져 있어 줄을 이루지 못함)
    order = np.argsort(cxf)
    xs_, ys_, hs_ = cxf[order], cyf[order], hg[order]
    inline = np.zeros(len(order), bool)
    for i in range(len(order)):
        tol_y = 0.6 * hs_[i]
        reach = 4.0 * hs_[i]
        j = i - 1
        left = False
        while j >= 0 and xs_[i] - xs_[j] <= reach:
            if abs(ys_[j] - ys_[i]) <= tol_y:
                left = True
                break
            j -= 1
        j = i + 1
        right = False
        while j < len(order) and xs_[j] - xs_[i] <= reach:
            if abs(ys_[j] - ys_[i]) <= tol_y:
                right = True
                break
            j += 1
        inline[i] = left and right
    inl = np.zeros(len(order), bool)
    inl[order] = inline
    n_in = int((inl & inside).sum())
    n_out = int((inl & ~inside).sum())
    if n_out < 30:
        return 0.0
    return n_out / max(n_in, 1)


def hull_mask(m):
    """가장 큰 덩어리의 볼록 껍질(0/255). 한 쪽 페이지는 거의 볼록하므로 가장자리가 패인 곳을 메운다.
    (양면 전체에 쓰면 가운데 접힌 곳 위아래의 바닥까지 덮으므로 페이지마다 따로 쓴다)"""
    if m is None or not m.any():
        return None
    n, lbl, st, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lbl == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(max(cnts, key=cv2.contourArea)), 255)
    return out


def crop_box(box, w, h, margin=0.04):
    """경계 상자를 여유 있게 넓혀 사진 안으로 자른 (x0, y0, x1, y1). 거의 전체면 None"""
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(w, int(np.ceil(x1 + mx))), min(h, int(np.ceil(y1 + my)))
    if (x1 - x0) * (y1 - y0) > 0.92 * w * h:
        return None
    return x0, y0, x1, y1
