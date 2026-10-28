"""
결과 이미지의 품질 수치표. 사용: python3 measure_outputs.py DIR [DIR2 ...]
  글줄 수 / 글줄 기울기(도, 중앙값·90%) / 휨(px@1000, 중앙값) / 줄 간격 고름(위·아래 1/5 의 비, 1 이 이상적)
  글자 가로세로비(위·중·아래) / 종이 밝기 / 글자 진하기
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textlines import measure  # noqa: E402


def glyph_aspect(g, y0, y1):
    h, w = g.shape
    c = g[int(h * y0):int(h * y1), int(w * 0.05):int(w * 0.95)]
    m = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    ws, hs, ar = st[1:, 2].astype(float), st[1:, 3].astype(float), st[1:, 4]
    ok = (hs > 5) & (hs < 60)
    if ok.sum() < 20:
        return float("nan")
    mh = np.median(hs[ok])
    sel = (hs > mh * 0.7) & (hs < mh * 1.3) & (ws > mh * 0.5) & (ws < mh * 1.6) & (ar > 10)
    return float(np.median(ws[sel] / hs[sel])) if sel.sum() >= 10 else float("nan")


def pitch_ratio(g):
    """글줄 간격: 아래쪽 1/5 중앙값 / 위쪽 1/5 중앙값 (덜 펴진 말림이 남으면 1 에서 멀어짐)"""
    h, w = g.shape
    ink = (g[:, int(0.15 * w):int(0.85 * w)] < 128).sum(1).astype(np.float32)
    ink = cv2.GaussianBlur(ink.reshape(-1, 1), (1, 0), 2).ravel()
    on = ink > ink.max() * 0.15
    idx = np.flatnonzero(np.diff(on.astype(int)) == 1) + 1
    ends = np.flatnonzero(np.diff(on.astype(int)) == -1) + 1
    cs = []
    for s in idx:
        e = ends[ends > s]
        if len(e) == 0:
            break
        if e[0] - s > 6:
            cs.append((s + e[0]) / 2)
    cs = np.array(cs)
    if len(cs) < 8:
        return float("nan")
    p = np.diff(cs)
    cc = cs[:-1]
    med = np.median(p)
    m = (p > med * 0.7) & (p < med * 1.3)
    top = p[m & (cc < h / 5)]
    bot = p[m & (cc >= 4 * h / 5)]
    if len(top) < 2 or len(bot) < 2:
        return float("nan")
    return float(np.median(bot) / np.median(top))


def main():
    print("%-28s %4s %6s %6s %6s %6s  %-18s %5s %5s" % ("파일", "줄", "기울기", "기울90", "휨", "간격비", "가로세로비(상/중/하)", "종이", "글자"))
    for d in sys.argv[1:]:
        for p in sorted(glob.glob(os.path.join(d, "*_p*.jpg"))):
            im = cv2.imread(p)
            g = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            s = 1600.0 / max(g.shape)
            gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
            m = measure(gs)
            asp = [glyph_aspect(gs, a, b) for a, b in [(0.05, 0.35), (0.35, 0.65), (0.65, 0.95)]]
            pr = pitch_ratio(gs)
            paper = float(np.percentile(gs, 60))
            tm = cv2.adaptiveThreshold(gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15) > 0
            ink = float(np.percentile(gs[tm], 20)) if tm.sum() > 100 else float("nan")
            print("%-28s %4d %6.2f %6.2f %6.1f %6.3f  %5.3f/%5.3f/%5.3f   %5.0f %5.0f" % (
                os.path.relpath(p, d)[:28], m["n"], m["angle"], m.get("angle90", float("nan")), m["bend"], pr, asp[0], asp[1], asp[2], paper, ink))


if __name__ == "__main__":
    main()
