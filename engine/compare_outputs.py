"""
파이썬 기준 구현과 앱 엔진(Kotlin, JVM 에서 실행)의 결과를 비교한다.
사용: python3 compare_outputs.py PY_DIR KT_DIR [--strict]
  - 페이지 수·크기가 다르면 실패
  - 글자 마스크 겹침(IoU)·흐린 밝기 차이는 표로 출력 (--strict 면 기준 미달도 실패)
"""
import glob
import os
import sys

import cv2
import numpy as np

SIZE_TOL = 0.05     # 크기 차이 허용(비율)
IOU_MIN = 0.70      # 글자 마스크(3px 넓힘) 겹침 최소 (높이 1000 기준)
MAD_MAX = 10.0      # 흐린(σ=2) 회색 밝기 평균 차이 최대 (0~255, 높이 1000 기준)


def text_mask(g):
    m = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    return cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0


def compare(py_path, kt_path):
    a = cv2.imread(py_path, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(kt_path, cv2.IMREAD_GRAYSCALE)
    sa, sb = a.shape, b.shape
    size_ok = abs(sa[0] - sb[0]) <= SIZE_TOL * sa[0] and abs(sa[1] - sb[1]) <= SIZE_TOL * sa[1]
    # 높이 1000 으로 맞춰 비교 (여백·자르기·비율의 작은 차이가 글자 하나 크기보다 작아지도록)
    s = 1000.0 / sa[0]
    a = cv2.resize(a, (int(round(sa[1] * s)), 1000), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    ma, mb = text_mask(a), text_mask(b)
    fa = cv2.GaussianBlur(a, (0, 0), 2).astype(np.float32)
    fb = cv2.GaussianBlur(b, (0, 0), 2).astype(np.float32)
    # 칸마다(4x4) 위상 상관으로 밀린 양을 재서 맞춘 뒤 글자 마스크 겹침을 잰다 (칸마다 다른 밀림 = 비율 차이도 흡수)
    inter = union = 0
    diff = []
    shifts = []
    h, w = a.shape
    for r in range(4):
        for c in range(4):
            y0, y1, x0, x1 = r * h // 4, (r + 1) * h // 4, c * w // 4, (c + 1) * w // 4
            ta, tb = fa[y0:y1, x0:x1], fb[y0:y1, x0:x1]
            (dx, dy), resp = cv2.phaseCorrelate(ta, tb)
            if not (abs(dx) < 0.1 * (x1 - x0) and abs(dy) < 0.1 * (y1 - y0)):
                dx = dy = 0.0
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            pb = cv2.warpAffine(mb[y0:y1, x0:x1].astype(np.uint8), M, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0
            gb = cv2.warpAffine(fb[y0:y1, x0:x1], M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            pa = ma[y0:y1, x0:x1]
            inter += int((pa & pb).sum()); union += int((pa | pb).sum())
            diff.append(np.abs(ta - gb).mean())
            shifts.append(np.hypot(dx, dy))
    iou = inter / max(1, union)
    return size_ok, iou, float(np.mean(diff)), sa, sb, float(np.max(shifts))


def main():
    py_dir, kt_dir = sys.argv[1], sys.argv[2]
    strict = "--strict" in sys.argv
    names = sorted({os.path.basename(p).rsplit("_p", 1)[0] for p in glob.glob(os.path.join(py_dir, "*_p*.jpg"))})
    bad = 0
    print("%-26s %-4s %-12s %-12s %-6s %-6s" % ("사진", "쪽", "파이썬", "앱 엔진", "IoU", "밝기차"))
    for n in names:
        py = sorted(glob.glob(os.path.join(py_dir, n + "_p*.jpg")))
        kt = sorted(glob.glob(os.path.join(kt_dir, n + "_p*.jpg")))
        if len(py) != len(kt):
            print("%-26s 페이지 수 다름: 파이썬 %d, 앱 %d" % (n, len(py), len(kt)))
            bad += 1
            continue
        for i, (p, k) in enumerate(zip(py, kt)):
            size_ok, iou, mad, sa, sb, shift = compare(p, k)
            flag = ""
            if not size_ok:
                flag = "크기 다름"; bad += 1
            elif iou < IOU_MIN or mad > MAD_MAX:
                flag = "주의"
                if strict:
                    bad += 1
            print("%-26s p%-3d %-12s %-12s %.3f  %5.1f  최대밀림 %.1fpx %s" % (n, i + 1, "%dx%d" % (sa[1], sa[0]), "%dx%d" % (sb[1], sb[0]), iou, mad, shift, flag))
    if bad:
        print("%d 건 불일치" % bad)
        sys.exit(1)
    print("일치")


if __name__ == "__main__":
    main()
