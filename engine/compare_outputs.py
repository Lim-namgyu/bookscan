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
IOU_MIN = 0.70      # 글자 마스크(3px 넓힘) 겹침 최소
MAD_MAX = 10.0      # 흐린(σ=3) 회색 밝기 평균 차이 최대 (0~255)


def text_mask(g):
    m = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    return cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0


def compare(py_path, kt_path):
    a = cv2.imread(py_path, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(kt_path, cv2.IMREAD_GRAYSCALE)
    size_ok = abs(a.shape[0] - b.shape[0]) <= SIZE_TOL * a.shape[0] and abs(a.shape[1] - b.shape[1]) <= SIZE_TOL * a.shape[1]
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    ma, mb = text_mask(a), text_mask(b)
    iou = float((ma & mb).sum()) / max(1, (ma | mb).sum())
    mad = float(np.abs(cv2.GaussianBlur(a, (0, 0), 3).astype(np.float32) - cv2.GaussianBlur(b, (0, 0), 3).astype(np.float32)).mean())
    return size_ok, iou, mad, a.shape, b.shape


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
            size_ok, iou, mad, sa, sb = compare(p, k)
            flag = ""
            if not size_ok:
                flag = "크기 다름"; bad += 1
            elif iou < IOU_MIN or mad > MAD_MAX:
                flag = "주의"
                if strict:
                    bad += 1
            print("%-26s p%-3d %-12s %-12s %.3f  %5.1f  %s" % (n, i + 1, "%dx%d" % (sa[1], sa[0]), "%dx%d" % (sb[1], sb[0]), iou, mad, flag))
    if bad:
        print("%d 건 불일치" % bad)
        sys.exit(1)
    print("일치")


if __name__ == "__main__":
    main()
