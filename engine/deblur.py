"""
흐린 글자 복원 모델(ONNX, 입력 1x1xHxW float 0~1, 출력 같은 크기). 합성 글자 데이터로 학습한 작은 CNN.
tone_map 에서 글자 선명도가 낮은 곳에만 섞어 쓴다. 모델 파일이 없으면 None → 건너뜀.
"""
import os
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

TILE_ROWS = 512
TILE_PAD = 16


class TextDeblur:
    def __init__(self, path):
        if ort is None or not os.path.exists(path):
            raise FileNotFoundError(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2))
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def run(self, g):
        """g: float32 (H, W) 0~255 → float32 (H, W) 0~255. 가로 띠로 나눠 돌린다(메모리)"""
        h, w = g.shape
        out = np.empty_like(g, dtype=np.float32)
        y0 = 0
        while y0 < h:
            y1 = min(h, y0 + TILE_ROWS)
            ya, yb = max(0, y0 - TILE_PAD), min(h, y1 + TILE_PAD)
            x = (g[ya:yb] / 255.0).astype(np.float32)[None, None]
            y = self.sess.run(None, {self.inp: x})[0][0, 0]
            out[y0:y1] = y[y0 - ya:y1 - ya]
            y0 = y1
        return np.clip(out * 255.0, 0, 255)


def load(path):
    try:
        return TextDeblur(path)
    except Exception:
        return None
