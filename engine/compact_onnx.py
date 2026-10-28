"""
가중치를 float16으로 저장해 모델 용량을 절반으로 줄인다.
연산은 float32 그대로(로드 시 Cast가 상수 접힘) → 정확도·호환성 유지.
사용법: python3 -I compact_onnx.py uvdoc.onnx uvdoc_fp16w.onnx
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

m = onnx.load(sys.argv[1])
g = m.graph
new_inits, casts = [], []
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    if arr.dtype == np.float32 and arr.size > 256:
        half_name = init.name + "_fp16"
        new_inits.append(numpy_helper.from_array(arr.astype(np.float16), half_name))
        casts.append(helper.make_node("Cast", [half_name], [init.name], to=TensorProto.FLOAT))
    else:
        new_inits.append(init)
del g.initializer[:]
g.initializer.extend(new_inits)
nodes = list(g.node)
del g.node[:]
g.node.extend(casts + nodes)
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
print("saved", sys.argv[2])
