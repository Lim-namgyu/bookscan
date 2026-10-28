"""
UVDoc(MIT, Tanguy Magne 2023) 체크포인트를 PyTorch 없이 읽어서
모바일용 ONNX 그래프로 변환한다. BatchNorm은 Conv에 접어 넣는다(fold).

입력 : image  float32 [1,3,712,488]  RGB 0~1
출력 : grid2d float32 [1,2,45,31]    grid_sample 좌표(-1~1, x,y 순)

사용법: python3 -I build_uvdoc_onnx.py <best_model.pkl> <out.onnx>
"""
import pickle
import sys
import zipfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# ---------------------------------------------------------------- 체크포인트 로더
_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
}


class _StorageType:
    def __init__(self, name):
        self.dtype = _DTYPES[name]


def _rebuild_tensor_v2(storage, offset, size, stride, *args):
    itemsize = storage.itemsize
    if len(size) == 0:
        return storage[offset].copy()
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
    ).copy()


class _Unpickler(pickle.Unpickler):
    """torch.save 포맷만 해석하는 제한된 언피클러(임의 코드 실행 차단)."""

    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in _DTYPES:
            return lambda: _StorageType(name)
        if module == "torch" and name == "Size":
            return tuple
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"허용되지 않은 클래스: {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, stype, key, _, _ = pid
        dtype = stype.dtype if isinstance(stype, _StorageType) else stype().dtype
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(raw, dtype=dtype)


def load_state_dict(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl.rsplit("/", 1)[0]
    with zf.open(pkl) as f:
        ckpt = _Unpickler(f, zf, prefix).load()
    return ckpt["model_state"]


# ---------------------------------------------------------------- 그래프 빌더
class G:
    def __init__(self, sd):
        self.sd = sd
        self.nodes, self.inits = [], []
        self.n = 0

    def name(self, p):
        self.n += 1
        return f"{p}_{self.n}"

    def const(self, arr, p):
        nm = self.name(p)
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), nm))
        return nm

    def conv_bn(self, x, conv, bn=None, stride=1, pad=0, dil=1, reflect=False):
        w = self.sd[f"{conv}.weight"].astype(np.float32)
        b = self.sd.get(f"{conv}.bias")
        b = np.zeros(w.shape[0], np.float32) if b is None else b.astype(np.float32)
        if bn is not None:
            g, beta = self.sd[f"{bn}.weight"], self.sd[f"{bn}.bias"]
            mean, var = self.sd[f"{bn}.running_mean"], self.sd[f"{bn}.running_var"]
            s = g / np.sqrt(var + 1e-5)
            w = w * s[:, None, None, None]
            b = (b - mean) * s + beta
        if reflect and pad > 0:
            pads = self.const(np.array([0, 0, pad, pad, 0, 0, pad, pad], np.int64), "pads")
            y = self.name("refpad")
            self.nodes.append(helper.make_node("Pad", [x, pads], [y], mode="reflect"))
            x, pad = y, 0
        y = self.name("conv")
        k = w.shape[2]
        self.nodes.append(
            helper.make_node(
                "Conv",
                [x, self.const(w, "w"), self.const(b, "b")],
                [y],
                kernel_shape=[k, k],
                strides=[stride, stride],
                pads=[pad] * 4,
                dilations=[dil, dil],
            )
        )
        return y

    def op(self, t, ins, **kw):
        y = self.name(t.lower())
        self.nodes.append(helper.make_node(t, ins, [y], **kw))
        return y

    def relu(self, x):
        return self.op("Relu", [x])


def build(sd):
    g = G(sd)
    x = "image"
    # resnet_head
    x = g.relu(g.conv_bn(x, "resnet_head.0", "resnet_head.1", stride=2, pad=2))
    x = g.relu(g.conv_bn(x, "resnet_head.3", "resnet_head.4", stride=2, pad=2))
    # resnet_down: layer1~3, 블록 수 [3,4,6], stride [1,2,2]
    for li, (nb, st) in enumerate([(3, 1), (4, 2), (6, 2)], start=1):
        for bi in range(nb):
            p = f"resnet_down.layer{li}.{bi}"
            top = bi == 0
            s = st if top else 1
            pad, dil = (2, 1) if top else (6, 3)
            if f"{p}.downsample.0.weight" in sd:
                res = g.conv_bn(x, f"{p}.downsample.0", f"{p}.downsample.1", stride=s, pad=2)
            else:
                res = x
            c1 = f"{p}.conv1" if top else f"{p}.conv1.0"
            c2 = f"{p}.conv2" if top else f"{p}.conv2.0"
            o = g.relu(g.conv_bn(x, c1, f"{p}.bn1", stride=s, pad=pad, dil=dil))
            o = g.conv_bn(o, c2, f"{p}.bn2", stride=1, pad=pad, dil=dil)
            x = g.relu(g.op("Add", [o, res]))
    down = x
    # bridges
    outs = []
    for name, dils in [
        ("bridge_1", [1]),
        ("bridge_2", [2]),
        ("bridge_3", [5]),
        ("bridge_4", [8, 3, 2]),
        ("bridge_5", [12, 7, 4]),
        ("bridge_6", [18, 12, 6]),
    ]:
        y = down
        for i, d in enumerate(dils):
            y = g.relu(g.conv_bn(y, f"{name}.{i}.0", f"{name}.{i}.1", pad=d, dil=d))
        outs.append(y)
    cat = g.op("Concat", outs, axis=1)
    br = g.relu(g.conv_bn(cat, "bridge_concat.0", "bridge_concat.1"))
    # 2D 출력 헤드 (3D 헤드는 앱에서 불필요하므로 생략)
    h = g.conv_bn(br, "out_point_positions2D.0", "out_point_positions2D.1", pad=2, reflect=True)
    slope = sd["out_point_positions2D.2.weight"].astype(np.float32).reshape(1, -1, 1, 1)
    h = g.op("PRelu", [h, g.const(slope, "slope")])
    h = g.conv_bn(h, "out_point_positions2D.3", None, pad=2, reflect=True)
    g.nodes.append(helper.make_node("Identity", [h], ["grid2d"]))

    graph = helper.make_graph(
        g.nodes,
        "uvdoc",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 712, 488])],
        [helper.make_tensor_value_info("grid2d", TensorProto.FLOAT, [1, 2, 45, 31])],
        g.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    sd = load_state_dict(sys.argv[1])
    m = build(sd)
    onnx.save(m, sys.argv[2])
    print("saved", sys.argv[2], f"{len(m.graph.node)} nodes")
