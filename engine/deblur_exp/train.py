#!/usr/bin/env python3
"""Train a tiny residual text-deblur CNN and export it to ONNX.

Architecture (gray, 1 channel, values in 0..1):
    conv3x3(1->16)+ReLU, 4 x [conv3x3(16->16)+ReLU], conv3x3(16->1)
    out = clip(x + net(x), 0, 1)                         (9,585 parameters)

Backends
    * torch  : used automatically when `import torch` works (set_num_threads(2)).
    * numpy  : exact same model/loss/optimiser written with im2col + manual
               backprop. Used when torch is not importable (this environment:
               download.pytorch.org is blocked by the egress proxy and the PyPI
               wheel is the CUDA build, several GB of deps).
Both backends write the same weights.npz (OIHW layout) and the ONNX file is
built from that with onnx.helper (opset 17, input "img" 1x1xHxW, dynamic H/W).

Usage:  python3 train.py [train_minutes] [data_tag]      (default 30, "")
"""
import os, sys, time, math
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
TAG = sys.argv[2] if len(sys.argv) > 2 else ""      # e.g. "_v2": uses train_v2_*.npy, writes weights_v2.npz / deblur_v2.onnx
BATCH = 32
LR = 1e-3
CH = 16
NLAYERS_MID = 4
SEED = 0
OFFS = [(dy, dx) for dy in range(3) for dx in range(3)]
DT = np.float32   # numpy-backend compute dtype (float64 only for gradient checks)

try:
    import torch
    torch.set_num_threads(2)
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

# ---------------------------------------------------------------- data
def load():
    Xtr = np.load(os.path.join(HERE, f"train{TAG}_x.npy")).astype(np.float32) / 255.0
    Ytr = np.load(os.path.join(HERE, f"train{TAG}_y.npy")).astype(np.float32) / 255.0
    Xva = np.load(os.path.join(HERE, f"val{TAG}_x.npy")).astype(np.float32) / 255.0
    Yva = np.load(os.path.join(HERE, f"val{TAG}_y.npy")).astype(np.float32) / 255.0
    return Xtr, Ytr, Xva, Yva

def augment(x, y, rng):
    """Random flips / transpose (same op on both), cheap and keeps blur statistics."""
    k = rng.integers(0, 8)
    if k & 1: x, y = x[:, :, ::-1], y[:, :, ::-1]
    if k & 2: x, y = x[:, ::-1, :], y[:, ::-1, :]
    if k & 4: x, y = x.transpose(0, 2, 1), y.transpose(0, 2, 1)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)

# ---------------------------------------------------------------- numpy backend
def kaiming_uniform(shape, fan_in, rng):
    b = 1.0 / math.sqrt(fan_in)
    return rng.uniform(-b, b, size=shape).astype(DT)

class NumpyNet:
    """Conv weights stored as Wm[(ky*3+kx)*Cin + cin, cout] (im2col layout); activations NHWC."""
    def __init__(self, rng):
        chans = [1] + [CH] * (NLAYERS_MID + 1) + [1]
        self.W, self.b = [], []
        for i in range(len(chans) - 1):
            cin, cout = chans[i], chans[i + 1]
            self.W.append(kaiming_uniform((9 * cin, cout), 9 * cin, rng))
            self.b.append(kaiming_uniform((cout,), 9 * cin, rng))
        self.W[-1][:] = 0.0; self.b[-1][:] = 0.0   # start as identity mapping
        self.nconv = len(self.W)

    def params(self):
        return self.W + self.b

    def n_params(self):
        return sum(p.size for p in self.params())

    # 3x3 'same' convolution as 9 shifted matmuls on the zero-padded NHWC input
    # (no im2col: the strided 5-D copies were the bottleneck on this CPU).
    def conv(self, xp, i):
        B, Hp, Wp, cin = xp.shape; H, W = Hp - 2, Wp - 2
        out = np.empty((B, H, W, self.W[i].shape[1]), DT); out[:] = self.b[i]
        for k, (dy, dx) in enumerate(OFFS):
            out += xp[:, dy:dy + H, dx:dx + W, :] @ self.W[i][k * cin:(k + 1) * cin]
        return out

    def forward(self, x, train=False):
        """x: (B,H,W) in 0..1 -> y (B,H,W). Caches for backward when train=True."""
        B, H, W = x.shape
        a = x[..., None].astype(DT, copy=False)
        cache = []
        for i in range(self.nconv):
            xp = np.pad(a, ((0, 0), (1, 1), (1, 1), (0, 0)))
            pre = self.conv(xp, i)
            if i < self.nconv - 1:
                a = np.maximum(pre, 0.0)
                cache.append((xp, pre > 0))
            else:
                a = pre
                cache.append((xp, None))
        s = x + a[..., 0]
        y = np.clip(s, 0.0, 1.0)
        if train:
            self._cache = cache
            self._clipmask = (s > 0.0) & (s < 1.0)
            self._shape = (B, H, W)
        return y

    def backward(self, dy):
        """dy: dL/dy (B,H,W). Returns list of grads matching params()."""
        B, H, W = self._shape
        d = (dy * self._clipmask)[..., None].astype(DT, copy=False)   # dL/d(residual)
        gW, gb = [None] * self.nconv, [None] * self.nconv
        for i in reversed(range(self.nconv)):
            xp, relumask = self._cache[i]
            if relumask is not None:
                d = d * relumask
            cin = xp.shape[-1]; cout = d.shape[-1]
            dflat = d.reshape(B * H * W, cout)
            g = np.empty_like(self.W[i])
            for k, (dy_, dx_) in enumerate(OFFS):
                xk = np.ascontiguousarray(xp[:, dy_:dy_ + H, dx_:dx_ + W, :]).reshape(B * H * W, cin)
                g[k * cin:(k + 1) * cin] = xk.T @ dflat
            gW[i] = g
            gb[i] = dflat.sum(0)
            if i > 0:
                dxp = np.zeros_like(xp)
                for k, (dy_, dx_) in enumerate(OFFS):
                    dxp[:, dy_:dy_ + H, dx_:dx_ + W, :] += (dflat @ self.W[i][k * cin:(k + 1) * cin].T).reshape(B, H, W, cin)
                d = dxp[:, 1:-1, 1:-1, :]
        self._cache = None
        return gW + gb

    def to_oihw(self):
        """Return list of (W[o,i,ky,kx], b[o]) for export."""
        out = []
        for Wm, b in zip(self.W, self.b):
            cout = Wm.shape[1]; cin = Wm.shape[0] // 9
            out.append((Wm.reshape(3, 3, cin, cout).transpose(3, 2, 0, 1).copy(), b.copy()))
        return out

class Adam:
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.p = params; self.lr = lr; self.b1, self.b2 = betas; self.eps = eps
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]; self.t = 0
    def step(self, grads, lr):
        self.t += 1
        c1 = 1 - self.b1 ** self.t; c2 = 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * (g * g)
            p -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)

def l1(a, b):
    return float(np.abs(a - b).mean())

def eval_numpy(net, X, Y, bs=50):
    tot = 0.0
    for i in range(0, len(X), bs):
        tot += np.abs(net.forward(X[i:i + bs]) - Y[i:i + bs]).sum()
    return tot / Y.size

def train_numpy(Xtr, Ytr, Xva, Yva):
    rng = np.random.default_rng(SEED)
    net = NumpyNet(rng)
    opt = Adam(net.params(), LR)
    print(f"[numpy backend] params = {net.n_params()}")
    n = len(Xtr); steps_per_epoch = n // BATCH
    base_val = l1(Xva, Yva)
    print(f"val L1 identity baseline = {base_val:.5f}   model (init, identity) = {eval_numpy(net, Xva, Yva):.5f}")

    # time a few steps to size the run to the budget
    t0 = time.time(); warm = 6
    perm = rng.permutation(n)
    for s in range(warm):
        idx = perm[s * BATCH:(s + 1) * BATCH]
        xb, yb = augment(Xtr[idx], Ytr[idx], rng)
        yhat = net.forward(xb, train=True)
        grads = net.backward(np.sign(yhat - yb) / yhat.size)
        opt.step(grads, LR)
    step_time = (time.time() - t0) / warm
    budget = TRAIN_MIN * 60.0
    epochs = max(1, int(budget / (step_time * steps_per_epoch * 1.08)))   # ~8% headroom for eval/overhead
    total_steps = epochs * steps_per_epoch
    print(f"step time ~{step_time * 1000:.0f} ms  -> {epochs} epochs x {steps_per_epoch} steps = {total_steps} steps (budget {TRAIN_MIN:.0f} min)")

    best = (1e9, None); step = 0; tstart = time.time(); hist = []
    for ep in range(epochs):
        perm = rng.permutation(n); run = 0.0; te = time.time()
        for s in range(steps_per_epoch):
            lr = 0.5 * LR * (1 + math.cos(math.pi * step / total_steps))
            idx = perm[s * BATCH:(s + 1) * BATCH]
            xb, yb = augment(Xtr[idx], Ytr[idx], rng)
            yhat = net.forward(xb, train=True)
            run += l1(yhat, yb)
            grads = net.backward(np.sign(yhat - yb) / yhat.size)
            opt.step(grads, lr); step += 1
        v = eval_numpy(net, Xva, Yva)
        hist.append((ep + 1, run / steps_per_epoch, v))
        if v < best[0]:
            best = (v, [p.copy() for p in net.params()])
        print(f"epoch {ep + 1:3d}/{epochs}  train L1 {run / steps_per_epoch:.5f}  val L1 {v:.5f}  lr {lr:.2e}  ({time.time() - te:.1f}s, total {(time.time() - tstart) / 60:.1f} min)")
    # restore best
    for p, q in zip(net.params(), best[1]):
        p[:] = q
    ttrain = time.time() - tstart
    v = eval_numpy(net, Xva, Yva)
    print(f"training time {ttrain / 60:.1f} min, best val L1 = {v:.5f} (identity {base_val:.5f}, {100 * (1 - v / base_val):.1f}% lower)")
    np.savez(os.path.join(HERE, f"weights{TAG}.npz"), **{f"w{i}": w for i, (w, b) in enumerate(net.to_oihw())},
             **{f"b{i}": b for i, (w, b) in enumerate(net.to_oihw())})
    with open(os.path.join(HERE, f"train_log{TAG}.txt"), "w") as f:
        f.write(f"backend numpy; params {net.n_params()}; epochs {epochs}; steps {total_steps}; batch {BATCH}; "
                f"train_time_min {ttrain / 60:.2f}; step_ms {step_time * 1000:.0f}\n")
        f.write(f"val_L1_identity {base_val:.6f}\nval_L1_model {v:.6f}\n")
        for e, tr, va in hist:
            f.write(f"{e}\t{tr:.6f}\t{va:.6f}\n")
    ref = lambda x: net.forward(x)
    return ref, net.n_params()

# ---------------------------------------------------------------- torch backend (used only when torch imports)
def train_torch(Xtr, Ytr, Xva, Yva):
    import torch, torch.nn as nn
    torch.manual_seed(SEED)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Conv2d(1, CH, 3, padding=1), nn.ReLU()]
            for _ in range(NLAYERS_MID):
                layers += [nn.Conv2d(CH, CH, 3, padding=1), nn.ReLU()]
            layers += [nn.Conv2d(CH, 1, 3, padding=1)]
            self.net = nn.Sequential(*layers)
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        def forward(self, x):
            return torch.clamp(x + self.net(x), 0.0, 1.0)
    net = Net(); nparams = sum(p.numel() for p in net.parameters())
    print(f"[torch backend] params = {nparams}")
    xtr = torch.from_numpy(Xtr)[:, None]; ytr = torch.from_numpy(Ytr)[:, None]
    xva = torch.from_numpy(Xva)[:, None]; yva = torch.from_numpy(Yva)[:, None]
    base_val = (xva - yva).abs().mean().item()
    def evalv():
        net.eval()
        with torch.no_grad():
            v = (net(xva) - yva).abs().mean().item()
        net.train(); return v
    print(f"val L1 identity baseline = {base_val:.5f}   model (init) = {evalv():.5f}")
    n = len(xtr); spe = n // BATCH
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    # time a few steps
    t0 = time.time()
    for s in range(6):
        idx = torch.from_numpy(rng.permutation(n)[:BATCH])
        loss = (net(xtr[idx]) - ytr[idx]).abs().mean(); opt.zero_grad(); loss.backward(); opt.step()
    step_time = (time.time() - t0) / 6
    epochs = max(1, int(TRAIN_MIN * 60 / (step_time * spe * 1.08))); total = epochs * spe
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    print(f"step time ~{step_time * 1000:.0f} ms -> {epochs} epochs ({total} steps)")
    best = (1e9, None); tstart = time.time(); hist = []
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)); run = 0.0; te = time.time()
        for s in range(spe):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            xb, yb = xtr[idx], ytr[idx]
            k = int(rng.integers(0, 8))
            if k & 1: xb, yb = xb.flip(-1), yb.flip(-1)
            if k & 2: xb, yb = xb.flip(-2), yb.flip(-2)
            if k & 4: xb, yb = xb.transpose(-1, -2), yb.transpose(-1, -2)
            loss = (net(xb) - yb).abs().mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += loss.item()
        v = evalv(); hist.append((ep + 1, run / spe, v))
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in net.state_dict().items()})
        print(f"epoch {ep + 1:3d}/{epochs}  train L1 {run / spe:.5f}  val L1 {v:.5f}  ({time.time() - te:.1f}s)")
    net.load_state_dict(best[1]); ttrain = time.time() - tstart; v = evalv()
    print(f"training time {ttrain / 60:.1f} min, best val L1 = {v:.5f} (identity {base_val:.5f})")
    convs = [m for m in net.net if isinstance(m, nn.Conv2d)]
    np.savez(os.path.join(HERE, f"weights{TAG}.npz"), **{f"w{i}": c.weight.detach().numpy() for i, c in enumerate(convs)},
             **{f"b{i}": c.bias.detach().numpy() for i, c in enumerate(convs)})
    with open(os.path.join(HERE, f"train_log{TAG}.txt"), "w") as f:
        f.write(f"backend torch; params {nparams}; epochs {epochs}; steps {total}; batch {BATCH}; train_time_min {ttrain / 60:.2f}; step_ms {step_time * 1000:.0f}\n")
        f.write(f"val_L1_identity {base_val:.6f}\nval_L1_model {v:.6f}\n")
        for e, tr, va in hist: f.write(f"{e}\t{tr:.6f}\t{va:.6f}\n")
    def ref(x):
        with torch.no_grad():
            return net(torch.from_numpy(x)[:, None])[:, 0].numpy()
    return ref, nparams

# ---------------------------------------------------------------- ONNX export (shared)
def export_onnx(path):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    w = np.load(os.path.join(HERE, f"weights{TAG}.npz"))
    nconv = len([k for k in w.files if k.startswith("w")])
    inits, nodes = [], []
    prev = "img"
    for i in range(nconv):
        inits.append(numpy_helper.from_array(w[f"w{i}"].astype(np.float32), f"conv{i}_w"))
        inits.append(numpy_helper.from_array(w[f"b{i}"].astype(np.float32), f"conv{i}_b"))
        out = f"conv{i}"
        nodes.append(helper.make_node("Conv", [prev, f"conv{i}_w", f"conv{i}_b"], [out],
                                      kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], name=f"Conv{i}"))
        if i < nconv - 1:
            nodes.append(helper.make_node("Relu", [out], [f"relu{i}"], name=f"Relu{i}")); prev = f"relu{i}"
        else:
            prev = out
    inits.append(numpy_helper.from_array(np.array(0.0, np.float32), "zero"))
    inits.append(numpy_helper.from_array(np.array(1.0, np.float32), "one"))
    nodes.append(helper.make_node("Add", ["img", prev], ["sum"], name="Residual"))
    nodes.append(helper.make_node("Clip", ["sum", "zero", "one"], ["out"], name="Clip"))
    graph = helper.make_graph(nodes, "tiny_text_deblur",
                              [helper.make_tensor_value_info("img", TensorProto.FLOAT, [1, 1, "H", "W"])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, "H", "W"])], inits)
    model = helper.make_model(graph, producer_name="deblur-prototype", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return model

def verify_onnx(path, ref, Xva):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    worst = 0.0
    for x in [Xva[0], Xva[7], Xva[42]]:
        o = sess.run(None, {"img": x[None, None]})[0][0, 0]
        worst = max(worst, float(np.abs(o - ref(x[None])[0]).max()))
    # odd, non-square size to exercise dynamic H/W
    rng = np.random.default_rng(1)
    x = np.clip(rng.normal(0.85, 0.15, (1, 1, 101, 183)).astype(np.float32), 0, 1)
    o = sess.run(None, {"img": x})[0]
    worst = max(worst, float(np.abs(o - ref(x[:, 0])[:, None]).max()))
    print(f"ONNX vs reference forward: max |diff| = {worst:.2e}  (shape check {o.shape})")
    return worst

if __name__ == "__main__":
    Xtr, Ytr, Xva, Yva = load()
    print(f"train {Xtr.shape}  val {Xva.shape}  backend = {'torch' if HAVE_TORCH else 'numpy'}")
    ref, nparams = (train_torch if HAVE_TORCH else train_numpy)(Xtr, Ytr, Xva, Yva)
    onnx_path = os.path.join(HERE, f"deblur{TAG}.onnx")
    export_onnx(onnx_path)
    d = verify_onnx(onnx_path, ref, Xva)
    print("ONNX export", "OK" if d < 1e-4 else "MISMATCH", f"-> {onnx_path}  ({os.path.getsize(onnx_path)} bytes)")
