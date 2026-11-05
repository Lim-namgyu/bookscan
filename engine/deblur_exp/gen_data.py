#!/usr/bin/env python3
"""Synthetic blurred/clean text patch pairs for a tiny deblur CNN.

Renders gray pages of Korean+English text with PIL, degrades them (Gaussian
blur, mild motion blur, sensor noise, JPEG, contrast loss) and samples 64x64
patches that contain text.  Target = the clean render (same geometry).

Output (in this directory):
  train_x.npy / train_y.npy : (N,64,64) uint8, degraded / clean
  val_x.npy   / val_y.npy   : (200,64,64) uint8
  sample_page.png           : one degraded|clean page pair for a sanity check
"""
import os, sys, time, random, io
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.dirname(os.path.abspath(__file__))
PATCH = 64
V2 = "--v2" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
N_TRAIN = int(args[0]) if args else 3000
N_VAL = 200
TAG = "_v2" if V2 else ""
# --v2: widened degradation model after the first real-photo evaluation (see REPORT.md):
#   paper tone 150..255 instead of 235..255, stronger low-frequency paper texture,
#   15% near-sharp inputs (sigma 0..0.5), 40% "noise before blur" (upscaled-photo
#   look: grain becomes smooth blobs), noise sigma up to 9, optional 1.3-2x down/up resample.
SEED = 1234
random.seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------- fonts
NOTO = "/usr/share/fonts/opentype/noto/"
KR_FONTS = [  # (path, index of the KR face)
    (NOTO + "NotoSansCJK-Regular.ttc", 1),
    (NOTO + "NotoSansCJK-Medium.ttc", 1),
    (NOTO + "NotoSansCJK-Bold.ttc", 1),
    (NOTO + "NotoSansCJK-DemiLight.ttc", 1),
    (NOTO + "NotoSerifCJK-Regular.ttc", 1),
    (NOTO + "NotoSerifCJK-Medium.ttc", 1),
    (NOTO + "NotoSerifCJK-Bold.ttc", 1),
]
EN_FONTS = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 0),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 0),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0),
    ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 0),
    ("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", 0),
]
_font_cache = {}
def font(path, idx, size):
    k = (path, idx, size)
    if k not in _font_cache:
        _font_cache[k] = ImageFont.truetype(path, size, index=idx)
    return _font_cache[k]

# ---------------------------------------------------------------- text
KR_WORDS = ("학교 선생님 문제 다음 중 옳은 것은 정답 해설 단원 평가 수학 과학 국어 영어 사회 "
            "학생 공부 시험 교과서 문장 단어 의미 설명 그림 표 번호 보기 가장 적절한 않은 "
            "하였다 있다 없다 때문에 그러나 그리고 또한 대하여 위하여 통해 따라 경우 결과 이유 "
            "서울 부산 대한민국 사람 시간 오늘 내일 어제 날씨 바람 사랑 가족 친구 이름 "
            "컴퓨터 프로그램 데이터 네트워크 보안 서버 클라우드 방화벽 정책 점검 문서").split()
EN_WORDS = ("the of and to in is that for with as on by this from are be at or an it "
            "which not have one you were all can their has more will when there were "
            "text book page chapter section exercise answer question number example "
            "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
            "Python Android camera image scan blur sharp filter model network").split()
PUNCT = list(".,;:!?()[]「」『』·-—/%\"'")

def rand_hangul(n):
    return "".join(chr(random.randint(0xAC00, 0xD7A3)) for _ in range(n))

def rand_token(lang):
    r = random.random()
    if lang == "kr":
        if r < 0.45: return random.choice(KR_WORDS)
        if r < 0.80: return rand_hangul(random.randint(1, 5))
        if r < 0.90: return str(random.randint(0, 9999)) + random.choice(["", ".", ")", "번", "쪽"])
        return random.choice(EN_WORDS)
    else:
        if r < 0.80: return random.choice(EN_WORDS)
        if r < 0.90: return str(random.randint(0, 99999))
        return "".join(random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(random.randint(2, 8)))

def rand_line(lang, nwords):
    toks = []
    for i in range(nwords):
        t = rand_token(lang)
        if random.random() < 0.15: t += random.choice(PUNCT)
        toks.append(t)
    if random.random() < 0.25:
        toks.insert(0, random.choice(["1.", "2.", "3.", "4.", "5.", "①", "②", "③", "④", "(1)", "(2)", "가.", "나.", "다.", "※", "•"]))
    return " ".join(toks)

# ---------------------------------------------------------------- page rendering
def render_page(W=640, H=640):
    """Returns clean uint8 gray page with text, plus a text-ink mask."""
    paper = random.uniform(150, 255) if V2 else random.uniform(235, 255)
    img = Image.new("L", (W, H), int(paper))
    d = ImageDraw.Draw(img)
    lang_page = "kr" if random.random() < 0.7 else "en"
    ink = int(random.uniform(20, 90))
    # choose a base font family per page, vary size per paragraph
    y = random.randint(4, 30)
    x0 = random.randint(4, 30)
    while y < H - 12:
        lang = lang_page if random.random() < 0.85 else ("en" if lang_page == "kr" else "kr")
        fpath, fidx = random.choice(KR_FONTS if lang == "kr" else EN_FONTS)
        # glyph heights 10..32 px; font size is ~ glyph height for CJK, a bit larger for Latin
        gh = random.uniform(10, 32)
        size = int(round(gh if lang == "kr" else gh * 1.35))
        f = font(fpath, fidx, max(size, 8))
        spacing = random.uniform(1.15, 1.9)
        nlines = random.randint(1, 6)
        local_ink = min(255, max(0, ink + random.randint(-15, 15)))
        for _ in range(nlines):
            if y >= H - 12: break
            nwords = random.randint(2, 14)
            txt = rand_line(lang, nwords)
            x = x0 + random.randint(0, 12)
            d.text((x, y), txt, fill=local_ink, font=f)
            # occasional underline / box lines
            if random.random() < 0.08:
                bb = d.textbbox((x, y), txt, font=f)
                d.line([(bb[0], bb[3] + 1), (min(bb[2], W - 2), bb[3] + 1)], fill=local_ink, width=random.randint(1, 2))
            y += int(size * spacing)
        y += random.randint(0, int(size * 0.8))
        if random.random() < 0.10:  # rule / box line
            yy = min(H - 2, y + random.randint(0, 6))
            d.line([(random.randint(0, 40), yy), (W - random.randint(0, 40), yy)], fill=local_ink, width=random.randint(1, 3))
            y += 8
        if random.random() < 0.06 and y < H - 40:  # box
            bh = random.randint(30, 90)
            d.rectangle([x0, y, W - random.randint(5, 40), min(H - 2, y + bh)], outline=local_ink, width=random.randint(1, 2))
            y += 6
    arr = np.asarray(img).astype(np.float32)
    # slight paper texture (low-frequency + fine grain)
    grain = np.random.normal(0, random.uniform(0.5, 3.0 if V2 else 2.0), arr.shape).astype(np.float32)
    lowf = cv2.resize(np.random.normal(0, random.uniform(1, 8 if V2 else 4), (8, 8)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    if V2 and random.random() < 0.5:   # mid-frequency paper mottle (fibres / print texture)
        arr = arr + cv2.resize(np.random.normal(0, random.uniform(1, 5), (W // 6, H // 6)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    arr = arr + grain + lowf
    arr = np.clip(arr, 0, 255)
    mask = (np.asarray(img) < 200)
    return arr, mask

def motion_kernel(length, angle):
    k = np.zeros((int(length) + 3, int(length) + 3), np.float32)
    c = k.shape[0] // 2
    k[c, c - int(length) // 2 : c - int(length) // 2 + max(1, int(length))] = 1.0
    M = cv2.getRotationMatrix2D((c, c), angle, 1.0)
    k = cv2.warpAffine(k, M, k.shape[::-1], flags=cv2.INTER_LINEAR)
    s = k.sum()
    return k / s if s > 0 else None

def degrade(clean):
    x = clean.copy()
    pre_noise = V2 and random.random() < 0.4
    if pre_noise:   # grain captured BEFORE the blur / upscaling (upscaled or soft photos)
        x = x + np.random.normal(0, random.uniform(2, 9), x.shape).astype(np.float32)
    # Gaussian (optionally slightly anisotropic) blur
    s = random.uniform(0.0, 0.5) if (V2 and random.random() < 0.15) else random.uniform(0.5, 2.2)
    if V2 and random.random() < 0.3:   # low-res capture then upscale (as in spread3_up)
        f = random.uniform(1.3, 2.0); h, w = x.shape
        x = cv2.resize(cv2.resize(x, (int(w / f), int(h / f)), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if s < 0.3: s = 0.3
    sx, sy = s, s
    if random.random() < 0.3:
        sy = s * random.uniform(0.6, 1.4)
    x = cv2.GaussianBlur(x, (0, 0), sigmaX=sx, sigmaY=sy)
    # mild motion blur 0..3 px
    ml = random.uniform(0, 3)
    if ml >= 1.0:
        k = motion_kernel(ml, random.uniform(0, 180))
        if k is not None:
            x = cv2.filter2D(x, -1, k)
    # slight contrast reduction (paper darkens, ink lightens)
    if random.random() < 0.7:
        a = random.uniform(0.80, 1.0)
        m = random.uniform(120, 200)
        x = (x - m) * a + m + random.uniform(-6, 6)
    # sensor noise
    x = x + np.random.normal(0, random.uniform(1, 9 if V2 else 6) * (0.5 if pre_noise else 1.0), x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    # JPEG round trip
    q = random.randint(70, 95)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
    x = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return x

def sample_patches(deg, clean, mask, n):
    H, W = clean.shape
    xs, ys = [], []
    tries = 0
    while len(xs) < n and tries < n * 30:
        tries += 1
        y = random.randint(0, H - PATCH); x = random.randint(0, W - PATCH)
        m = mask[y:y + PATCH, x:x + PATCH]
        # require some ink (>=1.5% of pixels), allow a few nearly-empty ones
        if m.mean() < 0.015 and random.random() > 0.05:
            continue
        xs.append(deg[y:y + PATCH, x:x + PATCH])
        ys.append(np.clip(np.round(clean[y:y + PATCH, x:x + PATCH]), 0, 255).astype(np.uint8))
    return xs, ys

def build(n_total, per_page=12):
    X, Y = [], []
    t0 = time.time()
    pages = 0
    while len(X) < n_total:
        clean, mask = render_page()
        deg = degrade(clean)
        xs, ys = sample_patches(deg, clean, mask, per_page)
        X += xs; Y += ys; pages += 1
        if pages == 1:
            cv2.imwrite(os.path.join(OUT, f"sample_page{TAG}.png"), np.hstack([deg, clean.astype(np.uint8)]))
    X = np.stack(X[:n_total]); Y = np.stack(Y[:n_total])
    print(f"  {len(X)} patches from {pages} pages in {time.time() - t0:.1f}s")
    return X, Y

if __name__ == "__main__":
    print("Generating training patches...")
    Xtr, Ytr = build(N_TRAIN)
    np.save(os.path.join(OUT, f"train{TAG}_x.npy"), Xtr); np.save(os.path.join(OUT, f"train{TAG}_y.npy"), Ytr)
    random.seed(SEED + 1); np.random.seed(SEED + 1)
    print("Generating validation patches...")
    Xva, Yva = build(N_VAL)
    np.save(os.path.join(OUT, f"val{TAG}_x.npy"), Xva); np.save(os.path.join(OUT, f"val{TAG}_y.npy"), Yva)
    # identity baseline (mean abs error degraded vs clean, in 0..1)
    print("train shape", Xtr.shape, Ytr.shape, "val shape", Xva.shape)
    print("identity L1 (train) = %.4f   (val) = %.4f" % (
        np.abs(Xtr.astype(np.float32) - Ytr).mean() / 255, np.abs(Xva.astype(np.float32) - Yva).mean() / 255))
    # save a contact sheet of 32 val pairs
    k = 32
    sheet = np.vstack([np.hstack([Xva[i], Yva[i]]) for i in range(k)])
    sheet = sheet.reshape(8, 4, PATCH, PATCH * 2).transpose(0, 2, 1, 3).reshape(8 * PATCH, 4 * PATCH * 2)
    cv2.imwrite(os.path.join(OUT, f"sample_patches{TAG}.png"), sheet)
