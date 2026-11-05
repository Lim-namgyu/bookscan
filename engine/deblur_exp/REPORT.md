# Tiny learned text-deblur prototype — report

Working dir: `/tmp/claude-0/-home-claude/d3460888-10d6-50e2-87cb-0d3a11dad1bf/scratchpad/deblur/`
Nothing under `/home/claude/bookscan` was touched.

## 0. Important deviation: PyTorch could not be installed

* `pip install torch --index-url https://download.pytorch.org/whl/cpu` -> the egress proxy answers
  **403 (policy denial)** for `download.pytorch.org`; not retryable.
* Plain `pip install torch` resolves to the **CUDA build** on PyPI (torch 2.15.0 wheel 539 MB + ~3 GB of
  `nvidia-*` deps). Measured download rate was ~320 KB/s, i.e. roughly 3 hours, far outside the
  60-90 min budget, and a CUDA torch wheel does not even import without the nvidia libs. Aborted.
* Instead of stopping, `train.py` implements the **exact same model, loss and optimiser in numpy**
  (3x3 convs as 9 shifted matmuls, hand-written backprop verified against finite differences to
  ~3e-6 relative error, Adam with cosine decay, batch 32, L1 loss). `train.py` also contains a torch
  code path that is selected automatically when `import torch` works; that path was **not executed
  here** and is untested. The ONNX file is built with the `onnx` helper API (opset 17) from the saved
  OIHW weights, so it is independent of the backend. If a strict "torch only" result is required,
  treat this as a numpy re-implementation of the spec.

## 1. What was built

| file | purpose |
|---|---|
| `gen_data.py` | Renders gray Korean/English text pages with PIL (Noto Sans/Serif CJK KR Regular/Medium/Bold/DemiLight, DejaVu, Liberation; glyph heights 10-32 px; random Hangul syllables + real words + numbers + punctuation; underlines, rules, boxes; paper 235-255, ink 20-90, paper grain). Degrades with Gaussian blur σ 0.5-2.2 (30 % anisotropic), motion blur 0-3 px, contrast reduction, sensor noise σ 1-6, JPEG q70-95. Samples 64x64 patches containing text. `--v2` = widened degradation (see §5). |
| `train.py` | Model + training + ONNX export + onnxruntime verification. `python3 train.py [minutes] [tag]`. |
| `eval.py` | Runs the ONNX model on the real photos in 512-row tiles with 8-px overlap, writes side-by-side crops (3x nearest zoom) and metrics. `--norm` = flat-field normalisation experiment, `--model=_v2` = v2 weights. |
| `deblur.onnx` | v1 model (spec data). opset 17, input `img` 1x1xHxW float32 (H, W dynamic), output `out`, 39.6 KB. |
| `deblur_v2.onnx` | v2 model (widened degradation), same graph. **Recommended of the two.** |

Model: `conv3x3(1→16)+ReLU, 4×[conv3x3(16→16)+ReLU], conv3x3(16→1)`, `out = clip(img + net(img), 0, 1)`.
**9,585 parameters.** Last conv zero-initialised so training starts from the identity. Receptive field 13x13.

## 2. Data

* v1: 3,000 train + 200 val patches, 64x64 uint8, from 250 + 17 rendered 640x640 pages (15 s to generate).
  Files `train_x.npy / train_y.npy / val_x.npy / val_y.npy`; previews `sample_page.png`, `sample_patches.png`.
* v2: same sizes, `train_v2_*.npy / val_v2_*.npy`, `sample_page_v2.png`, `sample_patches_v2.png`.

## 3. Training (numpy backend, 2 CPU threads)

| run | data | step time | epochs / steps | wall time | val L1 identity | val L1 model | gain |
|---|---|---|---|---|---|---|---|
| v1 | spec degradation | 457 ms | 39 / 3,627 | 27.7 min | 0.0728 | **0.0363** | −50.2 % |
| v2 | widened degradation | 450 ms | 13 / 1,209 | 9.1 min | 0.0606 | **0.0457** | −24.7 % |

(L1 on 0..1 scale; identity = blurred input vs clean target; per-epoch curves in `train_log.txt`,
`train_log_v2.txt`, stdout in `train_stdout*.txt`.) v2's smaller relative gain is expected: its val
set contains near-sharp and much noisier samples where little can be gained, and it trained 3x shorter.
v1 val curve: 0.0568 (ep 1) → 0.0390 (ep 10) → 0.0371 (ep 20) → 0.0363 (ep 39), i.e. converged.

Synthetic behaviour (`eval_val_patches*.png`, input | model | target): mild/moderate blur (σ ≲ 1.5) is
recovered convincingly; at σ ≈ 2 strokes stay soft (expected for a 13-px receptive field, 10k params).

## 4. ONNX export

* `deblur.onnx` / `deblur_v2.onnx`: opset 17, IR 8, `onnx.checker` passes, input `img` `[1,1,H,W]` with
  symbolic H/W, Conv(pads 1) → Relu … → Add(img, residual) → Clip(0,1).
* onnxruntime 1.29 CPU vs the reference forward (numpy): **max |diff| 5.4e-7 (v1), 4.2e-7 (v2)**,
  including an odd-sized 101x183 input (dynamic shape works).

## 5. Real-photo evaluation

Images: `closeup_p2.jpg` (sharp, 1659x2212), `closeup_p2_blur1.0.jpg` (same photo, σ 1.0 — gives a real
pixel-level reference), `spread3_up.jpg` (soft 4000x3000 two-page spread; it is an ~1.8x upscale of
`spread3.jpg`, so its softness is partly interpolation blur and its grain is magnified into smooth blobs).

Inference speed (onnxruntime, 2 threads, 512-row tiles, whole image): **0.12-0.20 s per megapixel**
(closeup 3.67 MP: 0.44-0.59 s; spread 12 MP: 2.0-2.3 s). The optional flat-field step adds ~0.06 s/MP.

Metrics on `closeup_p2_blur1.0.jpg` vs the sharp original (`/255` L1, PSNR, and "ink-F1" = overlap of
Otsu-binarised ink pixels with the binarised sharp image — a stroke-shape proxy that ignores the tone change):

| variant | L1 in→out | PSNR in→out | ink-F1 in→out | mean abs change on **sharp** image | ink-F1 of sharp out vs sharp in |
|---|---|---|---|---|---|
| v1 raw | 7.51 → **22.55** (worse) | 24.3 → 20.0 dB | 0.876 → 0.837 | 24.0/255 | 0.80 |
| v1 + flat-field (`--norm`) | 7.51 → 5.55 | 24.3 → 28.1 dB | 0.876 → 0.888 | 14.3/255 | 0.72 |
| **v2 raw** | 7.51 → 7.00 | 24.3 → 26.9 dB | 0.876 → **0.935** | **6.5/255** | 0.69* |
| v2 + flat-field | 7.51 → 7.47 | 24.3 → 27.1 dB | 0.876 → 0.937 | 6.8/255 | 0.71* |

\* the "sharp out vs sharp in" ink-F1 is dominated by Otsu picking a different threshold once the paper is
denoised/flattened; by eye the strokes are unchanged (see crops), so read this column with care.

Full metric dumps: `eval_metrics.txt`, `eval_metrics_norm.txt`, `eval_metrics_v2.txt`, `eval_metrics_v2_norm.txt`.

### What the crops show (3-4 crops per image, `eval_<name>[_v2][_norm].png`; whole outputs `*_full.jpg`)

**v1 (`deblur.onnx`, trained exactly on the spec data) — not usable as is.**
* The synthetic targets have paper ≈ 0.95 and ink ≈ 0.2, so v1 learned a *tone mapper + binariser* as much
  as a deblurrer: real paper (≈ 0.75-0.8) is pushed to white, ink to black. That is why the
  pixel L1 vs the sharp photo gets *worse* even though strokes look crisper.
* **Failure (artifact) crops:** `eval_closeup_p2_blur1.0.png` crop 2 ("body text 2", dim-lit area, paper ≈ 0.65):
  the paper turns into a dense field of black speckles. `eval_spread3_up.png` crops 1-2: dark specks and
  worm-like squiggles all over the paper (magnified grain/JPEG blobs read as faint blurred ink and get
  "sharpened" into strokes). `eval_closeup_p2.png` (sharp input): speckles on paper, worst in the dim region.
* Strokes themselves: no broken strokes or hallucinated glyph parts seen; edges become slightly ragged.
* Flat-field normalisation (divide by local paper level, run model, multiply back — `--norm`) fixes v1 on the
  blurred closeup completely (`eval_closeup_p2_blur1.0_norm.png` is clean, L1 5.55, PSNR +3.9 dB) but does
  **not** fix the spread speckles (`eval_spread3_up_norm.png`) and still speckles the sharp image's dim region.

**v2 (`deblur_v2.onnx`) — widened degradation model, 10 min retrain.** Changes vs spec data: paper 150-255,
stronger low/mid-frequency paper texture, 15 % near-sharp inputs (σ 0-0.5), 40 % "noise before blur",
30 % 1.3-2x down/up resampling, noise σ up to 9.
* `eval_spread3_up_v2.png`: strokes visibly tighter and darker on all four crops (left/middle/right body text,
  far-right small table), paper tone preserved and clean. Remaining artifacts: stroke edges look a bit
  "crunchy"/staircased at 3x zoom, a faint bright halo along strokes (mild ringing), and a handful of tiny
  dark specks near some glyphs (middle-column crop). Nothing that changes a glyph's identity.
* `eval_closeup_p2_blur1.0_v2.png`: clean, crisp, the dim region is handled correctly, the ⑤ marker and small
  answer text are recovered; ink-F1 0.876 → 0.935.
* `eval_closeup_p2_v2.png` (sharp input): glyphs unchanged/slightly crisper, bright paper slightly denoised,
  **but the dim-lit region (crop 2) still gets sporadic dark specks** — far fewer than v1, but visible at 3x.

## 6. Verdict

* **Does it visibly improve legibility of the soft spread photo?** Yes, with v2: strokes are clearly tighter and
  higher-contrast on all crops at ~0.19 s/MP on two CPU cores. v1 also sharpens strokes but covers the paper
  in speckles, which is worse for reading than the blur.
* **Without damaging the sharp one?** Not entirely. v2 leaves the glyphs of the sharp photo intact and is
  close to the identity on bright paper (mean change 6.5/255), but on dim, grainy paper it still hallucinates
  isolated dark specks. v1 damages the sharp photo badly (tone change + speckles).
* Recommendation if this goes further: (1) use the v2-style degradation model and train it for the full
  30 min (v2 got only 13 epochs and was still improving); (2) add a blur gate or blend — only apply where a
  local sharpness measure says the input is soft, or blend output with input in regions with no ink nearby;
  (3) optionally a cheap post-filter removing dark connected components < ~4 px that do not touch a glyph;
  (4) if the app already flat-fields pages, train on flat-fielded-looking data (paper ≈ 1.0) and run the
  model after that step — v1 + `--norm` on the blurred closeup shows how clean that regime can be.
* Honest caveat on the numbers: the only pixel-exact real reference is the synthetically blurred closeup
  (σ 1.0 Gaussian, inside the training range). The spread has no ground truth; its assessment is visual.

## 7. All files (all under `/tmp/claude-0/-home-claude/d3460888-10d6-50e2-87cb-0d3a11dad1bf/scratchpad/deblur/`)

Scripts: `gen_data.py`, `train.py`, `eval.py`, this `REPORT.md`
Data: `train_x.npy`, `train_y.npy`, `val_x.npy`, `val_y.npy`, `train_v2_x.npy`, `train_v2_y.npy`, `val_v2_x.npy`, `val_v2_y.npy`,
`sample_page.png`, `sample_patches.png`, `sample_page_v2.png`, `sample_patches_v2.png`
Models: `deblur.onnx`, `weights.npz` (v1); `deblur_v2.onnx`, `weights_v2.npz` (v2)
Logs: `train_log.txt`, `train_stdout.txt`, `train_log_v2.txt`, `train_stdout_v2.txt`,
`eval_metrics.txt`, `eval_metrics_norm.txt`, `eval_metrics_v2.txt`, `eval_metrics_v2_norm.txt`
Crops (input | model, 3x): `eval_closeup_p2.png`, `eval_closeup_p2_blur1.0.png`, `eval_spread3_up.png` (v1 raw);
`eval_*_norm.png` (v1 + flat-field); `eval_*_v2.png` (v2 raw); `eval_*_v2_norm.png` (v2 + flat-field);
`eval_val_patches*.png` (synthetic val: input | model | target). Whole-image outputs: `eval_*_full.jpg`.
