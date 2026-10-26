# Development plan — alignEarthSarSeg

> Living document. Update as milestones land or scope shifts. Conventions and
> the gate live in `CONVENTIONS.md` and `scripts/check.py`; this file is the
> roadmap that ties them to the project's purpose.

## Goal

An importable Python library (`import alignEarthSarSeg`) that takes an
amplitude-float or complex SAR image as a `.npy` array on an evenly spaced grid
and returns a `[H, W]` label image, using the AlignEarth-SAR ViT-B/16 model
([`BiliSakura/AlignEarth-SAR-ViT-B-16`](https://huggingface.co/BiliSakura/AlignEarth-SAR-ViT-B-16))
as the open-vocabulary segmentation backbone and
[`imageProcToolkit`](https://github.com/yapitsmejs/imageProcToolkit) for
preprocessing.

## Audience

SAR researchers who want a one-call path from a `.npy` SAR array to a semantic
label map. The package is a library, not a CLI or a service.

## Scope

**In scope (v1):**
- Load single-channel SAR `.npy` input (real amplitude or complex → magnitude).
- Preprocess amplitude → 8-bit greyscale: zero-floored linear scaling clipped at
  a high percentile (default; matches the model's training imagery), or the
  `imageProcToolkit` dB-clamp + log stretch (`scaling="log"`); optional FFT
  resampling.
- Replicate the single greyscale channel equally into the three RGB channels
  (R = G = B) to satisfy the CLIP image processor's 3-channel expectation.
- Delegate segmentation to the reference SegEarth-OV-2 `SegEarthPipeline`, with
  224×224 sliding-window inference on by default.
- One public entry function (`segmentSar`) with a small, typed surface.

**Non-goals (will not do, even if assumed):**
- Model training, fine-tuning, or weight production. Inference only.
- Shipping model weights — downloaded from Hugging Face at runtime and cached.
- Optical / multi-band imagery. SAR single-channel input only.
- Real-time or served inference; batch / offline library use only.

## Architecture / component breakdown

Segmentation is **delegated to the reference SegEarth-OV-2 `SegEarthPipeline`**
shipped in the AlignEarth-SAR HF repo, loaded from the cached snapshot at runtime
(not reimplemented). The reference code lives in the HF cache, outside this package,
so it is not in our mypy/coverage scope; only the thin wrapper is gated.

Modules under `src/alignEarthSarSeg/` (camelCase per `CONVENTIONS.md`):

- `ingest.py` — `loadSarImage(path | ndarray) -> (amplitude, metadata)`. `.npy`
  loader; complex → magnitude; 2-D / non-empty validation.
- `preprocess.py` — `preprocessSar(amplitude, upsampleFactor=0, scaling="linear",
  clipPercentile=99.9) -> PIL.Image` (RGB uint8). `linear`: amplitude / p99.9,
  clipped to [0, 1], zero stays black. `log`: amplitude → intensity → `clamp` →
  log10 → `normalizeToUint8`. Then 3-channel replication → PIL. Optional
  `fourierUpsample` before scaling.
- `segment.py` — `loadSegEarthPipeline(...)` (the wrapper): `snapshot_download`,
  a transformers-5.x `get_text_features` shim, import the reference `pipeline`
  module from the snapshot dir, construct and config-key-cache
  `SegEarthPipeline`. Also `_resolveDevice`.
- `pipeline.py` — `segmentSar(image, classNames | classNamesPath, ...) -> ndarray`
  (the public entry): ingest → preprocess → resolve the class-name path (temp
  file for a list) → `loadSegEarthPipeline` → `pipe(pilImage)` → nearest-resize
  the label map back to the input `(H, W)` → `int64` ndarray. Handles the
  reference's small-image `slide_crop` edge case. Also `loadClassNames` (a
  standalone vocab-inspection helper) and `_resolveClassNamesPath`.
- `__init__.py` — re-exports `segmentSar`, `loadSegEarthPipeline`, `__version__`.

The custom decoding modules from the earlier reimplementation
(`_upsampler.py`, `_prompts.py`, `tile.py`, the `Segmenter` class) were removed in
favor of the reference pipeline.

## Dependencies

Runtime (in `pyproject.toml` `[project] dependencies`):
- `numpy>=2.3`, `scipy>=1.15` — array backbone and resampling glue.
- `image-proc-toolkit` (GitHub source via `[tool.uv.sources]`) — preprocessing.
- `transformers>=4.45` — the reference pipeline loads the AlignEarth-SAR CLIP
  model from Hugging Face (a shim keeps it compatible with 5.x).
- `torch>=2.2`, `torchvision>=0.17` — model inference + image transforms (used
  by the reference pipeline at runtime).
- `pillow>=10` — PIL image input to the reference pipeline.
- `huggingface-hub>=0.25` — `snapshot_download` of the model + reference code +
  SimFeatUp checkpoints.
- `tqdm>=4.67` — progress.
- `truststore>=0.10` — makes Python ssl use the OS cert store so HF downloads
  succeed on managed/corporate hosts (proxy CA in the system store, not certifi).

Optional / manual (not declared dependencies):
- **CuPy** — GPU fast-path for `imageProcToolkit`. Installed on demand by a
  repo-local script (to be added); NumPy fallback otherwise. mypy silenced.
- **featup (SimFeatUp)** — CUDA JBU kernel that accelerates the reference's
  SimFeatUp upsampler. Not a declared dependency (its `setup.py` imports `torch`
  at build time and needs a CUDA toolkit, which breaks `uv` build isolation). The
  reference ships a pure-PyTorch `adaptive_conv` fallback and runs without it.
  Install manually on a GPU host:
  `uv pip install --no-build-isolation git+https://github.com/likyoo/SimFeatUp.git`.
  `torchvision` and `truststore` are silenced in mypy (untyped / stubless).

## Milestones

### M0 — Skeleton (done)
- Package renamed to `alignEarthSarSeg`, dependencies declared, README and this
  plan written, example files removed. Gate green.

### M1 — Ingest (done)
- `.npy` loader for real amplitude and complex arrays; complex → magnitude;
  2-D / non-empty validation; path and ndarray inputs.
- DoD: hermetic tests, 100% branch coverage, gate green.

### M2 — Preprocess (done)
- `imageProcToolkit` integration (clamp → log10 → `normalizeToUint8`); greyscale
  → 3-channel replication (R = G = B); PIL construction; optional
  `fourierUpsample`.
- DoD: deterministic tests vs reference wiring; property test that the 3 RGB
  channels are byte-identical; gate green.

### M3–M5 — Custom decoding (done, then superseded)
- M3–M5 originally vendored upsamplers/prompts/tiling and a custom `Segmenter`
  reimplementing the SegEarth-OV-2 decoding. Label quality on real scenes was
  poor, so the custom decoding was **replaced** by delegating to the reference
  `SegEarthPipeline` (see "Pivot" below). Those modules/tests were removed.

### M3′ — Reference-pipeline wrapper (done)
- `segment.py` `loadSegEarthPipeline`: `snapshot_download` → transformers-5.x
  `get_text_features` shim → import the reference `pipeline` from the snapshot →
  construct + config-key-cache `SegEarthPipeline`. Hermetic tests: shim
  idempotent + extracts `pooler_output`; cache hit/miss; param mapping. Gate green.

### M4′ — Delegating `segmentSar` (done)
- `pipeline.py` `segmentSar`: ingest → preprocess → `_resolveClassNamesPath`
  (temp file for a list) → `loadSegEarthPipeline` → `pipe(pilImage)` →
  nearest-resize to input `(H, W)` → `int64` ndarray; small-image `slide_crop`
  edge-case handling; `loadClassNames` kept as a helper.
- DoD: hermetic plumbing tests (mocked `loadSegEarthPipeline` + `.npy` fixture);
  README usage runs; gate green. `@pytest.mark.slow` real end-to-end auto-skips
  without a GPU.

### M5′ — Segmentation-quality fixes (done, 2026-10-26)
- Diagnosis on the reference YESeg demo (`demo_YESeg-OPT-SAR`, `bilinear`,
  `probThd=0.3`): the log-stretch preprocessing produced a mid-grey image (median
  107/255) unlike the model's dark linear training imagery (median 7/255) —
  42.8% pixel acc vs 60.8% for the reference input. Default `slideCrop=0`
  additionally squashed large scenes to a 14×14 patch grid.
- Fixes: default `scaling="linear"` (p99.9 clip, zero floor; `log` kept as an
  option); default `slideCrop=224`; `modelPath` accepts a local snapshot dir.
- Result on the demo (`bilinear`): linear 61.6% (≈ reference 60.8%) vs log 45.1%;
  a 4× upsampled 1024 px copy with the new defaults scores 63.6%.
- Open: `jbu_one` scores *lower* than `bilinear` on the demo even in the
  unmodified reference pipeline (35.7% vs 60.8%, CPU, pure-PyTorch JBU
  fallback). Unverified whether this is image-specific or a fallback issue;
  evaluate on more labelled scenes before trusting the `jbu_one` default.

## CI hermeticity & coverage

- Slow / network / GPU tests use `@pytest.mark.slow` (registered in
  `pyproject.toml`) and auto-skip without a GPU, so the gate never downloads a
  model or builds CUDA.
- The ≥90% branch-coverage bar is met by hermetic tests alone: the wrapper
  (`loadSegEarthPipeline`, `_resolveClassNamesPath`, `segmentSar` plumbing) is
  tested with mocked `loadSegEarthPipeline` + a synthetic `.npy` fixture. The
  reference pipeline itself is third-party code outside the package, so it is
  not counted.
- The only uncovered lines are the real `importlib` import path in
  `_importRefPipeline` (exercised only by the GPU-gated slow test).

## Verification

- Per milestone (and before any PR): `uv run python scripts/check.py` — ruff
  check, ruff format --check, mypy strict, pytest with `--cov-branch
  --cov-fail-under=90`.
- End-to-end (network; CPU works but slow, GPU recommended for large scenes):
  ```bash
  uv pip install --no-build-isolation git+https://github.com/likyoo/SimFeatUp.git  # optional CUDA kernel
  python -c "import alignEarthSarSeg, numpy as np; m=alignEarthSarSeg.segmentSar('tests/fixtures/sample.npy', classNamesPath='tests/fixtures/classes.txt'); print(m.shape, m.dtype, np.unique(m))"
  ```

## Notes / deferred

- **Pivot rationale:** the custom-reimplemented decoding produced poor label
  quality on real scenes; delegating to the reference `SegEarthPipeline` (loaded
  from the HF snapshot) matches the reference's validated quality. A spike
  confirmed the reference runs on transformers 5.14 with one `get_text_features`
  shim.
- `segmentSar` defaults `upsampler="jbu_one"` (best quality); CPU + jbu_one is
  correct but slow — `upsampler="bilinear"` is the fast preview fallback, and a
  GPU host is recommended for large scenes.
- The reference's 80-prompt OpenAI ensemble is used by default (`templateSet`
  may be `"sub"` for a faster 7-prompt set).
- `featup` is intentionally not a `uv`-managed dependency (see Dependencies);
  the reference's pure-PyTorch fallback keeps the gate hermetic and the SimFeatUp
  path functional on CPU (slow) and GPU.