# alignEarthSarSeg

Ingest amplitude/complex SAR images, preprocess, run AlignEarth-SAR
open-vocabulary segmentation, and return a label image.

## Purpose

AlignEarth-SAR (Hugging Face: [`BiliSakura/AlignEarth-SAR-ViT-B-16`](https://huggingface.co/BiliSakura/AlignEarth-SAR-ViT-B-16))
is a CLIP-style ViT-B/16 distilled from optical vision-language models for SAR
remote sensing. It produces dense features suitable for open-vocabulary
semantic segmentation, but it expects 224×224 RGB PIL images and exposes a
vision encoder rather than a one-call "SAR `.npy` → label map" interface.

`alignEarthSarSeg` closes that gap. It is an **importable library** (no CLI) for
SAR researchers: give it an amplitude-float or complex SAR image stored as a
`.npy` array on an evenly spaced grid, and it returns a `[H, W]` label image.
Internally it scales amplitude to 8-bit greyscale (by default zero-floored linear
amplitude clipped at the 99.9th percentile, matching the SAR imagery the model was
trained on; a dB-clamp + log stretch via [`imageProcToolkit`](https://github.com/yapitsmejs/imageProcToolkit)
is available as `scaling="log"`), optionally FFT-resamples, replicates the single
greyscale channel equally into the three RGB channels, tiles to 224×224, runs
the AlignEarth-SAR vision encoder, upsamples the dense features, and decodes an
open vocabulary of class names into a per-pixel label map.

## Install

This project is managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev          # create .venv and install dev dependencies (pytest, ruff, mypy)
```

`imageProcToolkit` is pulled from its GitHub source (see `[tool.uv.sources]` in
`pyproject.toml`); it is not on PyPI. The committed `uv.lock` is a
last-known-good snapshot — refresh it with
`uv sync --upgrade-package image-proc-toolkit` when tracking the sibling repo's
latest `main`.

On **Windows**, `torch`/`torchvision` are resolved from PyTorch's CUDA 13.0 index
(`[tool.uv.sources]` in `pyproject.toml`), so inference runs on an NVIDIA GPU
(cu128+ is required for Blackwell cards). Other platforms, including Linux CI, get
the CPU builds from PyPI. Check with
`uv run python -c "import torch; print(torch.cuda.is_available())"`.

GPU acceleration of the preprocessing fast-path uses CuPy, which is **not** a
declared dependency — it is installed on demand by `scripts/install_cupy.py` (see
`plans/developmentPlan.md`). The package runs on a NumPy fallback without it.
`uv sync` removes packages not in the lockfile, so re-run
`uv run python scripts/install_cupy.py` after every `uv sync`.

## Segmentation backend

Segmentation is delegated to the **reference SegEarth-OV-2 `SegEarthPipeline`**
shipped in the AlignEarth-SAR Hugging Face repo (`pipeline.py`, `upsamplers.py`,
`simfeatup/`, `prompts/`, `configs/`). On the first call the full model snapshot is
downloaded and cached; the reference pipeline (CLIP encode → SimFeatUp JBU upsample →
open-vocabulary decode → sliding window) is loaded from it at runtime. A small shim
keeps it compatible with transformers 5.x.

The upsampler (`upsampler=`) defaults to **`jbu_one`** (SimFeatUp, best quality);
`jbu_stack`, `jbu_stack_maskclip`, and `bilinear` (fast, lower quality) are
alternatives. On a **CPU**, `jbu_one` is correct but slow — use `upsampler="bilinear"`
for a fast preview, or run on a GPU host for full quality at speed. The optional
`featup` CUDA kernel that accelerates SimFeatUp is **not** a declared dependency
(its `setup.py` needs a CUDA toolkit and imports `torch` at build time); the
reference ships a numerically-equivalent pure-PyTorch fallback and runs without it.
To enable the CUDA kernel on a GPU host:

```bash
uv pip install --no-build-isolation git+https://github.com/likyoo/SimFeatUp.git
```

`truststore` is a declared dependency so Hugging Face downloads work on hosts whose
proxy CA is in the OS cert store but not in `certifi`.

## Usage

The package is imported as `alignEarthSarSeg`. The v1 entry point takes a SAR
`.npy` (or in-memory array) plus an open vocabulary and returns a label image:

```python
import alignEarthSarSeg

# Vocabulary as a list (one class per entry, comma-separated synonyms):
labelImage = alignEarthSarSeg.segmentSar(
    "scene.npy",
    classNames=["building,roof", "road", "water,river", "vegetation,tree,forest"],
)
# labelImage: numpy.ndarray of shape (H, W), int label per pixel

# ...or from a one-class-per-line class file (see tests/fixtures/classes.txt):
labelImage = alignEarthSarSeg.segmentSar("scene.npy", classNamesPath="classes.txt")
```

Images larger than 224 px are segmented with a 224×224 sliding window (stride
112) by default, so large scenes keep full-resolution detail; pass `slideCrop=0`
to squash the whole scene through a single 224×224 pass instead (fast, but coarse).
Segmentation quality depends strongly on the greyscale scaling matching the
model's training data — keep the default `scaling="linear"` unless you have a
reason to change it, and tune `clipPercentile` (default 99.9) if your scene looks
washed out or too dark. `modelPath` accepts a Hugging Face repo id or a local
snapshot directory. The package import alone is enough to check the version:

```python
import alignEarthSarSeg

print(alignEarthSarSeg.__version__)
```

### Running on a chosen image

`alignEarthSarSeg.runScene` wraps `segmentSar` for hand runs: it loads a `.npy`
(amplitude or complex) or a PNG/JPEG/single-band TIFF, segments it, and writes
`<stem>_labels.npy` plus three matplotlib figures (drawn with `origin="lower"`) —
the colour-coded `<stem>_labels.png`, the model-input preview `<stem>_input.png`,
and `<stem>_overview.png` (input | prediction) — the label figures carry a legend
mapping each colour to its class name and area % — to `<image dir>/<stem>_seg/`
(or `--out-dir`):

```bash
uv run python -m alignEarthSarSeg.runScene scene.npy \
    --class background --class "building,roof,house" --class road --class "water,river" \
    --upsampler bilinear --prob-thd 0.3
uv run python -m alignEarthSarSeg.runScene scene.png --classes-file classes.txt \
    --ground-truth scene_label.png      # also prints pixel accuracy
uv run python -m alignEarthSarSeg.runScene --help   # all options
```

From Python: `alignEarthSarSeg.runScene.runSceneFile(path, outDir, classNames=[...], **segmentSarKwargs)`.

Inputs:
- SAR image as a `.npy` file or `numpy.ndarray`, single-channel, sampled on an
  evenly spaced grid. Amplitude may be real (float) or complex — complex input
  is converted to amplitude (magnitude) before preprocessing.

Outputs:
- A `numpy.ndarray` of shape `(H, W)` with an integer class label per pixel.

## Development

```bash
uv run pytest                 # run tests (dev loop, no coverage gate)
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src/alignEarthSarSeg     # type check
uv run python scripts/check.py   # GATE: lint + format + types + tests + branch coverage ≥ 90%
```

Run `scripts/check.py` before pushing or opening a PR — it fails if ruff is
unhappy, formatting would change, **mypy** finds a type error, or branch coverage
on the package drops below 90%. See [`CONVENTIONS.md`](CONVENTIONS.md) for the
full naming, code-style, and test-driven conventions.

### Automation

These run the same gate as you do locally, so a red PR cannot merge:

- **CI** (`.github/workflows/ci.yml`) runs `scripts/check.py` on every push and PR.
- **Git hooks** (`.githooks/`) are opt-in per clone — fast `ruff` checks on
  commit, the full gate on push. One-time setup:
  ```bash
  git config core.hooksPath .githooks
  git update-index --chmod=+x .githooks/pre-commit .githooks/pre-push
  ```
  Use `git commit --no-verify` during active red-phase work; the pre-push hook
  and CI still enforce before anything lands.
- **Dependabot** (`.github/dependabot.yml`) opens weekly dependency PRs (uv +
  GitHub Actions) that must pass the gate like any other.
- **PR template** (`.github/PULL_REQUEST_TEMPLATE.md`) prompts contributors for
  the red-green workflow and the gate checklist.