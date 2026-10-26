"""Run the full segmentation pipeline on one image file and save viewable results.

Orchestrates load -> :func:`alignEarthSarSeg.segmentSar` -> save, for running the
pipeline by hand on a chosen scene::

    uv run python -m alignEarthSarSeg.runScene scene.npy --class "building,roof" \\
        --class road --class "water,river" --upsampler bilinear --prob-thd 0.3

Accepted inputs: ``.npy`` (real amplitude or complex) or any single-image file PIL
can read (8-bit PNG/JPEG SAR, single-band float/int TIFF); colour images are
converted to greyscale. Outputs, written to ``--out-dir`` (default
``<image dir>/<stem>_seg/``):

- ``<stem>_labels.npy``: the ``int64 (H, W)`` label map.
- ``<stem>_labels.png``: figure of the label map colour-coded with :data:`PALETTE`,
  with a legend (class name + area %).
- ``<stem>_input.png``: figure of the greyscale image the model actually saw.
- ``<stem>_overview.png``: input | prediction side by side, with the legend.

Figures are drawn with matplotlib using ``origin="lower"`` (row 0 at the bottom).

An optional ``--ground-truth`` label image (``.npy`` or PNG of class indices, same
size as the input) adds a pixel-accuracy score.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image

from alignEarthSarSeg.pipeline import loadClassNames, segmentSar
from alignEarthSarSeg.preprocess import SCALINGS, preprocessSar

__all__ = ["runSceneFile", "loadSceneArray", "colorizeLabels", "plotScene", "main", "PALETTE"]

# Matplotlib tab20 (bright then pale), so labels look like the reference demo figures.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (174, 199, 232),
    (255, 187, 120),
    (152, 223, 138),
    (255, 152, 150),
    (197, 176, 213),
    (196, 156, 148),
    (247, 182, 210),
    (199, 199, 199),
    (219, 219, 141),
    (158, 218, 229),
)

_FIGURE_DPI = 150


def loadSceneArray(path: str | Path) -> np.ndarray:
    """Load a scene file as a 2-D array.

    Args:
        path: ``.npy`` file (returned as stored, complex kept) or any image file PIL
            can open. Colour/palette images are converted to greyscale; numeric
            single-band images (e.g. float TIFF) keep their values.

    Returns:
        The image array (``float32`` for image files).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    with Image.open(path) as src:
        img = src if src.mode in ("L", "F", "I", "I;16", "I;16B", "I;16L") else src.convert("L")
        return np.asarray(img).astype(np.float32)


def colorizeLabels(labels: np.ndarray) -> np.ndarray:
    """Map an integer label map to RGB with :data:`PALETTE` (cycling past 20 classes).

    Args:
        labels: ``(H, W)`` non-negative integer label map.

    Returns:
        ``(H, W, 3)`` uint8 RGB image.
    """
    palette = np.asarray(PALETTE, dtype=np.uint8)
    return palette[np.asarray(labels, dtype=np.int64) % len(PALETTE)]


def _legendNames(
    classNames: list[str] | None, classNamesPath: str | Path | None, labels: np.ndarray
) -> list[str]:
    """First synonym of each class, or ``class k`` when the vocabulary is the default."""
    if classNames is None and classNamesPath is None:
        return [f"class {k}" for k in range(int(labels.max()) + 1)]
    names, indices = loadClassNames(classNames, classNamesPath)
    legend: dict[int, str] = {}
    for name, idx in zip(names, indices, strict=True):
        legend.setdefault(idx, name)
    return [legend[k] for k in sorted(legend)]


def _rgbFloat(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """uint8 RGB -> matplotlib's 0-1 float RGB."""
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def plotScene(
    inputGrey: np.ndarray | None,
    labels: np.ndarray | None,
    legend: list[str],
    fractions: dict[int, float],
    title: str = "",
) -> Figure:
    """Figure of the model input and/or the prediction, drawn with ``origin="lower"``.

    Panels are laid out left to right (input, then prediction); when the prediction
    is shown, a legend maps each :data:`PALETTE` colour to its class name and area %.

    Args:
        inputGrey: ``(H, W)`` greyscale model input, or ``None`` to omit the panel.
        labels: ``(H, W)`` integer label map, or ``None`` to omit the panel.
        legend: Class names, indexed by label value.
        fractions: Label value -> fraction of pixels.
        title: Optional figure title.

    Returns:
        The matplotlib figure (not shown; caller saves it).
    """
    panels = [p for p in (inputGrey, labels) if p is not None]
    fig = Figure(figsize=(6.5 * len(panels) + (3.0 if labels is not None else 0.0), 6.5))
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    k = 0
    if inputGrey is not None:
        axes[k].imshow(inputGrey, cmap="gray", vmin=0, vmax=255, origin="lower")
        axes[k].set_title("Model input")
        k += 1
    if labels is not None:
        axes[k].imshow(colorizeLabels(labels), origin="lower", interpolation="nearest")
        axes[k].set_title("Prediction")
        handles = [
            Patch(
                facecolor=_rgbFloat(PALETTE[i % len(PALETTE)]),
                edgecolor="black",
                linewidth=0.5,
                label=f"{i}: {name} ({fractions.get(i, 0.0) * 100:.1f}%)",
            )
            for i, name in enumerate(legend)
        ]
        axes[k].legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            title="Class (area %)",
        )
    for ax in axes:
        ax.set_xlabel("column (px)")
        ax.set_ylabel("row (px)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def runSceneFile(
    image: str | Path,
    outDir: str | Path,
    classNames: list[str] | None = None,
    classNamesPath: str | Path | None = None,
    groundTruth: str | Path | None = None,
    **segmentKwargs: Any,
) -> dict[str, Any]:
    """Segment one image file and write the label map plus viewable figures.

    Args:
        image: Scene file (see :func:`loadSceneArray`).
        outDir: Output directory (created if missing).
        classNames: Vocabulary list (one class per entry, comma synonyms).
        classNamesPath: Vocabulary file (one class per line). Neither -> the
            reference's default OpenEarthMap-SAR classes.
        groundTruth: Optional label image of class indices, same ``(H, W)`` as the
            scene, to compute pixel accuracy.
        **segmentKwargs: Forwarded to :func:`alignEarthSarSeg.segmentSar`
            (``upsampler``, ``probThd``, ``scaling``, ``clipPercentile``, ``slideCrop``,
            ``device``, ...).

    Returns:
        Dict with ``labels`` (ndarray), ``legend`` (class names), ``classFractions``
        (class index -> area fraction), ``pixelAccuracy`` (float or ``None``) and the
        output paths ``labelsNpy``, ``labelsPng``, ``inputPng``, ``overviewPng``.

    Raises:
        ValueError: If ``groundTruth`` does not match the scene shape.
    """
    image = Path(image)
    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    stem = image.stem

    scene = loadSceneArray(image)
    labels = segmentSar(
        scene, classNames=classNames, classNamesPath=classNamesPath, **segmentKwargs
    )

    pixelAccuracy: float | None = None
    if groundTruth is not None:
        gt = loadSceneArray(groundTruth)
        if gt.shape != labels.shape:
            raise ValueError(f"Ground truth shape {gt.shape} != prediction shape {labels.shape}.")
        pixelAccuracy = float((gt.astype(np.int64) == labels).mean())

    values, counts = np.unique(labels, return_counts=True)
    fractions = {int(v): float(c) / labels.size for v, c in zip(values, counts, strict=True)}
    legend = _legendNames(classNames, classNamesPath, labels)

    preprocessKeys = ("upsampleFactor", "scaling", "clipPercentile")
    inputImg = preprocessSar(
        np.abs(scene).astype(np.float32),
        **{k: segmentKwargs[k] for k in preprocessKeys if k in segmentKwargs},
    )
    if inputImg.size != (labels.shape[1], labels.shape[0]):
        inputImg = inputImg.resize((labels.shape[1], labels.shape[0]), Image.Resampling.BILINEAR)
    inputGrey = np.asarray(inputImg)[..., 0]

    paths = {
        "labelsNpy": outDir / f"{stem}_labels.npy",
        "labelsPng": outDir / f"{stem}_labels.png",
        "inputPng": outDir / f"{stem}_input.png",
        "overviewPng": outDir / f"{stem}_overview.png",
    }
    np.save(paths["labelsNpy"], labels)
    figures = {
        "inputPng": plotScene(inputGrey, None, legend, fractions, title=stem),
        "labelsPng": plotScene(None, labels, legend, fractions, title=stem),
        "overviewPng": plotScene(inputGrey, labels, legend, fractions, title=stem),
    }
    for key, fig in figures.items():
        fig.savefig(paths[key], dpi=_FIGURE_DPI, bbox_inches="tight")

    return {
        "labels": labels,
        "legend": legend,
        "classFractions": fractions,
        "pixelAccuracy": pixelAccuracy,
        **paths,
    }


def _buildParser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m alignEarthSarSeg.runScene",
        description="Segment one SAR image and save the label map and figures.",
    )
    p.add_argument("image", type=Path, help=".npy (amplitude/complex) or PNG/JPEG/TIFF scene")
    p.add_argument(
        "--out-dir", type=Path, help="output directory (default: <image dir>/<stem>_seg)"
    )
    vocab = p.add_mutually_exclusive_group()
    vocab.add_argument(
        "--class",
        dest="classNames",
        action="append",
        metavar="NAME[,SYNONYM...]",
        help="one class (repeat the flag); commas separate synonyms",
    )
    vocab.add_argument("--classes-file", type=Path, help="class file, one class per line")
    p.add_argument(
        "--ground-truth", type=Path, help="label image of class indices, for pixel accuracy"
    )
    p.add_argument(
        "--upsampler",
        default="jbu_one",
        choices=["jbu_one", "jbu_stack", "jbu_stack_maskclip", "bilinear"],
        help="feature upsampler (bilinear is much faster on CPU)",
    )
    p.add_argument("--prob-thd", type=float, default=0.0, help="low-confidence pixels -> class 0")
    p.add_argument("--scaling", default="linear", choices=SCALINGS, help="greyscale scaling")
    p.add_argument("--clip-percentile", type=float, default=99.9, help="linear-scaling white point")
    p.add_argument(
        "--slide-crop", type=int, default=224, help="sliding-window size (0 = one resized pass)"
    )
    p.add_argument("--slide-stride", type=int, default=112, help="sliding-window stride")
    p.add_argument(
        "--upsample-factor", type=int, default=0, help="FFT upsampling factor before scaling"
    )
    p.add_argument("--device", help="torch device, e.g. cpu or cuda (default: auto)")
    return p


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns a process exit code.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success.
    """
    args = _buildParser().parse_args(argv)
    image: Path = args.image
    outDir: Path = args.out_dir or image.parent / f"{image.stem}_seg"
    result = runSceneFile(
        image,
        outDir,
        classNames=args.classNames,
        classNamesPath=args.classes_file,
        groundTruth=args.ground_truth,
        upsampler=args.upsampler,
        probThd=args.prob_thd,
        scaling=args.scaling,
        clipPercentile=args.clip_percentile,
        slideCrop=args.slide_crop,
        slideStride=args.slide_stride,
        upsampleFactor=args.upsample_factor,
        device=args.device,
    )
    print(f"labels   {result['labels'].shape} -> {result['labelsNpy']}")
    print(f"overview -> {result['overviewPng']}")
    for k, name in enumerate(result["legend"]):
        print(f"  {k}: {name:<20s} {result['classFractions'].get(k, 0.0) * 100:5.1f}%")
    if result["pixelAccuracy"] is not None:
        print(f"pixel accuracy vs ground truth: {result['pixelAccuracy'] * 100:.2f}%")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
