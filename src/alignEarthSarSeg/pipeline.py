"""Public one-call API: a SAR ``.npy``/ndarray -> per-pixel label image.

Wires ingest -> preprocess -> the reference SegEarth-OV-2 ``SegEarthPipeline``
(loaded and cached via :func:`alignEarthSarSeg.segment.loadSegEarthPipeline`). The
pipeline does CLIP normalization, vision encoding, SimFeatUp upsampling,
open-vocabulary decoding, and sliding-window tiling; this module owns the SAR
plumbing and returns the label map at the input resolution.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alignEarthSarSeg.ingest import loadSarImage
from alignEarthSarSeg.preprocess import preprocessSar
from alignEarthSarSeg.segment import MODEL_REPO, loadSegEarthPipeline

__all__ = ["segmentSar", "loadClassNames", "loadSegEarthPipeline"]

_CLASS_FILE_CACHE: dict[tuple[str, ...], str] = {}


def loadClassNames(
    classNames: list[str] | None = None,
    classNamesPath: str | Path | None = None,
) -> tuple[list[str], list[int]]:
    """Parse the open vocabulary into per-synonym names and parent-class indices.

    Either a ``classNames`` list (one class per entry, comma-separated synonyms) or a
    ``classNamesPath`` text file (one class per line, comma-separated synonyms; blank
    lines ignored) — mirroring the reference ``cls_*.txt`` format.

    Args:
        classNames: List of class entries, e.g. ``["building,roof", "road", "water"]``.
        classNamesPath: Path to a class-name file with one class per line.

    Returns:
        ``(names, indices)`` where ``names`` is a flat list of synonyms and
        ``indices[i]`` is the parent class index of ``names[i]``.

    Raises:
        ValueError: If both or neither input is given, or the vocabulary is empty.
    """
    if classNames is not None and classNamesPath is not None:
        raise ValueError("Provide classNames or classNamesPath, not both.")
    if classNamesPath is not None:
        names: list[str] = []
        indices: list[int] = []
        idx = 0
        for line in Path(classNamesPath).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            for syn in line.split(","):
                syn = syn.strip()
                if syn:
                    names.append(syn)
                    indices.append(idx)
            idx += 1
        if not names:
            raise ValueError(f"No class names found in {classNamesPath!r}.")
        return names, indices
    if classNames is None:
        raise ValueError("Provide classNames or classNamesPath.")
    names = []
    indices = []
    for idx, entry in enumerate(classNames):
        for syn in entry.split(","):
            syn = syn.strip()
            if syn:
                names.append(syn)
                indices.append(idx)
    if not names:
        raise ValueError("classNames is empty.")
    return names, indices


def _resolveClassNamesPath(
    classNames: list[str] | None, classNamesPath: str | Path | None
) -> str | None:
    """Resolve a class-name file path for the reference pipeline.

    ``classNamesPath`` is returned as-is; a ``classNames`` list is written to a
    process-cached temp file (one class per line, comma synonyms) so repeated calls
    reuse one file; ``None`` lets the reference use its default class set.
    """
    if classNames is not None and classNamesPath is not None:
        raise ValueError("Provide classNames or classNamesPath, not both.")
    if classNamesPath is not None:
        return str(classNamesPath)
    if classNames is None:
        return None
    if not classNames:
        raise ValueError("classNames is empty.")
    key = tuple(classNames)
    cached = _CLASS_FILE_CACHE.get(key)
    if cached is not None:
        return cached
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="aess_classes_")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(classNames) + "\n")
    _CLASS_FILE_CACHE[key] = path
    return path


def segmentSar(
    image: str | Path | np.ndarray,
    classNames: list[str] | None = None,
    classNamesPath: str | Path | None = None,
    *,
    upsampleFactor: int = 0,
    scaling: str = "linear",
    clipPercentile: float = 99.9,
    upsampler: str = "jbu_one",
    device: str | None = None,
    clsTokenLambda: float = -0.3,
    logitScale: float = 50.0,
    probThd: float = 0.0,
    bgIdx: int = 0,
    slideCrop: int = 224,
    slideStride: int = 112,
    templateSet: str = "openai",
    modelPath: str | Path = MODEL_REPO,
) -> np.ndarray:
    """Segment a SAR image into a per-pixel label map.

    Args:
        image: A ``.npy`` path or in-memory ``ndarray`` (real amplitude or complex).
        classNames: List of class entries (comma-separated synonyms per entry).
        classNamesPath: Path to a one-class-per-line class-name file. If neither is
            given, the reference's default ``cls_openearthmap_sar.txt`` is used.
        upsampleFactor: Optional integer FFT upsampling factor (preprocessing). The
            model runs on the upsampled image; labels are downsampled (nearest)
            back to the input resolution.
        scaling: Greyscale scaling — ``"linear"`` (default; zero-floored amplitude
            clipped at ``clipPercentile``, matching the model's training imagery) or
            ``"log"`` (dB clamp + log stretch).
        clipPercentile: Amplitude percentile mapped to white for ``"linear"``.
        upsampler: Reference upsampler — ``jbu_one`` (default, best quality),
            ``jbu_stack``, ``jbu_stack_maskclip``, or ``bilinear`` (fast, lower
            quality). CPU + JBU is correct but slow; ``bilinear`` is the fast fallback.
        device: Torch device override (``None`` auto-selects cuda/cpu).
        clsTokenLambda: CLS-token Global Bias Alleviation weight.
        logitScale: Temperature applied before the per-pixel softmax.
        probThd: Pixels with max class probability below this fall back to ``bgIdx``
            (0 disables the threshold).
        bgIdx: Background class index for sub-threshold pixels.
        slideCrop: Sliding-window crop size (default 224, the encoder's native size,
            so large scenes keep full-resolution detail). 0 resizes the whole image
            to one 224×224 pass. For images no larger than ``slideCrop`` the resize
            path is used automatically.
        slideStride: Sliding-window stride.
        templateSet: Text-prompt ensemble (``openai`` or ``sub``).
        modelPath: Hugging Face model id or local snapshot directory.

    Returns:
        An ``int64`` ``[H, W]`` ``numpy.ndarray`` with one class label per pixel, at
        the input image resolution.
    """
    amplitude, _meta = loadSarImage(image)
    pilImage = preprocessSar(
        amplitude, upsampleFactor=upsampleFactor, scaling=scaling, clipPercentile=clipPercentile
    )
    # Labels are returned at the input resolution; the model sees the (possibly
    # FFT-upsampled) image, whose size decides tiling.
    origH, origW = amplitude.shape
    imgW, imgH = pilImage.size
    classesPath = _resolveClassNamesPath(classNames, classNamesPath)

    # The reference keeps the native size when slide_crop>0 and then rejects images
    # smaller than the crop (the encoder needs 224). Only slide when the image is
    # actually larger than the crop; otherwise fall back to the resize-to-224 path.
    effectiveSlideCrop = slideCrop if (imgH > slideCrop or imgW > slideCrop) else 0

    pipe = loadSegEarthPipeline(
        modelPath=modelPath,
        featupModel=upsampler,
        classNamesPath=classesPath,
        device=device,
        clsTokenLambda=clsTokenLambda,
        logitScale=logitScale,
        probThd=probThd,
        bgIdx=bgIdx,
        slideCrop=effectiveSlideCrop,
        slideStride=slideStride,
        templateSet=templateSet,
    )
    seg = pipe(pilImage)
    segTensor = torch.as_tensor(seg).detach().cpu()
    if tuple(segTensor.shape) != (origH, origW):
        segTensor = (
            F.interpolate(
                segTensor.unsqueeze(0).unsqueeze(0).float(), size=(origH, origW), mode="nearest"
            )
            .squeeze(0)
            .squeeze(0)
            .to(torch.int64)
        )
    return segTensor.numpy().astype(np.int64)
