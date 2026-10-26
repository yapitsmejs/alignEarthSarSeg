"""Reference SegEarth-OV-2 pipeline wrapper.

Delegates segmentation to the ``SegEarthPipeline`` shipped in the AlignEarth-SAR
Hugging Face repo (``pipeline.py`` + ``upsamplers.py`` + ``prompts/`` +
``simfeatup/`` + ``configs/``), loaded from the local snapshot at runtime. A small
shim restores 4.x-style ``get_text_features`` behavior on transformers 5.x. The
reference code lives in the HF cache (not this package), so it stays outside our
mypy/coverage scope; only this wrapper is gated.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download
from transformers import CLIPModel

__all__ = ["loadSegEarthPipeline", "MODEL_REPO"]

MODEL_REPO = "BiliSakura/AlignEarth-SAR-ViT-B-16"

_SHIM_APPLIED = False
_REF_PIPELINE_MODULE: Any = None
_PIPELINE_CACHE: dict[tuple[Any, ...], Any] = {}


def _resolveDevice(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _applyTextFeaturesShim() -> None:
    """Make ``CLIPModel.get_text_features`` return a tensor on transformers 5.x.

    5.x returns a ``BaseModelOutputWithPooling`` with the projected features in
    ``.pooler_output``; the reference pipeline expects a tensor (4.x behavior). This
    shim restores that. Idempotent and a no-op on 4.x (a tensor has no
    ``pooler_output``).
    """
    global _SHIM_APPLIED
    if _SHIM_APPLIED:
        return
    original = CLIPModel.get_text_features

    def shim(self: CLIPModel, *args: Any, **kwargs: Any) -> Any:
        result = cast(Any, original(self, *args, **kwargs))
        return result.pooler_output if hasattr(result, "pooler_output") else result

    CLIPModel.get_text_features = shim  # type: ignore[method-assign]
    _SHIM_APPLIED = True


def _importRefPipeline(snapDir: str) -> Any:
    """Import the reference ``pipeline`` module from the snapshot dir (cached)."""
    global _REF_PIPELINE_MODULE
    if _REF_PIPELINE_MODULE is not None:
        return _REF_PIPELINE_MODULE
    # The reference does ``from upsamplers import ...`` / ``from prompts... import``,
    # so the snapshot dir must be on sys.path for those top-level imports to resolve.
    if snapDir not in sys.path:
        sys.path.insert(0, snapDir)
    _REF_PIPELINE_MODULE = cast(Any, importlib.import_module("pipeline"))
    return _REF_PIPELINE_MODULE


def loadSegEarthPipeline(
    modelPath: str | Path = MODEL_REPO,
    *,
    featupModel: str = "jbu_one",
    classNamesPath: str | Path | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    clsTokenLambda: float = -0.3,
    logitScale: float = 50.0,
    probThd: float = 0.0,
    bgIdx: int = 0,
    slideCrop: int = 0,
    slideStride: int = 112,
    templateSet: str = "openai",
) -> Any:
    """Load (and cache) the reference ``SegEarthPipeline``.

    Args:
        modelPath: Hugging Face repo id, or a local snapshot directory (used as-is,
            no download).
        featupModel: Upsampler (``jbu_one`` default, ``jbu_stack``,
            ``jbu_stack_maskclip``, or ``bilinear`` for a fast low-quality fallback).
        classNamesPath: Class-name file (one class per line, comma synonyms). If
            ``None`` the reference uses its default ``cls_openearthmap_sar.txt``.
        device: Torch device (auto: cuda if available else cpu).
        dtype: Torch dtype (auto: float16 on cuda, float32 on cpu).
        clsTokenLambda: CLS-token Global Bias Alleviation weight.
        logitScale: Temperature before the per-pixel softmax.
        probThd: Sub-threshold pixels fall back to ``bgIdx`` (0 disables).
        bgIdx: Background class index.
        slideCrop: Sliding-window crop size (0 resizes the whole image to 224).
        slideStride: Sliding-window stride.
        templateSet: Text-prompt ensemble (``openai`` or ``sub``).

    Returns:
        A cached ``SegEarthPipeline`` instance ready to call on a PIL image.
    """
    resolvedDevice = _resolveDevice(device)
    resolvedDtype = (
        dtype
        if dtype is not None
        else (torch.float16 if resolvedDevice.type == "cuda" else torch.float32)
    )
    key = (
        str(modelPath),
        featupModel,
        str(classNamesPath),
        str(resolvedDevice),
        str(resolvedDtype),
        clsTokenLambda,
        logitScale,
        probThd,
        bgIdx,
        slideCrop,
        slideStride,
        templateSet,
    )
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached

    # A local snapshot dir is used as-is; anything else is a Hugging Face repo id.
    snapDir = str(modelPath) if Path(modelPath).is_dir() else str(snapshot_download(str(modelPath)))
    _applyTextFeaturesShim()
    ref = _importRefPipeline(snapDir)
    pipe = ref.SegEarthPipeline(
        snapDir,
        featup_model=featupModel,
        class_names_path=str(classNamesPath) if classNamesPath else None,
        device=str(resolvedDevice),
        dtype=resolvedDtype,
        cls_token_lambda=clsTokenLambda,
        logit_scale=logitScale,
        prob_thd=probThd,
        bg_idx=bgIdx,
        slide_crop=slideCrop,
        slide_stride=slideStride,
        template_set=templateSet,
    )
    _PIPELINE_CACHE[key] = pipe
    return pipe
