"""Preprocess SAR amplitude into the RGB PIL image the CLIP image encoder expects.

Two greyscale scalings are available:

- ``"linear"`` (default): amplitude is divided by a high percentile of the scene
  (``clipPercentile``, default 99.9), clipped to ``[0, 1]`` and mapped to uint8
  with zero amplitude kept at black. This reproduces the look of the 8-bit SAR
  PNGs AlignEarth-SAR was trained and demoed on (dark scene, sparse bright
  scatterers, no min-max stretch), which segments markedly better than the log
  stretch on the reference YESeg demo sample.
- ``"log"``: the sibling repos' visualization chain: amplitude -> intensity
  (``x**2``) -> dB-mode dynamic-range clamp (``imageProcToolkit.clamp``) ->
  log-compression -> min-max normalization to uint8 (``normalizeToUint8``). Good
  for viewing, but a different input distribution from the model's training data.

The single greyscale channel is replicated equally into the three RGB channels
(R = G = B) to satisfy the CLIP image processor's 3-channel expectation.
"""

from __future__ import annotations

import warnings

import numpy as np
from imageProcToolkit.clamp import clamp
from imageProcToolkit.fftUpsample import fourierUpsample
from imageProcToolkit.normalizeArray import normalizeToUint8
from PIL import Image

__all__ = ["preprocessSar", "SCALINGS"]

SCALINGS = ("linear", "log")


def _upsampleAmplitude(amplitude: np.ndarray, factor: int) -> np.ndarray:
    """Band-limited integer-factor upsampling along both axes.

    Args:
        amplitude: 2-D ``(H, W)`` array.
        factor: Integer upsampling factor (``> 1`` to activate).

    Returns:
        The upsampled ``(factor*H, factor*W)`` array.
    """
    up = fourierUpsample(amplitude, factor, axis=0)
    up = fourierUpsample(up, factor, axis=1)
    # FFT zero-pad interpolation of a real signal returns a complex array whose
    # imaginary part is numerical noise (~1e-7); keep the real band-limited result.
    if np.iscomplexobj(up):
        up = np.ascontiguousarray(up.real)
    return np.asarray(up)


def _linearToUint8(amplitude: np.ndarray, clipPercentile: float) -> np.ndarray:
    """Zero-floored linear amplitude scaling with a high-percentile clip.

    Args:
        amplitude: Non-negative ``(H, W)`` amplitude; NaN / non-finite map to 0.
        clipPercentile: Percentile of the finite amplitudes mapped to 255.

    Returns:
        uint8 ``(H, W)`` image.
    """
    finite = amplitude[np.isfinite(amplitude)]
    if finite.size == 0:
        return np.zeros(amplitude.shape, dtype=np.uint8)
    hi = float(np.percentile(finite, clipPercentile))
    if hi <= 0:
        # The bulk of the scene is zero; fall back to the max so sparse returns show.
        hi = float(finite.max())
        if hi <= 0:
            return np.zeros(amplitude.shape, dtype=np.uint8)
    scaled = np.clip(amplitude / hi, 0.0, 1.0) * 255.0
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)  # NaN -> 0 sentinel
    return np.rint(scaled).astype(np.uint8)


def _logToUint8(amplitude: np.ndarray) -> np.ndarray:
    """Intensity -> dB-mode clamp -> log10 -> min-max uint8 (visualization chain)."""
    intensity = amplitude**2
    # clamp emits an All-NaN RuntimeWarning on fully-empty inputs; that degenerate
    # case is handled below (normalizeToUint8 -> black), so silence it here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clamped = clamp(intensity)
        # log10(0) -> -inf is expected where the clamped intensity vanishes (e.g. NaN
        # borders, true zero); normalizeToUint8 masks non-finite values to black.
        with np.errstate(divide="ignore", invalid="ignore"):
            logImg = np.log10(clamped)
    return np.asarray(normalizeToUint8(logImg))


def preprocessSar(
    amplitude: np.ndarray,
    upsampleFactor: int = 0,
    scaling: str = "linear",
    clipPercentile: float = 99.9,
) -> Image.Image:
    """Reduce SAR amplitude to an RGB uint8 PIL image for the vision encoder.

    Args:
        amplitude: Non-negative ``(H, W)`` float array (complex is reduced to
            magnitude by :func:`alignEarthSarSeg.ingest.loadSarImage` before this).
            NaN borders are tolerated and map to black.
        upsampleFactor: Optional integer FFT upsampling factor applied to the
            amplitude before normalization (0 or 1 disables it).
        scaling: ``"linear"`` (default; matches the model's training imagery) or
            ``"log"`` (dB clamp + log stretch, for visualization parity).
        clipPercentile: For ``"linear"``, the amplitude percentile mapped to 255;
            brighter pixels saturate. Must be in ``(0, 100]``. Ignored for ``"log"``.

    Returns:
        An RGB ``PIL.Image`` of shape ``(W, H)`` with the greyscale channel
        replicated into all three channels (R = G = B).

    Raises:
        ValueError: If ``scaling`` is unknown or ``clipPercentile`` is out of range.
    """
    if scaling not in SCALINGS:
        raise ValueError(f"scaling must be one of {SCALINGS}; got {scaling!r}.")
    if not 0.0 < clipPercentile <= 100.0:
        raise ValueError(f"clipPercentile must be in (0, 100]; got {clipPercentile!r}.")

    if upsampleFactor and upsampleFactor > 1:
        amp = _upsampleAmplitude(amplitude, int(upsampleFactor))
    else:
        amp = amplitude

    uint8Img = _linearToUint8(amp, clipPercentile) if scaling == "linear" else _logToUint8(amp)
    rgb = np.stack([uint8Img, uint8Img, uint8Img], axis=-1)
    return Image.fromarray(rgb, mode="RGB")
