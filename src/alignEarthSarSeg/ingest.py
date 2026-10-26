"""Ingest SAR images from ``.npy`` files or in-memory arrays.

A SAR scene is fed in as a 2-D ``.npy`` array sampled on an evenly spaced grid.
Complex samples are reduced to amplitude (magnitude); real samples are taken as
amplitude directly (``|x|``). Downstream preprocessing treats the result as
non-negative amplitude.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["loadSarImage"]


def _validateGrid(array: np.ndarray) -> None:
    """Validate that ``array`` is a non-empty 2-D image on an even grid.

    A pixel array is evenly spaced by construction (integer indices along each
    axis), so the grid check reduces to a rank / size check.

    Args:
        array: Candidate SAR array.

    Raises:
        ValueError: If the array is not 2-D or is empty.
    """
    if array.ndim != 2:
        raise ValueError(f"SAR image must be 2-D (H, W); got {array.ndim}-D shape {array.shape!r}.")
    if array.size == 0:
        raise ValueError("SAR image must be non-empty.")


def loadSarImage(image: str | Path | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a SAR image and reduce it to non-negative amplitude.

    Args:
        image: A path to a ``.npy`` file or an in-memory ``numpy.ndarray``. The
            array must be 2-D ``(H, W)``; samples may be real (amplitude) or
            complex (reduced to magnitude).

    Returns:
        A ``(amplitude, metadata)`` tuple where ``amplitude`` is a float32
        ``(H, W)`` non-negative array and ``metadata`` records the source shape,
        dtype, whether the input was complex, and the load origin.

    Raises:
        FileNotFoundError: If ``image`` is a path that does not exist.
        ValueError: If the array is not 2-D or is empty.
    """
    if isinstance(image, np.ndarray):
        array = image
        source: str | None = "ndarray"
    else:
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"SAR image not found: {path}")
        array = np.load(path, allow_pickle=False)
        source = str(path)

    wasComplex = np.iscomplexobj(array)
    _validateGrid(array)

    amplitude = np.abs(array).astype(np.float32, copy=False)
    metadata: dict[str, Any] = {
        "sourceShape": tuple(array.shape),
        "sourceDtype": str(array.dtype),
        "wasComplex": wasComplex,
        "amplitudeShape": tuple(amplitude.shape),
        "nRows": int(amplitude.shape[0]),
        "nCols": int(amplitude.shape[1]),
        "source": source,
    }
    return amplitude, metadata
