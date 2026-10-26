from __future__ import annotations

import numpy as np
import pytest

from alignEarthSarSeg.ingest import loadSarImage


def testLoadRealNpy(tmp_path):
    arr = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    path = tmp_path / "real.npy"
    np.save(path, arr)

    amplitude, meta = loadSarImage(path)

    assert amplitude.shape == (2, 2)
    assert amplitude.dtype == np.float32
    assert meta["wasComplex"] is False
    assert meta["sourceShape"] == (2, 2)
    assert meta["nRows"] == 2
    assert meta["nCols"] == 2
    np.testing.assert_allclose(amplitude, np.abs(arr))


def testLoadComplexNpyConvertsToMagnitude(tmp_path):
    arr = np.array([[3 + 4j, 0 + 5j], [1 + 0j, 0 - 1j]], dtype=np.complex64)
    path = tmp_path / "complex.npy"
    np.save(path, arr)

    amplitude, meta = loadSarImage(path)

    assert meta["wasComplex"] is True
    assert amplitude.dtype == np.float32
    np.testing.assert_allclose(amplitude, np.abs(arr).astype(np.float32))


def testLoadNdarrayDirectly():
    arr = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)

    amplitude, meta = loadSarImage(arr)

    assert meta["source"] == "ndarray"
    np.testing.assert_allclose(amplitude, np.abs(arr))


def testRealNegativeBecomesAmplitude():
    arr = np.array([[-1.0, 2.0], [-3.0, 0.0]], dtype=np.float32)

    amplitude, _ = loadSarImage(arr)

    assert np.all(amplitude >= 0)
    np.testing.assert_allclose(amplitude, np.array([[1, 2], [3, 0]], dtype=np.float32))


def testRejectsNon2dArray():
    oneD = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    threeD = np.zeros((4, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        loadSarImage(oneD)
    with pytest.raises(ValueError):
        loadSarImage(threeD)


def testRejectsEmptyArray():
    empty = np.zeros((0, 0), dtype=np.float32)

    with pytest.raises(ValueError):
        loadSarImage(empty)


def testRejectsMissingFile(tmp_path):
    with pytest.raises(FileNotFoundError):
        loadSarImage(tmp_path / "does_not_exist.npy")


def testRejectsNon2dNpyFile(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((5,), dtype=np.float32))

    with pytest.raises(ValueError):
        loadSarImage(path)
