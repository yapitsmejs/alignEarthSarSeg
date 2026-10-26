from __future__ import annotations

import numpy as np
import pytest
from imageProcToolkit.clamp import clamp
from imageProcToolkit.normalizeArray import normalizeToUint8
from PIL import Image

from alignEarthSarSeg.preprocess import preprocessSar


def _ramp(rows: int, cols: int) -> np.ndarray:
    return (np.linspace(0.01, 1.0, cols, dtype=np.float32)[None, :] * np.ones((rows, 1))).astype(
        np.float32
    )


def testReturnsPilRgb():
    img = preprocessSar(_ramp(8, 8))

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (8, 8)


def testThreeChannelsByteIdentical():
    img = preprocessSar(_ramp(16, 16))
    arr = np.asarray(img)

    assert arr.shape == (16, 16, 3)
    np.testing.assert_array_equal(arr[..., 0], arr[..., 1])
    np.testing.assert_array_equal(arr[..., 1], arr[..., 2])


def testUint8Range():
    arr = np.asarray(preprocessSar(_ramp(32, 32)))

    assert arr.dtype == np.uint8
    assert arr.min() >= 0 and arr.max() <= 255
    assert arr.max() > 0  # not all-black for a real ramp


def testLogChainMatchesReferenceWiring():
    amp = _ramp(16, 16)

    img = preprocessSar(amp, scaling="log")
    ref = normalizeToUint8(np.log10(clamp(amp**2)))
    refRgb = np.stack([ref, ref, ref], axis=-1)

    np.testing.assert_array_equal(np.asarray(img), refRgb)


def testLinearIsDefaultAndMatchesFormula():
    amp = _ramp(16, 16)

    img = preprocessSar(amp)
    hi = np.percentile(amp, 99.9)
    ref = np.rint(np.clip(amp / hi, 0.0, 1.0) * 255.0).astype(np.uint8)

    np.testing.assert_array_equal(np.asarray(img)[..., 0], ref)


def testLinearKeepsZeroFloorAndSaturatesTail():
    amp = np.full((10, 10), 1.0, dtype=np.float32)
    amp[0, 0] = 0.0
    amp[9, 9] = 1000.0  # bright scatterer above the clip percentile

    arr = np.asarray(preprocessSar(amp, clipPercentile=90.0))[..., 0]

    assert arr[0, 0] == 0  # zero amplitude stays black (no min-max stretch)
    assert arr[9, 9] == 255  # tail saturates
    assert arr[5, 5] == 255  # clip level = p90 = 1.0 -> bulk maps to full scale


def testLinearSceneIsDarkLikeTrainingData():
    # Rayleigh-distributed speckle: the linear image should be dark with a long
    # bright tail (like the reference SAR PNGs), not a mid-grey log stretch.
    amp = np.random.default_rng(0).rayleigh(1.0, (128, 128)).astype(np.float32)

    linear = np.asarray(preprocessSar(amp))[..., 0]
    logImg = np.asarray(preprocessSar(amp, scaling="log"))[..., 0]

    assert np.median(linear) < np.median(logImg)
    assert np.median(linear) < 128


def testRejectsUnknownScaling():
    with pytest.raises(ValueError):
        preprocessSar(_ramp(8, 8), scaling="sqrt")


def testRejectsBadClipPercentile():
    with pytest.raises(ValueError):
        preprocessSar(_ramp(8, 8), clipPercentile=0.0)
    with pytest.raises(ValueError):
        preprocessSar(_ramp(8, 8), clipPercentile=101.0)


@pytest.mark.parametrize("scaling", ["linear", "log"])
def testAllZeroReturnsBlack(scaling):
    arr = np.asarray(preprocessSar(np.zeros((8, 8), dtype=np.float32), scaling=scaling))

    assert arr.shape == (8, 8, 3)
    np.testing.assert_array_equal(arr, np.zeros((8, 8, 3), dtype=np.uint8))


@pytest.mark.parametrize("scaling", ["linear", "log"])
def testAllNanReturnsBlackNoCrash(scaling):
    arr = np.asarray(preprocessSar(np.full((8, 8), np.nan, dtype=np.float32), scaling=scaling))

    assert arr.shape == (8, 8, 3)
    np.testing.assert_array_equal(arr, np.zeros((8, 8, 3), dtype=np.uint8))


def testLinearMostlyZeroSceneFallsBackToMax():
    amp = np.zeros((10, 10), dtype=np.float32)
    amp[3, 3] = 4.0  # p50 is 0 -> clip level falls back to the max

    arr = np.asarray(preprocessSar(amp, clipPercentile=50.0))[..., 0]

    assert arr[3, 3] == 255
    assert arr.sum() == 255


def testLinearNanBorderMapsToBlack():
    amp = _ramp(8, 8)
    amp[:, 0] = np.nan

    arr = np.asarray(preprocessSar(amp))[..., 0]

    assert np.all(arr[:, 0] == 0)
    assert arr[:, -1].max() > 0


def testUpsampleFactorDoublesSize():
    img = preprocessSar(_ramp(16, 16), upsampleFactor=2)

    assert img.size == (32, 32)
    assert np.asarray(img).shape == (32, 32, 3)


def testNoUpsampleByDefault():
    img = preprocessSar(_ramp(16, 16))

    assert img.size == (16, 16)
