from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import alignEarthSarSeg.pipeline as pipeline
from alignEarthSarSeg.pipeline import loadClassNames, segmentSar


def _saveSampleNpy(path: Path, h: int = 64, w: int = 80) -> None:
    rng = np.random.RandomState(0)
    arr = rng.rand(h, w).astype(np.float32) * 10 + 1e-3
    np.save(path, arr)


class _FakePipe:
    """Stand-in for the reference SegEarthPipeline."""

    def __init__(self, outShape: tuple[int, int] = (224, 224)) -> None:
        self.outShape = outShape
        self.calls: list[object] = []

    def __call__(self, pilImage):
        self.calls.append(pilImage)
        return torch.zeros(self.outShape, dtype=torch.int64)


def _mockPipe(monkeypatch: pytest.MonkeyPatch, outShape: tuple[int, int] = (224, 224)):
    fake = _FakePipe(outShape)
    captured: dict = {}

    def fakeLoad(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(pipeline, "loadSegEarthPipeline", fakeLoad)
    return fake, captured


# --- loadClassNames ---


def testLoadClassNamesFromListWithSynonyms():
    names, indices = loadClassNames(["building,roof", "road", "water,river"])

    assert names == ["building", "roof", "road", "water", "river"]
    assert indices == [0, 0, 1, 2, 2]


def testLoadClassNamesFromFile(tmp_path):
    path = tmp_path / "classes.txt"
    path.write_text("background\nbareland,barren\ngrass\n\n  \nwater,river\nbuilding,roof,house\n")

    names, indices = loadClassNames(classNamesPath=path)

    assert names == [
        "background",
        "bareland",
        "barren",
        "grass",
        "water",
        "river",
        "building",
        "roof",
        "house",
    ]
    assert indices == [0, 1, 1, 2, 3, 3, 4, 4, 4]


def testLoadClassNamesRejectsBoth():
    with pytest.raises(ValueError):
        loadClassNames(["a"], classNamesPath="x.txt")


def testLoadClassNamesRejectsNeither():
    with pytest.raises(ValueError):
        loadClassNames()


def testLoadClassNamesRejectsEmptyList():
    with pytest.raises(ValueError):
        loadClassNames([])


def testLoadClassNamesRejectsEmptyFile(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n\n  \n")

    with pytest.raises(ValueError):
        loadClassNames(classNamesPath=path)


# --- _resolveClassNamesPath ---


def testResolveClassNamesPathWritesTempFile(monkeypatch):
    pipeline._CLASS_FILE_CACHE.clear()
    p = pipeline._resolveClassNamesPath(["building,roof", "road", "water,river"], None)

    assert Path(p).read_text() == "building,roof\nroad\nwater,river\n"

    # same list -> cached same path
    p2 = pipeline._resolveClassNamesPath(["building,roof", "road", "water,river"], None)
    assert p2 == p


def testResolveClassNamesPathPassesPathThrough():
    p = pipeline._resolveClassNamesPath(None, "some/classes.txt")
    assert p == "some/classes.txt"


def testResolveClassNamesPathNoneGivesDefault():
    assert pipeline._resolveClassNamesPath(None, None) is None


def testResolveClassNamesPathRejectsBoth():
    with pytest.raises(ValueError):
        pipeline._resolveClassNamesPath(["a"], "x.txt")


def testResolveClassNamesPathRejectsEmptyList():
    with pytest.raises(ValueError):
        pipeline._resolveClassNamesPath([], None)


# --- segmentSar plumbing (mocked loadSegEarthPipeline) ---


def testSegmentSarReturnsLabelAtInputResolution(monkeypatch, tmp_path):
    _, captured = _mockPipe(monkeypatch)
    npy = tmp_path / "scene.npy"
    _saveSampleNpy(npy, 64, 80)

    labelMap = segmentSar(npy, classNames=["building,roof", "road"])

    assert isinstance(labelMap, np.ndarray)
    assert labelMap.shape == (64, 80)  # resized back from the 224x224 fake output
    assert labelMap.dtype == np.int64
    assert captured["featupModel"] == "jbu_one"  # default upsampler


def testSegmentSarAcceptsNdarray(monkeypatch):
    _mockPipe(monkeypatch)
    arr = np.random.RandomState(1).rand(48, 48).astype(np.float32) * 5 + 1e-3

    labelMap = segmentSar(arr, classNames=["a", "b"])

    assert labelMap.shape == (48, 48)
    assert labelMap.dtype == np.int64


def testSegmentSarWritesClassNamesListToTempFile(monkeypatch, tmp_path):
    pipeline._CLASS_FILE_CACHE.clear()
    _, captured = _mockPipe(monkeypatch)
    npy = tmp_path / "scene.npy"
    _saveSampleNpy(npy, 32, 32)

    segmentSar(npy, classNames=["building,roof", "road"])

    classesPath = captured["classNamesPath"]
    assert Path(classesPath).read_text() == "building,roof\nroad\n"


def testSegmentSarPassesClassNamesPathThrough(monkeypatch, tmp_path):
    _, captured = _mockPipe(monkeypatch)
    npy = tmp_path / "scene.npy"
    _saveSampleNpy(npy, 32, 32)
    classes = tmp_path / "classes.txt"
    classes.write_text("background\nship,boat\n")

    segmentSar(npy, classNamesPath=classes)

    assert captured["classNamesPath"] == str(classes)


def testSegmentSarDisablesSlideForSmallImage(monkeypatch, tmp_path):
    _, captured = _mockPipe(monkeypatch)
    npy = tmp_path / "scene.npy"
    _saveSampleNpy(npy, 64, 64)  # <= 224

    segmentSar(npy, classNames=["a"], slideCrop=224)

    assert captured["slideCrop"] == 0  # forced to resize path


def testSegmentSarKeepsSlideForLargeImage(monkeypatch, tmp_path):
    _, captured = _mockPipe(monkeypatch)
    npy = tmp_path / "scene.npy"
    _saveSampleNpy(npy, 256, 256)  # > 224

    segmentSar(npy, classNames=["a"], slideCrop=224)

    assert captured["slideCrop"] == 224


def testSegmentSarSlidesLargeImageByDefault(monkeypatch, tmp_path):
    _, captured = _mockPipe(monkeypatch)
    npy = tmp_path / "scene.npy"
    _saveSampleNpy(npy, 256, 300)

    segmentSar(npy, classNames=["a"])

    assert captured["slideCrop"] == 224
    assert captured["slideStride"] == 112


def testSegmentSarForwardsScalingToPreprocess(monkeypatch, tmp_path):
    _mockPipe(monkeypatch)
    seen: dict = {}
    realPreprocess = pipeline.preprocessSar

    def spyPreprocess(amplitude, **kwargs):
        seen.update(kwargs)
        return realPreprocess(amplitude, **kwargs)

    monkeypatch.setattr(pipeline, "preprocessSar", spyPreprocess)
    npy = tmp_path / "scene.npy"
    _saveSampleNpy(npy, 32, 32)

    segmentSar(npy, classNames=["a"])
    assert seen["scaling"] == "linear"
    assert seen["clipPercentile"] == 99.9

    segmentSar(npy, classNames=["a"], scaling="log", clipPercentile=99.0)
    assert seen["scaling"] == "log"
    assert seen["clipPercentile"] == 99.0


def testSegmentSarUpsampleFactorReturnsInputResolution(monkeypatch, tmp_path):
    fake, captured = _mockPipe(monkeypatch, outShape=(256, 320))
    npy = tmp_path / "scene.npy"
    _saveSampleNpy(npy, 128, 160)

    labelMap = segmentSar(npy, classNames=["a"], upsampleFactor=2)

    assert fake.calls[0].size == (320, 256)  # model sees the upsampled image
    assert captured["slideCrop"] == 224  # slide decided on the upsampled size
    assert labelMap.shape == (128, 160)  # but labels come back at input resolution


def testSegmentSarRejectsBadShape(monkeypatch):
    _mockPipe(monkeypatch)
    bad = np.zeros((5,), dtype=np.float32)

    with pytest.raises(ValueError):
        segmentSar(bad, classNames=["a"])


def testSegmentSarRejectsMissingFile(monkeypatch, tmp_path):
    _mockPipe(monkeypatch)

    with pytest.raises(FileNotFoundError):
        segmentSar(tmp_path / "missing.npy", classNames=["a"])


def testSegmentSarRejectsEmptyVocab(monkeypatch, tmp_path):
    _mockPipe(monkeypatch)
    npy = tmp_path / "scene.npy"
    _saveSampleNpy(npy, 32, 32)

    with pytest.raises(ValueError):
        segmentSar(npy, classNames=[])


def testSegmentSarRejectsBothVocabInputs(monkeypatch, tmp_path):
    _mockPipe(monkeypatch)
    npy = tmp_path / "scene.npy"
    _saveSampleNpy(npy, 32, 32)
    classes = tmp_path / "classes.txt"
    classes.write_text("a\n")

    with pytest.raises(ValueError):
        segmentSar(npy, classNames=["a"], classNamesPath=classes)


@pytest.mark.slow
def testSlowSegmentSarReal():
    if not torch.cuda.is_available():
        pytest.skip("real-model end-to-end test needs a GPU")

    labelMap = segmentSar("tests/fixtures/sample.npy", classNamesPath="tests/fixtures/classes.txt")
    assert labelMap.shape == (64, 80)
    assert labelMap.dtype == np.int64
