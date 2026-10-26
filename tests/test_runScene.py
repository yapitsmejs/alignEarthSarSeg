from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import alignEarthSarSeg.runScene as runScene
from alignEarthSarSeg.runScene import colorizeLabels, loadSceneArray, plotScene


def _fakeSegment(monkeypatch: pytest.MonkeyPatch, labels: np.ndarray | None = None) -> dict:
    captured: dict = {}

    def fake(image, classNames=None, classNamesPath=None, **kwargs):
        captured["image"] = image
        captured["classNames"] = classNames
        captured["classNamesPath"] = classNamesPath
        captured["kwargs"] = kwargs
        h, w = image.shape
        return (
            labels if labels is not None else (np.arange(h * w).reshape(h, w) % 3).astype(np.int64)
        )

    monkeypatch.setattr(runScene, "segmentSar", fake)
    return captured


# --- loadSceneArray ---


def testLoadSceneArrayNpyKeepsComplex(tmp_path):
    arr = (np.ones((4, 5)) + 1j).astype(np.complex64)
    np.save(tmp_path / "s.npy", arr)

    out = loadSceneArray(tmp_path / "s.npy")

    assert np.iscomplexobj(out)
    assert out.shape == (4, 5)


def testLoadSceneArrayGreyPng(tmp_path):
    grey = (np.arange(20).reshape(4, 5) * 10).astype(np.uint8)
    Image.fromarray(grey, mode="L").save(tmp_path / "s.png")

    out = loadSceneArray(tmp_path / "s.png")

    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, grey.astype(np.float32))


def testLoadSceneArrayRgbPngConvertsToGrey(tmp_path):
    rgb = np.full((4, 5, 3), 100, dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(tmp_path / "s.png")

    out = loadSceneArray(tmp_path / "s.png")

    assert out.shape == (4, 5)
    assert np.all(out == 100)


def testLoadSceneArrayFloatTiff(tmp_path):
    arr = np.linspace(0, 5, 20, dtype=np.float32).reshape(4, 5)
    Image.fromarray(arr, mode="F").save(tmp_path / "s.tif")

    out = loadSceneArray(tmp_path / "s.tif")

    np.testing.assert_allclose(out, arr)


def testLoadSceneArrayMissingFile(tmp_path):
    with pytest.raises(FileNotFoundError):
        loadSceneArray(tmp_path / "nope.png")


# --- colorizeLabels ---


def testColorizeLabelsDistinctColorsAndShape():
    labels = np.array([[0, 1], [2, 0]])

    rgb = colorizeLabels(labels)

    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0, 0], rgb[1, 1])
    assert not np.array_equal(rgb[0, 0], rgb[0, 1])
    assert not np.array_equal(rgb[0, 1], rgb[1, 0])


def testColorizeLabelsCyclesBeyondPalette():
    n = len(runScene.PALETTE)

    rgb = colorizeLabels(np.array([[0, n]]))

    np.testing.assert_array_equal(rgb[0, 0], rgb[0, 1])


# --- plotScene ---


def testPlotSceneOverviewOriginLowerWithLegend():
    grey = np.zeros((6, 8), dtype=np.uint8)
    labels = np.array([[0] * 8] * 3 + [[2] * 8] * 3)

    fig = plotScene(grey, labels, ["water", "road", "tree"], {0: 0.5, 2: 0.5}, title="scene")

    inputAx, labelAx = fig.axes
    assert inputAx.images[0].origin == "lower"
    assert labelAx.images[0].origin == "lower"
    texts = [t.get_text() for t in labelAx.get_legend().get_texts()]
    assert texts == ["0: water (50.0%)", "1: road (0.0%)", "2: tree (50.0%)"]
    colors = [h.get_facecolor()[:3] for h in labelAx.get_legend().legend_handles]
    np.testing.assert_allclose(colors, np.asarray(runScene.PALETTE[:3]) / 255.0)
    assert fig._suptitle.get_text() == "scene"


def testPlotSceneSinglePanels():
    grey = np.zeros((4, 4), dtype=np.uint8)
    labels = np.zeros((4, 4), dtype=np.int64)

    inputOnly = plotScene(grey, None, ["a"], {0: 1.0})
    labelsOnly = plotScene(None, labels, ["a"], {0: 1.0})

    assert len(inputOnly.axes) == 1 and inputOnly.axes[0].get_legend() is None
    assert len(labelsOnly.axes) == 1 and labelsOnly.axes[0].get_legend() is not None
    assert inputOnly._suptitle is None


# --- runSceneFile ---


def testRunSceneFileWritesOutputs(monkeypatch, tmp_path):
    captured = _fakeSegment(monkeypatch)
    np.save(tmp_path / "scene.npy", np.random.default_rng(0).rayleigh(1.0, (30, 40)))
    outDir = tmp_path / "out"

    result = runScene.runSceneFile(
        tmp_path / "scene.npy", outDir, classNames=["building,roof", "road", "water"]
    )

    assert result["labels"].shape == (30, 40)
    for key in ("labelsNpy", "labelsPng", "inputPng", "overviewPng"):
        assert Path(result[key]).exists()
    np.testing.assert_array_equal(np.load(result["labelsNpy"]), result["labels"])
    assert result["legend"] == ["building", "road", "water"]
    assert result["pixelAccuracy"] is None
    assert captured["classNames"] == ["building,roof", "road", "water"]
    assert sum(result["classFractions"].values()) == pytest.approx(1.0)


def testRunSceneFileForwardsSegmentOptions(monkeypatch, tmp_path):
    captured = _fakeSegment(monkeypatch)
    np.save(tmp_path / "scene.npy", np.ones((8, 8), dtype=np.float32))

    runScene.runSceneFile(
        tmp_path / "scene.npy",
        tmp_path / "out",
        classNames=["a"],
        upsampler="bilinear",
        probThd=0.3,
        scaling="log",
        clipPercentile=99.0,
    )

    assert captured["kwargs"]["upsampler"] == "bilinear"
    assert captured["kwargs"]["probThd"] == 0.3
    assert captured["kwargs"]["scaling"] == "log"
    assert captured["kwargs"]["clipPercentile"] == 99.0


def testRunSceneFileLegendFromClassFile(monkeypatch, tmp_path):
    captured = _fakeSegment(monkeypatch)
    np.save(tmp_path / "scene.npy", np.ones((8, 8), dtype=np.float32))
    classes = tmp_path / "classes.txt"
    classes.write_text("background\nship,boat\n")

    result = runScene.runSceneFile(tmp_path / "scene.npy", tmp_path / "out", classNamesPath=classes)

    assert result["legend"] == ["background", "ship"]
    assert captured["classNamesPath"] == classes


def testRunSceneFileDefaultVocabHasGenericLegend(monkeypatch, tmp_path):
    _fakeSegment(monkeypatch)
    np.save(tmp_path / "scene.npy", np.ones((8, 8), dtype=np.float32))

    result = runScene.runSceneFile(tmp_path / "scene.npy", tmp_path / "out")

    assert result["legend"] == ["class 0", "class 1", "class 2"]


def testRunSceneFileScoresGroundTruth(monkeypatch, tmp_path):
    pred = np.array([[0, 1], [1, 1]], dtype=np.int64)
    _fakeSegment(monkeypatch, labels=pred)
    np.save(tmp_path / "scene.npy", np.ones((2, 2), dtype=np.float32))
    Image.fromarray(np.array([[0, 1], [0, 1]], dtype=np.uint8), mode="L").save(tmp_path / "gt.png")

    result = runScene.runSceneFile(
        tmp_path / "scene.npy",
        tmp_path / "out",
        classNames=["a", "b"],
        groundTruth=tmp_path / "gt.png",
    )

    assert result["pixelAccuracy"] == pytest.approx(0.75)


def testRunSceneFileRejectsGroundTruthShapeMismatch(monkeypatch, tmp_path):
    _fakeSegment(monkeypatch)
    np.save(tmp_path / "scene.npy", np.ones((4, 4), dtype=np.float32))
    np.save(tmp_path / "gt.npy", np.zeros((3, 3), dtype=np.int64))

    with pytest.raises(ValueError):
        runScene.runSceneFile(
            tmp_path / "scene.npy",
            tmp_path / "out",
            classNames=["a"],
            groundTruth=tmp_path / "gt.npy",
        )


# --- CLI ---


def testMainParsesArgsAndRuns(monkeypatch, tmp_path, capsys):
    captured = _fakeSegment(monkeypatch)
    np.save(tmp_path / "scene.npy", np.ones((8, 8), dtype=np.float32))

    rc = runScene.main(
        [
            str(tmp_path / "scene.npy"),
            "--out-dir",
            str(tmp_path / "out"),
            "--class",
            "building,roof",
            "--class",
            "road",
            "--upsampler",
            "bilinear",
            "--prob-thd",
            "0.3",
            "--slide-crop",
            "0",
            "--device",
            "cpu",
        ]
    )

    assert rc == 0
    assert captured["classNames"] == ["building,roof", "road"]
    assert captured["kwargs"]["slideCrop"] == 0
    assert captured["kwargs"]["device"] == "cpu"
    out = capsys.readouterr().out
    assert "overview" in out and "building" in out


def testMainDefaultOutDirNextToImageAndPrintsAccuracy(monkeypatch, tmp_path, capsys):
    _fakeSegment(monkeypatch, labels=np.zeros((8, 8), dtype=np.int64))
    np.save(tmp_path / "scene.npy", np.ones((8, 8), dtype=np.float32))
    np.save(tmp_path / "gt.npy", np.zeros((8, 8), dtype=np.int64))

    runScene.main(
        [str(tmp_path / "scene.npy"), "--class", "a", "--ground-truth", str(tmp_path / "gt.npy")]
    )

    assert (tmp_path / "scene_seg" / "scene_overview.png").exists()
    assert "pixel accuracy vs ground truth: 100.00%" in capsys.readouterr().out


def testRunSceneFileResizesPreviewWhenUpsampled(monkeypatch, tmp_path):
    _fakeSegment(monkeypatch, labels=np.zeros((8, 8), dtype=np.int64))
    np.save(tmp_path / "scene.npy", np.ones((8, 8), dtype=np.float32))
    plottedShapes: list[tuple[int, ...]] = []
    realPlot = runScene.plotScene

    def spyPlot(inputGrey, labels, *args, **kwargs):
        if inputGrey is not None:
            plottedShapes.append(inputGrey.shape)
        return realPlot(inputGrey, labels, *args, **kwargs)

    monkeypatch.setattr(runScene, "plotScene", spyPlot)

    runScene.runSceneFile(
        tmp_path / "scene.npy", tmp_path / "out", classNames=["a"], upsampleFactor=2
    )

    assert plottedShapes == [(8, 8), (8, 8)]  # 16x16 upsampled preview resized to labels
