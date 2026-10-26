from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import CLIPModel

import alignEarthSarSeg.segment as segment


def testResolveDeviceAutoCpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert segment._resolveDevice(None) == torch.device("cpu")


def testResolveDeviceExplicit():
    assert segment._resolveDevice("cpu") == torch.device("cpu")
    assert segment._resolveDevice(torch.device("cpu")) == torch.device("cpu")


def testApplyTextFeaturesShimWrapsAndExtractsPoolerOutput(monkeypatch):
    def fakeOriginal(self, *args, **kwargs):
        return SimpleNamespace(pooler_output=torch.tensor([1.0, 2.0]))

    monkeypatch.setattr(CLIPModel, "get_text_features", fakeOriginal)
    monkeypatch.setattr(segment, "_SHIM_APPLIED", False)

    segment._applyTextFeaturesShim()
    out = CLIPModel.get_text_features(object())

    assert torch.equal(out, torch.tensor([1.0, 2.0]))  # pooler_output extracted -> tensor


def testApplyTextFeaturesShimIdempotent(monkeypatch):
    monkeypatch.setattr(segment, "_SHIM_APPLIED", True)
    before = CLIPModel.get_text_features

    segment._applyTextFeaturesShim()

    assert CLIPModel.get_text_features is before  # not re-wrapped


def testLoadSegEarthPipelineCachesByConfig(monkeypatch):
    segment._PIPELINE_CACHE.clear()
    constructed: list[object] = []

    class FakePipe:
        def __init__(self) -> None:
            constructed.append(self)

    class FakeRef:
        @staticmethod
        def SegEarthPipeline(*args, **kwargs):
            return FakePipe()

    monkeypatch.setattr(segment, "snapshot_download", lambda p: "/fake/snap")
    monkeypatch.setattr(segment, "_importRefPipeline", lambda snap: FakeRef)
    monkeypatch.setattr(segment, "_applyTextFeaturesShim", lambda: None)

    p1 = segment.loadSegEarthPipeline(featupModel="jbu_one")
    p2 = segment.loadSegEarthPipeline(featupModel="jbu_one")  # same key -> cached
    p3 = segment.loadSegEarthPipeline(featupModel="bilinear")  # diff key -> new

    assert p1 is p2
    assert p1 is not p3
    assert len(constructed) == 2  # p2 reused p1; only p1 and p3 constructed


def testLoadSegEarthPipelinePassesParams(monkeypatch):
    segment._PIPELINE_CACHE.clear()
    captured: dict = {}

    class FakeRef:
        @staticmethod
        def SegEarthPipeline(snapDir, **kwargs):
            captured["snapDir"] = snapDir
            captured["kwargs"] = kwargs
            return object()

    monkeypatch.setattr(segment, "snapshot_download", lambda p: "/fake/snap")
    monkeypatch.setattr(segment, "_importRefPipeline", lambda snap: FakeRef)
    monkeypatch.setattr(segment, "_applyTextFeaturesShim", lambda: None)

    segment.loadSegEarthPipeline(
        featupModel="jbu_stack", classNamesPath="classes.txt", slideCrop=224, clsTokenLambda=-0.5
    )

    assert captured["kwargs"]["featup_model"] == "jbu_stack"
    assert captured["kwargs"]["class_names_path"] == "classes.txt"
    assert captured["kwargs"]["slide_crop"] == 224
    assert captured["kwargs"]["cls_token_lambda"] == -0.5
    assert captured["snapDir"] == "/fake/snap"


def testLoadSegEarthPipelineUsesLocalSnapshotDirWithoutDownload(monkeypatch, tmp_path):
    segment._PIPELINE_CACHE.clear()
    captured: dict = {}

    class FakeRef:
        @staticmethod
        def SegEarthPipeline(snapDir, **kwargs):
            captured["snapDir"] = snapDir
            return object()

    def noDownload(p):
        raise AssertionError("snapshot_download must not be called for a local dir")

    monkeypatch.setattr(segment, "snapshot_download", noDownload)
    monkeypatch.setattr(segment, "_importRefPipeline", lambda snap: FakeRef)
    monkeypatch.setattr(segment, "_applyTextFeaturesShim", lambda: None)

    segment.loadSegEarthPipeline(modelPath=tmp_path)

    assert captured["snapDir"] == str(tmp_path)


def testLoadSegEarthPipelineDefaultsclassNamesPathToNone(monkeypatch):
    segment._PIPELINE_CACHE.clear()
    captured: dict = {}

    class FakeRef:
        @staticmethod
        def SegEarthPipeline(snapDir, **kwargs):
            captured["kwargs"] = kwargs
            return object()

    monkeypatch.setattr(segment, "snapshot_download", lambda p: "/fake/snap")
    monkeypatch.setattr(segment, "_importRefPipeline", lambda snap: FakeRef)
    monkeypatch.setattr(segment, "_applyTextFeaturesShim", lambda: None)

    segment.loadSegEarthPipeline()  # no classNamesPath

    assert captured["kwargs"]["class_names_path"] is None  # reference uses its default
