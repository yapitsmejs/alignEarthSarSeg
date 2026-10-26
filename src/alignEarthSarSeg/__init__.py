"""Ingest amplitude/complex SAR images, preprocess, run AlignEarth-SAR open-vocabulary segmentation, and return a label image."""

from __future__ import annotations

# Use the OS native cert store for Hugging Face downloads. On managed/corporate
# hosts a MITM proxy's CA sits in the system store (Windows/macOS/Linux) but not in
# certifi's bundle, which would make httpx (used by huggingface_hub) fail SSL
# verification. Injecting here makes the package work out of the box there; it is a
# safe no-op on standard hosts. Degrades gracefully if truststore is unavailable.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001 - best-effort cert setup, never fatal
    pass

from alignEarthSarSeg.pipeline import loadSegEarthPipeline, segmentSar

__version__ = "0.1.0"

__all__ = ["segmentSar", "loadSegEarthPipeline", "__version__"]
