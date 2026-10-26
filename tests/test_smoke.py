from __future__ import annotations

import alignEarthSarSeg


def testVersionIsSet():
    assert isinstance(alignEarthSarSeg.__version__, str)
    assert alignEarthSarSeg.__version__
