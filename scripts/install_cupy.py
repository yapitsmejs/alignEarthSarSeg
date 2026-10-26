#!/usr/bin/env python3
"""install_cupy.py — detect the NVIDIA GPU + CUDA toolkit and install the right cupy.

cupy packaging (pip / uv):
  - cupy-cuda{11,12,13}x  : toolkitless — uses the SYSTEM CUDA of that major
                            version (located at runtime via cuda-pathfinder).
                            Install when a matching CUDA toolkit is present.
  - cupy                  : bundled — self-contained, needs only the driver.
                            Install when no CUDA toolkit is present.
  (cupy-core is the conda-forge toolkitless name; not used by this pip/uv project.)

Policy:
  - No NVIDIA GPU            -> install nothing; the pipeline uses the NumPy FFT.
  - GPU + CUDA toolkit       -> cupy-cuda{M}x  (M = toolkit major, e.g. 13 -> cupy-cuda13x)
  - GPU, no toolkit (driver) -> cupy          (bundled, self-contained)

The cupy wheels are mutually exclusive (all install the `cupy` package), so the
script first uninstalls any existing variant, then installs exactly the chosen
one, then verifies the import + device count.

Run from the repo root:  python scripts/install_cupy.py
"""

import os
import re
import shutil
import subprocess
import sys

# cupy wheel variants we manage (all provide the `cupy` package -> mutually exclusive)
_CUPY_VARIANTS = ["cupy", "cupy-core", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x"]
_KNOWN_TOOLKITLESS_MAJORS = (11, 12, 13)


def _run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def detect_gpu():
    """Return (has_nvidia_gpu, driver_max_cuda_version_str) via nvidia-smi.

    The "CUDA Version: X.Y" field nvidia-smi prints is the max CUDA version the
    installed driver supports.
    """
    if shutil.which("nvidia-smi") is None:
        return False, None
    r = _run(["nvidia-smi"])
    if r is None or r.returncode != 0:
        return False, None
    m = re.search(r"CUDA Version:\s*([\d.]+)", r.stdout)
    return True, (m.group(1) if m else None)


def detect_toolkit():
    """Return (has_toolkit, toolkit_version_str).

    A "CUDA toolkit" means nvcc is findable (on PATH, or under CUDA_HOME/CUDA_PATH).
    """
    nvcc = shutil.which("nvcc")
    if nvcc:
        r = _run([nvcc, "--version"])
        if r is not None:
            m = re.search(r"release\s+([\d.]+)", r.stdout)
            return True, (m.group(1) if m else None)
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home and os.path.isfile(os.path.join(home, "bin", "nvcc")):
        return True, None
    return False, None


def pick_wheel(has_gpu, gpu_cuda, has_toolkit, toolkit_ver):
    """Decide which wheel to install, or None to install nothing."""
    if not has_gpu:
        return None
    if has_toolkit:
        # toolkitless wheel matching the system CUDA major (fall back to driver ver)
        ver_str = toolkit_ver or gpu_cuda
        try:
            major = int((ver_str or "").split(".")[0])
        except ValueError:
            major = None
        if major in _KNOWN_TOOLKITLESS_MAJORS:
            return f"cupy-cuda{major}x"
        # unknown major: fall through to the bundled wheel
    # no toolkit (driver only), or toolkit major with no published wheel -> bundled
    return "cupy"


def _installer_base():
    """Prefer uv (this is a uv-managed project) if on PATH, else python -m pip."""
    if shutil.which("uv") is not None:
        # --native-tls lets uv use the system certificate store, required behind a
        # corporate TLS-intercepting proxy (otherwise: "invalid peer certificate").
        return ["uv", "pip", "--native-tls"]
    return [sys.executable, "-m", "pip"]


def _run_installer(action, packages):
    base = _installer_base() + [action]
    extra = []
    if action == "uninstall" and base[0] != "uv":
        # uv pip uninstall is non-interactive (no -y); pip uninstall needs -y.
        extra = ["-y"]
    cmd = base + extra + list(packages)
    print("  $ " + " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def main():
    print("Detecting NVIDIA GPU and CUDA toolkit ...")
    has_gpu, gpu_cuda = detect_gpu()
    has_toolkit, toolkit_ver = detect_toolkit()
    print(f"  NVIDIA GPU       : {'yes' if has_gpu else 'no'}")
    print(f"  driver max CUDA  : {gpu_cuda or '(n/a)'}")
    toolkit_str = (
        ("yes (" + toolkit_ver + ")")
        if has_toolkit and toolkit_ver
        else ("yes" if has_toolkit else "no")
    )
    print(f"  CUDA toolkit     : {toolkit_str}")

    wheel = pick_wheel(has_gpu, gpu_cuda, has_toolkit, toolkit_ver)
    if wheel is None:
        print("\nNo NVIDIA GPU detected — not installing cupy.")
        print("The pipeline will use the NumPy")
        return 0

    print(f"\nSelected cupy wheel: {wheel}")
    print("Removing any existing cupy variant to avoid conflicts ...")
    _run_installer("uninstall", _CUPY_VARIANTS)  # ignore failures (not-installed)

    print(f"Installing {wheel} ...")
    if not _run_installer("install", [wheel]):
        print(f"\nERROR: install of {wheel} failed.")
        return 1

    print("\nVerifying cupy import + visible CUDA devices ...")
    try:
        import cupy as cp

        n = cp.cuda.runtime.getDeviceCount()
        print(f"  cupy {cp.__version__} installed; CUDA devices visible: {n}")
        if n > 0:
            print("  -> GPU FFT path is active.")
        else:
            print(
                "  -> WARNING: cupy installed but no CUDA device visible; "
                "NumPy fallback will be used."
            )
    except Exception as e:
        print(f"  -> ERROR: cupy import failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
