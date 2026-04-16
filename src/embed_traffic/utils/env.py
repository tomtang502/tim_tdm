"""Environment helpers.

Ensures NVIDIA CUDA libraries bundled inside `site-packages/nvidia/*/lib/` are
discoverable at runtime when the conda env isn't configured to prepend them to
LD_LIBRARY_PATH. Safe to call multiple times.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


def setup_nvidia_libs() -> None:
    """Prepend NVIDIA CUDA lib dirs in site-packages to LD_LIBRARY_PATH.

    This is a no-op if no NVIDIA packages are installed in the current Python
    environment. On systems where the conda env already handles this, calling
    it is harmless (just duplicates entries).
    """
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"

    # Candidate site-packages roots. Try conda env first, then ~/.local.
    candidates = []
    for prefix in (sys.prefix, os.path.expanduser("~/.local")):
        candidates.append(Path(prefix) / "lib" / py_ver / "site-packages" / "nvidia")

    nvidia_lib_dirs: list[str] = []
    for base in candidates:
        if base.exists():
            nvidia_lib_dirs.extend(
                str(p) for p in sorted(base.glob("*/lib")) if p.is_dir()
            )

    if not nvidia_lib_dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    new = ":".join(nvidia_lib_dirs + ([current] if current else []))
    os.environ["LD_LIBRARY_PATH"] = new
