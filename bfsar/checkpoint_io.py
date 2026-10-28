"""Write a finished checkpoint to the path later stages read."""

from __future__ import annotations

import os
import shutil


def publish_checkpoint(src, dest) -> str:
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.islink(dest):
        os.remove(dest)
    if src != dest:
        shutil.copy2(src, dest)
    return dest
