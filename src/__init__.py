"""Project-wide setup. Imported before torch in any entry point.

Sets PYTORCH_ENABLE_MPS_FALLBACK=1 by default on macOS so unsupported MPS
ops silently fall back to CPU instead of erroring. Override by exporting
`PYTORCH_ENABLE_MPS_FALLBACK=0` before invoking any script.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
