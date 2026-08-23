"""HTTP layer for the Banque Misr agentic research assistant."""

"""Import-time bootstrap: see _bootstrap.py for why this is here."""

import pathlib as _pathlib
import sys as _sys

# Resolved from this file, not the working directory, so the server can be
# launched from anywhere. Guarded so a missing repo root cannot raise at import.
_repo_root = str(_pathlib.Path(__file__).resolve().parent.parent)
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)
try:
    from _bootstrap import ensure_src_on_path as _ensure_src_on_path

    _ensure_src_on_path()
except Exception:  # pragma: no cover - never let a path helper break an import
    pass

