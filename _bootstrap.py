"""
Making ``src/person_b`` importable at runtime, without an environment variable.

What it does
    Puts the repository's ``src/`` directory on ``sys.path`` when the app
    starts, so ``import person_b`` works from a bare ``uvicorn api.app:app`` or
    ``python scripts/live_navigate.py`` in a fresh terminal.

Why it is needed
    The vendored intelligence layer lives at ``src/person_b`` and imports
    absolutely as ``person_b.*`` (see ``src/person_b/VENDORED.md``). pytest was
    told about it via ``pythonpath`` in ``pyproject.toml`` -- but that setting
    is pytest's alone. The server knew nothing about it, so the whole suite
    passed while ``uvicorn`` died at the first validate() call with
    ``ModuleNotFoundError: No module named 'person_b'`` unless PYTHONPATH had
    been exported by hand.

    Typing an environment variable before every run is not a deployment step,
    it is a trap. This removes it.

Why here, and why called from several places
    ``ensure_src_on_path()`` is idempotent and cheap, and it is called from
    ``agent/__init__.py`` and ``api/__init__.py`` -- the two packages every
    entry point goes through. Whichever the process imports first, the path is
    set before any ``person_b`` import runs. Doing it in one of them only would
    leave the other broken the moment someone imports it directly.

    It resolves the location from this file rather than from the working
    directory, so it does not matter where the server is launched from.
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"


def ensure_src_on_path() -> bool:
    """Put ``src/`` on ``sys.path`` if it is there. Returns whether it is."""
    if not SRC.is_dir():
        # A checkout without the vendored layer still runs the browsing and
        # navigation halves, so this is not fatal here -- the import that
        # needs it will say so, with a clearer message than a path bug.
        return False
    path = str(SRC)
    if path not in sys.path:
        sys.path.insert(0, path)
    # The repo root too: a process started from elsewhere (a service manager,
    # an IDE run configuration) may not have it, and ``agent``/``api``/
    # ``browsing`` are imported by name throughout.
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.append(root)
    return True
