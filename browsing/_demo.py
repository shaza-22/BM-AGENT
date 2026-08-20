"""
Support for the ``__main__`` inspection blocks in ``fetcher`` and
``extract_links``.

What it does
    Resolves the command-line arguments of those demo blocks into a concrete
    list of fixture files, and works out which URL each fixture was originally
    fetched from.

Inputs
    ``resolve_fixture_paths(args, default_dir)`` -- the raw ``sys.argv[1:]``
    (files, directories, or glob patterns) and the directory to scan when no
    arguments are given.
    ``base_url_for(path, default)`` -- a fixture file.

Outputs
    ``resolve_fixture_paths`` -> an ordered, deduplicated ``list[Path]``, or
    raises :class:`FixtureNotFound` with a message naming what was missing.
    ``base_url_for`` -> the URL recorded for that file in the sibling
    ``manifest.json``, else the supplied default.

Why it is needed
    The demo blocks are how the parsing layers get exercised by hand against
    saved pages, so they have to accept what someone actually types: one file,
    a directory, or a glob that the shell did not expand (cmd.exe and
    PowerShell do not). Keeping this here rather than inline makes it testable,
    and keeps it out of the agent-facing API -- nothing in the browsing layer
    proper imports this module.
"""

from __future__ import annotations

import glob as globlib
import json
import pathlib
from typing import Sequence


class FixtureNotFound(Exception):
    """Raised when an argument names nothing readable."""


def _html_files(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in directory.glob("*.html") if path.is_file())


def resolve_fixture_paths(
    args: Sequence[str], default_dir: pathlib.Path
) -> list[pathlib.Path]:
    """Turn demo-block arguments into the files to read.

    Accepts, in this order of preference per argument: an existing directory
    (scanned for ``*.html``), an existing file (used as-is, whatever its
    extension), or a glob pattern. Forward slashes work on every platform --
    ``pathlib`` splits them natively on Windows too.
    """
    if not args:
        found = _html_files(default_dir)
        if not found:
            raise FixtureNotFound(
                f"no *.html files in {default_dir} -- run scripts/save_fixtures.py first"
            )
        return found

    resolved: list[pathlib.Path] = []
    for arg in args:
        path = pathlib.Path(arg).expanduser()
        if path.is_dir():
            found = _html_files(path)
            if not found:
                raise FixtureNotFound(
                    f"no *.html files in directory {path} -- run scripts/save_fixtures.py first"
                )
            resolved.extend(found)
        elif path.is_file():
            # An explicit file is read as given: a saved fixture may be named
            # anything, and it is not this helper's business to second-guess it.
            resolved.append(path)
        else:
            # Neither cmd.exe nor PowerShell expands globs, so a pattern that
            # reaches us unexpanded still has to work.
            matches = [
                pathlib.Path(match)
                for match in sorted(globlib.glob(str(path), recursive=True))
            ]
            matches = [match for match in matches if match.is_file()]
            if not matches:
                raise FixtureNotFound(f"no such file, directory or match: {path}")
            resolved.extend(matches)

    ordered: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for path in resolved:
        try:
            identity = path.resolve()
        except OSError:  # pragma: no cover - unreadable path on some platforms
            identity = path
        if identity not in seen:
            seen.add(identity)
            ordered.append(path)
    return ordered


def base_url_for(path: pathlib.Path, default: str) -> str:
    """The URL a fixture was saved from, per its sibling ``manifest.json``.

    Relative hrefs resolve against this, so using the homepage for every
    fixture would silently produce wrong URLs for pages saved from deeper in
    the site. Falls back to ``default`` when there is no manifest -- the
    hand-written fixtures in ``fixtures/synthetic/`` have none.
    """
    manifest = path.parent / "manifest.json"
    if not manifest.is_file():
        return default
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
    for record in data.get("pages", []):
        if not isinstance(record, dict):
            continue
        if path.name in (record.get("files") or []) and record.get("url"):
            return str(record["url"])
    return default
