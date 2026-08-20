"""Tests for the demo-block argument handling in ``browsing/_demo.py``.

Regression origin: ``python -m browsing.extract_links fixtures/live/index.html``
reported "no fixtures" for a file that existed, because the argument was only
ever globbed as a directory.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from browsing._demo import FixtureNotFound, base_url_for, resolve_fixture_paths

SEED = "https://www.banquemisr.com/"


@pytest.fixture
def live_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A stand-in for fixtures/live, named the way save_fixtures.py names files."""
    directory = tmp_path / "live"
    directory.mkdir()
    (directory / "index.html").write_text("<html><body>home</body></html>", encoding="utf-8")
    (directory / "home-pages-fees.html").write_text("<html><body>fees</body></html>", encoding="utf-8")
    (directory / "index.txt").write_text("home", encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "pages": [
                    {"url": SEED, "files": ["index.html", "index.txt"]},
                    {
                        "url": "https://www.banquemisr.com/Home/Pages/Fees",
                        "files": ["home-pages-fees.html"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return directory


class TestExplicitFilePath:
    def test_explicit_file_is_read_not_scanned(self, live_dir):
        # The reported bug: this returned nothing because the file was globbed
        # as if it were a directory.
        target = live_dir / "index.html"
        assert resolve_fixture_paths([str(target)], live_dir) == [target]

    def test_explicit_file_works_with_forward_slashes(self, live_dir):
        # pathlib splits "/" natively on Windows too, so a path typed with
        # forward slashes must resolve identically.
        argument = f"{live_dir.as_posix()}/index.html"
        assert resolve_fixture_paths([argument], live_dir) == [pathlib.Path(argument)]

    def test_explicit_file_may_have_any_extension(self, live_dir):
        # A saved fixture is whatever save_fixtures.py wrote; an explicit
        # argument is not second-guessed.
        target = live_dir / "index.txt"
        assert resolve_fixture_paths([str(target)], live_dir) == [target]

    def test_relative_path_resolves_against_the_working_directory(self, live_dir, monkeypatch):
        monkeypatch.chdir(live_dir.parent)
        assert resolve_fixture_paths(["live/index.html"], live_dir) == [
            pathlib.Path("live/index.html")
        ]

    def test_missing_file_names_itself_in_the_error(self, live_dir):
        with pytest.raises(FixtureNotFound, match="nope.html"):
            resolve_fixture_paths([str(live_dir / "nope.html")], live_dir)


class TestDirectoryMode:
    def test_directory_argument_still_scans_for_html(self, live_dir):
        found = resolve_fixture_paths([str(live_dir)], live_dir)
        assert [path.name for path in found] == ["home-pages-fees.html", "index.html"]

    def test_no_arguments_scans_the_default_directory(self, live_dir):
        found = resolve_fixture_paths([], live_dir)
        assert [path.name for path in found] == ["home-pages-fees.html", "index.html"]

    def test_empty_directory_gives_an_actionable_message(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FixtureNotFound, match="save_fixtures.py"):
            resolve_fixture_paths([str(empty)], empty)

    def test_synthetic_fixtures_are_found_by_default(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        found = resolve_fixture_paths([], root / "fixtures" / "synthetic")
        assert "homepage.html" in {path.name for path in found}


class TestGlobAndMultipleArguments:
    def test_unexpanded_glob_is_expanded(self, live_dir):
        # Neither cmd.exe nor PowerShell expands globs, so the pattern arrives
        # verbatim and has to be handled here.
        found = resolve_fixture_paths([f"{live_dir.as_posix()}/*.html"], live_dir)
        assert [path.name for path in found] == ["home-pages-fees.html", "index.html"]

    def test_glob_matching_nothing_raises(self, live_dir):
        with pytest.raises(FixtureNotFound):
            resolve_fixture_paths([f"{live_dir.as_posix()}/*.pdf"], live_dir)

    def test_several_arguments_are_concatenated_in_order(self, live_dir):
        found = resolve_fixture_paths(
            [str(live_dir / "index.html"), str(live_dir / "home-pages-fees.html")], live_dir
        )
        assert [path.name for path in found] == ["index.html", "home-pages-fees.html"]

    def test_repeated_arguments_are_deduplicated(self, live_dir):
        found = resolve_fixture_paths(
            [str(live_dir / "index.html"), str(live_dir), str(live_dir / "index.html")], live_dir
        )
        assert [path.name for path in found] == ["index.html", "home-pages-fees.html"]


class TestBaseUrl:
    def test_url_comes_from_the_manifest(self, live_dir):
        # Relative hrefs resolve against this; using the homepage for a page
        # saved from deeper in the site would produce wrong URLs.
        assert base_url_for(live_dir / "home-pages-fees.html", SEED) == (
            "https://www.banquemisr.com/Home/Pages/Fees"
        )

    def test_homepage_maps_to_the_seed(self, live_dir):
        assert base_url_for(live_dir / "index.html", SEED) == SEED

    def test_default_when_there_is_no_manifest(self, tmp_path):
        page = tmp_path / "page.html"
        page.write_text("<html></html>", encoding="utf-8")
        assert base_url_for(page, SEED) == SEED

    def test_default_when_the_file_is_not_in_the_manifest(self, live_dir):
        stray = live_dir / "stray.html"
        stray.write_text("<html></html>", encoding="utf-8")
        assert base_url_for(stray, SEED) == SEED

    def test_corrupt_manifest_falls_back_instead_of_crashing(self, live_dir):
        (live_dir / "manifest.json").write_text("{not json", encoding="utf-8")
        assert base_url_for(live_dir / "index.html", SEED) == SEED

    def test_unexpected_manifest_shape_falls_back(self, live_dir):
        (live_dir / "manifest.json").write_text(json.dumps({"pages": ["oops"]}), encoding="utf-8")
        assert base_url_for(live_dir / "index.html", SEED) == SEED
