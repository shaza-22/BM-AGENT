"""Can the app import its own vendored layer, without an environment variable?

Every test in this suite runs under pytest, which is told about ``src/`` by
``pythonpath`` in ``pyproject.toml``. That setting is pytest's alone: the
server knew nothing about it, so the entire suite passed green while
``uvicorn api.app:app`` in a fresh terminal died at the first ``validate()``
call with ``ModuleNotFoundError: No module named 'person_b'``.

So these tests deliberately do **not** import anything themselves. They launch
a **subprocess** with a scrubbed environment -- no ``PYTHONPATH``, no inherited
``sys.path`` -- and import the way a real launch does. A test that imports
normally would pass under the very configuration that hides the bug.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def run_clean(code: str, cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    """Run a snippet with PYTHONPATH removed, as a bare terminal would."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # PYTHONSAFEPATH stops the interpreter adding the script's directory, so
    # nothing is on the path except what the code puts there itself.
    env["PYTHONSAFEPATH"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd or REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestTheAppFindsItsVendoredLayer:
    def test_importing_agent_makes_person_b_importable(self):
        result = run_clean(
            "import sys; sys.path.insert(0, '.');"
            "import agent.loop;"
            "from person_b.api import validate;"
            "print('ok')"
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_importing_api_makes_person_b_importable(self):
        result = run_clean(
            "import sys; sys.path.insert(0, '.');"
            "import api.app;"
            "from person_b.api import validate;"
            "print('ok')"
        )
        assert result.returncode == 0, result.stderr

    def test_the_asgi_app_builds_the_way_uvicorn_builds_it(self):
        """``uvicorn api.app:app`` resolves the attribute after importing.

        This is the exact failure that was reaching the terminal: the import
        succeeded and the ``person_b`` call inside a request blew up later.
        """
        result = run_clean(
            "import sys; sys.path.insert(0, '.');"
            "from api.app import app;"
            "import person_b.api;"
            "print('ok', bool(app))"
        )
        assert result.returncode == 0, result.stderr
        assert "ok True" in result.stdout

    def test_the_loop_can_reach_validate_at_call_time(self):
        """agent/loop.py imports person_b lazily, inside functions.

        A module-level import test would not have caught the original bug,
        because the module imported fine and only the call failed.
        """
        result = run_clean(
            "import sys; sys.path.insert(0, '.');"
            "from agent.loop import ResearchLoop;"
            "from person_b.api import plan_task, validate;"
            "sg = plan_task('what are the fees?').sub_goals[0];"
            "v = validate(sg, 'Fees and charges\\nIssuance EGP 250', source_url='https://x/y');"
            "print('ok', v['resolved'] in (True, False))"
        )
        assert result.returncode == 0, result.stderr
        assert "ok True" in result.stdout

    def test_it_works_when_launched_from_another_directory(self, tmp_path: pathlib.Path):
        """A service manager or IDE may start the process anywhere."""
        result = run_clean(
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
            "import api.app;"
            "from person_b.api import validate;"
            "print('ok')",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr

    def test_a_script_entry_point_works_too(self):
        result = run_clean(
            "import sys; sys.path.insert(0, '.');"
            "import scripts.live_navigate;"
            "from person_b.api import validate;"
            "print('ok')"
        )
        assert result.returncode == 0, result.stderr


class TestTheBootstrapItself:
    def test_it_is_idempotent(self):
        result = run_clean(
            "import sys; sys.path.insert(0, '.');"
            "from _bootstrap import ensure_src_on_path, SRC;"
            "ensure_src_on_path(); ensure_src_on_path(); ensure_src_on_path();"
            "print('ok', sys.path.count(str(SRC)))"
        )
        assert result.returncode == 0, result.stderr
        assert "ok 1" in result.stdout, "src/ was added to sys.path more than once"

    def test_it_does_not_depend_on_the_working_directory(self, tmp_path: pathlib.Path):
        result = run_clean(
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
            "from _bootstrap import ensure_src_on_path, SRC;"
            "print('ok', ensure_src_on_path(), SRC.is_dir())",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert "ok True True" in result.stdout
