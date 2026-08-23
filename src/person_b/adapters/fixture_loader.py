"""Offline fixture loader for local development, unit tests, and evaluation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from person_b.errors import FixtureError
from person_b.models import PageContext


class FixtureLoader:
    """Loads offline Banque Misr fixtures without performing live network requests."""

    def __init__(self, fixtures_dir: Optional[str] = None) -> None:
        if fixtures_dir:
            self.base_dir = Path(fixtures_dir)
        else:
            # Look relative to repository root
            candidates = [
                Path("fixtures/live"),
                Path("../fixtures/live"),
                Path("../../fixtures/live"),
                Path(__file__).resolve().parent.parent.parent.parent / "fixtures" / "live",
            ]
            self.base_dir = None
            for c in candidates:
                if c.exists() and c.is_dir():
                    self.base_dir = c
                    break

            if self.base_dir is None:
                self.base_dir = Path("fixtures/live")

        self.manifest_path = self.base_dir / "manifest.json"

    def exists(self) -> bool:
        """Check if fixtures directory exists on disk."""
        return self.base_dir.exists() and self.base_dir.is_dir()

    def get_manifest(self) -> Dict[str, Any]:
        """Read and return manifest.json metadata."""
        if not self.manifest_path.exists():
            raise FixtureError(f"Fixture manifest not found at: {self.manifest_path}")
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise FixtureError(f"Failed to read fixture manifest: {e}", {"path": str(self.manifest_path)})

    def list_pages(self) -> List[Dict[str, Any]]:
        """List all page records in manifest."""
        manifest = self.get_manifest()
        return manifest.get("pages", [])

    def load_text(self, file_name: str) -> PageContext:
        """Load cleaned text (.txt) file into PageContext."""
        path = self.base_dir / file_name
        if not path.exists():
            raise FixtureError(f"Text fixture not found: {file_name}", {"path": str(path)})
        try:
            text = path.read_text(encoding="utf-8")
            return PageContext(
                content=text,
                content_type="text",
                status_code=200,
                metadata={"file_name": file_name, "path": str(path)},
            )
        except Exception as e:
            raise FixtureError(f"Failed to read text fixture {file_name}: {e}", {"path": str(path)})

    def load_html(self, file_name: str) -> PageContext:
        """Load raw HTML (.html) file into PageContext."""
        path = self.base_dir / file_name
        if not path.exists():
            raise FixtureError(f"HTML fixture not found: {file_name}", {"path": str(path)})
        try:
            html = path.read_text(encoding="utf-8")
            return PageContext(
                content=html,
                content_type="html",
                status_code=200,
                metadata={"file_name": file_name, "path": str(path)},
            )
        except Exception as e:
            raise FixtureError(f"Failed to read HTML fixture {file_name}: {e}", {"path": str(path)})

    def load_pdf(self, file_name: str) -> PageContext:
        """Load PDF file as bytes and path into PageContext."""
        path = self.base_dir / file_name
        if not path.exists():
            raise FixtureError(f"PDF fixture not found: {file_name}", {"path": str(path)})
        try:
            pdf_bytes = path.read_bytes()
            return PageContext(
                content="",
                content_type="pdf",
                status_code=200,
                pdf_bytes=pdf_bytes,
                pdf_path=str(path),
                metadata={"file_name": file_name, "path": str(path), "pdf_size_bytes": len(pdf_bytes)},
            )
        except Exception as e:
            raise FixtureError(f"Failed to read PDF fixture {file_name}: {e}", {"path": str(path)})

    def load_page(self, file_name: str) -> PageContext:
        """Load a fixture automatically detecting its type from extension."""
        lower = file_name.lower()
        if lower.endswith(".txt"):
            return self.load_text(file_name)
        elif lower.endswith(".html") or lower.endswith(".htm"):
            return self.load_html(file_name)
        elif lower.endswith(".pdf"):
            return self.load_pdf(file_name)
        else:
            path = self.base_dir / file_name
            if not path.exists():
                raise FixtureError(f"Fixture file not found: {file_name}", {"path": str(path)})
            text = path.read_text(encoding="utf-8", errors="ignore")
            return PageContext(content=text, metadata={"file_name": file_name, "path": str(path)})

    def load_page_by_url(self, url: str, prefer_text: bool = True) -> PageContext:
        """Find page in manifest matching URL and load its PageContext."""
        manifest = self.get_manifest()
        matched = None
        for page in manifest.get("pages", []):
            if page.get("url") == url or page.get("requested_url") == url:
                matched = page
                break

        if not matched:
            raise FixtureError(f"No fixture found for URL: {url}", {"url": url})

        content_type = matched.get("content_type", "html")
        files = matched.get("files", [])
        if not files:
            raise FixtureError(f"Manifest page record has no associated files: {url}")

        chosen_file = None
        if content_type == "pdf":
            for f in files:
                if f.endswith(".pdf"):
                    chosen_file = f
                    break
        elif prefer_text:
            for f in files:
                if f.endswith(".txt"):
                    chosen_file = f
                    break
            if not chosen_file and files:
                chosen_file = files[0]
        else:
            chosen_file = files[0]

        if not chosen_file:
            chosen_file = files[0]

        ctx = self.load_page(chosen_file)
        ctx.source_url = matched.get("url", url)
        ctx.status_code = matched.get("status", 200)
        ctx.content_type = content_type
        ctx.metadata.update({
            "manifest_record": matched,
            "url": matched.get("url"),
        })
        return ctx
