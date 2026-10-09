"""MkDocs hooks: fill the API reference page from the live app's OpenAPI schema.

``docs/api-reference.md`` holds the introduction and the ``MARKER`` line. On
every build the hook assembles the app without a model (export_openapi),
replaces ``MARKER`` with the rendered schema, and publishes the raw schema as
``openapi.json`` next to the page.
"""

import json
import sys
from pathlib import Path

from mkdocs.structure.files import File

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_openapi  # noqa: E402
import openapi_markdown  # noqa: E402

MARKER = "<!-- openapi-reference -->"
PAGE = "api-reference.md"
SCHEMA_FILE = "openapi.json"

_schema: dict | None = None


def _get_schema() -> dict:
    global _schema
    if _schema is None:
        _schema = export_openapi.build_schema()
    return _schema


def on_files(files, config):
    """Add the raw schema to the site as ``openapi.json``."""
    text = json.dumps(_get_schema(), indent=2, sort_keys=True) + "\n"
    files.append(File.generated(config, SCHEMA_FILE, content=text))
    return files


def on_page_markdown(markdown, page, config, files):
    """Replace the marker on the API reference page with the rendered schema."""
    if page.file.src_uri != PAGE:
        return markdown
    if MARKER not in markdown:
        raise RuntimeError(f"{PAGE} must contain the line {MARKER}")
    return markdown.replace(MARKER, openapi_markdown.render(_get_schema()))
