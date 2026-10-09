"""Write the server's OpenAPI schema to a JSON file, without loading a model.

Assembles the HTTP app with no engine, in a throwaway LOCALM_HOME so no
installed plugin, user config or session data reaches the schema, and writes
``app.openapi()`` to ``--out``. The schema therefore describes the kernel
routes plus the built-in plugins only.

Usage:
    python scripts/export_openapi.py --out openapi.json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_schema() -> dict:
    """Return the OpenAPI schema of an engine-less app.

    Sets LOCALM_HOME to a fresh temporary directory before localm is imported,
    removes it again before returning, stamps ``info.version`` with the
    installed localm version, and raises if the schema has no paths.
    """
    home = tempfile.mkdtemp(prefix="localm-openapi-")
    previous = os.environ.get("LOCALM_HOME")
    os.environ["LOCALM_HOME"] = home
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import localm
        from localm.inference import http_server

        try:
            schema = http_server.create_app(None).openapi()
            schema["info"]["version"] = localm.__version__
        finally:
            audit = getattr(http_server, "_audit", None)
            if audit is not None:
                audit.close()
    finally:
        if previous is None:
            os.environ.pop("LOCALM_HOME", None)
        else:
            os.environ["LOCALM_HOME"] = previous
        shutil.rmtree(home)
    if not schema.get("paths"):
        raise RuntimeError("the exported OpenAPI schema has no paths")
    return schema


def write_schema(out: Path) -> int:
    """Write the schema to *out* and return the number of paths it documents."""
    schema = build_schema()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(schema["paths"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("openapi.json"),
                        help="file to write (default: ./openapi.json)")
    args = parser.parse_args(argv)
    count = write_schema(args.out)
    print(f"wrote {args.out} ({count} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
