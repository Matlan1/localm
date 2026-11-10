# API reference

Generated from the server's OpenAPI schema on every documentation build.
For behaviour, authentication, scopes and worked examples, read
[HTTP Server API](server-api.md); this page lists every route, parameter and
schema the schema declares. The published site also serves the raw schema as
`openapi.json`; `python scripts/export_openapi.py --out openapi.json` writes it.

The schema comes from the server assembled with no model loaded and only the
always-enabled chat plugin, so the routes of plugins that ship disabled (the
`rag` and `memory` endpoints, for example) and of the GUI are documented in
[HTTP Server API](server-api.md) and not listed here.

A running server serves its own schema (which also lists the routes of the
plugins enabled there) at `/openapi.json`, with interactive
pages at `/docs` and `/redoc`, on a loopback bind only. A server bound to a
network address answers `404` for all of them.

<!-- openapi-reference -->
