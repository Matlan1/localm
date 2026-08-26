# SPDX-License-Identifier: AGPL-3.0-or-later
"""A bare `localm serve` (api mode) should not 404 at /.

create_app builds the inference app with no root route; the GUI's own "/"
handler is only added when the GUI surface is attached, which `serve` never
does. With api_landing=True the root redirects to the auto-generated /docs.
"""

import unittest

from fastapi.testclient import TestClient

from localm.inference.http_server import create_app
from tests.test_http_server import _make_mock_engine


class TestApiLandingRoute(unittest.TestCase):
    def test_root_redirects_to_docs_when_api_landing(self):
        app = create_app(_make_mock_engine(loaded=True), api_landing=True)
        client = TestClient(app)
        r = client.get("/", follow_redirects=False)
        self.assertEqual(r.status_code, 307)
        self.assertTrue(r.headers["location"].endswith("/docs"))
        # The redirect target exists (FastAPI serves /docs by default).
        self.assertEqual(client.get("/docs").status_code, 200)

    def test_root_is_404_without_api_landing(self):
        # Default construction (e.g. when mounted under the GUI) must NOT add
        # the redirect; the GUI's own "/" shell owns the root path.
        app = create_app(_make_mock_engine(loaded=True))
        client = TestClient(app)
        self.assertEqual(client.get("/", follow_redirects=False).status_code, 404)


if __name__ == "__main__":
    unittest.main()
