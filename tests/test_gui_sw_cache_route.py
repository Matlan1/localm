# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm/plugins/gui/web.py's server-computed sw.js CACHE value.

GET /sw.js computes CACHE fresh on every request from a content digest of the
actual static assets being served, so nothing is checked into git for two
branches to conflict over and nothing needs a human to remember to bump. See
tests/test_check_hygiene_sw_cache.py for what the hygiene check still enforces:
SHELL precache coverage and the placeholder line's shape.
"""

import logging

import pytest
from fastapi import HTTPException

from localm.plugins.gui import web


@pytest.fixture()
def fake_static(tmp_path, monkeypatch):
    """A static dir shaped like the real one: sw.js with the real placeholder
    shape, a couple of cacheable assets, and one path under the never-cached
    prefixes sw.js's own fetch handler returns early on."""
    static = tmp_path / "static"
    (static / "app").mkdir(parents=True)
    (static / "api").mkdir(parents=True)
    (static / "sw.js").write_text(
        'const CACHE = "localm-shell-dev";\n// rest of sw.js\n', encoding="utf-8")
    (static / "index.html").write_text("<html></html>", encoding="utf-8")
    (static / "app" / "main.js").write_text("// main v1\n", encoding="utf-8")
    (static / "api" / "thing.json").write_text('{"never":"cached"}', encoding="utf-8")
    monkeypatch.setattr(web, "STATIC_DIR", static)
    return static


class TestDeterminism:
    def test_repeated_calls_on_unchanged_tree_are_identical(self, fake_static):
        v1 = web._compute_sw_cache_value()
        v2 = web._compute_sw_cache_value()
        assert v1 == v2

    def test_response_bytes_are_identical_across_repeated_requests(self, fake_static):
        r1 = web._sw_js_response()
        r2 = web._sw_js_response()
        assert r1.body == r2.body

    def test_etag_is_identical_across_repeated_requests(self, fake_static):
        """The ETag is what actually gates the 304 in TestConditionalGet - a body
        that is byte-identical across requests proves nothing about caching
        efficiency on its own if the ETag it is hashed into varied per request
        (e.g. from something non-deterministic sneaking into the digest). Asserted
        explicitly and separately from body-identity so a regression here cannot
        hide behind that other, weaker assertion."""
        etags = {web._sw_js_response().headers.get("etag") for _ in range(3)}
        assert len(etags) == 1
        assert next(iter(etags))


class TestChangeDetection:
    def test_editing_a_cacheable_file_changes_the_value(self, fake_static):
        before = web._compute_sw_cache_value()
        (fake_static / "app" / "main.js").write_text("// main v2\n", encoding="utf-8")
        after = web._compute_sw_cache_value()
        assert before != after

    def test_adding_a_new_cacheable_file_changes_the_value(self, fake_static):
        before = web._compute_sw_cache_value()
        (fake_static / "app" / "new.js").write_text("// new\n", encoding="utf-8")
        after = web._compute_sw_cache_value()
        assert before != after

    def test_removing_a_cacheable_file_changes_the_value(self, fake_static):
        before = web._compute_sw_cache_value()
        (fake_static / "index.html").unlink()
        after = web._compute_sw_cache_value()
        assert before != after

    def test_renaming_a_file_changes_the_value(self, fake_static):
        """Same bytes, different path: still a real change to what a client sees
        at each URL, so it must not hash identically to the pre-rename state -
        the digest includes each file's path, not just its content."""
        before = web._compute_sw_cache_value()
        content = (fake_static / "index.html").read_bytes()
        (fake_static / "index.html").unlink()
        (fake_static / "index2.html").write_bytes(content)
        after = web._compute_sw_cache_value()
        assert before != after

    def test_editing_an_uncached_path_does_not_change_the_value(self, fake_static):
        """Mirrors sw.js's own fetch handler, which never caches /api/* (and /v1,
        /plugins, /localm-ca.crt) - a change there is correctly outside this
        digest's business."""
        before = web._compute_sw_cache_value()
        (fake_static / "api" / "thing.json").write_text('{"changed":true}',
                                                         encoding="utf-8")
        after = web._compute_sw_cache_value()
        assert before == after

    def test_editing_sw_js_itself_does_not_change_the_value(self, fake_static):
        """sw.js gates the others and cannot gate itself: its own bytes changing
        already triggers the browser's independent byte-diff reinstall check, and
        folding it into its OWN digest would only make the served value depend on
        comment/formatting churn unrelated to what a client has cached."""
        before = web._compute_sw_cache_value()
        (fake_static / "sw.js").write_text(
            'const CACHE = "localm-shell-dev";\n// a harmless comment edit\n',
            encoding="utf-8")
        after = web._compute_sw_cache_value()
        assert before == after


class TestMissingFileIsSafeNotSilent:
    """A file listed a moment ago but gone by read time (a genuine TOCTOU window,
    not simulated via chmod - Windows does not enforce POSIX-style unreadable-but-
    present permissions the way this failure mode needs) must still force a
    cache-bust rather than either crashing the request or silently serving a
    digest that looks unchanged."""

    def _with_a_ghost_file(self, fake_static, monkeypatch):
        ghost = fake_static / "app" / "vanished.js"   # never created
        orig = web._sw_cacheable_files
        monkeypatch.setattr(
            web, "_sw_cacheable_files",
            lambda: sorted(orig() + [ghost],
                           key=lambda p: p.relative_to(web.STATIC_DIR).as_posix()))
        return ghost

    def test_forces_a_bust_and_logs(self, fake_static, monkeypatch, caplog):
        before = web._compute_sw_cache_value()
        self._with_a_ghost_file(fake_static, monkeypatch)
        with caplog.at_level(logging.WARNING, logger="localm"):
            after = web._compute_sw_cache_value()

        assert after != before, "a missing file must still force a cache-bust"
        assert any("vanished.js" in r.message for r in caplog.records), \
            "the read failure must be logged, not silent (AGENTS.md rule 5)"

    def test_route_still_returns_200(self, fake_static, monkeypatch):
        self._with_a_ghost_file(fake_static, monkeypatch)
        resp = web._sw_js_response()
        assert resp.status_code == 200


class TestSubstitution:
    def test_served_bytes_never_contain_the_placeholder(self, fake_static):
        resp = web._sw_js_response()
        assert b"localm-shell-dev" not in resp.body

    def test_served_bytes_contain_the_computed_value(self, fake_static):
        value = web._compute_sw_cache_value()
        resp = web._sw_js_response()
        assert value.encode("utf-8") in resp.body

    def test_media_type_and_cache_control(self, fake_static):
        resp = web._sw_js_response()
        assert resp.media_type == "text/javascript"
        assert resp.headers.get("content-type") == "text/javascript; charset=utf-8"
        assert resp.headers.get("cache-control") == "no-cache"

    def test_unparseable_source_fails_loud_not_silent(self, fake_static):
        (fake_static / "sw.js").write_text("const CACHE = 'wrong-quotes';\n",
                                           encoding="utf-8")
        with pytest.raises(HTTPException) as ei:
            web._sw_js_response()
        assert ei.value.status_code == 500

    def test_two_matching_lines_fails_loud_not_a_silent_wrong_substitution(
            self, fake_static):
        """re.subn(..., count=1) reports n=1 whether the pattern matched once or
        several times (it caps how many it REPLACES, not how many it FOUND), so
        checking that return count alone cannot detect a second match - it would
        silently substitute the FIRST occurrence (e.g. an illustrative example
        string in a future comment) and leave the REAL const CACHE line holding
        the literal placeholder text, reporting success while quietly defeating
        the whole mechanism. Two matches must fail loud, never resolve to one
        of them by accident."""
        (fake_static / "sw.js").write_text(
            'const CACHE = "localm-shell-dev";\n'
            '// example: const CACHE = "some-other-value";\n',
            encoding="utf-8")
        with pytest.raises(HTTPException) as ei:
            web._sw_js_response()
        assert ei.value.status_code == 500
        assert "matched 2 times" in str(ei.value.detail)


class TestConditionalGet:
    """_RevalidatingStatic's own contract for every other asset: no-cache does
    not mean no caching, an unchanged file still gets a cheap 304 via ETag.
    Routing /sw.js around that mount must not silently drop it."""

    def test_matching_if_none_match_returns_304_with_no_body(self, fake_static):
        first = web._sw_js_response()
        etag = first.headers.get("etag")
        assert etag

        second = web._sw_js_response(if_none_match=etag)

        assert second.status_code == 304
        assert second.body == b""

    def test_stale_if_none_match_returns_full_body(self, fake_static):
        first = web._sw_js_response()
        etag = first.headers.get("etag")
        (fake_static / "app" / "main.js").write_text("// changed\n", encoding="utf-8")

        second = web._sw_js_response(if_none_match=etag)

        assert second.status_code == 200
        assert second.body != b""
        assert second.headers.get("etag") != etag

    def test_etag_reflects_sw_js_own_logic_changing_even_when_no_watched_asset_did(
            self, fake_static):
        """The ETag must be over the FINAL served body, not just the computed
        cache value - an edit to sw.js's own logic changes what is served even
        though _compute_sw_cache_value (which excludes sw.js itself) does not
        change, and a stale ETag there would wrongly 304 an actually-different
        response."""
        first = web._sw_js_response()
        value_before = web._compute_sw_cache_value()
        (fake_static / "sw.js").write_text(
            'const CACHE = "localm-shell-dev";\n// sw.js logic actually changed\n',
            encoding="utf-8")
        value_after = web._compute_sw_cache_value()
        assert value_before == value_after, \
            "sanity: this edit must NOT change the watched-asset digest"

        second = web._sw_js_response(if_none_match=first.headers.get("etag"))

        assert second.status_code == 200, \
            "an actually-different body must never 304 against a stale ETag"


class TestRealStaticTree:
    """Pins the mechanism against the REAL, unmocked static/ directory - a
    synthetic-fixture-only suite could keep passing forever after a real
    regression (e.g. STATIC_DIR pointed somewhere empty)."""

    def test_real_tree_has_a_substantial_cacheable_set(self):
        files = web._sw_cacheable_files()
        assert len(files) > 20, \
            f"the real static tree only reports {len(files)} cacheable files - " \
            "STATIC_DIR or the walk logic may be broken"

    def test_real_sw_js_response_succeeds_and_is_deterministic(self):
        r1 = web._sw_js_response()
        r2 = web._sw_js_response()
        assert r1.status_code == 200
        assert r1.body == r2.body
        assert b"localm-shell-dev" not in r1.body
