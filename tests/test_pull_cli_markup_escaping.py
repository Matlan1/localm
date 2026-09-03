# SPDX-License-Identifier: AGPL-3.0-or-later
"""model_manager/pull.py interpolates repo_id/filename/url/dest/exception text
straight into rich.console.Console.print() f-strings. Rich parses any
"[...]" in ANY interpolated string as markup, including when the value sits
INSIDE an already-open [style]...[/style] tag - and several sites here are
exactly that worse case: markup TAG-INJECTION, not just bracket-drop or
bogus-style-consumption. A crafted repo_id/filename/url can CLOSE the real
tag and OPEN NEW ones, injecting fake styled text of the attacker's choosing
into what the user sees as localm's own pull output, not merely corrupting
its own display.

Confirmed HTTP-reachable with no auth by default: POST /api/models/pull
(localm/plugins/gui/routes/models.py) takes req.spec/req.mmproj verbatim and
spawns `localm pull -- <spec>` as a subprocess (localm/plugins/gui/jobs.py),
whose stdout lines - every console.print() call in this file - are re-pushed
into the GUI's job/activity log verbatim (job.push({"type": "line", ...})).

Repro against this venv's rich, showing the difference between plain
bracket-drop and real tag injection:

    Console().print('report[draft].txt')                    -> "report.txt"
    Console().print('[bold cyan]x][red]FAKE[/red][/bold cyan]')
        -> renders "FAKE" in ACTUAL red styling, not as literal text

rich.markup.escape() neutralizes both: every bracket (including the
attacker's own fake tag pair) renders as plain, unstyled literal text.

TAG-INJECTION sites are proven with a REAL rich.console.Console(record=True)
swapped in for the module's shared console, so the assertion is against what
Rich actually renders - checked two ways: export_text(styles=False) must
contain the WHOLE crafted payload verbatim (nothing dropped/consumed), and
export_text(styles=True) - which encodes each rendered segment's style as an
ANSI SGR code, making an injected style directly observable - must NOT
contain the ANSI code the injected style would produce. console.print()
itself is never mocked away: only the module's `console` name is swapped for
a recording Console, so Rich's own markup parser still runs for real on the
(escaped) f-string the fix produces. An UNESCAPED site produces the inverse:
the payload is NOT found verbatim in plain output, and the injected ANSI code
IS present in styled output.

Plain-corruption sites (the interpolated value sits outside any open tag, so
a crafted value cannot escape into someone else's span, but can still lose
its own brackets or open unstyled fake tags of its own) follow the
BRACKET_DROP/BRACKET_STYLE convention used across the markup-escaping tests.

The network layer (huggingface_hub, requests, localm.netpolicy) is mocked to
hand back attacker-controlled values.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm
from localm.model_manager import pull as pull_mod


# ---------------------------------------------------------------------------
# Shared payloads and the recording-console fixture
# ---------------------------------------------------------------------------

# The two shapes test_rag_cli_markup_escaping.py's docstring describes: one
# name Rich DROPS outright, one it consumes as a (bogus) style tag.
BRACKET_DROP = "repo[legacy]-name"
BRACKET_STYLE = "repo[bold red]xyz-name"

# A tag-injection payload: closes whatever *open_tag* it lands inside, opens
# its own distinctly-styled span carrying a distinctive marker, then reopens
# *open_tag* so a literal "[/{open_tag}]" written by the caller AFTER this
# value still balances even though the payload smuggled its own tag pair
# into the middle of the real one.
_INJECT_STYLE = "on white"
_INJECT_MARKER = "PWNED-MARKER"
# The SGR "47" (white background) code "on white" produces, matched as a
# REGEX rather than a fixed literal: when the injected style CLOSES the
# surrounding tag first (_tag_injection_payload), "47" renders alone
# ("\x1b[47m"); when it merely OPENS without closing
# (_tag_injection_payload_no_slash), Rich MERGES it with whatever style was
# already active ("\x1b[33;47m" alongside an open [yellow], etc.) - both
# shapes occur in this file's real sites. A fixed-string search for
# "\x1b[47m" alone misses the merged form and would silently pass on genuine
# injection - the exact "test that cannot fail" trap this regex exists to
# avoid.
_INJECT_ANSI_RE = re.compile(r"\x1b\[[0-9;]*47(;[0-9]+)*m")


def _tag_injection_payload(open_tag: str) -> str:
    """For positions with NO filename-safety filter (repo_id, url, dest,
    exception text): CLOSES the real tag, injects its own styled span, then
    REOPENS the real tag so a literal trailing "[/{open_tag}]" the caller
    writes still balances."""
    return (f"evil][/{open_tag}][{_INJECT_STYLE}]{_INJECT_MARKER}"
            f"[/{_INJECT_STYLE}][{open_tag}]repo")


def _tag_injection_payload_no_slash() -> str:
    """For FILENAME-derived positions, which are always run through
    _safe_models_filename first and so can never contain '/' - the character
    every real Rich CLOSING tag needs. Injection is still real without one:
    a bare "[style]" OPENS a new span that is simply never closed, which
    still applies real styling to (and swallows, in the plain export) the
    marker text."""
    return f"evilname[{_INJECT_STYLE}]{_INJECT_MARKER}"


@pytest.fixture()
def rich_capture(monkeypatch):
    """Swap pull.py's shared console for a REAL recording Console - so
    console.print()'s own Rich markup parser still runs for real against
    whatever the (fixed) f-string actually contains, and the assertion is
    against genuine rendered output rather than a mocked stand-in."""
    from rich.console import Console
    cap = Console(record=True, width=400, force_terminal=True, no_color=False,
                  highlight=False)
    monkeypatch.setattr(pull_mod, "console", cap)
    return cap


def _plain(cap) -> str:
    return cap.export_text(styles=False, clear=False)


def _styled(cap) -> str:
    return cap.export_text(styles=True, clear=False)


def _assert_injection_blocked(cap, payload: str):
    """The one assertion every TAG-INJECTION test shares: the WHOLE crafted
    payload (including its own fake tag pair) survives verbatim as plain
    text, and the injected style never actually applied."""
    plain = _plain(cap)
    styled = _styled(cap)
    assert payload in plain, (
        f"the crafted value must render as literal text verbatim, not be "
        f"parsed as markup: {plain!r}")
    assert not _INJECT_ANSI_RE.search(styled), (
        f"the injected '[{_INJECT_STYLE}]' span must NEVER actually apply - "
        f"finding its ANSI code (standalone or merged with a surrounding "
        f"style) in the styled render means real tag injection occurred: "
        f"{styled!r}")


# ---------------------------------------------------------------------------
# Shared network-layer fixtures (mirror test_pull_mmproj_autofetch.py /
# test_hf_endpoint_pinned.py's fake_registry fixture)
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    store: dict = {}
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "load_registry", lambda: dict(store))

    def _save(reg):
        store.clear()
        store.update(reg)

    monkeypatch.setattr(mm, "save_registry", _save)

    def _update(mutator):
        reg = dict(store)
        mutator(reg)
        store.clear()
        store.update(reg)
        return dict(store)

    monkeypatch.setattr(mm, "update_registry", _update)
    monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, fn: None)
    monkeypatch.setattr(mm, "gguf_registry_metadata", lambda p: {})
    monkeypatch.setattr(mm, "gguf_is_mmproj", lambda p: False)
    monkeypatch.setattr(mm, "gguf_embedding_signal", lambda p: False)
    monkeypatch.setattr("requests.head", lambda *a, **k: MagicMock(
        headers={"content-length": "4"}))
    return store, models_dir


def _wire_hf_hub_download(monkeypatch, content: bytes = b"GGUF"):
    def _fake_download(repo_id, filename, local_dir, **kw):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return str(p)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)


@pytest.fixture()
def url_env(tmp_path, monkeypatch):
    """A models dir with the disk-space and dir-creation checks stubbed out."""
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])
    monkeypatch.setattr(mm, "_register", MagicMock())
    monkeypatch.setattr(mm, "_register_with_dedup", MagicMock())
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    # Avoid the Rich Progress bar entirely (its Live-display ANSI would
    # clutter the recording console with unrelated content) - GUI mode
    # streams JSON sentinels through a different path instead.
    monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
    return models


def _resp(status, body: bytes, content_length=None):
    r = MagicMock()
    r.status_code = status
    r.raise_for_status = MagicMock()
    cl = len(body) if content_length is None else content_length
    r.headers = {"content-length": str(cl)}

    def _iter(chunk_size):
        for i in range(0, len(body), chunk_size):
            yield body[i:i + chunk_size]
    r.iter_content = _iter
    return r


def _wire_http(monkeypatch, head_total: int, response):
    def fake_pinned_request(method, url, **kwargs):
        if method == "HEAD":
            h = MagicMock()
            h.status_code = 200
            h.headers = {"content-length": str(head_total)}
            return h
        return response

    monkeypatch.setattr("localm.netpolicy.pinned_request", fake_pinned_request)


# ---------------------------------------------------------------------------
# TAG-INJECTION: _pull_gguf_file's "Pulling [bold cyan]{repo_id}[/bold cyan]
# / [bold]{filename}[/bold]" message
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Widen the console for every test in this module.

    rich.console.Console.size returns 80x25 outright on a dumb terminal,
    before it ever consults COLUMNS - patching is_dumb_terminal is what makes
    the COLUMNS override below actually take effect."""
    import rich.console
    monkeypatch.setattr(rich.console.Console, "is_dumb_terminal", False)
    monkeypatch.setenv("COLUMNS", "300")


class TestPullGgufFileTagInjection:
    def test_repo_id_tag_injection_blocked(
            self, rich_capture, fake_registry, monkeypatch):
        _wire_hf_hub_download(monkeypatch)
        payload = _tag_injection_payload("bold cyan")
        spec = f"{payload}:model.gguf"

        ok = mm._pull_gguf_file(spec, None)

        assert ok is True
        _assert_injection_blocked(rich_capture, payload)

    def test_filename_tag_injection_blocked(
            self, rich_capture, fake_registry, monkeypatch):
        # filename is confined by _safe_models_filename ahead of this print, so
        # it can never carry a real closing '[/...]' tag - the no-slash
        # payload (open-without-close) is the exploitable shape here.
        _wire_hf_hub_download(monkeypatch)
        payload = _tag_injection_payload_no_slash()
        spec = f"owner/repo:{payload}.gguf"

        ok = mm._pull_gguf_file(spec, None)

        assert ok is True
        _assert_injection_blocked(rich_capture, payload)


# ---------------------------------------------------------------------------
# Plain-corruption sites within _pull_gguf_file
# ---------------------------------------------------------------------------

class TestPullGgufFileBracketDrop:
    def test_unsafe_model_filename_message_survives_verbatim(
            self, rich_capture, fake_registry):
        """The 'genuinely ironic' site: a filename that FAILS the traversal
        safety check must not be able to weaponize the warning about it.
        A literal '/' in the spec's filename half is rejected by
        _safe_models_filename before this print, so use a bracketed name
        that still contains a rejected character (a colon is unreachable
        here since ':' splits the spec - use a NUL-adjacent reserved char
        instead: '<' is Windows-reserved and still trips the guard)."""
        spec = f"owner/repo:{BRACKET_STYLE}<.gguf"

        ok = mm._pull_gguf_file(spec, None)

        assert ok is False
        out = _plain(rich_capture)
        assert f"{BRACKET_STYLE}<.gguf" in out, (
            f"the unsafe-filename warning must show the real (unsafe) name "
            f"verbatim: {out!r}")

    def test_already_downloaded_message_survives_verbatim(
            self, rich_capture, fake_registry, tmp_path):
        store, models_dir = fake_registry
        name = f"{BRACKET_DROP}.gguf"
        (models_dir / name).write_bytes(b"GGUF")
        spec = f"owner/repo:{name}"

        ok = mm._pull_gguf_file(spec, None, register=False)

        assert ok is True
        out = _plain(rich_capture)
        assert name in out, (
            f"'Already downloaded' must show the real filename verbatim: "
            f"{out!r}")

    def test_download_failed_message_survives_verbatim(
            self, rich_capture, fake_registry, monkeypatch):
        payload = BRACKET_STYLE
        spec = f"owner/repo:{payload}.gguf"

        import huggingface_hub

        def _raise(*a, **kw):
            raise RuntimeError(f"boom {BRACKET_DROP}")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _raise)

        ok = mm._pull_gguf_file(spec, None)

        assert ok is False
        out = _plain(rich_capture)
        assert f"{payload}.gguf" in out, (
            f"'Download failed' must show the real filename verbatim: {out!r}")
        assert BRACKET_DROP in out, (
            f"'Download failed' must show the real exception text verbatim: "
            f"{out!r}")


# ---------------------------------------------------------------------------
# TAG-INJECTION: _pull_hf_snapshot's several open-tag sites
# ---------------------------------------------------------------------------

class TestPullHfSnapshotTagInjection:
    def test_downloading_full_model_repo_id_tag_injection_blocked(
            self, rich_capture, fake_registry, monkeypatch):
        def _fake_snapshot_download(**kw):
            Path(kw["local_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(kw["local_dir"]) / "config.json").write_text("{}")
            return kw["local_dir"]

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "snapshot_download",
                             _fake_snapshot_download)
        # model_type defaults to "llm" (not "unknown"), so
        # _resolve_snapshot_type returns immediately without ever calling
        # _detect_local_model_type - nothing to mock there.

        class _EmptyHfApi:
            def __init__(self, *a, **kw):
                pass

            def model_info(self, repo_id, files_metadata=True):
                raise RuntimeError("offline")

        monkeypatch.setattr(huggingface_hub, "HfApi", _EmptyHfApi)

        payload = _tag_injection_payload("bold cyan")

        ok = mm._pull_hf_snapshot(payload, "snap")

        assert ok is True
        _assert_injection_blocked(rich_capture, payload)

    def test_refusing_to_pull_collision_repo_id_tag_injection_blocked(
            self, rich_capture, fake_registry, monkeypatch):
        """The dest-collision refusal interpolates repo_id AND dest directly
        inside an OPEN [red]...[/red] tag."""
        store, models_dir = fake_registry
        model_name = "snap"
        dest = models_dir / model_name
        # dest exists but has NO config.json, so _snapshot_is_complete()
        # returns False and the "Already downloaded" fast path (which would
        # return before ever reaching the collision check) is not taken.
        dest.mkdir()
        # A DIFFERENT source already registered at this exact dest -> the
        # collision-refusal branch fires before any real download.
        store["other-name"] = {"path": str(dest), "source": "hf:some/other-repo"}

        class _OfflineHfApi:
            def __init__(self, *a, **kw):
                pass

            def model_info(self, repo_id, files_metadata=True):
                raise RuntimeError("offline")

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _OfflineHfApi)

        payload = _tag_injection_payload("red")

        ok = mm._pull_hf_snapshot(payload, model_name)

        assert ok is False
        _assert_injection_blocked(rich_capture, payload)

    def test_warn_could_not_check_bundled_code_tag_injection_blocked(
            self, rich_capture):
        """_warn_if_repo_ships_code's early-return OSError branch - the one
        documented gap in the pre-existing escaping this unit closes.
        repo_id AND the exception text both sit inside the OPEN
        [yellow]...[/yellow] tag."""
        payload = _tag_injection_payload("yellow")

        class _RaisingDest:
            def rglob(self, pattern):
                raise OSError(f"scan failed: {payload}")

        pull_mod._warn_if_repo_ships_code(_RaisingDest(), payload)

        plain = _plain(rich_capture)
        styled = _styled(rich_capture)
        # Two independent instances of the payload appear (repo_id and the
        # exception text) - both must survive verbatim and neither may
        # actually apply the injected style.
        assert plain.count(payload) == 2, (
            f"both repo_id and the exception text must survive verbatim: "
            f"{plain!r}")
        assert not _INJECT_ANSI_RE.search(styled), (
            f"the injected style must never actually apply: {styled!r}")


# ---------------------------------------------------------------------------
# Plain-corruption sites within _pull_hf_snapshot
# ---------------------------------------------------------------------------

class TestPullHfSnapshotBracketDrop:
    # No "Already downloaded" bracket-drop case here: the message at that
    # site shows _pull_hf_snapshot's OWN model_name, which is always
    # _sanitize_name()-derived (A-Za-z0-9._- only) by the time it reaches this
    # print, so a bracket cannot reach it - it is escaped anyway as
    # defense-in-depth, but a bracket-survival test on a value that is
    # stripped before it ever arrives would pass whether or not escape() ran.
    # _pull_gguf_file's own "Already downloaded" case
    # (TestPullGgufFileBracketDrop above) covers the same message shape with a
    # genuinely unsanitized filename instead.

    def test_sha256_not_supported_message_survives_verbatim(
            self, rich_capture, fake_registry):
        payload = f"owner/{BRACKET_STYLE}"

        ok = mm._pull_hf_snapshot(payload, "snap", expected_sha256="deadbeef")

        assert ok is False
        out = _plain(rich_capture)
        assert payload in out, (
            f"the --sha256-not-supported message must show repo_id "
            f"verbatim: {out!r}")


# ---------------------------------------------------------------------------
# TAG-INJECTION: _pull_url_locked's "Resuming/Downloading
# [bold cyan]{url}[/bold cyan]" messages - url is FULLY attacker-controlled
# (the raw spec string), the highest-severity single site in this file.
# ---------------------------------------------------------------------------

class TestPullUrlTagInjection:
    def test_downloading_url_tag_injection_blocked(
            self, rich_capture, url_env, monkeypatch):
        payload = _tag_injection_payload("bold cyan")
        # A crafted URL - the whole string after "https://" is attacker text,
        # so the payload can sit directly in the path with no encoding needed.
        url = f"https://example.com/{payload}.gguf"
        _wire_http(monkeypatch, 4, _resp(200, b"GGUF"))

        ok = mm._pull_url(url, "mymodel")

        assert ok is True
        _assert_injection_blocked(rich_capture, payload)

    def test_resuming_url_tag_injection_blocked(
            self, rich_capture, url_env, monkeypatch):
        # The "Resuming [bold cyan]{url}[/bold cyan]" print uses the WHOLE
        # raw url, unfiltered. The destination FILENAME, though, is derived
        # from only the url's LAST path segment and passed through
        # _safe_models_filename (real traversal/reserved-char guard) - so
        # the payload (which contains '/', rejected by that guard) is placed
        # in an EARLIER path segment, leaving the last segment ("plain.gguf")
        # safe for the .part-file resume setup below. The raw url printed
        # still carries the full payload either way.
        payload = _tag_injection_payload("bold cyan")
        url = f"https://example.com/{payload}/plain.gguf"
        # Pre-existing .part bytes so the resume branch (not the fresh one)
        # fires the "Resuming" message instead of "Downloading".
        (url_env / "plain.gguf.part").write_bytes(b"01")
        _wire_http(monkeypatch, 4, _resp(206, b"UF", content_length=2))

        ok = mm._pull_url(url, "mymodel")

        assert ok is True
        _assert_injection_blocked(rich_capture, payload)


# ---------------------------------------------------------------------------
# Plain-corruption sites within _pull_url / _pull_url_locked
# ---------------------------------------------------------------------------

class TestPullUrlBracketDrop:
    def test_invalid_url_message_survives_verbatim(self, rich_capture, url_env):
        url = f"https://example.com/{BRACKET_DROP}"   # no filename segment

        ok = mm._pull_url(url, "m")

        assert ok is False
        out = _plain(rich_capture)
        assert url in out, f"the invalid-URL message must echo the URL verbatim: {out!r}"

    def test_unsafe_filename_derived_from_url_message_survives_verbatim(
            self, rich_capture, url_env):
        """Another 'genuinely ironic' site: a URL-derived filename that just
        FAILED the safety check must not weaponize the warning about it."""
        url = f"https://example.com/{BRACKET_STYLE}<.gguf"

        ok = mm._pull_url(url, "m")

        assert ok is False
        out = _plain(rich_capture)
        assert f"{BRACKET_STYLE}<.gguf" in out, (
            f"the unsafe-filename-from-URL warning must show the real "
            f"(unsafe) derived name verbatim: {out!r}")

    def test_could_not_reach_message_survives_verbatim(
            self, rich_capture, url_env, monkeypatch):
        import requests
        url = f"https://example.com/{BRACKET_STYLE}.gguf"

        def fake_pinned_request(method, u, **kwargs):
            if method == "HEAD":
                h = MagicMock()
                h.status_code = 200
                h.headers = {"content-length": "4"}
                return h
            raise requests.ConnectionError(f"refused: {BRACKET_DROP}")

        monkeypatch.setattr("localm.netpolicy.pinned_request", fake_pinned_request)

        ok = mm._pull_url(url, "m")

        assert ok is False
        out = _plain(rich_capture)
        assert url in out, f"'Could not reach' must show the URL verbatim: {out!r}"
        assert BRACKET_DROP in out, (
            f"'Could not reach' must show the real exception text verbatim: "
            f"{out!r}")

    def test_sha256_mismatch_after_download_message_survives_verbatim(
            self, rich_capture, url_env, monkeypatch):
        url = "https://example.com/plain.gguf"
        # The print only shows the first 16 chars (bad_sha[:16]) - the
        # COMPLETE bracket tag must fit inside that slice, or an unescaped
        # site would pass this assertion for the wrong reason (a dangling,
        # unterminated '[' is left as literal text by Rich's own parser
        # regardless of escaping).
        bad_sha = "[bold red]" + "c" * 40
        assert "[bold red]" in bad_sha[:16], "payload must fit its own tag in the visible slice"
        _wire_http(monkeypatch, 4, _resp(200, b"GGUF"))

        ok = mm._pull_url(url, "m", expected_sha256=bad_sha)

        assert ok is False
        out = _plain(rich_capture)
        assert bad_sha[:16] in out, (
            f"the SHA256-mismatch message must show the raw --sha256 value "
            f"verbatim (it is not format-validated before this print): "
            f"{out!r}")


# ---------------------------------------------------------------------------
# TAG-INJECTION: mmproj auto-attach / explicit-mmproj "does not look like a
# valid vision projector" messages, where the filename sits right after the
# opening [yellow] tag
# ---------------------------------------------------------------------------

class TestMmprojTagInjection:
    def test_autoattach_bad_projector_candidate_tag_injection_blocked(
            self, rich_capture, tmp_path, monkeypatch):
        # The candidate filename is confined by _safe_models_filename before
        # it is even considered a candidate (_pick_mmproj_from_listing), so
        # it can never carry a real closing '[/...]' tag - the no-slash
        # (open-without-close) payload is the exploitable shape here. It
        # must also contain "mmproj" to be picked up as a candidate at all.
        payload = _tag_injection_payload_no_slash()
        candidate = f"mmproj-{payload}.gguf"

        class _FakeHfApi:
            def __init__(self, *a, **kw):
                pass

            def list_repo_files(self, repo_id):
                return ["main.gguf", candidate]

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
        _wire_hf_hub_download(monkeypatch)
        # gguf_is_mmproj is checked AFTER the fetch and here reports the
        # fetched file is NOT actually a valid projector (a real, if rare,
        # outcome: filename pattern matched, GGUF metadata did not).
        monkeypatch.setattr(pull_mod._mm, "gguf_is_mmproj", lambda p: False)

        result = pull_mod._maybe_fetch_repo_mmproj(
            "owner/repo", "main.gguf", tmp_path)

        assert result is None
        _assert_injection_blocked(rich_capture, payload)

    def test_explicit_mmproj_bad_projector_tag_injection_blocked(
            self, rich_capture, tmp_path, monkeypatch):
        # Same _safe_models_filename confinement as above applies to the
        # explicit --mmproj filename.
        payload = _tag_injection_payload_no_slash()
        mmproj_spec = f"owner/repo:{payload}.gguf"
        _wire_hf_hub_download(monkeypatch)
        monkeypatch.setattr(pull_mod._mm, "gguf_is_mmproj", lambda p: False)

        result = pull_mod._fetch_explicit_mmproj(mmproj_spec, tmp_path)

        assert result is None
        _assert_injection_blocked(rich_capture, payload)


# ---------------------------------------------------------------------------
# Plain-corruption sites within the mmproj helpers
# ---------------------------------------------------------------------------

class TestMmprojBracketDrop:
    def test_unsafe_mmproj_filename_message_survives_verbatim(
            self, rich_capture, tmp_path):
        """The 'genuinely ironic' site in _fetch_explicit_mmproj: a filename
        that just FAILED the safety check must not weaponize its own
        warning."""
        bad_name = f"{BRACKET_STYLE}<.gguf"
        mmproj_spec = f"owner/repo:{bad_name}"

        result = pull_mod._fetch_explicit_mmproj(mmproj_spec, tmp_path)

        assert result is None
        out = _plain(rich_capture)
        assert bad_name in out, (
            f"the unsafe-mmproj-filename warning must show the real name "
            f"verbatim: {out!r}")

    def test_mmproj_download_failed_message_survives_verbatim(
            self, rich_capture, tmp_path, monkeypatch):
        mmproj_spec = f"owner/repo:{BRACKET_STYLE}.gguf"

        import huggingface_hub

        def _raise(*a, **kw):
            raise RuntimeError(f"network gone: {BRACKET_DROP}")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _raise)

        result = pull_mod._fetch_explicit_mmproj(mmproj_spec, tmp_path)

        assert result is None
        out = _plain(rich_capture)
        assert BRACKET_DROP in out, (
            f"the mmproj-download-failed message must show the real "
            f"exception text verbatim: {out!r}")


# ---------------------------------------------------------------------------
# pull_model()'s own sites
# ---------------------------------------------------------------------------

class TestPullModelBracketDrop:
    def test_unknown_spec_message_survives_verbatim(self, rich_capture, fake_registry):
        spec = BRACKET_STYLE   # no "/" and no scheme -> the "Unknown spec" branch

        ok = mm.pull_model(spec)

        assert ok is False
        out = _plain(rich_capture)
        assert spec in out, f"'Unknown spec' must echo the real spec verbatim: {out!r}"

    def test_dest_dir_single_file_refusal_survives_verbatim_and_is_quoted(
            self, rich_capture, fake_registry, tmp_path):
        """repr()-avoidance: the value is hand-quoted with escape(), not via
        !r/repr(), so this also confirms no double-escaping artifact (a
        literal backslash) leaks through. Must use a BRACKETED spec, not a
        clean literal, or this cannot distinguish escaped from unescaped."""
        spec = f"owner/{BRACKET_STYLE}"   # not single-file: no ':', no '.gguf'
        dest_dir = tmp_path / "comfy"

        ok = mm.pull_model(spec, dest_dir=dest_dir)

        assert ok is False
        out = _plain(rich_capture)
        assert f"'{spec}'" in out, (
            f"the dest_dir refusal must show the spec verbatim, hand-quoted: "
            f"{out!r}")

    def test_local_sha256_mismatch_message_survives_verbatim(
            self, rich_capture, fake_registry, tmp_path):
        local_file = tmp_path / f"{BRACKET_STYLE}.gguf"
        local_file.write_bytes(b"GGUF")
        bad_sha = f"deadbeef-{BRACKET_DROP}"

        ok = mm.pull_model(str(local_file), expected_sha256=bad_sha)

        assert ok is False
        out = _plain(rich_capture)
        assert str(local_file) in out, (
            f"the local SHA256-mismatch message must show the real path "
            f"verbatim: {out!r}")
        assert bad_sha[:16] in out, (
            f"the local SHA256-mismatch message must show the raw --sha256 "
            f"value verbatim (not format-validated before this print): "
            f"{out!r}")
