# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression audit 2026-07-14, REG-579: a present but readable-and-EMPTY auth.key locked the owner out of their own server."""

import pytest


@pytest.fixture
def auth(tmp_path, monkeypatch):
    """localm.auth with a throwaway data dir and a clean auth environment."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    import localm.auth as a
    monkeypatch.setattr(a, "_empty_owner_key_warned", False, raising=False)
    return a


def _write_key_file(auth, text):
    p = auth.key_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.mark.parametrize("content, label", [
    ("", "zero-byte"),
    ("\n", "a bare newline"),
    ("   \n\t \n", "whitespace only"),
])
def test_present_but_empty_owner_key_is_open_not_a_lockout(auth, content, label):
    """THE REGRESSION."""
    _write_key_file(auth, content)
    assert auth.get_api_key() is None, f"{label} auth.key is not a key"
    assert auth.any_key_configured() is False, (
        f"a readable {label} auth.key put auth IN EFFECT with no key to verify "
        "against - every request 401s and the owner cannot recover via the API")


@pytest.mark.parametrize("raw, label", [
    (b"\xef\xbb\xbf", "BOM only"),
    (b"\xef\xbb\xbf\r\n", "BOM + a blank line"),
    (b"\x00\x00\x00\x00", "NUL-truncated"),
    (b"\xef\xbb\xbf  \n", "BOM + whitespace"),
])
def test_key_file_holding_no_presentable_key_is_open_not_a_lockout(auth, raw, label):
    """The same lockout by another route (found by a fresh-context review of the first cut of this fix, which only handled str.strip()-able whitespace)."""
    p = auth.key_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    assert auth.get_api_key() is None, f"{label} is not a presentable key"
    assert auth.any_key_configured() is False, (
        f"a {label} auth.key put auth IN EFFECT holding a key nobody can "
        "present - the REG-579 lockout by another route")


def test_bom_prefixed_real_key_still_matches_what_the_owner_typed(auth):
    """The other half of the BOM bug: a real key saved WITH a BOM must not come back as '\\ufeff<key>', or the owner presenting the key they typed is rejected (and, since U+FEFF is non-ASCII, hmac.compare_digest raises rather than returning False - a 500 on every request)."""
    p = auth.key_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\xef\xbb\xbfs3cret-key-value\n")
    assert auth.get_api_key() == "s3cret-key-value"
    assert auth.any_key_configured() is True
    assert auth.verify("s3cret-key-value") is not None, (
        "the owner's own correct key did not verify against a BOM'd key file")


def test_notice_re_arms_so_a_later_downgrade_is_not_silent(auth):
    """A server that drops KEYED -> OPEN mid-run (the file is truncated while it runs) must say so."""
    from localm.debuglog import install_ring_buffer, recent_activity
    install_ring_buffer()
    path = _write_key_file(auth, "")
    assert auth.any_key_configured() is False
    assert len([ln for ln in recent_activity() if str(path) in ln]) == 1

    _write_key_file(auth, "a-real-key-value\n")          # owner sets a key
    assert auth.any_key_configured() is True

    _write_key_file(auth, "")                            # ... and it is truncated
    assert auth.any_key_configured() is False
    assert len([ln for ln in recent_activity() if str(path) in ln]) == 2, (
        "the KEYED -> OPEN downgrade was silent the second time")


def test_empty_owner_key_leaves_the_open_mode_recovery_path_intact(auth):
    """The concrete consequence the finding turns on: routes/keys.py mints the first key only when the server was_open."""
    _write_key_file(auth, "")
    was_open = not auth.any_key_configured()
    assert was_open, "the open -> protected recovery transition is unreachable"


def test_empty_owner_key_is_surfaced_not_silent(auth):
    """Rule 5: localm itself never writes an empty auth.key (set_api_key('') unlinks it), so one is always an anomaly - a half-finished setup, or a file truncated by a crash or a sync."""
    from localm.debuglog import install_ring_buffer, recent_activity
    install_ring_buffer()
    path = _write_key_file(auth, "")

    assert auth.any_key_configured() is False
    # The ring is a process-wide singleton shared by the whole test session, so
    # match on THIS test's own key path rather than counting every line in it.
    assert [ln for ln in recent_activity() if str(path) in ln], (
        "an empty owner key silently dropped the server to open mode - the "
        "notice never reached the always-on breadcrumb buffer")


def test_empty_owner_key_warning_does_not_spam_the_request_path(auth):
    """any_key_configured() runs on EVERY request (via _require_auth), so the notice must be throttled - an unthrottled warning would put one line per request in the log (and in every bug report) for a persistent state."""
    from localm.debuglog import install_ring_buffer, recent_activity
    install_ring_buffer()
    path = _write_key_file(auth, "")

    for _ in range(20):
        auth.any_key_configured()
    # Match on THIS test's own key path: the ring is a process-wide singleton and
    # carries other tests' notices too.
    hits = [ln for ln in recent_activity() if str(path) in ln]
    assert len(hits) == 1, f"warned {len(hits)} times across 20 requests"


# --------------------------------------------------------------------------- #
#  NEGATIVE CASES: everything the fix must NOT buy                             #
# --------------------------------------------------------------------------- #

def test_unreadable_owner_key_still_fails_closed(auth):
    """NEGATIVE CASE, the important one. 'Readable and empty' must not be conflated with 'cannot be read'."""
    auth.key_file().mkdir(parents=True, exist_ok=True)
    assert auth.any_key_configured() is True, (
        "an UNREADABLE owner key must fail CLOSED - the fix must only change the "
        "readable-and-empty case")


def test_non_utf8_owner_key_still_fails_closed(auth):
    """NEGATIVE CASE: undecodable bytes are also 'cannot read it', not 'empty' - a truncated/corrupt key file must never be mistaken for no key."""
    p = auth.key_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\xff\xfe\x00\x81\x8dnot utf-8")
    assert auth.any_key_configured() is True


def test_real_owner_key_still_puts_auth_in_effect(auth):
    """NEGATIVE CASE: the ordinary keyed install must be untouched."""
    _write_key_file(auth, "s3cret-key-value\n")
    assert auth.get_api_key() == "s3cret-key-value"
    assert auth.any_key_configured() is True


def test_absent_owner_key_is_still_open(auth):
    """NEGATIVE CASE: the absent case must stay distinct and stay open."""
    assert not auth.key_file().exists()
    assert auth.any_key_configured() is False


def test_env_key_still_wins_over_an_empty_file(auth, monkeypatch):
    """NEGATIVE CASE: an empty auth.key must not mask a real env key."""
    _write_key_file(auth, "")
    monkeypatch.setenv("LOCALM_API_KEY", "envkey-value")
    assert auth.any_key_configured() is True
    assert auth.get_api_key() == "envkey-value"


def test_empty_owner_key_with_a_configured_keystore_stays_in_effect(auth):
    """NEGATIVE CASE: a scoped-keys-only install must not be dropped to open just because the (unused) owner key file happens to be empty."""
    _write_key_file(auth, "")
    auth.keystore_file().parent.mkdir(parents=True, exist_ok=True)
    auth.keystore_file().write_text(
        '[{"hash": "abc", "scopes": ["chat"]}]', encoding="utf-8")
    assert auth.any_key_configured() is True


def test_empty_owner_key_still_locks_when_require_auth_is_on(auth, monkeypatch):
    """NEGATIVE CASE: opening up on an empty key must not defeat the explicit require_auth kill-switch."""
    from fastapi import HTTPException
    _write_key_file(auth, "")
    monkeypatch.setenv("LOCALM_REQUIRE_AUTH", "1")
    assert auth.any_key_configured() is False
    assert auth.require_auth_enabled() is True

    from starlette.requests import Request

    from localm.inference.http_server import _require_auth
    req = Request({"type": "http", "method": "GET", "headers": [],
                   "path": "/", "query_string": b"", "client": ("test", 0)})
    with pytest.raises(HTTPException) as ei:
        _require_auth(req)
    assert ei.value.status_code == 401, (
        "require_auth is on and no key exists - the server must refuse, not "
        "serve open just because the empty auth.key reads as 'no key'")
