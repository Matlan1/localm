"""The owner-key memo handle is not an unsalted hash of the secret."""
from __future__ import annotations

import hashlib


def test_memo_key_is_not_an_unsalted_hash_of_the_secret():
    from localm import auth
    key = "owner-key-123"
    handle = auth._memo_key(key)
    assert handle == auth._memo_key(key)
    assert handle != auth._memo_key(key + "x")
    assert handle != hashlib.sha256(key.encode()).hexdigest()
    assert handle != auth._fast_digest(key)


def test_memo_key_survives_a_lone_surrogate():
    from localm import auth
    assert auth._memo_key("a\ud800b") == auth._memo_key("a\ud800b")
