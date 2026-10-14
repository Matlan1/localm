# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real Ed25519 keys and signature envelopes for update-signing tests."""

import base64
import json

from cryptography.hazmat.primitives import serialization as _ser
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from localm import updater


def keypair():
    """A fresh (private key, public key hex) pair."""
    priv = Ed25519PrivateKey.generate()
    return priv, pub_hex(priv)


def pub_hex(priv) -> str:
    return priv.public_key().public_bytes(
        encoding=_ser.Encoding.Raw, format=_ser.PublicFormat.Raw).hex()


def sig_b64(priv, data: bytes) -> str:
    return base64.b64encode(priv.sign(data)).decode("ascii")


def keyset(seq, release, recovery, *, release_threshold=1, recovery_threshold=2,
           signers=()):
    """A key set dict over the given private keys, signed by *signers* (private keys)
    when any are given."""
    ks = {
        "seq": seq,
        "release": {"threshold": release_threshold, "keys": [pub_hex(k) for k in release]},
        "recovery": {"threshold": recovery_threshold, "keys": [pub_hex(k) for k in recovery]},
    }
    if signers:
        body = updater.canonical_keyset_bytes(ks)
        ks["signatures"] = [{"key": pub_hex(k), "sig": sig_b64(k, body)} for k in signers]
    return ks


def envelope(data: bytes, signers, *, keyset=None) -> str:
    """The `.sig` text for build bytes *data*, signed by each private key in *signers*."""
    env = {
        "format": updater._SIG_FORMAT,
        "signatures": [{"key": pub_hex(k), "sig": sig_b64(k, data)} for k in signers],
    }
    if keyset is not None:
        env["keyset"] = keyset
    return json.dumps(env)


class Trust:
    """A pinned trust root of real keys: release keys R, recovery keys A, B, C."""

    def __init__(self, monkeypatch):
        self.release = [Ed25519PrivateKey.generate()]
        self.recovery = [Ed25519PrivateKey.generate() for _ in range(3)]
        monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(1, self.release, self.recovery))

    @property
    def r(self):
        return self.release[0]
