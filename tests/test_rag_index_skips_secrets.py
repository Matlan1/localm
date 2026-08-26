# SPDX-License-Identifier: AGPL-3.0-or-later
"""Directory indexing must skip model weights and secret files.

The folder-walk filter is a suffix BLACKLIST, not a whitelist, so every file
whose suffix is not explicitly blocked is read, sniffed, and (if text) indexed.
The blacklist must therefore cover:

  - model weights (.gguf/.safetensors/.pt/.pth/.onnx/...): multi-GB binaries
    that would be fully read into RAM and sha256-hashed (twice) before being
    rejected, repeated on every re-add;
  - secret material (.pem/.key/... and extensionless .env / id_rsa / .netrc):
    plain text that would otherwise land in a searchable, model-visible index.

This is only the recursive FOLDER-WALK filter - a user who explicitly picks a
single secret file is still honoured.
"""

from localm.rag.store import Collection


def test_folder_walk_skips_weights_and_secrets(tmp_path):
    (tmp_path / "notes.txt").write_text("genuinely indexable text")
    (tmp_path / "readme.md").write_text("# doc")
    # model weights
    (tmp_path / "model.gguf").write_bytes(b"GGUF\x00\x00")
    (tmp_path / "w.safetensors").write_bytes(b"\x00" * 32)
    (tmp_path / "ckpt.pt").write_bytes(b"\x00" * 32)
    (tmp_path / "net.onnx").write_bytes(b"\x00" * 32)
    # secrets by suffix (content is irrelevant - the skip is by name/suffix)
    (tmp_path / "server.pem").write_text("cert placeholder")
    (tmp_path / "tls.key").write_text("key placeholder")
    # more key/credential formats
    (tmp_path / "putty.ppk").write_text("PuTTY-User-Key-File placeholder")
    (tmp_path / "signing.p8").write_text("pkcs8 placeholder")
    (tmp_path / "client.ovpn").write_text("<key>embedded key placeholder</key>")
    # secrets by name (extensionless / dotfiles)
    (tmp_path / ".env").write_text("AWS_SECRET_ACCESS_KEY=xyz")
    (tmp_path / "id_rsa").write_text("ssh key placeholder")
    (tmp_path / ".netrc").write_text("machine x login y password z")
    (tmp_path / ".envrc").write_text("export AWS_SECRET_ACCESS_KEY=xyz")

    got = {p.name for p in Collection._expand([tmp_path])}

    assert got == {"notes.txt", "readme.md"}, (
        f"folder walk indexed weights/secrets it should skip: "
        f"{got - {'notes.txt', 'readme.md'}}")


def test_explicit_secret_file_is_still_honoured(tmp_path):
    """A user explicitly adding a single file (not a folder) keeps working - the
    skip is only for the recursive folder walk."""
    env = tmp_path / ".env"
    env.write_text("K=v")
    got = {p.name for p in Collection._expand([env])}
    assert got == {".env"}
