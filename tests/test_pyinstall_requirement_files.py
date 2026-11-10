# SPDX-License-Identifier: AGPL-3.0-or-later
"""check_command reads the requirements and constraints files an install names:
every option line and direct URL inside them goes through the network policy."""

from __future__ import annotations

import pytest

from localm import pyinstall
from localm.config import update_config

PYPI_ALLOW = ["pypi.org", "files.pythonhosted.org"]


def _allow_only_pypi():
    update_config(lambda c: c.update({"net_allow": PYPI_ALLOW}))


def _write(tmp_path, text, name="req.txt"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _cmd(*args):
    return ["python", "-m", "pip", "install", *args]


LINES = [
    "--index-url https://elsewhere.example/simple",
    "--extra-index-url https://elsewhere.example/simple",
    "--find-links https://elsewhere.example/wheels",
    "-i https://elsewhere.example/simple",
    "-f=https://elsewhere.example/wheels",
    "--index-url=https://elsewhere.example/simple",
    "pkg @ https://elsewhere.example/pkg-1.0.whl",
    "git+https://elsewhere.example/repo.git#egg=pkg",
    "-e git+https://elsewhere.example/repo.git#egg=pkg",
    "https://elsewhere.example/pkg-1.0.whl",
]


@pytest.mark.parametrize("line", LINES)
@pytest.mark.parametrize("flag", ["-r", "--requirement", "-c", "--constraint"])
def test_a_url_line_in_a_file_is_refused_by_net_allow(tmp_path, line, flag):
    _allow_only_pypi()
    f = _write(tmp_path, "requests\n" + line + "\n")
    with pytest.raises(pyinstall.InstallRefused, match="elsewhere.example"):
        pyinstall.check_command(_cmd(flag, str(f)))


@pytest.mark.parametrize("line", LINES)
def test_a_url_line_in_a_file_is_refused_under_off_with_model_downloads(
        tmp_path, monkeypatch, line):
    update_config(lambda c: c.update({"net_allow_model_downloads": True,
                                      "net_deny": ["elsewhere.example"]}))
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    f = _write(tmp_path, line + "\n")
    with pytest.raises(pyinstall.InstallRefused, match="elsewhere.example|deny"):
        pyinstall.check_command(_cmd("-r", str(f)))


def test_attached_and_equals_forms_of_the_file_option_are_read(tmp_path):
    _allow_only_pypi()
    f = _write(tmp_path, "--index-url https://elsewhere.example/simple\n")
    for arg in ("-r" + str(f), "--requirement=" + str(f)):
        with pytest.raises(pyinstall.InstallRefused, match="elsewhere.example"):
            pyinstall.check_command(_cmd(arg))


def test_a_clean_file_passes(tmp_path, monkeypatch):
    update_config(lambda c: c.update({"net_allow_model_downloads": True}))
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    f = _write(tmp_path, "# comment\nrequests>=2  # trailing\n\nnumpy==1.26.0 ; python_version > '3'\n"
                         "--index-url https://pypi.org/simple\n--find-links ./wheels\n")
    pyinstall.check_command(_cmd("-r", str(f), "-c", str(f)))


def test_a_url_in_a_comment_is_not_a_requirement(tmp_path, monkeypatch):
    update_config(lambda c: c.update({"net_allow_model_downloads": True}))
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    f = _write(tmp_path, "requests  # see https://elsewhere.example/x\n"
                         "# --index-url https://elsewhere.example/simple\n")
    pyinstall.check_command(_cmd("-r", str(f)))


def test_an_unreadable_file_is_refused(tmp_path):
    with pytest.raises(pyinstall.InstallRefused, match="cannot read"):
        pyinstall.check_command(_cmd("-r", str(tmp_path / "missing.txt")))
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(pyinstall.InstallRefused, match="cannot read"):
        pyinstall.check_command(_cmd("-r", str(d)))
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00\xd8 not utf8 \x80\x81")
    with pytest.raises(pyinstall.InstallRefused, match="cannot read"):
        pyinstall.check_command(_cmd("-r", str(bad)))


def test_a_remote_requirements_file_is_refused():
    with pytest.raises(pyinstall.InstallRefused, match="remote"):
        pyinstall.check_command(_cmd("-r", "https://pypi.org/req.txt"))


def test_nested_includes_are_followed_relative_to_their_file(tmp_path):
    _allow_only_pypi()
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(sub, "--extra-index-url https://elsewhere.example/simple\n", "inner.txt")
    _write(tmp_path, "-r inner.txt\n", "sub/mid.txt")
    top = _write(tmp_path, "-r sub/mid.txt\n", "top.txt")
    with pytest.raises(pyinstall.InstallRefused, match="elsewhere.example"):
        pyinstall.check_command(_cmd("-r", str(top)))


def test_a_nested_constraint_include_is_followed(tmp_path):
    _allow_only_pypi()
    _write(tmp_path, "pkg @ https://elsewhere.example/p.whl\n", "cons.txt")
    top = _write(tmp_path, "--constraint cons.txt\n", "top.txt")
    with pytest.raises(pyinstall.InstallRefused, match="elsewhere.example"):
        pyinstall.check_command(_cmd("-r", str(top)))


def test_a_nested_unreadable_include_is_refused(tmp_path):
    top = _write(tmp_path, "-r gone.txt\n", "top.txt")
    with pytest.raises(pyinstall.InstallRefused, match="cannot read"):
        pyinstall.check_command(_cmd("-r", str(top)))


def test_an_include_cycle_terminates_and_passes_when_clean(tmp_path, monkeypatch):
    update_config(lambda c: c.update({"net_allow_model_downloads": True}))
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    _write(tmp_path, "-r b.txt\nrequests\n", "a.txt")
    _write(tmp_path, "-r a.txt\n", "b.txt")
    pyinstall.check_command(_cmd("-r", str(tmp_path / "a.txt")))


def test_an_include_cycle_still_finds_a_bad_url(tmp_path):
    _allow_only_pypi()
    _write(tmp_path, "-r b.txt\n", "a.txt")
    _write(tmp_path, "-r a.txt\n-i https://elsewhere.example/simple\n", "b.txt")
    with pytest.raises(pyinstall.InstallRefused, match="elsewhere.example"):
        pyinstall.check_command(_cmd("-r", str(tmp_path / "a.txt")))


def test_nesting_beyond_the_depth_cap_is_refused(tmp_path):
    n = pyinstall._MAX_REQ_DEPTH + 3
    for i in range(n):
        _write(tmp_path, f"-r f{i + 1}.txt\n", f"f{i}.txt")
    _write(tmp_path, "requests\n", f"f{n}.txt")
    with pytest.raises(pyinstall.InstallRefused, match="nest"):
        pyinstall.check_command(_cmd("-r", str(tmp_path / "f0.txt")))


def test_a_relative_path_resolves_against_cwd(tmp_path):
    _allow_only_pypi()
    _write(tmp_path, "-i https://elsewhere.example/simple\n", "rel.txt")
    with pytest.raises(pyinstall.InstallRefused, match="elsewhere.example"):
        pyinstall.check_command(_cmd("-r", "rel.txt"), cwd=tmp_path)


def test_the_command_line_urls_are_still_checked(tmp_path):
    _allow_only_pypi()
    with pytest.raises(pyinstall.InstallRefused, match="elsewhere.example"):
        pyinstall.check_command(_cmd("-i", "https://elsewhere.example/simple"))
