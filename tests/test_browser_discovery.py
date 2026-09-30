# SPDX-License-Identifier: AGPL-3.0-or-later
"""Finding the browser the "system" engine drives: each browser on each
platform, the order they are preferred in, and the machine with none of them."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from localm.browser import discovery


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    path.chmod(0o755)
    return path


def _win_env(tmp_path: Path) -> dict:
    return {"LOCALAPPDATA": str(tmp_path / "local"),
            "PROGRAMFILES": str(tmp_path / "pf"),
            "PROGRAMFILES(X86)": str(tmp_path / "pf86")}


def _find_win(tmp_path: Path) -> list:
    return discovery.find_system_browsers(
        platform="win32", env=_win_env(tmp_path), which=lambda command: None)


def _found(names_and_paths: dict, *, platform: str, on_path: dict = None) -> list:
    """Run discovery on a fake machine holding exactly *names_and_paths*'s
    paths, with *on_path* mapping program names to the file PATH resolves."""
    present = set(names_and_paths) | set((on_path or {}).values())
    return discovery.find_system_browsers(
        platform=platform, env={"PATH": ""},
        which=lambda command: (on_path or {}).get(command),
        runnable=lambda path: path in present)


# --------------------------------------------------------------------------- #
#  Windows: a real directory tree standing in for the program files folders   #
# --------------------------------------------------------------------------- #

WINDOWS_BROWSERS = [
    ("pf", "Google/Chrome/Application/chrome.exe", "Google Chrome", "chrome"),
    ("pf86", "Google/Chrome/Application/chrome.exe", "Google Chrome", "chrome"),
    ("local", "Google/Chrome/Application/chrome.exe", "Google Chrome", "chrome"),
    ("pf", "Google/Chrome Beta/Application/chrome.exe", "Google Chrome Beta",
     "chrome-beta"),
    ("pf", "Google/Chrome Dev/Application/chrome.exe", "Google Chrome Dev",
     "chrome-dev"),
    ("pf86", "Microsoft/Edge/Application/msedge.exe", "Microsoft Edge", "msedge"),
    ("pf", "Microsoft/Edge Beta/Application/msedge.exe", "Microsoft Edge Beta",
     "msedge-beta"),
    ("pf", "Microsoft/Edge Dev/Application/msedge.exe", "Microsoft Edge Dev",
     "msedge-dev"),
    ("local", "Chromium/Application/chrome.exe", "Chromium", None),
    ("pf", "BraveSoftware/Brave-Browser/Application/brave.exe", "Brave", None),
    ("local", "BraveSoftware/Brave-Browser/Application/brave.exe", "Brave", None),
]


@pytest.mark.parametrize("base,relative,name,channel", WINDOWS_BROWSERS)
def test_windows_browser_is_found_where_it_installs(
        tmp_path, base, relative, name, channel):
    exe = _exe(tmp_path.joinpath(base, *relative.split("/")))

    found = _find_win(tmp_path)

    assert [(b.name, b.channel, b.path) for b in found] == [
        (name, channel, str(exe))]


def test_windows_with_no_browser_finds_nothing(tmp_path):
    (tmp_path / "pf").mkdir()
    assert _find_win(tmp_path) == []


def test_windows_directories_that_are_not_set_are_skipped(tmp_path):
    exe = _exe(tmp_path / "pf" / "Microsoft" / "Edge" / "Application" / "msedge.exe")

    found = discovery.find_system_browsers(
        platform="win32", env={"PROGRAMFILES": str(tmp_path / "pf")},
        which=lambda command: None)

    assert [b.path for b in found] == [str(exe)]


# --------------------------------------------------------------------------- #
#  Linux and macOS: a fake machine, since the real locations are absolute     #
# --------------------------------------------------------------------------- #

LINUX_CHANNELS = [
    ("/opt/google/chrome/chrome", "Google Chrome", "chrome"),
    ("/opt/google/chrome-beta/chrome", "Google Chrome Beta", "chrome-beta"),
    ("/opt/google/chrome-unstable/chrome", "Google Chrome Dev", "chrome-dev"),
    ("/opt/microsoft/msedge/msedge", "Microsoft Edge", "msedge"),
    ("/opt/microsoft/msedge-beta/msedge", "Microsoft Edge Beta", "msedge-beta"),
    ("/opt/microsoft/msedge-dev/msedge", "Microsoft Edge Dev", "msedge-dev"),
]


@pytest.mark.parametrize("path,name,channel", LINUX_CHANNELS)
def test_linux_chrome_and_edge_are_found_by_their_playwright_channel(
        path, name, channel):
    found = _found({path: 1}, platform="linux")
    assert [(b.name, b.channel, b.path) for b in found] == [(name, channel, path)]


@pytest.mark.parametrize("path", ["/usr/bin/chromium", "/usr/bin/chromium-browser",
                                  "/snap/bin/chromium"])
def test_linux_chromium_is_found_by_path_only(path):
    found = _found({path: 1}, platform="linux")
    assert [(b.name, b.channel, b.path) for b in found] == [("Chromium", None, path)]


@pytest.mark.parametrize("command", ["chromium", "chromium-browser"])
def test_linux_chromium_is_found_on_path_under_either_distro_name(command):
    found = _found({}, platform="linux", on_path={command: "/custom/bin/" + command})
    assert [(b.name, b.channel, b.path) for b in found] == [
        ("Chromium", None, "/custom/bin/" + command)]


@pytest.mark.parametrize("path", ["/opt/brave.com/brave/brave", "/snap/bin/brave"])
def test_linux_brave_is_found_at_its_install_locations(path):
    found = _found({path: 1}, platform="linux")
    assert [(b.name, b.channel, b.path) for b in found] == [("Brave", None, path)]


@pytest.mark.parametrize("command", ["brave-browser", "brave-browser-stable", "brave"])
def test_linux_brave_is_found_on_path(command):
    found = _found({}, platform="linux", on_path={command: "/custom/bin/" + command})
    assert [(b.name, b.path) for b in found] == [("Brave", "/custom/bin/" + command)]


def test_chrome_outside_its_channel_location_is_launched_by_path():
    found = _found({}, platform="linux",
                   on_path={"google-chrome-stable": "/home/u/opt/google-chrome"})

    assert [(b.name, b.channel, b.path) for b in found] == [
        ("Google Chrome", None, "/home/u/opt/google-chrome")]
    assert found[0].launch_options() == {"executable_path": "/home/u/opt/google-chrome"}


def test_edge_on_path_is_launched_by_path():
    found = _found({}, platform="linux",
                   on_path={"microsoft-edge-stable": "/usr/bin/microsoft-edge-stable"})
    assert [(b.name, b.channel) for b in found] == [("Microsoft Edge", None)]


def test_a_browser_at_its_channel_location_and_on_path_is_listed_once():
    found = _found({"/opt/google/chrome/chrome": 1}, platform="linux",
                   on_path={"google-chrome-stable": "/usr/bin/google-chrome-stable"})

    assert [(b.name, b.channel) for b in found] == [("Google Chrome", "chrome")]


def test_a_browser_is_reported_by_its_channel_when_one_exists():
    (chrome,) = _found({"/opt/google/chrome/chrome": 1}, platform="linux")
    assert chrome.launch_options() == {"channel": "chrome"}


def test_a_browser_with_no_channel_is_launched_by_its_executable():
    (brave,) = _found({"/opt/brave.com/brave/brave": 1}, platform="linux")
    assert brave.launch_options() == {"executable_path": "/opt/brave.com/brave/brave"}


MAC_BROWSERS = [
    ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
     "Google Chrome", "chrome"),
    ("/Applications/Chromium.app/Contents/MacOS/Chromium", "Chromium", None),
    ("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
     "Microsoft Edge", "msedge"),
    ("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser", "Brave", None),
]


@pytest.mark.parametrize("path,name,channel", MAC_BROWSERS)
def test_macos_browsers_are_found_in_applications(path, name, channel):
    found = _found({path: 1}, platform="darwin")
    assert [(b.name, b.channel, b.path) for b in found] == [(name, channel, path)]


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32", "freebsd14"])
def test_a_machine_with_no_browser_finds_nothing(platform):
    assert _found({}, platform=platform) == []


def test_every_browser_family_is_named_in_what_is_looked_for():
    assert discovery.LOOKED_FOR == ("Google Chrome", "Chromium", "Microsoft Edge",
                                    "Brave")


def test_stable_releases_are_preferred_to_beta_and_dev_builds():
    machine = {
        "/opt/google/chrome/chrome": 1,
        "/opt/google/chrome-beta/chrome": 1,
        "/opt/google/chrome-unstable/chrome": 1,
        "/usr/bin/chromium": 1,
        "/opt/microsoft/msedge/msedge": 1,
        "/opt/microsoft/msedge-dev/msedge": 1,
        "/opt/brave.com/brave/brave": 1,
    }

    names = [b.name for b in _found(machine, platform="linux")]

    assert names == ["Google Chrome", "Chromium", "Microsoft Edge", "Brave",
                     "Google Chrome Beta", "Google Chrome Dev",
                     "Microsoft Edge Dev"]


# --------------------------------------------------------------------------- #
#  The real PATH and real permission bits                                      #
# --------------------------------------------------------------------------- #

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="needs POSIX permission bits and PATH lookup")


@posix_only
def test_a_file_that_is_not_executable_is_not_a_browser(tmp_path):
    (tmp_path / "chromium").write_bytes(b"")
    (tmp_path / "chromium").chmod(0o644)

    found = discovery.find_system_browsers(
        platform="linux", env={"PATH": str(tmp_path)})

    assert found == []


@posix_only
def test_an_executable_on_the_real_path_is_found(tmp_path):
    exe = _exe(tmp_path / "chromium-browser")

    found = discovery.find_system_browsers(
        platform="linux", env={"PATH": str(tmp_path)})

    assert [(b.name, b.channel, b.path) for b in found] == [
        ("Chromium", None, str(exe))]


def test_the_default_search_reads_this_machine_without_raising():
    assert isinstance(discovery.find_system_browsers(), list)


# --------------------------------------------------------------------------- #
#  The channel locations are the ones the pinned playwright itself uses       #
# --------------------------------------------------------------------------- #

def _driver_bundle() -> str:
    playwright = pytest.importorskip("playwright")
    bundle = Path(playwright.__file__).parent / "driver" / "package" / "lib" / "coreBundle.js"
    if not bundle.exists():
        pytest.skip("this playwright build has no driver bundle to read")
    return bundle.read_text(encoding="utf-8")


def test_every_channel_location_matches_what_playwright_resolves():
    """Each channel location in the discovery table is a location the installed
    playwright resolves for that channel, on every platform."""
    bundle = _driver_bundle()
    checked = 0
    for spec in discovery._SPECS:
        if not spec.channel:
            continue
        marker = '_createChromiumChannel("%s"' % spec.channel
        start = bundle.find(marker)
        assert start >= 0, "playwright has no '%s' channel" % spec.channel
        block = bundle[start:start + 700].replace("\\\\", "/")
        for platform, path in spec.channel_paths.items():
            expected = ("/" + path) if platform == "win32" else path
            assert expected in block, (
                "%s on %s: playwright does not look at %s" % (
                    spec.channel, platform, expected))
            checked += 1
    assert checked >= 18
