# SPDX-License-Identifier: AGPL-3.0-or-later
"""A failed browser launch reads as localm's advice, not playwright's.

The pip extra installs the playwright PACKAGE, not the Chromium build it
drives, so an install that followed the documented steps can still fail at
launch. Playwright words that failure for its own command line: a boxed banner
telling the reader to run ``playwright install``. These pin that the message a
user sees states the reason and localm's remedy and carries none of that, that
the raw text goes to the debug log, and that the system engine launches the
browser it found.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import types

import pytest

from localm.browser import discovery, launch_errors, provision
from localm.browser import session as bsession

_TL, _TR, _BL, _BR, _V, _H = (chr(c) for c in (0x2554, 0x2557, 0x255A, 0x255D,
                                               0x2551, 0x2550))


def _box(*lines: str, width: int = 60) -> str:
    """Text framed the way playwright frames its banners."""
    rows = [_TL + _H * width + _TR]
    rows += [_V + (" " + line).ljust(width) + _V for line in lines]
    rows.append(_BL + _H * width + _BR)
    return "\n".join(rows)


HEADLESS_SHELL = ("/home/user/.cache/ms-playwright/chromium_headless_shell-1243/"
                  "chrome-headless-shell-linux64/chrome-headless-shell")

#: The message playwright raises on Linux when the bundled build is not there.
BUNDLED_MISSING_LINUX = (
    "BrowserType.launch: Executable doesn't exist at " + HEADLESS_SHELL + "\n"
    + _box("Looks like Playwright was just installed or updated.",
           "Please run the following command to download new browsers:",
           "", "    playwright install", "", "<3 Playwright Team"))

#: The message for a channel whose browser is not installed.
CHANNEL_MISSING_LINUX = (
    "BrowserType.launch: Chromium distribution 'chrome' is not found at "
    "/opt/google/chrome/chrome\nRun \"playwright install chrome\"")

CHANNEL_MISSING_WINDOWS = (
    "BrowserType.launch: Chromium distribution 'chrome-dev' is not found at "
    "C:\\Users\\user\\AppData\\Local\\Google\\Chrome Dev\\Application\\chrome.exe")

EXECUTABLE_PATH_MISSING = (
    "BrowserType.launch: Failed to launch chromium because executable doesn't "
    "exist at /usr/bin/chromium")

#: The shape of a launch whose browser exits at once for a missing shared
#: library (launch arguments elided).
MISSING_LIBRARY_LINUX = (
    "BrowserType.launch: Target page, context or browser has been closed\n"
    "Browser logs:\n\n"
    "<launching> " + HEADLESS_SHELL + " --disable-field-trial-config --headless\n"
    "<launched> pid=763\n"
    "[pid=763][err] " + HEADLESS_SHELL + ": error while loading shared libraries: "
    "libnspr4.so: cannot open shared object file: No such file or directory\n"
    "Call log:\n"
    "  - <launching> " + HEADLESS_SHELL + " --disable-field-trial-config --headless\n"
    "  - <launched> pid=763\n"
    "  - [pid=763][err] " + HEADLESS_SHELL + ": error while loading shared "
    "libraries: libnspr4.so: cannot open shared object file: No such file or "
    "directory\n"
    "  - [pid=763] <process did exit: exitCode=127, signal=null>\n")

#: The banner playwright raises when it knows the distribution's package names.
MISSING_DEPENDENCIES_DEBIAN = (
    "BrowserType.launch: \n"
    + _box("Host system is missing dependencies to run browsers.",
           "Please install them with the following command:",
           "", "    sudo playwright install-deps", "",
           "Alternatively, use apt:",
           "    sudo apt-get install libnss3\\",
           "        libnspr4\\",
           "        libasound2t64", "", "<3 Playwright Team"))

#: The banner playwright raises for a distribution it has no package names for.
MISSING_DEPENDENCIES_OTHER = (
    "BrowserType.launch: \n"
    + _box("Host system is missing dependencies to run browsers.",
           "Missing libraries:", "    libnss3.so", "    libnspr4.so",
           "    libasound.so.2"))

NOT_FOR_USERS = ("playwright install", "playwright-cli", "npx playwright",
                 "install-deps", "Playwright Team", _V, _TL, _BL, _H,
                 "BrowserType.launch", "Call log", "Browser logs", "<launching>")


def _assert_reads_as_localm(message: str) -> None:
    assert "\n" not in message, message
    for forbidden in NOT_FOR_USERS:
        assert forbidden not in message, (
            f"{forbidden!r} reached the user: {message}")


# --------------------------------------------------------------------------- #
#  The words                                                                   #
# --------------------------------------------------------------------------- #

def test_a_missing_bundled_build_names_localms_download_not_playwrights():
    kind, message = launch_errors.launch_failure(
        BUNDLED_MISSING_LINUX, engine="bundled")

    assert kind == launch_errors.BUNDLED_MISSING
    _assert_reads_as_localm(message)
    assert "localm setup-browser" in message
    assert "Download browser" in message


def test_a_missing_bundled_build_is_recognised_for_a_headed_launch_too():
    raw = BUNDLED_MISSING_LINUX.replace(
        "chromium_headless_shell-1243/chrome-headless-shell-linux64/"
        "chrome-headless-shell", "chromium-1243/chrome-linux64/chrome")

    kind, _ = launch_errors.launch_failure(raw, engine="bundled")

    assert kind == launch_errors.BUNDLED_MISSING


@pytest.mark.parametrize("raw", [CHANNEL_MISSING_LINUX, CHANNEL_MISSING_WINDOWS,
                                 EXECUTABLE_PATH_MISSING])
def test_a_system_browser_that_is_gone_says_so_without_playwrights_advice(raw):
    kind, message = launch_errors.launch_failure(
        raw, engine="system", browser="Google Chrome")

    assert kind == launch_errors.SYSTEM_MISSING
    _assert_reads_as_localm(message)
    assert "Google Chrome" in message
    assert "bundled" in message


def test_a_browser_that_cannot_start_for_a_missing_library_names_it():
    kind, message = launch_errors.launch_failure(
        MISSING_LIBRARY_LINUX, engine="bundled")

    assert kind == launch_errors.MISSING_LIBRARIES
    _assert_reads_as_localm(message)
    assert "libnspr4.so" in message
    assert "system libraries" in message
    assert "package manager" in message


def test_playwrights_dependency_banner_becomes_plain_words_with_the_packages():
    kind, message = launch_errors.launch_failure(
        MISSING_DEPENDENCIES_DEBIAN, engine="bundled")

    assert kind == launch_errors.MISSING_LIBRARIES
    _assert_reads_as_localm(message)
    assert ("On Debian and Ubuntu these come from the packages: "
            "libnss3, libnspr4, libasound2t64.") in message


def test_the_libraries_a_banner_lists_for_an_unknown_distribution_are_named():
    kind, message = launch_errors.launch_failure(
        MISSING_DEPENDENCIES_OTHER, engine="bundled")

    assert kind == launch_errors.MISSING_LIBRARIES
    _assert_reads_as_localm(message)
    for library in ("libnss3.so", "libnspr4.so", "libasound.so.2"):
        assert library in message, message
    assert "Debian" not in message


def test_another_failure_keeps_its_reason_and_loses_playwrights_framing():
    raw = ("BrowserType.launch: Failed to launch: Error: spawn EACCES\n"
           "Call log:\n  - <launching> /opt/thing --flag\n")

    kind, message = launch_errors.launch_failure(raw, engine="bundled")

    assert kind == launch_errors.OTHER
    _assert_reads_as_localm(message)
    assert "spawn EACCES" in message


def test_a_boxed_banner_inside_an_unrecognised_failure_is_dropped():
    raw = ("BrowserType.launch: Something odd happened\n"
           + _box("Please run the following command to download new browsers:",
                  "", "    playwright install", "", "<3 Playwright Team"))

    kind, message = launch_errors.launch_failure(raw, engine="bundled")

    assert kind == launch_errors.OTHER
    _assert_reads_as_localm(message)
    assert message == "Could not start the bundled browser: Something odd happened."


def test_a_browser_that_exits_with_no_known_cause_shows_what_it_printed():
    raw = ("BrowserType.launch: Target page, context or browser has been closed\n"
           "Browser logs:\n\n<launching> /opt/thing --flag\n<launched> pid=9\n"
           "[pid=9][err] FATAL: no display available\n"
           "Call log:\n  - <launching> /opt/thing --flag\n")

    kind, message = launch_errors.launch_failure(raw, engine="system",
                                                 browser="Brave")

    assert kind == launch_errors.OTHER
    _assert_reads_as_localm(message)
    assert "Target page, context or browser has been closed" in message
    assert "FATAL: no display available" in message
    assert "Brave" in message
    assert "--flag" not in message


def test_an_overlong_reason_is_cut():
    kind, message = launch_errors.launch_failure(
        "BrowserType.launch: " + "x" * 5000, engine="bundled")

    assert kind == launch_errors.OTHER
    assert len(message) < 600


def test_no_installed_browser_lists_every_family_looked_for():
    message = launch_errors.no_system_browser(discovery.LOOKED_FOR)

    _assert_reads_as_localm(message)
    for name in discovery.LOOKED_FOR:
        assert name in message
    assert "'bundled'" in message
    assert "download" in message.lower()


def test_installed_browsers_that_all_fail_are_each_named():
    kind, message = launch_errors.system_launch_failure([
        ("Chromium", RuntimeError(MISSING_LIBRARY_LINUX)),
        ("Brave", RuntimeError(EXECUTABLE_PATH_MISSING))])

    assert kind == launch_errors.OTHER
    _assert_reads_as_localm(message)
    assert "Chromium: missing system libraries: libnspr4.so" in message
    assert "Brave:" in message
    assert "'bundled'" in message


def test_a_single_failing_browser_gets_its_own_specific_message():
    kind, message = launch_errors.system_launch_failure(
        [("Chromium", RuntimeError(MISSING_LIBRARY_LINUX))])

    assert kind == launch_errors.MISSING_LIBRARIES
    assert message.startswith("Chromium is installed but cannot start")


# --------------------------------------------------------------------------- #
#  The dynamic loader names one library per start; ldd names them all         #
# --------------------------------------------------------------------------- #

LDD_OUTPUT = (
    "\tlinux-vdso.so.1 (0x00007ffd)\n"
    "\tlibnspr4.so => not found\n"
    "\tlibnss3.so => not found\n"
    "\tlibnssutil3.so => not found\n"
    "\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f0)\n"
    "\tlibasound.so.2 => not found\n")


def _pretend_linux(monkeypatch, output: str):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    monkeypatch.setattr(launch_errors, "sys", types.SimpleNamespace(platform="linux"))
    monkeypatch.setattr(launch_errors.subprocess, "run", fake_run)
    return calls


def test_every_library_ldd_reports_missing_is_named(tmp_path, monkeypatch):
    binary = tmp_path / "chrome"
    binary.write_bytes(b"")
    calls = _pretend_linux(monkeypatch, LDD_OUTPUT)
    raw = ("BrowserType.launch: Target page, context or browser has been closed\n"
           "Browser logs:\n\n<launched> pid=7\n"
           f"[pid=7][err] {binary}: error while loading shared libraries: "
           "libnspr4.so: cannot open shared object file\n"
           "Call log:\n"
           f"  - [pid=7][err] {binary}: error while loading shared libraries: "
           "libnspr4.so: cannot open shared object file\n")

    kind, message = launch_errors.launch_failure(raw, engine="bundled")

    assert kind == launch_errors.MISSING_LIBRARIES
    assert "Missing: libnspr4.so, libnss3.so, libnssutil3.so, libasound.so.2." in message
    assert len(calls) == 1, "the same executable was inspected twice"
    assert calls[0] == ["ldd", str(binary)]


def test_ldd_is_not_asked_about_a_file_that_is_not_there(tmp_path, monkeypatch):
    calls = _pretend_linux(monkeypatch, LDD_OUTPUT)
    gone = tmp_path / "gone" / "chrome"
    raw = (f"[pid=7][err] {gone}: error while loading shared libraries: "
           "libnspr4.so: cannot open shared object file")

    libraries, _ = launch_errors.missing_libraries(raw)

    assert libraries == ["libnspr4.so"]
    assert calls == []


def test_the_loaders_own_library_is_kept_when_ldd_cannot_run(tmp_path, monkeypatch):
    binary = tmp_path / "chrome"
    binary.write_bytes(b"")
    _pretend_linux(monkeypatch, "")
    raw = (f"[pid=7][err] {binary}: error while loading shared libraries: "
           "libnspr4.so: cannot open shared object file")

    libraries, _ = launch_errors.missing_libraries(raw)

    assert libraries == ["libnspr4.so"]


def test_ldd_failing_to_start_leaves_the_loaders_answer(tmp_path, monkeypatch):
    binary = tmp_path / "chrome"
    binary.write_bytes(b"")
    monkeypatch.setattr(launch_errors, "sys", types.SimpleNamespace(platform="linux"))

    def boom(cmd, **kw):
        raise FileNotFoundError("ldd")

    monkeypatch.setattr(launch_errors.subprocess, "run", boom)
    raw = (f"[pid=7][err] {binary}: error while loading shared libraries: "
           "libnspr4.so: cannot open shared object file")

    assert launch_errors.missing_libraries(raw)[0] == ["libnspr4.so"]


def test_an_executable_path_with_spaces_is_inspected_whole(tmp_path, monkeypatch):
    folder = tmp_path / "4TB DATA" / "pw"
    folder.mkdir(parents=True)
    binary = folder / "chrome"
    binary.write_bytes(b"")
    calls = _pretend_linux(monkeypatch, LDD_OUTPUT)
    raw = (f"[pid=7][err] {binary}: error while loading shared libraries: "
           "libnspr4.so: cannot open shared object file")

    launch_errors.missing_libraries(raw)

    assert calls == [["ldd", str(binary)]]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="ldd is Linux only")
def test_ldd_finds_nothing_missing_for_a_system_binary():
    assert launch_errors._ldd_missing("/bin/sh") == []


def test_ldd_is_never_run_off_linux(tmp_path, monkeypatch):
    binary = tmp_path / "chrome"
    binary.write_bytes(b"")
    monkeypatch.setattr(launch_errors, "sys", types.SimpleNamespace(platform="win32"))

    def boom(cmd, **kw):
        raise AssertionError("ldd must not run off Linux")

    monkeypatch.setattr(launch_errors.subprocess, "run", boom)

    assert launch_errors._ldd_missing(str(binary)) == []


# --------------------------------------------------------------------------- #
#  Against playwright's own launch failures                                   #
# --------------------------------------------------------------------------- #

def _real_launch_failure(**launch):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(**launch)
        except sync_api.Error as exc:
            return exc
        browser.close()
    pytest.skip("that browser launched here, so there is no failure to read")


def test_the_real_missing_build_error_loses_its_banner(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "no-browsers"))

    exc = _real_launch_failure(headless=True)

    assert "playwright install" in str(exc), (
        "playwright no longer phrases this failure as the test expects")
    kind, message = launch_errors.launch_failure(exc, engine="bundled")
    assert kind == launch_errors.BUNDLED_MISSING
    _assert_reads_as_localm(message)
    assert "localm setup-browser" in message


def test_the_real_headed_missing_build_error_loses_its_banner(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "no-browsers"))

    exc = _real_launch_failure(headless=False)

    kind, message = launch_errors.launch_failure(exc, engine="bundled")
    assert kind == launch_errors.BUNDLED_MISSING
    _assert_reads_as_localm(message)


def test_the_real_missing_channel_error_is_a_missing_system_browser():
    exc = _real_launch_failure(channel="chrome-canary")

    assert "chrome-canary" in str(exc)
    kind, message = launch_errors.launch_failure(
        exc, engine="system", browser="Google Chrome Canary")
    assert kind == launch_errors.SYSTEM_MISSING
    _assert_reads_as_localm(message)


def test_the_real_missing_executable_error_is_a_missing_system_browser(tmp_path):
    gone = tmp_path / "nowhere" / "chromium"

    exc = _real_launch_failure(executable_path=str(gone))

    kind, message = launch_errors.launch_failure(
        exc, engine="system", browser="Chromium")
    assert kind == launch_errors.SYSTEM_MISSING
    _assert_reads_as_localm(message)


# --------------------------------------------------------------------------- #
#  The session launches what it found and reports what it could not           #
# --------------------------------------------------------------------------- #

class _FakeChromium:
    """Answers each launch with the next outcome: an exception to raise, or
    the browser to return. Records every call's keyword arguments."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def launch(self, **kw):
        self.calls.append(kw)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakePlaywright:
    def __init__(self, *outcomes):
        self.chromium = _FakeChromium(*outcomes)

    async def stop(self):
        return None


def _session(engine, *outcomes, **kw):
    sess = bsession.BrowserSession("t-" + engine, engine=engine, **kw)
    sess._pw = _FakePlaywright(*outcomes)
    return sess


def _browsers(monkeypatch, *found):
    monkeypatch.setattr(discovery, "find_system_browsers", lambda: list(found))


CHROME = discovery.SystemBrowser("Google Chrome", "/opt/google/chrome/chrome", "chrome")
BRAVE = discovery.SystemBrowser("Brave", "/usr/bin/brave-browser")
CHROMIUM = discovery.SystemBrowser("Chromium", "/usr/bin/chromium")


def test_bundled_launch_failure_names_the_chromium_download():
    sess = _session("bundled", RuntimeError(BUNDLED_MISSING_LINUX))

    with pytest.raises(bsession.BrowserUnavailableError) as ei:
        asyncio.run(sess._launch_bundled())

    assert ei.value.kind == launch_errors.BUNDLED_MISSING
    _assert_reads_as_localm(str(ei.value))
    assert "localm setup-browser" in str(ei.value)


def test_the_raw_launch_error_is_kept_in_the_debug_log(caplog):
    sess = _session("bundled", RuntimeError(BUNDLED_MISSING_LINUX))

    with caplog.at_level(logging.DEBUG, logger="localm.browser.session"):
        with pytest.raises(bsession.BrowserUnavailableError):
            asyncio.run(sess._launch_bundled())

    assert "Executable doesn't exist at" in caplog.text
    assert "playwright install" in caplog.text


@pytest.fixture(autouse=True)
def _forget_missing_executables():
    def clear():
        with provision._missing_lock:
            provision._missing_executables.clear()
    clear()
    yield
    clear()


def test_a_missing_bundled_executable_is_remembered_until_it_exists(
        tmp_path, monkeypatch):
    full = tmp_path / "full" / "chrome"
    full.parent.mkdir()
    full.write_bytes(b"")
    shell = tmp_path / "shell" / "chrome-headless-shell"
    monkeypatch.setattr(provision, "chromium_executable_path", lambda: full)
    sess = _session("bundled", RuntimeError(
        BUNDLED_MISSING_LINUX.replace(HEADLESS_SHELL, str(shell))))
    assert provision.is_chromium_installed() is True

    with pytest.raises(bsession.BrowserUnavailableError):
        asyncio.run(sess._launch_bundled())

    assert provision.is_chromium_installed() is False
    shell.parent.mkdir()
    shell.write_bytes(b"")
    assert provision.is_chromium_installed() is True


def test_a_failure_with_another_cause_does_not_mark_the_browser_not_installed(
        tmp_path, monkeypatch):
    full = tmp_path / "full" / "chrome"
    full.parent.mkdir()
    full.write_bytes(b"")
    monkeypatch.setattr(provision, "chromium_executable_path", lambda: full)
    sess = _session("bundled", RuntimeError(MISSING_LIBRARY_LINUX))

    with pytest.raises(bsession.BrowserUnavailableError):
        asyncio.run(sess._launch_bundled())

    assert provision.is_chromium_installed() is True


def test_the_real_missing_headless_shell_marks_a_half_installed_build(
        tmp_path, monkeypatch):
    """Only the full browser is on disk: a headless launch names the headless
    shell as the missing executable, and the build stops counting as installed."""
    async_api = pytest.importorskip("playwright.async_api")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "browsers"))
    full = provision.chromium_executable_path()
    assert full is not None
    full.parent.mkdir(parents=True)
    full.write_bytes(b"")
    assert provision.is_chromium_installed() is True
    sess = bsession.BrowserSession("t-half", engine="bundled", headless=True)

    async def launch():
        sess._pw = await async_api.async_playwright().start()
        try:
            await sess._launch_bundled()
        finally:
            await sess._pw.stop()

    with pytest.raises(bsession.BrowserUnavailableError) as ei:
        asyncio.run(launch())

    assert ei.value.kind == launch_errors.BUNDLED_MISSING
    assert provision.is_chromium_installed() is False


def test_a_bundled_launch_passes_the_headless_setting_and_no_channel():
    sentinel = object()
    sess = _session("bundled", sentinel, headless=False)

    assert asyncio.run(sess._launch_bundled()) is sentinel

    assert sess._pw.chromium.calls == [{"headless": False}]
    assert sess.browser_name == "Bundled Chromium"


def test_the_system_engine_launches_chrome_by_its_channel(monkeypatch):
    _browsers(monkeypatch, CHROME)
    sentinel = object()
    sess = _session("system", sentinel)

    assert asyncio.run(sess._launch_system()) is sentinel

    assert sess._pw.chromium.calls == [{"headless": True, "channel": "chrome"}]
    assert sess.browser_name == "Google Chrome"


def test_the_system_engine_launches_a_browser_with_no_channel_by_its_path(monkeypatch):
    _browsers(monkeypatch, BRAVE)
    sess = _session("system", object())

    asyncio.run(sess._launch_system())

    assert sess._pw.chromium.calls == [
        {"headless": True, "executable_path": "/usr/bin/brave-browser"}]
    assert sess.browser_name == "Brave"


def test_the_system_engine_moves_on_when_the_first_browser_will_not_start(monkeypatch):
    _browsers(monkeypatch, CHROMIUM, BRAVE)
    sentinel = object()
    sess = _session("system", RuntimeError(MISSING_LIBRARY_LINUX), sentinel)

    assert asyncio.run(sess._launch_system()) is sentinel

    assert [c.get("executable_path") for c in sess._pw.chromium.calls] == [
        "/usr/bin/chromium", "/usr/bin/brave-browser"]
    assert sess.browser_name == "Brave"


def test_the_system_engine_with_no_browser_launches_nothing_and_says_what_it_sought(
        monkeypatch):
    _browsers(monkeypatch)
    sess = _session("system")

    with pytest.raises(bsession.BrowserUnavailableError) as ei:
        asyncio.run(sess._launch_system())

    assert ei.value.kind == launch_errors.SYSTEM_MISSING
    assert sess._pw.chromium.calls == []
    message = str(ei.value)
    _assert_reads_as_localm(message)
    for name in discovery.LOOKED_FOR:
        assert name in message
    assert "'bundled'" in message


def test_the_system_engine_reports_every_browser_that_failed(monkeypatch):
    _browsers(monkeypatch, CHROMIUM, BRAVE)
    sess = _session("system", RuntimeError(MISSING_LIBRARY_LINUX),
                    RuntimeError(EXECUTABLE_PATH_MISSING))

    with pytest.raises(bsession.BrowserUnavailableError) as ei:
        asyncio.run(sess._launch_system())

    message = str(ei.value)
    _assert_reads_as_localm(message)
    assert "Chromium" in message and "Brave" in message
    assert len(sess._pw.chromium.calls) == 2
    assert sess.browser_name is None


def test_a_single_system_failure_is_the_message_the_user_sees(monkeypatch):
    _browsers(monkeypatch, CHROME)
    sess = _session("system", RuntimeError(CHANNEL_MISSING_LINUX))

    with pytest.raises(bsession.BrowserUnavailableError) as ei:
        asyncio.run(sess._launch_system())

    assert ei.value.kind == launch_errors.SYSTEM_MISSING
    assert str(ei.value).startswith("Google Chrome could not be found")
    _assert_reads_as_localm(str(ei.value))
