# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm setup-browser` must wrap `python -m playwright install chromium`
honestly: refuse cleanly under the network policy instead of letting
playwright's own downloader surface a raw error, do nothing on the network
when Chromium is already present, and never report success it has not
verified on disk - including the case where playwright's
own exit code says success but the binary still is not there, and the case
where a `--force` reinstall fails after playwright has already removed the
previous build.
"""

from __future__ import annotations

import sys
import types

import pytest

from localm.browser import provision as bprovision


def _stub_playwright_importable(monkeypatch):
    """install_chromium()'s first line is `import playwright`, which fails
    outright in this suite's own CI environment (the [browser] extra is not
    part of .[dev,rag]). Every test below mocks is_chromium_installed/
    _stream_install to drive logic PAST that check, so it needs `import
    playwright` to merely succeed, not to be a real, usable package - a bare
    stub module does that without installing anything."""
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))


_REAL_LAUNCH_BUILD_LOCATIONS = bprovision.launch_build_locations


@pytest.fixture(autouse=True)
def _builds_cannot_be_listed(monkeypatch):
    """By default the build listing is unavailable, so is_chromium_installed
    checks only the full build's executable. Tests of the listing replace it."""
    monkeypatch.setattr(bprovision, "launch_build_locations", lambda: None)


# --------------------------------------------------------------------------- #
#  install_chromium: the playwright package itself is missing                 #
# --------------------------------------------------------------------------- #

def test_playwright_missing_returns_pip_install_message(monkeypatch):
    # sys.modules[name] = None is the standard way to make any `import name`
    # (bare or `from name.sub import x`) raise ImportError, without needing to
    # actually uninstall the real package from the suite's venv.
    monkeypatch.setitem(sys.modules, "playwright", None)

    result = bprovision.install_chromium()

    assert result.ok is False
    assert "localm[browser]" in result.message


# --------------------------------------------------------------------------- #
#  Already installed: free, no policy gate, no subprocess                     #
# --------------------------------------------------------------------------- #

def test_already_installed_skips_download_and_policy(monkeypatch):
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setattr(bprovision, "is_chromium_installed", lambda: True)
    monkeypatch.setattr(bprovision, "chromium_executable_path",
                        lambda: "/fake/chrome")

    def _boom(*a, **kw):
        raise AssertionError("must not invoke the installer when already present")
    monkeypatch.setattr(bprovision, "_stream_install", _boom)

    result = bprovision.install_chromium()

    assert result.ok is True
    assert result.already_installed is True
    assert "/fake/chrome" in result.message


# --------------------------------------------------------------------------- #
#  Network policy gate (only reached when a download is actually needed)      #
# --------------------------------------------------------------------------- #

def test_net_mode_off_refuses_before_any_subprocess(cli_runner, monkeypatch):
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setattr(bprovision, "is_chromium_installed", lambda: False)
    monkeypatch.setenv("LOCALM_NET_MODE", "off")

    def _boom(*a, **kw):
        raise AssertionError("must not invoke the installer under net_mode=off")
    monkeypatch.setattr(bprovision, "_stream_install", _boom)

    result = bprovision.install_chromium()

    assert result.ok is False
    assert "net_mode=off" in result.message
    assert "localm config net_mode ask" in result.message


def test_net_mode_off_exempted_by_config_proceeds(cli_runner, monkeypatch):
    _stub_playwright_importable(monkeypatch)
    from localm.config import update_config
    update_config(lambda c: c.update({"net_allow_model_downloads": True}))
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    calls = iter([False, True])   # not installed before, installed after
    monkeypatch.setattr(bprovision, "is_chromium_installed", lambda: next(calls))
    monkeypatch.setattr(bprovision, "chromium_executable_path",
                        lambda: "/fake/chrome")
    monkeypatch.setattr(bprovision, "_stream_install",
                        lambda cmd, **kw: (0, ["done"]))

    result = bprovision.install_chromium()

    assert result.ok is True, result.message


def test_force_reinstalls_even_when_already_present(cli_runner, monkeypatch):
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    monkeypatch.setattr(bprovision, "is_chromium_installed", lambda: True)
    monkeypatch.setattr(bprovision, "chromium_executable_path",
                        lambda: "/fake/chrome")
    seen = {}

    def _fake_stream(cmd, **kw):
        seen["cmd"] = cmd
        return 0, ["done"]
    monkeypatch.setattr(bprovision, "_stream_install", _fake_stream)

    result = bprovision.install_chromium(force=True)

    assert result.ok is True
    assert "--force" in seen["cmd"]


# --------------------------------------------------------------------------- #
#  Honest reporting of the subprocess outcome                                 #
# --------------------------------------------------------------------------- #

def test_subprocess_failure_is_reported_honestly(cli_runner, monkeypatch):
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    monkeypatch.setattr(bprovision, "is_chromium_installed", lambda: False)
    monkeypatch.setattr(bprovision, "_stream_install",
                        lambda cmd, **kw: (1, ["Error: connect ECONNREFUSED"]))

    result = bprovision.install_chromium()

    assert result.ok is False
    assert "code 1" in result.message
    assert "ECONNREFUSED" in result.message


def test_subprocess_success_with_missing_binary_is_not_reported_as_success(
        cli_runner, monkeypatch):
    # The installer's own exit code says 0, but the executable is still not on
    # disk afterward - this must NOT be reported as success (rule 5: never
    # trust the exit code alone).
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    monkeypatch.setattr(bprovision, "is_chromium_installed", lambda: False)
    monkeypatch.setattr(bprovision, "_stream_install",
                        lambda cmd, **kw: (0, ["nothing useful happened"]))

    result = bprovision.install_chromium()

    assert result.ok is False, (
        "a 0 exit code must not be trusted when the binary still is not there")
    assert "still not on disk" in result.message


def test_subprocess_success_with_binary_present_reports_success(
        cli_runner, monkeypatch):
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    calls = iter([False, True])   # not installed before, installed after
    monkeypatch.setattr(bprovision, "is_chromium_installed", lambda: next(calls))
    monkeypatch.setattr(bprovision, "chromium_executable_path",
                        lambda: "/fake/chrome-installed")
    monkeypatch.setattr(bprovision, "_stream_install",
                        lambda cmd, **kw: (0, ["downloaded to /fake/chrome-installed"]))

    result = bprovision.install_chromium()

    assert result.ok is True, result.message
    assert "/fake/chrome-installed" in result.message


def test_force_failure_warns_browser_now_uninstalled(cli_runner, monkeypatch):
    # was_installed=True (the pre-check), then False afterward: playwright's
    # --force removed the old build before the reinstall failed, so the
    # machine now has NO Chromium at all - the message must say so.
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    calls = iter([True, False])
    monkeypatch.setattr(bprovision, "is_chromium_installed", lambda: next(calls))
    monkeypatch.setattr(bprovision, "chromium_executable_path",
                        lambda: "/fake/chrome")
    monkeypatch.setattr(bprovision, "_stream_install",
                        lambda cmd, **kw: (1, ["Failed to install browsers"]))

    result = bprovision.install_chromium(force=True)

    assert result.ok is False
    assert "no Chromium build is installed right now" in result.message


# --------------------------------------------------------------------------- #
#  A launch that found the build incomplete                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _forget_missing_executables():
    def clear():
        with bprovision._missing_lock:
            bprovision._missing_executables.clear()
    clear()
    yield
    clear()


def _the_full_build_is_there(monkeypatch, tmp_path):
    exe = tmp_path / "chromium-1" / "chrome"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setattr(bprovision, "chromium_executable_path", lambda: exe)
    return exe


def _missing_shell_error(shell) -> str:
    return (f"BrowserType.launch: Executable doesn't exist at {shell}\n"
            "Looks like Playwright was just installed or updated.")


def test_a_reported_missing_executable_marks_the_browser_not_installed_until_it_exists(
        monkeypatch, tmp_path):
    _the_full_build_is_there(monkeypatch, tmp_path)
    shell = tmp_path / "shell-1" / "chrome-headless-shell"
    assert bprovision.is_chromium_installed() is True

    bprovision.note_missing_executable(_missing_shell_error(shell))

    assert bprovision.is_chromium_installed() is False
    shell.parent.mkdir()
    shell.write_bytes(b"")
    assert bprovision.is_chromium_installed() is True
    assert bprovision._missing_executables == set()


def test_an_error_that_names_no_missing_executable_changes_nothing(
        monkeypatch, tmp_path):
    _the_full_build_is_there(monkeypatch, tmp_path)

    bprovision.note_missing_executable("BrowserType.launch: Target closed")
    bprovision.note_missing_executable(
        "Chromium distribution 'chrome' is not found at /opt/google/chrome/chrome")

    assert bprovision.is_chromium_installed() is True


def test_the_full_build_being_absent_is_not_installed_whatever_was_reported(
        monkeypatch, tmp_path):
    gone = tmp_path / "gone" / "chrome"
    monkeypatch.setattr(bprovision, "chromium_executable_path", lambda: gone)

    assert bprovision.is_chromium_installed() is False


def test_the_installer_repairs_a_build_a_launch_found_incomplete(
        monkeypatch, tmp_path):
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    _the_full_build_is_there(monkeypatch, tmp_path)
    shell = tmp_path / "shell-1" / "chrome-headless-shell"
    bprovision.note_missing_executable(_missing_shell_error(shell))
    ran = []

    def fake_install(cmd, **kw):
        ran.append(cmd)
        shell.parent.mkdir()
        shell.write_bytes(b"")
        return 0, ["downloaded the headless shell"]
    monkeypatch.setattr(bprovision, "_stream_install", fake_install)

    result = bprovision.install_chromium()

    assert len(ran) == 1, "an incomplete build was reported as already installed"
    assert result.ok is True, result.message
    assert result.already_installed is False


def test_an_install_that_does_not_supply_the_missing_executable_is_not_success(
        monkeypatch, tmp_path):
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    _the_full_build_is_there(monkeypatch, tmp_path)
    bprovision.note_missing_executable(
        _missing_shell_error(tmp_path / "shell-1" / "chrome-headless-shell"))
    monkeypatch.setattr(bprovision, "_stream_install",
                        lambda cmd, **kw: (0, ["nothing useful happened"]))

    result = bprovision.install_chromium()

    assert result.ok is False
    assert "still not on disk" in result.message


# --------------------------------------------------------------------------- #
#  Both launch builds must have finished installing                          #
# --------------------------------------------------------------------------- #

_DRY_RUN_WINDOWS = r"""Chrome for Testing 149.0.7827.55 (playwright chromium v1228)
  Install location:    C:\Users\u\AppData\Local\ms-playwright\chromium-1228
  Download url:        https://cdn.playwright.dev/builds/cft/149.0.7827.55/win64/chrome-win64.zip

FFmpeg (playwright ffmpeg v1011)
  Install location:    C:\Users\u\AppData\Local\ms-playwright\ffmpeg-1011
  Download url:        https://cdn.playwright.dev/builds/ffmpeg/1011/ffmpeg-win64.zip
  Download fallback 1: https://playwright.download.prss.microsoft.com/ffmpeg-win64.zip

Chrome Headless Shell 149.0.7827.55 (playwright chromium-headless-shell v1228)
  Install location:    C:\Users\u\AppData\Local\ms-playwright\chromium_headless_shell-1228
  Download url:        https://cdn.playwright.dev/builds/cft/149.0.7827.55/win64/chrome-headless-shell-win64.zip
"""

_DRY_RUN_LINUX = """Chrome for Testing 150.0.7839.4 (playwright chromium v1243)
  Install location:    /home/u/.cache/ms-playwright/chromium-1243
  Download url:        https://cdn.playwright.dev/builds/cft/150.0.7839.4/linux64/chrome-linux64.zip

Chrome Headless Shell 150.0.7839.4 (playwright chromium-headless-shell v1243)
  Install location:    /home/u/.cache/ms-playwright/chromium_headless_shell-1243
  Download url:        https://cdn.playwright.dev/builds/cft/150.0.7839.4/linux64/chrome-headless-shell-linux64.zip
"""


def test_the_dry_run_listing_maps_each_build_to_its_location():
    found = bprovision.parse_install_locations(_DRY_RUN_WINDOWS)

    assert set(found) == {"chromium", "ffmpeg", "chromium-headless-shell"}
    assert str(found["chromium-headless-shell"]) == (
        r"C:\Users\u\AppData\Local\ms-playwright\chromium_headless_shell-1228")
    assert str(found["chromium"]) == r"C:\Users\u\AppData\Local\ms-playwright\chromium-1228"


def test_a_linux_install_location_is_read_whole():
    found = bprovision.parse_install_locations(_DRY_RUN_LINUX)

    assert found["chromium"] == bprovision.Path("/home/u/.cache/ms-playwright/chromium-1243")
    assert found["chromium-headless-shell"] == bprovision.Path(
        "/home/u/.cache/ms-playwright/chromium_headless_shell-1243")


def _builds(tmp_path, *, full_done=True, shell_done=True, shell_dir=True):
    full = tmp_path / "registry" / "chromium-1"
    shell = tmp_path / "registry" / "chromium_headless_shell-1"
    full.mkdir(parents=True)
    if full_done:
        (full / "INSTALLATION_COMPLETE").write_text("")
    if shell_dir:
        shell.mkdir()
        if shell_done:
            (shell / "INSTALLATION_COMPLETE").write_text("")
    return {"chromium": full, "chromium-headless-shell": shell}


def test_a_missing_headless_shell_is_not_installed(monkeypatch, tmp_path):
    _the_full_build_is_there(monkeypatch, tmp_path)
    builds = _builds(tmp_path, shell_dir=False)
    monkeypatch.setattr(bprovision, "launch_build_locations", lambda: builds)

    assert bprovision.is_chromium_installed() is False


def test_a_build_whose_install_did_not_finish_is_not_installed(monkeypatch, tmp_path):
    _the_full_build_is_there(monkeypatch, tmp_path)
    builds = _builds(tmp_path, shell_done=False)
    monkeypatch.setattr(bprovision, "launch_build_locations", lambda: builds)

    assert bprovision.is_chromium_installed() is False


def test_both_builds_finished_is_installed(monkeypatch, tmp_path):
    _the_full_build_is_there(monkeypatch, tmp_path)
    builds = _builds(tmp_path)
    monkeypatch.setattr(bprovision, "launch_build_locations", lambda: builds)

    assert bprovision.is_chromium_installed() is True


def test_setup_runs_the_installer_when_only_the_headless_shell_is_missing(
        monkeypatch, tmp_path):
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    _the_full_build_is_there(monkeypatch, tmp_path)
    builds = _builds(tmp_path, shell_dir=False)
    monkeypatch.setattr(bprovision, "launch_build_locations", lambda: builds)
    ran = []

    def fake_install(cmd, **kw):
        ran.append(cmd)
        builds["chromium-headless-shell"].mkdir()
        (builds["chromium-headless-shell"] / "INSTALLATION_COMPLETE").write_text("")
        return 0, ["downloaded the headless shell"]
    monkeypatch.setattr(bprovision, "_stream_install", fake_install)

    result = bprovision.install_chromium()

    assert ran, "a build without its headless shell was reported as already installed"
    assert result.ok is True, result.message
    assert result.already_installed is False


def test_an_install_that_leaves_a_build_unfinished_is_not_success(monkeypatch, tmp_path):
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    _the_full_build_is_there(monkeypatch, tmp_path)
    builds = _builds(tmp_path, shell_done=False)
    monkeypatch.setattr(bprovision, "launch_build_locations", lambda: builds)
    monkeypatch.setattr(bprovision, "_stream_install",
                        lambda cmd, **kw: (0, ["interrupted"]))

    result = bprovision.install_chromium()

    assert result.ok is False


def test_a_failed_dry_run_is_logged_and_lists_nothing(monkeypatch, caplog):
    monkeypatch.setattr(bprovision, "launch_build_locations", _REAL_LAUNCH_BUILD_LOCATIONS)
    monkeypatch.setattr(
        bprovision.subprocess, "run",
        lambda *a, **kw: bprovision.subprocess.CompletedProcess(
            a[0], 2, stdout="", stderr="unknown option --dry-run"))

    with caplog.at_level("WARNING", logger=bprovision.logger.name):
        assert bprovision.launch_build_locations() is None
    assert "unknown option --dry-run" in caplog.text


def test_without_a_listing_only_the_full_build_is_checked(monkeypatch, tmp_path):
    _the_full_build_is_there(monkeypatch, tmp_path)

    assert bprovision.is_chromium_installed() is True


def test_the_real_driver_lists_both_launch_builds(monkeypatch):
    pytest.importorskip("playwright")
    monkeypatch.setattr(bprovision, "launch_build_locations", _REAL_LAUNCH_BUILD_LOCATIONS)

    found = bprovision.launch_build_locations()

    assert found is not None
    assert set(found) == {"chromium", "chromium-headless-shell"}
    for path in found.values():
        assert path.is_absolute()


# --------------------------------------------------------------------------- #
#  The installer's output, line by line                                       #
# --------------------------------------------------------------------------- #

def test_installer_output_reaches_the_progress_sink_without_colour_codes():
    # A real child process that prints what playwright's installer prints:
    # dimmed text around a URL, then a progress bar line.
    script = (
        "import sys\n"
        "sys.stdout.write('Downloading Chrome \\x1b[2mfrom https://cdn.example/x.zip"
        "\\x1b[22m\\n')\n"
        "sys.stdout.write('|\\x1b[32m####\\x1b[0m    |  50% of 10 MiB\\n')\n")
    seen = []

    code, lines = bprovision._stream_install(
        [sys.executable, "-c", script], on_progress=seen.append, timeout=60)

    assert code == 0
    expected = ["Downloading Chrome from https://cdn.example/x.zip",
                "|####    |  50% of 10 MiB"]
    assert seen == expected
    assert lines == expected


def test_a_failing_installers_tail_carries_no_colour_codes(monkeypatch):
    _stub_playwright_importable(monkeypatch)
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    monkeypatch.setattr(bprovision, "is_chromium_installed", lambda: False)
    script = "import sys; sys.stdout.write('\\x1b[31mError: boom\\x1b[39m\\n'); sys.exit(3)"
    real_stream = bprovision._stream_install
    monkeypatch.setattr(
        bprovision, "_stream_install",
        lambda cmd, **kw: real_stream([sys.executable, "-c", script], **kw))

    result = bprovision.install_chromium()

    assert result.ok is False
    assert "Error: boom" in result.message
    assert "\x1b" not in result.message


# --------------------------------------------------------------------------- #
#  download_allowed: the policy answer the GUI route shares with the CLI      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["ask", "allow"])
def test_downloads_are_allowed_whenever_network_access_is_not_off(monkeypatch, mode):
    monkeypatch.setenv("LOCALM_NET_MODE", mode)
    assert bprovision.download_allowed() is True


def test_downloads_are_refused_when_network_access_is_off(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    assert bprovision.download_allowed() is False


def test_downloads_are_allowed_while_off_when_the_config_exempts_them(monkeypatch):
    from localm.config import update_config
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    update_config(lambda c: c.update({"net_allow_model_downloads": True}))
    assert bprovision.download_allowed() is True


# --------------------------------------------------------------------------- #
#  CLI wiring (localm/cli/browser.py + its registration on main)              #
# --------------------------------------------------------------------------- #

def test_setup_browser_registered_on_main_group():
    from localm.cli import main
    assert "setup-browser" in main.commands


def test_setup_browser_cli_success(cli_runner, monkeypatch):
    from localm.cli.browser import setup_browser
    monkeypatch.setattr(
        "localm.browser.provision.install_chromium",
        lambda force=False, on_progress=None: bprovision.ProvisionResult(
            ok=True, message="Chromium installed at /fake/chrome."))

    result = cli_runner.invoke(setup_browser, [])

    assert result.exit_code == 0, result.output
    assert "Chromium installed at /fake/chrome." in result.output


def test_setup_browser_cli_failure_exits_nonzero(cli_runner, monkeypatch):
    from localm.cli.browser import setup_browser
    monkeypatch.setattr(
        "localm.browser.provision.install_chromium",
        lambda force=False, on_progress=None: bprovision.ProvisionResult(
            ok=False, message="Network access is disabled (net_mode=off)."))

    result = cli_runner.invoke(setup_browser, [])

    assert result.exit_code != 0
    assert "net_mode=off" in result.output


def test_setup_browser_cli_force_flag_forwarded(cli_runner, monkeypatch):
    from localm.cli.browser import setup_browser
    seen = {}

    def _fake_install(force=False, on_progress=None):
        seen["force"] = force
        return bprovision.ProvisionResult(ok=True, message="ok")
    monkeypatch.setattr("localm.browser.provision.install_chromium", _fake_install)

    result = cli_runner.invoke(setup_browser, ["--force"])

    assert result.exit_code == 0, result.output
    assert seen["force"] is True
