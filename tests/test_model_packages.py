# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adding the packages a trusted model's own code needs: only reviewed modules,
a plan resolved with every installed package held, consent before anything is
installed, an install of exactly the plan, removal again when the result does
not import, and the routes and command that drive it."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import install_manifest
from localm.inference.backends import hf as hf_backend
from localm.model_manager import missing_packages as mp
from localm.model_manager import model_packages as mpk

H1 = "sha256:" + "a" * 64
H2 = "sha256:" + "b" * 64


def _compile_output(*pins) -> str:
    out = []
    for name, version in pins:
        out.append(f"{name}=={version} \\\n    --hash={H1} \\\n    --hash={H2}\n")
    return "".join(out)


class FakeRun:
    """Stands in for subprocess.run of ``uv pip compile``: records each call and
    the constraints file it was given, and answers with *stdout*/*rc*."""

    def __init__(self, stdout="", rc=0, stderr=""):
        self.stdout, self.rc, self.stderr = stdout, rc, stderr
        self.calls = []
        self.constraints = None

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        self.constraints = Path(cmd[cmd.index("-c") + 1]).read_text(encoding="utf-8")
        return SimpleNamespace(returncode=self.rc, stdout=self.stdout, stderr=self.stderr)


@pytest.fixture
def online(monkeypatch):
    """Network policy allows the indexes; uv is found; torch is the PyPI build."""
    monkeypatch.setattr(mpk, "check_network", lambda backend: None)
    monkeypatch.setattr(mpk, "_uv", lambda: "uv")
    monkeypatch.setattr(mpk, "torch_from_elsewhere", lambda: None)
    monkeypatch.setattr(mpk, "installed_distributions",
                        lambda: {"torch": "2.9.0", "numpy": "2.3.1", "pyyaml": "6.0.2"})


@pytest.fixture
def available(monkeypatch):
    monkeypatch.setattr(mpk, "_importable", lambda m: True)


# ----------------------------------------------------------------- the list

def test_a_module_not_on_the_list_is_never_installable():
    assert mpk.unavailable_reason("localm_no_such_package_zz").startswith(
        "it is not on the list")
    assert not mpk.installable("localm_no_such_package_zz")


def test_a_refused_module_gives_its_reason():
    assert mpk.unavailable_reason("ebfm") == mpk.REFUSED["ebfm"]
    assert "source code" in mpk.unavailable_reason("flash_attn")


def test_a_listed_module_for_another_platform_is_not_installable(monkeypatch, available):
    monkeypatch.setattr(sys, "platform", "win32")
    assert "operating system" in mpk.unavailable_reason("fla")
    monkeypatch.setattr(sys, "platform", "linux")
    assert mpk.unavailable_reason("fla") is None


def test_a_listed_module_whose_prerequisite_is_missing_is_not_installable(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(mpk, "_importable", lambda m: m != "triton")
    assert mpk.unavailable_reason("fla") == "it needs triton, which this install does not have"


def test_the_lists_do_not_overlap_each_other_or_what_localm_ships():
    assert not set(mpk.ON_DEMAND) & set(mpk.REFUSED)
    assert not set(mpk.ON_DEMAND) & set(mp.SHIPPED_MODULES)
    assert not set(mpk.REFUSED) & set(mp.SHIPPED_MODULES)


def test_the_census_reads_the_same_lists():
    from tests.test_missing_packages import _census_module
    census = _census_module()
    assert census.ON_DEMAND is mpk.ON_DEMAND
    assert census.ACCEPTED_REFUSALS is mpk.REFUSED


# ----------------------------------------------------------------- planning

def test_plan_refuses_a_module_not_on_the_list_without_running_uv(online):
    run = FakeRun()
    with pytest.raises(mpk.PlanError, match="localm cannot add"):
        mpk.plan(["localm_no_such_package_zz"], run=run)
    assert run.calls == []


def test_plan_refuses_when_the_network_is_off_without_running_uv(monkeypatch, available):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    monkeypatch.setattr(mpk, "_uv", lambda: "uv")
    monkeypatch.setattr(mpk, "torch_from_elsewhere", lambda: None)
    run = FakeRun()
    with pytest.raises(mpk.PlanError, match="net_mode=off"):
        mpk.plan(["timm"], run=run)
    assert run.calls == []


def test_plan_holds_every_installed_package_and_lists_only_new_ones(online, available):
    run = FakeRun(_compile_output(("timm", "1.0.30"), ("torch", "2.9.0"), ("pyyaml", "6.0.2"),
                                  ("torchvision", "0.24.0")))
    p = mpk.plan(["timm"], run=run)
    assert p.adds == [("timm", "1.0.30"), ("torchvision", "0.24.0")]
    assert p.requirements.splitlines() == [
        f"timm==1.0.30 --hash={H1} --hash={H2}", f"torchvision==0.24.0 --hash={H1} --hash={H2}"]
    cmd, kw = run.calls[0]
    assert cmd[:3] == ["uv", "pip", "compile"]
    for flag in ("--generate-hashes", "--no-config"):
        assert flag in cmd
    assert cmd[cmd.index("--only-binary") + 1] == ":all:"
    assert cmd[cmd.index("--default-index") + 1] == mpk.PYPI_INDEX
    assert cmd[cmd.index("--python") + 1] == sys.executable
    assert "--torch-backend" not in cmd
    assert kw["input"] == "timm>=1.0.30\n"
    assert run.constraints.splitlines() == ["numpy==2.3.1", "pyyaml==6.0.2", "torch==2.9.0"]


def test_plan_takes_torch_built_parts_from_the_index_of_the_installed_torch(online, available,
                                                                          monkeypatch):
    monkeypatch.setattr(mpk, "installed_distributions", lambda: {"torch": "2.9.0+cu126"})
    run = FakeRun(_compile_output(("torchaudio", "2.9.0+cu126"), ("torch", "2.9.0+cu126")))
    p = mpk.plan(["torchaudio"], run=run)
    cmd = run.calls[0][0]
    assert cmd[cmd.index("--torch-backend") + 1] == "cu126"
    assert p.torch_backend == "cu126"
    assert p.as_dict()["adds"] == [{"name": "torchaudio", "version": "2.9.0+cu126",
                                    "source": "download.pytorch.org (cu126)"}]


@pytest.mark.parametrize("version,label", [
    ("2.9.0", None), ("2.9.0+cpu", "cpu"), ("2.9.0+cu126", "cu126"),
    ("2.9.0+rocm6.4", "rocm6.4"), ("2.9.0+xpu", "xpu"), ("2.9.0+weird;x", None),
])
def test_torch_backend_label(version, label):
    assert mpk.torch_backend({"torch": version}) == label
    assert mpk.torch_backend({}) is None


def test_plan_refuses_a_resolution_that_changes_an_installed_package(online, available):
    run = FakeRun(_compile_output(("timm", "1.0.30"), ("numpy", "1.26.4")))
    with pytest.raises(mpk.PlanError, match="change numpy 2.3.1 to 1.26.4"):
        mpk.plan(["timm"], run=run)


def test_plan_refuses_to_add_torch(online, available, monkeypatch):
    monkeypatch.setattr(mpk, "installed_distributions", lambda: {"numpy": "2.3.1"})
    run = FakeRun(_compile_output(("timm", "1.0.30"), ("torch", "2.9.0")))
    with pytest.raises(mpk.PlanError, match="would add torch"):
        mpk.plan(["timm"], run=run)


def test_plan_refuses_a_pin_without_a_hash(online, available):
    run = FakeRun("timm==1.0.30\n")
    with pytest.raises(mpk.PlanError, match="no hash"):
        mpk.plan(["timm"], run=run)


def test_plan_reports_what_uv_said_when_it_cannot_resolve(online, available):
    run = FakeRun(rc=1, stderr="No solution found: timm has no wheel")
    with pytest.raises(mpk.PlanError, match="No solution found: timm has no wheel"):
        mpk.plan(["timm"], run=run)


def test_plan_refuses_a_torch_installed_from_a_file_or_url_without_running_uv(online, available,
                                                                          monkeypatch):
    monkeypatch.setattr(mpk, "torch_from_elsewhere", lambda: "2.11.0+rocm7.13.0")
    run = FakeRun()
    with pytest.raises(mpk.PlanError, match=r"PyTorch \(2.11.0\+rocm7.13.0\) did not come from"):
        mpk.plan(["timm"], run=run)
    assert run.calls == []


def test_a_torch_label_uv_does_not_know_is_reported_as_an_unmatched_torch(online, available,
                                                                          monkeypatch):
    monkeypatch.setattr(mpk, "installed_distributions", lambda: {"torch": "2.11.0+rocm9.9"})
    run = FakeRun(rc=2, stderr="error: invalid value 'rocm9.9' for '--torch-backend <BACKEND>'")
    with pytest.raises(mpk.PlanError, match=r"PyTorch \(2.11.0\+rocm9.9\) did not come from"):
        mpk.plan(["timm"], run=run)


def test_an_unreachable_index_is_reported_as_such(online, available):
    run = FakeRun(rc=2, stderr="error: Failed to fetch: `https://pypi.org/simple/timm/`")
    with pytest.raises(mpk.PlanError, match="could not reach the package index"):
        mpk.plan(["timm"], run=run)


def test_torch_from_elsewhere_reads_direct_url(monkeypatch):
    class Dist:
        version = "2.11.0+rocm7.13.0"

        def __init__(self, direct):
            self.direct = direct

        def read_text(self, name):
            return self.direct if name == "direct_url.json" else None

    monkeypatch.setattr(mpk.md, "distribution", lambda name: Dist('{"url": "https://x/t.whl"}'))
    assert mpk.torch_from_elsewhere() == "2.11.0+rocm7.13.0"
    monkeypatch.setattr(mpk.md, "distribution", lambda name: Dist(None))
    assert mpk.torch_from_elsewhere() is None

    def absent(name):
        raise mpk.md.PackageNotFoundError(name)
    monkeypatch.setattr(mpk.md, "distribution", absent)
    assert mpk.torch_from_elsewhere() is None


def test_parse_pins_reads_continuations_and_rejects_a_loose_requirement():
    pins = mpk.parse_pins(_compile_output(("Open_Clip-Torch", "3.3.0")) + "# comment\n")
    assert pins == {"open-clip-torch": ("Open_Clip-Torch", "3.3.0", [H1, H2])}
    with pytest.raises(mpk.PlanError):
        mpk.parse_pins("timm>=1.0\n")


def test_installed_distributions_leave_out_localm():
    assert "localm" not in mpk.installed_distributions()
    assert mpk.installed_distributions().get("pytest")


# ----------------------------------------------------------------- the plan store

def test_a_plan_is_taken_once_and_expires(monkeypatch):
    p = mpk.Plan(modules=["timm"], adds=[], requirements="\n")
    mpk.remember(p)
    assert mpk.take(p.id) is p
    assert mpk.take(p.id) is None
    old = mpk.Plan(modules=["timm"], adds=[], requirements="\n")
    old.created = time.monotonic() - mpk._PLAN_TTL_S - 1
    mpk.remember(old)
    assert mpk.take(old.id) is None
    assert mpk.take("nope") is None and mpk.take(None) is None


# ----------------------------------------------------------------- installing

@pytest.fixture
def fake_uv(monkeypatch):
    """Records the uv commands install runs and the requirements file it passed."""
    calls = SimpleNamespace(cmds=[], reqs=None, import_error=None, rc=0, uninstall_rc=0,
                            recorded=[], installed={}, leaves_nothing=False, install_error=None,
                            uninstall_error=None)

    def stream(cmd, on_progress):
        calls.cmds.append(cmd)
        if "-r" in cmd:
            calls.reqs = Path(cmd[cmd.index("-r") + 1]).read_text(encoding="utf-8")
            if not calls.leaves_nothing:
                calls.installed["timm"] = "1.0.30"
            if calls.install_error is not None:
                raise calls.install_error
            return calls.rc, "uv output"
        if calls.uninstall_error is not None:
            raise calls.uninstall_error
        if calls.uninstall_rc == 0:
            for name in cmd[cmd.index("--python") + 2:]:
                calls.installed.pop(name, None)
        return calls.uninstall_rc, "uv output"

    monkeypatch.setattr(mpk, "check_network", lambda backend: None)
    monkeypatch.setattr(mpk, "_uv", lambda: "uv")
    monkeypatch.setattr(mpk, "_stream", stream)
    monkeypatch.setattr(mpk, "_imports_fail", lambda modules: calls.import_error)
    monkeypatch.setattr(mpk, "_record", lambda p: calls.recorded.append(p))
    monkeypatch.setattr(mpk, "installed_distributions", lambda: dict(calls.installed))
    return calls


def _plan():
    return mpk.Plan(modules=["timm"], adds=[("timm", "1.0.30")],
                    requirements=f"timm==1.0.30 --hash={H1}\n")


def test_install_installs_exactly_the_plan_hash_checked(fake_uv):
    p = _plan()
    result = mpk.install(p)
    assert result.ok and result.installed == ["timm==1.0.30"]
    cmd = fake_uv.cmds[0]
    assert cmd[:3] == ["uv", "pip", "install"]
    for flag in ("--no-deps", "--require-hashes", "--no-config"):
        assert flag in cmd
    assert cmd[cmd.index("--only-binary") + 1] == ":all:"
    assert fake_uv.reqs == p.requirements
    assert fake_uv.recorded == [p]


def test_install_removes_what_it_added_when_a_module_does_not_import(fake_uv):
    fake_uv.import_error = "ImportError: boom"
    result = mpk.install(_plan())
    assert not result.ok
    assert "do not import: ImportError: boom" in result.error
    assert "removed again" in result.error
    assert fake_uv.cmds[-1][:3] == ["uv", "pip", "uninstall"] and fake_uv.cmds[-1][-1] == "timm"
    assert "--no-config" in fake_uv.cmds[-1]
    assert fake_uv.recorded == []


def test_an_added_torchvision_must_import_too(fake_uv, monkeypatch):
    checked = []
    monkeypatch.setattr(mpk, "_imports_fail", lambda modules: checked.append(modules) or None)
    p = mpk.Plan(modules=["timm"], adds=[("timm", "1.0.30"), ("torchvision", "0.24.0")],
                 requirements="x\n")
    assert mpk.install(p).ok
    assert checked == [["timm", "torchvision"]]


def test_a_failed_uv_install_removes_what_it_left_and_records_nothing(fake_uv):
    fake_uv.rc = 2
    result = mpk.install(_plan())
    assert not result.ok and "uv could not install" in result.error
    assert "removed again" in result.error
    assert fake_uv.cmds[-1][:3] == ["uv", "pip", "uninstall"] and fake_uv.cmds[-1][-1] == "timm"
    assert fake_uv.recorded == []


def test_a_failed_removal_is_reported(fake_uv):
    fake_uv.import_error = "ImportError: boom"
    fake_uv.uninstall_rc = 1
    result = mpk.install(_plan())
    assert not result.ok and "removing the packages added failed too: uv output" in result.error


def test_a_failed_uv_install_that_left_nothing_uninstalls_nothing(fake_uv):
    fake_uv.leaves_nothing = True
    fake_uv.rc = 2
    result = mpk.install(_plan())
    assert not result.ok and "removed again" not in result.error
    assert len(fake_uv.cmds) == 1 and fake_uv.recorded == []


def test_a_run_that_does_not_finish_is_stopped():
    started = time.monotonic()
    rc, out = mpk._stream([sys.executable, "-c", "import time; print('x', flush=True); "
                           "time.sleep(60)"], None, timeout=2)
    assert rc != 0 and "stopped after 2 seconds" in out and "x" in out
    assert time.monotonic() - started < 30


def test_uv_index_settings_from_the_environment_are_not_passed_on(monkeypatch):
    monkeypatch.setenv("UV_INDEX_URL", "https://evil.example/simple")
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://evil.example/simple")
    monkeypatch.setenv("UV_INDEX_STRATEGY", "unsafe-best-match")
    env = mpk._run_kwargs()["env"]
    assert not any(k.upper().startswith("UV_") and k.upper() != "UV_CACHE_DIR" for k in env)
    assert env["UV_CACHE_DIR"]


def test_a_second_install_at_the_same_time_is_refused(fake_uv):
    assert mpk._INSTALL_LOCK.acquire(blocking=False)
    try:
        result = mpk.install(_plan())
    finally:
        mpk._INSTALL_LOCK.release()
    assert not result.ok and "another package install" in result.error
    assert fake_uv.cmds == []


def test_install_rechecks_the_network_policy(fake_uv, monkeypatch):
    def refuse(backend):
        raise mpk.PlanError("localm cannot download the packages: off")
    monkeypatch.setattr(mpk, "check_network", refuse)
    result = mpk.install(_plan())
    assert not result.ok and "off" in result.error and fake_uv.cmds == []


def test_the_import_check_runs_in_a_fresh_interpreter():
    assert mpk._imports_fail(["json", "pytest"]) is None
    assert "localm_no_such_package_zz" in mpk._imports_fail(["localm_no_such_package_zz"])


def test_the_import_check_does_not_run_a_module_planted_in_the_current_folder(
        tmp_path, monkeypatch):
    marker = tmp_path / "ran.txt"
    (tmp_path / "localm_planted_zz.py").write_text(
        f"open({str(marker)!r}, 'w').write('ran')\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    error = mpk._imports_fail(["localm_planted_zz"])
    assert error is not None and "localm_planted_zz" in error
    assert not marker.exists()


def test_stream_reads_utf8_output_whatever_the_locale_code_page():
    text = "Using environment at: C:\\Users\\\u7528\u6237\\.venv"
    code = f"import sys; sys.stdout.buffer.write({text!r}.encode('utf-8'))"
    rc, out = mpk._stream([sys.executable, "-c", code], None)
    assert rc == 0 and out == text


def test_stream_survives_bytes_that_are_not_text():
    code = "import sys; sys.stdout.buffer.write(b'before \\xff\\xfe after\\n')"
    rc, out = mpk._stream([sys.executable, "-c", code], None)
    assert rc == 0 and "before" in out and "after" in out and "\ufffd" in out


def test_plan_decodes_uv_output_as_utf8_with_replacement(online, available):
    run = FakeRun(stdout=_compile_output(("timm", "1.0.30")))
    mpk.plan(["timm"], run=run)
    kw = run.calls[0][1]
    assert kw["encoding"] == "utf-8" and kw["errors"] == "replace" and "text" not in kw


def test_an_install_that_crashes_mid_stream_still_removes_what_it_added(fake_uv):
    fake_uv.install_error = UnicodeDecodeError("cp936", b"\xff", 0, 1, "illegal multibyte sequence")
    result = mpk.install(_plan())
    assert not result.ok and "stopped unexpectedly" in result.error
    assert "removed again" in result.error
    assert fake_uv.installed == {} and fake_uv.recorded == []
    assert fake_uv.cmds[-1][:3] == ["uv", "pip", "uninstall"]


def test_an_install_whose_uv_cannot_be_started_reports_it(fake_uv):
    fake_uv.leaves_nothing = True
    fake_uv.install_error = FileNotFoundError("uv")
    result = mpk.install(_plan())
    assert not result.ok and "uv could not be started" in result.error
    assert fake_uv.recorded == []


def test_a_removal_that_cannot_start_uv_is_reported(fake_uv):
    fake_uv.import_error = "ImportError: boom"
    fake_uv.uninstall_error = OSError("no uv")
    result = mpk.install(_plan())
    assert not result.ok and "removing the packages added failed too: no uv" in result.error


def test_a_stale_plan_whose_package_another_install_added_installs_nothing(fake_uv):
    """The first of two plans that both list torchvision has installed it since."""
    second = mpk.Plan(modules=["open_clip"],
                      adds=[("open-clip-torch", "3.3.0"), ("torchvision", "0.24.0")],
                      requirements="y\n")
    fake_uv.installed.update({"timm": "1.0.30", "torchvision": "0.24.0"})
    fake_uv.import_error = "ImportError: boom"
    result = mpk.install(second)
    assert not result.ok and "out of date" in result.error and "torchvision" in result.error
    assert fake_uv.cmds == []
    assert fake_uv.installed == {"timm": "1.0.30", "torchvision": "0.24.0"}
    assert fake_uv.recorded == []


def test_a_plan_made_after_the_first_install_still_installs(fake_uv):
    fake_uv.installed.update({"torchvision": "0.24.0"})
    p = mpk.Plan(modules=["timm"], adds=[("timm", "1.0.30")], requirements="x\n")
    result = mpk.install(p)
    assert result.ok and fake_uv.installed == {"torchvision": "0.24.0", "timm": "1.0.30"}


# ----------------------------------------------------------------- the install record

def _manifest(root: Path, **extra) -> None:
    data = {"schema": install_manifest.SCHEMA_VERSION, "root": str(root), "venv": "v", **extra}
    (root / install_manifest.MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")


def test_record_model_packages_adds_and_replaces_by_project(tmp_path):
    _manifest(tmp_path, model_packages=[{"project": "timm", "version": "1.0.0"}])
    install_manifest.record_model_packages(tmp_path, [
        {"project": "timm", "version": "1.0.30", "for_modules": ["timm"]},
        {"project": "torchvision", "version": "0.24.0", "for_modules": ["timm"]}])
    data = install_manifest.load(tmp_path)
    assert data["venv"] == "v"
    assert data["model_packages"] == [
        {"project": "timm", "version": "1.0.30", "for_modules": ["timm"]},
        {"project": "torchvision", "version": "0.24.0", "for_modules": ["timm"]}]


def test_record_model_packages_without_a_manifest_writes_nothing(tmp_path):
    assert install_manifest.record_model_packages(tmp_path, [{"project": "timm"}]) is None
    assert list(tmp_path.iterdir()) == []


def test_record_model_packages_refuses_a_newer_manifest(tmp_path):
    _manifest(tmp_path, schema=install_manifest.SCHEMA_VERSION + 1)
    with pytest.raises(ValueError):
        install_manifest.record_model_packages(tmp_path, [{"project": "timm"}])


def test_the_record_names_each_package_its_modules_and_the_day(monkeypatch):
    seen = []
    monkeypatch.setattr(install_manifest, "record_model_packages",
                        lambda root, entries: seen.append(entries))
    mpk._record(_plan())
    assert seen == [[{"project": "timm", "version": "1.0.30", "for_modules": ["timm"],
                      "installed": time.strftime("%Y-%m-%d")}]]


def test_recorded_lists_what_the_install_record_holds(tmp_path, monkeypatch):
    from localm import uv_tool
    monkeypatch.setattr(uv_tool, "install_root", lambda: tmp_path)
    assert mpk.recorded() == []
    _manifest(tmp_path, model_packages=[{"project": "timm", "version": "1.0.30"}, "junk"])
    assert mpk.recorded() == [{"project": "timm", "version": "1.0.30"}]


def test_a_record_that_cannot_be_written_is_logged_and_install_still_succeeds(monkeypatch, caplog):
    def broken(root, entries):
        raise OSError("read-only")
    monkeypatch.setattr(install_manifest, "record_model_packages", broken)
    with caplog.at_level("WARNING", logger="localm"):
        mpk._record(_plan())
    assert "could not list the packages added (timm): read-only" in caplog.text


# ----------------------------------------------------------------- what the user is told

@pytest.fixture
def trusted(monkeypatch):
    monkeypatch.setattr(hf_backend, "_trust_remote_code_enabled", lambda: True)


def _model(tmp_path: Path, imports: str) -> Path:
    folder = tmp_path / "Some-Model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "architectures": ["XForCausalLM"], "model_type": "x",
        "auto_map": {"AutoModelForCausalLM": "modeling_x.XForCausalLM"}}), encoding="utf-8")
    (folder / "modeling_x.py").write_text(imports, encoding="utf-8")
    return folder


def test_an_addable_package_names_the_command_and_the_models_page(tmp_path, trusted, monkeypatch):
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "timm")
    reason = mp.folder_refusal(_model(tmp_path, "import timm\n"))
    assert "localm can add to its own environment" in reason
    assert "`localm install-packages Some-Model`" in reason and "Models page" in reason
    assert "pip" not in reason.lower()


def test_a_package_localm_cannot_add_gives_the_reason(tmp_path, trusted, monkeypatch):
    monkeypatch.setattr(mp, "is_importable", lambda m: m not in ("timm", "flash_attn"))
    reason = mp.folder_refusal(_model(tmp_path, "import timm\nimport flash_attn\n"))
    assert "localm cannot add it to its environment (flash-attn: " in reason
    assert "cannot run this model" in reason and "install-packages" not in reason


def test_an_addable_package_beside_a_missing_shipped_one_says_to_repair_too(tmp_path, trusted,
                                                                          monkeypatch):
    monkeypatch.setattr(mp, "is_importable", lambda m: m not in ("timm", "einops"))
    reason = mp.folder_refusal(_model(tmp_path, "import timm\nimport einops\n"))
    assert "needs the Python package timm, which localm can add" in reason
    assert "It also needs einops, which localm normally includes" in reason


def test_folder_state_sorts_the_missing_packages(tmp_path, trusted, monkeypatch):
    monkeypatch.setattr(mp, "is_importable", lambda m: m not in ("timm", "einops", "ebfm"))
    state = mpk.folder_state(_model(tmp_path, "import timm\nimport einops\nimport ebfm\n"))
    assert (state.shipped, state.blocked, state.addable) == (["einops"], ["ebfm"], ["timm"])
    assert state.as_dict() == {"missing": ["ebfm", "einops", "timm"], "addable": ["timm"],
                               "installable": False}


@pytest.mark.parametrize("plat, machine, label, module, source", [
    ("linux", "aarch64", "Linux ARM64", "decord", "decord"),
    ("win32", "ARM64", "Windows ARM64", "tiktoken", "tiktoken"),
    ("win32", "ARM64", "Windows ARM64", "cv2", "opencv"),
    ("darwin", "x86_64", "macOS x64", "monai", "monai"),
])
def test_a_module_with_no_build_for_this_platform_is_blocked(
        tmp_path, trusted, monkeypatch, plat, machine, label, module, source):
    monkeypatch.setattr(mp.sys, "platform", plat)
    monkeypatch.setattr(mp.platform, "machine", lambda: machine)
    monkeypatch.setattr(mp, "is_importable", lambda m: m != module)
    state = mpk.folder_state(_model(tmp_path, f"import {module}\n"))
    assert (state.shipped, state.blocked, state.addable) == ([], [module], [])
    assert state.as_dict() == {"missing": [source], "addable": [], "installable": False}
    assert mpk.unavailable_reason(module) == f"it is not available for this platform ({label})"
    assert f"not available for this platform ({label})" in mp.refusal_for("m", state.missing)
    with pytest.raises(mpk.PlanError) as raised:
        mpk.plan([module])
    assert f"not available for this platform ({label})" in str(raised.value)


def test_a_module_shipped_for_this_platform_is_still_an_incomplete_install(
        tmp_path, trusted, monkeypatch):
    monkeypatch.setattr(mp.sys, "platform", "linux")
    monkeypatch.setattr(mp.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "decord")
    state = mpk.folder_state(_model(tmp_path, "import decord\n"))
    assert (state.shipped, state.blocked, state.addable) == (["decord"], [], [])
    assert "normally includes" in mp.refusal_for("m", state.missing)


def test_folder_state_scans_the_code_again_only_after_it_changed(tmp_path, trusted, monkeypatch):
    calls = []
    real = mp.folder_third_party_imports

    def counting(folder):
        calls.append(folder)
        return real(folder)
    monkeypatch.setattr(mp, "folder_third_party_imports", counting)
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "timm")
    folder = _model(tmp_path, "import timm\n")
    assert mpk.folder_state(folder).missing == ["timm"]
    assert mpk.folder_state(folder).missing == ["timm"]
    assert len(calls) == 1
    (folder / "modeling_x.py").write_text("import timm\nimport open_clip\n", encoding="utf-8")
    monkeypatch.setattr(mp, "is_importable", lambda m: m not in ("timm", "open_clip"))
    assert mpk.folder_state(folder).missing == ["open_clip", "timm"]
    assert len(calls) == 2


TIMM_TEXT = ("TimmWrapperModel requires the timm library but it was not found in your "
             "environment. You can install it with pip: `pip install timm`.")


def _folder_without_auto_map(tmp_path: Path) -> Path:
    folder = tmp_path / "Some-Model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "architectures": ["Gemma3nForConditionalGeneration"], "model_type": "gemma3n"}),
        encoding="utf-8")
    return folder


def _offers_the_install(sentence: str) -> bool:
    return "install-packages" in sentence or "Models page" in sentence


def test_a_package_a_native_model_needs_is_not_offered_for_install(tmp_path, trusted, monkeypatch):
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "timm")
    folder = _folder_without_auto_map(tmp_path)
    assert mpk.folder_state(folder).addable == []
    sentence = mp.rewrite_import_error("Some-Model", TIMM_TEXT, str(folder))
    assert sentence is not None and not _offers_the_install(sentence)
    assert "package timm" in sentence and "cannot add it for this model" in sentence
    assert "pip" not in sentence.lower()


def test_a_package_behind_a_guarded_import_is_not_offered_for_install(tmp_path, trusted,
                                                                    monkeypatch):
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "timm")
    folder = _model(tmp_path, "try:\n    import timm\nexcept ImportError:\n    timm = None\n")
    assert mpk.folder_state(folder).addable == []
    sentence = mp.rewrite_import_error("Some-Model", TIMM_TEXT, str(folder))
    assert sentence is not None and not _offers_the_install(sentence)


def test_a_package_the_models_code_imports_is_still_offered_for_install(tmp_path, trusted,
                                                                      monkeypatch):
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "timm")
    folder = _model(tmp_path, "import timm\n")
    assert mpk.folder_state(folder).addable == ["timm"]
    sentence = mp.rewrite_import_error("Some-Model", TIMM_TEXT, str(folder))
    assert sentence is not None and "`localm install-packages Some-Model`" in sentence


def test_a_load_error_with_no_folder_offers_no_install():
    sentence = mp.rewrite_import_error("Some-Model", TIMM_TEXT)
    assert sentence is not None and not _offers_the_install(sentence)


def test_the_load_error_of_a_native_model_does_not_offer_an_install(tmp_path, trusted,
                                                                  monkeypatch):
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "timm")
    folder = _folder_without_auto_map(tmp_path)
    out = hf_backend._without_pip_advice(str(folder), RuntimeError(TIMM_TEXT))
    assert not _offers_the_install(str(out))


def test_folder_state_checks_nothing_while_custom_code_is_not_trusted(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_backend, "_trust_remote_code_enabled", lambda: False)
    state = mpk.folder_state(_model(tmp_path, "import timm\n"))
    assert not state.trusted and state.missing == []


# ----------------------------------------------------------------- the routes

@pytest.fixture
def routes(tmp_path, monkeypatch, trusted):
    """The model routes on a bare app in open mode, on a loopback bind, with a
    registry holding one model whose code needs timm."""
    from localm.plugins.gui.routes import models as models_routes
    from localm.plugins.gui.routes.models import packages as packages_routes
    folder = _model(tmp_path, "import timm\n")
    registry = {"some-model": {"path": str(folder), "source": "hf"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "timm")
    app = FastAPI()
    app.state.bind_host = "127.0.0.1"
    models_routes.register(app, SimpleNamespace(active_model=lambda: "", switch_model=None,
                                                jobs=None))
    planned = []

    def fake_plan(modules):
        planned.append(modules)
        return mpk.Plan(modules=modules, adds=[("timm", "1.0.30")], requirements="timm\n")

    installs = []

    def fake_install(p, on_progress=None):
        installs.append(p)
        on_progress("installing timm")
        return mpk.InstallResult(ok=True, installed=["timm==1.0.30"])

    monkeypatch.setattr(mpk, "plan", fake_plan)
    monkeypatch.setattr(mpk, "install", fake_install)
    monkeypatch.setattr(packages_routes, "_task", {"current": None})
    return SimpleNamespace(client=TestClient(app), app=app, planned=planned, installs=installs)


def test_plan_then_install_installs_the_plan_shown(routes):
    r = routes.client.post("/api/models/packages/plan", json={"model": "some-model"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["adds"] == [{"name": "timm", "version": "1.0.30", "source": "pypi.org"}]
    assert routes.planned == [["timm"]]
    r = routes.client.post("/api/models/packages/install", json={"plan_id": body["plan_id"]})
    assert r.status_code == 200, r.text
    with routes.client.stream("GET", "/api/models/packages/install/events") as s:
        events = [json.loads(line[6:]) for line in s.iter_lines() if line.startswith("data: ")]
    assert events[-1] == {"type": "end", "ok": True, "installed": ["timm==1.0.30"],
                          "failed": [], "error": ""}
    assert {"type": "log", "line": "installing timm"} in events
    assert [p.id for p in routes.installs] == [body["plan_id"]]
    again = routes.client.post("/api/models/packages/install", json={"plan_id": body["plan_id"]})
    assert again.status_code == 404


def test_an_unknown_plan_installs_nothing(routes):
    r = routes.client.post("/api/models/packages/install", json={"plan_id": "made-up"})
    assert r.status_code == 404 and routes.installs == []


def test_an_unregistered_model_or_a_path_is_refused(routes, tmp_path):
    for name in ("nope", str(tmp_path / "Some-Model")):
        r = routes.client.post("/api/models/packages/plan", json={"model": name})
        assert r.status_code == 404
    assert routes.planned == []


def test_a_network_bind_refuses_every_packages_route(routes):
    routes.app.state.bind_host = "0.0.0.0"
    assert routes.client.post("/api/models/packages/plan",
                              json={"model": "some-model"}).status_code == 403
    assert routes.client.post("/api/models/packages/install",
                              json={"plan_id": "x"}).status_code == 403
    assert routes.client.get("/api/models/packages/install/events").status_code == 403
    assert routes.planned == [] and routes.installs == []


def test_a_model_needing_a_package_localm_cannot_add_is_refused_with_the_reason(routes,
                                                                               monkeypatch):
    monkeypatch.setattr(mp, "is_importable", lambda m: m not in ("timm", "ebfm"))
    from localm.config import load_registry
    Path(load_registry()["some-model"]["path"], "modeling_x.py").write_text(
        "import timm\nimport ebfm\n", encoding="utf-8")
    r = routes.client.post("/api/models/packages/plan", json={"model": "some-model"})
    assert r.status_code == 409 and "unrelated project" in r.json()["detail"]
    assert routes.planned == []


def test_a_second_install_while_one_runs_is_refused(routes, monkeypatch):
    from localm.plugins import deps_task
    from localm.plugins.gui.routes.models import packages as packages_routes
    monkeypatch.setattr(packages_routes, "_task", {"current": deps_task.DepInstallTask("x")})
    p = mpk.Plan(modules=["timm"], adds=[], requirements="\n")
    mpk.remember(p)
    r = routes.client.post("/api/models/packages/install", json={"plan_id": p.id})
    assert r.status_code == 409 and routes.installs == []
    assert mpk.take(p.id) is p


def test_the_model_list_says_which_packages_a_model_needs(routes):
    rows = routes.client.get("/api/models").json()
    rows = rows["models"] if isinstance(rows, dict) else rows
    row = next(r for r in rows if r["name"] == "some-model")
    assert row["packages"] == {"missing": ["timm"], "addable": ["timm"], "installable": True}


# ----------------------------------------------------------------- the command

@pytest.fixture
def cli(tmp_path, monkeypatch, trusted):
    from localm.cli import main
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "timm")
    folder = _model(tmp_path, "import timm\n")
    monkeypatch.setattr(mpk, "model_folder", lambda name: folder if name == "some-model" else None)
    monkeypatch.setattr(mpk, "plan", lambda modules: mpk.Plan(
        modules=modules, adds=[("timm", "1.0.30")], requirements="timm\n"))
    installs = []

    def fake_install(p, on_progress=None):
        installs.append(p)
        return mpk.InstallResult(ok=True, installed=["timm==1.0.30"])
    monkeypatch.setattr(mpk, "install", fake_install)
    return SimpleNamespace(run=lambda *a, **kw: CliRunner().invoke(main, ["install-packages", *a],
                                                                   **kw),
                           installs=installs)


def test_the_command_lists_the_packages_and_installs_nothing_unless_agreed(cli):
    result = cli.run("some-model", input="\n")
    assert result.exit_code == 1
    assert "timm 1.0.30" in result.output and "from pypi.org" in result.output
    assert "Nothing was installed." in result.output
    assert cli.installs == []


def test_the_command_installs_after_yes(cli):
    result = cli.run("some-model", input="y\n")
    assert result.exit_code == 0, result.output
    assert len(cli.installs) == 1 and "can now be loaded" in result.output


def test_the_command_with_yes_does_not_ask(cli):
    result = cli.run("some-model", "--yes")
    assert result.exit_code == 0 and "Install these packages?" not in result.output
    assert len(cli.installs) == 1


def test_the_command_lists_what_was_added(cli, monkeypatch):
    monkeypatch.setattr(mpk, "recorded", lambda: [
        {"project": "timm", "version": "1.0.30", "for_modules": ["timm"],
         "installed": "2026-11-03"}])
    result = cli.run("--list")
    assert result.exit_code == 0 and "timm 1.0.30" in result.output
    assert "for timm, 2026-11-03" in result.output and cli.installs == []


def test_the_command_needs_a_model_or_list(cli):
    result = cli.run()
    assert result.exit_code == 2 and "Name a model, or pass --list." in result.output


def test_the_command_refuses_an_unknown_model(cli):
    result = cli.run("nope")
    assert result.exit_code == 1 and "not a registered Hugging Face model folder" in result.output
    assert cli.installs == []


def test_model_folder_resolves_a_name_a_folder_name_or_a_path(tmp_path, monkeypatch):
    folder = _model(tmp_path, "")
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"alias-name": {"path": str(folder)}})
    monkeypatch.setattr("localm.model_manager.registry.get_model_path",
                        lambda name: folder if name == "alias-name" else None)
    assert mpk.model_folder("alias-name") == folder
    assert mpk.model_folder("Some-Model") == folder
    assert mpk.model_folder(str(folder)) == folder
    assert mpk.model_folder("nope") is None


def test_uv_is_the_one_installed_with_localm(monkeypatch):
    from localm import uv_tool
    monkeypatch.setattr(uv_tool, "find_uv", lambda root: "uv-on-path")
    monkeypatch.setattr(uv_tool, "bundled_uv", lambda: None)
    with pytest.raises(mpk.PlanError, match="cannot find the uv"):
        mpk._uv()
    monkeypatch.setattr(uv_tool, "bundled_uv", lambda: "/venv/bin/uv")
    assert mpk._uv() == "/venv/bin/uv"


def test_subprocesses_open_no_window_on_windows():
    kw = mpk._run_kwargs()
    if sys.platform == "win32":
        assert kw["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert "UV_CACHE_DIR" in kw["env"]
