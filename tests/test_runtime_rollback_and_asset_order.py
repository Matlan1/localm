# SPDX-License-Identifier: AGPL-3.0-or-later
"""The runtime card offers no Roll back and no pin for builds localm cannot roll
back or pin (amd-rocm, a --from build); an online release listing is matched in
the declared preference order; folder paths with brackets reach the console
intact."""

from __future__ import annotations

import pytest

import localm.setup_llama as sl
from tests.test_llama_runtime_version_pin import home  # noqa: F401


@pytest.mark.parametrize("backend", ["amd-rocm", "custom"])
def test_no_rollback_target_for_a_build_localm_does_not_choose(monkeypatch, tmp_path,
                                                               home, backend):  # noqa: F811
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_runtime_history(backend, "b1363")
    sl._record_runtime_history(backend, "b1363-cpu-b12086")
    sl._record_provisioned_backend(tmp_path, backend, build="b1363-cpu-b12086")
    assert sl.previous_tag(backend) is None
    sl._record_runtime_history("vulkan", "b10300")
    assert sl.previous_tag("vulkan") == "b10300", "other backends keep their history"


def test_the_update_check_reports_no_pin_and_no_rollback_for_amd_rocm(monkeypatch, tmp_path,
                                                                      home):  # noqa: F811
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    sl._record_runtime_history("amd-rocm", "b1363")
    sl._record_provisioned_backend(tmp_path, "amd-rocm", build="b1363-cpu-b12086")
    sl.set_pinned_tag("b12086")
    out = sl.check_runtime_update()
    assert out["backend"] == "amd-rocm"
    assert out["pinned"] is None
    assert out["previous"] is None


def test_the_online_listing_honours_the_declared_preference(monkeypatch):
    assets = [
        {"name": "llama-b12087-bin-ubuntu-sycl-fp32-x64.tar.gz",
         "browser_download_url": "https://example.invalid/fp32.tar.gz"},
        {"name": "llama-b12087-bin-ubuntu-sycl-fp16-x64.tar.gz",
         "browser_download_url": "https://example.invalid/fp16.tar.gz"},
    ]
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: assets)
    monkeypatch.setattr(sl, "_platform_key", lambda: "linux")
    monkeypatch.setattr(sl, "_machine_arch", lambda: "x64")
    url, _sha, _tag = sl._resolve_backend_asset("sycl", tag="b12087")
    assert url.endswith("fp16.tar.gz")


def test_every_runtime_folder_line_escapes_rich_markup():
    import inspect

    from localm.setup_llama import cli
    src = inspect.getsource(cli)
    for line in ("Already provisioned[/green]{escape(label)} at {escape(str(target))}",
                 "Copying binaries from [bold]{escape(str(src))}[/bold]",
                 "Native runtime ready[/bold green] -> {escape(str(d))}"):
        assert line in src
