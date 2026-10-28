# SPDX-License-Identifier: AGPL-3.0-or-later
"""The backend menu setup.bat and setup.sh print, per platform.

Both scripts print the rows of ``python -m localm.setup_spec backend-menu`` as
they come, so which backends a platform's menu lists, and under which numbers,
is decided here: [1] is the recommendation, then every backend setup-llama has
a build of for that OS and CPU architecture, then "own", numbered without gaps.
"""

from __future__ import annotations

import pytest

from localm import setup_spec


def _keys(rows):
    return [k for _n, k, _l in rows]


@pytest.mark.parametrize("plat, arch, offered", [
    ("linux", "x64", ["vulkan", "cuda", "hip", "sycl", "cpu"]),
    ("linux", "arm64", ["vulkan", "cuda", "cpu"]),
    ("linux", "s390x", ["cpu"]),
    ("win32", "x64", ["vulkan", "cuda", "hip", "amd-rocm", "sycl", "cpu"]),
    ("win32", "arm64", ["vulkan", "cuda", "cpu"]),
    ("darwin", "arm64", ["metal", "cpu"]),
    ("darwin", "x64", ["cpu"]),
])
def test_each_platform_lists_only_the_backends_it_has_a_build_of(plat, arch, offered):
    rows = setup_spec.backend_menu("cpu", plat, arch)
    assert _keys(rows) == ["cpu", *offered, "own"]
    assert [n for n, _k, _l in rows] == [str(i) for i in range(1, len(rows) + 1)]
    assert setup_spec.backend_hint(plat, arch) == "|".join(offered)


def test_the_recommendation_is_entry_1_and_its_twin_is_marked():
    rows = setup_spec.backend_menu("cuda", "linux", "x64")
    assert rows[0][1] == "cuda" and rows[0][2].endswith(setup_spec.RECOMMENDED_SUFFIX)
    twin = [r for r in rows[1:] if r[1] == "cuda"]
    assert len(twin) == 1 and twin[0][2].endswith(setup_spec.SAME_SUFFIX)
    assert sum(setup_spec.SAME_SUFFIX in r[2] for r in rows) == 1


def test_sycl_says_what_it_needs_per_platform():
    assert "self-contained" in setup_spec.backend_label("sycl", "win32")
    assert "oneAPI" in setup_spec.backend_label("sycl", "linux")


def test_a_platform_with_no_build_has_a_placeholder_hint():
    assert setup_spec.backend_hint("linux", "riscv64") == "backend"
    assert _keys(setup_spec.backend_menu("cpu", "linux", "riscv64")) == ["cpu", "own"]


def test_the_cli_prints_tab_separated_rows(capsys):
    assert setup_spec.main(["backend-menu", "vulkan"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [ln.split("\t") for ln in lines] == [list(r) for r in setup_spec.backend_menu("vulkan")]
