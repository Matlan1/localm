# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_llama_pin.py: the mechanical half of advancing the llama.cpp pin.

Covers the three properties that make a bump safe to script:

  * the values that must move together (pin, offline digests, MTP source tag and
    allowlist) move in one rewrite, and nothing else in either file changes;
  * a write needs evidence: a receipt from confirm_llama_runtime.py naming the
    target tag with PASS on every required backend, and a _PIN_CONFIRMATION
    table that agrees with it;
  * every edited region is located exactly once, on the real tree as well as on
    fixtures, so a reshaped file refuses rather than editing the wrong lines.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_BUMP = _ROOT / "scripts" / "bump_llama_pin.py"
_CONFIRM = _ROOT / "scripts" / "confirm_llama_runtime.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    return _load(_BUMP, "bump_llama_pin")


AA, BB, CC, DD = "aa" * 32, "bb" * 32, "cc" * 32, "dd" * 32

SETUP_FIXTURE = f'''# header
_ROCM_TAG = "b1307"

_PINNED_TAG = "b100"

_PIN_CONFIRMATION = {{
    "cpu": "load + generate, measured (somewhere)",
    "vulkan": "load + generate, measured (somewhere "
              "else)",
    "cuda": "ABI only (shared llama library); generation NOT measured - no NVIDIA hardware",
    "amd-rocm": "out of scope for _PINNED_TAG - NOT measured",
}}

_PINNED_FALLBACK_SHA256 = {{
    # tag b100 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names and upstream re-uploads the same file each release.
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "{AA}",
    "llama-b100-bin-win-cpu-x64.zip": "{BB}",
    "llama-b100-bin-win-vulkan-x64.zip": "{CC}",
    # tag b1307 ROCm assets (something)
    "llama-b1307-windows-rocm-gfx103X-x64.zip": "{DD}",
}}

_ASSET_MATCH = {{}}
'''

API_FIXTURE = '''# header
MTP_ARCH_SOURCE_TAG = "b100"

MTP_GRAPH_ARCHITECTURES = frozenset({
    "deepseek2",
    "qwen35",
})

OTHER = 1
'''

NEW_DIGESTS = {
    "llama-b200-bin-win-vulkan-x64.zip": "11" * 32,
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "aa" * 32,
    "llama-b200-bin-win-cpu-x64.zip": "22" * 32,
}


def _receipt(tmp_path: Path, tag: str, verdicts: dict) -> Path:
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps({
        "tag": tag, "exit_code": 0,
        "backends": {b: {"verdict": v, "why": f"{b} says {v}"} for b, v in verdicts.items()},
        "lib_sha256": {}}), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
#  The rewrite                                                                 #
# --------------------------------------------------------------------------- #

def test_rewrite_moves_pin_digests_source_tag_and_allowlist_together(bump):
    setup, api = bump.rewrite(SETUP_FIXTURE, API_FIXTURE, "b200", NEW_DIGESTS,
                              {"qwen35", "glm4moe", "deepseek2"})

    assert '_PINNED_TAG = "b200"' in setup
    assert '_PINNED_TAG = "b100"' not in setup
    assert "# tag b200 upstream assets (_PINNED_TAG). The three cudart bundles carry no" in setup
    assert "# tag in their names and upstream re-uploads the same file each release." in setup
    block = setup.split("_PINNED_FALLBACK_SHA256 = {\n", 1)[1].split("# tag b1307", 1)[0]
    names = [line.split('"')[1] for line in block.splitlines() if line.strip().startswith('"')]
    assert names == sorted(NEW_DIGESTS), "entries are the new release's assets, sorted by name"
    assert f'"llama-b200-bin-win-cpu-x64.zip": "{"22" * 32}",' in setup
    assert "llama-b100-" not in setup, "no entry of the old release survives"
    assert '"llama-b1307-windows-rocm-gfx103X-x64.zip": "' + "dd" * 32 + '",' in setup, (
        "the ROCm block belongs to _ROCM_TAG and is untouched")
    assert '_ROCM_TAG = "b1307"' in setup
    assert "_PIN_CONFIRMATION = {" in setup and "measured (somewhere)" in setup

    assert 'MTP_ARCH_SOURCE_TAG = "b200"' in api
    assert api.split("frozenset({\n", 1)[1].split("})", 1)[0] == (
        '    "deepseek2",\n    "glm4moe",\n    "qwen35",\n')
    assert api.endswith("OTHER = 1\n")


def test_rewrite_is_idempotent_and_a_no_op_at_the_current_values(bump):
    once = bump.rewrite(SETUP_FIXTURE, API_FIXTURE, "b200", NEW_DIGESTS, {"qwen35"})
    twice = bump.rewrite(once[0], once[1], "b200", NEW_DIGESTS, {"qwen35"})
    assert twice == once

    same = bump.rewrite(SETUP_FIXTURE, API_FIXTURE, "b100", {
        "cudart-llama-bin-win-cuda-12.4-x64.zip": "aa" * 32,
        "llama-b100-bin-win-cpu-x64.zip": "bb" * 32,
        "llama-b100-bin-win-vulkan-x64.zip": "cc" * 32,
    }, {"deepseek2", "qwen35"})
    assert same == (SETUP_FIXTURE, API_FIXTURE)


def test_each_region_must_be_found_exactly_once(bump):
    with pytest.raises(bump.Refused, match="_PINNED_TAG"):
        bump.set_pin(SETUP_FIXTURE + '\n_PINNED_TAG = "b101"\n', "b200")
    with pytest.raises(bump.Refused, match="upstream block"):
        bump.set_sha_block(SETUP_FIXTURE.replace("(_PINNED_TAG)", "(renamed)"), "b200", NEW_DIGESTS)
    with pytest.raises(bump.Refused, match="MTP_ARCH_SOURCE_TAG"):
        bump.set_mtp_tag(API_FIXTURE.replace("MTP_ARCH_SOURCE_TAG", "MTP_TAG"), "b200")
    with pytest.raises(bump.Refused, match="MTP_GRAPH_ARCHITECTURES"):
        bump.set_mtp_set(API_FIXTURE.replace("frozenset({", "frozenset(["), {"x"})


def test_the_regions_match_exactly_once_on_the_real_tree(bump):
    """Bound to the shipped files: the shapes this script edits exist there,
    once each, so a dry run against the tree is a real dry run."""
    setup = bump.SETUP_PATH.read_text(encoding="utf-8")
    api = bump.API_PATH.read_text(encoding="utf-8")
    assert len(bump._PIN_RE.findall(setup)) == 1
    assert len(list(bump._SHA_BLOCK_RE.finditer(setup))) == 1
    assert len(bump._MTP_TAG_RE.findall(api)) == 1
    assert len(list(bump._MTP_SET_RE.finditer(api))) == 1

    from localm import setup_llama as sl
    from localm.inference.backends.llamacpp import _api
    m = bump._SHA_BLOCK_RE.search(setup)
    assert m.group("tag") == sl._PINNED_TAG, "the upstream block is labelled with the pin"
    entries = m.group("entries").count("\n")
    upstream = [k for k in sl._PINNED_FALLBACK_SHA256 if sl._ROCM_TAG not in k]
    assert entries == len(upstream), "the block spans every non-ROCm entry"
    assert bump.measured_backends(setup) == {
        b for b, note in sl._PIN_CONFIRMATION.items() if "load + generate, measured" in note}
    assert _api.MTP_ARCH_SOURCE_TAG == sl._PINNED_TAG


# --------------------------------------------------------------------------- #
#  Evidence                                                                    #
# --------------------------------------------------------------------------- #

def test_receipt_must_name_the_tag_and_pass_every_required_backend(bump, tmp_path):
    with pytest.raises(bump.Refused, match="not b200"):
        bump.load_receipt(_receipt(tmp_path, "b199", {"cpu": "PASS"}), "b200", ("cpu",))

    p = _receipt(tmp_path, "b200", {"cpu": "PASS", "vulkan": "INCONCLUSIVE"})
    with pytest.raises(bump.Refused) as e:
        bump.load_receipt(p, "b200", ("cpu", "vulkan"))
    assert "vulkan" in str(e.value) and "INCONCLUSIVE" in str(e.value)
    assert "vulkan says INCONCLUSIVE" in str(e.value), "the receipt's own reason is quoted"
    assert bump.load_receipt(p, "b200", ("cpu",)) == {"cpu"}

    with pytest.raises(bump.Refused, match="not run"):
        bump.load_receipt(_receipt(tmp_path, "b200", {"cpu": "PASS"}), "b200", ("cpu", "metal"))
    with pytest.raises(bump.Refused, match="could not read"):
        bump.load_receipt(tmp_path / "absent.json", "b200", ("cpu",))


def test_the_confirm_scripts_receipt_is_what_the_bump_script_reads(bump, tmp_path):
    """Both ends of the contract in one place: a receipt written by
    confirm_llama_runtime._write_receipt satisfies load_receipt for the backends
    it reports PASS and refuses the ones it does not."""
    confirm = _load(_CONFIRM, "confirm_llama_runtime")
    summary = {"tag": "b200",
               "results": {"cpu": {"verdict": "PASS", "why": "loaded and generated"},
                           "vulkan": {"verdict": "FAIL", "why": "no tokens"}},
               "lib_sha256": {"cpu": "0" * 64}}
    p = tmp_path / "r.json"
    confirm._write_receipt(p, summary, ["cpu", "vulkan", "metal"], 1)
    written = json.loads(p.read_text(encoding="utf-8"))
    assert written["exit_code"] == 1
    assert written["backends"]["metal"]["verdict"] == "INCONCLUSIVE", "asked for, not run"
    assert bump.load_receipt(p, "b200", ("cpu",)) == {"cpu"}
    with pytest.raises(bump.Refused, match="no tokens"):
        bump.load_receipt(p, "b200", ("cpu", "vulkan"))


def test_measured_backends_reads_the_confirmation_table(bump):
    assert bump.measured_backends(SETUP_FIXTURE) == {"cpu", "vulkan"}
    assert bump.measured_backends(SETUP_FIXTURE.replace(
        '"vulkan": "load + generate, measured (somewhere "',
        '"vulkan": "ABI only; generation NOT measured (somewhere "')) == {"cpu"}
    with pytest.raises(bump.Refused, match="_PIN_CONFIRMATION"):
        bump.measured_backends("nothing here")


def test_fetch_release_assets_refuses_short_or_undigested_listings(bump):
    good = [{"name": f"a{i}", "digest": "sha256:" + f"{i:064x}"} for i in range(bump.MIN_ASSETS)]

    def opener_for(payload):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")
        return lambda req, timeout=None: _Resp()

    assert bump.fetch_release_assets("b1", opener_for({"assets": good})) == {
        a["name"]: a["digest"][7:] for a in good}
    with pytest.raises(bump.Refused, match="no sha256 digest"):
        bump.fetch_release_assets("b1", opener_for({"assets": good[:-1] + [{"name": "z"}]}))
    with pytest.raises(bump.Refused, match="at least"):
        bump.fetch_release_assets("b1", opener_for({"assets": good[:-1]}))
    with pytest.raises(bump.Refused, match="no asset list"):
        bump.fetch_release_assets("b1", opener_for({"message": "Not Found"}))

    def dead(req, timeout=None):
        raise OSError("no network")
    with pytest.raises(bump.Refused, match="could not read"):
        bump.fetch_release_assets("b1", dead)


# --------------------------------------------------------------------------- #
#  main(): dry run versus --write                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def tree(bump, tmp_path, monkeypatch):
    setup = tmp_path / "setup_llama.py"
    api = tmp_path / "_api.py"
    setup.write_text(SETUP_FIXTURE, encoding="utf-8")
    api.write_text(API_FIXTURE, encoding="utf-8")
    monkeypatch.setattr(bump, "SETUP_PATH", setup)
    monkeypatch.setattr(bump, "API_PATH", api)
    monkeypatch.setattr(bump, "REPO", tmp_path)
    monkeypatch.setattr(bump, "fetch_release_assets", lambda tag, opener=None: NEW_DIGESTS)
    monkeypatch.setattr(bump, "derive_mtp_architectures", lambda tag: {"qwen35", "glm4moe"})
    return setup, api


def test_dry_run_writes_nothing_and_write_needs_a_receipt(bump, tree, capsys):
    setup, api = tree
    assert bump.main(["--tag", "b200"]) == 0
    out = capsys.readouterr().out
    assert "no receipt given" in out
    assert "+_PINNED_TAG = \"b200\"" in out and "+    \"glm4moe\"," in out
    assert "dry run: nothing written" in out
    assert "REMAINING STEPS" in out
    assert setup.read_text(encoding="utf-8") == SETUP_FIXTURE
    assert api.read_text(encoding="utf-8") == API_FIXTURE

    assert bump.main(["--tag", "b200", "--write"]) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert setup.read_text(encoding="utf-8") == SETUP_FIXTURE, "a refusal edits nothing"


def test_write_applies_once_and_then_reports_nothing_to_change(bump, tree, tmp_path, capsys):
    setup, api = tree
    receipt = _receipt(tmp_path, "b200", {"cpu": "PASS", "vulkan": "PASS"})
    assert bump.main(["--tag", "b200", "--receipt", str(receipt), "--write"]) == 0
    out = capsys.readouterr().out
    assert "wrote setup_llama.py" in out and "wrote _api.py" in out
    assert '_PINNED_TAG = "b200"' in setup.read_text(encoding="utf-8")
    assert 'MTP_ARCH_SOURCE_TAG = "b200"' in api.read_text(encoding="utf-8")

    assert bump.main(["--tag", "b200", "--receipt", str(receipt), "--write"]) == 0
    assert "nothing to change" in capsys.readouterr().out


def test_write_refuses_when_the_table_and_the_receipt_disagree(bump, tree, tmp_path, capsys):
    """The table claims cpu and vulkan were measured; the receipt confirms cpu
    only. A dry run says so and continues; a write refuses, because the
    constant would otherwise ship claiming a measurement nobody made."""
    setup, _ = tree
    receipt = _receipt(tmp_path, "b200", {"cpu": "PASS"})
    assert bump.main(["--tag", "b200", "--receipt", str(receipt), "--require", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "WARNING: _PIN_CONFIRMATION claims a measurement for cpu, vulkan" in out
    assert "the receipt confirms cpu" in out

    assert bump.main(["--tag", "b200", "--receipt", str(receipt), "--require", "cpu",
                      "--write"]) == 1
    assert "REFUSED: _PIN_CONFIRMATION claims" in capsys.readouterr().out
    assert setup.read_text(encoding="utf-8") == SETUP_FIXTURE


def test_a_receipt_for_the_wrong_backend_set_refuses_by_default(bump, tree, tmp_path, capsys):
    """Default --require is cpu and vulkan: a cpu-only receipt is not enough
    to write, whatever the table says."""
    receipt = _receipt(tmp_path, "b200", {"cpu": "PASS"})
    assert bump.main(["--tag", "b200", "--receipt", str(receipt), "--write"]) == 1
    assert "does not confirm b200 on vulkan" in capsys.readouterr().out


def test_a_malformed_tag_is_refused_before_anything_is_read(bump, tree, capsys):
    assert bump.main(["--tag", "latest"]) == 1
    assert "not an upstream build tag" in capsys.readouterr().out
