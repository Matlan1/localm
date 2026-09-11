# SPDX-License-Identifier: AGPL-3.0-or-later
"""auth.create_key()/model_source_credentials.set_credentials() must serialize
ACROSS OS processes, exactly like config.update_config() (see
test_config_cross_process_lock.py, which this mirrors).

Before the fix, auth._load_keystore() failed OPEN on any read error and
create_key()/revoke_key() read-modify-wrote on the fixed temp name
`auth.json.tmp` with no cross-process lock: two real localm processes each
minting keys concurrently could read the same list, and whichever finished
last silently overwrote the other's keys - or crashed on the shared temp
file. model_source_credentials.set_credentials() had the identical gap.

These tests spawn two REAL, independent Python processes - not threads - each
writing N keys/credential updates against ONE shared throwaway LOCALM_HOME,
and assert that the final store holds every write from both processes: zero
lost, by NAME/COUNT, not by return-value alone."""

import json
import os
import subprocess
import sys

import pytest

import localm.config as cfg


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir()
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


def _child_env(home_dir):
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(home_dir)
    env.pop("LOCALM_API_KEY", None)
    env.pop("LOCALM_REQUIRE_AUTH", None)
    # Make the child import the same localm this test runs (the worktree),
    # not the venv's editable install - see worktree-preflight.md.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return env


# Each worker is a genuinely separate python process. It mints/updates in a
# tight loop with no inter-call delay: create_key/set_credentials each hold the
# cross-process lock for their own whole read-modify-write, so back-to-back
# calls from two processes are exactly the contention pattern that raced
# before the fix (a fixed temp filename, no cross-process serialization).
_KEYS_WORKER = (
    "import json, sys\n"
    "import localm.auth as auth\n"
    "prefix, count = sys.argv[1], int(sys.argv[2])\n"
    "made = [auth.create_key(f'{prefix}-{i}', [])['id'] for i in range(count)]\n"
    "print(json.dumps(made))\n"
)

_CREDS_WORKER = (
    "import json, sys\n"
    "from localm.model_source_credentials import set_credentials\n"
    "key, prefix, count = sys.argv[1], sys.argv[2], int(sys.argv[3])\n"
    "for i in range(count):\n"
    "    set_credentials({key: f'{prefix}-{i}'})\n"
    "print(json.dumps(f'{prefix}-{count - 1}'))\n"
)


def _spawn_two(env, worker_src, args_a, args_b):
    procs = [
        subprocess.Popen([sys.executable, "-c", worker_src, *args_a],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
        subprocess.Popen([sys.executable, "-c", worker_src, *args_b],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
    ]
    outs = []
    for i, p in enumerate(procs):
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, (
            f"worker {i} crashed (rc={p.returncode}): "
            f"{err.decode('utf-8', 'replace')[-4000:]}")
        outs.append(out.decode("utf-8"))
    return outs


def test_keystore_two_process_create_key_no_lost_writes(home):
    """Two SEPARATE OS processes each running auth.create_key() in a loop
    against the same LOCALM_HOME must not lose either process's keys: the
    final keystore count must equal the sum of both, by id."""
    import localm.auth as auth
    count = 15
    out_a, out_b = _spawn_two(_child_env(home), _KEYS_WORKER,
                              ["A", str(count)], ["B", str(count)])
    ids_a = set(json.loads(out_a))
    ids_b = set(json.loads(out_b))
    assert len(ids_a) == count and len(ids_b) == count, (
        "each worker must itself report having minted every key it attempted")

    final_ids = {r["id"] for r in auth.list_keys()}
    lost = (ids_a | ids_b) - final_ids
    assert not lost, (
        f"{len(lost)} of {2 * count} concurrently created keys were silently "
        f"lost from the on-disk keystore: {sorted(lost)}")
    assert len(final_ids) == 2 * count, (
        f"expected exactly {2 * count} keys on disk (no loss, no duplication), "
        f"found {len(final_ids)}")


def test_credentials_two_process_set_no_lost_writes(home):
    """Two SEPARATE OS processes each set a DIFFERENT credential key
    (hf_token / civitai_api_key) in a loop against the same LOCALM_HOME.
    Both keys live in ONE read-modify-write file: a lost update (process B's
    read-modify-write reading a stale copy before process A's write landed,
    then persisting its own key plus that stale/missing hf_token) would
    silently drop process A's key entirely - the failure mode this store
    shared with the keystore before the fix."""
    from localm.model_source_credentials import (credentials_path,
                                                  get_civitai_api_key, get_hf_token)
    count = 15
    last_hf, last_civ = _spawn_two(
        _child_env(home), _CREDS_WORKER,
        ["hf_token", "A", str(count)], ["civitai_api_key", "B", str(count)])
    last_hf, last_civ = json.loads(last_hf), json.loads(last_civ)

    assert get_hf_token() == last_hf, (
        f"hf_token was silently dropped by a concurrent civitai_api_key "
        f"write from a separate process; expected {last_hf!r}, got "
        f"{get_hf_token()!r}")
    assert get_civitai_api_key() == last_civ, (
        f"civitai_api_key was silently dropped by a concurrent hf_token "
        f"write from a separate process; expected {last_civ!r}, got "
        f"{get_civitai_api_key()!r}")

    on_disk = json.loads(credentials_path().read_text(encoding="utf-8"))
    assert on_disk == {"hf_token": last_hf, "civitai_api_key": last_civ}
