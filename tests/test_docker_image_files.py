# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Docker image files: docker/entrypoint.sh behaviour, docker/Dockerfile
properties the docs promise, and the permissions and pins of
.github/workflows/docker.yml."""

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker.yml"

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="runs a POSIX shell script")


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture
def container(tmp_path):
    """A stand-in container: `python` is this interpreter with localm importable,
    `localm` records its arguments, LOCALM_HOME is empty."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "localm-calls.txt"
    python = bindir / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    localm = bindir / "localm"
    localm.write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\n', encoding="utf-8")
    for script in (python, localm):
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    home = tmp_path / "data"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALM_")}
    env.update(PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
               PYTHONPATH=str(REPO_ROOT), LOCALM_HOME=str(home))

    def run(*args, **extra_env):
        merged = {**env, **extra_env}
        proc = subprocess.run(["sh", str(ENTRYPOINT), *args], env=merged,
                              capture_output=True, text=True, timeout=120)
        invoked = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
        return proc, invoked

    run.env = env
    run.bindir = bindir
    return run


@posix_only
class TestEntrypoint:
    def test_refuses_to_serve_without_a_key(self, container):
        proc, invoked = container()
        assert proc.returncode == 2
        assert "API key" in proc.stderr
        assert invoked == []

    def test_refuses_a_key_shorter_than_the_network_floor(self, container):
        proc, invoked = container(LOCALM_API_KEY="abc")
        assert proc.returncode == 2
        assert invoked == []

    def test_serves_on_every_interface_with_an_env_key(self, container):
        proc, invoked = container(LOCALM_API_KEY="a-long-enough-key")
        assert proc.returncode == 0, proc.stderr
        assert invoked == ["serve -H 0.0.0.0 -p 8642"]

    def test_serves_with_a_key_created_in_the_data_volume(self, container):
        made = subprocess.run(
            [sys.executable, "-c",
             "from localm import auth; auth.set_api_key('volume-key-12345')"],
            env=container.env, capture_output=True, text=True, timeout=120)
        assert made.returncode == 0, made.stderr
        proc, invoked = container()
        assert proc.returncode == 0, proc.stderr
        assert invoked == ["serve -H 0.0.0.0 -p 8642"]

    def test_user_arguments_follow_the_defaults(self, container):
        proc, invoked = container("serve", "-c", "8192", LOCALM_API_KEY="a-long-enough-key")
        assert proc.returncode == 0, proc.stderr
        assert invoked == ["serve -H 0.0.0.0 -p 8642 -c 8192"]

    def test_a_leading_option_is_a_serve_option(self, container):
        proc, invoked = container("--no-tls", LOCALM_API_KEY="a-long-enough-key")
        assert proc.returncode == 0, proc.stderr
        assert invoked == ["serve -H 0.0.0.0 -p 8642 --no-tls"]

    def test_port_comes_from_the_container_port_variable(self, container):
        proc, invoked = container(LOCALM_API_KEY="a-long-enough-key",
                                  LOCALM_CONTAINER_PORT="9000")
        assert proc.returncode == 0, proc.stderr
        assert invoked == ["serve -H 0.0.0.0 -p 9000"]

    def test_insecure_serves_without_a_key(self, container):
        proc, invoked = container("serve", "--insecure")
        assert proc.returncode == 0, proc.stderr
        assert invoked == ["serve -H 0.0.0.0 -p 8642 --insecure"]

    def test_other_commands_run_as_localm_without_needing_a_key(self, container):
        proc, invoked = container("key", "generate")
        assert proc.returncode == 0, proc.stderr
        assert invoked == ["key generate"]

    def test_a_shell_runs_as_itself(self, container):
        proc, invoked = container("sh", "-c", "echo from-the-shell")
        assert proc.stdout.strip() == "from-the-shell"
        assert invoked == []

    def test_a_failing_key_check_is_reported_not_treated_as_open_or_ok(self, container):
        broken = container.bindir.parent / "broken-bin"
        broken.mkdir()
        python = broken / "python"
        python.write_text('#!/bin/sh\necho "no localm here" >&2\nexit 1\n', encoding="utf-8")
        python.chmod(python.stat().st_mode | stat.S_IEXEC)
        proc, invoked = container(LOCALM_API_KEY="a-long-enough-key",
                                  PATH=f"{broken}{os.pathsep}{container.env['PATH']}")
        assert proc.returncode == 1
        assert "could not check the API key configuration" in proc.stderr
        assert invoked == []


class TestDockerfile:
    text = DOCKERFILE.read_text(encoding="utf-8")

    def test_base_image_is_pinned_by_digest(self):
        assert re.search(r"^ARG UBUNTU_IMAGE=ubuntu:[\d.]+@sha256:[0-9a-f]{64}$",
                         self.text, re.M)

    def test_downloaded_uv_is_checksummed(self):
        assert re.search(r"^ARG UV_SHA256=[0-9a-f]{64}$", self.text, re.M)
        assert "sha256sum -c" in self.text

    def test_only_cpu_and_vulkan_backends_build(self):
        assert "cpu|vulkan" in self.text
        assert 'setup-llama --backend "${BACKEND}"' in self.text

    def test_server_does_not_run_as_root(self):
        lines = [ln.strip() for ln in self.text.splitlines()]
        assert "USER localm" in lines
        assert not any(ln.startswith("USER ") and ln != "USER localm" for ln in lines)

    def test_data_is_a_volume_and_home(self):
        assert 'VOLUME ["/data"]' in self.text
        assert "LOCALM_HOME=/data" in self.text

    def test_entrypoint_and_healthcheck_are_wired(self):
        assert "localm-docker-entrypoint" in self.text
        assert "HEALTHCHECK" in self.text
        assert (REPO_ROOT / "docker" / "healthcheck.sh").is_file()

    def test_dockerignore_keeps_the_runtime_lib_placeholder_and_drops_binaries(self):
        ignore = (REPO_ROOT / "docker" / "Dockerfile.dockerignore").read_text(encoding="utf-8")
        assert "runtime/localm_llama_runtime/lib/*" in ignore
        assert "!runtime/localm_llama_runtime/lib/.gitkeep" in ignore
        assert (REPO_ROOT / "runtime" / "localm_llama_runtime" / "lib" / ".gitkeep").is_file()


class TestWorkflow:
    def test_every_action_is_pinned_to_a_commit(self):
        uses = re.findall(r"^\s*(?:-\s*)?uses:\s*(\S+)", WORKFLOW.read_text(encoding="utf-8"), re.M)
        assert uses
        for ref in uses:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), ref

    def test_no_token_by_default_and_packages_write_only_where_pushing(self):
        wf = _workflow()
        assert wf["permissions"] == {}
        writers = {name: job["permissions"] for name, job in wf["jobs"].items()
                   if any(v == "write" for v in job["permissions"].values())}
        assert writers == {"publish": {"contents": "read", "packages": "write"}}

    def test_pull_requests_only_run_for_docker_paths_and_never_publish(self):
        wf = _workflow()
        triggers = wf[True] if True in wf else wf["on"]
        assert triggers["pull_request"]["paths"] == ["docker/**", ".github/workflows/docker.yml"]
        assert triggers["release"]["types"] == ["published"]
        assert triggers["workflow_dispatch"]["inputs"]["dry_run"]["default"] is True
        assert wf["jobs"]["publish"]["if"] == "github.event_name != 'pull_request'"

    def test_login_and_push_need_an_explicit_push_decision(self):
        steps = _workflow()["jobs"]["publish"]["steps"]
        login = next(s for s in steps if s.get("name") == "Log in to GHCR")
        assert login["if"] == "needs.prepare.outputs.push == 'true'"
        push = next(s for s in steps if s.get("id") == "push")
        assert 'if [ "$PUSH" = "true" ]; then' in push["run"]

    def test_the_smoke_test_the_workflow_runs_exists(self):
        assert "docker/smoke-test.sh" in WORKFLOW.read_text(encoding="utf-8")
        assert (REPO_ROOT / "docker" / "smoke-test.sh").is_file()
