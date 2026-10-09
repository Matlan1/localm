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
HEALTHCHECK = REPO_ROOT / "docker" / "healthcheck.sh"
RESOLVE = REPO_ROOT / "docker" / "resolve-release.sh"
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


@posix_only
class TestHealthcheck:
    @staticmethod
    def _run(tmp_path, https, http):
        """Run the healthcheck against a stand-in curl that answers HTTPS URLs with
        *https* and HTTP URLs with *http* ("000" is a refused connection)."""
        bindir = tmp_path / "bin"
        bindir.mkdir()
        curl = bindir / "curl"
        script = (
            "#!/bin/sh\n"
            'for last in "$@"; do :; done\n'
            'case "$last" in\n'
            f"  https://*) code={https} ;;\n"
            f"  *) code={http} ;;\n"
            "esac\n"
            'printf "%s" "$code"\n'
            '[ "$code" != 000 ]\n'
        )
        curl.write_text(script, encoding="utf-8")
        curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
        env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}
        return subprocess.run(["sh", str(HEALTHCHECK)], env=env, capture_output=True,
                              text=True, timeout=30).returncode

    def test_healthy_when_https_answers_200(self, tmp_path):
        assert self._run(tmp_path, https=200, http=308) == 0

    def test_healthy_on_plain_http_when_tls_is_off(self, tmp_path):
        assert self._run(tmp_path, https="000", http=200) == 0

    def test_a_redirect_is_not_healthy(self, tmp_path):
        assert self._run(tmp_path, https=404, http=308) == 1

    def test_an_error_status_is_not_healthy(self, tmp_path):
        assert self._run(tmp_path, https=401, http=401) == 1

    def test_nothing_listening_is_not_healthy(self, tmp_path):
        assert self._run(tmp_path, https="000", http="000") == 1


@posix_only
class TestResolveRelease:
    @pytest.fixture
    def resolve(self, tmp_path):
        """Run docker/resolve-release.sh in a directory whose VERSION is 1.2.3 with a
        stand-in `gh` whose latest release is $FAKE_LATEST. Returns
        (exit code, stdout, {output name: value})."""
        (tmp_path / "VERSION").write_text("1.2.3\n", encoding="utf-8")
        bindir = tmp_path / "bin"
        bindir.mkdir()
        gh = bindir / "gh"
        gh.write_text('#!/bin/sh\n[ -z "$FAKE_GH_FAIL" ] || exit 1\necho "$FAKE_LATEST"\n',
                      encoding="utf-8")
        gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
        out = tmp_path / "github-output"

        def run(event, *, ref="refs/heads/master", tag="", prerelease="", dry_run="",
                latest="v1.2.3", gh_fails=False, version=None):
            if version is not None:
                (tmp_path / "VERSION").write_text(version, encoding="utf-8")
            out.write_text("", encoding="utf-8")
            env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
                   "EVENT_NAME": event, "REF": ref, "RELEASE_TAG": tag,
                   "PRERELEASE": prerelease, "DRY_RUN": dry_run,
                   "GITHUB_REPOSITORY": "owner/repo", "GITHUB_OUTPUT": str(out),
                   "FAKE_LATEST": latest, "FAKE_GH_FAIL": "1" if gh_fails else ""}
            proc = subprocess.run(["sh", str(RESOLVE)], cwd=tmp_path, env=env,
                                  capture_output=True, text=True, timeout=30)
            values = dict(line.split("=", 1) for line in
                          out.read_text(encoding="utf-8").splitlines())
            return proc.returncode, proc.stdout, values

        return run

    def test_pull_request_builds_and_never_pushes(self, resolve):
        code, _, out = resolve("pull_request")
        assert code == 0
        assert out == {"version": "1.2.3", "push": "false", "floating": "false"}

    def test_latest_release_pushes_and_moves_the_floating_tags(self, resolve):
        code, _, out = resolve("release", tag="v1.2.3", prerelease="false")
        assert code == 0
        assert out == {"version": "1.2.3", "push": "true", "floating": "true"}

    def test_prerelease_pushes_version_tags_only(self, resolve):
        code, _, out = resolve("release", tag="v1.2.3", prerelease="true")
        assert code == 0
        assert out == {"version": "1.2.3", "push": "true", "floating": "false"}

    def test_rerun_of_an_older_release_does_not_move_latest(self, resolve):
        code, _, out = resolve("release", tag="v1.2.3", prerelease="false", latest="v2.0.0")
        assert code == 0
        assert out == {"version": "1.2.3", "push": "true", "floating": "false"}

    def test_release_tag_must_match_version(self, resolve):
        code, stdout, out = resolve("release", tag="v1.2.4", prerelease="false")
        assert code == 1
        assert "does not match VERSION" in stdout
        assert out == {}

    def test_unreadable_latest_release_fails_instead_of_dropping_latest(self, resolve):
        code, stdout, out = resolve("release", tag="v1.2.3", prerelease="false", gh_fails=True)
        assert code == 1
        assert "could not read the latest release" in stdout
        assert out == {}

    def test_dispatch_defaults_to_a_dry_run(self, resolve):
        code, _, out = resolve("workflow_dispatch", dry_run="true")
        assert code == 0
        assert out["push"] == "false"
        assert out["floating"] == "false"

    def test_dispatch_cannot_publish_from_a_branch(self, resolve):
        code, stdout, out = resolve("workflow_dispatch", dry_run="false")
        assert code == 1
        assert "requires the release tag ref refs/tags/v1.2.3" in stdout
        assert out == {}

    def test_dispatch_cannot_publish_another_versions_tag(self, resolve):
        code, _, out = resolve("workflow_dispatch", dry_run="false", ref="refs/tags/v1.2.2")
        assert code == 1
        assert out == {}

    def test_dispatch_on_the_release_tag_republishes_it(self, resolve):
        code, _, out = resolve("workflow_dispatch", dry_run="false", ref="refs/tags/v1.2.3")
        assert code == 0
        assert out == {"version": "1.2.3", "push": "true", "floating": "true"}

    def test_a_version_that_is_not_a_tag_is_refused(self, resolve):
        code, stdout, _ = resolve("pull_request", version="1.2.3+local\n")
        assert code == 1
        assert "not usable as an image tag" in stdout


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
