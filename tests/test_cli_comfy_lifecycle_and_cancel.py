# SPDX-License-Identifier: AGPL-3.0-or-later
"""The CLI drives ComfyUI's process and cancels a server operation.

Four capabilities: whether ComfyUI is alive and whether localm launched it,
starting it without generating, stopping/restarting it, and cancelling one
in-flight operation.

Everything here is about the honesty of the seams:

  - `launched_by_localm` is only knowable inside the process holding the
    subprocess handle (`comfy_client._spawned_procs` is a process-local module
    global). With no server to ask, the answer is UNKNOWN. Printing "no" would
    be inventing a negative out of not having asked.
  - The stop/restart routes answer HTTP 200 with `{"ok": false}` for a refusal
    they handled cleanly, so a 2xx is not by itself success.
  - A 404 means "this server has no such route" on `/v1/comfy/status` and "no
    such job" on `/api/jobs/<id>/cancel`. One status code, two unrelated
    answers. And a POST to a path the server does not serve is 405, not 404.
  - Cancelling an operation that already finished must not report "cancelling",
    which is what a blind POST would produce (the route returns the same body
    either way).
"""

from __future__ import annotations

import pytest
import requests
from click.testing import CliRunner

import localm.cli.comfy as comfy_cli
import localm.cli.media as media_cli
import localm.cli.models as models_cli
from localm.cli._core import report_server_failure, server_call


# --------------------------------------------------------------------------
#  Test doubles
# --------------------------------------------------------------------------

class _Resp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.text = text if text is not None else "{}"

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class _Server:
    """A fake localm server: records every request and answers from a route map.

    The recorded calls are what several assertions below read: they are about a
    request NOT being made (a blind cancel, a launch that would abort a live
    render), which the printed output cannot express.
    """

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, **kw):
        path = url.split("127.0.0.1:9999", 1)[-1]
        self.calls.append((method, path))
        answer = self.routes.get((method, path), self.routes.get(path))
        if answer is None:
            # A POST to a path the server does not serve comes back 405 Method
            # Not Allowed; only GET gives 404 there.
            return (_Resp(404, {"detail": "Not Found"}) if method == "GET"
                    else _Resp(405, {"detail": "Method Not Allowed"}))
        if isinstance(answer, Exception):
            raise answer
        return answer

    def paths(self):
        return [p for _, p in self.calls]


def _install(monkeypatch, module, server, *, discovered=True):
    """Point *module*'s server helpers at *server*."""
    monkeypatch.setattr(requests, "request", server)
    monkeypatch.setattr(
        module, "running_server",
        (lambda **kw: ("http://127.0.0.1:9999", {})) if discovered
        else (lambda **kw: None))
    return server


def _run(cmd, args=()):
    return CliRunner().invoke(cmd, list(args))


# --------------------------------------------------------------------------
#  server_call: one status code, two meanings
# --------------------------------------------------------------------------

def test_a_404_defaults_to_the_route_being_absent(monkeypatch):
    monkeypatch.setattr(requests, "request", lambda *a, **k: _Resp(404, {}))
    assert server_call("http://x", {}, "GET", "/v1/comfy/status")[0] == "unsupported"


def test_a_404_can_mean_the_object_is_absent_instead(monkeypatch):
    """THE ONE THAT MATTERS. Without this parameter a mistyped job id reports
    "this server predates the feature", which sends the reader to entirely the
    wrong conclusion."""
    monkeypatch.setattr(requests, "request", lambda *a, **k: _Resp(404, {}))
    state, _ = server_call("http://x", {}, "POST", "/api/jobs/abc/cancel",
                           not_found="missing")
    assert state == "missing"


def test_the_two_404_meanings_do_not_print_the_same_sentence(capsys):
    report_server_failure("unsupported", 404, "cancel that operation")
    absent_route = capsys.readouterr().out
    report_server_failure("missing", 404, "cancel that operation")
    absent_object = capsys.readouterr().out
    assert absent_route != absent_object
    assert "predates" in absent_route
    assert "no such item" in absent_object


def test_an_unserved_post_path_reports_405_as_no_such_route(monkeypatch):
    """A POST to a path the server does not serve answers 405, not 404, so
    treating only 404 as "no such route" makes `comfy start` fail on a server
    with no media plugin instead of falling back."""
    monkeypatch.setattr(requests, "request", lambda *a, **k: _Resp(405, {}))
    state, _ = server_call("http://x", {}, "POST", "/api/imagine/comfy-launch")
    assert state == "unsupported"


def test_405_is_never_read_as_a_missing_object(monkeypatch):
    """not_found only ever reinterprets a 404. A 405 is a statement about the
    path and method and can never mean "the job you named is gone"."""
    monkeypatch.setattr(requests, "request", lambda *a, **k: _Resp(405, {}))
    state, _ = server_call("http://x", {}, "POST", "/api/jobs/abc/cancel",
                           not_found="missing")
    assert state == "unsupported"


@pytest.mark.parametrize("code,expected", [(401, "unauthorized"),
                                           (403, "unauthorized"),
                                           (500, "http")])
def test_the_other_failure_states_stay_apart(monkeypatch, code, expected):
    monkeypatch.setattr(requests, "request", lambda *a, **k: _Resp(code, {}))
    assert server_call("http://x", {}, "GET", "/p")[0] == expected


def test_a_connection_failure_is_not_a_negative_answer(monkeypatch):
    def _boom(*a, **k):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(requests, "request", _boom)
    assert server_call("http://x", {}, "GET", "/p")[0] == "unreachable"


def test_a_200_that_is_not_json_is_not_an_empty_result(monkeypatch):
    monkeypatch.setattr(requests, "request",
                        lambda *a, **k: _Resp(200, None, text="<html>"))
    assert server_call("http://x", {}, "GET", "/p")[0] == "http"


# --------------------------------------------------------------------------
#  localm status: the id and the cancellable flag stop being discarded
# --------------------------------------------------------------------------

def _activity(monkeypatch, ops, now=1000.0):
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _Resp(200, {"now": now, "operations": ops}))


def test_a_running_operation_now_shows_the_id_you_would_cancel(monkeypatch, capsys):
    _activity(monkeypatch, [{"id": "7f3a2b91c4d5", "kind": "pull",
                             "label": "Model pull owner/repo", "status": "running",
                             "created_at": 940.0, "finished_at": None,
                             "cancellable": True}])
    models_cli._print_activity("http", 1234)
    out = capsys.readouterr().out
    assert "7f3a2b91c4d5" in out, "the id was in the payload and was dropped"
    assert "localm cancel" in out


def test_no_cancel_hint_when_nothing_is_cancellable(monkeypatch, capsys):
    """A hint offering to cancel a finished operation is a false affordance:
    `localm cancel` would correctly refuse it, so the hint promises something
    that cannot happen."""
    _activity(monkeypatch, [{"id": "abc123abc123", "kind": "pull",
                             "label": "Model pull", "status": "done",
                             "created_at": 940.0, "finished_at": 990.0,
                             "cancellable": False}])
    models_cli._print_activity("http", 1234)
    assert "localm cancel" not in capsys.readouterr().out


def test_an_operation_with_no_id_does_not_produce_a_cancel_hint(monkeypatch, capsys):
    """An older server can report an operation without an id. Offering to
    cancel something that cannot be named is the same false affordance."""
    _activity(monkeypatch, [{"kind": "pull", "label": "Model pull",
                             "status": "running", "created_at": 940.0,
                             "cancellable": True}])
    models_cli._print_activity("http", 1234)
    assert "localm cancel" not in capsys.readouterr().out


# --------------------------------------------------------------------------
#  Resolving an operation id
# --------------------------------------------------------------------------

def test_an_exact_id_wins_over_a_prefix_that_also_matches_another():
    """`abc` is both an exact id and a prefix of `abcdef`. Treating that as
    ambiguous would make a correct, complete id unusable."""
    ops = [{"id": "abc"}, {"id": "abcdef"}]
    op, err = models_cli._match_operation(ops, "abc")
    assert err is None and op["id"] == "abc"


def test_a_unique_prefix_resolves():
    op, err = models_cli._match_operation([{"id": "abcdef"}, {"id": "zz"}], "abc")
    assert err is None and op["id"] == "abcdef"


def test_an_ambiguous_prefix_is_an_error_not_a_guess():
    op, err = models_cli._match_operation([{"id": "abc1"}, {"id": "abc2"}], "abc")
    assert op is None and err == "ambiguous"


def test_an_unmatched_id_is_its_own_error():
    assert models_cli._match_operation([{"id": "abc"}], "zzz") == (None, "none")


# --------------------------------------------------------------------------
#  localm cancel
# --------------------------------------------------------------------------

_RUNNING_OP = {"id": "7f3a2b91c4d5", "kind": "pull", "label": "Model pull",
               "status": "running", "created_at": 1.0, "cancellable": True}
_DONE_OP = {"id": "aaaabbbbcccc", "kind": "pull", "label": "Old pull",
            "status": "done", "created_at": 1.0, "finished_at": 2.0,
            "cancellable": False}


def _cancel_server(ops, cancel=_Resp(200, {"status": "cancelling"})):
    return _Server({("GET", "/api/activity"): _Resp(200, {"now": 9.0,
                                                          "operations": ops}),
                    ("POST", f"/api/jobs/{_RUNNING_OP['id']}/cancel"): cancel})


def test_cancelling_a_running_operation_posts_to_its_full_id(monkeypatch):
    """A PREFIX is accepted, but the POST must carry the FULL id - the route
    matches on the whole id, so a prefix there would 404."""
    srv = _install(monkeypatch, models_cli, _cancel_server([_RUNNING_OP]))
    result = _run(models_cli.cancel_cmd, ["7f3a"])
    assert result.exit_code == 0, result.output
    assert f"/api/jobs/{_RUNNING_OP['id']}/cancel" in srv.paths()
    assert "Cancelling" in result.output


def test_it_says_cancelling_not_cancelled(monkeypatch):
    """The server sets a cooperative flag and terminates any subprocess; an
    in-process job stops at its next checkpoint. "Cancelled" would be a state
    this command never observed."""
    _install(monkeypatch, models_cli, _cancel_server([_RUNNING_OP]))
    out = _run(models_cli.cancel_cmd, ["7f3a"]).output.lower()
    assert "cancelling" in out
    assert "cancelled" not in out


def test_a_finished_operation_is_not_posted_to_and_is_not_claimed_cancelled(monkeypatch):
    """THE CENTRAL HONESTY TEST. POST /api/jobs/<id>/cancel answers
    {"status": "cancelling"} for a job that finished hours ago exactly as it
    does for a live one, so a command that posted blind would report a
    cancellation that never happened."""
    srv = _install(monkeypatch, models_cli, _cancel_server([_DONE_OP]))
    result = _run(models_cli.cancel_cmd, [_DONE_OP["id"]])
    assert result.exit_code == 0, result.output
    assert not [p for p in srv.paths() if "cancel" in p], \
        f"posted a cancel for a finished operation: {srv.paths()}"
    assert "nothing to cancel" in result.output.lower()
    assert "cancelling" not in result.output.lower()


def test_an_unknown_id_fails_and_posts_nothing(monkeypatch):
    srv = _install(monkeypatch, models_cli, _cancel_server([_RUNNING_OP]))
    result = _run(models_cli.cancel_cmd, ["nope"])
    assert result.exit_code == 1
    assert not [p for p in srv.paths() if "cancel" in p]
    assert "No operation matches" in result.output


def test_an_ambiguous_prefix_lists_the_candidates_and_posts_nothing(monkeypatch):
    second = dict(_RUNNING_OP, id="7f3aZZZZZZZZ", label="RAG re-embed")
    srv = _install(monkeypatch, models_cli,
                   _cancel_server([_RUNNING_OP, second]))
    result = _run(models_cli.cancel_cmd, ["7f3a"])
    assert result.exit_code == 1
    assert not [p for p in srv.paths() if "cancel" in p]
    assert "Model pull" in result.output and "RAG re-embed" in result.output


def test_no_running_server_says_so_and_exits_nonzero(monkeypatch):
    srv = _install(monkeypatch, models_cli, _cancel_server([_RUNNING_OP]),
                   discovered=False)
    result = _run(models_cli.cancel_cmd, ["7f3a"])
    assert result.exit_code == 1
    assert srv.calls == []
    assert "No running localm server" in result.output


def test_a_cancel_that_the_server_refuses_is_not_reported_as_done(monkeypatch):
    _install(monkeypatch, models_cli,
             _cancel_server([_RUNNING_OP], cancel=_Resp(401, {})))
    result = _run(models_cli.cancel_cmd, ["7f3a"])
    assert result.exit_code == 1
    assert "Cancelling" not in result.output


def test_a_job_that_vanished_between_the_read_and_the_post_says_so(monkeypatch):
    """404 on the cancel route, having already resolved the id from a live
    activity listing. That is an evicted job, not a server without the route."""
    _install(monkeypatch, models_cli,
             _cancel_server([_RUNNING_OP], cancel=_Resp(404, {})))
    result = _run(models_cli.cancel_cmd, ["7f3a"])
    assert result.exit_code == 1
    assert "no such item" in result.output.lower()
    assert "predates" not in result.output.lower()


# --------------------------------------------------------------------------
#  localm comfy status
# --------------------------------------------------------------------------

def _status_server(alive=True, ours=True, status_resp=None):
    return _Server({("GET", "/v1/comfy/status"):
                    status_resp or _Resp(200, {"alive": alive,
                                               "launched_by_localm": ours})})


def _no_direct_comfy(monkeypatch, alive=False):
    """Pin the direct liveness probe so the fallback never touches the network."""
    import localm.media.comfy_client as cc
    monkeypatch.setattr(cc, "_comfy_alive", lambda *a, **k: alive)


def test_status_reports_alive_and_whether_localm_launched_it(monkeypatch):
    _install(monkeypatch, comfy_cli, _status_server(alive=True, ours=True))
    _no_direct_comfy(monkeypatch)
    result = _run(comfy_cli.comfy_status)
    assert result.exit_code == 0, result.output
    assert "Launched by localm: yes" in result.output


def test_status_without_a_server_reports_unknown_never_no(monkeypatch):
    """THE CENTRAL ONE. Only the process holding the subprocess handle knows
    whether localm launched ComfyUI. With no server to ask, "no" would be a
    claim; the honest word is "unknown"."""
    _install(monkeypatch, comfy_cli, _status_server(), discovered=False)
    _no_direct_comfy(monkeypatch, alive=True)
    result = _run(comfy_cli.comfy_status)
    assert result.exit_code == 0, result.output
    assert "Launched by localm: unknown" in result.output
    assert "Launched by localm: no" not in result.output


def test_status_that_could_not_ask_the_server_still_never_says_no(monkeypatch):
    """Same rule one layer over: a server that refuses the question has told
    us nothing about who launched ComfyUI."""
    _install(monkeypatch, comfy_cli,
             _status_server(status_resp=_Resp(401, {})))
    _no_direct_comfy(monkeypatch, alive=True)
    result = _run(comfy_cli.comfy_status)
    assert "Launched by localm: no" not in result.output
    assert "Launched by localm: unknown" in result.output
    # ...and it still answers the half a direct probe CAN answer.
    assert "Running           : yes" in result.output


def test_no_ping_makes_no_request_at_all(monkeypatch):
    srv = _install(monkeypatch, comfy_cli, _status_server())
    result = _run(comfy_cli.comfy_status, ["--no-ping"])
    assert result.exit_code == 0, result.output
    assert srv.calls == []
    assert "not checked" in result.output


# --------------------------------------------------------------------------
#  localm comfy stop / restart
# --------------------------------------------------------------------------

def test_stop_posts_to_the_stop_route(monkeypatch):
    srv = _install(monkeypatch, comfy_cli, _Server(
        {("POST", "/v1/comfy/stop"): _Resp(200, {"ok": True,
                                                 "message": "Stopped it."})}))
    result = _run(comfy_cli.comfy_stop)
    assert result.exit_code == 0, result.output
    assert srv.paths() == ["/v1/comfy/stop"]
    assert "Stopped it." in result.output


def test_a_200_saying_ok_false_is_a_failure_not_a_success(monkeypatch):
    """The routes handle a refusal cleanly and answer 200 with ok=false, so
    reading only the status code would report a failed stop as done - the
    discarded-return-value shape."""
    _install(monkeypatch, comfy_cli, _Server(
        {("POST", "/v1/comfy/stop"):
         _Resp(200, {"ok": False, "message": "Could not stop the ComfyUI localm launched."})}))
    result = _run(comfy_cli.comfy_stop)
    assert result.exit_code == 1, result.output
    assert "Could not stop" in result.output


def test_stop_without_a_server_posts_nothing(monkeypatch):
    srv = _install(monkeypatch, comfy_cli, _Server({}), discovered=False)
    result = _run(comfy_cli.comfy_stop)
    assert result.exit_code == 1
    assert srv.calls == []


def test_restart_posts_to_the_restart_route(monkeypatch):
    srv = _install(monkeypatch, comfy_cli, _Server(
        {("POST", "/v1/comfy/restart"): _Resp(200, {"ok": True,
                                                    "message": "Restarted."})}))
    assert _run(comfy_cli.comfy_restart).exit_code == 0
    assert srv.paths() == ["/v1/comfy/restart"]


def test_the_client_waits_longer_than_the_server_will(monkeypatch):
    """A client that gives up first reports a failure the server never had.
    The restart route budgets 90s for the stop half plus the configured launch
    wait, so the client's own budget must exceed that sum."""
    from localm.media.comfy_client import comfy_launch_wait_seconds
    cfg = {"comfy_launch_timeout": 600}
    server_budget = 90.0 + comfy_launch_wait_seconds(cfg) + 30.0
    assert comfy_cli._restart_timeout(cfg) > server_budget
    assert comfy_cli._launch_timeout(cfg) > comfy_launch_wait_seconds(cfg) + 30.0


# --------------------------------------------------------------------------
#  localm comfy start
# --------------------------------------------------------------------------

_ALIVE = _Resp(200, {"alive": True, "launched_by_localm": True})
_DEAD = _Resp(200, {"alive": False, "launched_by_localm": False})
_LAUNCHED = _Resp(200, {"ok": True, "message": "ComfyUI is up."})


def test_start_never_touches_a_comfyui_that_is_already_running(monkeypatch):
    """The no-media-plugin fallback is POST /v1/comfy/restart, whose stop half
    ABORTS the in-flight render. On an already-running ComfyUI that would
    destroy work the user never asked to interrupt, so start has to establish
    it is down before it can reach any launch route."""
    srv = _install(monkeypatch, comfy_cli,
                   _Server({("GET", "/v1/comfy/status"): _ALIVE,
                            ("POST", "/v1/comfy/restart"): _LAUNCHED,
                            ("POST", "/api/imagine/comfy-launch"): _LAUNCHED}))
    result = _run(comfy_cli.comfy_start)
    assert result.exit_code == 0, result.output
    assert srv.paths() == ["/v1/comfy/status"], srv.paths()
    assert "already running" in result.output


def test_start_uses_the_media_plugins_own_launch_route(monkeypatch):
    srv = _install(monkeypatch, comfy_cli,
                   _Server({("GET", "/v1/comfy/status"): _DEAD,
                            ("POST", "/api/imagine/comfy-launch"): _LAUNCHED}))
    result = _run(comfy_cli.comfy_start)
    assert result.exit_code == 0, result.output
    assert "/api/imagine/comfy-launch" in srv.paths()
    assert "/v1/comfy/restart" not in srv.paths()


def test_start_falls_back_to_the_kernel_route_when_no_media_plugin_exists(monkeypatch):
    """All three per-plugin routes 404 because no media plugin is installed.
    That is a missing ROUTE, not a failed launch, so it must not be reported as
    one - and the kernel route can still start ComfyUI from the global config."""
    srv = _install(monkeypatch, comfy_cli,
                   _Server({("GET", "/v1/comfy/status"): _DEAD,
                            ("POST", "/v1/comfy/restart"): _LAUNCHED}))
    result = _run(comfy_cli.comfy_start)
    assert result.exit_code == 0, result.output
    assert srv.paths()[-1] == "/v1/comfy/restart"
    assert "/api/imagine/comfy-launch" in srv.paths()
    assert "/api/video/comfy-launch" in srv.paths()


def test_the_render_aborting_fallback_is_never_used_on_an_unconfirmed_comfyui(monkeypatch):
    """A server too old to answer /v1/comfy/status has told us nothing about
    whether a render is in flight. The per-plugin launch routes are safe there
    (they only bring ComfyUI up), but /v1/comfy/restart aborts the running
    prompt, so it must not be reached on a guess.

    The guard is on the status answer itself, not on an assumption that the
    three routes always shipped together."""
    srv = _install(monkeypatch, comfy_cli,
                   _Server({("GET", "/v1/comfy/status"): _Resp(404, {}),
                            ("POST", "/v1/comfy/restart"): _LAUNCHED}))
    result = _run(comfy_cli.comfy_start)
    assert result.exit_code == 1, result.output
    assert "/v1/comfy/restart" not in srv.paths(), srv.paths()
    # It still TRIED the safe per-plugin routes before giving up.
    assert "/api/imagine/comfy-launch" in srv.paths()


def test_an_explicitly_named_missing_plugin_never_falls_back(monkeypatch):
    """--media music asks for THAT plugin's ComfyUI settings. Silently
    launching from the global config instead would answer a question the user
    did not ask."""
    srv = _install(monkeypatch, comfy_cli,
                   _Server({("GET", "/v1/comfy/status"): _DEAD,
                            ("POST", "/v1/comfy/restart"): _LAUNCHED}))
    result = _run(comfy_cli.comfy_start, ["--media", "music"])
    assert result.exit_code == 1, result.output
    assert "/v1/comfy/restart" not in srv.paths()
    assert "/api/imagine/comfy-launch" not in srv.paths()
    assert "music" in result.output


def test_start_stops_when_it_cannot_establish_comfyui_is_down(monkeypatch):
    """An unauthorized status read tells us nothing about whether a render is
    in flight, so proceeding to a route that aborts one is not available."""
    srv = _install(monkeypatch, comfy_cli,
                   _Server({("GET", "/v1/comfy/status"): _Resp(401, {}),
                            ("POST", "/api/imagine/comfy-launch"): _LAUNCHED}))
    result = _run(comfy_cli.comfy_start)
    assert result.exit_code == 1
    assert srv.paths() == ["/v1/comfy/status"]


# --------------------------------------------------------------------------
#  Ctrl-C during a CLI generation
# --------------------------------------------------------------------------

class _Comfy:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.interrupted = []
        self.freed = []

    def install(self, monkeypatch):
        import localm.media.comfy_client as cc
        monkeypatch.setattr(cc, "interrupt_comfy",
                            lambda u: (self.interrupted.append(u), self.accepted)[1])
        monkeypatch.setattr(cc, "free_comfy_vram",
                            lambda u=None: (self.freed.append(u), True)[1])
        return self


def test_ctrl_c_tells_comfyui_to_abort_the_render_and_free_vram(monkeypatch):
    """Ctrl-C must not leave ComfyUI rendering and holding its VRAM.
    interrupt_comfy/free_comfy_vram are plain HTTP, so they work from any
    process, unlike stop_comfy, which needs a handle only the launching process
    has."""
    comfy = _Comfy().install(monkeypatch)

    def _boom():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        media_cli._generate_or_abort("http://comfy:8188", _boom)
    assert comfy.interrupted == ["http://comfy:8188"]
    assert comfy.freed == ["http://comfy:8188"]


def test_an_abort_comfyui_did_not_accept_is_not_reported_as_stopped(monkeypatch, capsys):
    """interrupt_comfy swallows its own transport errors and returns False.
    That is "I could not tell it to stop", and printing "render aborted" there
    would be the exact failure this whole change is about."""
    _Comfy(accepted=False).install(monkeypatch)

    def _boom():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        media_cli._generate_or_abort("http://comfy:8188", _boom)
    out = capsys.readouterr().out.lower()
    assert "aborted, queue cleared" not in out
    assert "did not accept" in out


def test_a_normal_generation_is_untouched(monkeypatch):
    comfy = _Comfy().install(monkeypatch)
    assert media_cli._generate_or_abort(
        "http://comfy:8188", lambda: (True, "done")) == (True, "done")
    assert comfy.interrupted == [] and comfy.freed == []
