# SPDX-License-Identifier: AGPL-3.0-or-later
"""Browser tools called together share one browser per coder session.

The browser tools are not destructive, so the calls of one model turn run on a
thread pool. Two calls that both find no browser open must end up driving the
same one, not launch a browser each and leave one running that nothing closes.

The fake browser blocks in start() until a test opens its gate, and the
registry check is spied on, so each test knows both calls have found no browser
before the start is allowed to finish.
"""

import threading

import pytest

from localm.browser.session import BrowserUnavailableError

OWNER = "parallel-open-a"
OTHER = "parallel-open-b"
WAIT = 10.0
URL_A = "https://example.com/a"
URL_B = "https://example.com/b"


class _Session:
    browser_enabled = True

    def __init__(self, owner):
        self.job_owner = owner


class _FakeBrowser:
    """A BrowserSession with no Chromium: start() blocks until its gate opens."""

    def __init__(self, sid, gate, entered, failure):
        self.session_id = sid
        self._gate = gate
        self._entered = entered
        self._failure = failure
        self.stopped = False
        self.visited = []

    def start(self):
        self._entered.set()
        self._gate.wait(WAIT)
        if self._failure is not None:
            raise self._failure

    def navigate(self, url):
        self.visited.append(url)
        return {"ok": True, "url": url, "title": "t", "status": 200}

    def stop(self):
        self.stopped = True


class _Fleet:
    """Every fake browser built, plus a gate and an entered flag per session id."""

    def __init__(self):
        self.built = []
        self.failure = None
        self.checked = set()
        self.both_checked = threading.Event()
        self._gates = {}
        self._entered = {}
        self._lock = threading.Lock()

    def gate(self, sid):
        with self._lock:
            return self._gates.setdefault(sid, threading.Event())

    def entered(self, sid):
        with self._lock:
            return self._entered.setdefault(sid, threading.Event())

    def build(self, sid, **_kwargs):
        fake = _FakeBrowser(sid, self.gate(sid), self.entered(sid), self.failure)
        with self._lock:
            self.built.append(fake)
        return fake

    def found_none(self, ident):
        with self._lock:
            self.checked.add(ident)
            if len(self.checked) >= 2:
                self.both_checked.set()

    def open_every_gate(self):
        with self._lock:
            gates = list(self._gates.values())
        for gate in gates:
            gate.set()


@pytest.fixture
def fleet(monkeypatch):
    from localm.browser import session as bsession
    from localm.plugins.coder.tools import browser as bt

    fleet = _Fleet()
    monkeypatch.setattr(bsession, "BrowserSession", fleet.build)
    monkeypatch.setattr(bt, "_browser_config", lambda: {
        "headless": True, "engine": "bundled", "deny": [], "allow": []})
    real_existing = bt._existing

    def existing(session):
        live = real_existing(session)
        if live is None:
            fleet.found_none(threading.get_ident())
        return live

    monkeypatch.setattr(bt, "_existing", existing)
    yield fleet
    fleet.open_every_gate()
    for owner in (OWNER, OTHER):
        bt.close_for_owner(owner)
        bsession.close("coder-" + owner)


def _navigate_from_threads(cwd, owner, urls):
    """Start one browser_navigate per URL, each on its own thread."""
    from localm.plugins.coder.tools import browser as bt

    results = [None] * len(urls)

    def run(index, url):
        try:
            results[index] = bt.tool_browser_navigate(
                cwd, url=url, _session=_Session(owner))
        except BaseException as exc:                # noqa: BLE001
            results[index] = exc

    threads = [threading.Thread(target=run, args=(i, url), daemon=True)
               for i, url in enumerate(urls)]
    for thread in threads:
        thread.start()
    return threads, results


def _join(threads):
    for thread in threads:
        thread.join(WAIT)
    assert not [t for t in threads if t.is_alive()], "a browser call never returned"


class TestCallsMadeTogetherShareOneBrowser:
    def test_two_first_calls_start_one_browser(self, tmp_path, fleet):
        from localm.browser import session as bsession
        sid = "coder-" + OWNER
        before = set(bsession.active_ids())
        threads, results = _navigate_from_threads(tmp_path, OWNER, [URL_A, URL_B])
        assert fleet.both_checked.wait(WAIT), "the calls never both reached the launch"
        fleet.gate(sid).set()
        _join(threads)
        assert len(fleet.built) == 1, (
            "%d browsers were started for one coder session" % len(fleet.built))
        assert set(bsession.active_ids()) - before == {sid}
        assert bsession.get(sid) is fleet.built[0]
        assert not [b for b in fleet.built if b is not bsession.get(sid)
                    and not b.stopped], "a browser was left running unregistered"
        assert all(getattr(r, "ok", False) for r in results), [
            getattr(r, "output", r) for r in results]
        assert sorted(fleet.built[0].visited) == sorted([URL_A, URL_B])

    def test_a_failed_start_is_shared_and_holds_nothing(self, tmp_path, fleet):
        from localm.browser import session as bsession
        from localm.plugins.coder.tools import browser as bt
        sid = "coder-" + OWNER
        before = set(bsession.active_ids())
        fleet.failure = BrowserUnavailableError("no chromium build here")
        threads, results = _navigate_from_threads(tmp_path, OWNER, [URL_A, URL_B])
        assert fleet.both_checked.wait(WAIT), "the calls never both reached the launch"
        fleet.gate(sid).set()
        _join(threads)
        assert len(fleet.built) == 1, (
            "%d browsers were started for one coder session" % len(fleet.built))
        assert set(bsession.active_ids()) == before, "a failed start still holds the id"
        assert bt.owned_session_id(OWNER) is None
        assert all(getattr(r, "ok", True) is False
                   and "no chromium build here" in r.output for r in results), [
            getattr(r, "output", r) for r in results]
        fleet.failure = None
        again = bt.tool_browser_navigate(tmp_path, url=URL_A, _session=_Session(OWNER))
        assert again.ok is True, again.output
        assert len(fleet.built) == 2
        assert bsession.get(sid) is fleet.built[1]

    def test_a_later_call_reuses_the_open_browser(self, tmp_path, fleet):
        from localm.plugins.coder.tools import browser as bt
        fleet.gate("coder-" + OWNER).set()
        session = _Session(OWNER)
        first = bt.tool_browser_navigate(tmp_path, url=URL_A, _session=session)
        second = bt.tool_browser_navigate(tmp_path, url=URL_B, _session=session)
        assert first.ok and second.ok, (first.output, second.output)
        assert len(fleet.built) == 1
        assert fleet.built[0].visited == [URL_A, URL_B]

    def test_reopening_after_a_close_starts_a_new_browser(self, tmp_path, fleet):
        from localm.browser import session as bsession
        from localm.plugins.coder.tools import browser as bt
        sid = "coder-" + OWNER
        fleet.gate(sid).set()
        session = _Session(OWNER)
        assert bt.tool_browser_navigate(tmp_path, url=URL_A, _session=session).ok
        assert bt.tool_browser_close(tmp_path, _session=session).ok
        assert fleet.built[0].stopped is True
        again = bt.tool_browser_navigate(tmp_path, url=URL_B, _session=session)
        assert again.ok is True, again.output
        assert len(fleet.built) == 2
        assert bsession.get(sid) is fleet.built[1]
        assert fleet.built[1].visited == [URL_B]

    def test_another_session_does_not_wait_for_a_slow_start(self, tmp_path, fleet):
        slow, quick = "coder-" + OWNER, "coder-" + OTHER
        fleet.gate(quick).set()
        slow_threads, slow_results = _navigate_from_threads(tmp_path, OWNER, [URL_A])
        assert fleet.entered(slow).wait(WAIT), "the slow start never began"
        quick_threads, quick_results = _navigate_from_threads(tmp_path, OTHER, [URL_B])
        quick_threads[0].join(3.0)
        finished_while_slow_was_starting = not quick_threads[0].is_alive()
        fleet.gate(slow).set()
        _join(slow_threads + quick_threads)
        assert finished_while_slow_was_starting, (
            "a start for one coder session held up another session's browser")
        assert all(getattr(r, "ok", False)
                   for r in slow_results + quick_results), [
            getattr(r, "output", r) for r in slow_results + quick_results]
