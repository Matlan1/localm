# SPDX-License-Identifier: AGPL-3.0-or-later
"""Characterization of `localm gui` startup: what each flag combination
visibly does, pinned so a restructuring of the startup code can be checked
for parity by running this module unchanged before and after it.

Every test drives the real command (CliRunner) through its real startup.
Only the edges that would leave the process are replaced by recorders:

* the socket server (``portmux.run_server``), so ``http_server.run_advertised``,
  the instance-registry advertisement and the shutdown teardown all run for real;
* the model backend (``inference.engine.Engine``), so no weights are loaded;
* the browser, the native app window and the tray/status window;
* the mDNS advertiser, the LAN/Tailscale probes and certificate minting;
* the Windows console calls and the HTTP call to an already-running instance;
* the loopback readiness probe (``socket.create_connection``), except in the
  one test that checks the real wait for a listening server.

Assertions are on effects only: exit codes, console text, which engine and
app were built, what was served where and how, which surfaces were opened,
what was advertised, and that everything startup started was stopped again.
They never name a private helper of the GUI command, so moving startup code
between functions leaves them untouched. The fixture's own teardown also
fails any test whose run left a thread running or its own instance-registry
entry behind.
"""

import contextlib
import os
import re
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse

import pytest
from click.testing import CliRunner

# Imported up front, at collection, so no module is imported for the FIRST time
# while a test has one of these patched (a module-level ``from x import y``
# executed inside the patch window would keep the double past teardown).
import localm.cli  # noqa: F401
import localm.inference.http_engine  # noqa: F401
import localm.plugins.gui.web as gui_web
from localm import appface, instances, netname, portmux, tls
from localm.config import home_dir, load_config, load_registry, save_config, save_registry
from localm.inference import engine as engine_mod
from localm.inference import http_server as hs
from localm.plugins.gui import cli as guicli
from tests.conftest import make_console_wide_and_plain

STRONG_KEY = "characterization-owner-key-0123456789"
PEER_PORT = 8793


def _flat(text: str) -> str:
    return " ".join(text.split())


def _banner(flat: str, name: str, url: str) -> bool:
    """True when *flat* has the startup banner line "<name> <arrow> <url>".
    The arrow glyph is left unpinned: a console may re-encode it."""
    return re.search(rf"{re.escape(name)} \S+ {re.escape(url)}", flat) is not None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _listening(port: int = 0):
    """A real listener on 127.0.0.1, so the port reads as in use."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        s.listen(8)
        yield s
    finally:
        s.close()


def _query(url) -> dict:
    return {k: v[0] for k, v in parse_qs(url.query).items()}


class _FakeEngine:
    """Stands in for inference.engine.Engine: records how it was built and
    whether (and on which thread) it was loaded and unloaded."""

    def __init__(self, model_path, *, load_error=None, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.display_name = kwargs.get("display_name") or "fake-model"
        self.active_requests = 0
        self.unloading = False
        self.gpu_placement = None
        self.load_error = load_error
        self.load_calls = 0
        self.load_on_main_thread = None
        self.unloaded = False
        self._loaded = False

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.load_calls += 1
        self.load_on_main_thread = threading.current_thread() is threading.main_thread()
        if self.load_error is not None:
            raise self.load_error
        self._loaded = True

    def unload(self):
        self.unloaded = True
        self._loaded = False

    def set_load_cancel(self, event):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


class _Face:
    """Stands in for the tray / status window appface.start_app_face returns."""

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.ready = threading.Event()
        self.errors = []
        self.closed = 0

    def set_ready(self):
        self.ready.set()

    def set_error(self, text):
        self.errors.append(text)

    def set_status(self, text):
        pass

    def close(self):
        self.closed += 1


class _Advertiser:
    def __init__(self, port, tls, addresses):
        self.port, self.tls, self.addresses = port, tls, addresses
        self.closed = 0

    def close(self):
        self.closed += 1


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


class _Startup:
    """One `localm gui` invocation's world: knobs set before invoke(), and
    recorders read after it. Every knob is read when the startup reaches it,
    so a test may set it any time before invoke()."""

    def __init__(self, monkeypatch, tmp_path):
        self.mp = monkeypatch
        self.tmp_path = tmp_path
        # knobs
        self.native = False          # appface.native_window_available()
        self.window_loads = True     # appface.run_native_window() result
        self.tray = False            # start_app_face returns a face
        self.mdns_ok = True          # netname.start_advertiser returns a handle
        self.mount_status = 200      # a running api-mode instance's mount reply
        self.engine_error = None     # Engine(...) raises this
        self.load_error = None       # engine.load() raises this
        self.real_readiness = False  # leave socket.create_connection real
        self.during_serve = None     # callable(app), run while "serving"
        # recorders
        self.served = []
        self.engines = []
        self.apps = []
        self.gui_mounts = []
        self.managers_closed = 0
        self.opened = []
        self.windows = []
        self.windows_closed = 0
        self.restart_ui = []
        self.titles = []
        self.console_handlers = []
        self.faces = []
        self.advertisers = []
        self.certs = []
        self.mounts = []
        self.hang_surfaces = []
        self.leaked = []
        self.live_ids = set()
        self._before = set()

    # -- world setup ---------------------------------------------------- #

    def register_model(self, name, *, model_type="llm"):
        path = self.tmp_path / "weights" / f"{name}.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF" + b"\x00" * 64)
        reg = load_registry()
        reg[name] = {"path": str(path), "source": "local", "model_type": model_type}
        save_registry(reg)
        return path

    def set_config(self, **updates):
        cfg = load_config()
        cfg.update(updates)
        save_config(cfg)

    def running_instance(self, *, mode="full", port=PEER_PORT, token="tok"):
        """Register a live localm already serving this project dir."""
        iid = f"peer{len(self.live_ids)}"
        instances.register_instance(
            home_dir(), instance_id=iid, port=port, host="127.0.0.1",
            root_dir=instances.resolve_root_dir(), mode=mode, token=token)
        self.live_ids.add(iid)
        return iid

    def restarting(self, surface):
        """This process is a server restart's re-exec, whose previous run
        showed *surface*."""
        self.mp.setenv("LOCALM_RESTART_IN_PROGRESS", surface)

    # -- the run ------------------------------------------------------- #

    def invoke(self, *args):
        self._before = set(threading.enumerate())
        result = CliRunner().invoke(guicli.main, list(args))
        for t in self._new_threads():
            t.join(10.0)
        self.leaked = [t.name for t in self._new_threads() if t.is_alive()]
        result.flat = _flat(result.output)
        return result

    def _new_threads(self):
        return [t for t in threading.enumerate()
                if t not in self._before and t is not threading.current_thread()]

    def _settle(self):
        """Let what startup set in motion finish while the server is up, so
        what it prints lands in this run's output. Only when serving holds the
        main thread: with the app window on it, the stop-signal relay started
        for the server thread lives exactly as long as serving does, and the
        post-run join in invoke() still covers everything else."""
        me = threading.current_thread()
        if me is not threading.main_thread():
            return
        for t in threading.enumerate():
            if t not in self._before and t is not me:
                t.join(10.0)

    @property
    def serve(self):
        assert len(self.served) == 1, f"served {len(self.served)} times"
        return self.served[0]

    @property
    def engine(self):
        assert len(self.engines) == 1, f"built {len(self.engines)} engines"
        return self.engines[0]

    @property
    def app(self):
        assert len(self.apps) == 1, f"built {len(self.apps)} apps"
        return self.apps[0][0]

    @property
    def app_engine(self):
        assert len(self.apps) == 1, f"built {len(self.apps)} apps"
        return self.apps[0][1]

    @property
    def face(self):
        assert len(self.faces) == 1, f"started {len(self.faces)} trays"
        return self.faces[0]

    def opened_url(self):
        assert len(self.opened) == 1, f"opened {self.opened}"
        return urlparse(self.opened[0])

    def own_entries(self, entries):
        return [e for e in entries if e.get("instance_id") not in self.live_ids]

    def nothing_started(self):
        """True when startup stopped before building or showing anything."""
        return (self.served == [] and self.apps == [] and self.engines == []
                and self.opened == [] and self.windows == [] and self.faces == []
                and self.advertisers == [])

    # -- doubles ------------------------------------------------------- #

    def install(self):
        mp = self.mp
        make_console_wide_and_plain(mp)
        mp.setattr("localm.winconsole.disable_quickedit", lambda: None)
        mp.setattr("localm.winconsole.register_console_handler",
                   lambda cleanup: self.console_handlers.append(cleanup) or True)
        mp.setattr("localm.winconsole.set_console_title",
                   lambda title: self.titles.append(title) or True)
        mp.setattr("localm.applaunch.apply_window_identity", lambda *a, **k: True)
        mp.delenv("LOCALM_RESTART_IN_PROGRESS", raising=False)
        mp.delenv("LOCALM_MODE", raising=False)
        mp.delenv("LOCALM_OWN_CONSOLE", raising=False)

        # The project this invocation runs in keys attach-or-spawn.
        proj = self.tmp_path / "proj"
        (proj / ".localcoder").mkdir(parents=True)
        mp.chdir(proj)
        mp.setattr(instances, "default_probe",
                   lambda entry, **k: entry.get("instance_id") in self.live_ids)

        # Model backend and app assembly (the real ones, observed).
        def build_engine(model_path, **k):
            if self.engine_error is not None:
                raise self.engine_error
            eng = _FakeEngine(model_path, load_error=self.load_error, **k)
            self.engines.append(eng)
            return eng
        mp.setattr(engine_mod, "Engine", build_engine)

        real_create_app = hs.create_app

        def create_app(engine, *a, **k):
            app = real_create_app(engine, *a, **k)
            self.apps.append((app, engine))
            return app
        mp.setattr(hs, "create_app", create_app)

        real_attach_gui = gui_web.attach_gui

        def attach_gui(app, **k):
            manager = real_attach_gui(app, **k)
            self.gui_mounts.append(k)
            real_close_all = manager.close_all

            def close_all():
                self.managers_closed += 1
                return real_close_all()
            manager.close_all = close_all
            return manager
        mp.setattr(gui_web, "attach_gui", attach_gui)

        # The server itself: records how it was asked to serve, then "serves"
        # until everything startup started has finished.
        def run_server(app, *a, **k):
            self.served.append({
                "app": app, "host": k.get("host"), "port": k.get("port"),
                "ssl_certfile": k.get("ssl_certfile"),
                "ssl_keyfile": k.get("ssl_keyfile"),
                "on_main_thread": threading.current_thread() is threading.main_thread(),
                "restart_flag": os.environ.get("LOCALM_RESTART_IN_PROGRESS"),
                "entries": instances.list_entries(home_dir()),
                "faces_closed": [f.closed for f in self.faces],
                "advertisers_closed": [a.closed for a in self.advertisers],
                "managers_closed": self.managers_closed,
                "windows_closed": self.windows_closed,
            })
            if self.during_serve is not None:
                self.during_serve(app)
            self._settle()
        mp.setattr(portmux, "run_server", run_server)
        mp.setattr(hs, "set_restart_ui", self.restart_ui.append)
        mp.setattr(hs, "set_hang_surface",
                   lambda surface, recovered: self.hang_surfaces.append(
                       (surface, recovered)))

        # Surfaces.
        mp.setattr(appface, "native_window_available", lambda: self.native)

        def run_native_window(url, *a, **k):
            self.windows.append({
                "url": url, **k,
                "on_main_thread": threading.current_thread() is threading.main_thread()})
            return self.window_loads
        mp.setattr(appface, "run_native_window", run_native_window)

        def close_native_window():
            self.windows_closed += 1
        mp.setattr(appface, "close_native_window", close_native_window)

        def start_app_face(**k):
            if not self.tray:
                return None
            face = _Face(k)
            self.faces.append(face)
            return face
        mp.setattr(appface, "start_app_face", start_app_face)
        mp.setattr("webbrowser.open", lambda url, *a, **k: self.opened.append(url))

        real_create_connection = socket.create_connection

        def create_connection(address, *a, **k):
            if self.real_readiness:
                return real_create_connection(address, *a, **k)
            return contextlib.nullcontext()
        mp.setattr(socket, "create_connection", create_connection)

        # Network naming, TLS, and a running instance's mount endpoint.
        def start_advertiser(port, *, tls, addresses=None):
            if not self.mdns_ok:
                return None
            adv = _Advertiser(port, tls, addresses)
            self.advertisers.append(adv)
            return adv
        mp.setattr(netname, "start_advertiser", start_advertiser)
        mp.setattr(netname, "network_targets",
                   lambda **k: [("LAN (IP)", "192.0.2.10")])
        mp.setattr(netname, "tailscale_rename_hint", lambda: None)
        mp.setattr(netname, "cert_hostnames", lambda: [])
        mp.setattr(tls, "companion_addresses", lambda: {"lan": "192.0.2.10"})

        def ensure_cert(home, hostnames=None, ips=None):
            self.certs.append(list(hostnames or []))
            return "FAKE.crt", "FAKE.key"
        mp.setattr(tls, "ensure_cert", ensure_cert)

        def post(url, *a, **k):
            self.mounts.append({"url": url, "headers": dict(k.get("headers") or {})})
            return _Response(self.mount_status)
        mp.setattr("requests.post", post)
        return self


@pytest.fixture
def gui(monkeypatch, tmp_path):
    world = _Startup(monkeypatch, tmp_path).install()
    yield world
    assert world.leaked == [], f"startup left threads running: {world.leaked}"
    left = world.own_entries(instances.list_entries(home_dir()))
    assert left == [], f"startup left its instance-registry entry behind: {left}"


# ------------------------------------------------------------------ #
#  Model selection and engine construction                           #
# ------------------------------------------------------------------ #

class TestModelSelection:
    def test_no_model_builds_no_engine_even_with_a_usable_model_registered(self, gui):
        gui.register_model("usable")
        r = gui.invoke("--no-model")
        assert r.exit_code == 0, r.output
        assert gui.engines == []
        assert gui.app_engine is None
        port = gui.serve["port"]
        assert _banner(r.flat, "localm GUI", f"http://127.0.0.1:{port}/")
        assert "Opening with no model loaded - pick one on the Models page." in r.flat
        assert "model: none yet - add one on the Models page" in r.flat
        assert "Ctrl+C to stop" in r.flat
        url = gui.opened_url()
        assert (url.scheme, url.netloc, url.path) == ("http", f"127.0.0.1:{port}", "/")
        assert _query(url) == {"view": "models"}
        assert f"Open the GUI: {gui.opened[0]}" in r.flat

    def test_first_usable_registered_model_is_built_preloaded_and_unloaded(self, gui):
        path = gui.register_model("usable")
        r = gui.invoke()
        assert r.exit_code == 0, r.output
        eng = gui.engine
        assert eng.model_path == str(path)
        assert eng.kwargs["display_name"] == "usable"
        assert gui.app_engine is eng
        # Preloaded once, in the background, while the server comes up ...
        assert eng.load_calls == 1
        assert eng.load_on_main_thread is False
        # ... and released by the stop sequence once serving ends.
        assert eng.unloaded
        port = gui.serve["port"]
        assert "model: usable" in r.flat
        assert gui.titles[-1] == f"LocaLM  -  usable  -  :{port}"
        url = gui.opened_url()
        assert (url.netloc, url.path, url.query) == (f"127.0.0.1:{port}", "/", "")

    def test_model_options_reach_the_engine(self, gui, tmp_path):
        gui.register_model("usable")
        proj = tmp_path / "vision-proj.gguf"
        proj.write_bytes(b"GGUF")
        r = gui.invoke("usable", "--ctx", "4096", "--gpu-layers", "12",
                       "--device", "cpu", "--mmproj", str(proj), "--no-browser")
        assert r.exit_code == 0, r.output
        kw = gui.engine.kwargs
        assert (kw["n_ctx"], kw["n_gpu_layers"], kw["device"], kw["mmproj_path"]) == (
            4096, 12, "cpu", str(proj))

    def test_a_model_path_on_disk_is_accepted(self, gui, tmp_path):
        loose = tmp_path / "loose" / "tiny.gguf"
        loose.parent.mkdir()
        loose.write_bytes(b"GGUF" + b"\x00" * 64)
        r = gui.invoke(str(loose), "--no-browser")
        assert r.exit_code == 0, r.output
        assert gui.engine.model_path == str(loose)
        assert "model: tiny" in r.flat

    def test_empty_registry_opens_model_less(self, gui):
        r = gui.invoke()
        assert r.exit_code == 0, r.output
        assert gui.engines == []
        assert gui.app_engine is None
        assert ("No models registered yet. Opening the GUI - add one on the "
                "Models page.") in r.flat
        assert _query(gui.opened_url()) == {"view": "models"}

    def test_registry_without_a_loadable_chat_model_opens_model_less(self, gui):
        gui.register_model("mystery", model_type="unknown")
        gui.register_model("embedder", model_type="embedding")
        r = gui.invoke()
        assert r.exit_code == 0, r.output
        assert gui.engines == []
        assert "No loadable chat models in the registry" in r.flat
        assert _query(gui.opened_url()) == {"view": "models"}

    def test_an_unknown_model_name_exits_1_before_anything_starts(self, gui):
        gui.register_model("usable")
        r = gui.invoke("no-such-model")
        assert r.exit_code == 1, r.output
        assert "Model not found: no-such-model" in r.flat
        assert gui.nothing_started()

    def test_engine_construction_failure_degrades_to_model_less(self, gui):
        gui.register_model("usable")
        gui.engine_error = RuntimeError("bad weights")
        r = gui.invoke()
        assert r.exit_code == 0, r.output
        assert "Could not load model 'usable': bad weights" in r.flat
        assert "Opening the GUI model-less - pick a model on the Models page." in r.flat
        assert gui.app_engine is None
        assert "model: none yet" in r.flat
        assert _query(gui.opened_url()) == {"view": "models"}
        assert gui.serve["port"]

    def test_background_load_failure_is_reported_and_serving_continues(self, gui):
        gui.register_model("usable")
        gui.load_error = RuntimeError("out of memory")
        r = gui.invoke("--no-browser")
        assert r.exit_code == 0, r.output
        assert gui.engine.load_calls == 1
        assert "Background model load failed: out of memory" in r.flat
        assert gui.app_engine is gui.engine
        assert gui.serve["port"]


# ------------------------------------------------------------------ #
#  What gets served, and the surfaces that open onto it               #
# ------------------------------------------------------------------ #

class TestServeAndSurfaces:
    def test_default_start_serves_full_gui_on_loopback_http(self, gui):
        r = gui.invoke("--no-model")
        assert r.exit_code == 0, r.output
        s = gui.serve
        port = s["port"]
        assert (s["host"], s["ssl_certfile"], s["ssl_keyfile"]) == ("127.0.0.1", None, None)
        assert s["on_main_thread"]
        assert s["app"] is gui.app
        assert gui.app.state.bind_host == "127.0.0.1"
        assert gui.app.state.bind_fallback is None
        # The GUI is mounted and talks to this server's own /v1.
        assert len(gui.gui_mounts) == 1
        assert gui.gui_mounts[0]["self_url"] == f"http://127.0.0.1:{port}/v1"
        # Advertised for attach while serving, as a full surface.
        [entry] = gui.own_entries(s["entries"])
        assert (entry["mode"], entry["port"], entry["host"], entry["scheme"]) == (
            "full", port, "127.0.0.1", "http")
        assert entry["root_dir"] == instances.resolve_root_dir()
        assert entry["pid"] == os.getpid()
        # No mDNS on loopback; the phone hint is printed instead.
        assert gui.advertisers == []
        assert "use from your phone: bind to your network with localm gui -H 0.0.0.0" in r.flat
        assert "Stopping localm..." in r.flat

    def test_console_close_cleanup_is_registered_once(self, gui):
        r = gui.invoke("--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        assert len(gui.console_handlers) == 1
        assert callable(gui.console_handlers[0])
        assert gui.titles[0] == "LocaLM"
        assert gui.titles[-1] == f"LocaLM  -  localhost:{gui.serve['port']}"

    def test_the_gui_manager_is_closed_after_serving_ends(self, gui):
        r = gui.invoke("--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        assert gui.serve["managers_closed"] == 0
        assert gui.managers_closed == 1

    def test_api_mode_serves_without_the_gui(self, gui):
        r = gui.invoke("--api-mode", "--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        port = gui.serve["port"]
        assert gui.gui_mounts == []
        assert gui.managers_closed == 0
        [entry] = gui.own_entries(gui.serve["entries"])
        assert entry["mode"] == "api"
        assert _banner(r.flat, "localm API server", f"http://127.0.0.1:{port}/")
        assert f"API base: http://127.0.0.1:{port}/" in r.flat
        assert "model: none yet - add one with `localm pull <name>`" in r.flat

    def test_no_browser_opens_no_surface_and_records_none(self, gui):
        gui.native = True
        r = gui.invoke("--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        assert gui.opened == []
        assert gui.windows == []
        assert gui.restart_ui == []
        assert gui.serve["on_main_thread"]

    def test_browser_mode_opens_one_tab_and_records_it(self, gui):
        r = gui.invoke("--no-model")
        assert r.exit_code == 0, r.output
        assert len(gui.opened) == 1
        assert gui.windows == []
        assert gui.restart_ui == ["browser"]

    def test_native_window_takes_the_main_thread_and_the_server_moves_off_it(self, gui):
        gui.native = True
        r = gui.invoke("--no-model")
        assert r.exit_code == 0, r.output
        [window] = gui.windows
        port = gui.serve["port"]
        assert window["on_main_thread"]
        assert window["url"] == f"http://127.0.0.1:{port}/?view=models"
        assert callable(window["on_quit"])
        assert window["server_stopped"].is_set()
        assert gui.serve["on_main_thread"] is False
        assert gui.opened == []
        assert gui.restart_ui == ["window"]
        # The window is released only once the server has stopped.
        assert gui.serve["windows_closed"] == 0
        assert gui.windows_closed == 1

    def test_a_native_window_that_fails_falls_back_to_a_tab(self, gui):
        gui.native = True
        gui.window_loads = False
        r = gui.invoke("--no-model")
        assert r.exit_code == 0, r.output
        assert gui.opened == [gui.windows[0]["url"]]
        assert gui.restart_ui == ["window", "browser"]

    def test_tab_and_tray_wait_for_the_server_to_accept_connections(self, gui):
        gui.real_readiness = True
        gui.tray = True
        port = _free_port()
        seen = {}

        def during_serve(app):
            seen["opened_before"] = list(gui.opened)
            seen["ready_before"] = gui.face.ready.is_set()
            with _listening(port):
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and not (
                        gui.opened and gui.face.ready.is_set()):
                    time.sleep(0.05)
            seen["opened_after"] = list(gui.opened)
            seen["ready_after"] = gui.face.ready.is_set()

        gui.during_serve = during_serve
        r = gui.invoke("--no-model", "-p", str(port))
        assert r.exit_code == 0, r.output
        assert seen["opened_before"] == []
        assert seen["ready_before"] is False
        assert seen["opened_after"] == [f"http://127.0.0.1:{port}/?view=models"]
        assert seen["ready_after"] is True


# ------------------------------------------------------------------ #
#  Host and port                                                      #
# ------------------------------------------------------------------ #

class TestHostAndPort:
    def test_an_explicit_free_port_is_used(self, gui):
        port = _free_port()
        r = gui.invoke("--no-model", "--no-browser", "-p", str(port))
        assert r.exit_code == 0, r.output
        assert gui.serve["port"] == port
        assert "Default port busy" not in r.flat

    def test_an_explicit_busy_port_exits_1_and_is_never_relocated(self, gui):
        gui.register_model("usable")
        with _listening() as busy:
            port = busy.getsockname()[1]
            r = gui.invoke("-p", str(port))
        assert r.exit_code == 1, r.output
        assert (f"Port {port} is already in use. Free it, or choose another "
                "with -p/--port.") in r.flat
        assert gui.nothing_started()

    def test_a_busy_default_port_auto_bumps(self, gui):
        with _listening() as busy:
            taken = busy.getsockname()[1]
            gui.set_config(port=taken)
            r = gui.invoke("--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        port = gui.serve["port"]
        assert port != taken
        assert f"Default port busy - using {port}." in r.flat
        assert _banner(r.flat, "localm GUI", f"http://127.0.0.1:{port}/")

    def test_a_free_default_port_is_used_as_configured(self, gui):
        port = _free_port()
        gui.set_config(port=port)
        r = gui.invoke("--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        assert gui.serve["port"] == port
        assert "Default port busy" not in r.flat

    @pytest.mark.parametrize("key, why", [
        (None, "WITHOUT authentication"),
        ("short", "WEAK API key"),
    ])
    def test_a_network_bind_without_a_strong_key_exits_2(self, gui, monkeypatch, key, why):
        gui.register_model("usable")
        if key:
            monkeypatch.setenv("LOCALM_API_KEY", key)
        r = gui.invoke("-H", "0.0.0.0")
        assert r.exit_code == 2, r.output
        assert why in r.flat
        assert ("Refusing to start: binding past loopback without auth. Set "
                "$env:LOCALM_API_KEY first, or pass --insecure to override.") in r.flat
        assert gui.nothing_started()
        assert gui.certs == []

    def test_insecure_overrides_the_key_requirement_and_still_serves_tls(self, gui):
        r = gui.invoke("-H", "0.0.0.0", "--insecure", "--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        assert "WITHOUT authentication" in r.flat
        assert "Proceeding anyway (--insecure set)." in r.flat
        s = gui.serve
        assert (s["host"], s["ssl_certfile"], s["ssl_keyfile"]) == (
            "0.0.0.0", "FAKE.crt", "FAKE.key")
        assert gui.app.state.bind_host == "0.0.0.0"

    def test_a_keyed_network_bind_serves_tls_advertises_and_withdraws(self, gui, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", STRONG_KEY)
        r = gui.invoke("-H", "0.0.0.0", "--no-model")
        assert r.exit_code == 0, r.output
        assert "Refusing to start" not in r.flat
        assert "Proceeding anyway" not in r.flat
        s = gui.serve
        port = s["port"]
        assert (s["host"], s["ssl_certfile"], s["ssl_keyfile"]) == (
            "0.0.0.0", "FAKE.crt", "FAKE.key")
        assert gui.certs == [["0.0.0.0"]]
        assert gui.app.state.bind_host == "0.0.0.0"
        assert gui.gui_mounts[0]["self_url"] == f"https://127.0.0.1:{port}/v1"
        [entry] = gui.own_entries(s["entries"])
        assert (entry["host"], entry["scheme"]) == ("0.0.0.0", "https")
        # Reach-by-name while serving, withdrawn once serving ends.
        [adv] = gui.advertisers
        assert (adv.port, adv.tls) == (port, True)
        assert s["advertisers_closed"] == [0]
        assert adv.closed == 1
        # Console: loopback banner, the LAN address, the certificate hint.
        assert _banner(r.flat, "localm GUI", f"https://127.0.0.1:{port}/")
        assert f"https://192.0.2.10:{port}/ (open it, then Install as app)" in r.flat
        assert "Install certificate" in r.flat
        assert f":{port}/localm-ca.crt" in r.flat
        assert "use from your phone" not in r.flat
        # The launching browser gets a one-time grant on top of the models view.
        url = gui.opened_url()
        assert (url.scheme, url.netloc) == ("https", f"127.0.0.1:{port}")
        q = _query(url)
        assert q["view"] == "models"
        assert q["localm_token"]

    def test_no_tls_serves_a_network_bind_in_plain_http(self, gui, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", STRONG_KEY)
        r = gui.invoke("-H", "0.0.0.0", "--no-tls", "--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        s = gui.serve
        assert (s["host"], s["ssl_certfile"]) == ("0.0.0.0", None)
        assert gui.certs == []
        [adv] = gui.advertisers
        assert adv.tls is False
        assert f"http://192.0.2.10:{s['port']}/" in r.flat
        assert "Install certificate" not in r.flat

    def test_an_isolated_network_bind_is_not_advertised(self, gui, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", STRONG_KEY)
        r = gui.invoke("-H", "0.0.0.0", "--isolated", "--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        assert gui.serve["host"] == "0.0.0.0"
        assert gui.advertisers == []
        assert gui.own_entries(gui.serve["entries"]) == []

    def test_a_loopback_start_with_a_key_hands_the_browser_a_grant(self, gui, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", STRONG_KEY)
        r = gui.invoke("--no-model")
        assert r.exit_code == 0, r.output
        q = _query(gui.opened_url())
        assert q["view"] == "models"
        assert q["localm_token"]

    def test_a_start_without_a_key_hands_out_no_grant(self, gui):
        r = gui.invoke("--no-model")
        assert r.exit_code == 0, r.output
        assert "localm_token" not in _query(gui.opened_url())


# ------------------------------------------------------------------ #
#  Attach to an already-running instance vs --new / --isolated        #
# ------------------------------------------------------------------ #

class TestAttachOrNew:
    def test_a_second_launch_attaches_instead_of_starting_a_server(self, gui):
        gui.register_model("usable")
        gui.running_instance(mode="full")
        gui.window_loads = False
        r = gui.invoke()
        assert r.exit_code == 0, r.output
        assert (f"Attaching to the localm already running for "
                f"{instances.resolve_root_dir()} (pid {os.getpid()}, "
                f"port {PEER_PORT}).") in r.flat
        assert gui.served == [] and gui.apps == [] and gui.engines == []
        assert gui.faces == [] and gui.advertisers == []
        assert gui.mounts == []   # already a full GUI: nothing to mount
        # The app window is tried first; when it cannot open, a fresh
        # (cache-busted) tab on the running server.
        [window] = gui.windows
        assert window["hide_on_close"] is False
        url = gui.opened_url()
        assert gui.opened == [window["url"]]
        assert (url.scheme, url.netloc, url.path) == ("http", f"127.0.0.1:{PEER_PORT}", "/")
        assert list(_query(url)) == ["lm"]
        assert f"Open the GUI: http://127.0.0.1:{PEER_PORT}/" in r.flat

    def test_an_attach_with_the_app_window_opens_no_tab(self, gui):
        gui.running_instance(mode="full")
        r = gui.invoke()
        assert r.exit_code == 0, r.output
        assert len(gui.windows) == 1
        assert gui.opened == []

    def test_an_attach_with_no_browser_opens_nothing(self, gui):
        gui.running_instance(mode="full")
        r = gui.invoke("--no-browser")
        assert r.exit_code == 0, r.output
        assert "Attaching" in r.flat
        assert gui.windows == [] and gui.opened == []
        assert gui.served == []

    def test_an_attach_to_an_api_instance_mounts_its_gui(self, gui):
        gui.running_instance(mode="api", token="peer-token")
        gui.window_loads = False
        r = gui.invoke()
        assert r.exit_code == 0, r.output
        assert gui.mounts == [{
            "url": f"http://127.0.0.1:{PEER_PORT}/v1/surfaces/gui",
            "headers": {"Authorization": "Bearer peer-token"}}]
        assert "Mounted the GUI on the running instance." in r.flat
        assert len(gui.opened) == 1
        assert gui.served == []

    def test_a_failed_mount_still_opens_the_address(self, gui):
        gui.running_instance(mode="api")
        gui.mount_status = 404
        gui.window_loads = False
        r = gui.invoke()
        assert r.exit_code == 0, r.output
        assert "Could not mount the GUI on it (an older instance?); opening its address anyway." in r.flat
        assert len(gui.opened) == 1

    @pytest.mark.parametrize("args, named", [
        (("--port", "8794"), f"--port 8794 (the running server is on {PEER_PORT})"),
        (("-H", "0.0.0.0"), "--host 0.0.0.0 (the running server bound 127.0.0.1)"),
        (("--pull", "org/repo"), "--pull"),
        (("--insecure",), "--insecure"),
        (("--ctx", "4096"), "--ctx"),
        (("--api-mode",), "--api-mode"),
        (("--no-tls",), "--no-tls"),
    ])
    def test_an_explicit_option_the_running_server_cannot_apply_exits_1(
            self, gui, args, named):
        gui.running_instance(mode="full")
        r = gui.invoke(*args)
        assert r.exit_code == 1, r.output
        assert "it cannot apply:" in r.flat
        assert f"- {named}" in r.flat
        assert "Start a SEPARATE server with your settings using --new" in r.flat
        assert "Attaching" not in r.flat
        assert gui.nothing_started()
        assert gui.mounts == []

    def test_naming_a_different_model_than_the_running_one_exits_1(self, gui, monkeypatch):
        gui.register_model("usable")
        gui.running_instance(mode="full")
        monkeypatch.setattr("localm.inference.http_engine.remote_model_status",
                            lambda *a, **k: ("loaded", "other-model"))
        r = gui.invoke("usable")
        assert r.exit_code == 1, r.output
        assert "- model usable (the running server serves other-model)" in r.flat
        assert gui.nothing_started()

    @pytest.mark.parametrize("args", [
        ("--port", str(PEER_PORT)), ("--no-model",), ("--project", "."),
    ])
    def test_options_that_agree_with_the_running_server_still_attach(self, gui, args):
        gui.running_instance(mode="full")
        r = gui.invoke(*args, "--no-browser")
        assert r.exit_code == 0, r.output
        assert "Attaching" in r.flat
        assert "cannot apply" not in r.flat
        assert gui.served == []

    def test_new_starts_a_separate_advertised_server(self, gui):
        peer = gui.running_instance(mode="full")
        port = _free_port()
        r = gui.invoke("--new", "--no-model", "--no-browser", "-p", str(port))
        assert r.exit_code == 0, r.output
        assert "Attaching" not in r.flat and "cannot apply" not in r.flat
        s = gui.serve
        assert s["port"] == port
        ids = {e["instance_id"] for e in s["entries"]}
        assert peer in ids and len(ids) == 2
        # Afterwards only the other instance's entry remains (fixture teardown).

    def test_isolated_starts_a_private_server_invisible_to_discovery(self, gui):
        peer = gui.running_instance(mode="full")
        r = gui.invoke("--isolated", "--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        assert "Attaching" not in r.flat
        assert [e["instance_id"] for e in gui.serve["entries"]] == [peer]


# ------------------------------------------------------------------ #
#  Restart re-exec                                                    #
# ------------------------------------------------------------------ #

class TestRestart:
    @pytest.mark.parametrize("previous", ["browser", "1"])
    def test_a_restart_from_a_browser_tab_opens_no_second_tab(self, gui, previous):
        gui.restarting(previous)
        r = gui.invoke("--no-model")
        assert r.exit_code == 0, r.output
        assert gui.opened == []
        assert gui.windows == []
        assert gui.restart_ui == ["browser"]
        # The flag is consumed before serving, so nothing the server starts
        # (or a later fresh launch) inherits it.
        assert gui.serve["restart_flag"] is None
        assert "LOCALM_RESTART_IN_PROGRESS" not in os.environ

    def test_a_restart_from_the_app_window_into_browser_mode_opens_a_tab(self, gui):
        gui.restarting("window")
        r = gui.invoke("--no-model")
        assert r.exit_code == 0, r.output
        assert len(gui.opened) == 1
        assert gui.restart_ui == ["browser"]
        assert gui.serve["restart_flag"] is None

    def test_a_restart_with_the_app_window_available_reopens_the_window(self, gui):
        gui.native = True
        gui.restarting("window")
        r = gui.invoke("--no-model")
        assert r.exit_code == 0, r.output
        assert len(gui.windows) == 1
        assert gui.opened == []
        assert gui.restart_ui == ["window"]

    def test_a_restart_waits_for_its_own_port_to_free(self, gui):
        busy = socket.socket()
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        release = threading.Timer(0.5, busy.close)
        release.start()
        try:
            gui.restarting("browser")
            r = gui.invoke("--no-model", "-p", str(port))
        finally:
            release.cancel()
            busy.close()
        assert r.exit_code == 0, r.output
        assert gui.serve["port"] == port
        assert "already in use" not in r.flat


# ------------------------------------------------------------------ #
#  --pull                                                             #
# ------------------------------------------------------------------ #

class TestPull:
    SPEC = "org/Some-Model-GGUF:some-model.Q4_K_M.gguf"

    def test_pull_on_an_empty_registry_deep_links_a_granted_download(self, gui):
        r = gui.invoke("--pull", self.SPEC)
        assert r.exit_code == 0, r.output
        assert "No models registered yet. Opening the GUI - add one on the Models page (download starting)" in r.flat
        assert gui.engines == []
        q = _query(gui.opened_url())
        assert (q["view"], q["pull"]) == ("models", self.SPEC)
        # The link's token is a live grant for exactly this spec.
        assert gui_web.consume_pull_grant(gui.app, self.SPEC, q["pull_token"])

    def test_pull_does_not_stop_the_startup_model_from_loading(self, gui):
        gui.register_model("usable")
        r = gui.invoke("--pull", self.SPEC)
        assert r.exit_code == 0, r.output
        assert gui.engine.kwargs["display_name"] == "usable"
        assert gui.engine.load_calls == 1
        q = _query(gui.opened_url())
        assert (q["view"], q["pull"]) == ("models", self.SPEC)
        assert gui_web.consume_pull_grant(gui.app, self.SPEC, q["pull_token"])


# ------------------------------------------------------------------ #
#  Tray / status window                                               #
# ------------------------------------------------------------------ #

class TestTray:
    def test_the_tray_starts_on_the_server_address_and_closes_after_serving(self, gui):
        gui.tray = True
        r = gui.invoke("--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        face = gui.face
        port = gui.serve["port"]
        assert face.kwargs["name"] == "LocaLM"
        assert face.kwargs["url"] == f"http://127.0.0.1:{port}/"
        assert face.kwargs["logfile"] == home_dir() / "logs" / "recent.log"
        assert callable(face.kwargs["on_restart"]) and callable(face.kwargs["on_stop"])
        assert face.ready.is_set()
        # Server problems surface in the tray window.
        [(surface, recovered)] = gui.hang_surfaces
        surface("stuck")
        assert face.errors == ["Server problem: stuck"]
        # Open while serving, closed exactly once after.
        assert gui.serve["faces_closed"] == [0]
        assert face.closed == 1

    def test_no_tray_leaves_the_server_running_headless(self, gui):
        r = gui.invoke("--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        assert gui.faces == []
        assert gui.hang_surfaces == []
        assert gui.serve["port"]

    def test_tray_restart_and_stop_target_this_instance(self, gui, monkeypatch):
        gui.tray = True
        calls = []
        monkeypatch.setattr(hs, "_do_restart", lambda **k: calls.append(("restart", k)))
        monkeypatch.setattr(hs, "_do_shutdown", lambda **k: calls.append(("stop", k)))
        seen = {}

        def during_serve(app):
            gui.face.kwargs["on_restart"]()
            gui.face.kwargs["on_stop"]()
            seen["instance_id"] = app.state.instance_id

        gui.during_serve = during_serve
        r = gui.invoke("--no-model", "--no-browser")
        assert r.exit_code == 0, r.output
        iid = seen["instance_id"]
        assert iid
        assert calls == [
            ("restart", {"instance_id": iid, "port": gui.serve["port"]}),
            ("stop", {"instance_id": iid}),
        ]

    def test_the_native_window_quit_action_is_the_tray_stop(self, gui, monkeypatch):
        gui.tray = True
        gui.native = True
        stops = []
        monkeypatch.setattr(hs, "_do_shutdown", lambda **k: stops.append(k))
        r = gui.invoke("--no-model")
        assert r.exit_code == 0, r.output
        gui.windows[0]["on_quit"]()
        gui.face.kwargs["on_stop"]()
        assert len(stops) == 2 and stops[0] == stops[1]
