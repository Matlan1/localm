# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI form of `localm ps` / `localm stop <id>`.

  GET  /api/instances              - every live instance on this machine, this
                                      one included (`self: true` on it) and
                                      instances of other installs flagged
                                      `same_install: false`, read-only.
  POST /api/instances/{id}/stop    - stop ONE OTHER instance OF THIS INSTALL by
                                      id or id prefix.

Listing spans installs; stopping does not. An instance of another install is
found through the machine-wide coordination registry, which records no attach
token for it, so the graceful `POST /v1/server/shutdown` below has no credential
to present and only a hard OS kill would remain. A hard kill skips that
instance's model unload and leaves ITS crash marker armed (this process can only
disarm markers under its OWN data dir), so the next start of that install would
report a crash that did not happen. Those rows carry no Stop button.

The stop route is gated on scopes.ADMIN (the owner), not CONFIG_WRITE like the
sibling /v1/server/shutdown|restart (localm/inference/routes/admin.py): stopping
a DIFFERENT instance reaches outside the calling instance's own blast radius,
into a process that may be serving an unrelated project or user. Open mode (no
key configured anywhere) still passes through unchanged, same as every other
owner-gated route.

Both routes are plain `def`, not `async def`: listing probes each entry over
loopback HTTP (default_probe, up to ~0.7s per entry) and stopping polls PID
liveness with blocking sleeps - Starlette threadpools a sync handler, so
neither blocks the event loop.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from localm import scopes
from localm.inference.http_server import require_scope

# The fields a listing may show a client. Whitelisted explicitly: instances
# .snapshot()'s row also carries `_path` (this machine's filesystem path to the
# OTHER instance's registry file), which is server-internal and must not reach a
# network client. `token` is already stripped by snapshot()'s own default.
_LIST_FIELDS = ("instance_id", "alive", "root_dir", "mode", "scheme", "host",
                "port", "pid", "started", "version", "same_install")


def register(app: FastAPI, ctx) -> None:

    @app.get("/api/instances", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    def instances_list_ep():
        """Every live localm instance on this machine - the GUI form of
        `localm ps`, widened past it.

        Two sources: this install's own registry (`snapshot`, which reaps dead
        entries first, same as the CLI) and the machine-wide coordination
        registry for instances of OTHER installs, which have their own
        LOCALM_HOME and appear in no registry this one can read. The latter
        carry `same_install: false` and are listed only on a verified /whoami
        handshake. The instance serving THIS request carries `self: true`.

        The second source is read ONLY by a server that advertises itself into
        that same registry - an `--isolated` run or a bare test app registers
        nowhere (http_server.py gates its coordination entry on exactly this
        condition), and reading a registry it does not write would let a run
        documented as invisible to discovery both see and probe every other
        localm on the box."""
        from localm import instances
        from localm.bindhost import url_host
        from localm.config import home_dir
        self_id = getattr(app.state, "instance_id", None)
        home = home_dir()
        rows = [dict(r, same_install=True) for r in instances.snapshot(home)]
        if self_id and not getattr(app.state, "instance_isolated", False):
            rows += instances.list_machine_peers(home)
        out = []
        for r in rows:
            row = {k: r.get(k) for k in _LIST_FIELDS}
            row["self"] = row["instance_id"] == self_id
            # Pre-bracketed display address (the BIND host, the same choice
            # cli/models.py's ps_cmd makes): this answers "what did this
            # instance bind", not "how would I connect to it". Bracketing an
            # IPv6 bind lives once, in url_host().
            row["address"] = (f"{row.get('scheme') or 'http'}://"
                              f"{url_host(row.get('host') or '127.0.0.1')}"
                              f":{row.get('port', '?')}")
            out.append(row)
        return {"instances": out}

    @app.post("/api/instances/{instance_id}/stop",
              dependencies=[Depends(require_scope(scopes.ADMIN))])
    def instance_stop_ep(instance_id: str):
        """Stop ONE running instance OF THIS INSTALL (matched by id or id
        prefix, same as `localm stop <id>`); an id belonging to another
        install's instance is refused with 409. Mirrors cli/models.py's stop_cmd for a single
        target: a graceful `POST /v1/server/shutdown` using the target's own
        attach token (selfclient.self_request, the same hoisted helper the
        CLI/MCP already share), falling back to instances.kill_pid - a direct
        OS terminate/kill - if the target declines, is unreachable, or does
        not confirm within the timeout. No --all equivalent: a GUI row acts on
        one instance at a time."""
        import time

        import requests

        from localm import instances
        from localm.bindhost import self_connect_host, url_host
        from localm.config import home_dir
        from localm.selfclient import self_request

        timeout = 10.0
        home = home_dir()
        instances.reap_stale(home)
        matches = [e for e in instances.list_entries(home)
                   if str(e.get("instance_id", "")).startswith(instance_id)]
        if not matches:
            # Same gate as the listing route: a server that registers in no
            # machine-wide registry does not read one either.
            peers = (instances.list_machine_peers(home)
                     if getattr(app.state, "instance_id", None)
                     and not getattr(app.state, "instance_isolated", False)
                     else [])
            if any(str(p.get("instance_id", "")).startswith(instance_id)
                   for p in peers):
                raise HTTPException(
                    409,
                    f"Instance {instance_id!r} belongs to a different localm "
                    "install, which keeps its own data directory. This server "
                    "holds no shutdown credential for it and will not kill it "
                    "outright, because that would skip its model unload and "
                    "leave it reporting a crash on next start. Stop it from "
                    "that install, or from the terminal it was started in.")
            raise HTTPException(404, f"No running instance matches {instance_id!r}.")
        if len(matches) > 1:
            candidates = ", ".join(str(e.get("instance_id", ""))[:8] for e in matches)
            raise HTTPException(
                400, f"{instance_id!r} matches {len(matches)} instances "
                     f"({candidates}) - use a longer id.")
        entry = matches[0]
        pid = entry.get("pid")
        scheme = entry.get("scheme", "http")
        host = url_host(self_connect_host(entry.get("host")))
        base_url = f"{scheme}://{host}:{entry.get('port')}/v1"

        stopped = False
        graceful_denied = False
        try:
            resp = self_request("POST", "/server/shutdown", base_url=base_url,
                                timeout=5, instance_token=entry.get("token"))
            if resp.status_code in (401, 403):
                # The target's open-mode management gate refuses an
                # unauthenticated shutdown from a caller with no shell/API-key
                # credential for THAT instance - the default case, not a
                # misconfiguration. Falls back to a direct kill.
                graceful_denied = True
            elif resp.status_code == 200:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline and not stopped:
                    if not instances.pid_alive(int(pid or -1)):
                        stopped = True
                    else:
                        time.sleep(0.25)
        except requests.RequestException:
            pass  # target unreachable (hung / already gone) - fall through to a direct kill

        if not stopped:
            stopped = instances.kill_pid(int(pid or -1), timeout=timeout)
            if stopped:
                # A direct kill bypasses the target's own clean-shutdown path,
                # which normally clears its crash marker, so the next start of
                # THAT instance would otherwise report this stop as a crash.
                try:
                    from localm import bugreport
                    bugreport.disarm_crash_guard(
                        home, instance_id=entry.get("instance_id"))
                except Exception:
                    pass

        path = entry.get("_path")
        if path:
            instances.unregister_instance(path)

        if not stopped:
            raise HTTPException(
                502, f"Could not confirm instance {instance_id!r} stopped "
                     f"(pid {pid}).")
        return {"status": "stopped", "instance_id": entry.get("instance_id"),
                "root_dir": entry.get("root_dir"),
                "graceful_denied": graceful_denied}
