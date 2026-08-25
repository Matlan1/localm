# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI form of `localm ps` / `localm stop <id>` (PARITY-AUDIT-CLI-GUI-2026-08-19.md, CLI-only gap #7)."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from localm import scopes
from localm.inference.http_server import require_scope

# The fields a listing may show a client. Whitelisted explicitly rather than
# passing instances.snapshot()'s row through as-is: that row carries `_path`
# (this machine's filesystem path to the OTHER instance's registry file),
# which is server-internal and was never meant to reach a network client - a
# network-bound instance can serve a phone/companion client this data has no
# business going to. `token` is already stripped by snapshot()'s own default.
_LIST_FIELDS = ("instance_id", "alive", "root_dir", "mode", "scheme", "host",
                "port", "pid", "started", "version")


def register(app: FastAPI, ctx) -> None:

    @app.get("/api/instances", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    def instances_list_ep():
        """Every registered localm instance on this machine - the GUI form of `localm ps`."""
        from localm import instances
        from localm.bindhost import url_host
        from localm.config import home_dir
        self_id = getattr(app.state, "instance_id", None)
        rows = instances.snapshot(home_dir())
        out = []
        for r in rows:
            row = {k: r.get(k) for k in _LIST_FIELDS}
            row["self"] = row["instance_id"] == self_id
            # Pre-bracketed display address (the BIND host, same choice
            # cli/models.py's ps_cmd makes and the same reason: this answers
            # "what did this instance bind", not "how would I connect to it" -
            # url_host() lives once so an IPv6 bind never grows a second,
            # divergent bracketing implementation on the GUI side.
            row["address"] = (f"{row.get('scheme') or 'http'}://"
                              f"{url_host(row.get('host') or '127.0.0.1')}"
                              f":{row.get('port', '?')}")
            out.append(row)
        return {"instances": out}

    @app.post("/api/instances/{instance_id}/stop",
              dependencies=[Depends(require_scope(scopes.ADMIN))])
    def instance_stop_ep(instance_id: str):
        """Stop ONE running instance (matched by id or id prefix, same as `localm stop <id>`)."""
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
                # misconfiguration (see cli/models.py's stop_cmd, identical
                # comment). Fall back to a direct kill rather than failing.
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
                # which normally clears its crash marker - without this, the
                # next start of THAT instance would misreport this intentional
                # stop as a crash. Same as cli/models.py's stop_cmd.
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
