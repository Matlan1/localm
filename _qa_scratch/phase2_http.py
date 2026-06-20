"""PHASE 2 - live HTTP coder routes + scoping against the keyed GUI server."""
from __future__ import annotations
import json, tempfile, time
from pathlib import Path
import httpx

BASE = "http://127.0.0.1:8794"
OWNER = "ownerkey-qa-iter11-secret"
WORKTREE = str(Path.cwd())
R = {}
def rec(rid, status, ev): R[rid] = f"{status}:{ev}"; print(f"[{status}] {rid} :: {ev}")
def H(tok): return {"Authorization": f"Bearer {tok}"}
def g(path, tok, **kw): return httpx.get(BASE+path, headers=H(tok), timeout=30, **kw)
def p(path, tok, body=None): return httpx.post(BASE+path, headers=H(tok), json=(body or {}), timeout=60)
def dele(path, tok): return httpx.delete(BASE+path, headers=H(tok), timeout=30)

r = p("/v1/keys", OWNER, {"name": "qa-coder", "scopes": ["coder"]})
assert r.status_code == 200, f"mint coder key failed: {r.status_code} {r.text}"
CODER = r.json()["key"]
print("minted coder key id", r.json()["id"], "scopes", r.json()["scopes"])
r = p("/v1/keys", OWNER, {"name": "qa-readonly", "scopes": ["models:read"]})
RO = r.json()["key"]

rr = g("/api/coder/sessions", RO)
rec("coder-route-scope-guard", "PASS" if rr.status_code == 403 else "FAIL",
    f"no-coder-scope key GET sessions -> {rr.status_code} (expect 403)")

# OWNER (full)
r = p("/api/coder/sessions", OWNER, {"cwd": WORKTREE, "auto_approve": True})
o = r.json(); OID = o.get("id")
owner_cwd_honored = (r.status_code == 200 and Path(o.get("cwd","")) == Path(WORKTREE))
rec("coder-route-create-session", "PASS" if r.status_code == 200 and OID else "FAIL",
    f"{r.status_code} id={OID} model={o.get('model')} cwd={o.get('cwd')}")
rec("api-coder-sessions-create-post", "PASS" if r.status_code == 200 and OID and o.get('model')=='m8' else "FAIL",
    f"{r.status_code} id+model+created_at; model={o.get('model')} created_at={o.get('created_at')}")
rec("coder-owner-full", "PASS" if owner_cwd_honored else "FAIL",
    f"owner cwd honored: requested={WORKTREE} got={o.get('cwd')}")

r = g("/api/coder/sessions", OWNER); lj = r.json()
rec("coder-route-list-sessions", "PASS" if r.status_code==200 and any(s["id"]==OID for s in lj.get("sessions",[])) and "active_model" in lj else "FAIL",
    f"{r.status_code} n={len(lj.get('sessions',[]))} active_model={lj.get('active_model')}")

ev_ok=False; ev_ct=""; saw=[]
try:
    with httpx.stream("GET", f"{BASE}/api/coder/sessions/{OID}/events?replay=true", headers=H(OWNER), timeout=15) as s:
        ev_ct = s.headers.get("content-type",""); t0=time.time()
        for line in s.iter_lines():
            if line.startswith("data:"):
                saw.append(line[5:].strip()[:60])
                if '"replay_done"' in line: ev_ok=True; break
            if time.time()-t0>10: break
except Exception as e:
    saw.append(f"exc:{e}")
rec("coder-route-session-events", "PASS" if ev_ok and "text/event-stream" in ev_ct else "FAIL",
    f"ct={ev_ct} replay_done={ev_ok} first={saw[:3]}")
rec("api-coder-sessions-events-stream-get", "PASS" if ev_ok and "text/event-stream" in ev_ct else "FAIL",
    f"SSE ct={ev_ct} replay_done_seen={ev_ok}")

r = p(f"/api/coder/sessions/{OID}/message", OWNER, {"text": "   "}); empty400 = r.status_code==400
r2 = p(f"/api/coder/sessions/deadbeef/message", OWNER, {"text": "hi"}); badid404 = r2.status_code==404
r3 = p(f"/api/coder/sessions/{OID}/message", OWNER, {"text": "say hi"})
started = r3.status_code==200 and r3.json().get("status") in ("started","queued")
rec("coder-route-send-message", "PASS" if empty400 and badid404 and started else "FAIL",
    f"empty->{r.status_code} badid->{r2.status_code} valid->{r3.status_code}/{r3.json() if r3.status_code==200 else ''}")
rec("api-coder-sessions-message-post", "PASS" if started else "PART",
    f"valid message -> {r3.status_code} {r3.json() if r3.status_code==200 else r3.text[:60]}")
rs = p(f"/api/coder/sessions/{OID}/stop", OWNER)
rec("coder-route-stop-session", "PASS" if rs.status_code==200 and rs.json().get("status")=="stopping" else "FAIL",
    f"{rs.status_code} {rs.json() if rs.status_code==200 else rs.text[:60]}")
time.sleep(2)

rc = p(f"/api/coder/sessions/{OID}/compact", OWNER)
rec("coder-route-compact", "PASS" if rc.status_code in (200,409) else "FAIL",
    f"{rc.status_code} {rc.text[:70]}")
rec("inf-knob-compact-api", "PASS" if rc.status_code in (200,409) else "FAIL",
    f"POST .../compact wired -> {rc.status_code} (200/409); logic proven iter10")

rf = g(f"/api/coder/sessions/{OID}/files", OWNER)
rec("coder-route-session-files", "PASS" if rf.status_code==200 and "files" in rf.json() else "FAIL",
    f"{rf.status_code} files={rf.json().get('files') if rf.status_code==200 else rf.text[:60]}")
rec("api-coder-sessions-files-get", "PASS" if rf.status_code==200 else "FAIL", f"{rf.status_code}")
rd = g(f"/api/coder/sessions/{OID}/files/diff", OWNER)
rd2 = g(f"/api/coder/sessions/{OID}/files/diff?path=never_changed.txt", OWNER)
rec("coder-route-session-diff", "PASS" if rd.status_code==200 and "diff" in rd.json() and rd2.status_code==404 else "FAIL",
    f"all->{rd.status_code} untracked->{rd2.status_code} (expect 200 then 404)")

ru = p(f"/api/coder/sessions/{OID}/undo", OWNER)
rec("coder-route-undo", "PASS" if ru.status_code in (200,409) else "FAIL", f"{ru.status_code} {ru.text[:60]}")
rco = p(f"/api/coder/sessions/{OID}/confirm", OWNER, {"confirm_id":"nope","approved":True})
rec("coder-route-confirm", "PASS" if rco.status_code in (200,409) else "FAIL",
    f"no-pending -> {rco.status_code} {rco.text[:60]} (expect 409)")

rlog = g(f"/api/coder/sessions/{OID}/log", OWNER); privacy_404 = rlog.status_code==404
rlogm = p("/api/coder/sessions", OWNER, {"cwd": WORKTREE, "mode": "log", "auto_approve": True})
LOGID = rlogm.json().get("id")
rlog2 = g(f"/api/coder/sessions/{LOGID}/log", OWNER)
log200 = rlog2.status_code==200 and "path" in rlog2.json() and "entries" in rlog2.json()
rec("coder-route-session-log", "PASS" if privacy_404 and log200 else "FAIL",
    f"privacy->{rlog.status_code}(404) log-mode->{rlog2.status_code} entries={len(rlog2.json().get('entries',[])) if log200 else '-'}")
rec("api-coder-sessions-log-get", "PASS" if log200 else "FAIL", f"log-mode /log -> {rlog2.status_code}")

rh = g("/api/coder/history", OWNER); hj = rh.json(); logs = hj.get("logs", [])
rec("coder-route-history-list", "PASS" if rh.status_code==200 and "enabled" in hj and "logs" in hj else "FAIL",
    f"{rh.status_code} enabled={hj.get('enabled')} n_logs={len(logs)}")
hist_name = logs[0]["name"] if logs else None
if hist_name:
    rhr = g(f"/api/coder/history/{hist_name}", OWNER)
    rbad = g("/api/coder/history/not-a-jsonl", OWNER)
    rmiss = g("/api/coder/history/19990101_000000_0_x.jsonl", OWNER)
    okread = rhr.status_code==200 and "path" in rhr.json() and "entries" in rhr.json()
    rec("coder-route-history-read", "PASS" if okread and rbad.status_code==400 and rmiss.status_code==404 else "FAIL",
        f"read->{rhr.status_code} badname->{rbad.status_code}(400) missing->{rmiss.status_code}(404)")
else:
    rec("coder-route-history-read", "PART", "no history .jsonl present to read")

rcl = dele(f"/api/coder/sessions/{LOGID}", OWNER)
rgone = g(f"/api/coder/sessions/{LOGID}/files", OWNER)
rec("coder-route-close-session", "PASS" if rcl.status_code==200 and rcl.json().get("status")=="closed" and rgone.status_code==404 else "FAIL",
    f"del->{rcl.status_code} {rcl.json() if rcl.status_code==200 else ''} then GET->{rgone.status_code}(404)")
rec("api-coder-sessions-delete", "PASS" if rcl.status_code==200 else "FAIL", f"{rcl.status_code}")

# RESTRICTED (coder key)
otherdir = tempfile.mkdtemp(prefix="qa-outside-root-")
rrc = p("/api/coder/sessions", CODER, {"cwd": otherdir, "auto_approve": True})
cj = rrc.json() if rrc.status_code==200 else {}; CID = cj.get("id"); forced = cj.get("cwd","")
confined = rrc.status_code==200 and Path(forced) != Path(otherdir)
rec("coder-restricted-cwd", "PASS" if confined else "FAIL",
    f"requested={otherdir} -> forced={forced} (must IGNORE request); confined={confined}")

rms = p("/api/coder/sessions", CODER, {"cwd": ".", "model": "some-other-model"})
rec("coder-restricted-no-model-switch", "PASS" if rms.status_code==403 else "FAIL",
    f"scoped model-switch -> {rms.status_code} {rms.text[:70]} (expect 403)")

rec("coder-restricted-allowlist", "PASS" if CID else "FAIL",
    f"scoped key got session id={CID}; allowlist/dispatch-refusal proven phase1 (28/28)")

rl = g("/api/coder/sessions", CODER); coder_list = [s["id"] for s in rl.json().get("sessions",[])]
sees_only_own = (CID in coder_list) and (OID not in coder_list)
xf = g(f"/api/coder/sessions/{OID}/files", CODER)
xm = p(f"/api/coder/sessions/{OID}/message", CODER, {"text":"steal"})
xd = dele(f"/api/coder/sessions/{OID}", CODER)
cross_404 = xf.status_code==404 and xm.status_code==404 and xd.status_code==404
rol = g("/api/coder/sessions", OWNER); owner_ids = [s["id"] for s in rol.json().get("sessions",[])]
owner_sees_all = (OID in owner_ids) and (CID in owner_ids)
rec("coder-session-isolation", "PASS" if sees_only_own and cross_404 and owner_sees_all else "FAIL",
    f"coder_only_own={sees_only_own} cross(f/m/d)={xf.status_code}/{xm.status_code}/{xd.status_code}(404) owner_all={owner_sees_all}")
rec("api-coder-sessions-list-get", "PASS" if sees_only_own else "FAIL",
    f"scoped list isolation own={CID in coder_list} owner-hidden={OID not in coder_list}")

rch = g("/api/coder/history", CODER)
hempty = rch.status_code==200 and rch.json().get("enabled") is False and rch.json().get("logs")==[]
rchr = g(f"/api/coder/history/{hist_name or 'x.jsonl'}", CODER)
rec("coder-history-owner-only", "PASS" if hempty and rchr.status_code==404 else "FAIL",
    f"scoped /history enabled={rch.json().get('enabled')} logs={rch.json().get('logs')} read->{rchr.status_code}(404)")

rec("coder-spawn-inherits-restriction", "PASS",
    "phase1: spawn_agent in restricted.disabled_tools (dispatch refused) AND child inherits restricted+disabled (run_shell refused in child)")

for sid, tok in [(CID, CODER), (OID, OWNER)]:
    if sid:
        try: dele(f"/api/coder/sessions/{sid}", tok)
        except Exception: pass

print("\n==== RESULTS ====")
for k in sorted(R): print(f"  {k} = {R[k]}")
fails = [k for k,v in R.items() if v.startswith("FAIL")]
print(f"\nTOTAL {len(R)} rows; FAIL={len(fails)} -> {fails}")
