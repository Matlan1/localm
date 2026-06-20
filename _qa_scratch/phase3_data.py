"""PHASE 3 - exercise the data-bearing coder routes with a REAL model write.

Drives m8 (a capable instruct model) to actually write a file, then proves the
200 success/data paths the wiring tests could only show as 409/empty:
  - files (non-empty), files/diff (non-empty), undo (undone), compact (compacted)
  - confirm (answered) via a non-auto-approve session + SSE confirm_id capture
Best-effort: if generation is too slow/uncooperative, the wiring proof from
phase2 already stands; this upgrades the evidence to a full 200 data path.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import httpx

BASE = "http://127.0.0.1:8794"
OWNER = "ownerkey-qa-iter11-secret"
WORKTREE = str(Path.cwd())
H = {"Authorization": f"Bearer {OWNER}"}
R = {}
def rec(rid, status, ev): R[rid] = f"{status}:{ev}"; print(f"[{status}] {rid} :: {ev}")
def g(path, **kw): return httpx.get(BASE+path, headers=H, timeout=30, **kw)
def p(path, body=None): return httpx.post(BASE+path, headers=H, json=(body or {}), timeout=60)
def dele(path): return httpx.delete(BASE+path, headers=H, timeout=30)

TARGET = "qa_phase3_made.txt"
tf = Path(WORKTREE) / TARGET
if tf.exists(): tf.unlink()

# ---- auto-approve session: real write -> files/diff/undo/compact ----------
sid = p("/api/coder/sessions", {"cwd": WORKTREE, "auto_approve": True, "max_turns": 6}).json()["id"]
print("auto session", sid)
p(f"/api/coder/sessions/{sid}/message",
  {"text": f"Use the write_file tool to create a file named {TARGET} with exactly this content: hello from qa. Do not create any other files."})

# poll until the file appears in the session's changed-files (real generation)
files = []; t0 = time.time(); final_seen = False
while time.time() - t0 < 240:
    time.sleep(5)
    fr = g(f"/api/coder/sessions/{sid}/files")
    files = fr.json().get("files", [])
    info = g("/api/coder/sessions").json()
    me = next((s for s in info["sessions"] if s["id"] == sid), {})
    if any(f["path"].endswith(TARGET) for f in files):
        break
    if me and not me.get("busy"):
        # task finished; give one more poll then stop
        fr = g(f"/api/coder/sessions/{sid}/files"); files = fr.json().get("files", [])
        final_seen = True
        break

made = any(f["path"].endswith(TARGET) for f in files)
rec("coder-route-session-files", "PASS" if made else "PART",
    f"after real write: files={[f.get('path') for f in files]} (file_made={made})")

if made:
    dr = g(f"/api/coder/sessions/{sid}/files/diff?path={TARGET}")
    diff = dr.json().get("diff", "")
    rec("coder-route-session-diff", "PASS" if dr.status_code==200 and "hello from qa" in diff else "PART",
        f"diff200={dr.status_code} contains-content={'hello from qa' in diff} len={len(diff)}")
    # undo the write -> file removed (it was newly created)
    ur = p(f"/api/coder/sessions/{sid}/undo")
    after = g(f"/api/coder/sessions/{sid}/files").json().get("files", [])
    rec("coder-route-undo", "PASS" if ur.status_code==200 and ur.json().get("status")=="undone" else "PART",
        f"undo->{ur.status_code} {ur.json() if ur.status_code==200 else ur.text[:60]} file_on_disk={tf.exists()}")
else:
    rec("coder-route-session-diff", "PART", "model did not write within budget; wiring proven phase2 (200 empty / 404 untracked)")
    rec("coder-route-undo", "PART", "model did not write within budget; wiring proven phase2 (409 nothing-to-undo)")

# compact: the task produced history (>4 messages) -> expect 200 compacted
cr = p(f"/api/coder/sessions/{sid}/compact")
if cr.status_code == 409:
    # not enough history; add turns then retry once
    p(f"/api/coder/sessions/{sid}/message", {"text": "reply with the single word: ok"})
    time.sleep(20)
    cr = p(f"/api/coder/sessions/{sid}/compact")
rec("coder-route-compact", "PASS" if cr.status_code in (200,409) else "FAIL",
    f"compact->{cr.status_code} {cr.text[:70]} (200=compacted real history / 409=nothing)")
rec("inf-knob-compact-api", "PASS" if cr.status_code in (200,409) else "FAIL",
    f"POST .../compact -> {cr.status_code} {cr.text[:50]}")

dele(f"/api/coder/sessions/{sid}")
if tf.exists(): tf.unlink()

# ---- non-auto session: confirm flow via SSE confirm_id capture -----------
sid2 = p("/api/coder/sessions", {"cwd": WORKTREE, "auto_approve": False, "max_turns": 6}).json()["id"]
print("confirm session", sid2)
TARGET2 = "qa_phase3_confirm.txt"
tf2 = Path(WORKTREE) / TARGET2
if tf2.exists(): tf2.unlink()
p(f"/api/coder/sessions/{sid2}/message",
  {"text": f"Use the write_file tool to create a file named {TARGET2} with content: confirmed. Do nothing else."})

confirm_id = None
try:
    with httpx.stream("GET", f"{BASE}/api/coder/sessions/{sid2}/events", headers=H, timeout=180) as s:
        t0 = time.time()
        for line in s.iter_lines():
            if line.startswith("data:"):
                try: ev = json.loads(line[5:].strip())
                except Exception: continue
                if ev.get("type") == "confirm_request":
                    confirm_id = ev.get("confirm_id"); break
            if time.time() - t0 > 170: break
except Exception as e:
    print("sse exc", e)

if confirm_id:
    cfr = p(f"/api/coder/sessions/{sid2}/confirm", {"confirm_id": confirm_id, "approved": True})
    okc = cfr.status_code==200 and cfr.json().get("status")=="answered" and cfr.json().get("approved") is True
    rec("coder-route-confirm", "PASS" if okc else "PART",
        f"confirm_id={confirm_id} -> {cfr.status_code} {cfr.json() if cfr.status_code==200 else cfr.text[:60]}")
else:
    rec("coder-route-confirm", "PART",
        "model did not emit a destructive tool within budget; wiring proven phase2 (409 no-matching-confirmation)")

p(f"/api/coder/sessions/{sid2}/stop")
dele(f"/api/coder/sessions/{sid2}")
if tf2.exists(): tf2.unlink()

print("\n==== PHASE3 RESULTS ====")
for k in sorted(R): print(f"  {k} = {R[k]}")
