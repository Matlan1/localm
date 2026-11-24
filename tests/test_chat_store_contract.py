# SPDX-License-Identifier: AGPL-3.0-or-later
"""The conversation store's routes answer the requests of
tests-js/fixtures/chat-store-contract.json as the file says. The GUI tests run the
same file against the stand-in they use for the store."""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.builtin.chat import plug

CONTRACT = json.loads((Path(__file__).resolve().parents[1] / "tests-js" / "fixtures"
                       / "chat-store-contract.json").read_text(encoding="utf-8"))


def _body(step, revs):
    if "picture" in step:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + step["picture"]}}]}]
    else:
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t}
                    for i, t in enumerate(step["texts"])]
    body = {"title": "t", "messages": messages}
    if "base" in step:
        body["base_rev"] = revs[step["base"]] if step["base"] else None
    for key in ("instance", "principal"):
        if key in step:
            body[key] = step[key]
    return body


def test_the_routes_answer_as_the_contract_says(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "memory_principal", lambda request: None)
    app = FastAPI()
    app.include_router(plug._router)
    client = TestClient(app)
    revs = {}
    assert len(CONTRACT["steps"]) > 20
    for n, step in enumerate(CONTRACT["steps"]):
        url = "/api/conversations/" + step["id"]
        if step["do"] == "PUT":
            r = client.put(url, json=_body(step, revs))
        else:
            headers = {}
            if "store" in step:
                here = plug._store_owner(None)
                headers["X-LocaLM-Chat-Store"] = step["store"].format(**here)
            r = client.request(step["do"], url, headers=headers)
        where = f"step {n}: {step}: {r.status_code} {r.text}"
        assert r.status_code == step["status"], where
        if "as" in step:
            revs[step["as"]] = r.json()["rev"]
        if "answer" in step:
            assert r.json()["status"] == step["answer"], where
        if r.status_code == 409:
            detail = r.json()["detail"]
            assert detail["rev"] == (revs[step["rev"]] if step["rev"] else None), where
            assert detail["deleted"] is step["deleted"], where
            assert (detail["row"] is None) == (step["rev"] is None), where
