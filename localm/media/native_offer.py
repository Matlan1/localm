# SPDX-License-Identifier: AGPL-3.0-or-later
"""The structured form of "this native media model is not downloaded yet".

The GUI shows its download box from these entries; the CLI and the API keep the
text message that names the command to run. An entry is
``{"filename", "native": True, "searchable": False, "source": {...}}`` with
``source`` holding ``repo``, ``file``, ``spec``, ``sha256``, ``name``,
``size_bytes``, ``model_type`` and ``origin``. An entry carries ``blocked``
(``"net_mode_off"``) when the network policy refuses model downloads, so the
GUI says why instead of offering a download that fails.

A job that stops because a model is missing pushes one ``MISSING_MODEL_EVENT``
event before its text line; ``text`` repeats that line so a GUI that offers the
download can leave the command out of its log.
"""
from __future__ import annotations

from typing import Optional

MISSING_MODEL_EVENT = "missing_model"
BLOCKED_NET_MODE_OFF = "net_mode_off"


def blocked_reason() -> Optional[str]:
    """``BLOCKED_NET_MODE_OFF`` when the network policy refuses an explicit
    model download right now, else None. Reads the policy and config."""
    from localm.netpolicy import downloads_allowed_when_off, network_mode
    if network_mode() == "off" and not downloads_allowed_when_off():
        return BLOCKED_NET_MODE_OFF
    return None


def _entry(filename: str, source: dict, slot: Optional[str] = None) -> dict:
    out = {"filename": filename, "native": True, "searchable": False,
           "source": {**source, "origin": "curated"}}
    if slot:
        out["slot"] = slot
    return out


def _mark_blocked(entries: list) -> list:
    reason = blocked_reason() if entries else None
    if reason:
        for e in entries:
            e["blocked"] = reason
    return entries


def from_pairing(missing: list) -> list:
    """The entries to offer for the companion slots *missing* of a model
    (``pairing.Pairing.missing``): one per slot that has a known download."""
    return _mark_blocked([
        _entry(m["download"]["filename"],
               {**m["download"], "name": m["download"].get("name") or None}, m["slot"])
        for m in missing if m.get("download")])


def from_status(s: dict, st: dict) -> dict:
    """The preflight answer for the native image or video backend with
    settings *s* and ``native.status(s)`` = *st*: ``{"status", "warning",
    "missing"}``. With no model set, ``missing`` lists every recommended file
    not yet downloaded; with a model set, the companion files it lacks that
    have a known download."""
    if not st["missing"]:
        return {"status": "verified", "missing": [], "warning": None}
    all_parts = st.get("missing_components") or []
    parts = [p for p in all_parts if p.get("download")]
    if parts:
        return {"status": "verified",
                "warning": st["missing"] if len(parts) < len(all_parts) else None,
                "missing": from_pairing(parts)}
    if (s.get("native") or {}).get("model"):
        return {"status": "verified", "missing": [], "warning": st["missing"]}
    rec = st["recommended"]
    files = rec["parts"] if "parts" in rec else [dict(rec, filename=rec["file"])]
    return {"status": "verified", "warning": None, "missing": _mark_blocked([
        _entry(f["filename"], {
            "repo": f["repo"], "file": f["file"], "spec": f["spec"], "sha256": f["sha256"],
            "name": f.get("name"), "size_bytes": f["size_bytes"],
            "model_type": f["model_type"]})
        for f in files])}


def from_music_status(s: dict, st: dict) -> dict:
    """The preflight answer for the native music backend: each default model
    not downloaded yet, offered for download."""
    from localm.media.koboldcpp.models import LABELS
    configured = s.get("native") or {}
    unresolved = [str(LABELS.get(c, c)) for c, v in st["models"].items()
                  if v is None and configured.get(c)]
    warning = (f"Configured music models not found: {', '.join(unresolved)}."
               if unresolved else None)
    return {"status": "verified", "warning": warning, "missing": _mark_blocked([
        _entry(m["file"], {
            "repo": m["repo"], "file": m["file"], "spec": m["spec"], "sha256": m["sha256"],
            "name": m["name"], "size_bytes": m["size_bytes"], "model_type": m["model_type"]})
        for m in st["missing"]])}


def vision_encoder() -> list:
    """The one-entry list that offers the Qwen3-VL vision encoder reference
    images need (``pairing.QWEN3_VL_8B_MMPROJ``)."""
    from localm.media.sdcpp import pairing
    dl = pairing.QWEN3_VL_8B_MMPROJ.as_dict()
    return _mark_blocked([_entry(dl["filename"], {
        "repo": dl["repo"], "file": dl["file"], "spec": dl["spec"], "sha256": dl["sha256"],
        "name": dl["name"] or None, "size_bytes": dl["size_bytes"],
        "model_type": dl["model_type"]})])


def job_missing(kind: str, native, s: dict) -> list:
    """The entries to offer for the native backend module *native* of media
    *kind* ("image", "video" or "music") with settings *s*; [] when nothing
    can be offered or the status cannot be read. Never raises."""
    try:
        st = native.status(s)
        answer = from_music_status(s, st) if kind == "music" else from_status(s, st)
    except Exception:  # noqa: BLE001
        from localm.debuglog import logger
        logger.debug("native %s missing-model lookup failed", kind, exc_info=True)
        return []
    return answer["missing"]


def push_missing(job, missing: list, text: str) -> None:
    """Push the ``MISSING_MODEL_EVENT`` for *missing* onto *job* ahead of its
    text line *text*. Does nothing when *missing* is empty."""
    if missing:
        job.push({"type": MISSING_MODEL_EVENT, "missing": missing, "text": text})
