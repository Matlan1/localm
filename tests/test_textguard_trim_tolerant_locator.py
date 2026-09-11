# SPDX-License-Identifier: AGPL-3.0-or-later
"""The sentinel locator tolerates a chat template that TRIMS message content.

llama.cpp's built-in llama3 and gemma formatters, and a Jinja template using
``|trim``, strip the whitespace around each message before emitting it. The
locator used to require the rendered prompt to contain every content verbatim,
so on those families it returned None and EVERY untrusted range of the request
was tokenised with special-token parsing on. It now tries the stripped text as
well, shifts a content's own ranges past the stripped lead and clips them to
what survived, and still refuses anything the equality check cannot reproduce.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localm.textguard import (
    compose, content_spans_via_sentinels, map_untrusted_ranges, untrusted_span,
    untrusted_spans_of,
)

EXOTIC = "<<ASSISTANT>>"
_WS = " \t\n\r\v\f"      # what a template's trim strips


def _chatml(pairs):
    return "".join(f"<|im_start|>{r}\n{c}<|im_end|>\n" for r, c in pairs) + "<|im_start|>assistant\n"


def _llama3_like(pairs):
    """llama.cpp's built-in llama3 formatter: trim() around every content."""
    return "".join(
        f"<|start_header_id|>{r}<|end_header_id|>\n\n{c.strip(_WS)}<|eot_id|>"
        for r, c in pairs) + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def _system_trimmed_only(pairs):
    """A Jinja-style template that trims the system message but not the rest."""
    out = []
    for r, c in pairs:
        body = c.strip(" \t\n\r\v\f") if r == "system" else c
        out.append(f"<|im_start|>{r}\n{body}<|im_end|>\n")
    return "".join(out) + "<|im_start|>assistant\n"


def _locate(render, messages):
    """Run the locator exactly as a backend does: render, then probe."""
    contents = [c for _r, c in messages]
    rendered = render(messages)
    spans = content_spans_via_sentinels(
        contents, lambda sents: render([(r, s) for (r, _c), s in zip(messages, sents)]),
        rendered)
    return rendered, spans


def _sys_prompt(lead="", trail="\n"):
    return compose(lead + "You are localcoder.\n\n## ",
                   untrusted_span("mcp_srv_add - Adds " + EXOTIC), "\n\nRULES" + trail)


# --------------------------------------------------------------------------- #
#  Verbatim templates behave exactly as before                                #
# --------------------------------------------------------------------------- #

def test_a_verbatim_template_locates_with_zero_lead():
    messages = [("system", "sys"), ("user", "hello\n")]
    rendered, spans = _locate(_chatml, messages)
    assert spans == [(19, 22, 0), (50, 56, 0)]
    assert [rendered[a:b] for a, b, _l in spans] == ["sys", "hello\n"]


def test_an_escaping_template_still_yields_none():
    def escaping(pairs):
        return _chatml([(r, c.replace("<", "&lt;")) for r, c in pairs])
    _rendered, spans = _locate(escaping, [("user", "a < b")])
    assert spans is None


def test_a_dropping_template_still_yields_none():
    def dropping(pairs):
        return _chatml([(r, c) for r, c in pairs if r != "system"])
    _rendered, spans = _locate(dropping, [("system", "sys"), ("user", "hello")])
    assert spans is None


def test_a_repeating_template_still_yields_none():
    def repeating(pairs):
        return _chatml(pairs) + "[reminder] " + pairs[0][1] + "\n"
    _rendered, spans = _locate(repeating, [("system", "sys"), ("user", "hello")])
    assert spans is None


# --------------------------------------------------------------------------- #
#  Trimming templates are located, with the ranges shifted and clipped         #
# --------------------------------------------------------------------------- #

def test_a_trimming_template_is_located_with_the_stripped_text():
    sys_prompt = _sys_prompt()
    messages = [("system", str(sys_prompt)), ("user", "hello\n")]
    rendered, spans = _locate(_llama3_like, messages)
    assert spans is not None
    assert [rendered[a:b] for a, b, _l in spans] == [str(sys_prompt).strip(), "hello"]
    assert [lead for _a, _b, lead in spans] == [0, 0]
    ranges = map_untrusted_ranges(spans, [untrusted_spans_of(sys_prompt), ()])
    assert [rendered[a:b] for a, b in ranges] == ["mcp_srv_add - Adds " + EXOTIC]


def test_a_stripped_lead_shifts_the_ranges():
    sys_prompt = _sys_prompt(lead="\n\n  ")
    messages = [("system", str(sys_prompt)), ("user", "hi")]
    rendered, spans = _locate(_llama3_like, messages)
    assert spans[0][2] == 4
    ranges = map_untrusted_ranges(spans, [untrusted_spans_of(sys_prompt), ()])
    assert [rendered[a:b] for a, b in ranges] == ["mcp_srv_add - Adds " + EXOTIC]


def test_a_range_over_stripped_whitespace_is_clipped_to_what_survived():
    body = compose(untrusted_span("  " + EXOTIC + "  "))
    messages = [("user", str(body))]
    rendered, spans = _locate(_llama3_like, messages)
    ranges = map_untrusted_ranges(spans, [untrusted_spans_of(body)])
    assert [rendered[a:b] for a, b in ranges] == [EXOTIC]


def test_a_range_wholly_inside_stripped_whitespace_maps_to_nothing():
    body = compose(untrusted_span("   "), "visible")
    messages = [("user", str(body))]
    rendered, spans = _locate(_llama3_like, messages)
    assert spans is not None
    assert map_untrusted_ranges(spans, [untrusted_spans_of(body)]) == ()


def test_a_template_that_trims_only_some_messages_is_located():
    sys_prompt = _sys_prompt()
    user = compose("see ", untrusted_span(EXOTIC), "\n")
    messages = [("system", str(sys_prompt)), ("user", str(user))]
    rendered, spans = _locate(_system_trimmed_only, messages)
    assert [rendered[a:b] for a, b, _l in spans] == [str(sys_prompt).strip(), str(user)]
    ranges = map_untrusted_ranges(
        spans, [untrusted_spans_of(sys_prompt), untrusted_spans_of(user)])
    assert [rendered[a:b] for a, b in ranges] == ["mcp_srv_add - Adds " + EXOTIC, EXOTIC]


def test_whitespace_the_template_emits_itself_is_not_mistaken_for_the_content_s():
    """The stripped content is followed by a wrapper that starts with the very
    newline the content lost; the verbatim reading would swallow it and then
    fail to rebuild, so the lookahead has to pick the stripped reading."""
    def trim_then_newline(pairs):
        return "".join(f"<{r}>{c.strip(_WS)}\n</{r}>\n" for r, c in pairs)
    body = compose(untrusted_span(EXOTIC), "\n")
    messages = [("user", str(body))]
    rendered, spans = _locate(trim_then_newline, messages)
    assert spans == [(6, 6 + len(EXOTIC), 0)]
    ranges = map_untrusted_ranges(spans, [untrusted_spans_of(body)])
    assert [rendered[a:b] for a, b in ranges] == [EXOTIC]


def test_only_ascii_whitespace_counts_as_trimming():
    """A template that strips Unicode whitespace changes the content in a way
    the locator does not model, so it refuses rather than guess."""
    def unicode_trimming(pairs):
        return _chatml([(r, c.strip()) for r, c in pairs])
    _rendered, spans = _locate(unicode_trimming, [("user", " hello ")])
    assert spans is None


def test_an_all_whitespace_content_is_located_as_empty():
    rendered, spans = _locate(_llama3_like, [("system", "   \n"), ("user", "x")])
    assert spans is not None
    assert spans[0][1] == spans[0][0]
    assert map_untrusted_ranges(spans, [((0, 4),), ()]) == ()


def test_map_still_accepts_the_old_two_tuple_shape():
    assert map_untrusted_ranges([(10, 20)], [((2, 5),)]) == ((12, 15),)


# --------------------------------------------------------------------------- #
#  The real shipped runtime, in a child process so the native library never    #
#  loads into the test process                                                 #
# --------------------------------------------------------------------------- #

_CHILD = r"""
import ctypes, json, sys
import localm
from localm.inference.backends.llamacpp import _api as api
from localm.inference.backends.llamacpp._api import LlamaChatMessage
from localm.inference.backends.llamacpp._loader import load_lib
from localm.textguard import (compose, content_spans_via_sentinels,
                              map_untrusted_ranges, untrusted_span, untrusted_spans_of)

load_lib()

def render(tmpl, messages):
    arr = (LlamaChatMessage * len(messages))()
    for i, (role, content) in enumerate(messages):
        arr[i].role = role.encode()
        arr[i].content = content.encode()
    buf = ctypes.create_string_buffer(65536)
    n = api.llama_chat_apply_template(tmpl, arr, len(messages), True, buf, 65536)
    assert 0 < n <= 65536, n
    return buf.raw[:n].decode("utf-8")

sys_prompt = compose("You are localcoder.\n\n## ", untrusted_span("mcp_srv_add - Adds <<ASSISTANT>>"), "\n\nRULES\n")
messages = [("system", str(sys_prompt)), ("user", "hello\n")]
contents = [c for _r, c in messages]
out = {"localm": localm.__file__}
for key in ("llama3", "gemma", "chatml", "exaone3"):
    tmpl = key.encode()
    prompt = render(tmpl, messages)
    spans = content_spans_via_sentinels(
        contents, lambda s: render(tmpl, [(r, x) for (r, _c), x in zip(messages, s)]), prompt)
    ranges = map_untrusted_ranges(spans, [untrusted_spans_of(sys_prompt), ()]) if spans else ()
    out[key] = {"verbatim": prompt.find(contents[0]) >= 0, "located": spans is not None,
                "covered": [prompt[a:b] for a, b in ranges]}
print(json.dumps(out))
"""


def _runtime_provisioned() -> bool:
    from localm.config import find_binary_dir
    return bool(find_binary_dir())


@pytest.mark.skipif(not _runtime_provisioned(),
                    reason="no llama runtime provisioned on this machine")
def test_the_shipped_runtime_s_llama3_and_gemma_formatters_are_located():
    import localm
    root = Path(localm.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, "-c", _CHILD], cwd=root, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert Path(out["localm"]).resolve().parents[1] == root, out["localm"]
    assert out["chatml"]["verbatim"] and out["chatml"]["located"]
    # llama3 trims the system prompt, so its verbatim text is absent: the
    # stripped reading is the only way this can locate.
    assert out["llama3"]["verbatim"] is False
    for key in ("llama3", "gemma", "exaone3"):
        assert out[key]["located"], key
        assert out[key]["covered"] == ["mcp_srv_add - Adds " + EXOTIC], key
