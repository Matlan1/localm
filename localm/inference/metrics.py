# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-memory Prometheus metrics for the inference server.

Nothing here is written to disk, and nothing here can carry content: every label
value comes from a closed set (a fixed HTTP method list, a numeric status, a
route TEMPLATE such as ``/v1/models/{model_id}`` and never the requested path)
and every sample value is a number. Prompts, replies, model names and model paths
never reach this module.

Collection is off until :func:`configure` enables it; while off every ``observe_*``
call returns immediately and :func:`render` has nothing to report.
"""

from __future__ import annotations

import re
import threading
from bisect import bisect_left
from typing import Iterable, Optional

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

OTHER_ROUTE = "other"

_KNOWN_METHODS = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})

_ROUTE_TEMPLATE_RE = re.compile(r"^[A-Za-z0-9/_{}.:\-]{1,120}$")

REQUEST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
                   10.0, 30.0, 60.0, 120.0, 300.0)
TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
TOKENS_PER_SECOND_BUCKETS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0,
                             100.0, 150.0, 200.0, 300.0, 500.0, 1000.0)

_HELP = {
    "localm_http_requests_total":
        ("counter", "HTTP requests served, by method, route template and status."),
    "localm_http_request_duration_seconds":
        ("histogram", "HTTP request duration in seconds, by method and route template."),
    "localm_http_requests_in_flight":
        ("gauge", "HTTP requests currently being served."),
    "localm_prompt_tokens_total":
        ("counter", "Prompt tokens processed."),
    "localm_generated_tokens_total":
        ("counter", "Completion tokens generated."),
    "localm_time_to_first_token_seconds":
        ("histogram", "Seconds from the start of generation to the first token."),
    "localm_tokens_per_second":
        ("histogram", "Decode throughput per generation, in tokens per second."),
    "localm_inference_queue_depth":
        ("gauge", "Requests waiting for a model's inference slot."),
    "localm_models_loaded":
        ("gauge", "Models currently loaded."),
    "localm_vram_used_bytes":
        ("gauge", "GPU memory in use, summed over the monitored GPUs."),
    "localm_vram_total_bytes":
        ("gauge", "GPU memory capacity, summed over the monitored GPUs."),
}


class _Histogram:
    __slots__ = ("bounds", "buckets", "total", "count")

    def __init__(self, bounds: tuple) -> None:
        self.bounds = bounds
        self.buckets = [0] * len(bounds)
        self.total = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        idx = bisect_left(self.bounds, value)
        if idx < len(self.buckets):
            self.buckets[idx] += 1
        self.total += value
        self.count += 1


_lock = threading.Lock()
_enabled = False
_counters: dict = {}
_histograms: dict = {}
_in_flight = 0


def configure(enabled: bool) -> None:
    """Switch collection on or off and drop everything collected so far."""
    global _enabled, _in_flight
    with _lock:
        _enabled = bool(enabled)
        _counters.clear()
        _histograms.clear()
        _in_flight = 0


def is_enabled() -> bool:
    return _enabled


def method_label(method: object) -> str:
    """The request method when it is one of the standard seven, else ``OTHER``."""
    return method if isinstance(method, str) and method in _KNOWN_METHODS else "OTHER"


def route_label(route: object) -> str:
    """The route's path TEMPLATE, or ``other`` for anything that is not a plain
    route (a mount, an unmatched path) or whose template is not a short run of
    URL-template characters. The requested path itself is never a label."""
    template = getattr(route, "path", None)
    if not isinstance(template, str) or not _ROUTE_TEMPLATE_RE.match(template):
        return OTHER_ROUTE
    return template


def _add(name: str, labels: tuple, amount: float) -> None:
    key = (name, labels)
    _counters[key] = _counters.get(key, 0.0) + amount


def _observe(name: str, labels: tuple, bounds: tuple, value: float) -> None:
    hist = _histograms.get((name, labels))
    if hist is None:
        hist = _histograms[(name, labels)] = _Histogram(bounds)
    hist.observe(value)


def request_started() -> None:
    global _in_flight
    if not _enabled:
        return
    with _lock:
        _in_flight += 1


def request_finished(method: object, route: str, status: int,
                     seconds: float) -> None:
    """Record a completed HTTP request. *route* must come from
    :func:`route_label`."""
    global _in_flight
    if not _enabled:
        return
    method = method_label(method)
    status = status if isinstance(status, int) and 100 <= status <= 599 else 500
    with _lock:
        _in_flight = max(0, _in_flight - 1)
        _add("localm_http_requests_total",
             (("method", method), ("route", route), ("status", str(status))), 1)
        _observe("localm_http_request_duration_seconds",
                 (("method", method), ("route", route)),
                 REQUEST_BUCKETS, max(0.0, seconds))


def observe_generation(prompt_tokens: Optional[int],
                       completion_tokens: Optional[int],
                       ttft_ms: Optional[float],
                       tokens_per_sec: Optional[float]) -> None:
    """Record one finished generation. A figure that could not be measured is
    passed as ``None`` and simply not recorded."""
    if not _enabled:
        return
    with _lock:
        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
            _add("localm_prompt_tokens_total", (), prompt_tokens)
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            _add("localm_generated_tokens_total", (), completion_tokens)
        if isinstance(ttft_ms, (int, float)) and ttft_ms >= 0:
            _observe("localm_time_to_first_token_seconds", (),
                     TTFT_BUCKETS, ttft_ms / 1000.0)
        if isinstance(tokens_per_sec, (int, float)) and tokens_per_sec > 0:
            _observe("localm_tokens_per_second", (),
                     TOKENS_PER_SECOND_BUCKETS, float(tokens_per_sec))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt_labels(labels: Iterable) -> str:
    items = [f'{k}="{_escape(v)}"' for k, v in labels]
    return "{" + ",".join(items) + "}" if items else ""


def _fmt_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _header(name: str) -> list:
    kind, help_text = _HELP[name]
    return [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]


def render(gauges: Iterable = ()) -> str:
    """The Prometheus text exposition of everything collected, plus *gauges*:
    ``(name, value)`` pairs read at scrape time, where *name* is a key of the
    metric table and a ``None`` value leaves that metric out."""
    lines: list = []
    with _lock:
        counters = dict(_counters)
        histograms = {k: (v.bounds, list(v.buckets), v.total, v.count)
                      for k, v in _histograms.items()}
        in_flight = _in_flight
    seen: set = set()
    for (name, labels), value in sorted(counters.items()):
        if name not in seen:
            seen.add(name)
            lines += _header(name)
        lines.append(f"{name}{_fmt_labels(labels)} {_fmt_number(value)}")
    for (name, labels), (bounds, buckets, total, count) in sorted(histograms.items()):
        if name not in seen:
            seen.add(name)
            lines += _header(name)
        running = 0
        for bound, n in zip(bounds, buckets, strict=True):
            running += n
            lines.append(f"{name}_bucket"
                         f"{_fmt_labels(labels + (('le', _fmt_number(bound)),))} {running}")
        lines.append(f"{name}_bucket{_fmt_labels(labels + (('le', '+Inf'),))} {count}")
        lines.append(f"{name}_sum{_fmt_labels(labels)} {_fmt_number(total)}")
        lines.append(f"{name}_count{_fmt_labels(labels)} {count}")
    for name, value in (("localm_http_requests_in_flight", in_flight), *gauges):
        if value is None:
            continue
        lines += _header(name)
        lines.append(f"{name} {_fmt_number(value)}")
    return "\n".join(lines) + "\n"
