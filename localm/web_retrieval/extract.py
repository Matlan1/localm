# SPDX-License-Identifier: AGPL-3.0-or-later
"""Main-content extraction from an HTML page.

``extract_page`` parses the document into a small element tree with the
standard library parser, removes non-content (scripts, styles, hidden elements,
elements with navigation, banner, complementary, contentinfo, search, dialog,
menu or toolbar roles), then selects the content region:

1. the largest ``<main>`` (or ``role="main"``) element;
2. otherwise the largest ``<article>``;
3. otherwise the whole body.

Inside a ``main``/``article`` region, ``nav``, ``aside``, ``form`` and
``footer`` elements are removed and ``header`` is kept. In the body fallback,
``header`` is removed as well and link-dense blocks (three or more links whose
text is at least 65 percent of the block's text) are removed bottom-up. A
region under 200 characters falls back to the body path; a body path that
removes everything falls back to the body with only scripts, styles, hidden
elements and chrome tags removed.
"""

from __future__ import annotations

import html.parser
import re
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

_SKIP_TAGS = frozenset({
    "script", "style", "head", "noscript", "svg", "template", "iframe",
    "object", "embed", "canvas", "video", "audio", "map",
})
_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
})
#: Removed inside a main/article region and in the body fallback.
_CHROME_TAGS = frozenset({"nav", "aside", "form", "footer", "menu", "dialog"})
#: Removed in the body fallback only.
_BODY_ONLY_CHROME_TAGS = frozenset({"header"})
_CHROME_ROLES = frozenset({
    "navigation", "banner", "contentinfo", "complementary", "search", "menu",
    "menubar", "toolbar", "dialog", "alertdialog", "tooltip",
})
#: Tags whose content is followed by a paragraph break.
_PARAGRAPH_TAGS = frozenset({
    "p", "div", "section", "article", "main", "header", "footer", "aside",
    "nav", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "ul",
    "ol", "dl", "table", "figure", "figcaption", "details", "summary",
    "address", "hr", "body", "html", "form", "fieldset",
})
#: Tags whose content is followed by a line break.
_LINE_TAGS = frozenset({"li", "tr", "dd", "dt", "caption", "option"})
_CELL_TAGS = frozenset({"td", "th"})
_LINK_BLOCK_TAGS = frozenset({"ul", "ol", "dl", "div", "section", "table",
                              "menu"})
_HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")

_MIN_REGION_CHARS = 200
#: Open elements beyond this depth are not pushed; their content attaches to
#: the deepest open element.
_MAX_DEPTH = 200
_LINK_DENSITY_THRESHOLD = 0.65
_LINK_DENSITY_MIN_ANCHORS = 3
_TITLE_CAP = 300

_HIDDEN_STYLE_RE = re.compile(
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\f\v ]+")
_SPACE_AROUND_NEWLINE_RE = re.compile(r" *\n *")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


class _Node:
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag: str, attrs: dict, parent: "Optional[_Node]"):
        self.tag = tag
        self.attrs = attrs
        self.children: list = []
        self.parent = parent


class _TreeBuilder(html.parser.HTMLParser):
    """Build a ``_Node`` tree. An end tag closes the nearest matching open
    element (and everything opened after it); an unmatched end tag is
    ignored; void elements never open; an element opened at depth
    ``_MAX_DEPTH`` or deeper is added but never opened, so its content
    attaches to the deepest open element."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", {}, None)
        self._open: list[_Node] = [self.root]

    def _add(self, tag: str, attrs) -> _Node:
        node = _Node(tag, {k.lower(): (v or "") for k, v in attrs},
                     self._open[-1])
        self._open[-1].children.append(node)
        return node

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        node = self._add(t, attrs)
        if t not in _VOID_TAGS and len(self._open) < _MAX_DEPTH:
            self._open.append(node)

    def handle_startendtag(self, tag, attrs):
        self._add(tag.lower(), attrs)

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in _VOID_TAGS:
            return
        for i in range(len(self._open) - 1, 0, -1):
            if self._open[i].tag == t:
                del self._open[i:]
                return

    def handle_data(self, data):
        if data:
            self._open[-1].children.append(data)


def _iter_nodes(node: _Node) -> Iterator[_Node]:
    for child in node.children:
        if isinstance(child, _Node):
            yield child
            yield from _iter_nodes(child)


def _find_first(node: _Node, tag: str) -> Optional[_Node]:
    for n in _iter_nodes(node):
        if n.tag == tag:
            return n
    return None


def _prune(node: _Node, drop: Callable[[_Node], bool]) -> None:
    """Remove every descendant for which *drop* is true, top-down."""
    kept = []
    for child in node.children:
        if isinstance(child, _Node):
            if drop(child):
                continue
            _prune(child, drop)
        kept.append(child)
    node.children = kept


def _is_noise(node: _Node) -> bool:
    if node.tag in _SKIP_TAGS:
        return True
    attrs = node.attrs
    if "hidden" in attrs:
        return True
    if attrs.get("aria-hidden", "").strip().lower() == "true":
        return True
    if attrs.get("role", "").strip().lower() in _CHROME_ROLES:
        return True
    style = attrs.get("style", "")
    if style and _HIDDEN_STYLE_RE.search(style):
        return True
    return False


def _text_length(node: _Node) -> int:
    total = 0
    for child in node.children:
        if isinstance(child, str):
            total += len(child.strip())
        else:
            total += _text_length(child)
    return total


def _link_stats(node: _Node, in_link: bool = False) -> tuple[int, int, int]:
    """(text chars, text chars inside links, anchor count) for the subtree."""
    total = linked = anchors = 0
    for child in node.children:
        if isinstance(child, str):
            n = len(child.strip())
            total += n
            if in_link:
                linked += n
            continue
        inside = in_link or child.tag == "a"
        if child.tag == "a":
            anchors += 1
        t, li, a = _link_stats(child, inside)
        total += t
        linked += li
        anchors += a
    return total, linked, anchors


def _is_link_dense(node: _Node) -> bool:
    total, linked, anchors = _link_stats(node)
    return (anchors >= _LINK_DENSITY_MIN_ANCHORS and total > 0
            and linked / total >= _LINK_DENSITY_THRESHOLD)


def _prune_link_dense(node: _Node) -> None:
    """Remove link-dense blocks, children before parents."""
    kept = []
    for child in node.children:
        if isinstance(child, _Node):
            _prune_link_dense(child)
            if child.tag in _LINK_BLOCK_TAGS and _is_link_dense(child):
                continue
        kept.append(child)
    node.children = kept


def _render(node: _Node, out: list[str]) -> None:
    for child in node.children:
        if isinstance(child, str):
            out.append(child)
            continue
        t = child.tag
        if t == "br":
            out.append("\n")
            continue
        if t in _VOID_TAGS:
            continue
        if t in _PARAGRAPH_TAGS:
            out.append("\n\n")
            _render(child, out)
            out.append("\n\n")
        elif t in _LINE_TAGS:
            out.append("\n")
            _render(child, out)
            out.append("\n")
        elif t in _CELL_TAGS:
            out.append(" ")
            _render(child, out)
            out.append(" ")
        else:
            _render(child, out)


def _normalise(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


def _render_text(node: _Node) -> str:
    out: list[str] = []
    _render(node, out)
    return _normalise("".join(out))


def _clone(node: _Node, parent: Optional[_Node] = None) -> _Node:
    copy = _Node(node.tag, dict(node.attrs), parent)
    copy.children = [c if isinstance(c, str) else _clone(c, copy)
                     for c in node.children]
    return copy


def _find_title(root: _Node) -> str:
    title_node = _find_first(root, "title")
    if title_node is not None:
        title = _normalise(_render_text(title_node))
        if title:
            return " ".join(title.split())[:_TITLE_CAP]
    for n in _iter_nodes(root):
        if n.tag == "meta" and \
                n.attrs.get("property", "").lower() == "og:title":
            content = " ".join(n.attrs.get("content", "").split())
            if content:
                return content[:_TITLE_CAP]
    for heading in _HEADINGS:
        h = _find_first(root, heading)
        if h is not None:
            text = " ".join(_render_text(h).split())
            if text:
                return text[:_TITLE_CAP]
    return ""


@dataclass(frozen=True)
class ExtractedPage:
    """``title`` from ``<title>``, ``og:title`` or the first heading (may be
    empty); ``text`` the extracted content; ``region`` one of ``main``,
    ``article`` or ``body``."""

    title: str
    text: str
    region: str


def extract_page(markup: str) -> ExtractedPage:
    """Extract the main content of an HTML document. Never raises on
    malformed markup: whatever the parser built before an error is used."""
    builder = _TreeBuilder()
    try:
        builder.feed(markup or "")
        builder.close()
    except Exception:
        # Malformed markup: keep the partial tree.
        pass
    root = builder.root
    title = _find_title(root)
    _prune(root, _is_noise)
    body = _find_first(root, "body") or root

    region: Optional[_Node] = None
    region_kind = "body"
    mains = [n for n in _iter_nodes(body)
             if n.tag == "main" or n.attrs.get("role", "").lower() == "main"]
    if mains:
        region = max(mains, key=_text_length)
        region_kind = "main"
    else:
        articles = [n for n in _iter_nodes(body) if n.tag == "article"]
        if articles:
            region = max(articles, key=_text_length)
            region_kind = "article"

    if region is not None:
        region_copy = _clone(region)
        _prune(region_copy, lambda n: n.tag in _CHROME_TAGS)
        text = _render_text(region_copy)
        if len(text) >= _MIN_REGION_CHARS:
            return ExtractedPage(title=title, text=text, region=region_kind)

    _prune(body, lambda n: n.tag in _CHROME_TAGS
           or n.tag in _BODY_ONLY_CHROME_TAGS)
    plain = _render_text(body)
    _prune_link_dense(body)
    text = _render_text(body)
    if len(text) < _MIN_REGION_CHARS and len(plain) > len(text):
        text = plain
    return ExtractedPage(title=title, text=text, region="body")


def html_to_main_text(markup: str) -> str:
    """``extract_page(markup).text``."""
    return extract_page(markup).text
