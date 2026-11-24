# SPDX-License-Identifier: AGPL-3.0-or-later
"""The agentic loop: the public run/chat entry points, the turn loop itself, and
parallel tool dispatch. Mixed into Agent (see core.py)."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import localm.plugins.coder.agent as _agent
from ..display import (
    ActivitySpinner, console, print_assistant_response, print_info,
    print_success, print_task_footer, print_tool_call, print_tool_error,
    print_tool_result, print_turn_divider, safe_markup,
)
from ..parser import (
    invented_result_at, invented_result_without_call, looks_like_tool_attempt,
    names_a_tool_as_command, split_response, strip_orphan_closers, strip_tool_calls, unfinished_call,
)
from ..tools import ToolResult
from ..audit import SessionMode
from .constants import (
    _ACTION_VERBS, _IMPERATIVE_LEADS, _IMPERATIVE_ONLY_VERBS, _MAX_NOCALL_ESCALATIONS, _MAX_TOOL_REPAIRS,
    _NON_FILE_SUFFIXES, _STATEMENT_LINKS, _STATEMENT_PARTICIPLES, _STATEMENT_PREDICATES,
    _REPEAT_HISTORY_MAX, _REPEAT_RESPONSE_ABORT, _REPEAT_SIMILARITY,
    _BROWSER_TOOLS, _SKILL_STATE_TOOLS, confirm_forced,
)
from .planmode import INTERACTION_TOOLS
from localm.textguard import compose, compose_join

_RE_CLAUSE_SPLIT = re.compile(r"[,!?:;\n]+|\.+(?=\s|$)")
_RE_WORD = re.compile(r"[a-z']+")
_RE_POLITE_VERB = re.compile(
    r"\b(?:(?:can|could|would|will)\s+you(?:\s+mind)?"
    r"|please|pls|kindly|let'?s|go\s+ahead\s+and|go\s+and|try\s+to|be\s+sure\s+to"
    r"|don'?t\s+forget\s+to|feel\s+free\s+to"
    r"|(?:i|we)(?:'d|\s+would|\s+will)?\s+(?:like|want|need)(?:\s+you)?\s+to"
    r"|you\s+(?:need|have|should|must)\s+to|(?:i\s+)?(?:need|want)\s+to)"
    r"(?:\s+(?:also|just|please|kindly|now|then|quickly|first|simply))*"
    r"\s+(?=([a-z]+)\b)", re.IGNORECASE)
_RE_NO_TOOL_NEEDED = re.compile(r"\s*\[no tool needed\][ \t]*", re.IGNORECASE)
_RE_RUN_REQUEST = re.compile(r"\b(?:run|rerun|re-run|execute)\b", re.IGNORECASE)
_RE_TOOL_WORD = re.compile(r"[a-z][a-z_]*")
# A tag this harness puts at the start of its own re-prompts.
_RE_HARNESS_TAG = re.compile(
    r"\s*\[(?:no tool call|nothing written|code not written|tool-call format"
    r"|unfinished|self-verification|unverified code|plan not submitted"
    r"|not verified|code review)\]", re.IGNORECASE)


def _echoes_harness_prompt(text: str) -> bool:
    """True when *text* opens with one of the harness's own re-prompt tags: the
    model has repeated what it was told, which is not an answer."""
    return _RE_HARNESS_TAG.match(text) is not None
_QUESTION_OPENERS = frozenset({
    "why", "what", "whats", "what's", "how", "when", "where", "who", "which",
    "is", "are", "was", "were", "does", "do", "did", "has", "have", "had",
    "should", "shall", "isn't", "aren't", "doesn't", "didn't",
    "explain", "tell", "clarify",
})
_RE_PATH_TOKEN_SPLIT = re.compile(r"[\s`'\"()\[\]{}<>,;*|]+")
_RE_FILE_NAME = re.compile(r"[\w.-]*[\w-]\.([A-Za-z0-9]{1,8})")
_RE_SEPARATED_PATH = re.compile(r"(?:~|\.{1,2})?[/\\][\w.-]+|[\w.-]+[/\\][\w.-]+")
_RE_WORKSPACE_NOUN = re.compile(
    r"\b(?:file|files|folder|directory|dir|repo|repository|project|codebase"
    r"|workspace|script|module|package|test|tests|suite|branch|commit"
    r"|readme|config|source|sources)\b", re.IGNORECASE)


def _names_a_path(text: str) -> bool:
    """True when *text* contains a token that is a file name with an extension
    or a path with a separator. A domain name (``github.com``), a version number
    (``3.12``), an abbreviation (``e.g``) and a bare ellipsis are not paths."""
    for token in _RE_PATH_TOKEN_SPLIT.split(text):
        token = token.rstrip(".,:;!?")
        if not token or "://" in token:
            continue
        m = _RE_FILE_NAME.fullmatch(token)
        if m is not None:
            ext = m.group(1)
            if (any(c.isalpha() for c in ext)
                    and ext.lower() not in _NON_FILE_SUFFIXES
                    and token.lower() not in ("e.g", "i.e")):
                return True
        if _RE_SEPARATED_PATH.fullmatch(token):
            return True
    return False


def _is_question(text: str) -> bool:
    """True when *text* ends with a question mark or opens with an interrogative
    word."""
    stripped = (text or "").strip()
    if stripped.endswith("?"):
        return True
    words = _RE_WORD.findall(stripped.lower()[:40])
    return bool(words) and words[0] in _QUESTION_OPENERS


def _is_action_verb(word: str) -> bool:
    return word in _ACTION_VERBS or word in _IMPERATIVE_ONLY_VERBS


def _is_statement_subject(rest: list[str]) -> bool:
    """True when *rest*, the words after a clause-initial action word, make that
    word the subject of a statement: an auxiliary or a failure verb ("Build is
    red", "Search fails"), or a participle that ends the clause or is followed
    by a linking word ("Build failed.", "Test failed on CI")."""
    if not rest:
        return False
    if rest[0] in _STATEMENT_PREDICATES:
        return True
    return rest[0] in _STATEMENT_PARTICIPLES and (
        len(rest) == 1 or rest[1] in _STATEMENT_LINKS)


def demands_action(text: str) -> bool:
    """True when *text* tells the agent to DO something.

    An action verb counts when it is used as an imperative: it starts a clause
    (clauses split on commas, sentence ends, colons and line breaks; a leading
    "now", "yes" or "quickly" is skipped), or it follows a request prefix such as
    "please", "can you", "I'd like you to", "try to". A clause that opens with
    an action word as the subject of a statement ("Build failed.", "Search
    fails with a 500") is not an imperative. A text that is a question
    (ends with a question mark or opens with an interrogative) demands an action
    only through such a request prefix, so "search fails with a 500, why?" does
    not.

    Pure and module-level, so it can be tested directly on request strings."""
    lowered = (text or "").lower()
    if not lowered.strip():
        return False
    for m in _RE_POLITE_VERB.finditer(lowered):
        word = m.group(1)
        if _is_action_verb(word) or (
                word.endswith("ing") and (_is_action_verb(word[:-3])
                                          or _is_action_verb(word[:-3] + "e"))):
            return True
    if _is_question(text):
        return False
    for clause in _RE_CLAUSE_SPLIT.split(lowered):
        words = _RE_WORD.findall(clause)
        for i, word in enumerate(words):
            if word in _IMPERATIVE_LEADS:
                continue
            if _is_action_verb(word) and not _is_statement_subject(words[i + 1:]):
                return True
            break
    return False


def _is_change_verb(word: str) -> bool:
    return word in _CHANGE_VERBS


def _change_base(word: str) -> str:
    """The change verb *word* is, or is an -ing, -ed or -ied form of; "" when
    it is neither."""
    if word in _CHANGE_VERBS:
        return word
    for suffix, repl in (("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"),
                         ("ied", "y")):
        if word.endswith(suffix) and word[:-len(suffix)] + repl in _CHANGE_VERBS:
            return word[:-len(suffix)] + repl
    return ""


def _is_inflected_change_verb(word: str) -> bool:
    """True for an -ing, -ed or -ied form of a change verb."""
    return word not in _CHANGE_VERBS and bool(_change_base(word))


def _is_negated_change(words: list[str], k: int, carry: bool) -> bool:
    """True when the change word at ``words[k]`` is negated.

    It is negated when a negation ("not", "never", "don't", "without", ...)
    comes before it in the clause with only conjunctions, adverbs, other change
    words or a "need/want/try ... to" or "should be" span in between ("do not
    edit or change", "don't even try to modify", "should not be changed"), or
    when "no" or "nothing" follows it ("make no changes", "change nothing").
    With *carry*, a run of such words that reaches the start of the clause
    continues a negation that ended the previous clause ("do not edit, modify
    or rewrite anything")."""
    if k + 1 < len(words) and words[k + 1] in _NEGATED_OBJECTS:
        return True
    for j in range(k - 1, -1, -1):
        w = words[j]
        if w in _CHANGE_NEGATIONS:
            return True
        if not (w in _NEGATION_SPAN or _change_base(w)):
            return False
    return carry


def _has_file_target(rest: str) -> bool:
    """True when *rest*, the text after a piece of prose a request asks for,
    sends it to a file: a path, or a file or workspace noun ("file", "the
    README", "a docstring"), after "to", "into" or "as", or after "in" or
    "inside" that does not follow "is", "are", "was" or "were" ("what is in
    config.py" names where the content is, not where the text goes)."""
    parts = rest.split()
    for i, part in enumerate(parts):
        if part not in _PROSE_TARGET_PREPS:
            continue
        if part in ("in", "inside") and i and parts[i - 1] in ("is", "are", "was", "were"):
            continue
        targets = parts[i + 1:i + 3]
        if targets and targets[0] in _PROSE_TARGET_DETERMINERS:
            targets = targets[1:]
        if not targets:
            continue
        target = targets[0].rstrip(".,:;!?")
        if (_names_a_path(target) or _RE_WORKSPACE_NOUN.fullmatch(target)
                or target in _PROSE_TARGET_NOUNS):
            return True
    return False


def _is_prose_output(clause: str, tokens: list[re.Match], k: int) -> bool:
    """True when the change verb at ``tokens[k]`` asks for text in the reply,
    not for a change to the project.

    That is a "write", "make", "create" or "generate" (or an -ing or -ed form)
    whose object, within the next four words, is a summary, an explanation, a
    description, an answer, a report or similar prose, with no code or
    workspace noun before it, not used as a modifier of a code noun ("a
    summary function"), not part of a file name ("summary.md"), and with no
    file target after it (see :func:`_has_file_target`)."""
    words = [t.group() for t in tokens]
    if _change_base(words[k]) not in _PROSE_VERBS:
        return False
    for j in range(k + 1, min(len(words), k + 5)):
        w = words[j]
        if w in _PROSE_NOUNS:
            start, end = tokens[j].start(), tokens[j].end()
            if (start and clause[start - 1] in "./\\_-") or (
                    end < len(clause) and clause[end] in "./\\_-"
                    and clause[end + 1:end + 2].strip()):
                return False
            if j + 1 < len(words) and (words[j + 1] in _CODE_NOUNS
                                        or _RE_WORKSPACE_NOUN.fullmatch(words[j + 1])):
                return False
            return not _has_file_target(clause[end:])
        if w in _CODE_NOUNS or _RE_WORKSPACE_NOUN.fullmatch(w):
            return False
    return False


def asks_for_change(text: str) -> bool:
    """True when *text* tells the agent to change the project: a change verb
    (``implement``, ``write``, ``add``, ``fix``, ...) that is not used as a noun
    or in a description, or an -ing or -ed form of one after "start", "try",
    "by" or an obligation ("needs", "should", "must").

    A change word after a determiner, a possessive, an auxiliary or a question
    word ("the code that loads the config", "where the file is created", "what
    was added") asks to read, and so does a participle with none of those
    leads. A change verb after any other word does not: "Open utils.py and in
    the retry function add a backoff" asks for a change. A noun-like change
    word ("code") counts only when it opens a clause.

    A negated change word never counts ("do not edit anything", "don't change
    it", "make no changes", see :func:`_is_negated_change`), and neither does a
    request for prose in the reply ("write a short summary", see
    :func:`_is_prose_output`). Any other change verb in the text still makes it
    a change request: "read a.py, then fix b.py; do not touch c.py" is one.

    Pure and module-level, so it can be tested directly on request strings."""
    lowered = (text or "").lower()
    if not lowered.strip():
        return False
    for m in _RE_POLITE_VERB.finditer(lowered):
        word = m.group(1)
        if _is_change_verb(word) or _is_inflected_change_verb(word):
            tail = _RE_CLAUSE_SPLIT.split(lowered[m.start(1):], maxsplit=1)[0]
            tail_tokens = list(_RE_WORD.finditer(tail))
            head = _RE_WORD.findall(_RE_CLAUSE_SPLIT.split(lowered[:m.start(1)])[-1])
            if _is_negated_change(head + [t.group() for t in tail_tokens], len(head), False):
                continue
            if not _is_prose_output(tail, tail_tokens, 0):
                return True
    carry = False
    for clause in _RE_CLAUSE_SPLIT.split(lowered):
        tokens = list(_RE_WORD.finditer(clause))
        words = [t.group() for t in tokens]
        first = next((k for k, w in enumerate(words) if w not in _IMPERATIVE_LEADS),
                     len(words))
        negated_last = False
        for k, word in enumerate(words):
            prev = words[k - 1] if k else ""
            if _is_change_verb(word):
                if _is_negated_change(words, k, carry):
                    negated_last = k == len(words) - 1
                    continue
                if prev in _CHANGE_NOUN_CONTEXT:
                    continue
                if word == "code" and k != first:
                    continue
                if k == first and _is_statement_subject(words[k + 1:]):
                    continue
                if _is_prose_output(clause, tokens, k):
                    continue
                return True
            if _is_inflected_change_verb(word) and k > 0 and (
                    prev in _CHANGE_FORM_LEADS
                    or any(w in _CHANGE_OBLIGATIONS for w in words[max(0, k - 6):k])):
                if _is_negated_change(words, k, carry) or _is_prose_output(clause, tokens, k):
                    continue
                return True
        carry = negated_last
    return False


def _asks_a_question(response: str) -> bool:
    """True when *response* is a short plain-text question: its last line ends
    with a question mark and it holds no code block."""
    body = (response or "").strip()
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return bool(lines) and lines[-1].endswith("?") and len(body) <= 600 and "```" not in body


def implies_action(text: str) -> bool:
    """True when *text* asks for something that needs a TOOL, not an
    explanation - the precondition for escalating a turn that produced no tool
    call.

    True for an imperative (:func:`demands_action`), for a named file or path
    (``main.py``, ``src/app``), and, for a text that is not a question, for an
    action word anywhere in it or a workspace noun ("the repo", "the tests"). A
    question without a file or an imperative ("why does web search sometimes
    fail?") is not an action request. Read verbs count: "show me what is in
    config.py" needs read_file exactly as much as "write config.py" needs
    write_file.

    Pure and module-level, so it can be tested directly on request strings
    without constructing an Agent."""
    if not text:
        return False
    if demands_action(text) or _names_a_path(text):
        return True
    if _is_question(text):
        return False
    if any(w in _ACTION_VERBS for w in _RE_WORD.findall(text.lower())):
        return True
    return bool(_RE_WORKSPACE_NOUN.search(text))


# A fenced code block's body.
_RE_FENCE_BLOCK = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
# A name a code block DEFINES (Python, JS/TS, Go, Rust), not one it only uses.
_RE_CODE_DEF = re.compile(
    r"^[ \t]*(?:export[ \t]+)?(?:pub[ \t]+)?(?:async[ \t]+)?"
    r"(?:def|class|function|func|fn)[ \t]+([A-Za-z_$][\w$]*)", re.MULTILINE)
# Prose is split into tokens on these before a token is tested as a path, so the
# path pattern is only ever matched against one short token.
_RE_TOKEN_SPLIT = re.compile(r"[\s`'\"()\[\]{}<>,;*|]+")
_RE_PATH_TOKEN = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]{1,8}")
_UNFOUNDED_MAX_BYTES = 2_000_000


def _cited_path(segment: str) -> str | None:
    """The last path-looking token (``dir/name.ext``) in *segment*, or None."""
    cited = None
    for token in _RE_TOKEN_SPLIT.split(segment):
        token = re.sub(r":\d+(?::\d+)?$", "", token.rstrip(".,:;!?"))
        if token and _RE_PATH_TOKEN.fullmatch(token):
            cited = token
    return cited


def unfounded_code(text: str, cwd) -> list[tuple[str, list[str]]]:
    """``[(path, names), ...]`` for each workspace file *text* shows code for
    whose code defines names that file does not contain.

    A fenced code block is attributed to the path named last in the prose
    right before it. When that path is an existing file inside *cwd*, every
    name the block defines (``def``/``class``/``function``/``func``/``fn``) is
    looked up in the file on disk; a name that occurs nowhere in it means the
    block is not the file's content. A block with no path before it, a path
    that is not an existing workspace file, or a file too large to check is
    skipped. Only the file on disk is consulted, never the reply's account of
    what it read. Pure, so it can be tested without an Agent."""
    from ..tools.base import _confine
    root = Path(cwd)
    found: dict[str, list[str]] = {}
    contents: dict[Path, str | None] = {}
    prev_end = 0
    for block in _RE_FENCE_BLOCK.finditer(text or ""):
        cited = _cited_path(text[prev_end:block.start()])
        prev_end = block.end()
        names = list(dict.fromkeys(_RE_CODE_DEF.findall(block.group(1))))
        if not cited or not names:
            continue
        try:
            path = _confine(root, cited.replace("\\", "/"))
        except (PermissionError, OSError, ValueError):
            continue
        if path not in contents:
            try:
                ok = path.is_file() and path.stat().st_size <= _UNFOUNDED_MAX_BYTES
                contents[path] = (path.read_text(encoding="utf-8", errors="replace")
                                  if ok else None)
            except OSError:
                contents[path] = None
        body = contents[path]
        if body is None:
            continue
        missing = [n for n in names
                   if not re.search(r"(?<![\w$])" + re.escape(n) + r"(?![\w$])", body)]
        if missing:
            try:
                rel = path.relative_to(root.resolve()).as_posix()
            except ValueError:
                rel = cited.replace("\\", "/")
            found.setdefault(rel, [])
            found[rel].extend(n for n in missing if n not in found[rel])
    return list(found.items())


# A fenced block's language tag and body.
_RE_FENCE_WITH_LANG = re.compile(r"```[ \t]*([\w+.-]*)[^\n`]*\n(.*?)```", re.DOTALL)
_SHELL_FENCE_LANGS = frozenset({
    "sh", "bash", "zsh", "shell", "console", "terminal", "powershell", "ps1",
    "pwsh", "cmd", "bat", "batch", "text", "txt", "plaintext", "output", "log",
})
_FILE_CODE_MIN_LINES = 4
# Verbs of a step a tool would carry out.
_STEP_VERBS = (
    "run|rerun|re-run|execute|test|check|verify|create|write|rewrite|add|update|"
    "modify|edit|change|fix|implement|read|open|inspect|search|grep|install|apply|"
    "build|refactor|remove|delete|rename|move|commit|launch|compile|configure|"
    "generate|replace|patch|insert|call")
# A last line that announces a step instead of reporting one.
_RE_ANNOUNCED_STEP = re.compile(
    r"^(?:(?:now|next|then|ok|okay|great|alright|first|finally)[,.!]?\s+)*"
    r"(?:let\s+me|let'?s|let\s+us|i'?ll|i\s+will|i'?m\s+going\s+to"
    r"|i\s+am\s+going\s+to|we'?ll|we\s+will)\s+"
    r"(?:(?:now|also|quickly|just|first|then|go\s+ahead\s+and)\s+)*"
    r"(?:" + _STEP_VERBS + r")\b",
    re.IGNORECASE)
# Verbs that, used as an imperative, ask for the project to change.
_CHANGE_VERBS = frozenset({
    "implement", "write", "create", "add", "build", "make", "fix", "change",
    "update", "modify", "edit", "refactor", "rewrite", "generate", "code",
    "develop", "extend", "replace", "insert", "apply", "port", "convert",
    "finish", "complete", "rename",
})
# Words right before a change word that make it a noun, a participle of a
# description or part of a question ("the update", "was added", "where users
# create").
_CHANGE_NOUN_CONTEXT = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "its",
    "their", "our", "his", "her", "of", "me", "us", "new", "existing", "some",
    "any", "all", "each", "every", "source", "same", "whole", "full", "last",
    "latest", "recent", "what", "which", "where", "when", "how", "whether",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "no", "one", "another", "other", "will", "would", "can", "could", "may",
    "might", "does", "do", "did", "doesn't", "don't",
})
# Words that make a following -ing or -ed form of a change verb a request.
_CHANGE_FORM_LEADS = frozenset({
    "start", "begin", "continue", "keep", "proceed", "try", "by"})
_CHANGE_OBLIGATIONS = frozenset({"needs", "need", "should", "must"})
# Words that negate a change word after them.
_CHANGE_NEGATIONS = frozenset({
    "not", "never", "don't", "dont", "doesn't", "doesnt", "won't", "wont",
    "shouldn't", "shouldnt", "mustn't", "mustnt", "can't", "cant", "cannot",
    "nor", "without", "nothing"})
# Words that may stand between a negation and the change word it negates.
_NEGATION_SPAN = frozenset({
    "or", "and", "nor", "even", "ever", "actually", "yet", "also", "really",
    "need", "needs", "want", "you", "to", "try", "attempt", "bother", "should",
    "must", "be", "been", "being", "get"})
# Objects that negate the change word before them ("make no changes").
_NEGATED_OBJECTS = frozenset({"no", "nothing"})
# Change verbs that can ask for prose in the reply ("write a summary").
_PROSE_VERBS = frozenset({"write", "make", "create", "generate"})
# Pieces of prose an answer can be.
_PROSE_NOUNS = frozenset({
    "summary", "summaries", "explanation", "explanations", "description",
    "descriptions", "answer", "answers", "report", "overview", "recap",
    "analysis", "breakdown", "outline", "walkthrough", "review", "critique",
    "assessment", "evaluation", "synopsis", "rundown", "comparison",
    "paragraph", "paragraphs", "sentence", "sentences", "response", "reply",
    "plan"})
# Nouns that make a piece of prose part of the code ("a summary function").
_CODE_NOUNS = frozenset({
    "function", "functions", "method", "methods", "class", "classes", "script",
    "scripts", "module", "modules", "test", "tests", "file", "files", "page",
    "endpoint", "command", "field", "section", "table", "view", "component",
    "helper", "tool", "feature", "generator", "route", "screen", "widget",
    "option", "flag", "doc", "docs", "document", "docstring", "docstrings",
    "comment", "comments", "line", "lines", "column", "template", "struct",
    "type", "api", "job", "hook", "handler", "parser", "cli", "button",
    "dialog", "tab", "panel", "schema", "query", "variable", "constant", "log",
    "logger", "step", "task", "card", "row", "header", "footer", "message"})
# Words before a file target of a piece of prose ("to SUMMARY.md").
_PROSE_TARGET_PREPS = frozenset({"to", "into", "as", "in", "inside"})
_PROSE_TARGET_DETERMINERS = frozenset({
    "the", "a", "an", "new", "my", "our", "your", "this", "that", "its"})
# File nouns a piece of prose can be sent to, besides the workspace nouns.
_PROSE_TARGET_NOUNS = frozenset({
    "comment", "comments", "docstring", "docstrings", "docs", "documentation",
    "changelog", "notes", "wiki", "doc", "document"})


# Folders a test run or this tool leaves behind, not a change a command made.
_CACHE_DIRS = frozenset({"__pycache__", ".pytest_cache", ".localcoder",
                         ".mypy_cache", ".ruff_cache", "node_modules"})
_MAX_RECORD_PATHS = 10
# A shell command that runs a project's tests.
_RE_TEST_COMMAND = re.compile(
    r"^\s*(?:(?:python[\d.]*|py)\s+-m\s+)?(?:pytest|unittest|tox|nox|jest|vitest|mocha)\b"
    r"|^\s*(?:npm|pnpm|yarn)\s+(?:run\s+)?test\b"
    r"|^\s*(?:cargo|go|dotnet|mvn|gradle)\s+test\b"
    r"|^\s*make\s+(?:test|check)\b",
    re.IGNORECASE)


def _may_write_files(call, cwd) -> bool:
    """True for a call that can change files without the write tracker seeing
    it: search_replace, generate_image, a sub-agent that can write, an MCP or
    plugin tool, or a shell command that is neither read-only nor a test run."""
    name = call.name
    if name in ("search_replace", "generate_image") or name.startswith(("mcp_", "plugin_")):
        return True
    if name in ("spawn_agent", "spawn_agent_background", "dispatch_parallel"):
        return _delegated_work_may_write(call)
    if name not in ("run_shell", "run_shell_background"):
        return False
    command = call.args.get("command") if isinstance(call.args, dict) else None
    if not isinstance(command, str):
        return True
    if _RE_TEST_COMMAND.match(command):
        return False
    from ..shell_guard import is_read_only
    from ..tools.shell import executed_forms
    try:
        return not all(is_read_only(form, cwd) for form in executed_forms(command))
    except Exception:
        return True


def _delegated_work_may_write(call) -> bool:
    """True unless *call* hands work to a sub-agent whose named role has no
    tool that writes a file."""
    role = call.args.get("role") if isinstance(call.args, dict) else None
    if call.name == "dispatch_parallel" or not role:
        return True
    from ..roles import resolve_role
    try:
        preset = resolve_role(role)
    except ValueError:
        return False
    return preset is None or bool(
        preset.allowed_tools & {"write_file", "edit_file", "patch_file", "search_replace"})


def _has_file_code_block(text: str) -> bool:
    """True when *text* holds a fenced block of at least four non-blank lines
    whose language is not a shell, console or plain-text one."""
    for m in _RE_FENCE_WITH_LANG.finditer(text or ""):
        if m.group(1).lower() in _SHELL_FENCE_LANGS:
            continue
        if sum(1 for ln in m.group(2).splitlines() if ln.strip()) >= _FILE_CODE_MIN_LINES:
            return True
    return False


def _announced_step(text: str) -> str:
    """The last non-blank line of *text* when it announces a next step ("Let me
    run the tests.", "Now I'll update main.py:") rather than asking a question;
    ``""`` otherwise. Text inside a code block is not considered."""
    prose = _RE_FENCE_WITH_LANG.sub("", text or "")
    lines = [ln.strip().strip("*_>#- ").strip() for ln in prose.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    last = lines[-1]
    if last.endswith("?") or len(last) > 200:
        return ""
    return last if _RE_ANNOUNCED_STEP.match(last) else ""


def _dedupe_calls(calls: list) -> tuple[list, int]:
    """``(calls, dropped)``: *calls* with every call whose name and arguments
    equal an earlier call's removed, in order, and how many were removed."""
    import json
    seen: set = set()
    kept = []
    for call in calls:
        try:
            key = (call.name, json.dumps(call.args, sort_keys=True, default=str))
        except (TypeError, ValueError):
            kept.append(call)
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(call)
    return kept, len(calls) - len(kept)


def _with_notices(response: str, *notices: str) -> str:
    """*response* as it is kept in the conversation, with the harness's own
    notices about how the run ended in front of it: the saved transcript cuts a
    long message at its end."""
    said = [n.strip() for n in notices if n.strip()]
    return "\n\n".join([*said, response]) if said else response


def response_similarity(a: str, b: str) -> float:
    """difflib ratio of two responses, whitespace-normalised. 1.0 is identical."""
    import difflib
    return difflib.SequenceMatcher(None, " ".join(a.split()),
                                   " ".join(b.split())).ratio()


class _LoopMixin:
    if TYPE_CHECKING:
        # Set by Agent (core.py) through _PlanModeMixin.
        @property
        def plan_mode(self) -> bool: ...

        def _add_user_with_plan_note(self, content) -> None: ...
        def _end_plan_autorun(self) -> None: ...

        # Set by Agent (core.py).
        parent: Any
        gen_kwargs: dict
        on_event: Any
        name: str
        turn_budget: int
        _messages: list

        def _call_llm(self, messages: list[dict], interactive: bool) -> str: ...

        _loose_check: tuple | None
        _consecutive_errors: dict[str, int]

        def _loose_calls_to_check(self, calls: list) -> list: ...

        def _ask_if_calls_meant(self, calls: list) -> bool | None: ...

        def _build_messages(self) -> list[dict]: ...

        _max_tokens_explicit: bool | None
        _reply_limit: int | None
        _writing_commands: int
        _files_read: set
        _shell_baseline_captured: bool
        confirm_handler: Any
        always_confirm: Any

        _git_baseline: frozenset | None

        def _git_status_map(self) -> dict[str, str] | None: ...

        @property
        def stop_pending(self) -> bool: ...

        def start_deferred_mcp(self) -> list: ...

        # Set by Agent (core.py) through _ExecutionMixin.
        def _hooks_may_ask(self, calls) -> bool: ...

        def _may_need_terminal_confirm(self, calls, interactive: bool) -> bool: ...

        def _verify_refusal(self, label: str, interactive: bool) -> str | None: ...

        # Set by Agent (core.py) through _ContextMixin.
        def _note_typed_request(self, text) -> None: ...

        def report_deferred_mcp(self, interactive: bool) -> None: ...

        def apply_project_trust(self, interactive: bool = False) -> bool: ...

        def can_force_tool_calls(self) -> bool: ...

        @property
        def disabled_tools(self) -> frozenset: ...

        max_turns: int
        _task_start_turns: int
        patch_mode: bool
        dry_run: bool
        _user_declines: int
        _lenient_denials: int
        _denied_lock: Any
        cwd: Path
        _unverified_writes: set
        _audit: Any

        def _add_user(self, content) -> None: ...

        def _add_assistant(self, content) -> None: ...

        def _emit(self, event_type: str, **data) -> None: ...

        def _record_typed_request(self, text: str) -> None: ...

    def run_task(self, task: str, attachments: str = "", *,
                 record_request: bool = True, interactive: bool = False) -> str:
        """
        Run a single task to completion (non-interactive unless *interactive*).

        *attachments* is appended to what the model receives only, as in
        :meth:`chat`. *interactive* runs the loop as a terminal session does:
        tool calls and results are printed, and a call that needs confirmation
        is asked at the terminal.

        Returns the agent's final text response.
        Used by spawn_agent and the CLI `run` command.
        """
        if self._episodic and not self._episode_task:
            self._episode_task = task
        if not self._review_task:
            self._review_task = task   # remembered for the pre-done diff review
        if not self._session_title:
            self._session_title = task   # resume-listing display title
        # The raw request, before _with_episodes can prepend a lessons preamble.
        self._last_user_request = task
        if getattr(self, "parent", None) is None:
            self._untrusted_in_context = False
        if record_request and getattr(self, "parent", None) is None:
            self._note_typed_request(task)
            self._record_typed_request(task)
        message = self._with_episodes(task)
        if attachments:
            message = compose(message, attachments)
        note = "" if interactive else self._unconfirmable_note()
        self._add_user_with_plan_note(compose(message, note) if note else message)
        return self._loop(interactive=interactive)

    def _always_denied(self) -> list[str]:
        """The tools this run can only deny: they need a confirmation even under
        auto_approve and nobody can give one."""
        if (self.confirm_handler is not None or getattr(self, "_interactive", False)
                or not self.always_confirm):
            return []
        disabled = self.disabled_tools
        return sorted(
            name for name, td in _agent.TOOL_REGISTRY.items()
            if name not in disabled and confirm_forced(self.always_confirm, name, td))

    def _unconfirmable_note(self) -> str:
        """Tells the model which tools this run can only deny, or ""."""
        names = self._always_denied()
        if not names:
            return ""
        return ("\n\n[this run] Nobody can confirm a call in this run, so these "
                f"tools are always denied: {', '.join(names)}. Do not call them. Do "
                "the rest of the task, and say in your final answer what is missing "
                "because of it.")

    def continue_task(self, message: str, *, interactive: bool = False) -> str:
        """
        Continue the current session with another non-interactive instruction,
        preserving history.

        Unlike :meth:`run_task` it does not re-inject episodic recall (already
        injected on the first task). Used by the CLI goal loop to feed a
        verification failure back to the agent for another fix attempt.
        *interactive* is as for :meth:`run_task`.
        """
        self._last_user_request = message
        self._add_user_with_plan_note(message)
        return self._loop(interactive=interactive)

    def chat(self, user_input: str, attachments: str = "", *,
             record_request: bool = True) -> str:
        """
        Send one user message in an ongoing interactive session.

        *attachments* (the REPL's @file blocks) is appended to the message the
        model receives but is not part of the request text the session title,
        action classification, episode recall and reviewer read.
        See test_attachment_text_does_not_turn_a_question_into_an_action.

        History is preserved between calls.
        Returns the agent's final text response for this turn.
        """
        if not self._session_title:
            # The raw input, before _with_episodes can prepend a lessons preamble.
            self._session_title = user_input
        # Recall relevant past lessons on the first turn of a session (the turn
        # that sets the session's task); later turns keep the same context.
        self._last_user_request = user_input   # raw, see run_task
        self._untrusted_in_context = False
        if record_request:
            self._note_typed_request(user_input)
            self._record_typed_request(user_input)
        if self._episodic and not self._episode_task:
            self._episode_task = user_input
            user_input = self._with_episodes(user_input)
        if not self._review_task:
            self._review_task = user_input
        self._add_user_with_plan_note(compose(user_input, attachments) if attachments else user_input)
        return self._loop(interactive=True)

    def _with_episodes(self, task: str) -> str:
        """Prepend relevant past lessons (episodic memory) to *task*, if any.
        Best-effort: a retrieval failure just returns the task unchanged.

        Also RECORDS which lessons were injected (id + text), on the agent, on the
        event stream, and in the audit trail, so a lesson that steered a run badly
        is identifiable afterwards and can be forgotten by id."""
        if not self._episodic or self._episode_store is None:
            return task
        try:
            from ..episodes import render_for_prompt
            episodes = self._episode_store.search(task)
            block = render_for_prompt(episodes)
        except Exception as e:
            # A failed recall is recorded rather than silently ignored; the task
            # still runs.
            self._record_episodes_used([], reason="recall failed: %s" % e)
            return task
        if not block:
            self._record_episodes_used([], reason="no relevant past lesson")
            return task
        self._record_episodes_used(episodes)
        return compose(block, "\n\n## Task\n", task)

    def _record_episodes_used(self, episodes: list, reason: str = "") -> None:
        """Stash + surface the recalled lessons for this run.

        Mirrors the chat plugin's ``_stash_memory_used``: metadata plus the lesson
        text that was ALREADY injected into the prompt, so this adds no disclosure
        beyond what the prompt itself carries (and the audit log already records
        the prompt). Best-effort and side-effect free - a surfacing failure must
        never break the run."""
        try:
            used = [{"id": e.id, "outcome": e.outcome,
                     "lesson": (e.lesson or e.summary or "")[:200]}
                    for e in episodes]
            self._episodes_used = used
            self._episodes_degrade_reason = reason
            from localm.debuglog import logger
            logger.debug("episodic recall: injected %d lesson(s)%s", len(used),
                         (" (%s)" % reason) if reason else "")
            if used:
                self._audit.episodes_recalled(used)
                self._emit("episodes_recalled", episodes=used)
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("episodic recall surfacing skipped: %s", e)

    def _loop(self, interactive: bool) -> str:
        """
        Agentic loop: call LLM → parse tool calls → execute → repeat.
        Returns the final response text.
        """
        # Record whether this session owns a terminal it can prompt on. A spawned
        # child always runs _loop(interactive=False).
        self._interactive = interactive
        # Project MCP servers held back as untrusted start here once trusted
        # (the GUI or another session may have trusted them since); while they
        # stay untrusted the session says so once.
        if self.parent is None:
            self.apply_project_trust(interactive)
            self.start_deferred_mcp()
            self.report_deferred_mcp(interactive)
        # Live attribute lookup, so a patched agent.parse_tool_calls / confirm /
        # print_warning / TOOL_REGISTRY is honoured.
        parse_tool_calls = _agent.parse_tool_calls
        confirm = _agent.confirm
        print_warning = _agent.print_warning
        TOOL_REGISTRY = _agent.TOOL_REGISTRY
        final_response = ""
        self._stop_requested = False       # a stale stop must not kill a new task
        self._user_stopped = False         # per-run: an earlier run's stop does not
                                           # mute this run's failure lesson
        self._last_run_ok = True           # per-run: an earlier run's failure does not
                                           # label this run failed. _had_any_failure is
                                           # the session-wide flag, set in the finally.
        self._last_verify_state = None     # per-run: this run is unverified until
                                           # its own gate says otherwise
        self._denied_unconfirmed = []      # per-run: see Agent.denied_unconfirmed
        self._reply_limit = None           # per-run: see _raise_reply_limit
        self._files_read = set()           # per-run
        start_turns = self._turns          # turns used by *this* task only
        self._task_start_turns = start_turns
        budget_escalated = False           # uncertainty escalation fires at most once per task
        # Per-task one-shot flags for the no-tool-calls handler: self-verification
        # and pre-done review fire once each, repair re-prompts are capped. Held on
        # a namespace so the helper can persist them across turns. The verify_*
        # fields drive the exit-code oracle gate (_run_verify_gate);
        # verify_checked_at is the write count the check last passed at, so the
        # gate fires only once this task has written something since.
        st = SimpleNamespace(verify_nudged=False, review_done=False, repair_count=0,
                             verify_retries=0, verify_settled=False,
                             verify_checked_at=self._write_total(),
                             partial_notice_count=0, partial_notice_cap_announced=False,
                             nocall_escalation=0, nocall_declined=False, tool_calls_made=0,
                             nocall_response="", code_nudged=False, code_forced=False, code_reminded=0,
                             announce_nudged=False,
                             possible_write=False, announced_step="",
                             writes_at_start=self._write_total(),
                             files_at_start={f["path"]: f["writes"]
                                             for f in self.changed_files()},
                             start_turns=start_turns,
                             budget_reminded=0, invented_count=0,
                             question_answered=False, example_replies=0,
                             cut_off_count=0,
                             unfounded_checked=False,
                             plan_submitted=False, plan_nudges=0,
                             declines_at_start=self._user_declines)

        try:
            while self._task_turns_left():
                if self.stop_pending:
                    self._stop_requested = False
                    final_response = self._stopped_text()
                    self._last_run_ok = False
                    self._user_stopped = True
                    break

                # Mid-task steering: deliver queued user messages before the
                # next LLM call so the agent reads them this turn.
                for queued in self._drain_queued():
                    self._note_typed_request(queued)
                    self._record_typed_request(queued)
                    self._add_user(
                        "[user steering note - read this before continuing, it "
                        f"overrides earlier instructions where they conflict]\n{queued}"
                    )
                    self._emit("info", text="steering note delivered to the agent")
                    if interactive:
                        print_info("(steering note delivered)")

                # Absorb background sub-agents that finished since the last turn,
                # on this thread only, at the same point as steering notes so the
                # model reads the result before its next call.
                for note in self._drain_background_agents():
                    self._add_user(compose("[background sub-agent result]\n", note))
                    self._emit("info", text="background sub-agent result absorbed")
                    if interactive:
                        print_info("(background sub-agent finished)")

                self._turns += 1
                prev_turn_tokens = self._last_turn_tokens
                self._last_turn_tokens = 0   # reset counter for this turn

                ctx_ratio = self._fill_ratio()
                if interactive:
                    print_turn_divider(self._turns, self._total_tokens,
                                       prev_turn_tokens, ctx_ratio=ctx_ratio)
                self._emit("turn", turn=self._turns,
                           total_tokens=self._total_tokens,
                           ctx_ratio=round(min(ctx_ratio, 1.0), 3))

                self._audit.set_turn(self._turns)

                # ---- uncertainty escalation ------------------------------
                # Past the turn budget: interactively ask the user whether to keep
                # going; non-interactively tell the model to surface blockers.
                task_turns = self._turns - start_turns
                if not budget_escalated and task_turns > self.turn_budget:
                    budget_escalated = True
                    if interactive:
                        print_warning(
                            f"This task has used {task_turns} turns "
                            f"(budget: {self.turn_budget})."
                        )
                        if not confirm("  Keep going?"):
                            final_response = (
                                f"[stopped by user after {task_turns} turns - "
                                "task exceeded its turn budget]"
                            )
                            self._last_run_ok = False
                            self._user_stopped = True
                            break
                    else:
                        self._emit(
                            "info",
                            text=f"Turn budget exceeded ({task_turns}/{self.turn_budget}) - "
                                 "asking the agent to surface blockers instead of guessing.",
                        )
                        self._add_user(
                            f"[turn budget] You have used {task_turns} turns on this "
                            f"task (budget: {self.turn_budget}). If you are stuck or "
                            "uncertain, STOP guessing: summarise what you tried, state "
                            "exactly what is blocking you, and ask for guidance instead "
                            "of continuing to experiment. If you are genuinely close to "
                            "done, finish with the minimal remaining steps."
                        )
                elif budget_escalated and not interactive:
                    self._remind_of_budget(task_turns, st)

                # ---- context-budget check --------------------------------
                self._maybe_compact(interactive=interactive)

                # ---- call LLM -------------------------------------------
                messages = self._build_messages()
                response = self._call_llm(messages, interactive=interactive)
                # One-shot: the forcing grammar applies only to the turn the
                # ladder armed it for. Cleared here, next to the single call that
                # consumes it, so no later path inherits a constrained sampler.
                self._force_tool_grammar = False

                if self.stop_pending:
                    # Stopped mid-generation: keep the partial text, run nothing
                    self._stop_requested = False
                    self._add_assistant(response)
                    final_response = response or self._stopped_text()
                    self._last_run_ok = False
                    self._user_stopped = True
                    break

                # ---- parse tool calls ------------------------------------
                # Pass the known tool names so the name-gated formats (bare JSON
                # and json / bare fences) are recognised.
                tool_names = set(TOOL_REGISTRY) - self.disabled_tools
                parsed_calls = parse_tool_calls(response, tool_names=tool_names)
                # What follows a result the model invented is dropped, calls
                # included. Judged on the whole reply, so a call in it is as
                # trusted or as lenient as it was written.
                cut_at = invented_result_at(response, tool_names) if isinstance(
                    response, str) else -1
                invented = cut_at >= 0
                if invented:
                    response = response[:cut_at]
                    parsed_calls = [c for c in parsed_calls if c.end <= cut_at]
                    self._audit.notice("invented_result", "the reply went on with a "
                                       "tool result it wrote itself; cut off there")
                calls, duplicate_calls = _dedupe_calls(parsed_calls)

                # ---- repeated-scaffold breaker ---------------------------
                # Counts a turn whose response is SIMILAR (not identical) to any
                # turn in a bounded history AND whose tool-call signature matches
                # that turn's. Both conditions must hold. Matching against the
                # whole history, not just the previous turn, also catches an
                # A-B-A-B alternation.
                fp = (response or "").strip()
                sig = " | ".join(
                    f"{c.name}({sorted((c.args or {}).items())!r})" for c in calls)
                if fp:
                    history = getattr(self, "_recent_finals", None)
                    if history is None:
                        history = self._recent_finals = []
                    match = max((response_similarity(fp, prev_fp)
                                 for prev_fp, prev_sig in history
                                 if prev_sig == sig), default=0.0)
                    if match >= _REPEAT_SIMILARITY:
                        self._repeat_response_count += 1
                    else:
                        self._repeat_response_count = 0
                    history.append((fp, sig))
                    del history[:-_REPEAT_HISTORY_MAX]
                    self._last_response_fp = fp
                if self._repeat_response_count >= _REPEAT_RESPONSE_ABORT - 1:
                    final_response = (
                        "[circuit breaker: the model repeated essentially the "
                        f"same response {self._repeat_response_count + 1} times "
                        "with no progress - stopping so you can adjust the "
                        "approach instead of burning more turns. The "
                        "conversation is intact.]")
                    print_warning(final_response)
                    self._emit("info", text=final_response)
                    self._add_assistant(_with_notices(response, final_response))
                    self._last_run_ok = False
                    break

                if not calls:
                    should_break, fr = self._handle_no_tool_calls(
                        response, interactive, st)
                    if should_break:
                        final_response = fr
                        break
                    continue

                # ---- there are tool calls --------------------------------
                # Recorded before execution: the escalation ladder asks whether
                # the model can produce a CALL at all, not whether it succeeded.
                st.tool_calls_made += len(calls)

                # A loosely written call is run, or offered for approval, only
                # once the model was asked whether it meant it.
                self._add_assistant(response)
                loose = self._loose_calls_to_check(calls)
                meant = self._ask_if_calls_meant(loose) if loose else None
                request = getattr(self, "_last_user_request", "") or ""
                if meant is False:
                    st.example_replies += 1
                change_asked = demands_action(request) and asks_for_change(request)
                if (meant is False
                        and ((len(loose) == len(calls) and not change_asked)
                             or (st.example_replies > 1
                                 and all(c.lenient for c in calls)))
                        and not self._change_still_owed(request, st)):
                    # Every call in the reply was an example, or this is the
                    # second such reply to a request for a change or with reads
                    # beside it: the reply is the answer, judged and recorded like
                    # any other final reply.
                    self._messages.pop()
                    self._show_examples(response, parsed_calls, interactive)
                    should_break, fr = self._handle_no_tool_calls(
                        response, interactive, st, examples=True)
                    if should_break:
                        final_response = fr
                        break
                    continue
                st.possible_write = st.possible_write or any(
                    _may_write_files(c, self.cwd) for c in calls
                    if meant is not False or not any(c is x for x in loose))

                # Show the non-tool-call text parts when not already streamed
                segments = split_response(response, parsed_calls)
                if interactive and not getattr(self, "_streamed_to_terminal", False):
                    for seg in segments:
                        if isinstance(seg, str) and seg.strip():
                            console.print(safe_markup(seg.strip()))

                # The event-sink (GUI) surface streamed the RAW response live,
                # before parse_tool_calls could know which spans were real calls,
                # and its live hider (context.py) only recognises the
                # <tool_call>/<|tool_call> wrappers. `calls`/`segments` above are
                # the source of truth for what was consumed as a call, so re-send
                # the real leftover text and the GUI replaces what it streamed,
                # for every call shape. A no-op when nothing leaked.
                self._emit("assistant_text",
                          text="".join(seg for seg in segments if isinstance(seg, str)))

                # Execute tools - run non-destructive batches in parallel
                self._loose_check = (loose, meant) if loose else None
                writes_before = self._write_total()
                try:
                    result_blocks = self._execute_tools(calls, interactive=interactive)
                finally:
                    self._loose_check = None
                st.plan_submitted = st.plan_submitted or any(
                    c.name == "exit_plan_mode" for c in calls)
                if invented:
                    result_blocks.append(
                        "[invented result] Your reply went on with a <tool_result> "
                        "you wrote yourself. It was cut off there and dropped: only "
                        "the results above are real. Make a call, stop, and read its "
                        "real result before the next step.")

                # parse_tool_calls silently drops a tool-call-shaped block whose
                # body could not be recovered, and the only downstream check
                # (_handle_no_tool_calls) fires solely when NO call parsed.
                # `segments` has the successfully-parsed call spans excised, so
                # anything tool-call-shaped left in it is what failed to parse:
                # tell the model that part was NOT run.
                #
                # Capped at _MAX_TOOL_REPAIRS: the notice's own example text is
                # itself tool-call-shaped, so a model echoing it back would
                # re-trigger the notice every turn.
                prose = "".join(seg for seg in segments if isinstance(seg, str))
                leftover = strip_orphan_closers(prose)
                cut_off = unfinished_call(prose, tool_names)
                as_command = (demands_action(request)
                              and names_a_tool_as_command(leftover, tool_names))
                if cut_off and st.partial_notice_count < _MAX_TOOL_REPAIRS:
                    st.partial_notice_count += 1
                    result_blocks.append(self._cut_off_note(cut_off, interactive))
                elif as_command or looks_like_tool_attempt(leftover, tool_names):
                    if st.partial_notice_count < _MAX_TOOL_REPAIRS:
                        st.partial_notice_count += 1
                        if interactive:
                            _agent.print_warning(
                                "  Part of the reply looked like a tool call but "
                                "could not be read, so it was not run. The model "
                                "was told.")
                        result_blocks.append(
                            "[tool-call format] Part of this response looked "
                            "like another tool call, but it could not be "
                            "parsed (the JSON body was likely malformed) - "
                            "it was NOT run, unlike the call(s) above. If you "
                            "still need it, re-emit just that one call in "
                            'the exact <tool_call>\n{"name": "TOOL_NAME", '
                            '"args": {...}}\n</tool_call> format.'
                        )
                    elif not st.partial_notice_cap_announced:
                        # The cap bounds repetition, not visibility: one final
                        # notice naming the change, then a debug trace for every
                        # further occurrence.
                        st.partial_notice_cap_announced = True
                        result_blocks.append(
                            "[tool-call format] Another part of this response "
                            "looked like an unparseable tool call, same as "
                            "the earlier notice(s) this task - further "
                            "occurrences will not be reported individually "
                            "from here on, but each one is still a call that "
                            "did NOT run."
                        )
                        from localm.debuglog import logger
                        logger.debug(
                            "partial tool-call parse failures capped at %d "
                            "notices this task; further drops logged only",
                            _MAX_TOOL_REPAIRS)
                    else:
                        from localm.debuglog import logger
                        logger.debug(
                            "tool-call parse likely dropped a call (leftover "
                            "text still looks like an attempt); notice cap "
                            "already reached, not reported to the model "
                            "this turn")

                tests_fail = (self._consecutive_errors.get("run_tests", 0) > 0
                              and self._write_total() == writes_before
                              and demands_action(request) and asks_for_change(request))
                if (st.code_reminded < 2 and _has_file_code_block(leftover)
                        and (tests_fail or self._change_still_owed(request, st))):
                    st.code_reminded += 1
                    if st.code_reminded > 1 and as_command and self.can_force_tool_calls():
                        self._force_tool_grammar = True
                    result_blocks.append(
                        "[nothing written] The code block in your reply changed no "
                        "file: only a tool call does. If it belongs in the project, "
                        "write it now with write_file or edit_file.")

                if duplicate_calls:
                    result_blocks.append(
                        f"[duplicate calls] {duplicate_calls} call(s) in this "
                        "response repeated an earlier call in it with the same "
                        "arguments; each distinct call was run once.")

                # Feed all results back as a user message, compressing large
                # outputs when the context is filling up
                result_blocks = self._compress_results(result_blocks)
                combined = compose_join("\n\n", result_blocks)
                self._add_user(combined)

                breaker_msg = self._check_post_batch_breakers()
                if breaker_msg is not None:
                    final_response = breaker_msg
                    break

            else:
                msg = f"[max_turns={self.max_turns} reached]"
                print_warning(msg)
                self._last_run_ok = False
                summary = self._closing_summary(interactive, st)
                final_response = (f"{summary}\n\n{msg}{self._grounding_footer()}"
                                  if summary else msg)
                if summary and not interactive and self.on_event is None:
                    print_assistant_response(final_response, name=self.name)

        except KeyboardInterrupt:
            if interactive and self._messages:
                if self.mode == SessionMode.PRIVACY:
                    if os.path.isfile(self._checkpoint_path):
                        print_info(
                            "(interrupted - privacy mode, this turn was not saved; "
                            "/resume returns to the last saved point.)"
                        )
                    else:
                        print_info(
                            "(interrupted - privacy mode, no checkpoint saved; "
                            "/resume is unavailable.)"
                        )
                elif self.save_checkpoint():
                    print_info(
                        "(interrupted - progress saved. "
                        "Type /resume to continue or start a new task.)"
                    )
                else:
                    print_warning(
                        "(interrupted - progress could not be saved; "
                        "/resume will not return to this point.)"
                    )
            raise

        else:
            if not self._last_run_ok and not self._user_stopped:
                self._record_unanswered_end(final_response)
            # Clean finish - discard any stale checkpoint
            self.clear_checkpoint()

        finally:
            # Auto-run from an approved plan lasts for this run only.
            self._end_plan_autorun()
            # Fold this run's outcome into the session-level flag before the next
            # run re-arms the per-run one, so every failure path (max_turns,
            # either circuit breaker) is captured in one place. A user stop is
            # excluded: _last_run_ok and _user_stopped are per-run and re-armed
            # above, while _had_any_failure is session-level and cleared only by
            # reset().
            if not self._last_run_ok and not self._user_stopped:
                self._had_any_failure = True

        return final_response

    def _show_examples(self, response: str, calls: list, interactive: bool) -> None:
        """Show a reply whose calls are all examples whole: the live display
        hides a call."""
        if self.on_event is not None:
            self._emit("assistant_text", text=response)
        elif interactive:
            if getattr(self, "_streamed_to_terminal", False):
                parts = [response[c.start:c.end].strip() for c in calls]
            else:
                parts = [response.strip()]
            for part in parts:
                console.print(safe_markup(part))
            print_info("(an example, as the model said: nothing was run)")

    def _remind_of_budget(self, task_turns: int, st) -> None:
        """Past the turn budget, tell the model how many turns are left: once
        when half of the turns after the budget are used, and once when three
        are left."""
        left = self.max_turns - task_turns
        over = max(1, (self.max_turns - self.turn_budget) // 2)
        due = (2 if left <= 3 else 1 if left <= over else 0)
        if due <= st.budget_reminded or left < 0:
            return
        st.budget_reminded = due
        self._emit("info", text=f"{left} turn(s) left of {self.max_turns}")
        self._add_user(
            f"[turn budget] {left} more turn(s) after this one, then this run "
            "ends. Stop collecting. Write what you have so far with your tools "
            "now, then give your final answer: what is done, what is not done, "
            "and what you did not check.")

    def _closing_summary(self, interactive: bool, st) -> str:
        """The model's own account of a run that used up its turns, or "".

        One more reply after the last turn, asked for in plain text; any tool
        call in it is dropped and nothing in it runs. Not asked for when the
        run made no tool call or was stopped."""
        if not st.tool_calls_made or self.stop_pending:
            return ""
        self._add_user(
            f"[run ending] All {self.max_turns} turns of this run are used. No "
            "tool call will run any more. In plain text and in a few sentences: "
            "what you did, what is not done, and what you did not check.")
        try:
            reply = self._call_llm(self._build_messages(), interactive=interactive)
        except Exception as e:
            from localm.debuglog import logger
            logger.warning("closing summary of a run that used its turns failed: %r", e)
            reply = ""
        if not isinstance(reply, str):
            reply = ""
        tool_names = set(_agent.TOOL_REGISTRY) - self.disabled_tools
        _calls, text, _malformed = strip_tool_calls(reply, tool_names=tool_names)
        text = text.strip()
        if not text or looks_like_tool_attempt(text, tool_names):
            if str(self._messages[-1].get("content", "")).startswith("[run ending]"):
                self._messages.pop()
            return ""
        return text

    def _record_unanswered_end(self, final_response: str) -> None:
        """Put the end of a failed run into the conversation when the model's
        last reply is not there as its end: a circuit breaker or the turn limit
        after tool results, or a closing summary. The saved transcript and a
        resumed session then end with how the run ended."""
        if self._messages and self._messages[-1].get("role") == "user":
            self._add_assistant(final_response)

    def _raise_reply_limit(self) -> int:
        """Double this run's reply limit, once, when the limit in force was
        filled in by default. Returns the limit in force now, 0 when the run
        has none of its own."""
        current = self.gen_kwargs.get("max_tokens")
        if not isinstance(current, int) or isinstance(current, bool):
            return 0
        if self._reply_limit is None and self._max_tokens_explicit is False:
            self._reply_limit = current * 2
        return max(current, self._reply_limit or 0)

    def _cut_off_note(self, name: str, interactive: bool) -> str:
        """Tell the model, and the person, that the reply ended in the middle of
        a call to *name*."""
        before = self._reply_limit or self.gen_kwargs.get("max_tokens")
        now = self._raise_reply_limit()
        limit = (f" One reply can be at most {before} tokens long."
                 if isinstance(before, int) else "")
        raised = ""
        if now and isinstance(before, int) and now > before:
            raised = f" The limit is {now} tokens from here on."
        self._audit.notice("call_cut_off",
                           f"the reply ended in the middle of a {name} call")
        self._emit("info", text=(
            f"the reply ended in the middle of a {name} call, so it was not run"))
        if interactive:
            _agent.print_warning(
                f"  The reply ended in the middle of a {name} call, so it was "
                "not run. The model was told.")
        return ("[tool-call format] Your reply ended in the middle of the "
                f"{name} call, before it was complete, so that call was NOT run."
                f"{limit}{raised} Do not send the same long call again: do it in "
                "smaller steps, each one a complete call. For a long file, write "
                "the first part with write_file, then add each next part with "
                "edit_file.")

    def _task_turns_left(self) -> bool:
        """True while the running task has used fewer than ``max_turns`` turns.

        Counts from ``_task_start_turns``, the session turn count when the task
        started, so turns earlier tasks in the session used do not count."""
        return self._turns - self._task_start_turns < self.max_turns

    def _handle_no_tool_calls(self, response, interactive, st,
                              examples: bool = False) -> tuple[bool, str]:
        """Handle a turn that produced no tool calls (split out of _loop).

        Runs, in order: the harness-run exit-code oracle (when a verify command is
        configured), the one-shot self-verification nudge, the tool-call repair
        re-prompt (capped), the give-up surface for an unparseable attempt, the
        one-shot pre-done reviewer pass, and finally accepting the response as the
        final answer. ``st`` carries the per-task one-shot flags (verify_nudged,
        repair_count, review_done, verify_retries/verify_settled) so they persist
        across turns. With *examples* every call in the response is an example, so
        none of it is read as a broken attempt at a call.

        Returns ``(should_break, final_response)``: ``(False, "")`` to continue the
        loop, ``(True, text)`` to end it with that final response."""
        # Live attribute lookup, so a patched agent.TOOL_REGISTRY is honoured;
        # disabled_tools can differ per agent.
        tool_names = set(_agent.TOOL_REGISTRY) - self.disabled_tools

        # The exit-code oracle runs FIRST among the finish gates, before the
        # self-graded nudge and the reviewer, so a failing check goes straight
        # back as a fix request. Skipped for a response that only looks like a
        # broken tool call - the repair turn below handles that.
        attempt = not examples and looks_like_tool_attempt(response, tool_names)
        if not attempt:
            gated = self._run_verify_gate(response, interactive, st)
            if gated is not None:
                return gated

        request = getattr(self, "_last_user_request", "") or ""
        invented = (-1 if examples or not demands_action(request)
                    else invented_result_without_call(response))
        if invented >= 0 and st.invented_count < 2 and self._task_turns_left():
            return self._invented_result_nudge(response[:invented], interactive, st)

        # Self-verification: nudge the agent to check its work while code changes
        # sit unverified. Fires at most once per task.
        if (
            self.self_verify
            and not st.verify_nudged
            and self._unverified_writes
            and not self.plan_mode
            and self._task_turns_left()
        ):
            st.verify_nudged = True
            files = ", ".join(sorted(self._unverified_writes))
            self._add_assistant(response)
            # Name the project's real check when one is known, so the nudge asks
            # for a command run rather than a re-read.
            if self.verify_cmd is not None:
                from ..verify import command_text
                how = (f"run `{command_text(self.verify_cmd)}` (via run_shell, "
                       "or run_tests if it is the project's test command)")
            elif "run_tests" in self._always_denied():
                how = "re-read the changed files to check for mistakes"
            else:
                how = ("run run_tests if this project has a test suite, "
                       "otherwise re-read the changed files to check for mistakes")
            self._add_user(
                f"[self-verification] You changed code files ({files}) "
                "but have not verified them. Before giving your final "
                f"answer: {how}. Then give your final answer."
            )
            if interactive:
                print_info(
                    "(self-verification: asking agent to verify "
                    f"changes to {files})"
                )
            return (False, "")

        # Repair turn: the response looks like a tool call that failed to parse
        # (a marker/fence, or a name+args object the lenient parser could not
        # recover). Re-prompt once with the exact format.
        # A call the reply ends in the middle of has its own count: it needs
        # smaller steps, not another format.
        cut_off = unfinished_call(response, tool_names) if attempt else ""
        if cut_off and st.cut_off_count < _MAX_TOOL_REPAIRS and self._task_turns_left():
            st.cut_off_count += 1
            self._add_assistant(response)
            self._add_user(self._cut_off_note(cut_off, interactive))
            return (False, "")
        if (
            not cut_off
            and st.repair_count < _MAX_TOOL_REPAIRS
            and self._task_turns_left()
            and (attempt
                 or (not examples and invented < 0
                     and demands_action(request)
                     and names_a_tool_as_command(response, tool_names)))
        ):
            st.repair_count += 1
            self._add_assistant(response)
            if st.repair_count > 1 and self.can_force_tool_calls() and (
                    self._change_still_owed(request, st)
                    or self._tool_demanded(request, st)):
                self._force_tool_grammar = True
            self._add_user(
                "[tool-call format] That looked like a tool call, "
                "but I could not parse it. Re-emit it in EXACTLY "
                "this format and nothing else:\n"
                "<tool_call>\n"
                '{"name": "TOOL_NAME", "args": {"PARAM": "VALUE"}}\n'
                "</tool_call>\n"
                "The args must be valid JSON: use a single double-quoted "
                'string with \\n escapes for multi-line "content" - NOT a '
                "Python triple-quoted string. Use one of the available "
                "tools by its exact name. If you did NOT mean to call a "
                "tool, ignore this and give your final answer as plain text."
            )
            if interactive:
                print_info(
                    "(re-prompting: tool call could not be parsed)"
                )
            return (False, "")

        # Still looks like an unparseable tool call after the repair attempts:
        # surface the raw attempt instead of finalising a hidden <tool_call>
        # block, which the streaming display hides.
        if attempt:
            self._emit("info", text=(
                "the model tried to call a tool but emitted invalid output "
                f"{st.repair_count + st.cut_off_count + 1} times - surfacing its "
                "raw attempt (nothing was run or written)"))
            notice = (
                f"[I tried to call {cut_off}, but each attempt ended before the "
                "call was complete, so it was not run. My last attempt was:]"
                if cut_off else
                "[I tried to call a tool but could not produce valid "
                "tool-call JSON after several attempts, so nothing was run "
                "or written. My raw attempt was:]"
            ) + "\n\n" + response
            if not interactive and self.on_event is None:
                print_assistant_response(notice, name=self.name)
            self._add_assistant(response)
            self._last_run_ok = False
            return (True, notice)

        # Zero-attempt escalation: the model produced nothing tool-shaped on a
        # request that needs a tool.
        if self._plan_owed(request, response, st):
            return self._plan_not_submitted(response, interactive, st)
        offered = st.nocall_escalation >= 1 or st.code_nudged
        marker = _RE_NO_TOOL_NEEDED.match(response)
        if marker is not None and (self._change_still_owed(request, st)
                                   or self._tool_demanded(request, st)):
            # The escape does not answer a request to change the project while
            # nothing was written, nor one that names a tool or says to run
            # something while no tool ran: the marker is dropped and the reply
            # is judged as written. See
            # test_the_escape_does_not_answer_a_change_request.
            response = response[marker.end():] or st.nocall_response
            marker = None
        declined = marker if offered else None
        if declined is not None and (not response[declined.end():].strip()
                                     or _echoes_harness_prompt(response[declined.end():])):
            # The marker with no answer after it: an imperative request keeps
            # climbing the ladder, anything else is answered by the reply the
            # nudge was sent for.
            if demands_action(request) and not st.code_nudged:
                declined = None
            else:
                st.nocall_declined = True
                response = st.nocall_response
        elif declined is not None:
            st.nocall_declined = True
            response = response[declined.end():]
        if declined is None and invented < 0:
            escalated = self._escalate_no_tool_attempt(response, interactive, st)
            if escalated is not None:
                return escalated

        unfinished = ""
        if invented >= 0:
            unfinished = ("\n\n[not finished: this reply shows a tool result the "
                          "model wrote itself. No tool produced it, so do not rely "
                          "on it.]")
        elif not st.nocall_declined:
            gated = self._unfinished_work_gate(response, request, interactive, st)
            if isinstance(gated, tuple):
                return gated
            unfinished = gated or ""

        # Code shown for a workspace file that defines names the file does not
        # contain is invented file content, whatever the reply says it read.
        # Checked against the file on disk, once per task, while turns remain.
        unfounded = unfounded_code(response, self.cwd)
        where = "; ".join(f"{', '.join(names)} not found in {path}"
                          for path, names in unfounded)
        if unfounded and not st.unfounded_checked and self._task_turns_left():
            st.unfounded_checked = True
            self._audit.notice("unfounded_code", where)
            self._add_assistant(response)
            self._add_user(
                "[unverified code] The code in your reply does not match the "
                f"files it names: {where}. It is not those files' content. Read "
                "the part you need first (read_file with offset and limit, or "
                "grep for a name) and answer from what the file actually "
                "contains. If the code is new code you are proposing, say so "
                "plainly instead of presenting it as the file's content."
            )
            if interactive:
                print_info("(unverified code: asking the agent to read the file)")
            self._emit("info", text=(
                f"the reply showed code that is not in the file it names ({where}) "
                "- asking the agent to read the file"))
            return (False, "")

        # Pre-done review: a reviewer model checks the cumulative diff and any
        # blocking issues go back for one more fix pass. Fires at most once per
        # loop, only when there is a real diff and turns remain. Fail-open: a
        # reviewer error never blocks the answer.
        if (
            self._reviewer is not None
            and not st.review_done
            and not self.plan_mode
            and self._task_turns_left()
        ):
            st.review_done = True
            diff = self.session_diff()
            if diff.strip():
                feedback = self._run_pre_done_review(diff)
                if feedback:
                    self._add_assistant(response)
                    self._add_user(feedback)
                    self._emit("info", text=(
                        "self-review: the reviewer flagged issues - "
                        "asking the agent to address them"))
                    if interactive:
                        print_info(
                            "(self-review: reviewer flagged issues - "
                            "feeding them back)")
                    return (False, "")

        # No tool calls, so this is the final answer; _grounding_footer grounds it
        # in what the session actually recorded.
        # Rung 3, and only here: the ladder ran and the model still never produced
        # a call. Reached when the rungs are exhausted, when forcing is
        # unavailable, or when turns ran out mid-ladder.
        enforcement = ""
        if (st.nocall_escalation and not st.nocall_declined
                and not self._used_tools_this_task(st)
                and demands_action(getattr(self, "_last_user_request", "") or "")):
            why = ("" if self.can_force_tool_calls() else
                   " Constrained sampling, which would have forced one, is not "
                   "available here (this backend cannot enforce a grammar, or "
                   "coder_tool_grammar is off in config).")
            enforcement = (
                "\n\n[tool use not achieved: this model was asked "
                f"{st.nocall_escalation} more time(s) to call a tool and did not, "
                f"so nothing was run or written.{why}]")
            if self._tool_demanded(getattr(self, "_last_user_request", "") or "", st):
                self._last_run_ok = False
            self._audit.notice(
                "no_tool_call",
                f"escalation exhausted after {st.nocall_escalation} attempt(s); "
                "no tool call was produced")
            self._emit("info", text=(
                "the model did not call any tool despite being asked again "
                "- nothing was run or written"))

        unverified = ""
        if unfounded:
            unverified = f"\n\n[unverified code: {where}]"
            self._emit("info", text=(
                f"unverified code: {where} - the code in this answer is not "
                "the content of the file it names"))
        footer = self._grounding_footer()
        sensitive = self._sensitive_changes_notice()
        if unfinished:
            self._last_run_ok = False
        final_text = (response + enforcement + unfinished + unverified
                      + (footer or "") + sensitive)
        if interactive and self.on_event is None:
            print_warning = _agent.print_warning
            if enforcement.strip():
                print_warning(enforcement.strip())
            if unfinished.strip():
                print_warning(unfinished.strip())
            if unverified.strip():
                print_warning(unverified.strip())
            task_files = self._files_changed_this_task(st)
            task_sensitive = self._sensitive_changes_notice(task_files) if task_files else ""
            if task_sensitive.strip():
                print_warning(task_sensitive.strip())
            if st.tool_calls_made > 0:
                print_task_footer(
                    turns_used=max(1, self._turns - getattr(st, "start_turns", self._turns - 1)),
                    files_changed=[c["path"] for c in task_files],
                    verify_status=self._last_verify_state,
                    ctx_ratio=self._fill_ratio(),
                )
        elif not interactive and self.on_event is None:
            print_assistant_response(final_text, name=self.name)
        self._add_assistant(_with_notices(response, enforcement, unfinished, unverified))
        return (True, final_text)

    def _invented_result_nudge(self, kept, interactive, st) -> tuple[bool, str]:
        """Answer a reply that shows the result of a command nothing ran. Only
        *kept*, the part before that result, stays in the conversation. The
        second time the next turn is held to a tool call where the backend can."""
        st.invented_count += 1
        forced = st.invented_count > 1 and self.can_force_tool_calls()
        if forced:
            self._force_tool_grammar = True
        self._audit.notice("invented_result", "the reply showed a result of a "
                           "command that no tool ran; cut off there")
        self._add_assistant(kept.strip() or "(a made-up tool result, dropped)")
        self._add_user(
            "[invented result] Nothing ran: the <tool_result> in your reply was "
            "written by you, so it is not real and was dropped. A code block runs "
            "nothing. Run what you showed with a tool call now, stop, and read its "
            "real result."
            + (" This turn is constrained: the only output accepted is a tool call."
               if forced else ""))
        note = "the reply showed a tool result nothing produced - asking for a real call"
        if interactive:
            print_info(f"({note})")
        self._emit("info", text=note)
        return (False, "")

    def _unfinished_work_gate(self, response, request, interactive, st):
        """Hold back a final answer that leaves the requested work undone.

        Four cases:

        * in a run nobody answers, a short question while a change is still owed:
          the agent is told once to find out for itself, and a second question
          stands as the answer;
        * the reply to the self-verification request is empty (or only the
          ``[no tool needed]`` marker) while changed code files are still
          unverified: the run ends as not finished, naming those files;
        * a change the request asked for is still owed (see
          :meth:`_change_still_owed`) and the reply is not a short clarifying
          question: the agent is told so once (worded for a
          reply that shows code when it does), unless the no-tool-call ladder
          already ran this task; a final answer with the change still owed after
          that ends the run as not finished;
        * the reply's last line announces a next step ("Let me run the tests.")
          after tools were used this task: nudged once, and ending on an
          announcement again ends the run as not finished.

        Returns ``(False, "")`` after a nudge (the loop continues), a note for
        the final answer when the run ends with the case still open (it is then
        reported as not finished), or ``""``."""
        turns_left = self._task_turns_left() and not self.plan_mode
        if (self._change_still_owed(request, st) and _asks_a_question(response)
                and turns_left and not interactive and self.confirm_handler is None
                and not st.question_answered):
            st.question_answered = True
            self._audit.notice("question_unattended",
                               "the reply asked the user something in a run "
                               "nobody answers")
            self._add_assistant(response)
            self._add_user(
                "[unattended run] Nobody can answer in this run. Find out what you "
                "can with your tools (list_dir, grep, read_file, run_tests), decide "
                "the rest yourself, and do the task. If it cannot be done without an "
                "answer, say in your final answer exactly what is missing.")
            self._emit("info", text=(
                "the model asked a question in a run nobody answers - asking it to "
                "find out for itself"))
            return (False, "")
        if self._change_still_owed(request, st) and not _asks_a_question(response):
            escalated = st.nocall_escalation and not self._used_tools_this_task(st)
            if (st.code_nudged and not st.code_forced and turns_left
                    and "```" in response and self.can_force_tool_calls()):
                # It shows the change again instead of making it.
                st.code_forced = True
                self._force_tool_grammar = True
                self._audit.notice("change_forced",
                                   "the change was shown again instead of made - "
                                   "the next turn is constrained to a tool call")
                self._add_assistant(response)
                self._add_user(
                    "[nothing written] Still nothing on disk changed. This turn is "
                    "constrained: the only output accepted is a tool call. Make the "
                    "change you showed.")
                if interactive:
                    print_info("(nothing written: forcing a tool call via constrained sampling)")
                self._emit("info", text=(
                    "the change was shown again instead of made - forcing a tool "
                    "call via constrained sampling"))
                return (False, "")
            if st.code_nudged or escalated or not turns_left:
                return ("\n\n[not finished: nothing was written to any file in this "
                        "task, so the change the request asked for was not made]")
            st.code_nudged = True
            st.nocall_response = response
            if _has_file_code_block(response):
                self._audit.notice("code_not_written",
                                   "reply carried code but nothing was written")
                prompt = ("[code not written] Your reply contains code, but no file "
                          "was written or edited in this task, so nothing on disk "
                          "changed. If the code belongs in the project, write it now "
                          "with a tool call: write_file for a new file or a complete "
                          "rewrite, edit_file to change part of a file.")
                info = "the reply showed code but wrote no file - asking the agent to write it"
            else:
                self._audit.notice("change_not_made",
                                   "change request answered with nothing written")
                prompt = ("[nothing written] This request asks for a change to the "
                          "project, but nothing has been written in this task, so "
                          "nothing on disk changed. Make the change now with a tool "
                          "call: write_file for a new file or a complete rewrite, "
                          "edit_file to change part of a file.")
                info = "the request asked for a change and nothing was written - asking the agent to make it"
            self._add_assistant(response)
            self._add_user(prompt)
            if interactive:
                print_info("(nothing written: asking the agent to make the change)")
            self._emit("info", text=info)
            return (False, "")
        if st.verify_nudged and self._unverified_writes and (
                not response.strip() or _RE_NO_TOOL_NEEDED.fullmatch(response.strip())):
            files = ", ".join(sorted(self._unverified_writes))
            return (f"\n\n[not verified: the reply to the request to verify the "
                    f"changed code files ({files}) was empty, so they were not "
                    "checked]")
        announced = _announced_step(response)
        if st.announce_nudged and not announced and (
                not response.strip() or _RE_NO_TOOL_NEEDED.fullmatch(response.strip())):
            # A reply with nothing in it after the nudge leaves the announced
            # step untaken.
            announced = st.announced_step
        if announced and self._used_tools_this_task(st):
            if not st.announce_nudged and turns_left:
                st.announce_nudged = True
                st.announced_step = announced
                st.nocall_response = response
                why = (f'Your reply ends with "{announced}", but it made no tool '
                       "call, so that step did not happen.")
                self._audit.notice("announced_step_not_taken", why)
                self._add_assistant(response)
                self._add_user(
                    f"[unfinished] {why} Do it now with a tool call. If the task "
                    "is already complete, give your final summary without "
                    "announcing further steps.")
                if interactive:
                    print_info("(unfinished: asking the agent to take the step it announced)")
                self._emit("info", text=(
                    "the reply announced a next step without taking it - asking "
                    "the agent to take it"))
                return (False, "")
            if st.announce_nudged:
                return (f'\n\n[not finished: the run ended on "{announced}" and '
                        "that step was never taken]")
        return ""

    def _change_still_owed(self, request: str, st) -> bool:
        """True when *request* tells the agent to change the project (an
        imperative with a change verb), the session can write files and is not
        in plan mode, and nothing that can write has run this task: no write
        the tracker saw, no unverified changed file, no patch-mode capture, no
        call :func:`_may_write_files` counts as a possible write, no dry-run
        session (its writes are shown, never made) and no call the user refused
        at an approval prompt (their answer stands)."""
        if self.plan_mode:
            return False
        if not {"write_file", "edit_file", "patch_file"} - self.disabled_tools:
            return False
        if not (demands_action(request) and asks_for_change(request)):
            return False
        return not (self._write_total() > st.writes_at_start or self.patch_mode
                    or self.dry_run or st.possible_write
                    or bool(self._unverified_writes)
                    or self._user_declines > st.declines_at_start)

    def _tool_demanded(self, request: str, st) -> bool:
        """True when *request* names an available tool or tells the agent to
        run something, and no tool has run this task: only a tool call answers
        it, so the ``[no tool needed]`` escape does not."""
        if self._used_tools_this_task(st) or not demands_action(request):
            return False
        if _RE_RUN_REQUEST.search(request):
            return True
        names = set(_agent.TOOL_REGISTRY) - self.disabled_tools
        return any(w in names for w in _RE_TOOL_WORD.findall(request.lower()))

    def _plan_owed(self, request: str, response: str, st) -> bool:
        """True in plan mode when the request asks for a change, the plan was
        never submitted with exit_plan_mode, and the reply is not a question."""
        return (self.plan_mode and "exit_plan_mode" not in self.disabled_tools
                and not st.plan_submitted
                and demands_action(request) and asks_for_change(request)
                and not _asks_a_question(response))

    def _plan_not_submitted(self, response, interactive, st):
        """A reply with no tool call in plan mode is not a plan the user can
        approve: ask for exit_plan_mode twice (the second turn constrained to a
        tool call where the backend can), then end the run as not finished."""
        if st.plan_nudges < 2 and self._task_turns_left():
            st.plan_nudges += 1
            self._audit.notice("plan_not_submitted",
                               "plan mode reply without exit_plan_mode")
            self._add_assistant(response)
            self._add_user(
                "[plan not submitted] You are in plan mode and this request asks "
                "for a change. A plan written as a reply never reaches the user's "
                "approval prompt, so nothing can go ahead. Call exit_plan_mode "
                "now with the whole plan as markdown in its plan argument "
                "(read-only tools first if you still need to look). If you need "
                "an answer from the user first, ask one short question instead.")
            if st.plan_nudges == 2 and self.can_force_tool_calls():
                self._force_tool_grammar = True
            if interactive:
                print_info("(plan mode: asking the agent to submit the plan)")
            self._emit("info", text=(
                "plan mode: the reply was not submitted with exit_plan_mode - "
                "asking the agent to submit it"))
            return (False, "")
        self._last_run_ok = False
        self._emit("info", text="plan mode: the plan was never submitted")
        notice = ("\n\n[not finished: plan mode needs the plan submitted with "
                  "exit_plan_mode, and it never was, so you were not asked to approve "
                  "it]")
        self._add_assistant(_with_notices(response, notice))
        return (True, response + notice + self._grounding_footer())

    def _used_tools_this_task(self, st) -> bool:
        """Has the model demonstrated, THIS TASK, that it can drive a tool?

        Three independent pieces of evidence, any one of which settles it. Only
        the first is about parsing; the other two are about what the harness
        actually recorded happening, which is the same grounding rule
        _grounding_footer follows - never the model's own account of itself.

        Both artifact checks are needed, and neither subsumes the other: a
        write is recorded in _unverified_writes the moment it lands, while
        _write_total() is the cumulative counter, snapshotted at task start so
        an EARLIER task's writes in a long REPL session cannot be mistaken for
        this one's. A task that has written something is self-evidently not a
        task in which the model refuses to act, and escalating at it would
        nag a model that is working."""
        return bool(st.tool_calls_made
                    or self._unverified_writes
                    or self._write_total() > st.writes_at_start)

    def _escalate_no_tool_attempt(self, response, interactive, st):
        """Escalate a turn that produced NO tool call and no attempt at one, on a
        request that needs a tool.

        Returns None to fall through to the remaining gates, or the same
        ``(should_break, final_response)`` pair the caller propagates.

        THE POINT IS TO MAKE THE CALL HAPPEN, NOT TO REPORT THAT IT DID NOT.
        Reporting already existed (_grounding_footer's "no files changed") and
        it left the user with a model that never touched a tool across six
        turns. The rungs, in order:

          1. Re-prompt with the exact format block - the same treatment the
             MALFORMED case has always had, triggered by ABSENCE instead. Some
             models simply never saw the wrapper in a prompt this long and
             produce a correct call the moment it is shown to them again.
          2. Re-run the turn with the tool-call grammar bound from the FIRST
             token (see context.can_force_tool_calls). At this point the
             sampler cannot emit anything except an optional reasoning block
             and a structurally valid call, so the model's willingness stops
             being the deciding factor.
          3. Only once forcing has actually been tried and still failed - or is
             genuinely unavailable on this backend - tell the user. That
             message is a report of FAILED ENFORCEMENT, never a suggestion to
             pick a different model: which model to run is the user's choice
             and this code's job is to make their choice work.

        Rung 1 offers an explicit way out: a reply that begins with
        "[no tool needed]" is accepted as the answer (handled by the caller,
        which strips the marker). Only an imperative request
        (:func:`demands_action`) climbs past rung 1 or is reported as failed
        enforcement. A short plain-text question from the model stands as the
        answer.

        Deliberately NOT gated on the response's wording. Every phrasing-based
        check inherits the unreliability of the self-report it is reading (see
        _grounding_footer); "did the harness parse a call" is an observable
        fact about this turn and "does the request need one" is a fact about
        the user's own text, so neither can be talked past."""
        if self._used_tools_this_task(st):
            return None                      # this model calls tools fine
        request = getattr(self, "_last_user_request", "") or ""
        if not implies_action(request):
            return None                      # a question, not an action
        demanded = demands_action(request)
        if _asks_a_question(response):
            return None                      # a clarifying question is the answer
        if not self._task_turns_left():
            return None                      # no turns left to escalate into
        if st.nocall_escalation >= _MAX_NOCALL_ESCALATIONS:
            return None                      # ladder exhausted; caller reports below
        if st.nocall_escalation >= 1 and not demanded:
            return None                      # rung 1 allowed a prose answer; accept it

        st.nocall_escalation += 1
        rung = st.nocall_escalation

        if rung == 1:
            self._audit.notice(
                "no_tool_call",
                "model produced no tool call and no attempt on an action request "
                "- re-prompting with the tool-call format")
            st.nocall_response = response
            self._add_assistant(response)
            self._add_user(
                "[no tool call] That request needs you to USE a tool - I run the "
                "tools on this machine for you, and nothing was run or written, "
                "so the task is not done. Do not describe the steps, do not hand "
                "back a script for me to run: emit the call itself, in EXACTLY "
                "this format:\n"
                "<tool_call>\n"
                '{"name": "TOOL_NAME", "args": {"PARAM": "VALUE"}}\n'
                "</tool_call>\n"
                "Use one of the available tools by its exact name, with valid "
                "JSON args. Emit one call now and I will run it and give you the "
                "result."
                + ("" if (self._change_still_owed(request, st)
                            or self._tool_demanded(request, st)) else
                   ' If this genuinely needs no tool at all, begin your reply '
                   'with "[no tool needed]" and answer in plain text, and I will '
                   "accept that as your answer.")
            )
            if interactive:
                print_info("(no tool call: re-prompting with the tool-call format)")
            self._emit("info", text=(
                "the model answered without using a tool - asking it again "
                "with the tool-call format"))
            return (False, "")

        # Rung 2: force it at the sampler.
        if not self.can_force_tool_calls():
            self._audit.notice(
                "no_tool_call",
                "re-prompt did not produce a tool call and grammar forcing is "
                "unavailable on this backend/config")
            return None                      # caller's own branch reports it
        self._force_tool_grammar = True
        self._audit.notice(
            "no_tool_call",
            "re-prompt did not produce a tool call - re-running the turn with "
            "the tool-call grammar bound from the first token")
        self._add_assistant(response)
        self._add_user(
            "[no tool call] Still nothing was run. This turn is constrained: "
            "the only output accepted is a tool call. Think first if you need "
            "to, then emit the call."
        )
        if interactive:
            print_info("(no tool call: forcing a tool call via constrained sampling)")
        self._emit("info", text=(
            "the model still did not use a tool - forcing a tool call via "
            "constrained sampling"))
        return (False, "")

    def _grounding_footer(self) -> str:
        """A factual line grounding the final answer in the session's own
        record, appended UNCONDITIONALLY - never gated on what the response
        text itself claims.

        Built from the facts loop.py already tracks for the self-verify nudge
        and the exit-code oracle (changed_files(), _last_verify_state), never
        from the response text, so it cannot be gamed by phrasing: it reports
        only what the harness recorded.
        """
        changed = self.changed_files()
        if changed:
            names = ", ".join(sorted(f["path"] for f in changed))
            parts = [f"{len(changed)} file(s) changed: {names}"]
        else:
            parts = []
        by_commands = self._changed_by_commands({f["path"] for f in changed})
        if by_commands:
            shown = ", ".join(by_commands[:_MAX_RECORD_PATHS])
            more = len(by_commands) - _MAX_RECORD_PATHS
            parts.append(f"{len(by_commands)} file(s) changed by commands: {shown}"
                         + (f" and {more} more" if more > 0 else ""))
        elif self._shell_baseline_captured and self._writing_commands:
            parts.append(f"{self._writing_commands} command(s) ran whose file "
                         "changes are not recorded")
        if not parts:
            parts = ["no files changed"]
        if self._files_read:
            parts.append(f"{len(self._files_read)} file(s) read")
        if self.verify_cmd is not None and self._last_verify_state:
            parts.append(f"verify: {self._last_verify_state}")
        return "\n\n[session record: " + "; ".join(parts) + "]"

    def _changed_by_commands(self, known: set) -> list[str]:
        """Paths a shell command of this session changed, from the git work
        tree before its first command and now: without the *known* paths the
        write tools recorded and without caches a test run leaves. Empty when no
        command ran or the folder is no git work tree."""
        if not self._shell_baseline_captured or self._git_baseline is None:
            return []
        current = self._git_status_map()
        found = []
        for path in sorted(set(current or ()) - self._git_baseline):
            if any(part in _CACHE_DIRS for part in path.split("/")):
                continue
            if path in known or (path.endswith("/")
                                 and any(k.startswith(path) for k in known)):
                continue
            found.append(path)
        return found

    def _files_changed_this_task(self, st) -> list[dict]:
        """changed_files() entries written since this task started."""
        before = getattr(st, "files_at_start", None) or {}
        return [f for f in self.changed_files()
                if f.get("writes", 0) > before.get(f.get("path"), 0)]

    def _sensitive_changes_notice(self, files: list[dict] | None = None) -> str:
        """A footer noting files a passing check cannot vouch for on its own
        (a rewritten test, a loosened CI/lint gate), or ``""`` when none.

        This used to be console-only (runner.py's ``warn_sensitive_changes``,
        reached only via ``finish_agent``), so a GUI session - whose
        ``CoderSession.close()`` never calls ``finish_agent`` at all - and an
        MCP ``run_coder_task`` call - whose caller sees only the returned
        response text, with every console print swallowed - never saw it.
        Folding it into the final answer itself, the same way
        ``_grounding_footer`` does, reaches every surface by construction
        instead of by remembering to wire up each one separately.
        """
        try:
            from ..review_guard import classify_sensitive_changes, render_warning
            message = render_warning(classify_sensitive_changes(
                self.changed_files() if files is None else files))
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("sensitive-changes notice skipped: %r", e)   # See test_classify_sensitive_changes_tolerates_malformed_entries.
            return ""
        if not message:
            return ""
        self._emit("info", text=message)
        return "\n\n" + message

    def _run_pre_done_review(self, diff: str) -> str:
        """Run the pre-done review over *diff* and return the feedback to feed
        back (``""`` when the answer stands).

        Fail-OPEN: a reviewer that crashes or emits garbage never blocks the
        agent. It is not silent either - ``ReviewResult.ok=False`` is surfaced as
        a warning plus an audit entry, distinct from an approval, so the user
        knows the diff went out unchecked. Visibility only; control flow is the
        same in both cases.

        A diff over the reviewer's size cap is truncated before it is sent -
        the reviewer then only ever saw part of the change, so its verdict
        (approved OR blocking) is surfaced with that caveat too, the same way
        as a failed review: printed, emitted, and recorded in the audit trail.
        """
        print_warning = _agent.print_warning  # live: honour a patched agent.print_warning
        sensitive_note = ""
        try:
            from ..review_guard import classify_sensitive_changes, render_reviewer_note
            sensitive_note = render_reviewer_note(
                classify_sensitive_changes(self.changed_files()))
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("sensitive-changes reviewer note skipped: %r", e)   # See test_classify_sensitive_changes_tolerates_malformed_entries.
        if getattr(self, "_interactive", False):
            with ActivitySpinner("Running pre-done code review..."):
                result = self._reviewer.review(diff, self._review_task, sensitive_note=sensitive_note)
        else:
            result = self._reviewer.review(diff, self._review_task, sensitive_note=sensitive_note)
        warning = self._reviewer.failure_warning(result)
        if warning:
            print_warning(warning)
            self._emit("info", text=warning)
            self._audit.notice("review_failed", warning)
        partial = self._reviewer.partial_warning(result)
        if partial:
            print_warning(partial)
            self._emit("info", text=partial)
            self._audit.notice("review_partial", partial)
        return self._reviewer.feedback_for(result)

    def _write_total(self) -> int:
        """Total file writes recorded this session. Compared against a per-task
        snapshot so the verify gate can tell "this task changed something" from
        "an earlier turn in this REPL session did"."""
        return sum(int(f.get("writes", 0)) for f in self._changed_files.values())

    def _run_verify_gate(self, response, interactive, st):
        """The harness-run exit-code oracle at the pre-done boundary.

        This is the same un-gameable check goal mode runs (``cli/goal.py``'s
        ``--until``), reaching the interactive REPL and the GUI, where the only
        finish gates were otherwise self-graded (the verify nudge) or advisory
        (the reviewer's diff opinion). The HARNESS runs the command and reads its
        exit code; the model's own claim of success is not consulted, so it
        cannot declare a premature one.

        Returns None to fall through to the remaining gates, or the
        ``(should_break, final_response)`` pair ``_handle_no_tool_calls`` returns.

        ``verify_cmd`` is set by the CLI's outer ``--until`` loop, an
        interactive/GUI session, or (tools/agents.py's ``_isolated_verify_cmd``)
        a worktree-isolated child - never more than one of these for the same
        Agent instance's own run, so the command is never executed twice per
        iteration. A parent that dispatches an isolated child and ALSO has its
        own verify_cmd runs two independent gates on two independent trees,
        which is by design, not a double-count of one."""
        if self.verify_cmd is None or st.verify_settled or self.plan_mode:
            return None
        # Nothing written since the last passing check means nothing to verify.
        if self._write_total() <= st.verify_checked_at:
            return None

        from .. import verify as _verify
        cmd = self.verify_cmd
        label = _verify.command_text(cmd)
        refusal = self._verify_refusal(label, interactive)
        if refusal is not None:
            st.verify_settled = True
            self._last_verify_state = "inconclusive"
            msg = (f"verification not run: `{label}` runs code the project "
                   f"controls and {refusal} - nothing was actually verified")
            self._emit("info", text=msg)
            _agent.print_warning(msg)
            return None
        self._emit("info", text=f"verification: running `{label}`")
        if interactive:
            print_info(f"(verification: running `{label}`)")
            with ActivitySpinner(f"Verifying with `{label}`..."):
                outcome = _verify.run_verify(cmd, self.cwd)
        else:
            outcome = _verify.run_verify(cmd, self.cwd)
        code, output = outcome
        # Whether the command STARTED is carried on the outcome, never inferred
        # from the exit code.
        did_not_start = _verify.launch_failed(outcome)

        if code == 0:
            # Passing is not terminal: anything written after this point is
            # checked again. Only the inconclusive and exhausted cases settle.
            st.verify_checked_at = self._write_total()
            # A real check just passed, so the self-verification nudge has
            # nothing left to ask for and its writes count as verified.
            st.verify_nudged = True
            self._unverified_writes.clear()
            self._last_verify_state = "passed"
            self._emit("info", text=f"verification passed: `{label}` exited 0")
            if interactive:
                print_success(f"Verification passed: `{label}` exited 0.")
            return None

        if _verify.is_inconclusive(cmd, code, did_not_start):
            # Not a pass and not a fixable failure: the check either could not
            # start or collected nothing, so settle instead of retrying, state
            # that nothing was verified, and record the third state.
            st.verify_settled = True
            self._last_verify_state = "inconclusive"
            msg = (f"verification inconclusive: `{label}` "
                   f"{_verify.inconclusive_reason(cmd, code, did_not_start)} "
                   f"(exit {code}) - nothing was actually verified")
            self._emit("info", text=msg)
            _agent.print_warning(msg)
            return None

        if st.verify_retries < self.verify_max_retries and self._task_turns_left():
            st.verify_retries += 1
            self._add_assistant(response)
            self._add_user(_verify.verify_feedback(cmd, code, output))
            msg = (f"verification failed (exit {code}); asking for a fix "
                   f"({st.verify_retries}/{self.verify_max_retries})")
            self._emit("info", text=msg)
            if interactive:
                _agent.print_warning(f"({msg})")
            return (False, "")

        # Retries exhausted and the check still fails: surface the answer but
        # mark it not-ok and state the failure.
        st.verify_settled = True
        self._last_run_ok = False
        self._last_verify_state = "failed"
        notice = (
            f"\n\n[verification FAILED] `{label}` still exits {code} after "
            f"{self.verify_max_retries} fix attempt(s). This task is NOT verified."
        )
        self._emit("info", text=(
            f"verification FAILED: `{label}` still exits {code} after "
            f"{self.verify_max_retries} fix attempt(s) - reporting failure, "
            "not a false success"))
        _agent.print_warning(notice.strip())
        self._add_assistant(_with_notices(response, notice))
        # The grounding footer is unconditional. _last_verify_state is already
        # "failed" here, so the same call includes it.
        return (True, response + notice + self._grounding_footer()
                + self._sensitive_changes_notice())

    def _check_post_batch_breakers(self) -> str | None:
        """After a tool batch, return a circuit-breaker message (and mark the run
        not-ok + emit it) if a breaker tripped, else None. Split out of _loop."""
        print_warning = _agent.print_warning  # live: honour a patched agent.print_warning
        # Circuit breaker: a tool that keeps failing identically wastes
        # the whole turn budget - stop and hand control back instead.
        if self._abort_streak_tool:
            tool = self._abort_streak_tool
            self._abort_streak_tool = None
            streak = self._consecutive_errors.get(tool, 0)
            final_response = (
                f"[circuit breaker: {tool} failed {streak} times in a "
                "row - stopping so you can take a look instead of "
                "burning more turns. The conversation is intact; "
                "adjust the approach and continue.]"
            )
            print_warning(final_response)
            self._emit("info", text=final_response)
            self._last_run_ok = False
            return final_response

        # No-progress breaker: many tool calls failed in a row across ANY tools.
        if self._abort_no_progress:
            self._abort_no_progress = False
            self._global_error_streak = 0
            final_response = (
                "[circuit breaker: many tool calls failed in a row with no "
                "progress - stopping so you can take a look instead of burning "
                "more turns. The conversation is intact; adjust and continue.]"
            )
            print_warning(final_response)
            self._emit("info", text=final_response)
            self._last_run_ok = False
            return final_response

        return None

    def _result_block(self, call, result) -> str:
        """The <tool_result> XML for a finished tool call, provenance-tagged.

        A result that :func:`provenance.taints_session` names sets
        ``_untrusted_in_context`` on this agent and every agent above it, which
        re-arms confirmation until each one's next user message.

        Results from untrusted (network / MCP) tools are re-framed as
        data-not-instructions with a hardened boundary (provenance.py); trusted
        tools keep the plain frame. The outer <tool_result> tag is preserved
        either way so the rest of the agent (audit / transcript skips) is
        unaffected. When the feature is off, every result uses the plain frame.
        """
        TOOL_REGISTRY = _agent.TOOL_REGISTRY  # live: honour a patched agent.TOOL_REGISTRY
        from ..provenance import build_result_block, is_untrusted_tool, taints_session
        if taints_session(call.name, TOOL_REGISTRY.get(call.name), result):
            agent = self
            while agent is not None:
                agent._untrusted_in_context = True
                agent = getattr(agent, "parent", None)
        if not getattr(self, "_untrusted_provenance", True):
            return result.to_xml(call.name)
        untrusted = is_untrusted_tool(call.name, TOOL_REGISTRY.get(call.name), result)
        return build_result_block(call.name, result, untrusted)

    def _stopped_text(self) -> str:
        """The final response of a run that stopped before finishing."""
        if self.cancelled:
            return f"[stopped: {self.cancel_reason}]"
        return "[stopped by user]"

    def _execute_tools(self, calls: list, interactive: bool) -> list[str]:
        """
        Execute a list of tool calls and return their XML result blocks.

        Groups consecutive non-destructive calls and runs each group in parallel
        with a ``ThreadPoolExecutor``.  Destructive calls are always run alone,
        in order, to avoid unintended interactions (file corruption, overlapping
        shell commands, etc.).  A group runs one call at a time, with
        *interactive* passed on, when a project hook may ask before it runs for
        one of its calls (``_hooks_may_ask``).

        The grouping strategy preserves the original ordering of destructive
        calls relative to non-destructive ones: given [read, read, write, read],
        the two leading reads run in parallel, then the write runs alone, then
        the final read runs alone.  This is conservative but safe.

        A call that ARMS a dispatch-time restriction runs alone too, whatever its
        destructive flag says.  See the segmentation below for why.
        """
        TOOL_REGISTRY = _agent.TOOL_REGISTRY  # live: honour a patched agent.TOOL_REGISTRY
        result_blocks: list[str] = []

        # Split into segments: each segment is (is_destructive, [calls]).
        #
        # A call that ARMS this session's active-skill restriction (use_skill,
        # _SKILL_STATE_TOOLS) is a segment boundary on BOTH sides, so it runs
        # alone despite being non-destructive.
        #
        # Ordering invariant any regrouping must preserve: everything emitted
        # BEFORE the arming call finishes before it runs, and nothing emitted
        # AFTER it starts until the restriction is armed. Narrowing therefore
        # applies FORWARD ONLY.
        #
        # A singleton segment runs serially and so has no batch deadline.
        segments: list[tuple[bool, list]] = []
        extendable = False              # may the last segment take another call?
        for call in calls:
            td = TOOL_REGISTRY.get(call.name)
            destructive = td.destructive if td else True
            solo = (call.name in _SKILL_STATE_TOOLS or call.name in INTERACTION_TOOLS
                    or call.name in _BROWSER_TOOLS)
            if (extendable and not solo
                    and segments and segments[-1][0] == destructive):
                segments[-1][1].append(call)
            else:
                segments.append((destructive, [call]))
            extendable = not solo

        # Non-destructive peers abandoned at a batch deadline, as (future, tool
        # name). The timeout path cancels (a no-op on a running future) and shuts
        # the pool down without joining, so an abandoned thread can still be
        # executing when the next segment starts.
        #
        # Stored on the agent, not in this frame, so it outlives the turn. Ones
        # that have since finished are dropped so the list cannot grow.
        self._abandoned_peers = [(f, n) for f, n in self._abandoned_peers
                                 if not f.done()]
        abandoned = self._abandoned_peers

        after_plan_verdict = False
        for destructive, group in segments:
            if after_plan_verdict:
                # Calls emitted after exit_plan_mode in the same reply wait for
                # the model to read the user's verdict.
                for call in group:
                    result = ToolResult.error(
                        f"{call.name} was not run: it was in the same reply as "
                        "exit_plan_mode. Read the user's verdict on the plan, "
                        "then call it again if it still applies.")
                    if interactive:
                        print_tool_error(call.name, result.output)
                    result_blocks.append(self._result_block(call, result))
                continue
            still_live: list[tuple] = []
            if destructive and abandoned:
                abandoned = self._await_abandoned_peers(abandoned)
                self._abandoned_peers = abandoned
                still_live = abandoned
            if (destructive or len(group) == 1 or self._hooks_may_ask(group)
                    or self._may_need_terminal_confirm(group, interactive)):
                # Serial execution
                for call in group:
                    if still_live:
                        # Refuse rather than run alongside it. The model can call
                        # it again next turn.
                        peers = ", ".join(sorted({n for _f, n in still_live}))
                        result = ToolResult.error(
                            f"{call.name} was not run: it is a destructive tool, so "
                            f"it runs alone, and {peers} is still running after an "
                            f"extra {self._ABANDONED_PEER_GRACE_S}s. Running them "
                            "together would stack more concurrency than this "
                            f"machine can serve. WAIT for {peers} to finish - "
                            "retrying immediately, this turn or the next, is "
                            "refused again for the same reason."
                        )
                        if interactive:
                            print_tool_error(call.name, result.output)
                    else:
                        result = self._execute_tool(call, interactive=interactive)
                    result_blocks.append(self._result_block(call, result))
                    if call.name == "exit_plan_mode":
                        after_plan_verdict = True
            else:
                # Parallel execution for a non-destructive batch. The pool is
                # shut down without waiting, so one hung tool cannot block the
                # batch past the deadline; the stuck thread is abandoned and
                # reported as a timeout.
                ordered: dict[int, str] = {}
                pool = ThreadPoolExecutor(max_workers=min(len(group), 8))
                futures = {
                    pool.submit(self._execute_tool, call, False): (i, call)
                    for i, call in enumerate(group)
                }
                batch_spinner = ActivitySpinner(
                    f"Running {len(group)} tools in parallel..."
                )
                if interactive:
                    batch_spinner.start()
                try:
                    for fut in as_completed(futures,
                                            timeout=self._PARALLEL_BATCH_TIMEOUT_S):
                        i, call = futures[fut]
                        try:
                            result = fut.result()
                        except Exception as exc:
                            result = ToolResult.error(f"Parallel execution error: {exc}")
                        ordered[i] = self._result_block(call, result)
                        if interactive:
                            batch_spinner.stop()
                            print_tool_call(call.name, call.args)
                            print_tool_result(
                                call.name, result, verbose=self.verbose, args=call.args
                            )
                            rem = len(group) - len(ordered)
                            if rem > 0:
                                batch_spinner.start(f"Running {rem} remaining tool(s)...")
                except TimeoutError:
                    if interactive:
                        batch_spinner.stop()
                    for fut, (i, call) in futures.items():
                        if i not in ordered:
                            fut.cancel()
                            if not fut.done():
                                # cancel() cannot stop a RUNNING future, so this
                                # one is still executing. Remember it: a later
                                # destructive segment must not start alongside it.
                                abandoned.append((fut, call.name))
                            result = ToolResult.error(
                                f"{call.name} did not finish within "
                                f"{self._PARALLEL_BATCH_TIMEOUT_S}s (parallel "
                                "batch timeout) - try a narrower target."
                            )
                            ordered[i] = self._result_block(call, result)
                            if interactive:
                                print_tool_error(call.name, result.output)
                finally:
                    if interactive:
                        batch_spinner.stop()
                    pool.shutdown(wait=False)
                for i in range(len(group)):
                    result_blocks.append(ordered[i])

        return result_blocks

    def _await_abandoned_peers(self, abandoned: list) -> list:
        """Wait a bounded grace for abandoned non-destructive peers to end.

        Returns the ones STILL running, so the caller can refuse to start a
        destructive tool beside them. The wait is bounded: an unbounded join here
        would defeat the batch deadline.
        """
        deadline = time.monotonic() + self._ABANDONED_PEER_GRACE_S
        for fut, _name in abandoned:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                fut.result(timeout=remaining)
            except Exception:
                # Only THAT it ended matters. Its outcome was already turned into
                # a result block in the segment that abandoned it.
                pass
        return [(f, n) for f, n in abandoned if not f.done()]

    # Wall-clock deadline for one parallel batch of non-destructive tools
    _PARALLEL_BATCH_TIMEOUT_S = 120

    # Extra grace a destructive tool gives an abandoned non-destructive peer from
    # the same batch to finish before it refuses to run beside it.
    _ABANDONED_PEER_GRACE_S = 30
