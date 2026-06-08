"""
Tool implementations for the coding agent.

Each tool is a plain function that takes a ``cwd`` (working directory Path)
plus keyword arguments matching the tool's parameter schema, and returns
a ``ToolResult``.

Tools are registered in the TOOL_REGISTRY dict at the bottom of this file.
"""

from __future__ import annotations

import glob as _glob
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .agent import Agent


# ---------------------------------------------------------------------------
#  ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    ok:      bool
    output:  str
    summary: str = ""       # one-line display shown in the console
    truncated: bool = False

    @classmethod
    def success(cls, output: str, summary: str = "") -> "ToolResult":
        return cls(ok=True, output=output, summary=summary)

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(ok=False, output=message, summary=f"ERROR: {message}")

    def to_xml(self, tool_name: str) -> str:
        status = "ok" if self.ok else "error"
        trunc  = ' truncated="true"' if self.truncated else ""
        return (
            f'<tool_result name="{tool_name}" status="{status}"{trunc}>\n'
            f"{self.output}\n"
            f"</tool_result>"
        )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

_MAX_OUTPUT = 8_000   # chars — truncate large outputs to spare context

def _truncate(text: str, max_chars: int = _MAX_OUTPUT) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    half = max_chars // 2
    return (
        text[:half] + f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n" + text[-half:],
        True,
    )


def _resolve(cwd: Path, path: str) -> Path:
    """Resolve a possibly-relative path against cwd."""
    p = Path(path)
    return p if p.is_absolute() else cwd / p


def _line_count(text: str) -> int:
    return text.count("\n") + 1


# ---------------------------------------------------------------------------
#  Tool functions
# ---------------------------------------------------------------------------

def tool_read_file(cwd: Path, path: str) -> ToolResult:
    p = _resolve(cwd, path)
    if not p.exists():
        return ToolResult.error(f"File not found: {p}")
    if not p.is_file():
        return ToolResult.error(f"Not a file: {p}")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ToolResult.error(str(e))
    rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
    lines = _line_count(text)
    output, trunc = _truncate(text)
    return ToolResult(
        ok=True,
        output=f"<path>{rel}</path>\n<lines>{lines}</lines>\n<content>\n{output}\n</content>",
        summary=f"{rel} — {lines} lines{' (truncated)' if trunc else ''}",
        truncated=trunc,
    )


def tool_write_file(cwd: Path, path: str, content: str) -> ToolResult:
    p = _resolve(cwd, path)
    existed = p.exists()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except Exception as e:
        return ToolResult.error(str(e))
    rel  = p.relative_to(cwd) if p.is_relative_to(cwd) else p
    verb = "updated" if existed else "created"
    lines = _line_count(content)
    return ToolResult.success(
        f"{verb} {rel} ({lines} lines)",
        summary=f"{verb} {rel} ({lines} lines)",
    )


def tool_edit_file(cwd: Path, path: str, old: str, new: str) -> ToolResult:
    """Replace the first occurrence of `old` with `new` in `path`."""
    p = _resolve(cwd, path)
    if not p.exists():
        return ToolResult.error(f"File not found: {p}")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ToolResult.error(str(e))

    if old not in text:
        snippet = textwrap.shorten(repr(old[:120]), width=120)
        return ToolResult.error(
            f"String not found in {path}.\n"
            f"Looking for: {snippet}\n"
            f"Hint: read the file first to get the exact text."
        )

    count = text.count(old)
    new_text = text.replace(old, new, 1)
    try:
        p.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return ToolResult.error(str(e))

    rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
    note = f" ({count - 1} more occurrence(s) unchanged)" if count > 1 else ""
    return ToolResult.success(
        f"Replaced 1 occurrence in {rel}{note}",
        summary=f"edited {rel}{note}",
    )


def tool_patch_file(cwd: Path, path: str, diff: str) -> ToolResult:
    """
    Apply a unified diff to a file.

    The diff must be in standard ``patch -u`` format::

        --- a/path/to/file.py
        +++ b/path/to/file.py
        @@ -10,4 +10,5 @@
         context line
        -old line
        +new line
        +added line

    File-header lines (``---``/``+++``) are optional but recommended.
    Line numbers in ``@@`` headers are used as hints only — minor off-by-one
    errors are tolerated.  Always read the file before generating the diff.
    """
    from ._patch import apply_diff, PatchError

    p = _resolve(cwd, path)
    if not p.exists():
        return ToolResult.error(f"File not found: {p}")
    if not p.is_file():
        return ToolResult.error(f"Not a file: {p}")

    try:
        original = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ToolResult.error(str(e))

    try:
        patched = apply_diff(original, diff)
    except PatchError as e:
        return ToolResult.error(str(e))
    except Exception as e:
        return ToolResult.error(f"Unexpected patch error: {e}")

    try:
        p.write_text(patched, encoding="utf-8")
    except Exception as e:
        return ToolResult.error(str(e))

    rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
    # Count changed lines for summary
    orig_lines  = set(original.splitlines())
    patch_lines = set(patched.splitlines())
    added   = len(patch_lines - orig_lines)
    removed = len(orig_lines - patch_lines)
    return ToolResult.success(
        f"Patched {rel} (+{added} / -{removed} lines)",
        summary=f"patched {rel} (+{added} / -{removed})",
    )


def tool_run_shell(
    cwd: Path,
    command: str,
    timeout: int = 30,
    _privacy: bool = False,
) -> ToolResult:
    """
    Execute a shell command.  Uses the system shell via a list invocation.

    In privacy mode (``_privacy=True``, injected by the agent) the subprocess
    environment has shell-history variables zeroed so that the command cannot
    be persisted to bash/sh/zsh history files.
    """
    shell_cmd: list[str]
    if sys.platform == "win32":
        shell_cmd = ["cmd", "/C", command]
    else:
        shell_cmd = ["/bin/sh", "-c", command]

    env: dict | None = None
    if _privacy:
        from .privacy import subprocess_privacy_env
        env = subprocess_privacy_env()

    try:
        proc = subprocess.run(
            shell_cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.error(f"Command timed out after {timeout}s")
    except Exception as e:
        return ToolResult.error(str(e))

    combined = ""
    if proc.stdout:
        combined += proc.stdout
    if proc.stderr:
        combined += ("\n" if combined else "") + "STDERR:\n" + proc.stderr

    combined = combined.strip() or "(no output)"
    output, trunc = _truncate(combined)
    rc = proc.returncode
    status = "ok" if rc == 0 else f"exit {rc}"

    return ToolResult(
        ok=(rc == 0),
        output=f"<exit_code>{rc}</exit_code>\n<output>\n{output}\n</output>",
        summary=f"$ {command[:60]}  [{status}]",
        truncated=trunc,
    )


def tool_list_dir(cwd: Path, path: str = ".") -> ToolResult:
    p = _resolve(cwd, path)
    if not p.exists():
        return ToolResult.error(f"Path not found: {p}")
    if not p.is_dir():
        return ToolResult.error(f"Not a directory: {p}")

    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    lines = []
    for e in entries[:200]:
        if e.is_dir():
            lines.append(f"  {e.name}/")
        else:
            size = e.stat().st_size
            sz   = f"{size/1e6:.1f}M" if size >= 1e6 else f"{size/1e3:.0f}k" if size >= 1e3 else f"{size}B"
            lines.append(f"  {e.name}  [{sz}]")

    rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
    output = f"{rel}/\n" + "\n".join(lines)
    if len(entries) > 200:
        output += f"\n  ... ({len(entries) - 200} more entries)"
    return ToolResult.success(output, summary=f"{rel}/ — {len(entries)} entries")


def tool_search_files(cwd: Path, pattern: str, path: str = ".") -> ToolResult:
    base = _resolve(cwd, path)
    full_pattern = str(base / pattern) if not Path(pattern).is_absolute() else pattern
    try:
        matches = sorted(_glob.glob(full_pattern, recursive=True))
    except Exception as e:
        return ToolResult.error(str(e))

    if not matches:
        return ToolResult.success("No files matched.", summary="0 matches")

    rel_matches = []
    for m in matches[:200]:
        try:
            rel_matches.append(str(Path(m).relative_to(cwd)))
        except ValueError:
            rel_matches.append(m)

    output = "\n".join(rel_matches)
    trunc  = len(matches) > 200
    if trunc:
        output += f"\n... ({len(matches) - 200} more)"
    return ToolResult(
        ok=True,
        output=output,
        summary=f"{len(matches)} file(s) matched '{pattern}'",
        truncated=trunc,
    )


def tool_grep(cwd: Path, pattern: str, path: str = ".", glob: str = "", context: int = 2) -> ToolResult:
    """Search file contents with a regex pattern (pure Python, no external tools)."""
    base = _resolve(cwd, path)
    file_glob = glob or "**/*"
    files = sorted(base.glob(file_glob)) if base.is_dir() else [base]

    try:
        rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error as e:
        return ToolResult.error(f"Invalid regex: {e}")

    results = []
    total_hits = 0
    for fp in files:
        if not fp.is_file():
            continue
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        hits = []
        for i, line in enumerate(lines):
            if rx.search(line):
                start = max(0, i - context)
                end   = min(len(lines), i + context + 1)
                hits.append((i + 1, lines[start:end], i - start))

        if hits:
            try:
                rel = fp.relative_to(cwd)
            except ValueError:
                rel = fp
            results.append(f"## {rel}")
            for lineno, ctx_lines, hit_offset in hits[:20]:
                for j, ctx_line in enumerate(ctx_lines):
                    marker = "→ " if j == hit_offset else "  "
                    results.append(f"{marker}{lineno - hit_offset + j:4d}: {ctx_line}")
                results.append("")
            total_hits += len(hits)
            if len(results) > 300:
                break

    if not results:
        return ToolResult.success(f"No matches for '{pattern}'", summary="0 matches")

    output, trunc = _truncate("\n".join(results))
    return ToolResult(
        ok=True,
        output=output,
        summary=f"{total_hits} match(es) for '{pattern}'",
        truncated=trunc,
    )


# ---------------------------------------------------------------------------
#  spawn_agent — sub-agent tool
# ---------------------------------------------------------------------------

def tool_spawn_agent(
    cwd: Path,
    task: str,
    name: str = "subagent",
    files: Optional[list] = None,
    model: Optional[str] = None,
    max_turns: int = 10,
    _parent_agent: Optional[Any] = None,
) -> ToolResult:
    """
    Spawn a child Agent with a focused task.

    The child inherits the parent's backend (or uses ``model`` if given),
    gets ``files`` pre-loaded into its first user message, and runs until
    it produces a final answer or ``max_turns`` is reached.

    Returns the child agent's final response as a string.
    """
    if _parent_agent is None:
        return ToolResult.error("spawn_agent requires a running parent agent")

    from .agent import Agent

    backend = _parent_agent.backend
    if model and model != backend.model_id:
        from .backends.http import make_localm_backend
        raw_url = getattr(backend, "_base_url", "http://127.0.0.1:8080/v1")
        try:
            port = int(raw_url.split(":")[-1].split("/")[0])
        except Exception:
            port = 8080
        try:
            backend = make_localm_backend(model, port=port)
        except Exception:
            backend = _parent_agent.backend

    preload_text = ""
    if files:
        for fp in files:
            r = tool_read_file(cwd, fp)
            preload_text += f"\n{r.output}\n"

    full_task = task
    if preload_text:
        full_task = f"Context files:\n{preload_text}\n\nTask:\n{task}"

    from .audit import SessionMode as _SessionMode
    inherited_mode = getattr(_parent_agent, "mode", _SessionMode.PRIVACY)

    child = Agent(
        backend=backend,
        cwd=cwd,
        name=name,
        max_turns=max_turns,
        verbose=False,
        auto_approve=True,
        parent=_parent_agent,
        mode=inherited_mode,
    )
    result_text = child.run_task(full_task)
    turns_used  = child.turns

    return ToolResult.success(
        result_text,
        summary=f"sub-agent '{name}' finished in {turns_used} turn(s)",
    )


def tool_fetch_url(cwd: Path, url: str, max_chars: int = 8000) -> ToolResult:
    """
    Fetch a URL and return its plain-text content (HTML tags stripped).

    Useful for documentation pages, GitHub raw files, Stack Overflow answers,
    and package changelogs.  Content is truncated to ``max_chars`` to avoid
    flooding the context window.
    """
    import html.parser
    import urllib.error
    import urllib.request

    class _HTMLStripper(html.parser.HTMLParser):
        _SKIP = {"script", "style", "head", "meta", "link", "noscript"}

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self._buf: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag.lower() in self._SKIP:
                self._skip += 1

        def handle_endtag(self, tag):
            if tag.lower() in self._SKIP and self._skip:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip:
                self._buf.append(data)

        def get_text(self) -> str:
            import re
            raw = "".join(self._buf)
            # Collapse excessive blank lines
            return re.sub(r"\n{3,}", "\n\n", raw).strip()

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "localm/0.1 (fetch_url tool)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(1_000_000).decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return ToolResult.error(f"Could not fetch {url}: {e}")
    except Exception as e:
        return ToolResult.error(str(e))

    if "html" in content_type.lower():
        stripper = _HTMLStripper()
        stripper.feed(raw)
        text = stripper.get_text()
    else:
        text = raw.strip()

    output, trunc = _truncate(text, max_chars)
    return ToolResult(
        ok=True,
        output=f"<url>{url}</url>\n<content>\n{output}\n</content>",
        summary=f"fetched {url[:60]} ({len(text):,} chars{', truncated' if trunc else ''})",
        truncated=trunc,
    )


def _git(cwd: Path, *args: str, timeout: int = 10) -> tuple[str, bool]:
    """Run a git command and return (output, ok)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        out = (proc.stdout + proc.stderr).strip() or "(no output)"
        return out, proc.returncode == 0
    except FileNotFoundError:
        return "git not found in PATH", False
    except subprocess.TimeoutExpired:
        return f"git {args[0]} timed out", False
    except Exception as e:
        return str(e), False


def tool_git_status(cwd: Path) -> ToolResult:
    """Return the output of `git status --short` in the working directory."""
    out, ok = _git(cwd, "status", "--short", "--branch")
    return ToolResult(ok=ok, output=out, summary=f"git status ({len(out.splitlines())} lines)")


def tool_git_diff(cwd: Path, path: str = "", staged: bool = False) -> ToolResult:
    """
    Return `git diff` output.

    Parameters
    ----------
    path   : limit diff to this file or directory (optional)
    staged : if True, show staged changes (`git diff --cached`)
    """
    args = ["diff", "--stat", "-p"]
    if staged:
        args.append("--cached")
    if path:
        args += ["--", path]
    out, ok = _git(cwd, *args, timeout=15)
    out, trunc = _truncate(out)
    return ToolResult(ok=ok, output=out, summary="git diff" + (" --cached" if staged else ""), truncated=trunc)


def tool_git_log(cwd: Path, n: int = 10, path: str = "") -> ToolResult:
    """Return the last n commits as a compact log."""
    args = ["log", f"--max-count={n}", "--oneline", "--decorate"]
    if path:
        args += ["--", path]
    out, ok = _git(cwd, *args)
    return ToolResult(ok=ok, output=out, summary=f"git log -{n}")


def _localm_unload() -> None:
    """
    Ask localm to release its model from GPU memory so FLUX can use the VRAM.

    Reads LOCALM_URL (set by ManagedServer when it starts localm serve).
    Silent no-op if the env var isn't set (external server, not our problem)
    or if the request fails for any reason.
    """
    import os, urllib.request, urllib.error
    localm_url = os.environ.get("LOCALM_URL", "").rstrip("/")
    if not localm_url:
        return
    try:
        req = urllib.request.Request(
            f"{localm_url}/models/unload",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass  # best-effort; don't block image generation if this fails


def tool_generate_image(cwd: Path, prompt: str, output_path: str = "output.png") -> ToolResult:
    """
    Generate an image from a text prompt using a local ComfyUI FLUX GGUF setup.
    Connects to ComfyUI (via FLUX_API_URL or default http://127.0.0.1:8188).

    Before queuing the image, the localm inference model is unloaded from GPU
    memory so ComfyUI has the full VRAM budget for FLUX.  The model reloads
    automatically on the next chat turn.
    """
    import json
    import os
    import random
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    from rich.console import Console as _Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    _con = _Console()

    # 1. Resolve output path
    out_p = _resolve(cwd, output_path)

    # 2. API URL
    api_url = os.environ.get("FLUX_API_URL", "http://127.0.0.1:8188")

    # 3. Unload LLM to free VRAM before FLUX loads
    _localm_unload()

    # 4. Load workflow template
    try:
        wf_path = Path(__file__).parent / "flux_workflow.json"
        workflow = json.loads(wf_path.read_text(encoding="utf-8"))
    except Exception as e:
        return ToolResult.error(f"Failed to load FLUX workflow template: {e}")

    # 5. Inject prompt — try node "6" first (default template), then scan for
    #    any CLIPTextEncode node with a PROMPT_PLACEHOLDER value.
    injected = False
    if "6" in workflow and workflow["6"].get("inputs", {}).get("text") is not None:
        workflow["6"]["inputs"]["text"] = prompt
        injected = True
    if not injected:
        for node in workflow.values():
            if node.get("class_type") == "CLIPTextEncode":
                node["inputs"]["text"] = prompt
                injected = True
                break
    if not injected:
        return ToolResult.error(
            "Could not find a text-prompt node in the workflow template.\n"
            "Export a fresh workflow from ComfyUI (Save → API format) and replace\n"
            f"{wf_path}"
        )

    # 6. Randomise seed so every run is unique.
    #    Handles both KSampler-style and SamplerCustomAdvanced-style (RandomNoise node).
    seed = random.randint(1, 10 ** 12)
    for node in workflow.values():
        cls = node.get("class_type", "")
        if cls in ("KSampler", "KSamplerAdvanced"):
            node["inputs"]["seed"] = seed
            break
        if cls == "RandomNoise":
            node["inputs"]["noise_seed"] = seed
            break

    # 7. Queue the prompt in ComfyUI
    try:
        req_data = json.dumps({"prompt": workflow}).encode("utf-8")
        req = urllib.request.Request(
            f"{api_url}/prompt",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode("utf-8"))
            prompt_id = res.get("prompt_id")

        if not prompt_id:
            return ToolResult.error(
                "ComfyUI accepted the request but returned no prompt_id.\n"
                "Check the ComfyUI console for workflow validation errors."
            )

    except urllib.error.URLError as e:
        return ToolResult.error(
            f"Could not connect to ComfyUI at {api_url}.\n"
            f"Error: {e}\n"
            "Make sure ComfyUI is running.  Setup guide: flux_local_setup_guide.md"
        )
    except Exception as e:
        return ToolResult.error(f"Error queuing prompt in ComfyUI: {e}")

    # 8. Poll /history with a visible progress spinner
    start_time = time.time()
    max_poll_time = 600
    finished = False
    filename = None
    subfolder = ""
    img_type = "output"

    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]{task.description}[/dim]"),
        TimeElapsedColumn(),
        transient=True,
        console=_con,
    ) as progress:
        task_id = progress.add_task("Generating image…", total=None)

        while time.time() - start_time < max_poll_time:
            elapsed = int(time.time() - start_time)
            progress.update(task_id, description=f"Generating image… ({elapsed}s)")

            try:
                hist_req = urllib.request.Request(
                    f"{api_url}/history/{prompt_id}"
                )
                with urllib.request.urlopen(hist_req, timeout=5) as response:
                    history = json.loads(response.read().decode("utf-8"))

                if prompt_id in history:
                    finished = True
                    outputs = history[prompt_id].get("outputs", {})
                    for node_output in outputs.values():
                        if "images" in node_output:
                            img_info = node_output["images"][0]
                            filename = img_info.get("filename")
                            subfolder = img_info.get("subfolder", "")
                            img_type  = img_info.get("type", "output")
                            break
                    break

            except Exception:
                pass

            time.sleep(2)

    if not finished:
        return ToolResult.error("Image generation timed out after 10 minutes.")

    if not filename:
        return ToolResult.error(
            "Generation finished but no output image was found in ComfyUI history.\n"
            "Check the ComfyUI console — a SaveImage node error is likely."
        )

    # 9. Fetch the image from ComfyUI /view endpoint and save it locally
    try:
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": img_type
        })
        img_url = f"{api_url}/view?{params}"

        out_p.parent.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(img_url, timeout=10) as response:
            out_p.write_bytes(response.read())

        rel = out_p.relative_to(cwd) if out_p.is_relative_to(cwd) else out_p
        return ToolResult.success(
            f"Image successfully generated and saved to {rel}",
            summary=f"generated {rel}"
        )
    except Exception as e:
        return ToolResult.error(f"Failed to download generated image from ComfyUI: {e}")


# ---------------------------------------------------------------------------
#  Tool registry
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    name:        str
    fn:          Callable
    description: str
    params:      dict
    destructive: bool = False


TOOL_REGISTRY: dict[str, ToolDef] = {
    "read_file": ToolDef(
        name="read_file",
        fn=tool_read_file,
        description="Read the contents of a file.",
        params={"path": {"type": "string", "description": "File path (relative to cwd)", "required": True}},
    ),
    "write_file": ToolDef(
        name="write_file",
        fn=tool_write_file,
        description="Write or overwrite a file with new content.",
        params={
            "path":    {"type": "string", "description": "File path",         "required": True},
            "content": {"type": "string", "description": "Full file content", "required": True},
        },
        destructive=True,
    ),
    "edit_file": ToolDef(
        name="edit_file",
        fn=tool_edit_file,
        description="Replace the first occurrence of `old` text with `new` text in a file.",
        params={
            "path": {"type": "string", "description": "File path",            "required": True},
            "old":  {"type": "string", "description": "Exact text to replace","required": True},
            "new":  {"type": "string", "description": "Replacement text",     "required": True},
        },
        destructive=True,
    ),
    "patch_file": ToolDef(
        name="patch_file",
        fn=tool_patch_file,
        description=(
            "Apply a unified diff (patch -u format) to a file. "
            "More reliable than edit_file for multi-hunk or large changes. "
            "Always read_file first so line numbers are accurate."
        ),
        params={
            "path": {"type": "string", "description": "File path (relative to cwd)",              "required": True},
            "diff": {"type": "string", "description": "Unified diff string (patch -u format)",    "required": True},
        },
        destructive=True,
    ),
    "run_shell": ToolDef(
        name="run_shell",
        fn=tool_run_shell,
        description="Execute a shell command in the working directory.",
        params={
            "command": {"type": "string", "description": "Shell command",      "required": True},
            "timeout": {"type": "int",    "description": "Timeout in seconds", "required": False},
        },
        destructive=True,
    ),
    "list_dir": ToolDef(
        name="list_dir",
        fn=tool_list_dir,
        description="List the contents of a directory.",
        params={"path": {"type": "string", "description": "Directory path (default: .)", "required": False}},
    ),
    "search_files": ToolDef(
        name="search_files",
        fn=tool_search_files,
        description="Find files matching a glob pattern.",
        params={
            "pattern": {"type": "string", "description": "Glob pattern, e.g. **/*.py", "required": True},
            "path":    {"type": "string", "description": "Root directory to search",    "required": False},
        },
    ),
    "grep": ToolDef(
        name="grep",
        fn=tool_grep,
        description="Search file contents with a regex pattern.",
        params={
            "pattern": {"type": "string", "description": "Regex pattern",                "required": True},
            "path":    {"type": "string", "description": "File or directory to search",   "required": False},
            "glob":    {"type": "string", "description": "File filter, e.g. **/*.py",     "required": False},
            "context": {"type": "int",    "description": "Lines of context (default 2)",  "required": False},
        },
    ),
    "git_status": ToolDef(
        name="git_status",
        fn=tool_git_status,
        description="Show working-tree status (git status --short --branch).",
        params={},
    ),
    "git_diff": ToolDef(
        name="git_diff",
        fn=tool_git_diff,
        description="Show git diff (unstaged by default; pass staged=true for staged changes).",
        params={
            "path":   {"type": "string", "description": "Limit to this file/dir",    "required": False},
            "staged": {"type": "bool",   "description": "Show staged diff (default false)", "required": False},
        },
    ),
    "git_log": ToolDef(
        name="git_log",
        fn=tool_git_log,
        description="Show recent commits (oneline format).",
        params={
            "n":    {"type": "int",    "description": "Number of commits (default 10)", "required": False},
            "path": {"type": "string", "description": "Limit to this file/dir",         "required": False},
        },
    ),
    "fetch_url": ToolDef(
        name="fetch_url",
        fn=tool_fetch_url,
        description="Fetch a URL and return its plain-text content (HTML stripped).",
        params={
            "url":       {"type": "string", "description": "Full URL to fetch",               "required": True},
            "max_chars": {"type": "int",    "description": "Truncate output (default 8000)",  "required": False},
        },
    ),
    "spawn_agent": ToolDef(
        name="spawn_agent",
        fn=tool_spawn_agent,
        description="Spawn a focused sub-agent to handle a specific sub-task and return its result.",
        params={
            "task":      {"type": "string", "description": "What the sub-agent should do",     "required": True},
            "name":      {"type": "string", "description": "Short name for this sub-agent",     "required": False},
            "files":     {"type": "array",  "description": "Files to pre-load into sub-agent", "required": False},
            "model":     {"type": "string", "description": "Override model for sub-agent",      "required": False},
            "max_turns": {"type": "int",    "description": "Max iterations (default 10)",       "required": False},
        },
    ),
    "generate_image": ToolDef(
        name="generate_image",
        fn=tool_generate_image,
        description="Generate a high-quality image from a detailed text prompt using the local FLUX model.",
        params={
            "prompt":      {"type": "string", "description": "Descriptive prompt detailing styles, objects, and composition.", "required": True},
            "output_path": {"type": "string", "description": "Path to save the generated image file (default: output.png)", "required": False},
        },
    ),
}

