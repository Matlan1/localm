# SPDX-License-Identifier: AGPL-3.0-or-later
"""Machine-name redaction in the bug-report scrubbers: localm/bugreport/scrub.py
``_scrub_secrets``, scripts/report_issue.py ``scrub``, scripts/report_issue.ps1
``Scrub`` and report-issue.sh ``scrub``. A UNC path's machine name, a host name
under a local-network suffix and this machine's own names become ``<host>``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

from localm.bugreport import scrub as br_scrub

_REPO = Path(__file__).resolve().parents[1]
_RI_PATH = _REPO / "scripts" / "report_issue.py"
_PS1_PATH = _REPO / "scripts" / "report_issue.ps1"
_SH_PATH = _REPO / "report-issue.sh"

_spec = importlib.util.spec_from_file_location("report_issue_hosts", _RI_PATH)
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)

BS = "\\"

# (input, expected output) pairs every scrubber must produce exactly.
_CASES = [
    ("open " + BS * 2 + "FILESRV01" + BS + "share" + BS + "m.gguf",
     "open " + BS * 2 + "<host>" + BS + "share" + BS + "m.gguf"),
    (repr(BS * 2 + "NAS-2" + BS + "media"),
     repr(BS * 2 + "<host>" + BS + "media")),
    ('{"p": "' + BS * 4 + "fs.corp.example.com" + BS * 2 + 'share"}',
     '{"p": "' + BS * 4 + "<host>" + BS * 2 + 'share"}'),
    (BS * 2 + "?" + BS + "UNC" + BS + "FILESRV01" + BS + "share",
     BS * 2 + "?" + BS + "UNC" + BS + "<host>" + BS + "share"),
    ("(" + BS * 2 + "srv9" + BS + "x)", "(" + BS * 2 + "<host>" + BS + "x)"),
    (BS * 2 + "wsl$" + BS + "Ubuntu", BS * 2 + "wsl$" + BS + "Ubuntu"),
    (BS * 2 + "wsl.localhost" + BS + "Ubuntu", BS * 2 + "wsl.localhost" + BS + "Ubuntu"),
    (BS * 2 + "localhost" + BS + "c$", BS * 2 + "localhost" + BS + "c$"),
    (BS * 2 + "?" + BS + "C:" + BS + "x", BS * 2 + "?" + BS + "C:" + BS + "x"),
    (BS * 2 + "." + BS + "pipe" + BS + "x", BS * 2 + "." + BS + "pipe" + BS + "x"),
    ("D:" + BS * 2 + "models" + BS * 2 + "a.gguf", "D:" + BS * 2 + "models" + BS * 2 + "a.gguf"),
    ("see nas.local:8080, pc.LAN and printer.home.arpa",
     "see <host>.local:8080, <host>.LAN and <host>.home.arpa"),
    ("db.internal/x box.localdomain.", "<host>.internal/x <host>.localdomain."),
    ("https://u:p@box.lan/x", "https://<redacted>@<host>.lan/x"),
    ("config.local.json, a .env.local file, github.com and huggingface.co",
     "config.local.json, a .env.local file, github.com and huggingface.co"),
    ("models on `" + BS * 2 + "NAS01" + BS + "llm`", "models on `" + BS * 2 + "<host>" + BS + "llm`"),
    ("path:" + BS * 2 + "NAS01" + BS + "x", "path:" + BS * 2 + "<host>" + BS + "x"),
    ("<" + BS * 2 + "NAS01" + BS + "x>", "<" + BS * 2 + "<host>" + BS + "x>"),
    ("*" + BS * 2 + "NAS01" + BS + "x", "*" + BS * 2 + "<host>" + BS + "x"),
    ("D:" + BS * 2 + "data" + BS * 2 + "x.gguf", "D:" + BS * 2 + "data" + BS * 2 + "x.gguf"),
    ("state = threading.local()", "state = threading.local()"),
    ("netname: advertising localm.local -> 10.0.0.5", "netname: advertising localm.local -> 10.0.0.5"),
    ("http://host.docker.internal:8188 and gateway.docker.internal",
     "http://host.docker.internal:8188 and gateway.docker.internal"),
]

_SCRUBBERS = [
    pytest.param(br_scrub._scrub_secrets, id="bugreport"),
    pytest.param(ri.scrub, id="report_issue_py"),
]


@pytest.mark.parametrize("scrub", _SCRUBBERS)
@pytest.mark.parametrize("text,want", _CASES)
def test_machine_names_are_redacted(scrub, text, want):
    assert scrub(text) == want
    assert scrub(want) == want


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_own_host_name_is_redacted_wherever_it_stands_alone(scrub, monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "zq-workstation.corp.example.com")
    monkeypatch.setenv("COMPUTERNAME", "ZQ-WORKSTATION")
    text = ("on ZQ-WORKSTATION as zq-workstation.corp.example.com; "
            "zq-workstation2 and my-zq-workstation stay")
    assert scrub(text) == "on <host> as <host>; zq-workstation2 and my-zq-workstation stay"


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_own_host_name_inside_a_dotted_or_path_name_is_kept(scrub, monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "zqbox7")
    monkeypatch.delenv("COMPUTERNAME", raising=False)
    text = "zqbox7.inference.http_server, src/zqbox7/x.py, C:" + BS + "zqbox7" + BS + "y; on zqbox7."
    assert scrub(text) == ("zqbox7.inference.http_server, src/zqbox7/x.py, C:" + BS + "zqbox7"
                           + BS + "y; on <host>.")


# A host name of letters only can be an ordinary word, so it is replaced only
# where it stands as a machine name. No list of words decides this.
_WORD_NAMES = ["ubuntu", "Fedora", "localm", "host", "redacted", "user", "server",
               "Desktop", "workstation", "gpu", "main", "build", "zqbox", "thomas"]


@pytest.mark.parametrize("scrub", _SCRUBBERS)
@pytest.mark.parametrize("name", _WORD_NAMES)
def test_a_letters_only_host_name_is_kept_in_prose(scrub, monkeypatch, name):
    monkeypatch.setattr(socket, "gethostname", lambda: name)
    monkeypatch.setenv("COMPUTERNAME", name.upper())
    text = (f"{name} 24.04 and <host> and <redacted>; the {name} crashed; "
            f"{name}.example.com and src/{name}/x.py")
    assert scrub(text) == text


@pytest.mark.parametrize("scrub", _SCRUBBERS)
@pytest.mark.parametrize("name", _WORD_NAMES)
def test_a_letters_only_host_name_is_replaced_where_it_names_a_machine(scrub, monkeypatch, name):
    monkeypatch.setattr(socket, "gethostname", lambda: name)
    monkeypatch.setenv("COMPUTERNAME", name.upper())
    text = (f"http://{name}:8080/v1, ssh me@{name.upper()}, {name}:5000 up, "
            f"hostname={name}, Host: {name}, \"computername\": \"{name}\", "
            f"Host Name:      {name.upper()}, node: {name}, Device_Name={name}")
    assert scrub(text) == ("http://<host>:8080/v1, ssh me@<host>, <host>:5000 up, "
                           "hostname=<host>, Host: <host>, \"computername\": \"<host>\", "
                           "Host Name:      <host>, node: <host>, Device_Name=<host>")
    assert scrub(scrub(text)) == scrub(text)


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_a_name_with_a_non_letter_is_replaced_in_a_url_too(scrub, monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "zq-ws1")
    monkeypatch.delenv("COMPUTERNAME", raising=False)
    text = "http://zq-ws1:8080/v1 and //zq-ws1/share and on zq-ws1"
    assert scrub(text) == "http://<host>:8080/v1 and //<host>/share and on <host>"


def test_python_reporters_share_the_own_name_rules():
    import inspect
    for name in ("_own_host_names", "_scrub_own_names"):
        assert inspect.getsource(getattr(ri, name)) == inspect.getsource(getattr(br_scrub, name)), name
    assert ri._WORD_NAME_RE.pattern == br_scrub._WORD_NAME_RE.pattern
    assert ri._HOST_KEY == br_scrub._HOST_KEY


def test_a_report_title_keeps_a_common_word_host_name(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "server")
    monkeypatch.setenv("COMPUTERNAME", "SERVER")
    title = "localm server crashed while stopping - native fault captured"
    assert br_scrub._scrub_secrets(title) == title


def test_python_reporters_share_the_local_network_scanner():
    import inspect
    for name in ("_lan_boundary", "_lan_token", "_scrub_lan_hosts"):
        assert inspect.getsource(getattr(ri, name)) == inspect.getsource(getattr(br_scrub, name)), name
    for name in ("_LAN_SUFFIXES", "_LAN_KEEP"):
        assert getattr(ri, name) == getattr(br_scrub, name), name
    assert ri._LAN_TOKEN_RE.pattern == br_scrub._LAN_TOKEN_RE.pattern


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_short_or_loopback_own_names_are_not_used(scrub, monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "pc")
    monkeypatch.setenv("COMPUTERNAME", "localhost")
    assert scrub("pc on localhost") == "pc on localhost"


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_unreadable_host_name_still_runs_the_other_rules(scrub, monkeypatch):
    def boom():
        raise OSError("no name")
    monkeypatch.setattr(socket, "gethostname", boom)
    monkeypatch.delenv("COMPUTERNAME", raising=False)
    assert scrub(BS * 2 + "srv9" + BS + "x nas.local") == BS * 2 + "<host>" + BS + "x <host>.local"


def test_the_real_host_name_is_redacted_in_a_report():
    import re
    name = socket.gethostname().strip().rstrip(".")
    if len(name) < 3 or name.lower() == "localhost":
        pytest.skip("this machine's name is too short to be redacted")
    line = _real_name_line()
    out = br_scrub._scrub_secrets(line)
    if re.fullmatch(r"[A-Za-z]+", name.split(".")[0]):
        assert out.startswith("crash on " + name.split(".")[0])
        assert "http://<host>:8080" in out
    else:
        assert name.lower() not in out.lower()
    assert ri.scrub(line) == out


_PS_DRIVER = (
    "param([string]$Target, [string]$InFile)\n"
    "$ast = [System.Management.Automation.Language.Parser]::ParseFile("
    "$Target, [ref]$null, [ref]$null)\n"
    "$fn = $ast.FindAll({ param($n) $n -is "
    "[System.Management.Automation.Language.FunctionDefinitionAst] -and "
    "$n.Name -eq 'Scrub' }, $true)\n"
    "$asg = $ast.FindAll({ param($n) $n -is "
    "[System.Management.Automation.Language.AssignmentStatementAst] -and "
    "$n.Left.Extent.Text -eq '$email' }, $false)\n"
    "if ($fn.Count -ne 1 -or $asg.Count -ne 1) { Write-Output 'SCRUB-NOT-FOUND'; exit 1 }\n"
    "Invoke-Expression $asg[0].Extent.Text\n"
    "Invoke-Expression $fn[0].Extent.Text\n"
    "$text = [System.IO.File]::ReadAllText($InFile, [System.Text.Encoding]::UTF8)\n"
    "$out = New-Object System.Collections.Generic.List[string]\n"
    "foreach ($line in ($text -split \"`n\")) { $out.Add((Scrub $line)) }\n"
    "[System.IO.File]::WriteAllText($InFile + '.out', ($out -join \"`n\"), "
    "(New-Object System.Text.UTF8Encoding($false)))\n"
)


def _real_name_line() -> str:
    name = socket.gethostname().strip().rstrip(".").split(".")[0]
    return "crash on " + name + " at boot, served at http://" + name + ":8080"


# Lines for COMPUTERNAME=SERVER: a letters-only name in prose and as a machine name.
_WORD_LINES = ["localm server crashed while stopping",
               "http://server:8080/v1 and me@SERVER and server:5000 up",
               "hostname=server, Host: server, Host Name:   SERVER, node: server"]


@pytest.mark.skipif(os.name != "nt",
                    reason="the PowerShell fallback reporter only runs on Windows")
def test_powershell_reporter_redacts_machine_names(tmp_path, monkeypatch):
    pwsh = shutil.which("powershell") or shutil.which("pwsh")
    if not pwsh:
        pytest.skip("no PowerShell interpreter on PATH")
    lines = [text for text, _ in _CASES] + [_real_name_line()] + _WORD_LINES
    monkeypatch.setenv("COMPUTERNAME", "SERVER")
    driver = tmp_path / "drive_scrub.ps1"
    driver.write_text(_PS_DRIVER, encoding="utf-8")
    infile = tmp_path / "in.txt"
    infile.write_text("\n".join(lines), encoding="utf-8")
    proc = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(driver), "-Target", str(_PS1_PATH), "-InFile", str(infile)],
        capture_output=True, text=True, timeout=120, env=dict(os.environ))
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "SCRUB-NOT-FOUND" not in out, out
    assert proc.returncode == 0, f"driver failed ({proc.returncode}): {out}"
    got = (tmp_path / "in.txt.out").read_text(encoding="utf-8").split("\n")
    assert got == [br_scrub._scrub_secrets(t) for t in lines]


def _sh_scrub_function() -> str:
    text = _SH_PATH.read_text(encoding="utf-8")
    start = text.index("\nscrub() {\n") + 1
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def test_shell_reporter_redacts_machine_names(tmp_path, monkeypatch):
    bash = shutil.which("bash")
    if not bash or "system32" in bash.lower() or not shutil.which("perl"):
        pytest.skip("no bash with perl on PATH")
    lines = [text for text, _ in _CASES] + [_real_name_line()] + _WORD_LINES
    monkeypatch.setenv("COMPUTERNAME", "SERVER")
    driver = tmp_path / "drive_sh_scrub.sh"
    driver.write_text(_sh_scrub_function() + 'scrub < "$1" > "$2"\n',
                      encoding="utf-8", newline="\n")
    infile = tmp_path / "sh_in.txt"
    infile.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    outfile = tmp_path / "sh_out.txt"
    env = dict(os.environ, HOME=(tmp_path / "nohome").as_posix(), RI_EMAIL=ri.MAINTAINER_EMAIL)
    proc = subprocess.run([bash, driver.as_posix(), infile.as_posix(), outfile.as_posix()],
                          capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr
    got = outfile.read_text(encoding="utf-8").split("\n")[:-1]
    assert got == [br_scrub._scrub_secrets(t) for t in lines]


def _best(fn, text: str, samples: int = 5) -> float:
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        fn(text)
        times.append(time.perf_counter() - start)
    return min(times)


_HOSTILE = [
    pytest.param(lambda n: BS * n, id="backslash-run"),
    pytest.param(lambda n: (" " + BS * 2 + "a") * n, id="many-unc-starts"),
    pytest.param(lambda n: "a." * n, id="many-labels-no-suffix"),
    pytest.param(lambda n: "a-" * n + ".x", id="one-long-label"),
    pytest.param(lambda n: "a.local." * n + "a", id="suffix-then-label"),
    pytest.param(lambda n: (BS * 2 + "?" + BS) * n, id="device-prefixes"),
    pytest.param(lambda n: "-" * n, id="hyphen-run"),
    pytest.param(lambda n: "-" + ".-" * n, id="dot-hyphen-run"),
    pytest.param(lambda n: "-." * n, id="hyphen-dot-run"),
    pytest.param(lambda n: "a." * n + "local(", id="labels-then-call"),
]


@pytest.mark.parametrize("make", _HOSTILE)
def test_host_patterns_do_not_grow_superlinearly(make):
    """10x input: linear costs ~10x, quadratic ~100x; the bound is 30x."""
    base = _best(br_scrub._scrub_hosts, make(10_000))
    ten_x = _best(br_scrub._scrub_hosts, make(100_000))
    assert ten_x < max(30 * base, 0.005), (
        f"{base * 1000:.2f}ms -> {ten_x * 1000:.2f}ms for a 10x input")
