# SPDX-License-Identifier: AGPL-3.0-or-later
"""``scripts/update_gate.bat`` for a program without a console: shows in a window that an
update of LocaLM is waited for or undone, and why when that fails. Run by the programs
that open windows, before they import anything of LocaLM; standard library only. Exit
status 0: LocaLM may start."""

import subprocess
import sys
import threading
from pathlib import Path

GATE = Path(__file__).resolve().parent.parent / "localm" / "__init__.py"
_HIDDEN = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def run(say) -> tuple:
    """Run the gate and give each line it prints to *say*. Returns its exit status and
    all it printed."""
    proc = subprocess.Popen([sys.executable, "-I", "-S", "-X", "utf8", str(GATE)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", **_HIDDEN)
    lines = []
    for line in proc.stderr:
        lines.append(line.rstrip())
        say(lines[-1])
    return proc.wait(), "\n".join(lines)


def _run_into(said: list, result: list) -> None:
    result.append(run(said.append))


def main() -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox
        window = tk.Tk()
    except Exception as e:
        print(f"[!] no window can be shown: {e}", file=sys.stderr)
        return run(lambda line: print(line, file=sys.stderr, flush=True))[0]
    window.title("LocaLM")
    window.protocol("WM_DELETE_WINDOW", lambda: None)
    text = tk.StringVar(window, "Checking an update of LocaLM...")
    tk.Label(window, textvariable=text, width=70, height=6, wraplength=480).pack(padx=16, pady=16)
    while True:
        said: list = []
        result: list = []
        worker = threading.Thread(target=_run_into, args=(said, result), daemon=True)
        worker.start()
        while worker.is_alive():
            if said:
                text.set(said[-1])
            window.update()
            worker.join(0.1)
        code, all_said = result[0] if result else (1, "The update could not be checked.")
        if code == 0:
            messagebox.showinfo("LocaLM", all_said or "LocaLM can be used.", parent=window)
            return 0
        if not messagebox.askretrycancel("LocaLM", all_said, parent=window):
            return code


if __name__ == "__main__":
    sys.exit(main())
