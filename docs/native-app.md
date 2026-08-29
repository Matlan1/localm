# Native app identity (LocaLM.exe)

This page covers two related but independent things: running the GUI in its own
app window instead of a browser tab, and the server's process identity
(`LocaLM.exe`, the taskbar/tray icon). Neither requires the other.

By default the localm server runs as `LocaLM.exe` in Task Manager (not
`python.exe`), launches from a double-click or a desktop shortcut, and carries the
LocaLM icon on the taskbar, in the tray, and on the file itself. `setup.bat` /
`setup.sh` build this launcher for you; you can also (re)build it yourself.

## App window, not a browser tab

Install the optional `desktop` extra (`localm[desktop]`, built on
[pywebview](https://pywebview.flowrl.com/)) and `localm gui` opens the GUI in its
own OS window instead of a browser tab - no address bar, no other tabs, and it
closes and reopens like any other app.

- **Setup asks up front.** `setup.sh` / `setup.bat` ask "Open localm's GUI as its
  own app window, or in your browser?" before installing anything. The default is
  the browser tab (no extra install); choosing the app window installs
  `localm[desktop]` alongside the rest.
- **Change your mind later without reinstalling**, from Settings &rsaquo; System &rsaquo;
  Desktop app: **Default window mode** (`auto`, the default: the app window when
  the extra is installed, else a browser tab; `browser`: always a browser tab,
  even with the extra installed) and **Quit when the app window is closed** (off
  by default: closing the window hides it and the server keeps running, same as
  closing a browser tab, use Stop to actually quit; on: closing the window quits
  the server too). **Default window mode** is read once, at launch, so switching
  it does not touch a window already open - it takes effect on the next
  `localm gui`. **Quit when the app window is closed** is read live, from disk,
  every time the window's close button is clicked - so switching it DOES change
  the behavior of a window that is already open, with no restart needed.
- **`localm gui --no-browser` suppresses the app window too**, along with the
  browser tab - it starts the server only, for a headless launch.
- **If the window fails to open** (a missing runtime dependency, a broken driver,
  or it does not finish loading within a few seconds), `localm gui` falls back to
  opening a browser tab automatically. It never fails to open something.
- There is no CLI flag to force the app window on for a single run; only the
  Settings toggle and whether the extra is installed decide.

Platform status:

- **Windows:** pywebview's default backend, hosting Microsoft Edge WebView2.
- **Linux:** installs entirely via `pip`, no system packages and no `sudo` -
  `localm[desktop]` pulls in `pywebview[qt]` rather than pywebview's default GTK
  backend, because GTK's Python bindings are a system package pywebview cannot see
  from inside localm's isolated virtual environment. A couple of small system
  libraries the Qt/X11 stack itself needs (e.g. `libxcb-cursor0`) are commonly
  already present on a desktop install but not on a minimal one; `setup.sh` notes
  this when you choose the app window.
- **macOS:** uses pywebview's default backend (WKWebView). Not independently
  verified on this project's own hardware - the same caveat this project applies
  to its other macOS-only paths.

This is a separate control surface from the tray/status window described below:
with the `desktop` extra installed, both can be visible at once - a tray icon or
status window for Open/Copy address/View logs/Restart/Stop, and the app window
itself showing the GUI content.

## First launch on Windows

The first time you start `LocaLM.exe` (from a double-click, the desktop shortcut, or
the Launcher), Windows SmartScreen may show a blue "Windows protected your PC" screen
warning about an unrecognized publisher. This is expected: `LocaLM.exe` is a copy of
your own interpreter built locally by `make-launcher` (see "What it actually is"
below), not signed with a paid code-signing certificate, so Windows has no publisher
reputation to check it against. Click **More info**, then **Run anyway** to continue.
There is no way to remove this warning without a paid code-signing certificate; it is
not a sign that anything is wrong with the build, and it only appears for the
`LocaLM.exe` launcher itself, not for `localm gui` run directly in a terminal.

## Build it

```
localm make-launcher            # build for this OS (idempotent)
localm make-launcher --force    # rebuild (use after a Python upgrade)
```

- **Windows** creates `.venv\localm-app\LocaLM.exe` and launches it as
  `LocaLM.exe -m localm gui`. Task Manager then shows `LocaLM.exe` as a single
  process (no `python.exe`). `setup.bat` points the desktop shortcut at it.
- **Linux** creates `.venv/bin/LocaLM` and writes a `LocaLM.desktop` you can copy
  to `~/.local/share/applications/`. A process monitor then shows `LocaLM`.
  (`setup.sh` also writes an application-menu entry.)

Everything stays **inside the clone's own `.venv`** - nothing is installed
system-wide.

The graphical launcher (`launcher.pyw` / the "Launcher" shortcut) automatically
spawns modes through `LocaLM.exe` when it is built, so the Web GUI's console-less
background-app path shows LocaLM, not python. When the Web GUI is launched as
`LocaLM.exe` in its own console (a double-click or the launcher's own console), it
hides that console once the server is up - the tray and status window are the
surface - so it runs like a real background app. Running `LocaLM.exe -m localm gui`
inside an existing terminal leaves that terminal visible.

## What it actually is

`LocaLM.exe` is a **branded copy of the venv's Python interpreter**, not a
compiled binary. It is small and low-risk: because it is the same interpreter
running the same installed package, plugin discovery, the native llama.cpp
libraries, and the GUI assets all keep resolving exactly as they do for a normal
`localm gui`. Only the process **name** and **icon** change.

Two Windows details make it robust:

- We copy the **real interpreter** (`sys._base_executable`) plus its loader DLLs
  (`python3*.dll`, `vcruntime*.dll`), not `.venv\Scripts\python.exe`. Under a
  uv-managed Python the venv `python.exe` is a *trampoline* that launches the base
  interpreter as a child - copying it would leave the real server named
  `python.exe`. Copying the base interpreter yields one genuine `LocaLM.exe`.
- The LocaLM icon is written into the copy's PE resources, and the running process
  also sets its taskbar identity (AppUserModelID) and console-window icon. A
  restart from the tray or Settings re-execs the same `LocaLM.exe`, so the
  identity survives in place.

`make-launcher` self-checks the result and refuses to leave a launcher that does
not start. If it cannot build one, `localm gui` still works (Task Manager may then
show `python.exe`).

## Linux caveat

The Linux `bin/LocaLM` is a copy of the interpreter; if that interpreter is not
relocatable (its shared runtime is not resolvable next to the copy), the launcher
falls back to the venv Python and a process monitor shows `python`. The robust
"real app" path on Linux is an **AppImage** (bundles the interpreter and a
`.desktop`); that is planned future work, not shipped yet.

## Why not a full freeze (PyInstaller/Nuitka)?

A frozen single-file binary was evaluated and deferred:

- localm's native inference libraries are **provisioned per-hardware at runtime**
  (`localm setup-llama`) and resolved by importing `localm_llama_runtime`. A frozen
  monolith would have to bake one backend in at build time, fighting that model.
- The optional HuggingFace stack (torch/transformers/rocm-sdk) is multi-gigabyte
  and fragile to freeze; a core-only freeze would silently drop HF-model support
  from the frozen exe.

The branded-interpreter approach gives the native identity today without those
trade-offs. A true compiled binary remains an option for a later release; the code
paths that matter for it (`applaunch`, the restart argv, the home/asset resolvers)
are already isolated.
