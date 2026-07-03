# Native app identity (LocaLM.exe)

By default the localm server runs as `LocaLM.exe` in Task Manager (not
`python.exe`), launches from a double-click or a desktop shortcut, and carries the
LocaLM icon on the taskbar, in the tray, and on the file itself. `setup.bat` /
`setup.sh` build this launcher for you; you can also (re)build it yourself.

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
