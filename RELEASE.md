# Release process: pre-publish live verification

The last gate before a release is published. CI proves the code passes the suite; the
smoke gate proves the build imports and runs; THIS proves that every feature the
changelog advertises actually WORKS front to back, on a COLD install, by real use, not
"it loaded fine". CI cannot do this: it needs real model runs, the GUI, media backends,
and judgment. So a human/agent does it, by hand, against the release build, and records
the verdict under `dev-notes/release-verify/<sha>.md` (local, gitignored: it is
machine-specific release evidence). `make_release --publish` refuses to publish without a
PASSING record for the exact release commit.

## Non-negotiable rules
- Exercise the REAL path (CLI and GUI where a feature has both). "Loads without error" is
  NOT a pass; the feature must produce the result the changelog claims.
- NEVER fake a PASS. If something is broken, it is FAIL and the release does not ship.
- BLK is RARE and only for a real hardware limit of THIS machine that localm genuinely
  cannot provision around: a cross-vendor GPU code path (the NVIDIA/Intel CUDA paths on
  the AMD dev box - route those to the NVIDIA box, do not skip). "No backend" is almost
  NEVER a valid BLK: localm provisions its own - `localm comfy setup` installs an isolated
  ComfyUI, `localm pull` gets models, `localm setup-embeddings` gets the embedder. If a
  feature needs a backend, STAND IT UP and verify for real; do not declare it blocked.
  BLK never silently becomes PASS, and a whole BLK category is called out in the record so
  the maintainer decides where to verify it.
- Every ADDED and CHANGED item must be exercised. FIXED/SECURITY items that cannot be
  re-triggered by hand are covered by their regression tests (note it); the ones with an
  observable behavior are spot-checked here.

## Scope: real-run what CHANGED, spot-check the rest
A full real-run of a feature (generate an image, run every model, index a big corpus) is
required only when THIS release CHANGED that path. An unchanged path is already covered by
its tests and the import/smoke gate, so re-generating an image or re-running every model
every release just makes the pass take forever. So:
```
python scripts/release_verify.py changed --since <last-release-tag>   # e.g. v0.1.0
```
Real-run the features whose code moved (map the changed top-level areas to sections 2-4);
for the rest, spot-check (launch + light exercise). The core loop (section 1) always runs.
The backend matters: provision the backend users actually run (the AMD GPU on the dev box),
NOT cpu, or a GPU-path regression is missed. cpu/vulkan is a secondary portability check.

## 0. Cold install (isolated, matches what a user gets)
```
# build the exact artifact for this commit (no publish)
python scripts/make_release.py --key <signing-key>        # -> dist/localm-<ver>.zip
# cold-install it: fresh venv + install + native runtime (backend defaults to the
# hardware-detected one, e.g. the AMD GPU - NOT cpu, so the real GPU path is exercised)
python scripts/release_verify.py cold-install --zip dist/localm-<ver>.zip --dest <TMP>/cold
# drive everything below from the cold install, with a THROWAWAY home:
export LOCALM_HOME=<TMP>/home          # never the real ~/.localm or a repo home/
PY=<TMP>/cold[/wrapper]/.venv/Scripts/python.exe   # (bin/python on Linux)
```
Add `--extras coder,voice,monitor,rag` to match the target, or `--backend cpu`/`vulkan`
for a secondary portability check. Everything below runs as `"$PY" -m localm ...` (and via
the GUI it serves).

## 1. Core loop (MUST pass)
| # | Exercise | Expected | P/F/BLK |
|---|---|---|---|
| 1 | `localm doctor` | all core checks green (or a documented, benign note) | |
| 2 | `localm pull <tiny gguf>` AND a small HF model (`HuggingFaceTB/SmolLM2-135M-Instruct`, ~270MB; or `Qwen/Qwen2.5-0.5B-Instruct`) | both download and register, show in `localm list` | |
| 3a | GGUF backend: `localm run <gguf> -p "..."` | a REAL answer via the bundled llama.cpp (non-empty, coherent) | |
| 3b | HF/torch backend: `localm run <hf-model> -p "..."` | a REAL answer via transformers. BOTH inference backends are advertised, so both are verified. Where the `[gpu]` extra is installed, confirm it uses the GPU (SmolLM2-135M at ~1 tok/s means it silently fell back to CPU) | |
| 4 | `localm gui`, open in a browser, pick the model, send a message | streamed reply in the GUI. The DEFAULT (privacy) mode is session-only: the chat clears on reload and the sidebar shows "privacy mode - this session only". Re-run with `--mode full` (or `log`) and confirm the conversation then persists across a reload | |
| 5 | `localm serve` + an OpenAI-style `curl /v1/chat/completions` (Bearer key) | a real completion; `/v1/models` marks the active model | |

## 2. Added (every one must be exercised)
| Feature | Exercise | Expected | P/F/BLK |
|---|---|---|---|
| Managed ComfyUI (image) | `localm comfy setup` (fresh isolated install) OR point at your existing ComfyUI; then generate an image via localm | ComfyUI installs isolated + coexists; a real PNG is produced. REQUIRED - localm provisions the backend, do not BLK | |
| Managed ComfyUI (music/video) | generate a clip via the managed ComfyUI (the __func__ shim, #482/#494, is what makes this work) | real audio/video output; if an upstream modality is broken past the shim that is a FAIL/finding, not a skip | |
| Signed self-update | `localm update --check` | reaches the proxy, reports current vs latest; apply verifies the signature (BLK the full apply if no newer signed release) | |
| Report a problem (broken install) | run the `report-issue` entry; also try it with the venv removed | previews exactly what will be sent; account-less; works with no working install; a declined/failed send never claims success | |
| Unified model browser | GUI Models page | registry types shown, ComfyUI scan runs, type filter works | |
| RAG richer indexing | add a text file, a .zip, and an image to a collection; query it | all index (archive extracted, image described); query retrieves; chunks show a format tag | |
| MCP tools | `npx mcporter` (or an MCP client) over stdio: list + call | setup/model-removal/diagnostics/plugin tools present; `chat`/`list_models` work; destructive tools annotated | |
| Model type-detection / --store | `add` a `.safetensors` dir and an ambiguous file; `add --store <file>` | correct type or `unknown` sentinel; `--store` copies into the store | |
| GPU selector / unload | GUI/CLI main-GPU select + model unload | selection honored; unload frees VRAM (watch `system_stats`) - verify the AMD path here; only the NVIDIA/Intel selection routes to the NVIDIA box | |
| Setup dead-ends | (covered by the cold install: uv auto-bootstrap; re-run setup-llama -> replace prompt) | install succeeded without a dead-end; re-provision prompts | |

## 3. Changed / QoL
| Item | Exercise | Expected | P/F/BLK |
|---|---|---|---|
| ComfyUI status polling | GUI network tab with ComfyUI configured | status checked on demand, not every 5s | |
| Native-stderr dedup | run a long generation, watch the log | repeated native lines collapsed | |
| RAG heuristic-first | index with NO chat model loaded | indexes without stalling / timing out | |
| Workflow-override migration | put a legacy `*_workflow*.json` under `localm/...`, start | migrated into `home/workflows/`, survives | |
| Dependency versions | `localm doctor` | lists the key dependency versions (huggingface-hub, fastapi, uvicorn, etc.), each resolved to an installed version - none reported "not installed" | |

## 4. Fixed / Security (spot-check the observable ones)
| Item | Exercise | Expected | P/F/BLK |
|---|---|---|---|
| Multi-model VRAM eviction | load two models that exceed VRAM | safe eviction, no crash/corruption | BLK if VRAM allows both |
| /v1/embeddings no force-load | POST `/v1/embeddings` with only a chat model registered | does not force-load the chat model | |
| RAG folder index skips secrets/weights | index a folder containing a `.gguf`/`.env` | weights + secrets skipped, not indexed | |
| Bug reporter: no GitHub login path | walk every report path | none ever asks to log into GitHub | |
| ComfyUI launch shell-injection | (code-level; covered by regression test) | note: verified by test, not re-triggered here | BLK/test |
| Job API privilege escalation | with a scoped key, confirm owner jobs are not reachable anonymously | isolation holds | |
| Do-not-hide-problems facades removed | (covered by regression tests) | note: test-covered | BLK/test |

## 5. Record the verdict
Write `dev-notes/release-verify/<full-sha>.md` (local, gitignored) with:
```
# Release verification <short-sha>  (localm <version>)
Date: <YYYY-MM-DD>   Platform: <os/gpu>
VERDICT: PASS            # or FAIL - only PASS lets make_release --publish proceed
- section 1 core loop: PASS (evidence...)
- section 2 Added: PASS  (per-item notes; BLK items + reason)
- ... FAIL items block the release; BLK items listed with why.
```
`make_release --publish` reads `VERDICT: PASS` for the current HEAD sha and refuses
otherwise. A record for an older sha does not count (it is keyed to the commit). If any
item is FAIL, fix it and re-verify; if BLK covers a shippable-elsewhere category, the
maintainer decides.
