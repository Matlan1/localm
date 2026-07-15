# Changelog

All notable changes to localm are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and localm aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Being pre-1.0,
minor versions may include breaking changes.

Each release adds its section on top. Published (versioned) sections are the
permanent public record of what shipped and are never rewritten; the in-progress
`[Unreleased]` section is maintained until it is cut into a release.

## [Unreleased]

### Added
- **Choose how your embedding model is pooled.** Settings > Models has a new
  Embedding pooling option. The default (`mean`) suits the bundled `bge-small`
  and `nomic` choices and matches everything you have already indexed, so nothing
  changes unless you want it to. If you point `embedding_model` at a
  decoder-based embedder such as Qwen3-Embedding, which is built for `last`-token
  pooling, localm now warns you that it is being pooled the wrong way and names
  the setting that fixes it, instead of silently giving you weaker embeddings.
  Changing the setting means re-indexing, since existing vectors were built the
  old way.
- **Offer to download a missing Flux model file.** When Image/Video/Music
  generation detects a missing ComfyUI model file that has a known-good source
  (currently the Flux UNET, text encoders, and VAE), it now offers to download
  it for you - showing the exact repository, file, and size, with a real
  confirm click before anything is fetched. It lands directly in the right
  ComfyUI models folder, whether you use the managed ComfyUI or your own
  install. A model without a known source still shows the same clear message
  as before, telling you to install it yourself.
- **See which memories a reply used.** When chat memory is on, each reply that drew
  on remembered facts now shows a small "🧠 N" chip; hovering lists the facts used,
  and clicking opens the memory panel so a wrong or stale fact is easy to correct.
  The chip's tooltip also says when recall fell back to keyword-only matching (for
  example, before an embedding model is installed), so semantic recall being off is
  no longer invisible.
- **Launch ComfyUI and pick its models from the Images/Music/Video pages.** The
  Workflow panel now has a "Launch ComfyUI" button (starts or confirms it, then
  opens it in a new tab) and, once it's reachable, a dropdown for every model-file
  slot (UNet/checkpoint, CLIP, VAE, ...) the active workflow exposes - no more
  needing to hand-edit the workflow JSON to pick a different model. Picks are sent
  along with the next generation and are remembered while you stay on that workflow.

### Changed
- **Memory recall is now relevant-only.** Chat memory used to inject the same handful
  of remembered facts into every reply regardless of the question, adding noise and
  distracting smaller models. It now surfaces only the facts that actually relate to
  what you asked - by keyword, or by meaning when an embedding model is installed -
  and stays quiet when nothing is relevant. Follow-up questions ("yes, do that")
  recall better too, and a long remembered fact is shown in full instead of being cut
  off mid-sentence.
- **Each conversation is remembered on its own.** Automatic memory used to blur all of
  your recent sessions into a single one-line recollection (and re-summarise the same
  ones every time it ran). It now records one summary per conversation, tagged to that
  conversation, and never re-processes a session it has already summarised - so past
  topics are recalled distinctly instead of collapsing into one vague note.
- **One control for which ComfyUI localm uses, not two.** Settings > Media used to
  have both a "Use localm's own managed ComfyUI" checkbox and a separate "ComfyUI to
  use: own/user" dropdown - two controls for one decision, and confusing when they
  disagreed (checkbox on, dropdown set to "user"). There is now just the dropdown:
  "own" routes to localm's managed ComfyUI once you set one up (`localm comfy
  setup`), "user" always uses your own install. Your own ComfyUI's settings and
  localm's managed instance both keep their state regardless of which one is
  currently selected, so switching back and forth never loses either. One real
  behavior change worth knowing: the old design needed a separate explicit "enable"
  step after setup; that step is gone. If you ran `comfy setup` on an earlier
  version but never flipped that switch, generation will now start using the
  managed instance automatically next time - `localm comfy status` shows what is
  currently targeted, and `localm config comfy_target user` (or the same dropdown
  in Settings > Media) opts back out if you'd rather it stay off.
- **The ComfyUI re-scan is easier to find.** The button that re-scans your configured
  `comfy_workdir` used to live in a spot that only appeared after switching to the
  Diffusion/Encoders/VAEs/LoRAs/Other tab - invisible by default and on the LLMs tab,
  with nothing hinting it existed. It's now a labelled **Re-scan ComfyUI folder**
  option in the "Add a model" card, next to **Import from ComfyUI…**, and it stays
  visible no matter which tab is active.

### Fixed
- **Your own saved facts come back when you ask about yourself.** Without an
  embedding model installed, memory could only match facts that shared an exact
  word with your question, so asking "what is my name" never surfaced "User is
  called Sam" - it silently answered as if you had saved nothing. When you ask
  about yourself, memory now falls back to a couple of your own saved facts
  instead of staying silent. Questions that are not about you still stay silent,
  and installing an embedding model (`localm setup-embeddings`) is still what
  gives you full meaning-based recall.
- **Picking a Main GPU no longer stops large transformers models from loading.**
  With a Main GPU selected, localm pinned the whole model onto that one card, so a
  transformers model bigger than its free memory failed with an out-of-memory error
  even though it used to load (more slowly) by keeping part of it in system memory.
  It now still uses only the card you picked, but falls back to system memory for
  whatever does not fit, as it did before. The same fallback was restored for a
  configured multi-GPU split.
- **A failing command no longer reports success.** Certain real failures (reading a
  folder where a file was expected, an invalid path, some native calls) were
  mistaken for "the output pipe closed", so localm exited quietly with a success
  code and filed no bug report - telling you, and any script checking the result,
  that a command had worked when it had not. Those now fail properly and offer the
  usual report. Piping into `head` or `findstr` still exits quietly, as it should.
- **A machine on a DSL/PPPoE line gets a working certificate again.** localm leaves
  a VPN's address out of the certificate it makes for itself, but the check matched
  far too loosely: an adapter whose name merely contained "ppp", "tun" or "tap" -
  including a real PPPoE internet link, or a network card with an unlucky name -
  was mistaken for a VPN, so its address was left out and browsers reaching localm
  there reported a certificate mismatch. Real VPN adapters are still excluded.
- **Privacy mode no longer lets a remembered fact reach the debug log.** If you ran
  in privacy mode with the debug log on for troubleshooting, and the embedding model
  hiccuped while saving a memory, a snippet of that memory - a fact about you, or a
  summary of your conversations - was written into the log file on disk. Privacy mode
  now withholds it, as it already did everywhere else. The failure itself is still
  reported, so the problem is not hidden, just the content.
- **Chat no longer pauses on every turn that uses a memory.** Each reply that recalled
  a fact re-read and rewrote the whole memory store, including its embedding file, on
  the server's main thread - so with a large memory the entire server (every chat, every
  progress update) stalled briefly on each turn. That work now happens off to the side.
  It also means loading the embedding model can free up video memory properly instead of
  squeezing in alongside the chat model.
- **A long prompt on a slow setup is no longer killed as "stalled".** Replies that
  needed more than two minutes to read the prompt before writing their first word -
  common when running on CPU, with most layers off the GPU, or with a very long
  prompt or document - were treated as a hung model and cancelled, and retrying hit
  the same wall every time. Reading the prompt now gets its own generous allowance,
  separate from the per-word one, and you can raise it in Settings > Engine
  ("First-token timeout") if your machine needs longer. A genuinely hung model is
  still detected.
- **Stopping a reply can no longer break that model until you restart.** If you
  stopped a reply while the model was busy in a step it could not interrupt, the
  model was shut down but still looked loaded, so every later message to it failed
  with an internal error, permanently. It now reloads itself on the next message.
- **An update no longer undoes itself.** After "Update now", localm checks that the
  new version comes back up healthy and rolls back if it does not. That check looked
  for the server on the port it had *before* restarting, which is not always the port
  it comes back on - so a perfectly good update could be silently reverted a minute
  or two later. Restarting now keeps the server on the port it was already using,
  which also means a plain "Restart" no longer moves it out from under an open tab.
- **Bug reports no longer blame an old freeze for a new problem.** localm attaches a
  freeze snapshot when it has one, but it attached the newest one it could find,
  however old and whatever it was from - so a report about something else entirely
  (a wrong answer, a failed download) could arrive captioned with an unrelated freeze
  from weeks ago, pointing at the wrong cause. Only a freeze from the run you are
  reporting on is attached now.
- **A crash report no longer arrives with the crash missing.** When the error was
  large (a deep native model-load failure, which is exactly when localm asks you to
  file a report), it did not fit the report's size limit and was dropped whole,
  leaving a report that said an error had been left out but not what it was. The
  error is now included, trimmed to fit and marked as trimmed, keeping the part that
  names the failure.
- **Knowledge folders no longer silently skip symlinked files.** If a folder you
  added contained a symlink to a document stored elsewhere (a very common way to
  collect docs from several projects), that document was quietly left out: you
  saw "indexed N chunks" with no hint anything was missing, and searching for its
  content found nothing. Linked files are indexed again. Linked folders are still
  not followed, so a looping shortcut cannot hang indexing.
- **One damaged archive no longer aborts a whole folder index.** A truncated or
  corrupt `.gz`/`.tar.gz` (a half-finished download, say) crashed the entire
  indexing run, so every other file in the folder went unindexed, and uploading
  one to chat returned a server error. The bad file is now reported on its own
  and everything around it indexes normally.
- **A video or database in a multi-select no longer blocks the whole add.**
  Picking several files where one happened to be a `.mp4`, `.db`, `.7z`, or model
  weights file rejected the entire request and indexed nothing. The files that do
  contain text are now indexed, and the one that doesn't is listed as a single
  skipped file with the reason. Key and credential files are still refused
  outright.
- **`.env.example` files are indexed again.** The template files that document
  which settings a project needs (`.env.example`, `.env.template`, `.env.sample`)
  were being dropped from knowledge folders as if they were secrets. They are
  documentation and are indexed again. Real `.env` files, including `.env.local`,
  are still skipped.
- **`localm rag repair` works again from scripts and scheduled jobs.** On a
  collection with semantic search, repair asked before dropping its embeddings -
  but run without a terminal (cron, CI, a script) there was nothing to answer, so
  it exited with an error and repaired nothing, on exactly the stale indexes it
  exists to rebuild. It now keeps the embeddings and repairs, telling you it did.
  Pass `--yes` to drop them instead, or `--embed` to recompute them.
- **Document search no longer embeds with the chat model itself.** If you ran a
  HuggingFace-format model, localm quietly made embeddings out of the chat model
  rather than the small dedicated embedding model, even when you had one
  installed. Those vectors look perfectly healthy but barely tell related text
  from unrelated (on one 0.5B chat model the *most unrelated* pair we tested
  scored higher than the *least related* one), so document (Knowledge/RAG) search
  and the `/v1/embeddings` API silently returned worse results with nothing to
  indicate it. localm now uses your installed embedding model, and when none is
  installed it says so and falls back to keyword search instead of handing back
  unusable vectors. GGUF models, the default, were never affected, and neither
  was chat memory.
- **A model you switched away from mid-load can be used again.** If you picked a
  model and then quickly picked a different one, the first model could be left
  permanently unusable: every later message to it failed with "Model load was
  superseded by a newer request", even though nothing was loading any more, until
  you explicitly switched to it again. It now loads normally.
- **Stopping or restarting while knowledge is indexing no longer leaves a model
  stuck in GPU memory, or hangs.** The embedding helper could survive the restart as
  an orphan still holding its model in VRAM, so a restarted server ran a second copy
  beside it and "Stop" did not actually free everything. Both now release it - and
  stopping while an embedding model is still loading no longer waits on that load
  before it can finish.
- **An embedding error no longer strands a helper in GPU memory.** If embedding a
  single piece of text failed, localm dropped its still-running embedding helper
  instead of reusing it: the helper kept your embedding model in GPU memory with
  nothing able to reach it, and the next request started a second one alongside. It
  now keeps the working helper, and only replaces one that has actually gone.
- **The app stays responsive while models load and unload.** On a multi-GPU setup
  with a specific main GPU selected, each load or unload could briefly freeze every
  other request (up to a few seconds) while it wrote coordination state and probed
  the GPU. That work now happens off the request-handling path.
- **A second localm no longer loses its models for nothing.** When two localm
  instances share a GPU, one could ask the other to drop every model it had loaded
  even when that could not possibly free enough room, and could keep re-asking in a
  loop that never finished. It now only asks when it would actually help, and asks
  each instance once. On a multi-GPU split it correctly counts every card the split
  uses, so a model that needs both cards still gets the room it needs instead of
  failing with "VRAM exhausted".
- **An empty `auth.key` no longer locks you out of your own server.** If an
  `auth.key` file existed but held no key - you created it by hand to paste a key
  in later, an editor or PowerShell saved it empty, or a backup left it truncated
  - localm treated it as "a key is set" while having no key to check against, so
  every request was rejected with 401 and there was no way back in short of
  deleting the file by hand. A key file that holds no key now means exactly what
  it says (no key, so the server runs open, as it did before), and localm says so
  in the log rather than changing your server's security posture silently. A key
  file that cannot be READ still locks the server, as it should - that is the case
  where localm genuinely cannot tell whether you have a key. A key saved with a
  byte-order mark (what Windows editors and PowerShell add) now also works
  instead of silently never matching.
- **Knowledge and memory no longer mix up results when two things are indexed at
  once.** If a background knowledge re-index overlapped with a chat memory lookup,
  the two could receive each other's results, storing or matching against the wrong
  text with no error shown - which could quietly leave wrong entries in your
  knowledge and memory stores and make recall return unrelated results. Overlapping
  requests are now kept apart, so each one always gets its own.
- **Settings changes and model-list updates could stop working for a whole run.**
  If a localm process was killed while saving settings or updating the model list,
  the lock file it left behind could later be misread as "this same process is
  already writing" - wedging every settings save and every model pull, remove, or
  alias for the rest of that run, with no way out but a restart. A leftover lock is
  now always reclaimed, and only a genuine re-entrant write is refused.
- **Saving settings no longer freezes the whole app.** Saving settings in the GUI
  while another localm process was also writing them (for example `localm config` in
  a terminal) could freeze the entire server - chat, streaming, health checks - for
  up to 10 seconds. That wait now happens off the request-handling path.
- **An unreadable config no longer stalls every request on Linux and macOS.** When
  config.json or registry.json could not be read because of file permissions, localm
  retried for about a second before falling back, on every affected read - including
  the one on the request-authentication path. A permission denial is now recognised
  as permanent and handled immediately, while the brief Windows retry (which rides
  out antivirus and indexer locks) is unchanged.
- **The Plugins page's "External plugins" card works again.** Listing, installing,
  and removing a third-party plugin from the GUI failed with "Could not load
  plugins: Not Found" for everyone, because the page still called an API that had
  been removed. The card now uses the same plugin engine the rest of the app (and
  `localm plugin install`) uses, and it again shows each external plugin's version,
  description, and the tools it exports to the coder.
- **Chat no longer refuses right after you load a model.** Picking a model in the
  sidebar and typing straight away could be rejected with "No model loaded - load a
  model on the sidebar before chatting" for up to 30 seconds, even though the model
  had loaded fine. The app now registers the model as active the moment the load
  lands.
- **Aliasing a model tells you the name it really created.** An alias containing a
  space (or a slash or colon) is stored in a cleaned-up form, but the GUI reported
  the name you typed, so "daily driver" was announced while only "daily-driver"
  existed. It now shows the actual name. If that cleaned-up name is already taken,
  the alias is refused with a clear message instead of reporting success without
  creating anything.
- **Moving a model onto the same drive no longer reports a false "not enough disk
  space".** `localm add <path> --on-duplicate move` refused whenever the drive had
  less free space than the model's size, even though moving within one drive needs
  no extra room at all - exactly the case where you would choose move over copy.
- **A downloaded HuggingFace model is no longer mislabelled "unknown".** Pulling a
  full transformers model whose repository lacks a type tag registered it as
  "unknown", so it was skipped by automatic chat selection and `localm gui` could
  open with no model even though the download worked. The type is now read from the
  model's own config file when the repository does not say.
- **A HuggingFace download can resume after the disk fills up.** Retrying required
  free space for the whole model again, ignoring the part already downloaded, so the
  retry it was meant to enable could never start. It now only asks for the space the
  remaining files need.
- **A ComfyUI model download no longer reports success without downloading.** If you
  already had the same file registered in localm, the "Missing model -> Download"
  offer reported success while nothing was written to the ComfyUI folder, and
  generation then failed with the model still missing.
- **Rotating your API key no longer breaks your own scheduled jobs.** If you had a
  scheduled coder job with shell access enabled and then rolled or cleared your API
  key, the job silently kept running but lost its shell step, forever, with no
  notice. Your own jobs now keep their shell access across a key change. A job whose
  access genuinely was revoked still loses shell, as intended, but now says so in
  the job's output instead of quietly doing less.
- **The coder can delegate real work again in the terminal.** When running
  `localm coder` without `--yes`, any task the assistant handed to a sub-agent had
  every file edit and command blocked outright, so the delegated work just failed.
  Sub-agents now ask you to approve their changes at the same prompt the main
  session already uses, so you can say yes and the work happens.
- **Quitting the coder is instant again.** Ending an ordinary look-around session
  could hang for many seconds while it ran a full model reflection you never asked
  for - triggered by nothing more than a couple of incidental errors, or by your own
  Ctrl-C. Sessions that changed no files now exit immediately; a run that genuinely
  failed still records what it learned.
- **The app stays responsive while a coder session closes, and while the model
  picker loads.** Closing a coder session from the web UI, and opening the Workflow
  panel on the Images/Music/Video pages, each ran slow work on the server's request
  path - freezing every other request, including chat streaming, until it finished.
- **A crashed ComfyUI is noticed instead of being reported as running.** If ComfyUI
  died mid-session (an out-of-memory on a big render, a crash, or you closed its
  window), localm kept believing it was up, so every later generation failed with a
  confusing error and it would not restart it for you. It now re-checks, and starts
  ComfyUI again or tells you exactly what to do.
- **Generating or deleting media no longer fails when antivirus is mid-scan.** On
  Windows, if a background process (an antivirus scan, the search indexer, a backup
  tool) happened to have the gallery's index file open at the wrong moment, the
  request failed with an error even though the file itself was fine, and left a
  stray temp file behind each time. It now waits briefly for the file to free up.
- **Saving a memory no longer freezes the app.** On some setups, saving, editing,
  or adding a remembered fact could make the whole app stop responding for several
  minutes the first time an embedding model needed to load, because that load ran on
  the request-handling path. Memory writes now load the embedder off that path, so
  the app stays responsive.
- **Clearer errors when indexing an image fails.** If the image-description step
  fails (a timeout, a connection error, or an error returned by the model), knowledge
  indexing now reports the real reason, instead of a misleading "returned an empty
  description" message that hid the actual failure.
- **The Models page shows the real error instead of "No models yet".** When the
  Models page could not load your model list because the session had expired or the
  API key lacked permission, it used to fall back to the empty "No models yet" state,
  as if you had none. It now opens the key prompt on an expired session and otherwise
  shows the actual status ("Could not load models (HTTP 403)"), so a sign-in or
  permission problem is no longer mistaken for an empty model library.
- **"Keep diagnostics" now warns when it cannot write its log.** If you turned on
  keeping a diagnostic log for bug reports but the log file could not be created (for
  example an unwritable or full data folder), localm used to start silently as though
  it had, leaving your bug reports quietly without one. It now prints a warning at
  startup that the diagnostic log could not be enabled, instead of appearing to
  succeed; startup itself is unaffected.
- **Adding documents to a knowledge collection reports the server's real error.**
  When a document add was refused with a conflict, the GUI could show an internal
  "body stream already read" message instead of the server's actual reason. It now
  surfaces the real error detail.
- **A damaged media-ownership record no longer exposes everyone's generated media.**
  On a multi-user server, the image, music, and video galleries record which key
  generated each file so a scoped key only ever sees its own. If that on-disk
  ownership record became unreadable (corrupt, truncated, or momentarily locked),
  localm used to treat every file as unowned and let any key view, download, delete,
  move, or rename all of them. It now fails closed: a scoped key is denied until the
  record is repaired, while the owner/admin key (and single-user open mode) keeps full
  access so the gallery stays usable. The record is also written atomically now, so an
  interrupted save can no longer leave it half-written in the first place.
- **Semantic knowledge search now works on password-protected servers.** When localm
  is started with an API key saved to disk (`localm key generate`, or the launcher),
  indexing a document used to silently fall back to keyword-only search: the server
  could not authenticate its own embedding call and quietly dropped to lexical
  retrieval while the embedding model sat ready. Keyed servers now embed correctly -
  whether the key comes from the environment or the saved key file - so semantic
  (hybrid) search works again, including `localm rag add/query --embed`.
- **A collection stuck on BM25 now says so where you can act on it.** The Knowledge
  page's "Ready ... semantic search is on" banner used to read as a blanket
  all-clear, even though it only covers indexing from that point forward - a
  collection indexed before the embedding model was set up (or whose vector index
  went stale) silently stayed lexical-only, with the only hint a small warning
  buried in that collection's info modal. A BM25 row now gets a visible "reindex
  needed" badge and a highlighted reindex button when the embedding model is ready,
  and the banner itself now says existing collections may need reindexing.
- **You can index documents on a headless `localm serve`.** A bare API server (no GUI)
  could not index into a knowledge collection at all - `POST /api/rag/collections/
  {name}/add` and `/upload` refused with a "run localm gui" error. They now index the
  documents directly and return the result, so the documented REST API works without
  the GUI (semantic embeddings included, when an embedding model is installed).
- A non-finite number (`NaN` or `Infinity`) can no longer be saved as a config value.
  Setting one, e.g. `localm config temperature nan` or a crafted Settings request, used
  to be accepted and written to `config.json`, after which the Settings page failed to
  load (HTTP 500) every time until the file was hand-edited. Such values are now
  rejected up front with a clear message and never persisted.
- A single malformed entry in `registry.json` (from a hand-edit or a partial write)
  no longer crashes `localm list` / `rm` / `add` / `pull` with an "unexpected error"
  and a bug-report offer. The bad entry is shown as `[corrupt]`, can be removed with
  `localm rm <name>`, and your other models still list normally (the same resilience
  applies to the MCP server's `list_models`).
- That same corrupt-entry resilience now also covers the surfaces that were still
  missed: the GUI **Models** page and the `/api/vram-estimate` readout, the model
  detail API, `localm pull`, and the ComfyUI model **Scan** no longer error out on a
  single malformed registry entry - the bad entry is skipped and your good models
  keep working.
- `localm pull <local file> --sha256 <hash>` now actually verifies the checksum and
  refuses on a mismatch, instead of registering the file and reporting success while
  silently ignoring the hash you asked it to check.
- `localm alias <existing> <new-name>` now cleans the new name the same way every
  other model name is cleaned, so it can no longer write an unsafe (`../x`, `a/b`) or
  empty key into the registry.
- A nonexistent model name made only of dots (e.g. `localm run ....`) now reports
  "Model not found" instead of crashing with an internal error on Windows.
- **A nested `LOCALM_HOME` is created for you:** pointing `LOCALM_HOME` at a fresh
  path a couple of levels deep (e.g. `D:\localm\data`) used to crash if the parent
  folders did not exist yet. localm now creates the whole path, like `mkdir -p`.
- **No more crash or lost settings when two processes touch config at once
  (Windows):** on Windows, saving `config.json` / `registry.json` could fail if
  another localm process, antivirus, a backup tool, or Windows Search happened to be
  reading the file at that instant, OR if two localm processes tried to save the same
  file at the same time (a CLI `pull`/`config` while the GUI is running); a read at
  that moment could momentarily fall back to defaults. Saves now use a unique temp
  file per write and both saves and reads ride out the brief lock, so concurrent
  access no longer crashes a save or drops settings.
- **A concurrent config/model-registry change from a SEPARATE localm process is no
  longer silently lost:** `localm config <key> <value>` (or a client's `PATCH
  /v1/config`) racing a different localm process's own config change - e.g. a
  running server's settings save, or two CLI invocations - could each read the
  file before the other had written it, so whichever finished last silently
  overwrote the other's already-saved change; the same applied to two `localm
  pull`/model-registry writes racing each other. Config and registry updates now
  hold a lock across the whole read-then-write, so a concurrent writer in another
  process waits its turn instead of clobbering the first one's change.
- **A `config.json` that is not a JSON object is no longer ignored silently:** if
  the file somehow becomes valid JSON but not an object (a list, a bare string, a
  number), localm now says it is ignoring it and using defaults, instead of quietly
  dropping your settings with no explanation. The file is left untouched so you can
  recover it.
- **Model loading no longer fails under the LocaLM.exe launcher (Windows).**
  Loading any GGUF model after launching via the branded LocaLM.exe (the desktop
  shortcut / launcher's default) failed with a misleading "Native llama runtime
  failed to load" error, in two different ways: first "[WinError 2] The system
  cannot find the file specified" (Python's own multiprocessing redirects a
  spawned worker to the base interpreter behind a renamed launcher, and could not
  find it under its new name), and then, once that was redirected to the venv's
  own interpreter instead, "[WinError 6] The handle is invalid" (that interpreter
  is itself a launcher that re-spawns the real one as a further child, one hop too
  many for Windows to hand the worker its synchronization handles correctly).
  Model loads and voice transcription now spawn correctly under LocaLM.exe.
- **`localm doctor` now actually verifies model loads will work.** Its native
  runtime and GPU checks run their probes in a plain subprocess, a different
  mechanism from the isolated worker process every real GGUF model load and
  the voice/STT engine use - so a machine could see every doctor check pass
  while every model load still failed (the exact LocaLM.exe launcher bug
  above). Doctor now spawns a worker the same way a real model load does and
  reports it as a failed check if that does not work.
- **Rebuilding the LocaLM.exe launcher after a Python upgrade (Windows) no
  longer fails when run from LocaLM.exe itself.** `localm make-launcher
  --force` - used to refresh the branded launcher after upgrading Python -
  failed outright ("could not locate the base interpreter to copy") when
  invoked from the already-built LocaLM.exe, since Python could no longer
  find itself under its own renamed identity. It now resolves correctly and
  replaces the running launcher's file in place.
- **A failed setup job (ComfyUI, a model pull, image generation) no longer
  hides its own error.** Two separate problems compounded: the job's live
  progress log in Settings disappeared the instant it failed - right as a
  toast told you to go read it - because the panel immediately re-rendered
  itself from scratch; and the job's real output (the actual git/pip/native
  error) was never written anywhere else either, so once that log vanished
  the reason was gone for good. The log now stays on screen after a failure,
  and it's also logged (so a bug report carries it too).
- **The managed-ComfyUI setup no longer fails cloning under a long
  `LOCALM_HOME` path.** A sufficiently nested data directory (a long
  username, a OneDrive-redirected profile, a custom install path) could push
  a cloned repo's internal file paths past Windows' legacy 260-character
  limit, failing with a cryptic "Filename too long" / "invalid index-pack
  output" git error.
- **The managed-ComfyUI setup no longer fails creating its own venv under the
  LocaLM.exe launcher (Windows).** After cloning ComfyUI, the "Creating a
  fresh localm venv" step could fail with the same misleading "[WinError 2]
  The system cannot find the file specified" as the model-loading bug above,
  for a related but distinct reason: Python's own venv module matches file
  names against the running interpreter's own name to decide what to copy
  into the new venv, and a renamed launcher's name never exists in the base
  install, so the new venv silently ended up with no interpreter of its own
  and the mandatory pip bootstrap then failed. Setup now creates the venv
  using the real base interpreter directly, so it succeeds under the branded
  launcher too.
- **`localm doctor` now also verifies venv creation will work.** Alongside
  the worker-spawn check above, doctor now also creates (and immediately
  discards) a throwaway venv the same way the managed-ComfyUI installer
  does, so a machine that cannot create nested venvs is flagged up front
  instead of failing silently mid-setup.
- **Selecting localm's own managed ComfyUI now actually uses it for
  generation.** Two separate bugs meant a correctly installed, correctly
  selected managed instance could still go unused: (1) nothing knew how to
  *start* it - image/music/video generation could only discover a launcher
  script for a user-provided ComfyUI install, so it failed with "ComfyUI is
  not reachable... point localm at your ComfyUI install" even though the
  managed instance was right there; (2) if you had ever set a ComfyUI folder
  for your own install - even long before setting up the managed one - that
  old value silently kept overriding the managed instance forever, for the
  same underlying reason. Generation now correctly launches and targets the
  managed instance in both cases; a genuine per-plugin ComfyUI override
  (Settings > Media's Image/Music/Video panels) still always wins, as before.
- **The managed-ComfyUI status no longer reports "installed" while it is
  still installing.** Whether a managed instance is considered ready (the
  Settings pill, `localm comfy status`, and the actual routing that decides
  where Generate sends a request) used to be based only on the ComfyUI
  checkout and its venv existing - which happens within the first few
  seconds of a fresh install, well before the much longer torch/requirements/
  custom-nodes steps that follow. For the entire rest of that install
  (several minutes on a normal connection), the status looked fully ready and
  a Generate click would try to use a ComfyUI with no PyTorch installed yet.
  Readiness now also requires the install's own completion marker, so the
  status - and Generate - correctly wait for the whole setup to actually
  finish.
- **Bug reports no longer lose the actual error.** The "Recent log (tail)"
  section used to be a blind cut of the last ~120 log lines, so a session that
  kept running afterward (even routine polling) could push the real error out
  before the report was filed. It now keeps every warning/error from the whole
  run and collapses long runs of near-identical routine lines (e.g. repeated
  status polling) into one line with a repeat count, so the actual failure is
  never buried or pushed out - no matter how long the session ran after it.
- **A failed automatic model preload is no longer silent in the log.** When
  `localm gui` warms up the last-used model in the background at startup and
  that load fails, the failure now reaches the debug log (with its full
  traceback) in addition to the console notice - previously it was
  console-only, so a report filed afterward showed no sign anything had gone
  wrong.
- A couple of status messages no longer claim success when the step was actually
  refused or skipped. A chat-model reload after image, music, or video generation
  now says the reload was deferred to your next message (instead of a false "Chat
  model ready.") when the server declined it, and installing the global `localm`
  command tells you to add its folder to your PATH by hand when it could not do so
  itself, instead of claiming the folder "was already on PATH".

### Security
- **Bug reports no longer leak your username in the fields you type or the issue
  title.** When you file a bug (through the app's "Report a bug", `localm bug-report`,
  or the standalone reporter used when localm will not start), the automatically
  collected diagnostics were already stripped of your home-folder path - which
  contains your account name - and of any pasted credential. The one-line summary, the
  description, and the extra "reason" you type were not, and neither was the public
  issue title built from them, so a home path or a key pasted into those fields could
  reach the public tracking issue even though the preview claims it shows exactly what
  will be sent. Those fields are now scrubbed at the point of upload too, so what is
  filed matches the redacted preview.
- **Disabling the private-network (SSRF) guard now requires an owner key.** The
  "Allow private/loopback targets" setting (`net_allow_private`) turns off the
  guard that blocks model-initiated requests to localhost, your LAN, and
  cloud-metadata addresses. It was changeable by any key with `config:write`;
  now, like the other trust-widening settings (the RAG indexing folders, a media
  backend's launch command), it is owner-only, so a scoped device key can no
  longer weaken this protection. No change for an owner running the app normally.
- **`localm key recover` and `localm key clear` now sign out browser sessions.**
  Rotating the owner key with `key recover` (the compromise-recovery path) or
  removing it with `key clear` now also invalidates every active browser (cookie)
  session, matching what the in-app "clear key" button already did. A browser
  session is deliberately decoupled from the key so a routine key roll does not
  log you out - but that meant a captured session cookie could keep owner access
  after a recovery meant to lock an attacker out. Your scoped device keys are
  untouched, so devices keep working; just sign in again in the browser with the
  new key.
- **Knowledge-base indexing refuses credential files named directly through the
  API:** the folder scan already skipped key and secret material (`.pem`, `.key`,
  `id_rsa`, `.env`, and the like) and model-weight binaries, but a file named
  explicitly in an API "add to collection" request slipped past that filter, so a
  scoped or remote client could point the indexer straight at a private key and
  read it back through search. Such files are now refused (HTTP 400) whenever
  indexing goes through the API, for every caller. Indexing your own key material
  on your own machine still works from the `localm rag add` command line.
- **The embedding model now shares VRAM management with your chat model.**
  Loading an embedding model (for memory/knowledge search) while a chat model was
  already resident used to just pile both into VRAM/RAM at once instead of
  swapping the chat model out first, the way image/music/video generation already
  does - on a tight card this could push system memory uncomfortably high. And
  "Unload all" on the Models page only ever freed the chat model: the embedding
  model stayed loaded and unaccounted for, so the button under-reported how much
  VRAM was actually released. Both are fixed: loading a large embedding model now
  swaps the chat model out first when needed, and "Unload all" (and the Models
  page's per-row Unload) now release the embedding model too and show it as
  loaded when it's the one resident.

## [0.1.2] - 2026-07-10

### Added
- **Automatic server-hang diagnostics:** an event-loop stall watchdog runs by
  default and, if the server ever freezes, dumps every thread's stack to
  `<home>/logs/hang_*.log` (the file is created only when a real stall happens, so
  a healthy run leaves nothing behind). A captured trace is bundled into a bug
  report automatically, so an intermittent "it just hung" becomes diagnosable with
  no setup on the reporter's part. It respects privacy: in privacy mode (the
  default) it writes nothing automatically, unless you turn on "Keep diagnostics
  for bug reports" (below). `LOCALM_HANG_WATCHDOG=0` turns it off entirely (and `=1`
  forces it on with verbose logging even in privacy mode); a loopback-only
  `GET /debug/stacks` returns thread and task state on demand.
- **"Keep diagnostics for bug reports" privacy setting:** privacy mode saves
  nothing automatically, which also means a hang or crash leaves nothing to
  report. This new toggle (off by default) keeps the diagnostic bits a report
  needs - the hang stack trace, the restart breadcrumb log, and a debug log -
  even in privacy mode. It is available in Settings > Privacy (in-app) and as a
  checkbox in the desktop launcher (and `--keep-diagnostics` on the command line).
  It keeps operational diagnostics only - see Security below for the chat-content
  guarantee this is held to.
- **Clearer bug-report send failures:** when a bug report cannot be filed, the app
  now tells you WHERE it failed - you appear offline, the server is unreachable, a
  secure-connection problem, or the server rejected it - instead of a raw error. It
  always keeps your report and offers to retry, or to download the report file so
  you can send it by email/Discord yourself (works from a phone or another device
  where a server-side path is useless). Both the WebUI and `localm bug-report`.

### Fixed
- **Launcher model selector lists only chat models:** the desktop launcher's model
  dropdown listed every registered model, including non-LLM ones (embedding /
  text-encoder / VAE / LoRA / diffusion components, or an unclassified `unknown`
  entry) that cannot be launched as a chat model. It now shows only LLM models,
  matching the rule the Models page already uses.
- **`localm doctor` no longer cries "CPU mode only" when your GPU is in use:**
  doctor decided GPU capability from `nvidia-smi` / `rocm-smi` / torch alone, none
  of which see localm's default GPU paths - Vulkan (Intel/NVIDIA/AMD via the
  display driver), Metal on Apple Silicon, or the bundled AMD-on-Windows ROCm
  build - so it wrongly reported "CPU mode only" on the majority of non-CUDA-toolkit
  GPU setups while inference was actually running on the GPU. It now reports the
  GPU from what localm will actually load: the real device the provisioned
  llama.cpp runtime registers (e.g. "GPU: ROCm0 - used for inference"), falling
  back to the provisioned backend name; the smi/torch lines remain as extra detail.
- **Server freezing while the system is idle:** reading GPU/VRAM state (opening
  Settings > Performance, the Models page, or the periodic cross-instance GPU
  heartbeat) ran a synchronous GPU driver query directly on the server's single
  event loop; if the driver was momentarily busy or wedged, the whole web UI froze
  even though the machine was idle. GPU probes are now time-bounded and run off the
  event loop, so a slow or stuck driver can no longer stall the server.
- **A cancelled reply no longer blocks the next request:** if a client
  disconnected in the middle of a response (closed the tab, pressed stop, or
  dropped the connection), the model kept generating the abandoned reply all the
  way to the end while still holding that model's single-inference lock, so the
  very next request to the same model stalled until the discarded generation ran
  itself out. A disconnect now stops the abandoned generation promptly and
  releases the model, so the next request starts right away - for both streaming
  and plain (non-streaming) requests.
- **Multi-GPU split fit checks:** a model too large for the single main GPU
  alone, but that fits combined across a configured multi-GPU split, was
  wrongly refused ("Not enough VRAM") when loading, and mis-badged "too big"
  in model search - the pre-load check, the search/CLI/MCP fit badges, the
  Settings performance-slider estimate, the scheduled-job/media-generation
  VRAM swap decisions, model-unload VRAM-release detection, and the GUI's
  hardware-monitor VRAM readout only ever weighed a load against one GPU's
  capacity, never the split's combined capacity. They now correctly sum
  capacity across the configured split.
- **Multi-GPU split vs. multiple loaded models:** a GGUF model load (chat
  model or the embedding model) could pass the pre-load VRAM check (enough
  COMBINED free VRAM across a configured split) and still fail or crash on
  a single GPU, if another already-loaded model left that specific device
  with less free room than its configured share of the new model needed -
  the aggregate check alone cannot catch an uneven split. Loading now also
  checks each configured GPU's own share before handing off to the native
  loader, evicting further (chat models) or refusing clearly instead of
  risking a native crash. That VRAM check also now runs off the server's
  event loop (like the idle-freeze fix above), so a slow GPU driver during
  a model load can no longer stall other requests.
- **GGUF context/KV-cache VRAM checks:** loading a GGUF model with a large
  context window ignored the KV cache entirely when judging whether it would
  fit, only weighing model weights - so a model whose weights fit could still
  ask the driver to reserve a KV cache far bigger than VRAM, which on some
  drivers either silently spilled into slow system memory or crashed the GPU
  driver with nothing shown to the user (it looked like the model "loaded"
  then went silent on the first prompt). The preflight now accounts for the
  KV cache at the requested context size and refuses clearly, before the
  native load, when it cannot fit; conversation growth (which can double the
  context on literally the first prompt, since the default reply budget
  already exceeds the default starting context) is now checked the same way,
  so a request that would overflow VRAM fails with a clear message instead of
  silently returning nothing.
- **Phone pairing with a VPN active:** a VPN client's virtual tunnel adapter can
  become the machine's default route and hand out an ordinary private address,
  which used to get picked as the "LAN" address for the Companion pairing card
  and the `<name>.local` mDNS advertisement - pointing a phone at an address
  only reachable through the VPN. VPN-like adapters are now excluded, so the
  real LAN address is shown (and advertised) instead.
- **Setup:** a silent fallback during AMD ROCm asset lookup (when the upstream
  release-asset check fails) now prints a warning instead of happening
  invisibly.
- **Log export reports real failures instead of "no logs found":** exporting logs
  to a folder that could not be written (full disk, read-only, or a permissions
  problem) used to report success with "No log files were found to export", hiding
  the write failure. It now fails with the actual reason when nothing could be
  copied, and a partial export lists the files it could not copy.
- **`localm setup-embeddings` no longer overstates what changed:** the completion
  message claimed memory and RAG would now use semantic search, but existing RAG
  collections stay lexical until re-indexed with embeddings. It now says memory
  retrieves semantically right away and names the re-index step
  (`localm rag add <name> <path> --embed`, queried with `--embed`) that RAG
  collections need.
- **`localm run` stops contradicting its own auto-start:** after it tried to start
  a background server that did not come up in time, it used to say "no server is
  serving this directory; start one with `localm serve`". It now acknowledges the
  timed-out start and loads the model in this process instead of advising exactly
  what it just attempted.

### Security
- **Privacy mode never logs chat content, even with diagnostics on.** Two
  code paths (the GGUF backend's raw model output, and the coder's web-tool
  call logging) could write your actual chat content - prompts and replies -
  to the debug log while in privacy mode, if `--debug` or the new "keep
  diagnostics" setting above was active. Both now stay off in privacy mode
  regardless of diagnostic settings; operational details (timings, request
  shapes) are unaffected.

## [0.1.1] - 2026-07-09

### Added
- **Unified model browser (phase 1):** registry model types, a ComfyUI model
  scan, and a type-filtered model list.
- **Search HuggingFace for HF (transformers) models, not just GGUF.** The Models
  page search has GGUF / HF format toggles; HF results show a total size and a
  VRAM fit badge estimated from the model's parameter count (or "size unknown"
  when the metadata is absent), and pull the whole repo, with a non-blocking hint
  when no transformers runtime is installed (the files still download). Both
  formats are interleaved so one never crowds the other out of the results.
- **Model management:** deterministic model type-detection with an explicit
  `unknown` sentinel and a `.safetensors` directory scan; a `--store` option when
  adding a model; a main-GPU selector with multi-instance GPU coordination and
  explicit model unload.
- **Split a model across multiple GPUs.** A model too large for any single
  card's VRAM can now load using the combined VRAM of 2 or more GPUs
  (`localm config gpu_split_indices 0,1`, or the "Split across GPUs"
  checkboxes next to the Main GPU selector in Settings), alongside the
  existing single-GPU selector. The model search results also hint when a
  model would fit split across your GPUs but not on the largest one alone.
- **Media: localm can manage its own ComfyUI (opt-in).** For image, music, and
  video generation, localm can install and run an isolated, hardware-matched
  ComfyUI kept separate from any existing one, via `localm comfy setup`
  (copy-path or a fresh install) and GUI setup/status/remove on the media pages;
  `localm doctor` hints at it. It coexists with a user's own ComfyUI rather than
  replacing it, and an in-memory shim works around an upstream ComfyUI crash.
- **RAG indexes more:** arbitrary text files by content sniffing, zip/tar archive
  extraction, and image description via a vision model. Each chunk is tagged with
  its document format (json, yaml, python, ...), heuristic-first, with the AI
  classifier only a tie-break for an unclear extension when a chat model is loaded.
- **MCP server exposes more tools:** setup, model removal, diagnostics, and plugin
  management, with annotations so clients can confirm destructive calls.
- **View the changelog in the app.** A "Show changelog" button in Settings >
  System renders the full release history in-app (backed by a read-only
  `/api/changelog`), so what changed is visible without leaving localm.
- **Signed, out-of-the-box self-update.** `localm update` works with no
  per-install setup: builds are assembled from a declared file manifest and
  signed, and the client verifies the signature before applying (see Security).
- **Recover from a bad update.** `localm update --rollback` restores the previous
  build from the last update's backup; and a standalone `rollback.bat` /
  `rollback.sh` in the install folder does the same WITHOUT needing localm to
  start, so an update that breaks the app can always be undone to a working one.
- **Report a problem even when localm will not start.** A standalone,
  account-less reporter (the `report-issue` entry) files a bug report through the
  localm proxy with no working install and no GitHub login, previewing exactly
  what will be sent first.
- **Setup dead-ends removed:** bootstrap `uv` automatically when it is missing,
  and prompt to replace already-provisioned native binaries instead of exiting
  (non-interactive runs keep the existing binaries).

### Changed
- **Media:** managed-ComfyUI status is checked only when it matters, not every 5s.
- **Media:** legacy in-package personal workflow overrides are migrated to the
  data directory on startup, so a self-update cannot wipe them.
- **RAG:** an embedding-only index no longer stalls or burns the request timeout,
  because format tagging is heuristic-first and the AI classifier runs only as a
  tie-break with a chat model loaded.
- **Inference:** repeated native-stderr lines are de-duplicated during generation,
  and native stderr redirection is tightened.
- **Dependencies:** huggingface-hub 1.22, transformers 5.x, fastapi 0.139,
  pillow 12.3, plus dev/tooling bumps; Dependabot now tracks the native `uv`
  ecosystem.

### Fixed
- **Models and API:** a local model file registers fully offline with no
  HuggingFace path leak; `/api/models` no longer 500s on a forward reference; a
  hidden native-load failure cause is surfaced.
- **HF snapshot pulls:** a full-repo HuggingFace pull now preflight-checks free
  disk space (like the GGUF/URL pull paths already did) instead of running
  until the OS hits ENOSPC mid-transfer; and "already downloaded" now compares
  every file the repo lists against what's actually on disk, not just
  `config.json`'s presence, so a disk-full mid-download no longer gets
  silently registered as a complete, ready model on retry.
- **RAG safety:** archive extraction is bounded (zip/tar bombs), compressed tars
  are handled, and a folder index skips model weights and secrets and does not
  index member-read errors.
- **RAG API under `localm serve` (api-mode):** indexing, upload, and embedding
  setup used to crash with an opaque HTTP 500 when hit directly (not through the
  GUI) because they read GUI-only server state; they now degrade to a clean 503
  ("run `localm gui`"), and querying falls back to lexical-only search instead of
  crashing. Attachment extraction (`/api/rag/extract`) no longer runs
  synchronously on the event loop, so a large or crafted archive attachment can
  no longer freeze every other request on the server while it extracts.
- **RAG on Windows:** a transient `PermissionError` during a collection write's
  atomic rename (antivirus or the search indexer briefly holding the file handle)
  is now retried instead of failing the write.
- **VRAM and multi-model handling:** safe multi-model VRAM eviction; idle-unload
  keeps the engine for a lazy reload; concurrent loads of different models no
  longer preempt each other; the active model is marked in `/v1/models` and
  attached to instead of the first entry; `/v1/embeddings` no longer force-loads
  the chat model; a stale VRAM estimate and a cross-instance conversation leak are
  fixed.
- **Do not hide problems:** removed production code paths that detected
  pytest/mocks and fabricated behavior; a swallowed VRAM-gate failure is now logged.
- **Inference:** a zero-n_tokens decode failure, batch memory-safety and
  token-position bugs, and a tool-call grammar that let small models loop; the
  lenient tool-call JSON parser no longer mangles backslashes (Windows paths).
- **Chat:** `localm run`'s default attach mode (the common case) no longer
  silently drops a thinking model's reasoning - it is now dimmed in the terminal
  like the in-process path already did.
- **Chat:** trimmed the injected web-access prompts so weak models stop fixating on them.
- **Coder:** `coder_confirm_timeout=0` now means wait forever, as documented.
- **Coder episodic memory:** a concurrent read racing the atomic write (a GUI poll
  landing mid session-close reflection) could hit a transient Windows
  `PermissionError`; both sides now retry briefly instead of raising.
- **Coder confirmations:** a coder session tracked only one pending confirmation
  at a time, so two tool calls needing approval in the same turn (e.g. two
  `fetch_url`/`web_search` calls under `net_mode=ask`, which the agent runs
  concurrently) would have the second clobber the first - the first could never
  be answered and just sat until the 10-minute timeout auto-rejected it. Pending
  confirmations are now tracked per call, so concurrent approvals no longer
  collide.
- **Coder reasoning:** the coder's HTTP backend (the common case - `localm serve`,
  OpenAI, Anthropic, or any OpenAI-compatible endpoint) never read a thinking
  model's `reasoning_content`, so its reasoning was silently dropped - no
  "thinking" display in the terminal or GUI, unlike the CLI attach-mode fix
  above. It now streams separately (a dimmed terminal aside, its own GUI event
  and collapsible block, and its own audit-log field) rather than being mixed
  into the visible answer, which the coder loop resends to the model and stores
  in history verbatim with no splitter of its own.
- **Memory:** the owner's chat memory stays in the shared `owner` namespace; a
  missing per-namespace write lock let concurrent writers (two requests, or a
  background consolidation pass racing a live edit) silently drop each other's
  facts, now fixed with a lock mirroring RAG's existing one.
- **GUI:** an empty ComfyUI scan shows the reason instead of a bare "Added 0"; the
  GUI no longer sends a chat request when no model is loaded; a background job's
  progress stream (model pull, ComfyUI setup, image/music/video generation) now
  fans out to every viewer independently, so reloading the page or opening the
  same job in two tabs no longer splits its events between them (one tab could
  end up hanging forever with no completion event).
- **CLI:** `localm.bat` argument forwarding is fixed.
- **CLI:** a `localm serve` start path that skipped the bind-security gates now
  enforces them like every other start path.
- **CLI:** an `UnboundLocalError` in the chat runner is fixed.
- **Updater:** the "runtime" update class's post-swap command used a bare `localm`
  argv that resolved back to the native launcher exe itself on the default install
  and rolled back the whole update; it now re-invokes through the current
  interpreter, like every other self-invocation site.
- **Bug reporter:** removed the only path that could ask for a GitHub login; the
  non-interactive path now names the account-less send channel.
- **Setup:** warn when a llama.cpp download has no checksum to verify (and verify
  against a published checksum by default); discard stray keyboard input before the
  CUDA prompt; skip console-window hiding in debug mode; skip draft releases with
  no uploaded asset.

### Security
- **Open-mode metadata reads are now origin-bound too.** The default CORS policy
  trusts every `localhost`/`127.0.0.1` origin to read a matching response, so
  another local program could steal the loopback GUI shell's open-mode
  management token from a plain cross-origin `GET /` and replay it against
  metadata routes (named keys, server config, host stats, the filesystem
  browser) to read real local data with no credentials of its own. That token
  is now gated on the same same-origin/allowlist check state-changing routes
  already enforced; state changes themselves were never affected.
- **Request-body size cap now covers chunked uploads.** The 160MB body cap
  only checked the client-supplied `Content-Length` header, so a
  `Transfer-Encoding: chunked` request (which sends no `Content-Length` at
  all) bypassed it entirely - reachable pre-auth on the CORS-exempt inference
  routes, and capable of buffering a multi-gigabyte body into memory from one
  connection. The cap is now enforced on the actual byte stream.
- **Plugin/tool calls (rag, web, voice, coder sessions, GUI model routes) can no
  longer starve chat completions.** Every blocking offload in the server -
  inference (model load/unload, chat/completion generation) and plugin tool
  calls alike - drew from the SAME process-wide thread pool
  (`min(32, cpu_count+4)` workers). A caller holding only a narrow plugin
  scope (or any loopback caller under open/no-key mode) could pipeline enough
  slow tool calls (e.g. archive extraction, which can legitimately run
  8-30s+ per file) to occupy every worker thread in that pool, which starved
  the SAME pool's inference slot and stalled chat replies for every user of
  the server, including the admin. Plugin/tool work now runs on its own
  dedicated, equally-sized pool (`localm/plugins/executor.py`), completely
  isolated from the pool inference relies on.
- **Privacy mode now deletes ComfyUI's own on-disk output copy everywhere media
  is generated, not just from the GUI/API.** ComfyUI keeps its own copy of every
  generated image/track/clip with the full prompt and workflow embedded as
  metadata; privacy mode is supposed to remove it after use. That fold-in was
  only wired up on the GUI/API path - the `localm image`/`music`/`video` CLI
  commands, the `/generate-image`/`/generate-music`/`/generate-video` REPL
  commands, the MCP server's `generate_image` tool, and the coder agent's
  `generate_image` tool all suppressed the prompt sidecar in privacy mode but
  left ComfyUI's own copy (and any img2img source image) on disk indefinitely.
  All six call sites now fold privacy mode into `delete_outputs`, matching the
  GUI/API behaviour.
- **`?pull=` deep link no longer downloads a model with zero confirmation.**
  `localm gui --pull SPEC` opens the browser at `?pull=SPEC`, but any page (or a
  hidden iframe on any site, while localm runs locally) could forge the same
  link and silently start a real download. The CLI now mints a single-use,
  spec-bound token passed alongside the link, so ITS OWN deep link still
  auto-starts with zero clicks; a link without a valid token falls back to an
  explicit confirmation dialog instead of firing automatically.
- **Media gallery and share-inbox ownership fixed:** the image/music/video
  generated-media routes (serve/delete/move/rename/history) and the PWA
  share-inbox routes had no per-key ownership check, so any key holding the
  plugin's own (non-privileged) scope could enumerate, read, delete, move, or
  rename another principal's generated media or shared files. Both now stamp
  and check ownership the same way the jobs API already did.
- **Job API privilege escalation fixed:** correct principal-ID hashing for
  admin/owner keys, so owner-created jobs are no longer reachable by
  loopback-anonymous roles.
- **Coder `spawn_agent` no longer bypasses confirmation.** A child agent spawned
  via `spawn_agent` now inherits the parent session's `auto_approve`, `dry_run`,
  `always_confirm`, and `confirm_handler` instead of always auto-approving, so a
  parent that requires confirmation (or is running `--dry-run`) can no longer be
  routed around by having the model delegate destructive work to a sub-agent.
- **A scheduled coder job's shell access no longer outlives its creating key.**
  The autonomous job scheduler now re-validates the owning key is still live
  (not revoked, not expired) before running an `allow_shell` job, instead of
  trusting the stored opt-in forever; a dead key downgrades the run to the
  safe restricted coder rather than keeping shell access.
- **Coder privacy-mode history scrub** now warns when a shell history file
  cannot be *read* for scrubbing, matching the existing warning when it cannot
  be *written* - an unreadable file is no longer silently treated as clean.
- **Authenticated self-update.** Each release build is signed with an offline
  Ed25519 key and verified against a pinned public key before it is extracted or
  executed, so a compromised release channel cannot push a forged build.
  Anti-rollback refuses an older signed build, the download stays HTTPS-pinned, and
  a self-update no longer wipes the provisioned native runtime. Publishing is gated
  on a clean tree, a full CI pass over the repo, and a build that imports and runs;
  the build itself is now assembled from that exact CI-validated commit (not a
  live disk snapshot taken after CI finishes), so a change landing on disk during
  the wait can no longer ship unreviewed.
- **ComfyUI launch** no longer has a shell-injection vector on Windows.
- **RAG folder index** skips model weights and secrets rather than indexing them.
- **MCP** destructive tools are annotated so clients can confirm them.

## [0.1.0] - 2026-07-04

First tagged release. A self-contained, offline local-LLM platform.

### Added
- **Inference**: run GGUF models via a bundled llama.cpp runtime, and HuggingFace
  models via a torch backend. GPU acceleration on AMD (ROCm), NVIDIA (CUDA or
  Vulkan), and Intel/Vulkan, with automatic hardware detection and a CPU fallback.
  Auto context-window sizing from free VRAM.
- **CLI**: `pull`/`search`/`add`/`list`/`rm` for models, `run` (interactive or
  single-prompt, with image input), `serve` (OpenAI-compatible API), `gui`,
  `doctor`, `benchmark`, `config`, `info`, and shell completion.
- **GUI**: an installable PWA served by the local server - chat with conversation
  history, assistant memory, and personas; the coder agent; model management; and
  per-plugin pages. Loopback by default; an optional network bind for phone access.
- **Coder agent**: an offline AI coding agent with file, shell, search, and test
  tools, a project map, an agentic loop, and MCP interop, gated behind an explicit
  scope.
- **Plugins** (everything but chat is a plugin): image / music / video generation
  (ComfyUI backends), a knowledge base for retrieval-augmented chat (RAG), a job
  scheduler, an MCP server, text-to-speech, speech-to-text, and web search/fetch.
- **Security**: loopback-only by default; a network bind forces a strong API key
  and enables built-in TLS. Scoped API keys, an owner/admin model, HttpOnly session
  cookies with CSRF protection, path confinement, and an SSRF-guarded outbound
  fetch policy.
- **Privacy**: an opt-in privacy mode that keeps sessions in memory and writes no
  traces to disk; a configurable network policy (off / ask / allow, allow/deny
  lists).
- **Networking**: reach a network-bound instance by name over mDNS (`localm.local`)
  and Tailscale MagicDNS, folded into the TLS certificate.
- **Model tools**: HuggingFace search and local model registration.

### Known limitations
- Media generation (image / music / video) requires a local ComfyUI backend; RAG's
  semantic mode requires the on-device embedding model (`localm setup-embeddings`).
- The in-app bug-report upload and the self-updater require the maintainer's
  Cloudflare Worker to be deployed; until then those two features are inert.
- The NVIDIA GPU path is validated by design and CI-adjacent testing; the primary
  development hardware is AMD.

[Unreleased]: https://github.com/Matlan1/localm/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/Matlan1/localm/releases/tag/v0.1.2
[0.1.1]: https://github.com/Matlan1/localm/releases/tag/v0.1.1
[0.1.0]: https://github.com/Matlan1/localm/releases/tag/v0.1.0
