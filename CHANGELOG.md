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
- **`/review` in the coder REPL.** Asks the reviewer model for a second opinion
  on the current diff right now, instead of waiting for the automatic pre-done
  pass. Works even when `coder_review` (the automatic pass's own on/off switch,
  off by default) is off, and reports the same verdict that pass reads:
  approved, or the list of blocking issues plus the reviewer's notes.
- **Other AI tools can now read localm's memory of you, so you stop repeating
  yourself.** With the memory plugin on, `localm mcp` offers a `memory_recall`
  tool: any MCP client (Claude Desktop, an editor's agent) can look up the
  facts localm has remembered about you rather than starting from nothing.
  Reading changes nothing - it will not alter what localm remembers or how
  long it keeps it. Writing is a separate opt-in: pass `localm mcp
  --memory-write` to also offer `memory_append`, which lets a client suggest a
  fact. Anything it suggests is stored as unverified, never as something you
  said yourself, and if it contradicts a fact you typed, it goes to the same
  accept/reject review as any other suggestion instead of overwriting you.
  Both tools are refused in privacy mode.

### Fixed
- **A stuck GPU-detection helper could wedge VRAM measurement for the rest of
  the run.** When it happened, loading any model afterward refused with "free
  VRAM could not be measured," even though the graphics driver itself was
  fine, and only a server restart cleared it. The helper's own cleanup after
  giving up on a slow check could itself block forever waiting on a hung
  child process; that wait is now bounded, so a stuck helper is abandoned
  instead of wedging every check that comes after it.
- **A separate cause of the same "free VRAM could not be measured" refusal:**
  a GPU check running at the same time as a model load could read the driver
  library as ready before it had actually finished starting up, then wait on
  it anyway. On a slow first start this could take longer than the check's
  own budget, timing out even though nothing was actually stuck. That check
  now gives up on its own short budget and falls back to measuring VRAM a
  different way instead of waiting it out.
- **Loading a model that hit an inconclusive VRAM reading now retries a few
  times on its own before giving up**, instead of immediately refusing and
  asking you to try again yourself. If it still cannot get a clear reading
  after those automatic attempts, it says so plainly rather than suggesting
  something to go do about it.
- **Clicking an already-open coder session now switches to it instead of
  opening it again.** Clicking a session that was already running started a
  second, independent copy of it every time, each holding its own connection
  open indefinitely. Enough of these open at once could make the rest of the
  app stop responding - most visibly, the Plugins page's list would stop
  loading. Reopening a session you already have open now does nothing more
  than bring it to the front, the same as every other session in the list.
- **Restarting the server from Settings could leave the page stuck on a
  reconnecting screen instead of coming back.** The page could reload before
  the old server had actually finished shutting down, landing back on the
  server that was already on its way out rather than the freshly restarted
  one. It now waits for confirmation that a genuinely new server process has
  come up before reloading.
- **A blocked or failing Kokoro voice download could show a raw, unhelpful
  error instead of the "allow huggingface.co" guidance.** This happened
  whenever the browser had already cached part of the voice model from an
  earlier attempt, which made the failure look like an unrelated problem
  instead of a network one. A genuine network failure is now reported as
  blocked regardless of what is already cached.
- **If the server process ever died unexpectedly, such as a crash, it stayed
  down until manually restarted.** A watchdog now relaunches it automatically
  on the same port when this happens, so a crash recovers on its own instead
  of requiring you to notice and restart it yourself. A clean, intentional
  shutdown is unaffected.
- **An installed app whose certificate stopped being trusted, such as after
  switching devices or a new certificate being generated, still said
  "Running as an installed app" with no hint anything was wrong.** It now
  tells you the certificate needs to be trusted again and where to get it.
- **Right-clicking the tray icon could, in rare cases, crash the server
  outright.** The tray's background thread now sets up what Windows needs
  before it can show its menu, closing off the specific way this could
  happen. Combined with the automatic-restart fix above, this is also less
  likely to matter if it ever does.
- **The launcher's "Import from folder" gave no sign anything was happening
  and could sit for a while before failing, especially when pointed at the
  models folder itself (such as after copying it in from another install).**
  That case now syncs the folder instead of refusing it as "not a model."
  Every import shows a live, updating status now instead of a static
  message, and the model list refreshes automatically in the background
  when the launcher opens or when the new "rescan" button is used, rather
  than freezing the window while it scans.
- **In the coder REPL, running several `/goal` tasks in one session could let
  a task that rewrites the test it is judged against go unflagged until you
  quit the REPL.** The warning about edited test or CI-config files now shows
  right after each `/goal` task finishes, not only when the session ends.

### Security
- **The coder now refuses a small set of catastrophic shell commands outright,
  instead of relying on you being there to say no.** Until now the only thing
  standing between the model and a command like a recursive delete of your home
  directory was the confirmation prompt, and an unattended run skips that
  prompt entirely: with auto-approve on, "ls" and a command that wipes a disk
  took exactly the same path. A fixed safety check now runs on every command
  the model writes, before anything else, and cannot be approved past. It
  refuses a recursive delete aimed at a drive root, a home directory or a
  system directory; commands that format or overwrite a disk; writes that would
  overwrite or delete your SSH, GnuPG, AWS, Docker or Kubernetes credentials;
  downloaded scripts piped straight into a shell; a force push at master or
  main; and "git reset --hard", which throws away uncommitted work. Each
  refusal says which rule stopped it and what to do instead. Ordinary commands
  are unaffected, including ones that look similar such as deleting a build
  directory or force-pushing your own feature branch.

## [0.1.5] - 2026-08-26

### Added
- **`/goal <command>` in the coder REPL.** Iterates the current task until the
  command exits 0, feeding each failure back for another attempt (up to 5 by
  default), instead of stopping after a single reply. `/goal auto` re-detects
  the project's own check; `/goal off` turns it off again. The same iterating
  loop the non-interactive `--until` flag already used.
- **A coder session started in the browser can now choose which model server
  answers it.** The setup form offers this localm, any OpenAI-compatible URL
  (Ollama, LM Studio, vLLM), OpenAI, or Anthropic, per session, matching the
  choice the terminal has always had. The default is unchanged and offline.
  Because any other choice sends your prompts and the file contents the agent
  reads off your machine, it states that when you pick it and keeps a marker on
  the session for as long as it runs; off-machine models are refused in privacy
  mode, with a message naming the setting that enables them; choosing a model
  server needs the owner key; and an API key entered for a session is kept only
  for that session and never written to disk. A URL on your own machine counts
  as local and works in privacy mode.
- **Native HuggingFace AWQ model loading and inference.** HuggingFace AWQ
  (Activation-aware Weight Quantization) 4-bit checkpoints can now be loaded
  and executed natively across Windows ROCm, Linux ROCm, NVIDIA CUDA, Intel XPU,
  and CPU without external compiled binary dependencies (bypassing gptqmodel,
  autoawq, and torchao incompatibilities on Windows ROCm).
- **Multi-Token Prediction (MTP) model support.** Models trained with MTP or
  next-n prediction heads (such as Qwen MTP variants) can now speculatively
  generate draft tokens via a dedicated MTP draft context and verify them in
  batches on the main model graph, speeding up generation without requiring a
  separate draft model (structured or tool-calling replies generate one token
  at a time instead). It engages only where the runtime can genuinely build an
  MTP draft head: a model that merely carries next-n metadata, one whose cache
  cannot roll back a rejected draft, and a conversation that outgrows the draft
  context all fall back to ordinary autoregressive decoding, with the reason
  recorded in the debug log.

  The draft head is fed the hidden state it predicts from, so it now accepts
  about half its drafts instead of about one in ten, and it works on models
  whose cache keeps recurrent state (the Qwen3.5 and 3.6 MTP family, Nemotron
  and DeepSeek V4 among them). Those declined outright before: speculation has
  to take a rejected draft back out of the cache, and such a cache cannot be
  rewound unless it was asked to keep per-token snapshots, which nothing did.

  `mtp_enabled` **stays off by default.** Good drafts are necessary for
  speculation to pay and not sufficient: a rejected draft still costs a
  two-token verification, and on a small model that costs meaningfully more than
  verifying one token, so the arithmetic comes out slightly negative. It turns
  positive on a model large enough that the two cost about the same. Turn it on
  and keep it if your model gets faster.
- **A collection's individual documents can now be listed and removed from the
  terminal.** `localm rag docs NAME` shows each indexed document with its chunk
  count and whether its source file has since gone missing or was added via an
  upload; `localm rag rm-doc NAME PATH` drops just that one document from the
  index, leaving the original file and the rest of the collection untouched.
  Until now the only CLI removal was `rag rm`, which deletes the whole
  collection - so fixing one bad document meant losing every other one too.
- **Max resident models and pinned models are now settable from the GUI.** Two
  new controls sit in Settings > Model > Live tuning, beside Main GPU and Split
  across GPUs: "Max resident models" caps how many models may stay loaded at
  once (blank leaves it to free-VRAM arithmetic), and "Pinned models" takes a
  comma-separated list of names that are never evicted to make room for
  another. Both were previously reachable only through `localm config` or a
  raw API call.
- **Diagnostics moved into the app: Settings > System > Diagnostics.** Until now
  the only way to run localm's active self-checks was `localm doctor` in a
  terminal, so anyone using the app alone could not run them at all. The card
  runs the five checks that actually try something rather than reading a version
  number - the llama.cpp library is present and not truncated or missing its GPU
  kernel data, the runtime's struct layout matches this build, the worker process
  every model load depends on can really be spawned, a nested venv can really be
  created, and the transformers backend's classes really load - and shows a
  verdict per check with the same wording the terminal gives. A run takes about
  half a minute, names the check it is on while it works, and can be picked up
  from another tab or after a reload. Nothing is installed or changed, and the
  summary says it covers these checks rather than claiming your whole system is
  healthy.
- **Past coding sessions are now listed and can be picked up again.** The coder's
  session list shows what is open right now, past sessions in the folder you have
  selected, and collapsible groups for every other project you have worked in, so
  reaching earlier work no longer means retyping its path and hoping the right
  conversation comes back. Clicking a past session continues that particular one,
  in its own project. A folder that has been moved or deleted still lists its
  sessions and says the folder is missing, since the conversations do not live
  inside the project. Privacy-mode sessions never appear, because that mode writes
  nothing to disk, and the list says so permanently so a short list is never
  mistaken for a complete history. Remembering which projects you have worked in
  can be turned off or cleared in Settings > Coder, along with how many to keep.
- **The coder's session list can sit on either side.** It stays on the right by
  default; Settings > Coder, or the button on the list itself, moves it left.
- **What localm remembers about you can now be managed from the terminal.**
  `localm memory` lists the facts it has learned, shows one in full, saves a fact
  you type yourself, deletes one, reviews the corrections that background
  consolidation proposes, and erases everything. Until now all of that was only
  reachable in the app, which was awkward the other way round too: a scheduled
  memory job could quietly produce correction proposals that a terminal user had
  no way to read or resolve. Erasing now takes the recoverable copies with it, so
  "erased" means the text is off disk rather than moved to a file you cannot see.
- **Rolling the owner key no longer needs a terminal.** Settings > Security gains
  an **Owner key** card: **Generate new key** mints a fresh one, or paste a key you
  already use and press **Set this key**. The new key is shown once with a copy
  button, and either action asks first, since both cut off every other device
  holding the old key. This browser stays signed in, so you can rotate a key that
  may have leaked without locking yourself out, and remotely rather than only from
  the machine localm runs on. Scoped device keys are untouched. If
  `LOCALM_API_KEY` is set in the server's environment it still overrides the
  stored key, and localm now says so instead of reporting a rotation that has not
  taken effect.
- **A plugin's own settings are now editable from the terminal.**
  `localm plugin config <name>` lists one plugin's settings with the values
  actually in effect, and `localm plugin config <name> <key> <value>` sets one
  (a blank value clears it again). These are the per-plugin blocks the web UI
  edits under Settings; `localm config` could never reach them, because it
  writes top-level keys while these live under a plugin of their own.
  `image`, `music` and `video` can each hold their own value for a field now
  instead of sharing one global default, so two of them can point at different
  ComfyUI installs or run at different precisions, and each can pick its own
  workflow. The text-to-speech block (voice, speed, model, device, precision) is
  settable too. Those four work with no server running; any other plugin
  declares its settings as it loads, so listing or changing those uses a running
  localm, and says so plainly when there is none instead of showing an empty
  list.
- **The terminal can now start, stop and restart ComfyUI, and cancel something
  the server is busy with.** `localm comfy status` used to answer only from
  disk; it now also says whether the ComfyUI localm targets is actually
  running and whether localm launched it (`--no-ping` keeps the old, instant
  answer). `localm comfy start` brings it up without running a generation,
  `localm comfy stop` aborts the render, clears the queue and frees its VRAM,
  and `localm comfy restart` does both. A ComfyUI you started yourself is
  never killed, only aborted, and localm says so rather than implying
  otherwise. Separately, `localm status` now shows an id for each thing the
  server is working on and `localm cancel <id>` stops one of them - so a
  two-hour re-embed or a large download started from a phone can be called off
  from the terminal that could previously only watch it. Pressing Ctrl-C during
  `localm image`, `localm music` or `localm video` now tells ComfyUI to abort
  that render and release its VRAM, instead of leaving it going after localm
  has exited.
- **Installing, switching or pinning the llama.cpp runtime no longer needs a
  terminal.** Settings > Updates could only ever re-provision the backend you
  already had, and did nothing at all on a machine with no runtime yet - so
  choosing a backend, trying a specific llama.cpp build, or setting one up in
  the first place meant running `localm setup-llama` at a command line, which
  `localm doctor` and the Settings page itself both told you to do. The block,
  now called **Inference runtime**, adds a backend picker covering every
  option the command accepts (auto-detect, vulkan, cuda, sycl, hip, cpu, metal
  and the self-contained AMD ROCm build) and a **build** field that takes a
  llama.cpp release tag to install and pin, or `default` for the build localm
  ships and confirmed, or `latest` for upstream's newest. The button says which
  of install, reinstall, switch or update it is about to do, and asks first only
  when a switch would replace a runtime that currently works. A build that
  cannot load on your machine is still never kept, and a machine whose runtime
  is missing or broken can now be repaired from the same screen instead of
  being locked out of the one action that would fix it.
- **Binding the server to your network no longer needs a terminal.** Settings >
  Server gains a **Bind address** field (plus TLS controls: turn encryption
  off for a trusted network, or point at your own certificate pair), so the
  phone/Companion feature can be enabled entirely from the GUI: set an API
  key, set the bind address to `0.0.0.0`, click Restart server. The safety
  rules are unchanged and enforced at startup: without a strong API key the
  server refuses the configured network bind and stays on this computer only -
  the Companion card then says exactly why - and the unauthenticated
  `--insecure` override deliberately has no settings form, so it still
  requires a terminal. An explicit `-H` on the command line always wins over
  the setting, and a custom certificate pair that fails to load falls back to
  localm's built-in certificate (with a warning) instead of serving
  unencrypted or failing to start.
- **Managing what the coder remembers no longer needs a terminal, and two more
  session settings arrived.** The **lessons** panel under Coder now has a
  **stored** and a **dropped** tab, so you can see both what the agent recalls
  from past sessions on a project and what it has let go of. Any lesson can be
  forgotten, any dropped one brought back, and **consolidate** asks the model to
  merge related lessons into one - opt-in, manual, and every original kept so a
  bad merge is reversible. There is also an erase-everything button; that one
  takes the recoverable copies with it and cannot be undone, so it asks first.
  Alongside it the session form gains **seed**, which pins sampling so the same
  seed and prompt reproduce the same output, and **still confirm shell
  commands**, which lets file writes run unattended under auto-approve while
  shell commands still stop for you.
- **A plugin can now contribute a working settings section.** `add_settings()`
  was already documented as part of the plugin API, but calling it did
  nothing - the fields simply never appeared anywhere. Any plugin, built-in or
  third-party, that adds settings this way now gets a real section on the
  Settings > Plugins page, rendered with the same text, number, toggle,
  dropdown and masked-secret controls used everywhere else in Settings.
  Saving only writes the fields you actually changed, and a field marked
  admin-only (a webhook secret, a script path) stays hidden from and
  unwritable by a non-owner key, matching every other privileged setting.
- **The Image, Music and Video model pickers now say what each dropdown is
  for, and know what you already have.** Every model row in the Workflow panel
  was labelled with the raw name the workflow file happens to use (`unet_name`,
  `clip_name1`), which told you nothing about which file belongs there. Each row
  now carries a plain-English name for the part it fills, such as "Diffusion
  model (UNet)" or "Text encoder 1 (CLIP-L)", with the raw name kept beside it.
  If a model your ComfyUI does not have is registered in localm anyway, that is
  now pointed out on the row that needs it, along with what to do about it. And
  when ComfyUI is not running, the panel no longer stops at "not running": it
  lists the parts this workflow needs and which of your registered models could
  fill each one, which needs no ComfyUI at all.
- **A one-time "download it now" for the two small helper models, without
  touching your network settings.** Semantic search needs a small embedding
  model and the mic button needs a Whisper speech model, each fetched exactly
  once. With the network policy set to "ask", those one-time fetches used to
  leave only a bad choice: flip net_mode for the whole app, or go without the
  feature. Now the blocked state says exactly why (in the Knowledge page's
  embedding panel and the mic button's tooltip) and offers a one-click download
  of just that model. The permission that guards it is the same one that could
  change the network policy itself, the download happens once, and nothing is
  written to settings: net_mode stays where you put it for everything else,
  and net_mode=off remains absolute, with no bypass. The voice plugin also
  fetches its speech model right when it is installed (under the same rule),
  so the mic works on first click instead of stalling on a surprise download.
- **Six coder options that only worked from the terminal now work in the app
  too.** Under **Coder**: **estimate** (beside the composer) plans the task you
  have typed without running it, so you can see the approach and what it will
  cost before committing to it. **Patch mode** (a session toggle) collects every
  file change as a single diff and writes nothing to disk, downloadable with the
  new **patch** button. **Export** now offers the last task's result as JSON as
  well as the readable transcript. **Lessons** lists what the coder has learned
  from past sessions on a project. The session form gains a **verification
  command** - a command whose exit code, not the model's own say-so, decides
  whether a task is done - along with how many fix attempts it gets and a way to
  turn it off. And a **native tools API** toggle for model servers that support
  that protocol, which tells you plainly when the server you are on does not.
- **Images a model links in a reply can now be displayed, without the site that
  hosts them learning anything about you.** Until now a linked image showed as
  broken: localm refuses to let the page load anything from the internet. Other
  local-AI apps just let your browser fetch it, which tells that site your IP
  address, your browser and which page you were on. There is now a setting,
  **Settings > Server & network > Outbound access > "Show remote images in
  replies"**, which is **off by default**. Turned on, localm fetches the picture itself and hands it to the
  page, so your browser never contacts the remote site at all, and the request
  obeys the same protections as every other one localm makes (no local or
  private addresses, size limit, and your allowed/denied domain lists). Worth
  knowing before you turn it on, because it is the reason it ships off: a linked
  image is also how a model could smuggle information out, since the web address
  itself can carry it and it is fetched the moment the reply appears. Turning
  this on does not stop that, it only makes the request come from this machine
  instead of your browser.
- **Answers built on your indexed documents and remembered facts are now
  checked against the evidence they were given.** localm gains a regression
  check that indexes an invented fact, asks about it, then flips that fact and
  asks again: a correct answer has to change with it. It covers both the RAG
  retrieval path and the memory recall the assistant receives each turn, and it
  runs against a real model, so a future change to how retrieved text is
  presented cannot quietly leave answers relying on what the model already knew
  instead of on what you gave it.
- **Music and Video now have the same library as Images.** Generated tracks and
  clips were a plain list of filenames with play, move and delete. They are now
  a grid of cards you can tick to select, with bulk move and bulk delete, and a
  detail view holding the full generation metadata plus download, copy path,
  rename, move and delete. "Reuse settings" refills the form from any track or
  clip you already made, opening the Advanced section when it restores anything
  hidden there. A clip's card shows a real frame from just past the start; a
  track's card leads with its style tags and length, since audio has no frame to
  show, and plays in place from the card in one click. Long libraries show the
  newest 24 with a "show all" toggle.
- **The server now watches itself for hangs and recovers on its own.** A
  frozen server (nothing responding, no error, no crash) previously just sat
  there until someone noticed the GUI had gone dead. A built-in watchdog now
  detects a frozen server, turns the status window red with what is wrong,
  and automatically restarts the server on the same port within about a
  minute - no user action needed. Set `LOCALM_HANG_RECOVERY=surface` to keep
  the warning but disable the automatic restart, or `=off` to disable the
  watchdog entirely. When individual requests get stuck without the whole
  server being dead, nothing is shown (a warning that can be wrong is worse
  than none), but the debug log records exactly which requests were stuck,
  where they were blocked, and what to include in a bug report.
- **Registering models found in a ComfyUI folder now shows real progress.**
  The guided "Import from ComfyUI..." wizard and the "Re-scan ComfyUI folder"
  button used to sit there with no feedback (or just a static "Scanning..."
  message) while everything they found got registered. Both now show a live
  "registering model N of M" count, with the file name, as it works.
- **`localm gui` can now open as its own app window instead of a browser
  tab.** Setup now asks up front (default stays browser tab, no extra
  install); a Settings toggle ("Default window mode", under Desktop app)
  lets you change your mind later without re-running setup. By default the
  window's close button hides it to the tray and the server keeps running,
  same as closing a browser tab - another toggle ("Quit when the app window
  is closed") makes it quit the whole app instead. On Linux this installs
  entirely via pip - no system packages, no `sudo` needed.
- **The file/folder picker used throughout the GUI can now create a new folder
  and rename a file or folder.** A "New folder" button creates a folder in the
  location you are browsing and opens it; each listed file and folder has a
  rename control. Renaming never overwrites an existing file or folder with the
  same name.
- **Split across GPUs now has a relative-weight control.** Next to each checked
  device in Settings > Model > Live tuning, an optional weight input lets you
  pin how a model is split (e.g. 3 and 1 gives the first card three times the
  second's share) instead of only the automatic, free-VRAM-proportional split.
  Leave every weight blank to keep the automatic sizing. This was previously
  only settable by hand-editing `config.json`.
- **A scoped key can now be handed access to specific document folders only,
  instead of your whole default RAG reach.** `localm key create` gains a
  repeatable `--rag-root` option: set it and that key's whole Knowledge
  reach - indexing, querying, listing, and managing collections - is limited
  to collections built only from the folders you listed, not your home
  directory, working directory or configured allowed folders. Leave it
  unset and a key behaves exactly as before. The owner key is never
  affected by this.
- **The Models page gained an "Other" tab, a way to keep it out of your main
  list, and a group-by-type view.** A vision projector, or a model localm could
  not classify, used to appear only in the "All" list mixed in with the models
  you actually chat with, because no tab covered it. Those now have their own
  **Other** tab, and the All list leaves them out by default while telling you
  how many it is holding back and where they are; a **show other types here**
  tick puts them back in. A **group by type** tick breaks the list into one
  section per kind (LLMs, Embedding, VAEs, ...), sorted inside each section by
  whichever column you are sorting on. Both are remembered in this browser only.
  A model registered long enough ago that localm never recorded a type for it
  also lives on **Other** now, and its Role reads **not set** instead of
  claiming it is an LLM, which is what the list used to show. Picking a type
  there files it away properly. It stays selectable for chat throughout, exactly
  as before.
- **A mapped network drive (e.g. `Z:\`) is now correctly recognized as a
  network location, and a new setting lets you refuse them.** localm's checks
  for a remote path only caught `\\host\share`-style UNC paths; a Windows
  drive letter mapped to a network share looked like an ordinary local drive
  and was never distinguished from one. It still worked exactly the same as
  before, but a new toggle, **Settings > Security > "Allow network drives as
  filesystem locations"** (on by default, matching the existing behavior),
  now lets you refuse them across the folder picker, folder creation,
  renaming, log export, and document indexing, if you would rather keep
  localm confined to local disks.
- **Rolling back a bad update no longer needs a terminal.** Settings > Updates
  gains a **Roll back** button that restores the build you were running before
  the last update and restarts, so the restored build actually loads. It covers
  the case nothing else did: an update that installed cleanly, runs, and turns
  out worse (a build too broken to start is what `rollback.bat` / `rollback.sh`
  are for, and an update that never comes back is already rolled back for you).
  The button appears only when there is a backup to restore and names the build
  it would put back before you press it. It asks for the owner key rather than
  any key that may change settings, because restoring an older build can bring
  back something the newer one fixed. If an update is being applied at that
  moment, the roll back is refused with a clear message instead of running
  alongside it. Like `localm update --rollback`, it restores that build's files;
  it does not undo package installs an update made along the way.

- **A running coder session can now be steered from the app, not just at the
  start.** The session toolbar gains **controls**, **memory** and **bg**.
  **Controls** turns auto-approve on or off while the agent is working - so a run
  you no longer trust stops asking for nothing and starts asking you, including
  work it has handed to a sub-agent - and sets
  the scope glob, the verification command, and the project directory the session
  works in. Moving directory brings the conversation and its saved copy along
  rather than stranding it under the old project. **Memory** shows the project's
  LOCALCODER.md, the notes the agent reads every turn, and lets you add or drop
  one; the running session picks the change up immediately instead of on the next
  one. **bg** lists the background jobs this session started, which previously
  ran with nowhere to see them. The setup form can also now pick which saved
  session to continue when a project has several, rather than only the newest.
- **Pulling a model from the app can now verify its checksum.** The "Add a
  model" field gains an optional **sha256** box, matching `localm pull
  --sha256` on the terminal: enter the expected digest and a download, or a
  local file you point at, is checked against it before anything is kept. A
  downloaded file that does not match is deleted and never registered; a local
  file that does not match is simply refused. The box has no effect on a full
  HuggingFace repository, which has no single file to check.
- **A model whose file was moved can now be re-pointed from the Models page.**
  When a registered model lives outside the managed models folder and its
  file goes away (moved to a new drive or folder), the row now reads
  **missing**, the same flag `localm list` already showed in the terminal but
  that the app never surfaced at all. A **relocate** button on that row asks
  for the new location - a moved `.gguf` file, or a HuggingFace model folder -
  pre-filled with where it used to be. Until now the only way to recover was
  removing the entry and adding it back, which mints a new registration and
  drops its aliases, its recorded source and its stored sha256; relocating
  keeps all three, the same registry entry `localm relocate` re-points from
  the terminal.
- **`localm key create` can now set an expiry and mint privileged scopes, and
  `key list` now shows a key's age, expiry and last use.** A CLI-minted key was
  permanent with no way to set a deadline, and there was no way to see one
  either, even though the server already tracked it. `--expires-in <seconds>`
  sets a deadline now; `key list` gains Age/Expires/Used columns, and a key
  whose deadline has passed is marked rather than looking identical to a live
  one. `--allow-privileged` lets `key create` mint a key holding `admin`,
  `keys:admin`, `plugins:admin`, `config:write` or `coder:full` - refused by
  default, since a routine mint should not escalate itself by accident, but
  this machine's terminal is already fully trusted (it can already roll or
  recover the owner key with no credential presented), so the option exists to
  grant one deliberately.
- **A key minted from the GUI or the raw API can now be confined to specific
  RAG folders.** `localm key create --rag-root` already did this from the
  terminal, but `POST /v1/keys` silently dropped the field, so every GUI- or
  API-minted key stayed RAG-unconfined no matter what was asked for. Only the
  owner key may grant it, the same rule host filesystem access already follows.
- **Other localm servers running on this machine can now be seen and stopped
  from the app.** `localm ps` and `localm stop <id>` had no GUI form, so a
  second `localm gui`/`localm serve` running for a different project (or
  started from the terminal) was invisible from the app: nothing showed its
  address, and there was no way to stop it without a terminal. Settings >
  Server & network now lists every other instance registered on this machine
  (its directory, address and whether it answers) with a Stop button per row.
  The server behind the page you are looking at is not listed there; use
  Server controls, right above, for that one.
- **Settings can now rebuild the native app launcher.** `localm make-launcher
  --force` was terminal-only, and its one real use, refreshing the copied
  interpreter after a Python upgrade, is exactly when the GUI is what is
  running rather than a fresh terminal. A Rebuild launcher button under
  Settings > Updates does the same thing the command does, and reports the
  same notes back (for example when it could not stamp the icon, or removed a
  build that failed its own self-check) instead of a bare success or failure.
- **The ComfyUI update panel can now target a specific commit.** `localm comfy
  update --commit` was an advanced, terminal-only testing knob for trying a
  candidate ComfyUI version before it is pinned. Settings > Media's Update
  action gains the same optional field; left blank, an update behaves exactly
  as before and moves to the shipped pin.
- **`localm job show` and `localm job results` read a scheduled job from the
  terminal.** `job list` only ever printed a one-line summary, so seeing a
  job's prompt, model, working directory, file-access scope, or whether it can
  run the shell-capable coder (`allow_shell`) meant opening the Jobs tab.
  `localm job show ID` now prints the full definition, and `localm job results
  ID` lists past runs newest first with each one's status and output or error,
  paged with `--limit`/`--offset` for a long history.
- **ComfyUI workflows can now be uploaded, listed and deleted from the
  terminal.** `localm comfy workflow list <media>` shows every uploaded
  workflow and which one is active, `add` uploads one (`--use` to select it
  right away), `use` selects an already-uploaded one (or `--clear` to fall
  back to the shipped default), and `rm` deletes one - refusing to delete
  whichever is active. Until now a terminal-only user could point
  `localm image`/`music`/`video` at a chosen workflow (`localm plugin config
  <media> workflow <file>`), but had no way to see what was actually
  uploaded, add a new one, or remove one - the same picker the GUI's
  Image/Music/Video pages already had.
- **The inference runtime can now be rolled back from Settings, the same
  build `setup-llama --rollback` returns to.** The runtime picker could
  already install a specific build by its release tag, but had no way to
  know or reach the one that was actually working before a bad upstream
  release. A **Roll back** button appears next to it only when an earlier
  build is on record for the installed backend, names that build before you
  press it, and asks for confirmation since it replaces a runtime that
  currently works. The self-contained AMD ROCm build has nothing to offer
  here, matching the terminal command: its build is fixed by the localm
  release, not chosen from upstream.
- **Minting a key from the app can now grant three more privileged scopes:
  `keys:admin`, `plugins:admin` and `config:write`.** Only `admin` and
  `coder:full` had a checkbox, so an owner minting a device key from Settings
  could hand out full admin or shell access but not a narrower key that only
  manages other keys, only manages plugins, or only changes settings. All
  five scopes the server treats as privileged are now offered and, exactly
  like the two that were already there, greyed out and refused for anyone
  minting from a merely `keys:admin` device.
- **Music generation gained three new ACE-Step controls: sampler, scheduler and
  shift.** The workflow template hardcoded these; they can now be overridden
  per generation the same way seed, steps and cfg already were. Leaving them
  unset behaves exactly as before.
- **Chat's web search/fetch tool calls are now grammar-constrained, not just
  prompted for.** Once the model starts a `<tool_call>`, a lazy grammar forces
  it to be valid tool-call JSON instead of hoping the model followed the
  instructions - the same protection the coder plugin's tool calls already
  had. Applies to both the interactive chat web toggle and scheduled chat
  jobs, on local grammar-capable backends; toggle with the new "Grammar-
  constrain chat tool calls" setting.
- **"Show remote images in replies" can now ask you first, per site.** The
  setting gained a third choice between off and on: set it to "ask" and localm
  checks with you once per site per conversation before any image from that site
  is fetched. Nothing leaves your machine while you decide, and choosing not to
  load leaves a note in the reply saying so. A reply's image address is chosen by
  the model and the address itself can carry information out, so "on" only moves
  the request from your browser to this machine, it does not stop it; "ask" is
  the setting that stops it for a site you have not agreed to. Your answers last
  for that conversation in that browser tab only: they are never saved to disk,
  never shared with another conversation, and a reload asks again. Off is still
  the default, and an install that had this switched on before keeps working
  exactly as it did.
- **Explicit downloads can now proceed while Network access is off.** A model
  pull, a HuggingFace search, or a vision-projector/voice/embedding-model
  fetch you asked for used to be refused the same way a model's own request
  is. There is now a setting, **Settings > Server & network > Outbound
  access > "Allow model downloads while network access is off"**, **off by
  default**. Turning it on exempts only your own explicit downloads; Network
  access (net_mode) itself still refuses everything a model could trigger on
  its own, unconditionally.

### Changed
- **The chat parameters drawer and the image, music and video generation forms
  now keep their rarely-touched settings behind an "Advanced" section.** Chat
  leads with the persona, the system prompt and temperature; the Studio forms
  lead with the prompt and the settings that come with a real starting value.
  Seed, sampling, resolution, steps, CFG, denoise, LoRA strengths and the GBNF
  grammar fold away behind a single line that names what is inside, so you can
  tell whether it is worth opening. It opens itself whenever something fills
  those fields in - restoring an image's settings from history, or applying a
  persona - and a collapsed section that still holds values says how many, so a
  setting can never quietly change your output from behind a fold.
- **Status indicators across the GUI (the chat connection line, the coder
  session and tool-call states, model and plugin status tags, the Knowledge
  and Studio badges, and the Settings ComfyUI indicator) now render
  consistently as small colored pills**, matching the style already used for
  job/run status elsewhere. Two small display bugs are fixed along the way: a
  model that is merely loaded in memory no longer looks identical to (and no
  longer highlights its row like) the one actually active, and a missing
  Studio model file no longer shows a color that ignores your light/dark theme.
- **Settings > System no longer shows "Updates" twice.** The app-update card,
  the Runtime update card, and the update-behavior toggles (offer prereleases,
  ignore network access) are now one "Updates" card with three clearly
  labeled parts, instead of three separate panels.
- **Two places that said the same thing twice now say it once.** The sidebar
  showed the active model's name in a badge directly under the dropdown that
  was already showing it, and the Image, Music and Video workflow lists marked
  the workflow in use with a radio dot as well as the highlighted bar down the
  side of the row. The duplicate badge and the dot are gone. The badge still
  appears for everything the dropdown cannot tell you, which is the part worth
  keeping: while a model is loading or unloading, and for any problem, such as
  a failed load or the server becoming unreachable.

### Fixed
- **Indexing a broken image into a knowledge base no longer stores an internal
  error message as if it were the image's description.** If the model
  answering an image-description request crashed on unreadable image data, its
  own error text was saved and returned as a normal search result. That path
  is now treated as a failed description, the same as any other failure, and
  the file is skipped rather than indexed with the wrong content.
- **A ComfyUI instance localm launched itself is no longer left running after
  you stop or restart the server.** It runs in its own detached process so it
  can be closed cleanly on demand, but nothing closed it when the server
  itself stopped or restarted - it just kept running in the background,
  still holding whatever model it last had loaded, and a repeated
  restart could leave several of these behind at once. Stopping or
  restarting localm now stops any ComfyUI it launched along with it.
- **Starting localm no longer prints an unprompted note about a missing vision
  projector.** Launch used to include a network check for each vision-capable
  model's companion projector file, and if network access was off it announced
  that as a startup warning nobody had asked for. That check now runs quietly
  in the background once the app is already serving, instead of during
  startup, and it no longer shows up as a startup message either way.
- **A HuggingFace search or model pull refused because Network access is
  off now points you to Settings, not a command line the GUI has no way to
  run.** And on the Models page, that refusal no longer reloads the whole
  app and lands you back on Chat mid-search, discarding your query.
- **Ctrl+C in the server window now stops the server the same way the GUI's Stop
  button does.** It used to print a `gguf worker process crashed` report with a
  KeyboardInterrupt traceback, because the interrupt reaches every process
  sharing the console and the model worker took it as a crash - an intentional
  stop that looked like a failure. The stop now unloads the model, releases the
  embedding model, stops any download or setup still running, and clears the
  crash marker so the next start does not report a crash that never happened.
  A stop no longer waits for a chat that is still generating: in-flight replies
  get a few seconds to finish and are then ended, instead of holding the server
  open until the longest one completes.
- **The debug log no longer fills with the same few sentences about which VRAM
  source is being used.** On a Windows AMD box with the bundled runtime loaded,
  four notes explaining that choice were written every time the interface asked
  for GPU information, which is continuously while a page is open, so a debug log
  recorded dozens of copies a minute and anything else in it was buried. The
  choice cannot change while localm is running, so it is now worked out once,
  written once, and written again only if it changes. Free VRAM is still read
  fresh every time and the readings are unchanged.
- **A remote image a reply links to no longer disappears without explanation.**
  Showing remote images is off by default, and several other things can stop one
  loading: the host is not on your Allowed domains list, it is unreachable, the
  file is too big, or it is not an image. All of those used to look identical -
  the picture simply was not there, and when the model wrote no caption there was
  nothing on screen at all to say one had been meant. A short note now takes its
  place and says which of those happened, so a setting you can change is not
  mistaken for a broken link. The Allowed domains list is also now mentioned in
  the remote-images setting itself.
- **Your sampling settings now apply to every token on a multi-token-prediction
  model.** These models predict a token ahead and check the guess against the
  main model. That check ignored the temperature, top-k, top-p and repetition
  penalty on your request, so a large share of the reply came out as if you had
  asked for greedy decoding, and the repetition penalty that exists to stop a
  reply looping was not consulted for those tokens. When the guess turns out to
  be wrong, the model's own token is now used instead of being thrown away.
- **A stray turn marker no longer opens a reply.** Some models emit their own
  training-format turn markers as ordinary text. Several dialects were already
  removed before the reply reached you, but the ChatML, Llama 3 and Gemma ones
  were not, so a reply could begin with a stray marker or a bare role word such
  as "model". A role word the model writes in ordinary prose is untouched.
- **The app no longer reloads itself when a reply links a remote image.**
  Showing remote images in replies is off by default, and the request the page
  makes for one is refused while it is off. The app mistook that refusal for its
  own login being rejected, so it cleared its offline cache and reloaded the
  page in the middle of the reply that carried the image, losing it. A genuinely
  expired login still recovers the same way it did before.
- **Chat comes back on the model you were actually using after generating
  media.** Making an image, music, or video unloads the chat model to free up
  VRAM and reloads it when the job finishes. That reload asked for the model
  the server started with instead of the one you had switched to, so anyone
  who had picked a different model returned from a generation talking to a
  different model than they left, with nothing saying it had changed.
- **One request using a grammar no longer holds up everyone else.** Checking a
  grammar before generation has to wait for the model to answer, and that wait
  was blocking every other request on the server rather than just the one that
  asked for it, so a single constrained request could briefly stall chat for
  every other client. The check now runs alongside other work.
- **A model in use can no longer be deleted out from under itself.** Removing a
  model through an AI assistant (the MCP `remove_model` tool) or from a
  terminal (`localm rm`) deleted its file without checking whether anything
  was still using it, so a model you had just been chatting with could have
  its downloaded file destroyed while loaded. The same removal in the app has
  always refused this. It now refuses everywhere, naming what is holding the
  model, and it also checks a running localm server rather than trusting only
  its own state. When it cannot reach a running server to ask, it refuses
  rather than assuming the model is free.
- **Downloading the same model twice at once no longer corrupts the download.**
  Two downloads of the same direct URL wrote into a single partly-downloaded
  file, interleaving their bytes: the download finished and then failed its
  checksum, or, with no checksum to check against, was registered as a model
  that does not load. Downloads to the same destination are now serialised,
  including between the app and `localm pull` run in a terminal. A download
  interrupted by a crash is still picked up where it left off.
- **Starting a download that is already running is refused.** Asking the app to
  download a model it is already downloading started a second job that could
  only fail; it now points at the running one instead.
- **Oversized prompts exceeding model context capacity are now rejected up front
  without crashing the inference worker or evicting loaded models.** Sending a
  prompt whose token count exceeded `n_ctx_max` previously reached the native
  worker process, where an uncaught context ceiling error terminated the worker
  process (`worker exit 1`). This caused the model to be evicted from memory and
  VRAM for all users. The `/v1/chat/completions` and `/v1/completions` routes now
  validate prompt token length against context capacity before dispatch, returning
  HTTP 413 Payload Too Large with clear capacity guidance, and backend workers
  cleanly catch and marshal context overflow errors without process termination.
- **Loading a model no longer hangs indefinitely when the graphics driver is
  busy.** Before loading, localm reads how much video memory is free. That read
  had no time limit, so on a machine where the graphics driver was wedged or
  heavily contended it could simply never come back: loading stopped after
  "Loading <model>", nothing further was printed, no error appeared, and the
  server reported the model as not loaded for as long as it ran. The read is now
  time-limited, and it is skipped entirely when localm has already established
  that the graphics stack is not answering. If it does run out of time it says
  so once and falls back to its other measurement, so the load either continues
  or fails with a real message.
- **Starting a second embedding-model setup while one is running is now
  refused.** Choosing a new embedding model twice in quick succession left both
  setups stuck on "Loading and testing the model..." with no progress and no
  error, and made other pages stop responding for as long as it lasted. The
  second attempt is now declined with a message telling you the first one is
  still going, and the first one carries on normally.
- **Four things that could freeze the whole app, not just the request that
  caused them, no longer do.** Warming up the embedding model, the Knowledge
  page's own status check, an embedding request that fails, and sending a bug
  report each waited for something slow without letting anything else through -
  so while one of them waited, every open tab, every reply being streamed and
  every background job stopped too. The worst was a picture in a reply: with the
  proxy for remote images turned on, one link to a host that never answers froze
  everything for 15 seconds, and the link comes from the model, not from you.
  Each of them now waits on its own without stopping anything else - an unrelated
  request during that same stalled image fetch went from 14 seconds to a
  hundredth of a second - and they still do exactly what they did before.
- **The GPU load figure on AMD cards now reports the whole GPU.** It was reading
  a Windows per-engine counter and showing whichever engine was busiest, which
  could be another program entirely: with the card at 99 percent, the sidebar
  showed 7 percent, because that 7 percent was a screen-recording program's
  video encoder and the counter did not reflect what the card was really doing.
  localm now asks the card itself, through the same AMD interface the vendor's
  own tools use, so the figure covers all work on the GPU whoever caused it and
  matches what AMD's control panel and GPU-Z show. NVIDIA cards are unchanged,
  and other cards keep the previous counter. If the card cannot be asked, the
  figure is left out rather than shown as zero.
- **The sidebar no longer shows total VRAM as though it were VRAM in use.** When
  the amount in use could not be read, the readout printed the card's total
  under the same "VRAM" label used for used-of-total, so an empty card read as a
  completely full one. It now says "16.0 GB total" in that case.
- **Pages no longer jump back to the top when something on them changes.**
  Sorting or filtering the model list, using a row action on it, and saving any
  settings section all rebuilt the page in a way that briefly left it with
  nothing to scroll, so the browser sent you back to the top and you had to find
  your place again. Both pages now put their new contents in place in a single
  step, without emptying or hiding what is already on screen, so your scroll
  position stays exactly where it was. The model list also no longer flashes
  empty while it reloads.
- **IPv6 addresses can now be used as the bind address.** `localm gui -H ::`
  and `localm serve -H ::` (or any IPv6 literal, such as `::1` or one interface's
  own address) used to stop the server before it started, with an unexpected-error
  message and a bug-report offer; Settings > Server > Bind address refused IPv6
  values outright for the same reason. Both now work. `::` listens for IPv4 and
  IPv6 clients alike, so binding it reaches everything `0.0.0.0` did and more, and
  the addresses localm prints for you to open are written the way a browser needs
  them. An address with a zone suffix (`fe80::1%eth0`) is still declined, with a
  message saying so, because the zone number only means anything on the machine
  that wrote it.
- **The coder no longer carries a recording session into a project you marked
  private.** A session's persistence is fixed when it starts and cannot be
  lowered afterwards, but its transcript is written wherever the session has got
  to - so changing directory into a project whose `.localcoder/config.toml` asks
  for privacy would have left a full record inside it. That move is now refused,
  in the app and at the `/cd` prompt, naming the project's own setting; start a
  fresh session in that directory instead.
- **A memory write can no longer be silently lost to another localm process.**
  Two localm processes writing your memory at the same time - a `localm memory`
  command, or `localm setup-embeddings`, alongside a running server distilling
  facts in the background - could each save their own copy of the store, and
  whichever finished last won. The other write disappeared with nothing said:
  `localm memory add` printed "Remembered ..." and exited 0 for a fact that was
  no longer there. Writes to one memory store are now serialised across
  processes. They wait their turn, and if the store is genuinely busy for too
  long the command names what is holding it and changes nothing, instead of
  reporting a success that did not happen. Reading your memory is unaffected and
  never waits.
- **"Still confirm shell commands" refused the command instead of asking you.**
  Ticking it alongside auto-approve, as its own description invites, left a coder
  session in the app with no way to put the question to anyone: every shell
  command it reached was declined outright, reported as needing a confirmation
  that could not be obtained. The approval card now appears, and the session no
  longer needs to be restarted to get one.
- **A restart no longer forgets what the server was in the middle of, and
  stopping the server no longer leaves the work running behind it.** A model
  pull, a runtime install or a ComfyUI setup was only ever remembered in
  memory, so a restart (or a crash) came back reporting that nothing had been
  happening - while the download itself kept running, invisible to every
  surface, or was cut off part-way with no trace either way. localm now keeps a
  record of what is in flight, and after it comes back the Activity panel and
  `localm status` list anything that was interrupted, marked **interrupted**
  rather than "failed", because whether the work finished is genuinely unknown.
  Stopping or restarting the server now also stops the background work it
  started, instead of abandoning it to keep writing to your data folder.
- **An Image, Music or Video model dropdown could show a file that is not the
  one generation would use.** When the chosen workflow named a model your
  ComfyUI does not have, but your ComfyUI had other files of the same kind, the
  dropdown quietly displayed the first of those instead - so the panel read as
  though a model were selected while generating still used the missing one. The
  dropdown now shows the file the workflow actually names, marked "not
  installed", so picking a different one is a deliberate choice.
- **The mic button's speech-model download now obeys the network policy.** The
  one-time Whisper fetch used to run regardless of net_mode; now "off" truly
  blocks it, "ask" waits for your explicit go-ahead, and a model already on
  disk loads with no network access at all. When the download is blocked, the
  mic no longer lets you record and then fail: it is greyed out with the real
  reason up front.
- **A model pull can no longer be reported as failed after it actually
  succeeded.** Once a download finishes, localm prints a green checkmark to
  confirm the checksum was verified - and on some machines that confirmation
  itself could crash. The download, checksum check and registration had
  already completed by then, but the crash still made the whole pull come
  back as failed (and, from the GUI, file a bug report) even though the model
  was fully downloaded and ready to use. That confirmation can no longer turn
  a completed, verified download into a false failure.
- **Chat no longer refuses to send after a model is unloaded, when the model
  was going to reload by itself anyway.** Both the "unload model" button and
  the optional unload-after-idle setting say the model comes back on your next
  message, and the server really does do that. The chat box did not know it:
  it saw no model in memory, said "No model loaded - load a model on the
  sidebar before chatting", and refused to send the very message that would
  have brought the model back. The sidebar showed "No model loaded" too, so
  the only way out was picking the model again by hand. It now keeps showing
  the model that will answer your next message, and sending one reloads it, as
  both of those features already promised. A message still cannot be sent when
  there is genuinely nothing to load.
- **Reading replies aloud now works with no internet connection.** The
  neural voice needed a piece of its engine downloaded from a public CDN
  every time it started, so on a machine that was offline, air-gapped, or
  behind a filtering proxy or strict firewall the speech engine simply
  failed to start and localm fell back to the robotic system voice. That
  engine now ships inside localm, so the voice starts from your own
  machine. Nothing else changes: the same voices, the same quality. The
  voice model itself is still downloaded once on first use and then kept
  in the browser, as before.
- **A link straight to a media or tool page now opens that page.** Opening the
  GUI at a Images, Music, Video, Knowledge, Coder or Jobs link (for example a
  bookmark or a link you shared with yourself) quietly showed a different page
  instead, and the address it was given was discarded, so reloading did not help
  it either. Those links now land where they point. A link to a page that does
  not exist still falls back to Models.
- **Hybrid models (Qwen3-Next, Granite 4 H, LFM2, Jamba, Falcon-H1 and
  similar) are no longer charged several times too much VRAM for their
  context.** These architectures use a growing KV cache on only some of their
  layers; the rest keep a fixed-size state that costs nothing per token.
  localm charged every layer, so on a real Qwen3-Next it asked for 4x the KV
  cache actually needed (12 GB instead of 3 GB at a 128k context). Nothing
  failed visibly - localm just quietly put the cache in system RAM, or
  offloaded fewer layers to the GPU, on models that would have fit. localm now
  works out which layers actually hold a cache by reading the model file, so
  these models get the exact figure rather than an estimate. If a particular
  file does not record enough to tell, localm falls back to its general
  estimate instead of using a number it knows is wrong.
- **The "reload chat model after generation" toggle now applies only to the page
  you set it on.** It sat on the Images page but wrote a single shared setting,
  so turning it off there silently turned off the VRAM handover for Music and
  Video too, with nothing on those pages to say so or change it back. Each of
  the three pages now has its own toggle writing its own setting. An existing
  setting keeps its meaning: a generator you have never set individually still
  follows the shared default.
- **A generated file whose preview cannot be shown now says so.** A file the
  browser refused to decode left a silent blank tile; it now reads "Preview
  unavailable" and keeps its card, so you can still open, move or delete it.
- **Fixed a startup deadlock that could freeze the entire server for good.**
  Launching with a model to preload while the memory/knowledge plugins needed
  the embedding model could deadlock the model loader against the embedder
  (each waiting forever on a lock the other held). From then on every part of
  the GUI that touched model or embedding status silently stopped answering,
  the browser ran out of connections, and the whole app appeared dead - with
  the server process still running at 0% CPU, indefinitely. The two code
  paths now take those locks in one agreed order, so the deadlock cannot
  form. This was the cause of the 2026-08-18 "clicked Launch ComfyUI and
  everything stopped loading" hang: the click itself was just the first
  casualty a user could see.
- **The status window is bigger and its text actually readable.** The red
  status line no longer gets cut off mid-sentence (it wraps to the window
  width), and the log pane wraps long lines instead of auto-scrolling
  sideways into unreadable fragments.
- **Rows in the Models, Plugins, Knowledge and Jobs tables no longer break into
  pieces.** Each row's separator line stopped partway across and continued at a
  different height, and the buttons at the end of a row stacked one per line, so a
  single model or plugin sprawled over several lines of staggered rules. Every
  record is one row again, with one line under it and its controls side by side.
  On the Models page the type dropdown has moved into the Role column, where it
  now doubles as that column's coloured type tag instead of sitting twice in the
  same row, and a long source path is shortened with the full value on hover. In a
  window too narrow for the whole table, the table scrolls on its own rather than
  pushing the page sideways.
- **The Images/Music/Video workflow panel now matches the rest of the GUI.**
  The pick/delete buttons used one-off styling instead of the shared button
  classes, and hovering or selecting a workflow row showed no feedback.
- **Setup's download progress bars no longer come out garbled.** While it
  installed PyTorch and the other Python packages, setup printed a periodic
  "still working" line straight over the live download progress, which left
  half-finished progress bars stranded up the screen and made it look like
  something had gone wrong. That line is now only used for the one step that
  really is silent (creating the environment), so every install that shows its
  own progress renders normally.
- **The desktop-shortcut question at the end of setup now explains itself.**
  "Launcher" and "Web GUI directly" never said what either one actually opens,
  and "Web GUI" was simply wrong if you had chosen the app window earlier in
  setup. Both options now describe what they do, and the one that skips the
  menu names the window mode you picked. The closing "how to start" line
  follows your answer instead of always naming the launcher script, and a
  shortcut that could not be created is no longer described as though it
  exists. The step that builds the branded app executable no longer calls
  itself "the launcher" too, which is what made that screen so confusing.
- **Setup no longer invents a clash with an existing `localm` command.** On
  Windows, answering yes to "Make 'localm' runnable from any terminal?"
  warned that a `localm` command already existed and added this install at
  lower priority anyway. There was no other command: it had found this
  install's own launcher file, because Windows looks in the current folder
  first. It now only reports a `localm` that is genuinely on your PATH, and
  when there is one it asks whether this install or the existing one should
  run, instead of deciding for you and telling you to reorder PATH yourself.
- **The context-usage gauge now shows up on ordinary chat replies, not only
  while streaming.** A non-streaming `/v1/chat/completions` reply left the
  context-window figure out of its usage report even though streaming
  replies always included it, so the gauge only ever drew for a streamed
  answer. Both reply modes now report it the same way.
- **A coder skill's `allowed-tools` can no longer be outrun by the same reply
  that loads the skill.** The restriction was applied when `use_skill` finished,
  but the coder runs several read-only tools from one reply at the same time, so
  a tool the skill does not declare could be started alongside `use_skill` and
  finish before the restriction took hold. Loading a skill now happens on its
  own: anything the model asked for beforehand completes first, and nothing
  after it starts until the declared `allowed-tools` are in force.
- **Setting a model alias, renaming a model or a generated file, saving a
  named key preset, saving a persona, or moving a chat into a folder now
  works on every browser.** These used the browser's own text-input popup,
  which some mobile and PWA browsers block; on those, clicking any of them
  did nothing at all, with no error or message to explain why. They now use
  localm's own in-page dialog instead, the same one already used for
  delete-confirmation prompts.
- **`localm setup-llama --backend sycl` no longer tells Windows users they
  need a separate Intel oneAPI install.** The Windows SYCL build actually
  bundles the whole oneAPI runtime alongside the inference library, the same
  self-contained shape as the CUDA and ROCm builds, so nothing beyond the
  Intel GPU driver is needed there. The printed note was wrong for Windows;
  it still correctly asks for a system oneAPI install on Linux, where the
  runtime genuinely is not bundled.
- **Uploading a media workflow while a generation is reading it can no longer
  leave a corrupted file on disk.** The upload write and a generation's read
  raced with no protection between them; a generation starting at exactly the
  wrong moment could load a half-written, invalid workflow file. The write is
  now atomic, so a read always sees the complete file, before or after the
  upload.
- **`localm setup-llama --backend amd-rocm` now fetches the self-contained
  build that matches your AMD card, instead of always the RX 6000 one.**
  Explicitly requesting this backend on an RX 7000 or RX 9000 card silently
  downloaded the RX 6000 (RDNA2) build by name, the same file every card
  received. The GPU is now detected first and the matching build is
  requested; RX 6000 keeps its existing behavior unchanged.
- **Older saved memories could stay stuck in plain keyword search forever.**
  A memory saved before an embedding model was installed, or a backlog too
  large for the usual background pass to catch up on its own, used to need a
  manual `localm setup-embeddings` re-run to become semantically searchable
  again. A background pass now catches them up on its own while you use
  localm, so semantic recall keeps improving without a manual step.
- **Stopping a reply, or running out of memory as one starts, now tells you what
  actually happened.** Pressing Stop, unloading the model mid-reply, or asking
  for a context window that does not fit in free memory reported an internal
  error about a missing variable, instead of either stopping quietly or showing
  the out-of-memory message that tells you to start a new chat or lower
  `n_ctx_max`. The real reason now reaches you.
- **Structured replies from multi-token-prediction models no longer break their
  own format.** On a model with MTP heads, a JSON-schema, GBNF or tool-calling
  request drafted tokens ahead using a sampler that ignored the grammar, so the
  reply could contain text the schema forbids and feeding such a token back
  could abort generation outright. Constrained requests now generate one token
  at a time; ordinary chat keeps the speedup.
- **The coder's shell tools can now actually launch npm, yarn and npx on
  Windows.** `run_shell` and `run_shell_background` previously handed Windows
  the bare command name, which it cannot start directly for these three -
  even with npm installed and on PATH, any coder-run `npm test` or manual
  `npm`/`npx` command failed immediately with "the system cannot find the
  file specified". They now resolve to the real, launchable path first.
- **A coder sub-agent dispatched to its own worktree (`spawn_agent_background`,
  `dispatch_parallel`) now has its changes checked before being reported as
  finished.** Such a sub-agent's diff previously sat in a separate worktree
  that nothing ever verified, so it could be reported as done even when its
  change did not actually work. It now runs the project's own check (or
  whichever one you configured) against its own worktree, and a failing
  check is reported as a failure rather than a success.
- **HuggingFace-backend chats no longer crash instead of reporting a clean
  error.** A prompt too long for the model's context window reported a
  generic worker crash rather than the intended, actionable "conversation
  exceeded context window" message, and stopping the server while a reply
  was still streaming could itself crash instead of ending the reply
  cleanly.
- **The Linux CUDA fallback error no longer points at a file you can't
  open.** When no CUDA build is available for your llama.cpp tag on Linux
  and localm falls back to Vulkan, the error referenced an internal
  maintainer file that isn't part of the published repository. It now
  states the same condition without the dead reference.
- **`setup.sh` could quit partway through with a confusing error on a rare,
  filesystem-specific install hiccup.** Occasionally, mostly on a WSL clone
  under a Windows-drive mount, `uv pip install` reports success while the
  `localm` command itself does not get written; setup already retried once
  and warned loudly when it was still missing, but then went on to run that
  missing command anyway for the native runtime, which quit setup early with
  an unrelated-looking "failed" message instead of the real cause. Setup now
  skips the steps that need the command and finishes normally, and its
  closing summary says plainly which commands will not work until it is
  fixed (the graphical launcher is unaffected).
- **`localm gui` no longer disappears without a trace when the GUI fails to
  load.** A broken or partial install could make the GUI's own code fail to
  import, and the command was then dropped entirely - Click answered "No
  such command 'gui'", exactly what a typo would look like. It now tells you
  the GUI could not be loaded and how to see the underlying error.
- **`localm gui --api-mode` no longer points you at a web page it never
  serves.** With no model loaded, the model line said "add one on the Models
  page"; the address line was always labelled "Open the GUI" and carried a
  browser-only deep link. `--api-mode` mounts no GUI at all, so both now
  match: the hint points at `localm pull <name>`, and the address line shows
  the plain API base under "API base".
- **`localm run MODEL -p "..."` now exits non-zero when the model load
  hard-fails.** A crashed or unreachable load printed the real error in red
  but still exited 0 with empty output, so a script chaining on the exit
  code (or piping the output to a next step) could not tell that call apart
  from a normal reply that happened to be empty.
- **A worker crash no longer points you at a debug log that was never
  written.** The message after a native crash always said "full trace in
  the debug log", even when nothing had turned debug mode on and no such
  file existed. It now says so only when the log actually exists, and
  otherwise tells you to rerun with `--debug` to get one.
- **Setting `LOCALM_MTMD_CPU=1` to skip the GPU attempt for image
  understanding no longer reports that as a GPU failure.** With the
  variable set, the vision projector deliberately never tries the GPU -
  but the log said "the vision projector could not be loaded onto the
  GPU... Set LOCALM_MTMD_CPU=1 to skip the GPU attempt entirely", advising
  you to set the exact variable you had already set. It now says plainly
  that CPU encoding is being used as requested. A genuine GPU failure
  (the variable unset) still gets the original message.
- **`localm comfy status` no longer uses "own" for two opposite things.**
  With nothing configured, "Preferred target : own" and "Target now : your
  own ComfyUI" read as agreeing while naming opposite targets - the first
  means localm's managed ComfyUI, the second means a separate install you
  run yourself. The preferred-target line now spells out the same wording
  the second line already uses.
- **A model file whose copy or download was interrupted partway through could
  still be registered as a usable model.** localm already waits for a new
  file to stop changing before registering it, and rejects files too small
  to be real, but a file that had genuinely stopped changing - because the
  copy simply never finished, not because it was still mid-write - could
  clear both of those checks once enough of it had landed on disk. Loading
  it already failed cleanly with an error instead of crashing anything; such
  a file is now also skipped during registration, the same as any other file
  that is not really a usable model.
- **`localm relocate` no longer tells you a mid-copy file "is not a GGUF
  model file".** Pointing it at a file whose copy or download has not
  finished yet used to get the same generic rejection as a genuinely
  unrelated file, which sends you looking for the wrong problem. It now
  says the file looks incomplete and to try again once the copy or download
  has finished.
- **`setup.sh`'s automatic hardware detection now recognizes Apple Silicon.**
  It previously had no check for it at all, so every Mac was silently treated
  as having no GPU: the acceleration prompt never appeared, and the
  recommended llama.cpp backend was forced to CPU-only even though localm's
  own hardware policy already knew Metal was the right choice for that
  hardware. Apple Silicon is now detected correctly, so `setup.sh` offers and
  defaults to the Metal backend, which can also be picked directly from the
  interactive backend menu.

### Security
- **Two more model families' role markers are now defanged in untrusted text.**
  A fetched page, a tool result or a stored memory is escaped so it cannot forge
  a system or assistant turn using the model's own delimiters. EXAONE and GLM
  models use delimiters that were not in that list, so text aimed at them passed
  through unescaped.
- **Turning network access off no longer left the voice model able to
  download anyway.** The neural text-to-speech voice is fetched by your
  browser directly, so localm's network switch, which every other
  network-triggering action already obeys, had no way to see or stop that
  request. With network access set to "off" it is now refused outright, with
  a message telling you how to re-enable it; set to "ask first" (the
  default), it now asks for a one-time confirmation before the ~86 MB
  download starts, the same way an embedding model download already does.
  Nothing changes once the voice model has already been downloaded once.
- **A form inside a model's reply can no longer send anything off your
  machine.** Replies are rendered as HTML, so a reply could draw a form, and
  the app's security policy did not say where forms were allowed to be
  submitted. That is a separate setting from the one covering scripts, images
  and network requests, and it was not set, which left submission unrestricted.
  A reply that drew a convincing "confirm your key" box could therefore have
  sent whatever you typed into it to any address on the internet, and because
  no script is involved, neither the HTML sanitiser nor the script rules
  applied. Nothing in localm submits a form, so form submission is now refused
  outright, in the chat view and in the artifact pane alike.
- **A credential you paste into a bug report is now removed, including the ways
  a `.env` file actually writes one.** A report already stripped a secret that
  appeared as a URL query parameter, but not one written as a plain
  `api_key=...` line, one behind a prefix such as `OPENAI_API_KEY=`, or one
  whose value was in quotes. The quoted form was the worst of the three,
  because the secret stayed in the report with a "redacted" marker printed
  immediately in front of it, so the line read as though it had been dealt
  with. All three are now removed, in the report form, in `localm bug-report`,
  and in the fallback reporter used when localm will not start. Settings are
  deliberately left readable, so a report still shows you and the maintainer
  things like `require_auth=true` and `n_gpu_layers=35` that are needed to
  work out what went wrong.
- **A key minted through the API can now actually be granted access to your
  files on disk.** A `POST /v1/keys` request could ask for `fs_access` (host
  filesystem reach), and it was silently dropped - every key came back with
  no filesystem access no matter what was requested, even when the owner
  asked for it. The owner key can now actually grant it; the same request
  from any other key is refused instead of being quietly downgraded to none.
- **Model downloads from HuggingFace now always go to the real HuggingFace,
  even if something else on your machine points elsewhere.** The library
  localm uses to pull models honors an environment variable, `HF_ENDPOINT`,
  that redirects every download to a different host - something another
  program on your machine could set without your knowledge, and localm never
  exposed a setting of its own that could override it. Every model,
  projector and embedding download now pins the real HuggingFace address
  explicitly, so a stray `HF_ENDPOINT` can no longer change where your
  models come from.
- **A vulnerable version of `setuptools`, pulled in by the speech-to-text
  backend, is no longer installed.** CVE-2026-59890 is a flaw in how versions
  before 83.0.0 build source packages; localm never builds one itself, but
  the floor is now raised above the fix regardless.
- **The uninstaller's data-directory guard now refuses a delete it cannot
  fully check, instead of letting it through.** Before `--purge-data` removes
  a folder, several checks confirm it is not your home directory, the
  filesystem root, or the repository itself. If the home-directory check
  could not run, it was silently skipped rather than treated as a refusal.
  It now refuses whenever that check cannot be completed, matching how every
  other check in the same guard already behaved.
- **A bug report's error detail could still show your Windows account name,
  even though the same path elsewhere in the report was already hidden.**
  Some error messages quote the file path in a form the scrubber did not
  recognize, so the account name inside it slipped through untouched. Every
  form of the path is now caught.
- **The no-Python fallback bug reporter no longer files a report without
  asking you first.** If it could not show the confirmation prompt (no
  console attached, or nothing left to type into), it used to go ahead and
  send anyway. It now treats that exactly like you said no: the report is
  saved locally and nothing is sent, with a note on where to find it.
- **The weak-key warning no longer implies an 8-character key is "strong".**
  Binding to the network with a key under the minimum length told you to "set
  a strong key (>= 8 chars)", which read as if hitting that length made the
  key strong. It doesn't: the floor only rules out the shortest, easiest
  guesses, and a short human-chosen key can still be trivially guessable. The
  warning, and the matching message in the GUI's bind-fallback notice, now
  say the length is a floor and point at `localm key generate` for an
  actually strong, random key.
- **`localm memory clear` now actually erases everything it reports erasing.**
  It could print "Erased N remembered and M forgotten fact(s)" and exit
  successfully while two records of your own words stayed on disk: a pending
  suggestion to update or delete a saved fact, and the text of a suggestion
  you had already turned down. Both are now removed by the same command, and
  it refuses to report success if anything is still there afterward.

## [0.1.5rc3] - 2026-08-13

### Added
- **A coder skill's `allowed-tools` is now enforced, not just displayed.** A
  `SKILL.md` could declare that it only needs `read_file` and then, once loaded,
  call `run_shell`, `write_file` or `git_push` anyway - the declaration was shown
  to the model as a suggestion and nothing checked it. Loading a skill now
  restricts the session to the tools that skill declares, and anything else is
  refused before it runs. The restriction only ever narrows: it cannot re-enable
  a tool you have switched off, loading a second skill can only narrow it
  further, a sub-agent the skill spawns inherits it, and nothing the model itself
  can call lifts it - your next request does. Skills that declare no
  `allowed-tools` are unrestricted, exactly as before.
- **The llama.cpp compatibility check now catches an upstream change that
  redefines a VALUE, not just one that moves a field.** The check compared where
  each field sits in memory, which cannot see a change to which values are legal
  in it. That is exactly what happened: upstream added a new "let the build
  decide" model-loading mode numbered -1 and made it the default, no field
  moved, the weekly check stayed green, and localm refused to load every build
  from that point on. The same run now also compares the legal values, and keeps
  the two cases apart: a value upstream ADDS is reported loudly but does not
  fail the check, while a value that CHANGES out from under something localm
  already relies on does fail it.
- **Settings can now check for, and apply, a newer llama.cpp runtime build.**
  Previously the only way to move to a different build was the
  `setup-llama --tag`/`--rollback` command line. There is now a Runtime update
  card next to the ordinary Updates card: it reports whether a different build
  is available for your installed backend (honouring a pin if you have set
  one), and an Update button re-provisions it and streams progress, the same
  way the ComfyUI update panel does. A build that fails to load on your
  machine is never kept.
- **You can now pin the llama.cpp build localm installs, and go back to a
  previous one.** localm always fetched whichever llama.cpp release was newest
  on the day you installed, and kept no record of which one that was - so an
  upstream build that misbehaves on your hardware arrived on your next install
  with no way to name it or get away from it. `localm setup-llama --tag b10355`
  now installs exactly that build and keeps it, including through
  `localm update`, until you run `localm setup-llama --tag default` to go back
  to the build localm ships. `localm setup-llama --rollback` returns to the
  previous build you had, without needing to remember its number.
- **`localm doctor` and bug reports now say which llama.cpp build is
  installed.** It was previously impossible to answer that question without
  reading library filenames and guessing. Doctor shows the build and whether it
  is pinned, and warns if a pin has been set but not yet installed; a bug report
  carries the same detail. Installs made before this change say "not recorded"
  rather than guessing a version - re-running `localm setup-llama --force`
  records it.
- **Settings now shows which backend is actually installed, not just what
  would be picked fresh.** The Runtime & GPU section only ever showed the
  recommended default; Live tuning now shows what your install is really
  running. If you have an NVIDIA GPU but are still on the Vulkan backend, a
  dismissable hint says CUDA usually performs better - it only ever informs,
  and never switches anything for you.
- **Settings can now update localm's own ComfyUI.** The managed-ComfyUI box
  offered Set up, Repair and Remove but no way to update, so if you only use
  the GUI you could install localm's ComfyUI and never move it to a newer one.
  There is now an Update button next to Remove, which shows the version you
  have and the version localm ships, and streams the update as it runs. A
  tickbox next to it also reinstalls ComfyUI's dependencies, which is what you
  want when the new version needs packages the old one did not. If your
  ComfyUI was installed by copying rather than as a git checkout it cannot take
  a version update, and the button now says so instead of failing part way in.
- **The Knowledge page can now repair a damaged collection itself.** A
  collection with a corrupt or malformed index used to only tell you to run
  `localm rag repair` in a terminal. There is now a Repair button in the
  collections table and in a collection's info panel, and the damage message
  names what's actually wrong (for example how many chunk lines are
  unreadable) instead of a generic "index damaged". A document you added by
  uploading it from your device has no copy on the server to rebuild from, so
  repair says so plainly and fixes what it can rather than reporting success
  for a run that changed nothing.
- **The Models page now shows which of your models can accept images.** localm
  already worked this out internally - it is how an attached image gets routed
  to a model that can read it - but nothing ever showed you, so the only way to
  find out was to attach an image and see what happened. Models that can take
  image input now carry a "vision" pill in the model list and in the detail
  window. A model localm could not check, because its file is on a drive or
  network share that is not currently reachable, simply shows no pill: it is
  never labelled as text-only on the strength of a file nobody could open.
- **Changing the embedding model from the config API or the CLI now warns you
  what it will cost first, the same way the Knowledge page already did.**
  Switching embedders makes every existing collection unreadable until it is
  re-embedded, and only the GUI picker said so. `PATCH /v1/config` and
  `localm setup-embeddings` now produce the same pre-switch report, naming the
  collections that would be affected, and ask for confirmation before going
  ahead.

### Changed
- **Settings help text is much shorter, and says what a setting does instead of
  arguing for its default.** Thirty fields carried a paragraph each, the longest
  at 559 characters, so the page took three screens of scrolling for eleven
  controls and the warnings that mattered were buried in the ones that did not.
  Help is now capped at 200 characters and leads with the consequence; the
  reasoning it used to carry is kept in full beside the code. Settings that
  referred to "the option above" or "the toggles below" now name the setting
  they mean, which was already wrong in a two-column layout and wrong again
  after settings moved between tabs.
- **Live tuning, the GPU rows and two empty lists now match the rest of the
  Settings page.** The Live tuning controls ran their label and description
  together on one line, unlike every panel above them; "GPU layers" and "Context
  window" appeared twice in the same section with no way to tell the persisted
  setting from the one that applies right now, and are now marked "(next load)"
  and "(running model)". On a multi-GPU machine the note about device numbering
  was printed twice, and both GPU rows explained that they are "shown only when
  more than one GPU is detected" to the only people who could ever read it.
  Deleting all conversations is now styled as the destructive action it is, and
  the Issues and Uploaded files lists say they are empty instead of showing
  nothing at all.
- **localm now installs a llama.cpp build it has actually tested, instead of
  whatever was published most recently.** Until now every `setup-llama` asked
  GitHub for the newest llama.cpp release and installed that - on released
  copies of localm too - so a build nobody had ever run could arrive on your
  machine with no localm update involved, and a release that did not match this
  code left a fresh install unable to load anything. localm now ships a specific
  release that was downloaded, loaded and made to generate text before being
  chosen, and installs exactly that. If you want upstream's newest anyway,
  `localm setup-llama --tag latest` opts in and says plainly that the build is
  untested; `--tag default` comes back. Nothing about naming an exact build with
  `--tag b10355` or `--rollback` changes.
- **A runtime localm's own compatibility check refuses is no longer replaced
  with an older, less-tested one.** When that check rejects a build, setup-llama
  now installs the tested build instead - and only if you had opted in to
  tracking upstream. If the tested build is the one being refused, it says so,
  names it as a fault in localm rather than in the release, and stops rather
  than quietly reaching for something older. It previously tried up to three
  earlier releases and kept whichever one loaded, which could leave you on a
  build that had never been tested at all, reported as a success.
- **Offline and rate-limited installs are checksum-verified again for every
  backend.** When the release listing cannot be reached, setup-llama builds the
  download URL itself; that guess was constructed from a naming convention
  upstream has since changed, so the AMD ROCm downloads on Windows and Linux
  resolved to filenames that no longer exist (a 404) and, being unrecognised,
  carried no checksum. The known asset names of the shipped build are now used
  directly, so this path names a real file and verifies it.
- **localm's own managed ComfyUI now installs v0.31.1**, up from v0.9.2. The
  pinned version had not moved since July and was roughly 21 releases behind
  upstream, which is why "Split media across GPUs (experimental)" reported that
  localm's own ComfyUI was too old to offer per component placement. It now
  offers it. That setting still needs two or more graphics cards, so it
  continues to decline on a single card machine, but for a reason about your
  hardware rather than about the version localm ships.

  **If you already have a localm managed ComfyUI, run `localm comfy update
  --reinstall-requirements` when you update.** v0.31.1 needs several packages
  the old version did not, so a plain `localm comfy update` moves the checkout
  without installing them. The command does say so when it runs, but it is easy
  to miss. A fresh install is unaffected and needs nothing extra.
- **Neural text-to-speech now runs multi-threaded, and is roughly two and a
  half times faster.** The speech backend needs cross-origin isolation before
  the browser will give it the shared memory that threading requires, and the
  GUI did not send the two headers that switch it on, so it ran on a single
  core. Measured on the same sentence (6.3 seconds of audio, warm, median of
  three): 12.9 seconds before, 5.1 seconds after, on a twelve-core machine.
  Because isolation changes how the page is allowed to load anything from
  another origin, every subresource the GUI fetches is now served in a way
  that keeps working under it.

### Fixed
- **Two things localm could fail at and then say nothing about are now
  reported.** If a plugin's install or first-use hook raised, the action
  carried on and nothing was written anywhere at any level, so a plugin that
  only half set itself up was indistinguishable from one that set itself up
  properly. Separately, if the coder could not delete a saved session it was
  clearing (a file held open by something else, for instance), the checkpoint
  quietly stayed on disk, so a later `/resume` could offer you a session you
  believed was gone. Neither is treated as a failure of the thing it was
  attached to, so an install still installs and a finished task still
  finishes, but both now leave a warning naming what went wrong, and a bug
  report carries it.
- **The last case where a requested grammar was quietly ignored is closed: a
  GGUF model now refuses the request instead of answering without the
  grammar.** Asking for a "lazy" grammar (one that lets the model write freely
  until a trigger appears, then enforces the grammar from there) still fell
  through to unconstrained generation in one place - on a llama.cpp runtime too
  old to have the feature, or if the trigger patterns were left out. The reply
  came back as an ordinary success, so nothing told you it did not follow the
  grammar you asked for. It is now refused with a message naming which of the
  two it was and what to do instead, the loaded model stays loaded, and the
  coder carries on without trigger-based tool calls rather than failing.
- **A flood of tool-calling requests can no longer slow down the rest of the
  server.** Requests that use a trigger-gated ("lazy") grammar have their
  trigger patterns safety-checked first, and each check that had to wait its
  turn held one of the server's shared worker threads for up to five seconds.
  How long each one waited was capped; how many could be waiting at once was
  not, so a large enough burst tied up threads that loading a model, embedding
  and counting tokens also need. At most twelve of those checks can now be in
  progress or queued at a time; beyond that, a request is answered immediately
  with the existing "the validator is busy, retry shortly" 503 instead of
  joining the queue. Nothing is accepted without being checked, and an ordinary
  request that does not have to wait is unaffected.
- **Your API key was briefly readable by other accounts on the machine each
  time it was saved.** localm writes the key file, the named-key store and the
  key-derivation file by writing a temporary file and renaming it into place,
  which is what stops a crash mid-write from destroying the old one. The
  finished file was locked down to your account only, but the temporary one was
  not, so on a shared or multi-account machine the full contents sat readable
  by anyone for the length of the write. They are now locked down from the
  moment they are created, before anything is written into them. On Windows
  this matters most: files there inherit the folder's permissions, which
  commonly include read access for all local users.
- **`localm doctor` no longer reports "No GPU detected" when what actually
  happened is that it could not find out.** If the installed llama.cpp runtime
  failed to load while being probed (a graphics driver too old for it is the
  usual cause), doctor gave the same answer as a machine with no graphics card
  at all, and told you to run `localm setup-llama` to install a backend that
  was already installed. The reason the probe gave was collected and then
  thrown away. Doctor now reports that case as its own result, prints the
  reason, and says plainly that this is not a statement about your hardware.
  A machine that genuinely has no GPU is unaffected.
- **Hardware detection can now tell "this machine has no GPU" apart from "the
  detection could not run".** The tools it shells out to (the Windows display
  adapter query, `lspci` on Linux, `uname` on macOS) are missing or blocked
  often enough to matter, and a tool that never answered used to produce the
  same empty result as a tool that answered "nothing here" - including on
  macOS, where a failed check reported an Intel Mac and so passed over the
  Metal recommendation. Detection is still advisory and still falls back to the
  same safe default; it just no longer states a finding it did not make.
- **A download that was redirected off HTTPS onto plain HTTP was followed.**
  localm verifies the certificate of every outbound HTTPS request, but that only
  covers the FIRST hop: the underlying library follows up to ten redirects and
  accepts a plain-HTTP target, so a server (or anything able to answer as one)
  could answer a verified request with a redirect and have the rest arrive in
  cleartext, where it can be read or replaced in transit. Only the update
  download and the calls to your ComfyUI were guarded against this; the
  llama.cpp runtime download, its GitHub and PyPI lookups, the issues list and
  the bug-report upload were not. The runtime download is the one that matters
  most, because its bytes are loaded as a native library and its checksum check
  is opt-in. Every one of them now refuses a redirect that leaves HTTPS, and
  says so instead of failing as though the network had dropped. Redirects that
  stay on HTTPS still work, which is what the real downloads depend on.
- **Web search now reads the promising page instead of answering from the
  search engine's one-line summary.** `web_search` only ever returns snippets;
  reading a page needs a separate `fetch_url` call, and nothing told the model
  to make one, so answers were built from result summaries while looking like
  they came from the pages themselves. Chat and scheduled jobs now both tell it
  to follow up on a result that looks like it holds the answer, matching what
  the coder agent already did.
- **A second tool call in one reply is no longer thrown away in silence.** Chat
  and scheduled jobs run one web lookup per message, but a model that asked for
  two got no hint that the second never happened, so it answered as though it
  had those results too - and the chat transcript displayed both lookups, so it
  looked to you as if both had run. Only the first still runs (asking for two
  at once is not supported here), but the model is now told which call was
  ignored and can ask for it again on its next turn.
- **The Settings update check no longer says "localm is up to date" when it
  could not work out whether it is.** localm orders release tags to decide if
  one is newer than what you are running, and some tags cannot be ordered at
  all - a release named `nightly`, `stable` or `release-5` has no version
  number to compare. That produced the same "not newer" answer as a genuine
  tie, so the GUI reported you were up to date when the honest answer was that
  it could not tell. It now says so, and names the tag it could not read. The
  command line already got this right, and this check also runs by itself
  shortly after startup, so the wrong reassurance was appearing without anyone
  asking for it.
- **A couple of dismissable UI hints (the NVIDIA/Vulkan backend hint, and
  whether the Studio nav group is expanded) could survive turning on privacy
  mode, unlike every other saved GUI preference.** Both were already withheld
  from being written while privacy mode was on; they just were not cleared
  from a browser that had them saved from before it was turned on, so
  "privacy mode - this session only" was not quite true for those two. Fixed,
  and every such write now goes through one shared function so this class of
  gap cannot reopen unnoticed.
- **A grammar that could not be applied no longer produces unconstrained text
  that looks like a normal answer.** Asking a HuggingFace-format model for
  trigger-gated ("lazy") grammar, or sending a grammar it could not compile,
  used to drop the constraint and reply with an ordinary 200 the caller had no
  way to tell from a grammar-conformant one; the only trace was a log line
  inside a background process, which nobody sees. Such a request is now
  refused up front with a clear reason naming what is unsupported and what to
  do instead, before any of the reply is generated. GGUF models, whose sampler
  applies grammars natively, are unaffected and still support lazy grammar.
- **On Windows, the TLS certificate's private keys and a couple of internal
  coordination files were not restricted to your account the way other
  credential files (the API key, session data) already were.** The permission
  call they used only worked on POSIX systems and was a silent no-op on
  Windows, so these files were left as readable as whatever the data
  directory's own location happened to allow - normally fine, but wider than
  intended on a machine another local account can also sign into. They are now
  locked down the same Windows-aware way every other credential file in localm
  is.
- **A restored, rotated, or long-expired TLS certificate authority is no
  longer served past the point where a browser or app would actually trust
  it.** The check that decides whether to keep using the existing server
  certificate compared certificate-authority names, and localm always mints
  every one with the identical name - so it could not tell a genuinely
  different certificate authority (restored from a backup, or replaced after
  its own roughly ten-year life) from the one already on disk, and could keep
  serving a certificate chain that no longer verifies, with nothing in the log
  to explain why connections started failing. The check now verifies the
  actual signature and the certificate authority's own expiry, and a
  certificate authority nearing the end of its life is renewed together with
  the certificate it signs, not on its own.
- **"Scan for ComfyUI models" now finds models in localm's own managed
  ComfyUI.** The managed instance's downloaded models live in a directory next
  to the ComfyUI checkout, not inside a `models` folder under it, so the scan
  button always looked in the wrong place there and reported nothing even with
  real downloaded model files and a running instance serving generations. It
  now looks in the right place for a managed install, and an ordinary external
  ComfyUI folder still scans exactly as before.
- **`localm update` now actually replaces the llama.cpp runtime when a release
  needs it to.** A release that ships a newer llama.cpp build was supposed to
  re-provision your runtime as part of the update, but the step that does this
  always found the old binaries already in place and skipped re-provisioning
  entirely, silently keeping the outdated build while still reporting the
  update as successful.
- **localm no longer installs a llama.cpp runtime it then refuses to load.** A
  fresh `localm setup-llama` fetches the newest upstream build, and from build
  b10373 onward localm rejected what it had just installed with "incompatible
  struct layout (ABI mismatch)", on every backend - CUDA, Vulkan and CPU alike.
  Nothing was actually wrong with those builds: upstream added a new "auto"
  option for how model weights are loaded and made it the default, and localm's
  check did not know that option yet, so it read a perfectly good setting as
  corruption. The check now knows it. Because localm asks upstream for the
  newest build at install time, this could start happening to an installed copy
  of localm with no update on our side, so it is worth updating even if yours
  works today. If you already worked around it by pinning an older build with
  `localm setup-llama --tag <build>`, you can go back to the default.
- **A server that asks for an API key no longer loads any of your data before
  you have entered one.** Opening a link that pointed straight at a page
  (`?view=models`, `?view=settings`, a shared image, or simply returning to the
  tab you were last on) started that page's requests immediately, racing the
  key check instead of waiting for it. On a server that requires a key those
  requests went out before the key prompt appeared. They now wait until the key
  has actually been accepted. Nothing changes once you are signed in, and a
  link you followed still takes you to the right page afterwards.
- **The GUI recovers by itself when the server is restarted underneath it.** On
  a normal local setup the page is trusted using a pass that is issued fresh
  every time the server starts, so a page left open across a restart was still
  holding the old one. Everything kept looking fine while every button quietly
  did nothing, with no message explaining why, and the only way out was knowing
  to reload by hand. The page now notices, refreshes itself once, and carries
  on; if that still does not work it says so plainly instead of pretending to
  work. Restarting from Settings also waits for the new server to actually be
  up before reconnecting, rather than reconnecting to the one that is shutting
  down.
- **Setting up localm's own ComfyUI now says so immediately when the new
  install's venv has no working pip**, instead of failing two steps later with
  an opaque "No module named pip" buried in a PyTorch install transcript. This
  can happen when the base Python's own pip bootstrap is broken or stripped; the
  new message names the real cause and points at `localm doctor`, in both the
  fresh install and the copy-your-existing-ComfyUI paths. Replicating an
  existing ComfyUI whose venv has a package from a private index or a local
  wheel (a vendor-bundled driver package, for example) now also tells you the
  actual next step - clear the ComfyUI folder setting and run setup again for
  the fresh, hardware-matched install - instead of leaving you stuck on a copy
  that can never succeed.
- **An inference error sent back over the API no longer names folders on your
  machine.** When generation fails, the reason is returned to the caller so
  they know what happened. Some of those reasons quote a full path (a model
  that failed to load names the file), which meant your account name and your
  install layout travelled to whoever made the request. Paths are now replaced
  with a short placeholder on all four generation endpoints. The reason itself,
  the file name and any line number are kept, so nothing you need for
  diagnosing a problem is lost.
- **A failed generation on `/v1/completions` now tells you what went wrong.**
  If generation broke part way through (not enough free video memory for the
  prompt, a conversation that outgrew the context window), the non-streaming
  form of this endpoint answered with a bare "Internal server error" and threw
  the actual reason away, while the streaming form of the same endpoint, and
  both forms of `/v1/chat/completions`, reported it in full. All four now
  agree: you get the reason, and `finish_reason` is `"error"` so a program can
  tell a failed generation from a successful one. See the API docs for the
  whole error contract in one table.
- **One caller sending bad grammar trigger patterns no longer holds up
  everyone else.** Checking whether a caller-supplied trigger pattern is safe
  to run can take a couple of seconds when the pattern is a deliberately
  malicious one, and those checks used to happen strictly one at a time, so a
  burst of them queued up behind each other and delayed unrelated requests.
  Checks now run several at a time, and a request that would have to wait too
  long is told the validator is busy (and can simply retry) instead of sitting
  in the queue. Measured on a burst of 18 malicious patterns, the last caller
  waited 6 seconds instead of 36. Nothing unsafe is let through to gain this:
  a pattern that cannot be checked is refused, exactly as before.
- **A vision model's image projector no longer lands on a graphics card you
  told localm not to use.** If you split a model across specific GPUs with
  `gpu_split_indices`, or picked one with `main_gpu_index`, the text model went
  where you asked but the projector always went to the first card regardless,
  taking about a gigabyte of memory on it. It now follows the same card the
  rest of the model is anchored to. On setups where localm cannot work out
  which card that is with certainty (a machine with integrated graphics
  alongside a discrete card, for example) it leaves the projector where it was
  rather than guessing, and says so in the log. Nothing changes if you have not
  configured a specific GPU.
- **`localm update` no longer swaps your installed backend for a different
  one.** The runtime-provisioning step picked whichever backend the current
  hardware recommendation named, rather than the one you actually had
  installed, so a change to that recommendation (an NVIDIA-on-Linux install
  moving from Vulkan to CUDA, for example) could re-provision an already-
  working install onto a backend you never chose. Updates now keep whatever
  backend is already installed, no matter what the recommendation says.
- **A model process that dies from an ordinary error is no longer reported as a
  "native inference fault".** Any death of the model process used to be announced
  that way, including a plain Python error inside it - so a missing image library,
  for one real example, produced "Native inference fault (worker exit 1). See the
  debug log for the native stack trace" when there was no native fault, no such
  trace, and nothing wrong with the model. The wording is now decided by the
  evidence: a genuine crash still says so and points at its trace, while anything
  else says the process exited unexpectedly and sends you to the log for the real
  error. The internal log line that made the same claim is neutral now too, so it
  no longer pre-judges a crash before anyone has looked at it.
- **A working llama runtime is no longer refused over a setting it is allowed
  to change.** Before loading the native library localm checks that its
  internal layout is the one this build expects, and that check could pick the
  wrong layout when a runtime changed any one of a handful of defaults.
  Startup then stopped with "the native llama runtime has an incompatible
  struct layout", named a setting whose value was in fact perfectly normal,
  and told you to re-provision or to set LOCALM_SKIP_ABI_CHECK, all over a
  runtime that was completely fine. The layout is now identified in a way that
  does not fall over when a single default moves. Genuinely mismatched
  runtimes are still refused, and now only ever name the setting that actually
  moved.
- **A crashed model process now says HOW it died, not just a bare number.** The
  error used to read "worker exit -4" and leave you to look that up. On Linux and
  macOS a negative code is the signal that killed it, so that one now reads
  "killed by signal SIGILL" - which distinguishes an illegal instruction from a
  segfault or a deliberate abort, and those point at different causes. On Windows
  the well-known native fault codes are named the same way, including the one
  produced by a clashing GPU library version. Codes that are not faults are left
  exactly as they are rather than dressed up as one. This covers the GGUF,
  HuggingFace and embedding workers alike.
- **When a model or embedding process dies from a native crash, the error now
  tells you what crashed instead of only that something did.** The message has
  always ended with "see the debug log for the native stack trace", but for the
  worst kind of crash there was never any trace to find: a hard native fault
  never returns to Python, so nothing localm could run afterwards was able to
  record where it happened. These processes now arm a crash handler before they
  load anything, so a fault leaves a trace behind, and the error you get names
  the fault and points at the full trace in the debug log. If nothing could be
  captured the message now says so plainly rather than sending you to look for a
  trace that was never written. This covers the GGUF, HuggingFace and embedding
  workers alike.
- **Models are now sized against all your GPUs, not just one of them.** If you
  have more than one card, localm already spread a model across all of them at
  load time, but when working out how much it could offload it only ever looked
  at a single card's free memory. So it offloaded far less than your machine
  could hold and left the rest idle: on one three-card machine a load used
  39 GB and left 21.5 GB unused. Both the number of layers put on the GPU and
  the context size are now worked out from the free memory of every card the
  model will actually be spread over, including cards of different sizes. You
  do not need to configure anything for this. If you have one GPU, nothing
  changes.
- **Opening the web UI on an install that has an API key set now asks for the
  key, instead of signing you in automatically.** When a key was configured, the
  page would still hand a full owner session to whoever asked for it, with no key
  presented, as long as the request came from this machine. That could not be told
  apart from any other local program asking, so it is gone: presenting no key to a
  keyed install is now treated the same as presenting a wrong one. Launching from
  `localm gui` is unchanged and still signs you straight in, browsers that are
  already signed in stay signed in, including across rolling the key, and typing
  the key into the page works as before. The one visible change is that opening
  the address by hand now shows the key prompt.
- **Two ComfyUI updates can no longer run at once.** Updating localm's own
  ComfyUI rewrites its files in place, and until now nothing stopped a second
  update starting while the first was still going, which could leave the
  install in a broken half-updated state. Only one can run at a time now,
  whether started from the command line or the Settings button; a second one
  stops immediately and says which update is already in progress rather than
  waiting or interfering. If an update is ever interrupted, the next one tidies
  up after it automatically.
- **A chat or completion request that did not name a model reported back the
  literal word "localm" instead of the model that actually answered.**
  `/v1/chat/completions` and `/v1/completions` already resolved an unnamed
  request to whichever model was in use, but the reply's `model` field still
  said "localm" rather than that model's real name, because "localm" doubled
  as both the field's default and its own separate documented meaning of "no
  preference". A request that explicitly asks for "localm" still gets
  "localm" back; only the truly unnamed case changes.
- **An embeddings request that did not name a model was refused outright,
  even when your embedding model was already set up.** `/v1/embeddings`
  required an explicit `model` unconditionally, with no fallback at all - now
  an unnamed request uses your configured embedding model automatically, and
  only refuses when nothing is loaded to serve it with. Deliberately not
  simply "whatever model is active" like chat and completions: embeddings
  from different models cannot be mixed the way chat replies can, so an
  unnamed request always prefers your one configured embedder rather than
  whichever chat model you happen to be using at the time. A request that
  explicitly asks for "localm" still gets "localm" back unchanged.
- **Copying a model file into the models folder while the copy was still
  running could get it registered and loaded before the copy finished.** The
  existing checks already refused a non-GGUF file or an empty placeholder, but
  a real GGUF header appearing early in an in-progress copy was enough to pass
  them, well before the rest of the file existed on disk. localm now also
  waits for a new file's contents to stop changing for a few seconds before
  registering it, so an in-progress copy is picked up once it actually
  finishes, not while it is still arriving.
- **A browser that could not reach the server on startup no longer assumes it
  is talking to the same localm it was last connected to.** The GUI checks, on
  every load, that the server behind the address really is the one whose
  conversations your browser has cached, because a second install reusing the
  same address would otherwise show you the first one's chats. That check was
  skipped entirely when the request asking who the server is did not come back
  (it was down, still starting, or answered with an error), and the answer left
  standing was the optimistic one used before the first check completes, so
  cached conversations could be uploaded into an install that had never been
  identified. A failed check is now treated as what it is - no answer - so your
  own cached conversations still appear and nothing is deleted, but nothing is
  uploaded until a later check confirms the server. The console says so once
  while the server is unreachable, rather than repeating it every 30 seconds.
- **Shared items the server could not delete are no longer reported as
  cleared.** When you share images into localm from your phone, the app copies
  them into the chat and then tells the server to empty its share inbox. Any
  entry the server failed to delete (in use, or not permitted) was left out of
  the reply's count and never shown, so a share store that still held your
  files looked exactly like an emptied one. The GUI now tells you how many
  items could not be cleared.
- **A malformed request with deeply nested JSON now gets the clear rejection it
  should, instead of a generic server error.** Building the validation message
  walked the offending value without any limit on how deep it went, so a small
  body nested a few hundred levels deep exhausted the recursion limit inside the
  error handler itself. The request came back as an unexplained server error, and
  because each one occupied the server for a moment, a handful of them together
  slowed unrelated requests noticeably. Nesting past a sensible depth is now
  summarised rather than walked, so the response is the ordinary validation error
  and the cost is gone.
- **Indexing or attaching a hostile archive can no longer tie up the machine.**
  Archive extraction limited how much any single member could decompress to, and
  how many members it would look at, but not how much it decompressed in total.
  An archive of many small, highly compressible files could stay well under the
  upload limit while still expanding to many gigabytes and occupying a CPU core
  for over a minute before being correctly rejected as containing no text. There
  is now an overall budget for how much an archive may decompress to, and every
  member counts against it whether or not any text came out. Ordinary archives
  are unaffected; one that hits the budget stops early and says so in the
  extracted text, as it already did for the other limits.
- **Revoking an admin-scoped device key now reliably signs out its browser
  too.** Browser sessions are re-checked against the key that created them on
  every request, so revoking or expiring that key ends the session. The owner
  key is deliberately exempt, because rolling it must not sign you out of your
  own browser, but that exemption was granted to any session holding admin
  rights, including ones created by admin-scoped keys made with
  `localm keys create`. Those keys are meant to be revocable, so their browser
  sessions could outlive them. The exemption now follows the owner key itself.
  Rolling the owner key still keeps you signed in, including on sessions created
  by an older version.
- **A `jobs`-scoped key can no longer point a scheduled coder job at any folder
  on the machine.** Scheduled coder jobs took their working directory from
  whoever created them, checked only for being a local path, so a key you handed
  out with just the `jobs` scope could read and edit files anywhere the server
  could reach. Coder sessions over the API already confined a scoped key to the
  project folder; scheduled jobs now match. The owner key and `coder:full` keys
  still choose freely, a scoped key's job is placed in the project folder (shown
  in the API response and in the job's own output when an existing job is moved),
  and the check runs again at each run, so narrowing or revoking a key takes
  effect on its next tick.
- **Applying an update from two places at once could no longer be safely rolled
  back.** `localm update` had no protection against running twice concurrently
  (two browser tabs both pressing "Update now", or the CLI racing a live
  server), and both used the same fixed scratch and backup locations. The
  second call's "pre-update" backup could end up silently holding the first
  call's already-installed new build rather than the true original, leaving
  nothing to restore from if a rollback was ever needed. A second concurrent
  apply is now refused immediately with a clear message, and each apply uses
  its own private working directory until it succeeds.
- **Restarting the server from Settings no longer opens a second browser tab.**
  The restart re-launches the server process the same way a fresh `localm gui`
  launch does, which used to auto-open a browser tab regardless. The tab you
  restarted from already reconnects in place once the server is back up, so a
  restart no longer opens anything new.
- **`localm key clear` and `localm key recover` no longer report success when
  they could not sign browser sessions out.** If the session store could not be
  written (a locked or read-only file), both commands printed their normal
  success message and `POST /api/auth/key/clear` returned `"cleared": true`,
  while every signed-in browser stayed signed in. This mattered most for
  `key recover`, whose whole purpose is locking a compromised owner out and
  which always sets a new key, so a surviving session kept working against it.
  All three now say plainly that sessions were not signed out and name the file
  to fix, and the clear endpoint reports `"cleared": false` with the reason in
  `warnings`. `POST /api/session/logout` gained the same `warnings` list for the
  case where the browser cookie is cleared but the server-side session survives.
  Revoking a scoped key now also logs a warning if its sessions could not be
  dropped, instead of only a debug line.
- **The update check now obeys network access policy, with an opt-out.**
  Setting network access to Off in Settings blocked every model-initiated
  request but not the periodic update check, which kept quietly phoning the
  update server regardless. It now goes through the same policy as everything
  else - Off stops it too, unless you turn on "Check for updates even when
  network access is off" for that one channel. Pressing "Check for updates"
  while blocked now shows a short reason instead of either a silent failure or
  a false "you are up to date".
- **Scheduled jobs no longer lose their shell step when you roll the API key.**
  Rolling the key deliberately keeps you signed in to the web UI. But a
  scheduled coder job created from that still-signed-in browser afterwards was
  no longer recognised as yours, so every later run quietly dropped its shell
  access and there was no way to tell from the job itself. Signing in now
  records that it was the owner key that signed in, so your own automation keeps
  working across a key roll. Keys minted with `localm keys create` are
  unaffected and are still re-checked on every run, so revoking one still
  removes shell access from the jobs it created.
- **The GUI shell now applies the same same-origin check when a key is
  configured that it already applied when one is not.** On a loopback bind,
  the page that signs a keyed owner in was answering requests from any origin,
  not just its own; it now serves the plain shell to anything cross-origin, as
  the keyless path already did. Signing in locally is unchanged: a first visit
  still signs you in, a reload does not sign you in again, and a browser that
  is already signed in stays signed in after you roll the owner key. The
  one-time launcher handoff is unaffected and still works on any bind.
- **The coder no longer accepts "I can't do that" as a finished answer.** A
  model that replied in prose without calling any tool was taken at its word,
  so a request to create files could end with an explanation and nothing
  written. The coder now asks again with the tool-call format, and if that
  does not work it constrains the model's output so that a valid tool call is
  the only thing it can produce. Only if enforcement genuinely fails does it
  say so plainly, instead of reporting the task as done. The prompt also no
  longer describes the assistant as "running fully offline", which some models
  read as being unable to touch your files at all.
- The coder's stuck-loop detector now recognises a model that keeps rewording
  the same non-answer, not only one that repeats it character for character.
- **The `uninstall_plugin` MCP tool no longer reports success when a plugin's
  files could not actually be removed.** If the installed directory was locked,
  held by antivirus, or blocked by a permission denial, the plugin was still
  disabled and unloaded but its files stayed on disk; the tool nonetheless
  replied "successfully uninstalled". It now reports an error explaining the
  plugin is not fully uninstalled in that case, matching what the CLI already
  told you.
- **`localm rename` could lead to a loaded model's file being deleted.** The
  rename happened only in the CLI's own process, so a running server kept
  serving the model under its old name, and removing it from the Models page
  then deleted the GGUF out from under it. A model file that is loaded right
  now can no longer be deleted, whatever name it is registered or loaded
  under, and `localm rename` now hands the rename to the running server so a
  loaded model keeps serving under its new name.
- **A grammar with a very large repeat count now gets a clean error instead of
  possibly reaching the native parser unrejected.** The upfront structural
  check on a `grammar` field (chat completions and the coder) allowed a repeat
  count well above what the native GBNF parser actually accepts, so a request
  in that range used to clear the check and could fail later instead of
  getting an immediate, clear error.
- **`localm run` and `localcoder` no longer fail to reach your own server
  once you set an API key.** Attaching to an already-running local server
  presented the wrong credential once auth was turned on, so setting a key
  - the recommended way to protect an install from other local processes -
  broke the everyday `localm run <model>` / `localcoder` workflow with a
  bare connection error. Fixed for both, including `localcoder --no-server`.
- **When setup could not find a working llama.cpp backend at all, the error
  named the wrong cause.** After your chosen backend failed to load, setup
  falls back through the universal Vulkan and CPU builds; if those failed too,
  the final message always blamed whatever went wrong with your *original*
  pick, even though the fallback builds had failed for their own, different
  reasons. It now names every backend it tried and why each one failed, your
  original pick included, both as it happens and in the final summary (and
  the saved bug report, which only ever captured that final summary).
- **No llama.cpp backend could load on a freshly provisioned install.**
  Upstream inserted a new field into the struct localm passes to the native
  runtime, which shifted several fields' positions in the exact builds
  `localm setup-llama` was fetching, while the bundled AMD build still used
  the older layout. localm's own safety check correctly noticed the mismatch
  and refused to load rather than risk memory corruption - which is exactly
  what it is supposed to do - but until now it only recognized the older
  layout, so every freshly downloaded cuda, vulkan and cpu build was refused.
  Both layouts are recognized now, detected per install, so this affects
  every platform equally going forward as upstream keeps evolving.
- **Choosing CUDA on Linux could silently fetch a build with no kernels for
  your GPU.** Detecting which CUDA build line a card actually needs (the
  newer line for very recent architectures, the broadly-compatible one for
  everything else) only ran on Windows; Linux always fetched the older line
  regardless of what was actually installed. The runtime still loaded - it is
  a valid build, just not for that card - so `localm doctor` reported no
  usable GPU with no indication why. Linux now detects the same way Windows
  already did, so the build fetched actually matches the card.
- **The HuggingFace/PyTorch backend, and localm's own managed ComfyUI, always
  installed the same PyTorch build regardless of which NVIDIA GPU was
  present.** On the newest NVIDIA architectures that build has no compute
  kernels for the card at all, so PyTorch loaded but silently ran on the CPU
  - correctly detected GPUs, no error, just no acceleration, with a warning
  easy to miss in a long setup log. Both installers now pick a PyTorch build
  that matches the card.
- **Setup now recommends CUDA for NVIDIA on Linux, not Vulkan.** The
  self-contained Linux CUDA path was new and unconfirmed on real NVIDIA Linux
  hardware, so Vulkan stayed the safer default while it proved itself.
  Real-hardware testing has since confirmed it works and outperforms Vulkan,
  so a bare `setup.sh` / `localm setup-llama` on an NVIDIA Linux box now
  installs CUDA by default, matching what Windows has always done. The
  existing load-test-then-fallback safety net still applies if CUDA cannot
  load on your machine; Vulkan remains available with `--backend vulkan`.
- **Setup now recommends the ROCm/HIP build for AMD cards it could not
  self-contain a build for, when it detects you already have the toolkit
  installed.** Previously any AMD card outside the bundled RX 6000 build
  (RX 7000/9000, and every AMD card on Linux) always recommended Vulkan, even
  on a machine with a working ROCm/HIP install already present. Setup now
  detects that toolkit the same way it already detects the card itself, and
  recommends the faster vendor build when it finds one. Vulkan remains the
  recommendation when no toolkit is present, and the RX 6000 build is
  unaffected - it needs no toolkit at all, so it stays the default there.
- **Filing a bug report from the app could freeze the whole server for as
  long as it took to save.** Saving a report reads the current log, digests
  it, and writes the report file, and all of that ran directly on the same
  thread that serves every other request - so filing one, exactly when
  something was already going wrong, stalled everyone else for the duration.
  Saving now happens off to the side, and two reports filed close together
  can no longer collide and corrupt each other's file.
- **A GPU-compatibility warning in the debug log could be cut off before the
  useful part.** When your GPU's PyTorch build reports it as unsupported, the
  message logged for it was trimmed to 200 characters - shorter than the
  install path that comes first in it, so the list of architectures your
  PyTorch actually supports, the part that says what to do about it, never
  made it in. The limit is now much higher, and a message rare enough to
  still exceed it is now marked as cut short instead of stopping without a
  word.
- **`setup.sh` could misreport an NVIDIA machine as ROCm.** Its pre-venv
  hardware guess checked for leftover ROCm tooling (`rocminfo`/`rocm-smi`/a
  bare `/opt/rocm`) before ever checking `nvidia-smi`, so a box with real
  NVIDIA hardware but some ROCm tooling still on PATH (a shared ML rig, a
  base image bundling both vendor stacks) was detected as ROCm. `nvidia-smi`
  is now checked first, matching the same vendor priority the authoritative
  hardware recommendation already used. That guess also no longer prints
  itself as a standalone "Detected acceleration" verdict, since it could
  disagree with the real recommendation sourced from `python -m
  localm.hwdetect` a few lines later; it now only gates the pre-venv "use GPU
  acceleration?" prompt.
- **The network policy's domain deny list could be silently skipped by a
  config-read failure, in one narrow configuration.** Checking whether a
  request is allowed read the config twice, once to resolve the mode and
  once for the `net_deny`/`net_allow` lists; each read handled a failure on
  its own. With `LOCALM_NET_MODE` set in the environment, the mode check
  short-circuited before the config was ever touched, so a config that
  failed to read at the wrong moment fell through with an empty deny list
  instead of being refused. The private-address (SSRF) guard was never
  affected. The config is now read once per check, and a read failure now
  refuses the request outright, regardless of the environment variable.
- **The ComfyUI connection no longer follows a redirect.** The guard that
  keeps a configured ComfyUI address off link-local / cloud-metadata targets
  only ever checked the address you configured, not where a response from
  that server could redirect a request afterward - so a hostile or
  compromised ComfyUI (which SECURITY.md already documents as possibly being
  another machine, over plain http) could answer any of localm's requests to
  it with a redirect straight past that guard. The connection now refuses any
  redirect outright; ComfyUI's API never legitimately sends one.
- **The GUI's HTML sanitizer is updated to a build that fixes a published
  cross-site-scripting bypass.** localm ships DOMPurify with the GUI to clean
  everything a model writes before the browser renders it as markdown. The
  copy in place was 3.2.6, which falls inside the range affected by
  CVE-2026-0540: certain text could survive sanitizing with an HTML close tag
  intact and then run as script if the result was placed inside one of five
  particular elements. It is now 3.4.13. Whether localm's own rendering could
  be made to reach that case was not established in either direction, so the
  build was replaced on the version match alone. The sanitizer is now also
  covered by a test that reads the file localm actually ships and fails if it
  ever falls below the fixed version. Nothing watched it before: the library
  is vendored directly and appears in no lockfile, so no dependency scanner
  could ever have reported it.
- **Stopping the coder once no longer makes every later exit pause to think.**
  Stopping a task (or answering no to "Keep going?") was recorded as a failed
  session and never unrecorded, so from then on quitting `localm coder` waited
  on the model while it wrote a lesson about the session - even when everything
  after the stop went fine and nothing was changed. Stopping is you asking to
  leave, not a failure, so it no longer counts as one. A task that genuinely
  failed still records its lesson, including when you stop the next one.
- **The TLS documentation no longer promises that a device you trusted stays
  trusted forever.** `docs/tls.md` stated that the local certificate authority
  is always kept when the server certificate is regenerated. That is true when
  your addresses change or the certificate nears expiry, but localm does mint a
  fresh authority if the old one reaches its own expiry or if its files go
  missing or unreadable, and every device then has to trust the new one. The
  page now says which case is which, and documents deleting `<LOCALM_HOME>/tls/`
  to rebuild both from scratch, which was not written down anywhere.
- **The GUI's math renderer is updated off a published advisory.** localm ships
  KaTeX with the GUI to render LaTeX in replies. The copy in place was 0.16.11,
  which falls inside the range affected by CVE-2025-23207: `\htmlData` did not
  validate attribute names, so crafted math could emit an attribute it should
  not have. localm was not exposed to it in practice, because the one place it
  renders math already turns off KaTeX's trusted-input mode, which is the
  workaround the advisory itself names; the build was replaced because running a
  known-affected version and relying on a setting to stay put is not a position
  worth keeping. It is now 0.18.4, which also carries an upstream fix for
  prototype pollution in KaTeX's own options handling. Math rendering is
  unchanged, and the three KaTeX files localm ships are now covered by a test
  that reads them, fails below the fixed version, and fails if they ever come
  from different releases.
- **A bug report now says when it could not collect your debug log, instead of
  leaving the section out as though there was nothing to report.** A report
  attaches the log from the run it is about. If that file could not be read, or
  no log for that run could be found, the report simply had no log section, and
  that looks exactly like a run that logged nothing worth reporting. The report
  now says which of those happened and why (for example "permission denied"),
  so a missing log is visible rather than being mistaken for a clean one. The
  reason is the operating system's own short message: it never includes the
  file's location or your account name, and a log that was read normally and
  had nothing notable in it still adds nothing to the report.
- **An attached file was silently cut to about its first 24,000 characters.**
  Attaching a document to a chat or coder message sent only the opening of it,
  so a question about anything past that point was answered confidently from
  text the model had never seen. The whole file is now attached, for every file
  attached. A caller that genuinely wants an excerpt can still ask for one.
- **Text-to-speech was completely broken and now works.** The security policy
  the GUI started enforcing this cycle blocked speech at two independent
  points: the script that runs the neural voice was refused because its source
  was allowed for data but not for code, and compiling the voice model needed a
  permission that was never granted, so no backend could start even once
  downloaded. Both are now allowed explicitly, and the audio the browser
  produces is used unmodified rather than being re-assembled, which was
  corrupting longer replies while short ones happened to sound fine.
- **The sidebar's model dropdown could show a model that was not loaded.** With
  nothing active, no entry was marked as selected, so the browser fell back to
  displaying whichever model happened to be first in the list as though it were
  in use, while the line underneath correctly read "no model". The dropdown now
  shows an explicit placeholder whenever nothing is loaded, and the sidebar has
  its own Unload button for the model that is.
- **The Settings page could lose its whole Knowledge section on a cold load,
  making RAG indexing unreachable from the GUI.** Settings decided which fields
  needed a Browse button by guessing from their names, and six fields that are
  not paths matched. A path field stays hidden until the server reports what it
  supports, so on a cold render each of those vanished, taking the section
  around them with it, and the two-column layout appeared to reorder between
  reloads. Browse buttons are now attached only to fields actually declared as
  paths, and the page waits for a real answer from the server rather than for a
  timeout.
- **Settings > Show changelog listed changes that were not in the build you
  were running.** The in-progress section was served along with the released
  history, so it advertised work that had not shipped, including security fixes
  described before they were available. Only released sections are served now.
  The file itself is unchanged, and published prereleases still appear.
- **The web UI could stutter because reading system stats re-probed the GPU
  every time.** Each stats poll ran the full out-of-process VRAM probe, which
  takes seconds on a cold call, so a page that polls regularly paid it over and
  over. The reading is now refreshed in the background and shared between
  callers. A probe that fails is never cached as a confirmed empty reading.
- **localm no longer overwrites a file it could not read.** Your saved prompt
  library, the config file and the model registry were each treated as empty
  when they existed but could not be read or parsed. Every save is a
  read-then-write, so the next save replaced the whole file with just the entry
  being added, losing everything else. Each of those now refuses the write and
  says why, so an unreadable file stays intact instead of being replaced.
- **A `.localcoder/config.toml` that could not be parsed was treated as absent,
  silently dropping your safety settings.** Two of the keys lost that way are
  not preferences: the setting that makes the coder ask before running a shell
  command, and the one that limits where it may write. An unreadable project
  config is now refused with the parse error, rather than quietly falling back
  to defaults that permit more than you asked for.
- **A model that failed to load left its worker process behind.** Only a load
  that timed out was cleaned up; a load that was cancelled or that raised an
  error tore down nothing, so a model that keeps failing to load accumulated
  one stranded process per attempt, each holding memory. The worker is now
  reaped on every failure path.
- **Loading a vision model dumped raw native output into the console and hid
  the reason when it failed.** Unlike every other native call, the image half
  of a vision model was loaded without capturing its output, so the loader's
  tensor dump reached your terminal; and when it could not be opened, the
  actual reason was dropped and only a generic message survived. The output is
  now captured like everywhere else, and a genuine failure reports what the
  library said instead of silently falling back to text-only.
- **The coder could not force a tool call on a model served through the
  HuggingFace backend.** The check for grammar support only recognised GGUF
  models, so an HF-backed model never got the constrained output that makes it
  produce a valid tool call, even where the backend supports it. It is now
  asked what it supports rather than assumed.
- **The coder's approval preview could not tell a new file from one it could
  not read.** Both looked like empty content, so a write over a file that was
  locked or unreadable was previewed as though it were creating something from
  nothing, and you approved a replacement without being shown what it replaced.
  The preview now says when the existing content could not be read.
- **The update health check went through your HTTP proxy.** After applying an
  update, localm checks that the instance it just restarted is answering. That
  check targets this machine's own port, so a proxy is never right for it, and
  honouring one made the check fail on machines where a proxy is configured.
  It now connects directly.
- **Setting an API key got slower every time you set one.** Saving a key
  re-verified every historical key record it keeps, and that verification is
  deliberately expensive, so the cost grew with each rotation until the call
  took long enough to look like a hang. It now does only the work the save
  actually needs.
- **localm could misidentify what kind of device a graphics backend reported.**
  One of the constants describing device types was fixed at a value that
  upstream llama.cpp has since moved, because a new type was inserted ahead of
  it, so a device could be read as the wrong kind. The value is no longer
  declared where it could be wrong.
- **The file and folder picker could not browse into a dot-directory.** Folders
  that other tools use for models, such as `~/.ollama`, `~/.lmstudio` and
  `~/.cache/huggingface`, were skipped from every listing, so they could not be
  reached by clicking. A "Hidden" toggle now shows them, off by default so an
  ordinary pick is not buried in `.cache` and `.config`, and it switches itself
  on when you land inside one. Typing or pasting a dot-path always worked and
  is unchanged.
- **The setting that keeps graphics memory host-visible did the opposite of
  what it said.** The opt-out is there for machines that share memory between
  the processor and the graphics chip, and its own documentation promised that
  setting it to off would be respected. The underlying library reacts to the
  setting being present at all rather than to its value, so switching it off
  disabled exactly what it was meant to keep. Off now means off.
- **Unloading a model that was still generating reported success without
  freeing anything.** Pressing Unload on the Models page while a request was
  running against that model said "Unloaded" even though the model stayed
  resident and no video memory was released. Unload all got it wrong from the
  other side: it counted only what it had actually freed, so a run where
  everything was busy read as "Nothing was loaded". Both now say plainly that
  the model is still generating and to try again once it finishes, and Unload
  all names how many it had to skip.
- **Video and music would not play, and sending a generated image to chat or
  copying it failed.** The security policy the GUI started enforcing this cycle
  refused every internally-generated media file, so players sat dead and the two
  image buttons reported "Failed to fetch". Playback was the worse half because
  it was silent: the browser reports a refused source as an event rather than an
  error the page can catch, so nothing was shown at all and the button still
  said "hide". Both are allowed again, and a media file that fails to load now
  says so whatever the reason, instead of leaving you with a dead player.
- **Answering "no" to using your GPU during setup on Linux or macOS no longer
  recommends a GPU build anyway.** The question promises CPU only, but the next
  screen still suggested a GPU backend and made it the default, so pressing
  Enter downloaded and tested a runtime you had just declined. Every backend is
  still listed if you change your mind.
- **A command could stop with "localm hit an unexpected error" when a knowledge
  collection was briefly busy.** On Windows, a collection's lock file can be
  unreadable for a moment while another localm process releases it. That was
  treated as a failure rather than as something to wait for. It now waits, up to
  the same limit as any other busy collection, and says the collection is in use
  if it never clears.
- **Re-embedding a knowledge collection failed with an embedding model that is
  built on a chat model.** Every re-embed stopped with an error. Localm packs
  several chunks into one call and sized that batch against the whole context
  window, while the model was actually given a small slice of it per chunk, so
  any ordinary chunk was too big. Models of this kind (the multilingual and
  Qwen-based embedders) now work; the bundled default was never affected, which
  is why this went unnoticed. Embedding also uses about a quarter of the video
  memory it did before.
- **Updating localm's own ComfyUI could stop with "unable to read tree".** The
  update did not ask for the exact version it was moving to, so the download
  could arrive without it. It now requests that version specifically, and checks
  the download is complete before touching your install rather than discovering
  it half way through.
- **A simple request could be answered with something unrelated you talked about
  days ago.** Asking localm to greet a friend could produce a reply about a
  different person entirely. Three things combined to cause it: stored memories
  were never given the data semantic search needs, so recall fell back to
  offering whatever it considered most important rather than what was relevant;
  a request about somebody else ("greet my friend...") was read as a question
  about you, which opened that fallback; and a memory that merely mentioned some
  other person was close enough to count as related. All three are fixed, so a
  request now recalls what it is actually about, or nothing.
- **Memories saved before you installed the search model are now included.**
  `localm setup-embeddings` said memory would retrieve semantically, but nothing
  ever went back and processed what you had already saved, so it stayed keyword
  only, sometimes indefinitely. Setting up the model now processes those saved
  memories and tells you how many, including any it could not.
- **localm no longer searches the web for things it can just do.** Asking it to
  greet someone could produce a web search for greeting messages and a list of
  websites instead of a greeting. It is now told plainly not to search for
  writing, greeting, rephrasing, translating or summarising something already in
  the conversation. Searching for genuinely current information is unchanged.

### Security
- **The GUI's Content-Security-Policy now actually blocks, instead of only
  reporting.** It shipped in report-only mode, which never stops anything, so
  the HTML sanitiser was the only thing standing between a malicious model reply
  and script execution in the app - if it were ever bypassed, nothing was behind
  it. The policy now enforces: scripts must come from localm itself and carry a
  one-time-per-page marker, so injected script cannot run even if it reaches the
  page. Plugins, embedded objects and framing the app from another site are
  refused outright. Rendering an artifact still works exactly as before, and it
  remains sealed off from the rest of the app and from the network.
- **On Linux and macOS, the files holding your browser-session records and a
  running instance's attach token were briefly readable by other accounts on
  the machine while being written.** Each was created with the system's default
  permissions, filled with its contents, and only then restricted to your
  account, so the whole payload sat unprotected for the length of the write.
  They are now created already restricted, which closes the window rather than
  shortening it. Windows was not affected here: on that platform the
  restriction was always applied before the file was moved into place.
- **A bug report could leak a credential carried in a configured URL's query
  string.** The search, ComfyUI and code-review endpoint URLs you configure are
  echoed into a report to help diagnose setup problems; a credential written as
  `user:pass@` in one of those was already redacted, but one passed as
  `?api_key=...`/`?token=...` (or pasted in from an `X-Api-Key:` header) was
  not. Both shapes are now redacted by name, keeping the rest of the address
  visible so the report stays useful.
- **A fenced code block in a model reply could freeze the chat tab.** localm
  vendors highlight.js to color code blocks; upstream fixed two denial-of-
  service bugs in its C/C++ and XML tokenizers (a runaway regex triggered by
  ordinary-looking source) with no CVE and no security advisory of any kind,
  so nothing could have flagged the vulnerable copy that shipped here.
  highlight.js is now 11.12.0, and the fix is covered by a test so a future
  bump cannot silently lose it.
- **A pattern using punctuation could slip past the check that rejects
  regular expressions capable of hanging the tokenizer.** The check derives
  test inputs from the pattern itself, and the step that built them discarded
  every non-alphanumeric character, so a pattern keyed on punctuation was
  probed with inputs that could never trigger it and was let through. The
  derived inputs now keep punctuation.
- **Privacy mode left three kinds of trace behind when you used the coder.**
  Turning privacy mode on for the coder alone did not stop the debug log, if
  you had it switched on, from recording prompt and reply content: the check
  that gates this looked only at the chat and server settings, while the coder
  produces its replies through the same shared path. The coder's project
  index, which records your project's file paths and the names it pulled out
  of them, was written into localm's data directory whatever the mode. And
  checking a Python file for syntax errors before writing it compiled the
  content through a temporary copy, leaving a compiled form of your code in
  the machine's shared temporary folder that nothing ever removed, in any
  mode. All three are fixed: the log content is suppressed, the project index
  stays in memory, and the syntax check no longer writes anything to disk.

## [0.1.5rc2] - 2026-08-08

### Added
- **Self-contained CUDA support on Linux, as an explicit opt-in choice.**
  `localm setup-llama --backend cuda` now works on Linux the same way it
  already does on Windows: it fetches a compiled build plus the CUDA
  runtime libraries (cudart, cuBLAS) with no CUDA Toolkit install needed.
  Upstream llama.cpp does not publish a Linux CUDA binary itself, so this
  fetches one from an actively-maintained third-party builder instead -
  Vulkan remains the recommended backend on Linux until real hardware
  confirms the CUDA path, and the existing load-test-then-fallback safety
  net still applies if the CUDA build cannot load on your machine.

## [0.1.4] - 2026-08-06

### Added
- **Image generation now sends a "Rendering… (Ns elapsed)" heartbeat every
  15 seconds while ComfyUI is working**, matching the existing music and video
  generation feed. Previously an image job stayed completely silent on the job
  stream from submission until it finished - up to 10 minutes by default - so
  a slow render and a wedged one looked identical on the Images page; the
  local terminal spinner already showed elapsed time, but only when running
  from the CLI, never in the GUI.
- **The Models page now shows a registered model's real architecture and
  MoE-ness, not just a name guess.** The same header read the HuggingFace
  search page already uses to badge a remote repo is now captured once, at
  registration or pull time, and shown on your local list too - a confirmed
  architecture family, and an "MoE" badge when the file's own header says it
  has experts. A model registered before this existed shows neither badge
  until it is next re-registered or picked up by the models-folder sync
  (which backfills a few at a time on an ordinary launch, never all at once,
  so a large library never stalls startup) - shown as genuinely unknown in
  the meantime, never as a false "not MoE" claim about a model nobody has
  actually checked.
- **The Models page can now sort by column and rename a model.** Table headers for
  Name, Role, Source, Size and a new Modified column are clickable to sort
  ascending or descending, and the choice is remembered across reloads. A rename
  control (`localm rename OLD NEW`, matching API and button) moves a model to a
  new name outright, unlike the existing alias which keeps the old name working
  too - other aliases pointing at the same file are left alone. Renaming
  best-effort updates every other place that stores the plain name (pinned
  models, the embedding and coder-reviewer model settings, scheduled jobs, RAG
  collection metadata) and tells you what it could and could not update; a
  currently loaded model keeps running under its new name without a reload. One
  thing it cannot reach: a per-project `.localcoder/config.toml` that pins this
  model by name has to be updated by hand.
- **Loading a Mixture-of-Experts model with "MoE expert layers on CPU" turned on
  now tells you where the weights actually landed**, instead of leaving you to
  take it on faith. The load prints a line like "moe placement: 200.00 MiB
  system RAM / 800.00 MiB VRAM across 2 backend buffer(s)"; if nothing could be
  read back it says "not reported" rather than implying zero. Ordinary loads are
  silent as before - this only appears when the setting is actually on.
- **A "Warm up now" button in Settings loads the embedding model on demand,**
  with a live status line walking through resolving, downloading if needed,
  freeing VRAM, and loading. Previously the embedder loaded silently and
  invisibly on whatever request happened to need it first - even after running
  `localm setup-embeddings`, which only fetches the file and never actually
  loads it - so the first real embeddings or RAG call could stall for minutes
  with no explanation. Already-loaded shows "Already warm" instead of reloading.
  This is opt-in: nothing changes if you never click it.
- **The coder can now search your indexed Knowledge collections.** Two new
  tools, one to list what is available and one to search a named collection for
  matching excerpts, so the coder is no longer limited to files in the project
  directory. Search asks for confirmation by default, since collection content
  isn't scoped to the project the way file reads are; retrieved text goes
  through the same sanitiser used for anything else untrusted before it can
  reach the model. A restricted (shared, non-owner) coder session cannot use
  either tool. Search is lexical only for now, not the hybrid/embedding search
  the Knowledge page itself can do.
- **The Image generation page can now select a LoRA and set its strength.** A
  dropdown lists the LoRA files installed in your ComfyUI instance, with
  separate strength fields for the model and for CLIP (defaulting to 1.0 and 0.5
  when left blank). The name is validated before it is ever handed to ComfyUI's
  workflow graph, since ComfyUI resolves it against its own models folder rather
  than a path localm controls - the same check runs whether the request comes
  through the API or through the coder's own image-generation tool.
- **The Models page now shows a model switch actually happening.** The "use"
  button shows "loading…" for the real duration of the switch (which can take
  tens of seconds), instead of giving no feedback beyond being disabled. The
  sidebar already had an accurate status line for this, but it lives in a
  drawer that is closed by default on mobile - the button now carries its own
  cue everywhere.
- **HuggingFace model search now shows what a model actually is.** Results
  display the model's architecture family (e.g. `qwen3moe`, `mixtral`), an
  approximate parameter count, and a Mixture-of-Experts badge when one applies.
  The MoE badge is labelled `MoE` when the model's own header confirms it, or
  `MoE?` with a tooltip when it is only a guess from the repo name (some real
  MoE models, notably older Mixtral conversions, do not say so in their own
  header) - never presented as equally certain. Display only; it never changes
  which results a search returns.
- **Run a Mixture-of-Experts model in a fraction of the VRAM.** A new
  "MoE expert layers on CPU" setting (`n_cpu_moe`, off by default) keeps the
  expert weights of the first N layers in system RAM while the rest of the model
  still runs on the GPU. Measured on a 7B MoE: the GPU footprint dropped from
  3961 MiB to 241 MiB with all 16 layers set. It is a FOOTPRINT dial rather than
  a speed-up - at the same VRAM it runs at about the same speed - so it is worth
  reaching for when something else needs the card, or when a model would
  otherwise not fit at all. It has no effect on a normal (dense) model, and says
  so instead of silently doing nothing.
- **Background work now runs on a server started without the web interface.**
  Indexing a Knowledge collection, uploading documents to one, installing the
  embedding model, and generating images, music or video all used to need
  `localm gui`: on a bare `localm serve` the first two quietly ran to completion
  inside the request instead of in the background, and the rest simply refused
  with a message blaming the missing web interface. They all work now, the same
  way they do in the browser. See the Changed section for the response-shape
  break this creates for anyone scripting the first two.
- **Setting up localm's own ComfyUI by copying your existing install now reports
  each custom node as it copies it**, instead of going silent for the whole
  step - previously the only output during that step was a start line and a
  final count, so copying a large custom-nodes folder could sit with nothing on
  screen for as long as the copy took.

### Security
- **Clearing your API key could report success while that key still granted
  access.** `localm key clear` and the web interface's clear button both
  announced a completed clear unconditionally. When the key file could not
  actually be deleted (a virus scanner or search indexer holding it open, a
  permissions or profile change), the only notice went to the debug log, which
  you never see unless you run with `--debug`. So you were shown a tick and told
  the server was in open mode while the old key kept working. Both now report
  exactly what could not be removed and stop claiming success; the
  `/api/auth/key/clear` response gained a `warnings` list, and its `cleared`
  field is now false whenever any credential survived. Browser sessions are
  still signed out either way, so a partial clear is never worse than before.
- **Another website open in your browser could obtain the local management
  credential from a keyless install.** On an install running without an API
  key, the page that boots the web interface embeds the token that authorizes
  management actions, and it decided whether to include that token from the
  address the server had been bound to alone, never from who was asking. A
  different site you had open could therefore read the token out of that page
  and act against your instance. The token is now included only for a
  same-origin request (an ordinary navigation or reload of the interface
  itself) and withheld from every cross-origin one, whatever `cors_origins` is
  set to - that setting governs who may read a response, which is a different
  question from whether the credential was safe to put in one. Using the
  interface normally on your own machine is unchanged. An install that has an
  API key set was never affected.
- **A model or vision projector pulled from an untrusted repo could resolve to
  something other than the plain file it appeared to be.** The filename
  confinement check for a model download (used by `localm pull`, the same-repo
  vision-projector auto-attach, and the `--mmproj` flag) verified a downloaded
  file's destination stayed inside your models folder, but did not reject
  every Windows filename shape that can still name something other than what
  it appears to while staying inside that folder: a repo-supplied name
  containing a colon could open a hidden alternate data stream behind an
  ordinary-looking, apparently empty file, and a name shaped like a short
  8.3 alias could resolve to an unrelated file you already had, causing it to
  be silently registered under the pulled name instead of the file actually
  being downloaded. Both are now rejected, alongside reserved Windows device
  names and a trailing dot or space (which Windows silently strips, so two
  visually distinct names can refer to the same file). No effect on ordinary
  model filenames.
- **A bug report can no longer include your actual chat content, and asking for
  help no longer means filing three copies of the same complaint.** Attaching a
  debug log tail to a report could pull in a raw model reply, a snippet of your
  own text from an embedding failure, or a web-tool query built from your
  prompt - because those log lines looked like any other line to the report
  builder. Worse, if that leaked text happened to contain a word like "error"
  it was kept and prioritized ahead of the report's real errors. Report
  generation now recognizes the exact log lines that can carry your content and
  withholds them (with a "records withheld" note), including a check for a
  reply that fakes a log-header line to slip past it. Separately, the "What I
  was doing" and "What happened" fields used to be the same text twice, telling
  the maintainer nothing about what you expected versus what went wrong - the
  bug-report form now has three distinct fields (what you were doing, what you
  expected, what actually happened). A long run of repeated native log lines
  with no timestamp of its own (which could fill a whole report's character
  budget with one line copied dozens of times) is now collapsed before the
  budget is spent.
- **An oversized batch sent to a HuggingFace-backed embedding model could run
  for a very long time instead of failing fast.** `/v1/embeddings` passed its
  input straight through with no limit on how many texts, or how much text,
  one request could contain, and a HuggingFace embed runs one text at a time
  against a full-precision model with no batching of its own - so a large
  enough batch could plausibly run for however long the request's timeout
  allowed. A request against a HuggingFace-format model that exceeds a
  configurable text-count or character-count cap is now rejected immediately
  with a clear error instead. Not applicable to GGUF models (which cannot
  embed at all) or the dedicated on-device embedder (a separate, already
  small, purpose-built path).
- **Starting the coder in your home folder no longer puts your account name into
  its prompt.** The coder shows its working directory as `~/.` when that folder
  IS your home directory, specifically so the prompt never carries your OS user
  name. A clarifying note added beside it worked the folder name out separately,
  and for the home directory that name is exactly the account name - so it went
  into every system prompt and every sub-agent brief, and from there into
  anything that saves or logs one. The note now takes its name from the same
  text the model was shown, and says nothing at all when there is no folder name
  to mention. Projects anywhere else were never affected.
- **A file shared to localm can no longer be written somewhere localm cannot see
  it.** Sharing into localm from another app passes that app's own filename
  through. A name like `photo:stream.png` was used as-is, and on Windows a colon
  means "alternate data stream": the share landed in a hidden stream attached to
  a zero-byte file, so the pending-shares list showed an empty `photo` while the
  actual content sat somewhere the app cannot read. A name containing a NUL byte
  failed with a bare server error instead. Shared names now go through the same
  check uploads already used, and every name in a batch is checked before
  anything is written, so a refused share leaves no half-written entry behind.
- **A corrupt scheduled-jobs file is no longer copied aside with your key's
  fingerprint inside it.** When `jobs.json` cannot be parsed, localm copies it to
  `jobs.json.corrupt-<timestamp>` rather than lose your scheduled jobs. That copy
  was verbatim, so it carried each job's owner fingerprint - the same hash the
  keystore holds - and nothing ever removed old copies, so one could outlive the
  move to the slower key derivation indefinitely. The fingerprint is stripped
  from the copy now, and only the newest few copies are kept.
- **A model can no longer hang the coder with a search pattern.** The coder's
  `grep` and `search_replace` tools compile a regular expression the MODEL
  writes, so its shape is not under your control. Measured against a real
  repository: a 24-character pattern took 1.7 seconds, quadrupling every two
  characters - roughly 31 hours at 40 characters. Both tools are reachable by a
  restricted, shareable key, and `grep` asks for no confirmation because it
  changes nothing. Patterns are now capped in size and the match itself is
  time-bounded, with the cap set so an ordinary search never reaches the timeout.
- **An owner key you chose yourself is no longer stored as a fast, unsalted
  fingerprint.** localm lets you pick your own key (`localm key set`, the
  `LOCALM_API_KEY` variable, or writing `auth.key` by hand), and a key you chose
  can be short or memorable. Its fingerprint was recorded in `sessions.json` and
  `jobs.json` with a single fast hash, which anyone able to read those files
  could work backwards from to recover the key itself - and unlike the key file
  beside them, those two were readable by other accounts on Windows until
  recently. Chosen keys now use a slow, salted derivation (scrypt), worked out
  once per run instead of on every request, so day-to-day speed is unchanged.
  Existing installs upgrade themselves the next time the key is used: you are
  not signed out, scheduled jobs keep their owner, and nothing needs re-entering.
  Keys that localm generated for you were never at risk (they are random and long
  enough that hash speed is irrelevant) and keep working exactly as before.
- **A plugin name can no longer escape the plugins folder.** Installing or
  refreshing a plugin took the name you gave it and joined it straight onto a
  path, so a crafted name like `..\something` pointed at a directory NEXT TO
  the plugins folder - and installing reached its clean-up step and deleted
  that directory. Any name that is not a plain one-word plugin id is now
  refused before it becomes a path: the HTTP API answers 404, the CLI prints an
  error, and nothing on disk is touched. Normal plugin names are unaffected.
- **Installing a plugin can no longer delete a different plugin.** The clean-up
  that runs when an install fails deleted the destination folder even when the
  install had not created it. On Windows and macOS, where folder names are
  case-insensitive, installing `MyTool` while `mytool` was installed therefore
  destroyed `mytool` and everything in it. Clean-up now only removes a folder
  that the failed install itself created.
- **A third-party plugin can no longer smuggle a file out of your machine
  through a symlink.** Installing a plugin from a directory followed symlinks
  while copying, so a plugin shipping something like
  `web/notes.txt -> <your private key>` had that file's CONTENTS copied into
  the installed plugin folder, which localm then serves over HTTP. A plugin
  source containing any symlink or Windows directory junction is now refused
  with a message naming it, so an installed plugin is always plain files. That
  also stops a folder linking back to itself from driving a huge recursive copy
  (measured at 63 nested levels before it failed), which happened before any of
  the plugin's own code ran.
- **A model name from the API or an MCP client can no longer point at any file on
  your disk.** Naming a model used to fall back to treating the name as a path, so
  a scheduled job or an MCP tool call could hand the server any folder on the
  machine and have it read and load what it found. For a HuggingFace model folder
  that meant the folder's own bundled Python was imported and RUN, because localm
  passed transformers' "trust remote code" flag unconditionally. Scheduled jobs and
  MCP requests now have to name a model you have actually registered, and say which
  name they did not recognise when they do not. Running a model straight from a path
  on the command line (`localm run D:\models\foo.gguf`, `localm gui <path>`,
  `localm mcp --model <path>`) is unchanged: you typed it, so it is still allowed.
- **A model's own bundled code is no longer executed just because you loaded it.**
  Custom model code is off by default; a model that needs it is refused with an
  explanation and the setting to turn on, instead of silently running. The new
  owner-only "Allow model-bundled custom code" setting (Settings -> Security) turns
  it back on for a model you trust. `localm pull` now also tells you when a
  downloaded repository contains Python files.
- **A restricted key can no longer plant arbitrary server paths in your model
  registry.** Registering a model records an absolute path that localm re-reads
  every time it lists models, so deciding who may write one is a filesystem
  question, not just a models question. Two routes did not treat it that way.
  "Scan ComfyUI models" only checked for host filesystem access when the request
  named a folder explicitly, so the plain Scan button's request skipped the check
  and scanned whatever folder was configured, and that folder was itself settable
  without owner rights. Pulling a model forwarded its spec straight through, so
  naming a path that already existed on the server registered it where it sat. A
  key granted no filesystem access at all could use either one to add a path of
  its choosing and read back the filename and exact byte size, and a network path
  added that way made every later model listing hang while Windows tried to reach
  it. A third route, the curated ComfyUI model download, likewise wrote into a
  folder that key could choose. All three now require host filesystem access.
- **A malformed Ollama manifest can no longer send localm outside the blobs
  folder.** The digest recorded inside a manifest became a filename with no check
  on its shape, so a hand-written or hostile one could name a file anywhere on
  disk and have it opened as a model. Digests must now actually look like
  digests, and a manifest whose digest is malformed says so instead of being
  passed along. A manifest that is misshapen in other ways no longer crashes the
  command that read it; those cases are noted in the debug log rather than shown,
  because the commonest one is simply a folder that was never an Ollama manifest.
- **`localm rm` no longer deletes outside your models folder.** The check
  deciding whether a registered file was localm's to delete compared text, not
  real locations, so an entry pointing above the models folder could be removed,
  and so could a folder whose name merely started the same way (a `models-old`
  sitting next to `models`). It now compares resolved locations, and the models
  folder itself is never a deletion target. A registry entry containing `..` is
  now treated as corrupt everywhere it is read, so `localm list` shows it and
  `localm rm` clears it rather than any command acting on it.
- **A HuggingFace repo can no longer make an empty download look finished.** The
  check for "is this snapshot fully downloaded" trusted the filenames in the
  repo's own listing, so a listing naming files elsewhere on the disk could be
  satisfied by files localm never downloaded, and that half-present folder was
  then registered as a ready model. Names that point outside the download folder
  are refused and the snapshot is re-downloaded.
- **A chat request could crash or hang the model worker with a single
  carefully-shaped request.** Tool-call detection hands a pattern to the native
  model runtime to watch for while it generates, and that matching had no size
  or time bound: a pattern shaped to make it backtrack badly could take the
  worker down or freeze it, whether the pattern came from localm's own
  tool-call detection on ordinary (if awkwardly repetitive) model output, or
  was supplied directly in a request's `grammar_triggers` field. Both are
  fixed: localm's own pattern no longer has the shape that triggers it, and
  every caller-supplied pattern is now checked against known-dangerous shapes
  and run against adversarial test input in an isolated process before it is
  ever used, so a pattern that would hang stays contained to that check and
  never reaches your request. An oversized pattern is now also rejected
  outright before any of those checks run.
- **A downloaded HuggingFace model's own tokenizer could crash or hang the
  server.** Loading a model that ships a `tokenizer.json` compiles patterns
  from that file into a native matcher that then runs against every message
  you send it. Those patterns are not localm's own; they come from whatever
  produced the model. A pattern shaped to backtrack badly could freeze the
  whole server rather than one request, since this loader runs in the main
  process rather than a separate worker, and some shapes crash the matcher
  outright instead of hanging. A pulled model's tokenizer patterns are now
  tested against adversarial input in an isolated process before the model is
  used, and a model whose tokenizer fails that check is refused.
- **A single stuck request to a HuggingFace-format model could eventually
  freeze every model on the server, not just that one request.** Background
  work (loading a model, counting tokens, computing embeddings) shares one
  fixed-size pool of worker threads, and a HuggingFace model's native
  tokenizer and generation calls could not be interrupted once started - so
  one request that triggered a hang permanently used up a thread, with no
  way to get it back short of restarting. Enough of those (about a dozen and
  a half) exhausted the pool entirely, which then blocked every other
  model's requests, embeddings, and model loads too. HuggingFace-format
  models now run in their own isolated worker process, matching how GGUF
  models are already handled - a hang is now contained to that one request
  and cleaned up automatically, and the rest of the server keeps working. A
  client that disconnects mid-generation no longer forces that model to
  restart either: the worker now stops the generation in place and keeps
  serving your next request immediately, instead of reloading first.

### Changed
- **Windows installs no longer carry 49 command-line tools localm never runs.**
  `setup-llama` was copying every executable out of the upstream archive -
  `llama-cli`, `llama-server`, `llama-bench`, an RPC server daemon and ~45
  others - none of which localm invokes: it loads the runtime in-process. They
  are no longer installed, which is also what the macOS and Linux installs have
  always done. Re-run `localm setup-llama --force` to clear them from an
  existing install. No library is affected.
- **The bundled AMD (ROCm) llama.cpp runtime moves to build b1307**, from
  b1288. Two months of upstream llama.cpp, a newer ROCm runtime, and more
  gfx1030 GEMM kernels than the previous build shipped. Run
  `localm setup-llama --force` to pick it up; an existing installation keeps
  working untouched until you do.
- **Breaking, for scripts driving a headless `localm serve` over REST:
  `POST /api/rag/collections/{name}/add` and `.../upload` now return
  `{"job_id": ...}` instead of the finished index result, and
  `POST /api/rag/embedding` starts a job instead of refusing.** On a headless
  server those routes used to behave differently from the same routes under the
  GUI, because background jobs only existed when the GUI was attached: add and
  upload ran to completion inside the request and returned counts, and
  embedding setup returned an error telling you to run `localm gui`. Background
  jobs now exist on every server, so all three behave the same way everywhere
  and you follow progress on the job instead of waiting on one long request.
  If you parse the response of a headless add or upload, you need to update it.
- **Settings now shows you which fields you've actually changed.** A field
  still on its shipped default now renders blank with the default shown as a
  greyed placeholder, instead of looking identical to a value you chose
  yourself. A field you did override still renders solid, and clearing an
  override still sends an explicit "use the default" rather than silently
  reverting. The coder session panel's "Max turns" field gets the same
  treatment - blank means "use the server's default," not a client guess of 40.
- **Pulling or verifying a model file is measurably faster.** Hashing now reads
  in 4 MB blocks instead of 64 KB, roughly doubling throughput on a local SSD
  (233 to 479 MB/s hashing a 10 GB file in testing) with no change to the
  digest produced. Files under 32 MB are unaffected (too small for the overhead
  to pay off).
- **Three model routes now require host filesystem access:** scanning for ComfyUI
  models, pulling a model by naming a path already on the server, and downloading
  a curated ComfyUI model. This affects only additional keys you minted yourself
  without filesystem access. Running localm normally, or using an owner key, is
  unchanged, and pulling from HuggingFace by name is unaffected.

### Fixed
- **Asking about an image on an install without the PyTorch stack no longer
  fails as a "native inference fault".** Image understanding needs the Pillow
  imaging library, which until now was installed only alongside PyTorch and
  transformers. Setup skips that whole stack when it is not needed for chat, so
  a GGUF-only install had no image decoder, and attaching a picture reported
  "Native inference fault (worker exit 1). The model has been unloaded ... see
  the debug log for the native stack trace" and dropped the model out of memory.
  None of that was true: there was no native fault, there was no native stack
  trace, and the model was fine. Pillow is now installed with localm itself, so
  image understanding works on every install rather than only on one with a
  graphics stack. If it is missing anyway, the message now names Pillow and the
  model stays loaded.
- **Unloading a model while it was still loading no longer reports a broken
  llama runtime.** If a model was being loaded in the background while
  something else freed memory or swapped models, the load could fail with
  "Native llama runtime failed to load ... Provision or repair it with localm
  setup-llama" - pointing at a runtime that was perfectly fine. It now reports
  that the load was superseded, which is what actually happened. The same race
  during a reply or a token count now says the model was unloaded instead of
  failing with an internal error.
- **Image understanding now runs on the GPU instead of the CPU.** The vision
  projector was pinned to the CPU for every user, on every GPU, with every
  projector, to work around a failure that only affects one AMD card paired with
  a bf16 projector. On top of that it used a fixed 4 threads no matter how many
  cores the machine has. A screenshot took about ten minutes with the graphics
  card sitting idle, close enough to the timeout that it often failed outright.
  It now uses the GPU, and only falls back to the CPU if the GPU attempt really
  fails, saying so in the log. The same image now takes a couple of seconds. Set
  `LOCALM_MTMD_CPU=1` to force the old CPU behaviour.
- **Asking a GGUF vision model about an image crashed the model process.** The
  reply was "Native inference fault (worker exit 1). The model has been
  unloaded", and the model had to reload from scratch on the next message. A
  recent llama.cpp change added an explicit length field to the structure
  localm passes the prompt in; without it the runtime read a garbage length and
  cut every image prompt off after 257 bytes, losing the marker that says where
  the picture goes. Anything ahead of the image pushed it past that limit, so
  with the memory plugin recalling anything at all this happened on every image,
  even in a brand new chat. localm now detects which layout the installed
  runtime uses and passes the prompt whole. Separately, an image the projector
  genuinely cannot process is now reported as a failed message, leaving the
  model loaded, instead of being treated as a crash.
- **A low-VRAM warning could blame the running server for holding its own
  VRAM.** When possible, the warning names a concrete instance holding the
  memory, but the lookup never excluded the process printing the warning
  itself, so it could report "another localm instance (port N) is running
  ... - POST /v1/models/unload on port N to free it" about its own,
  just-loaded model - telling you to unload the model you were about to
  use. It now excludes itself from that lookup, and when the VRAM really is
  held by one of its own other resident models, says so plainly instead of
  blaming a nonexistent sibling.
- **Chat now reloads a model that was evicted to make VRAM room for the
  embedder.** Running an embedding or RAG task frees every loaded chat model,
  and the server is built to bring it back on the next turn. It could not: a
  chat request that did not name a model explicitly was refused with
  "Model parameter is required and cannot be empty" before the server ever got
  as far as working out which model the request meant, so the model was never
  reloaded and chat stayed dead until it was loaded by hand from the Models
  page. An unnamed request now resolves to the model in use, exactly as the
  documented `"localm"` default already did. A request that genuinely cannot be
  served (no model loaded and none configured) is still refused the same way.
- **A refused request now records WHY in the debug log, not just the status.**
  A failing request logged its status and timing and nothing else, so a chat
  failure reported with a full debug log still could not be diagnosed. The
  reason the server gave the client is now written to the log beside it.
- **An unnamed request after switching models could silently reload the
  wrong one.** The model an unnamed chat turn falls back to was only ever
  updated at server startup, not on a later model switch, so start model A,
  switch to model B, then trigger an eviction - the embedder freeing VRAM,
  unloading B by itself, or B idling out - and the next unnamed turn
  silently came back as A instead of B. It now tracks the model actually
  last in use, whichever of those paths freed it.
- **`GET /health` reported a plain 503 for a model that was about to reload
  itself.** After an eviction the server keeps the model on hand to reload on
  the next request, but health had no way to say so and just reported "no
  engine" - a likely reason to reach for a manual reload instead of just
  sending another chat turn. It now reports the recoverable model instead,
  and still 503s when there genuinely is nothing to recover.
- **Image, music and video generation, and RAG indexing/re-embedding/embedding
  setup, could report "failed" for a job that actually succeeded.** If VRAM
  handover back to the chat model (or, for embedding setup, the collection
  impact report) raised after the real work was already done, the job's
  background-thread runner read the exception as failure - the same class of
  bug already fixed for model pulls and managed-ComfyUI setup. Each of these
  now tells the job runner the true outcome directly, before that risky
  cleanup step runs, so a crash there can no longer misreport a completed
  operation.
- **Setting up or updating localm's managed ComfyUI could report "failed" for an
  install that actually succeeded.** If the final status line crashed after the
  real work (clone, venv, packages) was already done, the GUI read the crashed
  subprocess's non-zero exit code as failure - the same class of bug already
  fixed for model pulls. The CLI now tells the GUI the true outcome directly,
  before that risky display step runs, so a crash there can no longer
  misreport a completed install.
- **`localm doctor` no longer invents a reason when its native-ABI check cannot
  run.** If that probe timed out or crashed, doctor still reported the cause as
  "runtime not loadable" - a specific diagnosis it had not made, pointing at a
  repair command that cannot fix a probe which never ran. It now says the probe
  did not run, and still passes through the real reason whenever the probe
  reported one.
- **`localm doctor` no longer shows a green tick for a GPU driver that is
  broken.** It read whatever `nvidia-smi` or `rocm-smi` printed without checking
  whether the tool had actually succeeded, so a driver that fails to initialise -
  the usual state after a driver update with no reboot - was reported as a
  working GPU, with the error message itself shown as the graphics card's name.
  Worse, that false positive also suppressed the "No GPU detected ... CPU mode
  only" warning, so nothing anywhere told you something was wrong. doctor now
  counts only a tool that exits cleanly, and reports one that is installed but
  failing as its own warning with what the tool actually said. A tool that is
  simply not installed stays silent, exactly as before - localm's default GPU
  paths never need it.
- **Re-running `setup-llama` on an AMD box now tells you it is an upgrade, and
  from which build.** It records the build tag it installed, so a genuine move
  between llama.cpp builds says "Upgrading the amd-rocm build: b1288 -> b1307"
  instead of a bare "Re-downloading" that reads like a no-op. An install set up
  before this existed, or on a backend whose build tag cannot be determined
  without a network call, still says only what it actually knows rather than
  guessing a version. No re-provision is needed to pick this up: the tag is
  recorded next time you run setup-llama.
- **Warming up the embedding model no longer promises "up to a minute" for
  something it gives five.** The progress line shown while an embedding model
  loads quoted a ceiling five times shorter than the one the load actually runs
  under, so a slow but perfectly healthy first load - a large model, a cold
  cache, a slow disk - looked like it had hung with four minutes still legitimately
  left on the clock. It now states the real bound.
- **A model download could show a confident percentage it had no basis for.**
  Five related problems in the pull progress bar: a failed download announced
  100% before reporting failure; resuming an interrupted direct-URL download
  whose server sends no length could sit at a stuck 100% for the entire real
  transfer; resuming any download briefly claimed 0 bytes when it already had
  some on disk; a download whose size could not be determined went completely
  silent instead of showing a running byte count; and the in-flight byte count
  could include a different download running at the same time, or watch the
  wrong folder entirely when pulling into a custom destination. Progress now
  comes from what is actually on disk, and says "unknown" rather than guessing.
- **The setup menu no longer lists your recommended GPU backend twice.** Option
  [1] is a shortcut for whatever setup detected for your hardware, so it was
  always the same choice as one of the numbered entries below it - two lines that
  looked like different options and did the same thing. The twin is now marked as
  such. Re-provisioning an existing install also describes itself correctly
  instead of announcing "Replacing amd-rocm build with amd-rocm" for what is a
  re-download, or "with auto" when auto is how a backend gets picked rather than
  a backend you can install.
- **A self-update could silently swap your GPU runtime to a slower one.** When an
  update needed to re-provision the native binaries, it picked the backend from
  an older detection value that only ever recommends the universal Vulkan build
  or CPU, never the faster vendor-specific one. On a Windows AMD RX 6000 GPU,
  that meant a runtime update could quietly replace an installed ROCm build with
  Vulkan, with no notice. It now uses the same detection the installer and
  first-time setup already use, so an update reprovisions the backend you
  actually have.
- **Re-provisioning the GPU runtime while localm is running no longer damages the
  install.** `localm setup-llama` clears the old build before writing the new one,
  and on Windows a library that is currently loaded cannot be deleted. Those
  failures were being ignored, so the clear-out could half finish and the new
  build was then written over the survivors, leaving a runtime mixed from two
  different versions - or, when the copy failed too, an install missing files
  with nothing reporting it. Setup now checks whether anything is holding the
  runtime **before** deleting a single file, and if so it stops with the existing
  install completely untouched and tells you to close whatever is using it. A
  locked file also no longer silently switches you to a different backend: your
  chosen backend was never the problem, so it is no longer swapped out behind
  your back.
- **`localm doctor` now catches a GPU runtime whose maths kernels are missing.**
  On AMD ROCm installs, rocBLAS loads its GPU-specific kernels from a data folder
  next to the library rather than from the library itself. An install could be
  missing that folder entirely and still pass every check doctor had - the
  library was present, the right size, and loaded fine - right up until a
  workload that uses it (such as generating embeddings) crashed the process
  outright. Doctor now reports this as a failure with the command to fix it.
  Backends that do not use rocBLAS are unaffected and are not checked.
- **A chatty GPU log line no longer floods the console and the activity view.**
  On some workloads the native runtime prints a "CUDA Graph id N reused" line
  many times per second, with the number cycling through a set of values. The
  repeat-collapsing that already tidies the other native lines could not group
  these, because it matched on the whole line and the number kept changing, so
  they came through one by one - a captured session showed 8934 of 9012 lines
  being this single message. They are now collapsed into one counted line that
  also reports how many different values appeared, and the lines around them
  group properly again instead of being pushed out by the flood. Messages that
  genuinely differ, such as a per-layer load report, are still shown in full.
- **`localm doctor` no longer claims the native ABI check passed when it was
  switched off.** With `LOCALM_SKIP_ABI_CHECK` set, doctor reported "native ABI:
  struct layout matches this build" - stating the layout had been verified for a
  check that never ran, and making its own "check skipped" line unreachable. It
  now says the check was skipped, and reports which struct layout the runtime
  uses either way.
- **A newer llama.cpp runtime could silently apply the wrong model settings.**
  Upstream rearranged the fields inside one of the structures localm passes to
  the native library, without changing its size, so nothing detected the
  change. Against such a build localm's "use this GPU" setting was quietly
  discarded (the model loaded on the default device instead), its request to
  skip memory-mapping the weights had no effect, and the value meant for the
  GPU index landed on the setting that controls how the weights are mapped into
  memory. localm now recognises both arrangements and picks the right one for
  whichever runtime is installed, so old and new builds both behave correctly,
  and it refuses to load a runtime it cannot place confidently rather than
  guessing. This affected anyone who provisioned a very recent llama.cpp build,
  not only the AMD one.
- **Repetition penalty is now applied correctly on newer llama.cpp builds.**
  Upstream added an argument to the repetition-penalty sampler; on a build
  carrying that change localm was calling it the old way, which scrambled every
  value it passed. On a build where localm cannot tell which form that build
  expects, the repetition-penalty stage is now skipped with a warning naming
  the reason, rather than making a call that would misbehave silently.
- **`localm add --store move` (or `copy`) can no longer relocate your file and
  then report success while leaving it unregistered under any name.** When
  the destination name was already taken by a genuinely different model and
  nothing could ask you to confirm an overwrite (the GUI, a script, any
  non-interactive caller), the file was still moved into the models folder
  first - only afterward did registration get refused, and that refusal was
  never checked, so the command reported success anyway. The registry still
  pointed at the old file; the moved one sat unregistered until the next
  `localm list` or server start picked it up under an automatic name, never
  the one you asked for. The conflict is now checked before anything is moved
  or copied, so a refusal leaves your file exactly where it was; an
  interactive decline is also now correctly reported as a failure instead of
  a false success.
- **Stopping a reply mid-generation no longer loses it on reload.** A stopped
  reply rendered live with a "[stopped]" marker, but nothing was ever saved -
  reload the page or switch conversations and it was gone, even though it was
  still on screen a moment before. It is now saved as its own message, so it
  survives a reload; it is explicitly marked stopped so a later turn can never
  mistake it for a finished, continuable reply - it still cannot be spoken
  aloud or trigger a web-search follow-up on its own, the same as before.
  Separately: a completed reply's token-rate figures (tok/s, time to first
  token) reached the on-screen status line but never the saved message, so
  they were gone on reload for every turn, not just a stopped one - both the
  figures and the context-usage gauge are now saved with the reply and shown
  again on reload or when switching back to that conversation.
- **The GUI no longer needs a manual reminder to invalidate its own cache when
  a static file changes.** Every change to the app's HTML, JS or CSS used to
  need a hand-typed version bump so browsers would fetch the new copy; missing
  one meant an already-open browser could keep serving a stale, possibly broken
  page indefinitely, since a service worker only re-checks its cache when its
  own bytes change. The cache key is now computed automatically from the actual
  contents of every cacheable file, so it changes exactly when something
  relevant does.
- **Restarting or stopping localm from the tray icon (Windows) no longer
  reports a false crash on the next startup.** The tray's Restart and Stop
  buttons called the server's internal hooks without identifying which running
  instance they meant, which cleared the wrong bookkeeping and left the real
  one armed - so the next start reported a crash that never happened. They now
  correctly identify the instance (and, for Restart, the port it was really
  running on). Separately: when a real crash IS detected, its leftover trace
  file is now cleaned up instead of accumulating on disk forever, a failure to
  attach the crash detector is now logged instead of silently leaving you with
  no report if a crash does happen, and the report's stated cause now reflects
  the actual evidence found - a captured fault trace, a log that cuts off
  mid-word during native model loading, or an honest "no evidence found" -
  instead of a generic guess. This does not explain any specific past
  unexplained crash; it only stops these particular false positives and makes
  a real one easier to diagnose.
- **Listing or opening a Knowledge collection can no longer freeze the whole
  server.** Both actions used to fully re-parse a collection's stored chunks
  and vectors just to report counts, and because localm answers requests on a
  single thread, that parse blocked every other in-flight request for as long
  as it took - measured at up to several seconds for a multi-thousand-chunk
  collection. Listing and viewing a collection now read from a small cached
  summary instead, cutting a measured 3.3 second freeze to 0.003 seconds. A
  collection that predates this cache still falls back to the slower path the
  first time, but that fallback no longer blocks the server outright while it
  runs.
- **A failed chat request in the GUI no longer shows you raw JSON.** The error
  display used to take the server's response text and cut it off at 300
  characters, so a VRAM-overflow error's actual list of suggestions (lower the
  context, offload fewer layers, and so on) was silently cut away past that
  point, and what remained showed literal `{"detail":...` markup instead of a
  message. The full server-provided explanation is shown now, with no length
  cap, the same way every other error in the GUI already worked.
- **A coder session can now be repointed to a different model without losing
  its history.** Previously a session's model was fixed at creation - switching
  the active model elsewhere in the app could make the session's next message
  silently reload its original model back into VRAM, evicting whatever you had
  just switched to. A session can now be told to use a different model in
  place; only the account that owns the shared inference engine can trigger a
  switch that affects it for everyone, and a session currently mid-task refuses
  the change rather than corrupting it.
- **A scheduled job or memory auto-consolidation can no longer have its model
  unloaded out from under it.** If idle-unload is turned on (off by default)
  and the server was otherwise quiet, a scheduled chat/memory job or a
  background memory-consolidation pass could get evicted mid-run, because
  those paths called the inference engine directly instead of going through
  the same "mark this model busy" path an ordinary chat request uses. Both now
  pin the model for as long as they are actually using it.
- **A vision-capable model's projector file is now fetched and wired up
  automatically when you pull it, and every way of starting localm now
  actually uses it.** Pulling a vision GGUF previously downloaded only the
  language model - the separate projector file needed to see images had to be
  found and attached by hand, with no warning that anything was missing until
  you tried sending a picture and it silently failed. Pulling now checks the
  source repository for a projector sibling, verifies it really is one, and
  records it; `localm gui`, `localm serve`, and `localm run` all now pick that
  projector up automatically instead of only two of the four places that build
  a model doing so. An explicit `--mmproj` flag still always wins if you want
  to override it.
- **The coder plugin now shows you the server's actual error instead of just
  "Service Unavailable," and no longer wastes half a minute retrying a failure
  that was never going to succeed.** A non-auth error response from the coder's
  HTTP backend (local or remote) used to report only the bare status line, with
  no detail even when the server had already sent one - now the response body's
  detail is read and shown, capped at 500 characters. Separately, a 503 from
  localm's own local server used to be retried like any transient network
  error, up to 5 requests over roughly 30 seconds; since every such 503 from
  the local server has already either failed deterministically or exhausted
  its own wait before the response reaches the client, it now fails
  immediately on the first try instead. A remote/cloud backend's retry
  behavior is unchanged.
- **Setup and update output for the bundled ComfyUI installer now appears live
  instead of going silent for minutes at a time.** The log used to buffer all
  output from git/pip/venv steps and only show it once the whole command
  finished - one report saw over five minutes of apparent silence with only
  two log lines total, despite work actively happening in the background.
  Output now streams line by line as it happens, for both a fresh install and
  an update.
- **Switching the active model while another model has gone idle no longer
  risks the newly switched-to model being evicted before it ever answers a
  request.** The idle-unload timer only started counting a model's activity
  from its first served request, so a model that inherited the previous
  model's already-expired idle clock the instant it was loaded could be
  evicted by the very next idle sweep. A newly loaded or switched-to model's
  activity clock is now seeded at load time instead.
- **Native model-loading output (llama.cpp/ggml) from an embedding-model load
  is now grouped and de-duplicated the same way as every other model load's
  output**, instead of appearing as raw, unformatted spew with no timestamp and
  no repeat-collapsing. Only visible with `--debug`/`LOCALM_DEBUG` on; the
  persisted debug log file is unaffected either way.
- **A coder tool call that is blocked, denied, or skipped (patch mode
  interception, a scope restriction, a dry run) now correctly shows its result
  in the chat instead of leaving its card stuck on "..." forever**, which also
  desynced every tool result after it onto the wrong card for the rest of the
  session. The reported duration for a tool that never actually ran is now
  omitted rather than shown as a misleading "0.0s".
- **A grammar-constrained chat request that fails because the model worker
  itself faulted no longer tells you your grammar syntax was wrong.** It used
  to be reported as a plain "invalid grammar" error even when the real cause
  was an unrelated crash or timeout in the isolated worker process; it now
  reports the worker fault for what it is.
- **Native model-loading output that alternates between two or more distinct
  lines (common with CUDA's own warmup chatter) is now collapsed the same way
  a single repeating line already was.** The line-grouping logic used to only
  compare each new line to the one immediately before it, so an alternating
  cycle never matched and the console, GUI status window, and bug-report log
  tail all filled with the full repeated pair for the whole generation.
  Grammar-constrained requests also switch to this same grouped view - they
  previously suppressed native output entirely, hiding real grammar
  diagnostics along with the noise. Only affects the live console/GUI view;
  the persisted debug log file still gets every raw line.
- **`localm gpus` now tells you when its free-VRAM figure only accounts for
  this process**, not the whole card - on some driver/platform combinations
  (notably AMD ROCm/HIP on Windows) the number can look reassuringly high on a
  card that other running processes have actually filled. `localm doctor`
  already carried this caveat; `gpus` now matches it.
- **A connection that drops out mid-relay while the server is shutting down or
  restarting is now closed cleanly instead of abandoned**, and no longer leaves
  "Task was destroyed but it is pending!" warnings in the log.
- **Manually synthesizing memory ("Synthesize now") no longer fails with "load
  a model first" right after you've switched models, and no longer freezes the
  server for the length of the synthesis call.** The memory plugin used to
  cache which model was active only once, at startup, so switching models
  later left it pointing at a stale, unloaded reference - it now always
  resolves the model that is actually active. The synthesis call itself is now
  run off the main thread, so a 16-second synthesis no longer stalls every
  other request for 16 seconds.
- **Two more bugs affecting every GGUF model load are fixed.** Every templated
  token-count request to the model worker was silently failing (a
  bytes-versus-string mismatch) and falling back to a less accurate count that
  skipped the chat template's own markup - this had been happening on every
  prompt, on every load, since the isolated worker was introduced, without
  ever surfacing above debug level; a real recurrence would now log a warning
  instead of staying invisible. Separately, the "vram: X GB in use" line
  printed after a model loads is no longer read from a raw, process-blind
  measurement that could print a number far from reality (one case: "0.14 GB
  in use" on a card that actually had 10.53 GB in use); it now uses the same
  trusted reading the GUI's status bar relies on, and says so plainly when
  that reading can't be trusted rather than showing a number that might be
  wrong. For a CPU-only load, where nothing is ever placed on a GPU, this line
  is now skipped entirely instead of printing a technically-uncorrected
  reading (and paying the cost of checking one) for nothing to report.
- **The coder's episode store (its cross-session "lessons learned" log) no
  longer silently drops and then permanently deletes an episode** if its text
  happened to contain certain rare Unicode line-separator characters - the same
  class of bug already fixed in the coder's RAG store and in agent memory,
  applied here to the one file that had not received it yet.
- **The coder's reported tool execution time is now real, not a guess.** The
  GUI used to estimate how long a tool took from the gap between two UI render
  events, which usually landed in the same screen update and read as roughly
  0.0 seconds regardless of the tool's actual cost. The server now measures and
  reports the real duration; a tool that never ran (blocked, denied before
  execution) reports exactly 0.0 rather than nothing, so the GUI can tell
  "genuinely instant" apart from "did not run."
- **A model that fails to load now leaves a real diagnostic message instead of
  a blank one, for any kind of load failure** - a corrupted file, running out
  of memory, an unsupported quantization. The native runtime's own explanation
  was being captured to a temporary file and then deleted before the caller
  ever read it, so this affected every failed GGUF load, not just a rare case.
- **HuggingFace model search now recognizes MoE chat models it used to badge
  "unknown".** Type detection only read a repo's `pipeline_tag`, and a
  HF-repacked GGUF-only quantization upload commonly never sets that field -
  it belongs to the original checkpoint's model card, which a pure quantizer
  repo often leaves blank. Search results now also read the repo's tags and
  the model's own GGUF header architecture, so a GGUF-only MoE upload with no
  `pipeline_tag` at all is correctly badged as an LLM instead of "unknown".
  Quant labels like `MXFP4_MOE`, `TQ1_0`, and `TQ2_0` are now recognized too.
  A classified GGUF search query's stat fields (downloads, likes, last
  modified) no longer go missing from every result row as a side effect of
  the same request.
- **The coder's codebase map no longer goes stale after a shell command.** Files
  a shell command created, edited, or deleted (applying a patch, running a
  formatter, generated code) never updated the compact codebase summary
  injected into the model's context - only files the model wrote or edited
  directly did. The map now reconciles itself against the filesystem (a
  lightweight size/timestamp check, not re-reading every file) before the
  model's next turn, so it reflects what a shell command actually did.
- **The coder no longer skips a valid tool call that follows a malformed one.**
  When a reply contained a botched tool call before a good one, the scanner's
  recovery could step past the valid call so it never ran - the model believed it
  had acted and carried on. Smaller local models are affected most, being the
  likeliest to produce the malformed attempt in the first place.
- **The coder no longer writes doubled paths like `proj/proj/file.py`.** When
  your project sits outside your home folder its "working directory" line showed
  only the bare folder name, and models read that name as a prefix to put in
  front of every path they wrote - so every read and edit failed on a path that
  does not exist, over and over, until the run gave up. The line now says
  explicitly that paths are relative to that folder. Reproduced identically on a
  1.5B and a 7B model, so it was the wording rather than the model's size.
- **The coder now tells you when part of a reply looked like a tool call it could
  not read.** If one call in a reply parsed and another did not, the broken one
  was dropped in silence: it never ran, the model was never told, and nothing
  recorded it. It is reported back now, so the model can send it again.
- **A coder session's saved transcript or resume view no longer shows a tool call
  as raw JSON.** When a reply wrote its tool call as a fenced ```` ```json ````
  block or a bare JSON object instead of the usual wrapped form, the full-mode
  `.localcoder/sessions` transcript and the GUI's "resumed your last session"
  recap did not recognise it, so the raw JSON (fence markers and all) showed up
  verbatim instead of being summarised or cleanly removed, even though the call
  itself ran normally. Both places now recognise every form the coder itself
  understands.
- **Resuming a coder session no longer loses track of what it already changed.**
  After a server restart or a GUI reconnect, the list of files changed so far in
  that session came back empty, so both undo and the changed-files view were
  wrong about work that had already happened.
- **The "part of that looked like a tool call" notice no longer repeats every
  turn.** The notice quotes the correct format, and that quote is itself
  tool-call-shaped, so a model that echoed it back triggered the notice again on
  the next turn, indefinitely. It now appears at most twice per task and then
  says so once; every later occurrence still goes to the debug log rather than
  disappearing.
- **The coder's final answer now ends with a short factual record of what it
  did.** A line naming how many files actually changed - and your verify
  command's result, if you set one - is appended to every final answer, taken
  from what the session recorded rather than from anything the model wrote. A
  confident "done, tests pass" from a small model can now be checked against it
  at a glance.
- **`--patch-mode` no longer writes your files for real when the coder searches
  and replaces across them.** Patch mode promises to capture every file write as
  a diff and leave your files alone. That held for every edit tool except the
  across-files search-and-replace, which wrote straight to disk with patch mode
  on - the one tool that can rewrite a hundred files from a single pattern. Those
  writes were also missing from the session's changed-files record and could not
  be undone. All three now work for it as they do for every other edit tool.
- **Two more edit tools now update the coder's internal file index.** Applying a
  patch file and editing a notebook cell both wrote to disk without refreshing
  the map the coder reads, so it could serve a stale summary of a file it had
  just changed itself.
- **The Knowledge page's re-embed button now actually re-embeds.** Switching the
  embedding model leaves existing collections refusing new documents until their
  vectors are recomputed, and the fix for that shipped as a command and an API
  call - but the button re-read your original files instead of using it. That
  tripped the very mismatch you were trying to get past, needed the source folder
  to still exist, and silently skipped every document you had uploaded rather
  than pointed localm at on disk. It now calls the same re-embed the command line
  does: no original files needed, uploads included, nothing deleted.
- **A plugin that fails to uninstall now says so.** Removing a plugin's folder
  can fail on Windows - a locked file, antivirus holding a handle, a permission
  denial - and that failure was discarded silently, so localm reported the plugin
  uninstalled while its folder was still on disk. The failure is logged with the
  path now, and the reported outcome reflects whether the folder actually went.
- **A quarantined scheduled-jobs file now reports honestly what happened to it.**
  Three things were off at once: a leftover credential-shaped value in the copy
  was only noted in the debug log, so you never saw it; the copy's permissions
  were tightened without checking whether that worked, unlike everywhere else
  doing the same thing; and the message claimed localm was "refusing to silently
  discard" your jobs when in fact it warns and continues with an empty list. All
  three now match what the code actually does.
- **localm no longer reports "no GPU" when it simply could not ask.** After
  graphics-card detection moved out of process, a card whose driver library
  wedges left localm reporting an empty GPU list - indistinguishable from a
  machine with no GPU at all. Those two are now told apart, so a card that cannot
  be probed is reported as unknown rather than as absent.
- **A plugin whose dependencies fail to install no longer reports success.**
  Installing a plugin over MCP said "successfully installed and enabled" even
  when its Python extras had failed - a network problem, a version conflict. The
  plugin still stays enabled, matching what the command line does, but the reply
  now says the dependency step failed and names what went wrong.
- **A crashed model-loading process could point you to a debug log that had
  nothing in it.** The isolated process that loads and runs a GGUF model
  already told you to check the debug log when it crashed - but for most
  crash causes (anything other than a fault during the brief moment a reply
  was actively streaming) nothing was actually written there, because the
  process's own crash report was produced after the debug log had already
  stopped capturing it. It is now captured for a crash at any point in that
  process's life, not just while a reply is streaming.
- **A plain-text completion request could freeze every other request on the
  server while it counted tokens.** `/v1/completions` counted the prompt and
  reply tokens with a direct call to the model's tokenizer, which runs on
  Python's single request-handling thread - so a large prompt or reply could
  briefly stall every other concurrent request. `/v1/embeddings` already ran
  this off that thread; `/v1/completions` now does too.
- **Detecting your graphics card no longer freezes the app for several seconds
  on Windows.** The first time localm looked up your GPU it loaded PyTorch's
  graphics libraries inside the server itself. Windows only lets one thing load
  libraries at a time, and starting any new task needs that same permission, so
  while the lookup ran the whole app stopped answering - measured at 10.9 seconds
  on one machine, and longer where the graphics driver is slow to wake up. The
  lookup now happens in a separate helper process, so the app keeps responding
  while it runs, and a graphics driver that never answers is given up on after 20
  seconds instead of hanging startup.
- **A model registered on an unreachable network path no longer freezes the whole
  server.** The models list, a model's detail view and the VRAM estimate each
  measured registered files on the same thread that answers every other request,
  so one unreachable path stalled everything until Windows gave up on it, which
  can take minutes. All three now measure off that thread.
- **The status bar's GPU-utilisation reading no longer spawns a fresh `nvidia-smi`
  process on every 2.5-second poll, and no longer risks blocking a poll on it.**
  Utilisation (NVIDIA only) ran a bare `subprocess.run` inline on every poll with
  no cap on how many could overlap; it now runs single-flighted on its own
  background probe, so at most one `nvidia-smi` is ever in flight and a poll
  always returns immediately with the most recent reading. An efficiency and
  robustness fix scoped to this one reading - it does not touch how or when GPU
  memory is enumerated, and is unrelated to the separate GPU-enumeration freeze
  fixed elsewhere.

- **Changing the embedding model no longer strands your knowledge collections.**
  Switching from one embedding model to another left every existing collection
  refusing new documents, and the only advice was to delete and re-add it. Both
  advertised remedies failed with the same error, so there was no working way
  out. There is now: `localm rag reembed <collection>`, and a matching API
  action, rebuild a collection's vectors with the current model from the text
  already stored - no original source files needed, nothing deleted, and an
  interrupted run leaves the previous index untouched rather than a half-built
  one. The error you get on a mismatch now names your collection and the exact
  command instead of telling you to delete your data.
- **Knowledge collections silently lost documents.** A chunk whose text contained
  certain rarer separator characters was split in half when the collection was
  read back, so those documents were dropped, the collection was reported as
  damaged, semantic search quietly fell back to keyword-only matching, and the
  next save wrote the collection back WITHOUT them. On a real 1192-chunk
  collection this deleted 26 chunks every time it was opened and saved. Existing
  collections recover on their next load; nothing needs re-importing. The same
  fault is fixed in agent memory and in the coder's stored episodes.
- **A damaged collection no longer floods the log or fills the disk.** The
  "this index is unusable" warning was repeated on essentially every request
  (25+ times in one session); it now appears once. Set-aside copies of a broken
  index, which are kept so nothing is ever destroyed, are capped at the newest
  three instead of accumulating without limit (88 MB on one install).
- **Requests rejected for a bad value now say what is wrong.** Sending
  `max_tokens: 0` returned a raw validation dump; it now reads "max_tokens must
  be 1 or more - 0 is not 'no limit'. Omit max_tokens entirely to use the
  model's default." The machine-readable form is still there for API clients.
- **Semantic search no longer fails outright when numpy is unusable.** If numpy
  is present but broken, localm falls back to its built-in maths instead of
  erroring the query, and says which situation it hit - including naming the
  path when something has put an empty stand-in numpy on the import path, which
  is a broken environment rather than a missing optional dependency.
- **A ComfyUI server can no longer delete files outside its own folders.** When
  you turn on output containment (or run in privacy mode, which forces it on),
  localm deletes ComfyUI's duplicate copies of what it generated. It took the
  filenames for that straight from ComfyUI's replies, so a ComfyUI on your
  network, an attacker sitting on that plaintext connection, or a malicious
  custom node could name any file on your disk and have localm delete it. Those
  names are now confined to ComfyUI's own output and input folders. Nested
  outputs (ComfyUI's `subfolder`) still work exactly as before, and a name that
  gets refused is reported in the result rather than skipped silently, so you
  are never told a copy was removed when it was not. This affected the most
  privacy-conscious setups specifically, since containment is what turns it on.
- **The folder picker and the ComfyUI launcher no longer freeze the server on a
  network path.** Handing either one a Windows network path (`\\host\share`)
  made localm contact that host before it checked whether the path was allowed,
  which stalled every other request for as long as the connection took - over
  four minutes against an unreachable address - and would have handed the
  machine's Windows credentials to whoever answered. The path is now checked
  before anything touches the disk, the ComfyUI launcher matches your configured
  folder before it looks at the filesystem at all, and none of these handlers
  can block the server while they wait on a slow disk. Browsing your own drives
  is unchanged, and a ComfyUI folder that genuinely lives on a network share
  still works.
- **The file browser can no longer be read by another page on your machine.**
  A different local page could ask localm to list your folders and read the
  answer. It is now refused whether or not you have an API key set.
- **A tampered update record can no longer delete the wrong files during a
  rollback.** Rolling back an update read a list of file names from your data
  folder without checking them, so an edited list could point the rollback at
  files outside the installation - including, on a portable install, your own
  models and chat history. Those names are now validated, and a rejected one
  fails the rollback loudly with the backup left intact, rather than being
  quietly skipped.
- **A hostile web page or document can no longer freeze the server by being
  awkwardly punctuated.** The patterns that defang untrusted text and that pick
  tool calls out of a model's reply all backtracked badly on input crafted to
  make them. A fetched page that was nothing but a "<" and 60,000 spaces, exactly
  the size the fetch endpoint itself allows, took the better part of a minute to
  clean, and it was
  cleaned on the thread that answers every other request, so the whole server sat
  still for that minute. The tool-call patterns were worse per character: an
  unfinished `<tool_call>` followed by 2,000 spaces, a few hundred tokens of model
  output, cost 7 seconds, and because the coder re-reads every stored reply when
  it writes a session transcript, one such reply re-froze the export every time
  you closed a session from then on. A code fence followed by a long run of tabs
  reached 96 seconds. All of those patterns are now restructured, and the
  defanging happens on a worker thread, so a slow page is a slow request instead
  of a stalled server. On the same inputs the new code takes between 0.0001 and
  0.005 seconds, and it defangs exactly the same markers as before, including
  ones separated by arbitrarily long runs of whitespace. Two deliberate changes
  to what counts as a tool call come with it: a mangled `<|tool_call>` wrapper
  whose JSON body itself contains a `<tool_call>` marker is no longer recovered
  (the coder still notices and asks the model to reformat), and the session
  transcript now recognises the `<tool_call name="...">` form it previously
  printed as raw XML.
- **Only the owner can change the embedding model now.** The Knowledge page's
  "Set up / apply" button wrote that setting under the RAG plugin's own
  permission, so a deliberately restricted key (`--scope chat --scope rag`) could
  repoint localm at a file of its choosing, which is then opened and parsed by
  the native GGUF reader. That endpoint now requires the owner. Nothing changes
  for you: you are the owner, and the picker works exactly as before.
- Pointing the embedding model at a network location (a `\\server\share` path, a
  Windows device path, or a URL) is refused up front instead of being handed to
  the filesystem, and the reason is logged rather than silently ignored. On
  Windows, merely testing such a path could hang localm for minutes and send your
  Windows credentials to whatever machine was named.
- The Knowledge page no longer tells a non-owner key whether the owner's chosen
  embedding file exists, and no longer shows it that file's path.
- `localm mcp`'s `setup_embeddings` tool now only accepts a known embedding key
  or a model you already registered, not an arbitrary file path, since the caller
  driving it is usually a model acting on instructions it read somewhere. Use
  `localm setup-embeddings <path>` or the GUI to point it at a GGUF of your own.
- localm now logs a warning naming the folder when the native runtime is loaded
  from somewhere other than the bundled one, so an override you did not intend is
  visible instead of silent.

- **`localm rm` no longer offers to delete a file it will only unregister.** The
  confirmation prompt picked its wording with a weaker path check than the
  deletion itself used, so a model registered from a folder that merely shares a
  name prefix with your models folder (say `<data dir>/models-old`) was announced
  as "PERMANENTLY deletes ..." when removing it would in fact only drop the name.
  Nothing was ever deleted unexpectedly, the prompt simply over-warned, but it now
  describes exactly what will happen because the prompt and the deletion consult
  the same check. A registered file that has already gone missing is now described
  as missing, rather than as living outside your models folder.
- **The standalone bug reporter shows where a report is going, and sends only over
  http(s).** The preview you confirm now names the destination host, so a reporter
  pointed somewhere unexpected is visible instead of silent. An endpoint override
  that is not `http://` or `https://` is refused and said out loud rather than
  used or quietly swapped for the built-in one; the report is saved locally and
  nothing is sent. Pointing the reporter at your own proxy over http or https
  works exactly as before.
- **Error messages and diagnostics no longer hand out your folder layout or
  account name.** The Knowledge page's embedding status used to report a failed
  model load with the full path baked into the message, so anything that could
  read that status - including a restricted key holding only the `rag` scope -
  learned your data directory and, with it, your operating-system username. The
  reason you actually need ("not an embedding model") is unchanged; only the
  directory is replaced. The same redaction now applies to the GPU-fallback
  notice and to the `/debug/stacks` hang-diagnosis dump, which additionally now
  requires a credential: it was reachable with none at all on a default keyless
  install, and returned every thread's stack including source lines and install
  paths. Full detail still goes to the debug log, where the local operator wants
  it.
- **Asking to index a folder can no longer be used to probe the server's disk.**
  Adding paths to a knowledge collection checked whether each path existed
  before checking whether you were allowed to index it, and said so in the
  error - so a restricted key could ask about any absolute path on the machine
  and read the answer off the response. Permission is now decided first, and an
  out-of-bounds path gets the same reply whether or not it is there.
- **Saved sessions and scheduled jobs get the same file protection as the key
  file on Windows.** `sessions.json` and `jobs.json` record a hash of the API key
  that created each entry, but only `auth.key` - which holds the key itself - was
  locked to your account; the two files holding the hashes inherited permissions
  that let any local account read them. All three are now restricted the same
  way. Note this protects where those hashes are STORED; how the owner key
  itself is hashed is a separate, still-open issue if you set a short or
  guessable key by hand (prefer `localm key generate`).
- **The bug-report proxy no longer echoes internal errors to callers.** An
  unexpected failure returned the raw error text, readable from any web origin;
  it now returns an opaque request id and logs the real error, with its stack,
  where the operator can see it. (Deployed separately from the app, so this
  takes effect when the proxy is next deployed.)
- **A restricted key can no longer read localm's own secrets through img2img,
  nor create folders anywhere on your disk through the media galleries.** Three
  paths a caller supplies were reaching the filesystem with no real limit. The
  img2img / image-to-video "input image" was allowed to be any file in the
  localm data directory - which is where your owner key, sessions and RAG store
  live - and that file is uploaded to ComfyUI (which may legitimately be another
  machine, over plain http) before anything checks it is an image. It is now
  limited to the uploads folder and the generated-media galleries, and anything
  without an image signature is refused before a byte leaves the machine. The
  localm install folder is not readable through it either, which matters if you
  installed with `git clone`: that folder holds more than the program.
  "Move to folder..." on an image, clip or track took the destination as given
  and created it; a key without host filesystem access is now confined to the
  data directory, while the owner keeps any folder on the machine as before.
  Exporting logs now needs host filesystem access too, matching the folder
  picker that chooses where they go.
  Migration: if you kept a reference image loose in the localm data folder, move
  it into the `uploads` subfolder (or upload it from the Settings page) and it
  will work again; images already in a gallery are unaffected.

- **Mixture-of-Experts models now use the GPU they actually fit in.** Deciding how
  many layers to put on the GPU needs to know what the context cache costs, and
  localm estimated that from the model file's SIZE. That works for an ordinary
  model but badly overstates it for a Mixture-of-Experts model, whose file is
  large because it holds many experts that cost nothing to keep context for. The
  overstatement ate the VRAM budget, so localm quietly loaded fewer layers onto
  the GPU than would have fit, and generation was slower than your card allowed
  with no message saying why. The cost is now read from the model's own attention
  shape instead: on a 35B-class MoE with 12 GB free that is 13 layers on the GPU
  where it previously managed 8. Ordinary (dense) models are also measured more
  accurately, and a model whose file cannot be read falls back to the old
  estimate, so nothing that worked before stops working.
- **NVIDIA Blackwell GPUs (RTX 50-series, and datacenter B100/B200) can now use
  the CUDA backend.** `setup-llama` always fetched the older CUDA 12.x build,
  which has no code for Blackwell's architecture, so a Blackwell card either
  failed to load the CUDA backend or crashed partway through inference - even
  though upstream already publishes a newer build that supports it. Setup now
  detects your GPU's own architecture (not just your driver) and fetches the
  CUDA 13.x build automatically when it is needed; every earlier NVIDIA GPU is
  unaffected and keeps using the same 12.x build as before. If your driver is
  not new enough for the line your GPU needs, setup falls back to Vulkan and
  explains why, exactly as it already did for an old driver.
- **The MCP `pull_model` tool now actually loads the model before saying so.**
  Its default reply promised "load it - blocks until ready", but the load step
  only registered the model as available and reserved room for it; the real
  load only happened later, lazily, on the next chat or embed call. An MCP
  client that checked whether the model was resident right after `pull_model`
  returned could find it was not, despite the "ready to use" reply. `pull_model`
  now performs the real load itself before replying, matching what it says.

- **The Settings "Live tuning" VRAM estimate could get stuck on the previous
  model.** Switching the active model from the sidebar dropdown or the Models
  page always refreshed the live estimate right away. Switching it from
  somewhere else - another browser tab, another device, the CLI, an MCP
  client - did not: the model dropdown and status line caught up within
  seconds, but the VRAM estimate kept showing the model that was active when
  Settings was opened, with no way to tell it was stale. It is now kept in
  sync the same way the dropdown always was, picked up automatically within
  about 30 seconds regardless of where the switch came from.

- **Loading a model that only partly fits your GPU no longer reports plain
  success.** A model too large for free VRAM still loads deliberately,
  offloading as many layers as fit and running the rest on CPU - but the load
  response, the GUI, and the MCP `pull_model` tool all reported the same
  "loaded" success as a full GPU load. Loading a model now reports how many
  layers actually reached the GPU whenever it is fewer than the model has:
  the API response carries the counts, the GUI sidebar and Models page warn
  instead of toasting a plain "switched", and `pull_model`'s reply says how
  many layers landed on CPU. A model that fits fully is unaffected.
- **Restarting the server could occasionally crash right after coming back
  up.** The restart button unloaded the resident model and immediately
  relaunched the server, with nothing waiting for the freed VRAM to actually
  be reclaimed before the fresh process built a new model context in it - a
  race against the still-reclaiming graphics driver. Restart now waits for
  the release the same way switching models already did, before relaunching;
  a restart with nothing loaded pays no extra delay.
- **A vision model's projector file no longer shows up as its own, unusable
  chat model.** Pulling a vision model with `--mmproj` deliberately keeps the
  companion projector file out of the model list as its own entry - it cannot
  answer chat requests on its own. But the next time the models folder was
  rescanned, it was added back in anyway, labeled the same as a regular chat
  model and offered a "use" button that could never actually work. It is now
  recognized from its own file contents and labeled as what it is.
- **A bug report's log excerpt could drop a native crash entirely, leaving no
  error visible in the report at all.** Raw native (CUDA/HIP/ggml) crash
  output, such as `CUDA error: operation not permitted when stream is
  capturing`, is appended to the debug log with no timestamp or level of its
  own, so it always attaches to whichever ordinary log line came right before
  it - usually a routine request logged every few seconds. The digest that
  builds a report's log excerpt only recognized such an attached line as an
  error when it was a Python traceback; anything else inherited the routine
  line's harmless level and could be folded away as repeated noise, taking
  the actual crash text with it. It now also recognizes crash-shaped text in
  an unmarked line and never discards a log entry that carries one.
- **A coder session's reply, in both the GUI and the CLI terminal, could keep
  showing a tool call's raw text even after the tool had actually run.** Both
  stream a model's reply live as it is generated, before the harness can know
  which parts are real tool calls; a call written in some of the accepted
  formats (for example a ```` ```json ```` block) streamed through as plain
  visible text and then ran for real once the full reply arrived, leaving the
  raw JSON on screen right alongside the record of the tool that actually
  ran. Recognized tool-call formats are now hidden from the live stream as
  they arrive, in both the GUI and the CLI, not just corrected afterward. One
  narrow shape - a call written with no wrapper at all, just a bare `{...}`
  object, the least common way a model writes one - can still flash briefly
  in the CLI terminal, which has no way to un-print text it already showed;
  the GUI corrects even that case. Deciding is incremental, not a wait for
  the whole block: a genuine ```` ```json ```` example the model shows you
  (not a call) is recognized as not-a-call within about the first twenty
  characters in the common case and streams normally from there. The one
  remaining exception is a JSON example with no `"name"` field anywhere in
  it at all - that still appears all at once once the block closes, rather
  than line by line, because nothing can rule it out until every field has
  been seen. Ordinary code fences in other languages are unaffected and
  always stream normally.
- **A HuggingFace-format model's chat reply now correctly reports when it was
  cut off by the length limit.** The `finish_reason` field in a chat
  completion always said `"stop"` for this model type, even when the reply
  was actually cut short by the max-tokens setting rather than the model
  choosing to stop on its own - GGUF models already reported this correctly.
  A client relying on `finish_reason` to detect a truncated reply (to retry
  with more room, or to warn the user) could not tell the difference here.
  It now says `"length"` when generation was cut off by the limit, and
  `"stop"` when the model produced its own end-of-reply token.

## [0.1.3] - 2026-07-23

### Added
- **The coder can hand a sub-task off and keep working instead of waiting.**
  `spawn_agent` runs its sub-agent to completion before the coder does anything
  else, so delegating a ten-turn job to a local model costs you all of that time
  staring at a spinner. The new `spawn_agent_background` starts the sub-agent and
  hands back a job id straight away; you keep working, and `check_agent_job` tells
  you how it went. A finished sub-agent's result is also folded into the
  conversation on its own at the start of a later turn, so you do not have to
  remember to ask. Background sub-agents get the same isolation as parallel
  dispatch - each works in its own git worktree on its own branch, its changes are
  committed there and never merged into your working tree, and `/diff` points at
  the branch rather than pretending the work is in your files. Two can run at
  once (the same shared ceiling parallel dispatch uses); a third is refused with a
  clear message naming the two already running, rather than quietly queued. The
  new `/bg` command lists this session's background work, both shell commands and
  sub-agents. Starting one asks for confirmation exactly like `spawn_agent` does,
  and that one approval covers everything the sub-agent then does, because a
  background sub-agent cannot come back and ask you mid-run - so in a session that
  confirms at the terminal, localm refuses to start one rather than approving on
  your behalf. A one-shot `localcoder "task"` run now warns you if it is about to
  exit while a background sub-agent is still going, instead of dropping it
  silently.
- **The coder can run two sub-tasks at once, each in its own checkout.** A new
  `dispatch_parallel` tool gives every child agent its own git worktree on its
  own branch, so two children can work on the same files without interfering and
  your working tree is never touched. Each child's work is committed to its
  branch and its diff comes back for you to review; nothing is ever merged
  automatically, because a local model resolving a merge conflict unsupervised is
  not something to do behind your back. `/diff` and `/changes` gain a clearly
  labelled "Delegated work (NOT in your working tree)" section naming the branch
  that holds each change. Capped at two children at a time, shared across every
  way the coder spawns them, matching what one GPU can actually keep resident.
  When a child needs your approval to run something, the request queues up behind
  any other child's and says WHICH child is asking - on the approval card in the
  browser as well as at the terminal - so two children's requests can never be
  mistaken for each other.
  Note the isolation is real for file tools but best-effort for shell: a child
  can still run a command that reaches outside its worktree.
- **The sidebar shows the GPU split your loaded model actually got.** With a
  multi-GPU split configured, the model status now shows each card's share
  and how it was decided - for example "Split: GPU 0 33% · GPU 1 67% (by
  free VRAM)", or "(pinned)" for manual ratios and "(equal)" when free VRAM
  could not be measured. Single-GPU setups see no change.
- **Scheduled knowledge re-sync: an indexed folder can now stay current on its
  own.** Indexing a folder into a collection records the folder itself, not
  just the files it happened to hold, so localm can re-walk it later.
  `localm rag resync NAME` does that by hand, and a new `rag` job kind puts it
  on a schedule -
  `localm job add sync-docs --rag --collection NAME --cron "0 3 * * *"`, or the
  Jobs tab's new "rag" task. A run re-indexes incrementally (new files added,
  changed files re-indexed, unchanged files skipped by content hash) and loads
  no chat model. A document whose file has vanished is FLAGGED, not deleted:
  its chunks stay searchable, the flag clears by itself if the file comes back,
  and only `--prune-missing` actually removes it - so a moved file, an
  unplugged drive, or a half-finished cloud sync can never silently delete part
  of an index. A folder that is unreachable at run time is reported and skipped
  whole, with nothing under it indexed, flagged, or pruned. Scheduled runs
  apply the same allowed/denied folder policy as an interactive add, so a
  folder that has since fallen outside it is skipped and reported rather than
  indexed. There is still no filesystem watcher, deliberately: a watcher daemon
  would break the self-contained design.
- **Search the Settings page.** A box above the section nav filters every
  group at once, matching a setting's label, its config key, or its help
  text, so a setting can be found without knowing which of the seven
  sections holds it. Matches keep their section heading, so it is still
  clear what a setting belongs to, and the rows that have no config key
  behind them (Main GPU, Split across GPUs, the logo style picker, the key
  presets) are searchable by their own text. A control that is not
  currently offered - the multi-GPU rows on a single-GPU machine, the keys
  card for a non-owner key - is deliberately not findable. Clearing the box
  restores the normal grouped view.
- **The coder can run shell commands in the background.** `run_shell` waits for
  the command to finish, so the agent could not start a dev server and then talk
  to it, or run a long build while doing anything else. Three new tools fix
  that: `run_shell_background` starts a command and returns a job id
  immediately, `check_shell_job` reports whether it is still running (plus its
  exit code and captured output once done), and `kill_shell_job` stops it. So
  the coder can now start a server, curl it, and shut it down in one session.
  Output is captured into a capped buffer, so a chatty process cannot grow
  memory without limit; anything dropped is reported rather than presented as
  the whole output. Stopping a job kills its entire process tree, so a build
  that spawned children does not leave them running, and any job still running
  is stopped when localm exits rather than being orphaned. Up to four
  background commands run at once; asking for a fifth is refused with a clear
  message instead of quietly queueing. Starting a background command is
  arbitrary code execution and stopping one tears down a process tree, so both
  ask for confirmation exactly like `run_shell` does and get the same privacy
  handling; checking on a job only reads its status and output, so it does not
  prompt (otherwise every poll of a long build would need approval). All three
  are unavailable to shareable (restricted) coder sessions, and turning
  `run_shell` off turns them off too.
- **The coder's past lessons no longer expire by age alone, and nothing it
  forgets is gone for good.** Episodic memory used to be a plain queue: at 200
  stored lessons the oldest was discarded, however useful it was, with no
  notice and no way back. Now a new lesson that merely restates one you already
  have is merged into it (keeping both file lists and the better wording)
  instead of being stored twice, and when the store is full the LEAST USEFUL
  lesson goes rather than the oldest, so a hard-won "this is what went wrong
  and why" survives a run of throwaway one-liners. Everything dropped is
  archived first: `localcoder --episodes-archive` lists it and
  `localcoder --restore-episode ID` puts it back. `localcoder --episodes` now
  shows each lesson's id, and `--forget-episode ID` removes just that one
  instead of wiping the project's whole history (`--forget-episodes` still
  erases everything, archive included).
- **You can see which past lessons the coder is acting on.** When a session
  recalls lessons from earlier work, the GUI now says which ones it pulled in
  and shows the id for each, and the same list is written to the session audit
  log. A lesson that sends a run down the wrong path used to be invisible after
  the fact; now you can see it and remove it by id.
- **Optional: let the model merge related lessons into one**
  (`localcoder --consolidate-episodes`). Strictly opt-in and manual - it never
  runs on a timer or at session close - and it reports exactly what it merged.
  The originals are archived, so any merge you dislike is reversible with
  `--restore-episode`, and a group the model cannot summarize usefully is left
  untouched rather than lost.
- **Coder sub-agents can now be given a role, which narrows what they can
  do.** `spawn_agent` takes an optional `role`: `reviewer` (read the code and
  inspect git, change nothing), `researcher` (read-only investigation of the
  project, no writes and no network), or `test-writer` (read, write tests, and
  run them, but no shell and no commit or push). The role gives the sub-agent a
  focused brief and, more importantly, actually takes the other tools away, so a
  helper spawned to review a diff can no longer overwrite files, run shell
  commands, or push. A role can only ever REMOVE capability: it is applied on
  top of whatever the parent session already forbids, so it can never hand back
  a tool you disabled, and it never lets a shared, restricted session regain
  execution. Tools registered by MCP servers, plugins, and skills are excluded
  from a role by default rather than inherited. Omitting `role` keeps the
  previous behavior (the sub-agent gets the parent's full toolset).
- **Coder: `edit_files` applies one exact-string edit across several files at
  once, all-or-nothing.** The coder could already replace an exact snippet in a
  single file (`edit_file`) or run a regex substitution across many
  (`search_replace`), but not the common middle case: change this exact text in
  these five files. `edit_files` takes a list of `{path, old, new}` edits, checks
  every one before writing anything, and if any edit fails - a path outside the
  project, a missing file, text that does not match - it reports which one and
  why and leaves every file exactly as it was. If a write fails partway through,
  the files already written are restored from snapshots taken before the batch;
  should that restore itself fail, the result says so rather than claiming a
  clean rollback. Edits keep `edit_file`'s behaviour otherwise: first occurrence
  only, matched exactly, with the closest-match hint on a miss and the post-write
  syntax check on each file. It honours an active `--scope`, is undoable, and is
  captured (not written to disk) in patch mode.
- **The MCP server keeps several models loaded at once, like the HTTP server
  already did.** `localm mcp` used to hold exactly one model: asking it for a
  second unloaded the first, so alternating between two models reloaded them
  every time. It now keeps both resident when a live free-VRAM reading shows
  the second one fits (its estimated need plus a 1 GB headroom, and enough
  room on every split device when a GPU split is configured), and only evicts
  when it genuinely does not fit - least-recently-used first, never a model
  that is currently generating. On a machine where free VRAM cannot be
  measured, or where the reading is inconclusive, it stays single-resident
  exactly as before rather than stacking models until the graphics driver runs
  out. The HTTP server and the MCP server now share one implementation of this
  decision, so they cannot drift apart. Shutting the server down frees every
  model it still has loaded, and says so if one fails to free.
- **Two optional knobs to decide model residency yourself**, for when you would
  rather say "keep these loaded" than rely on free-VRAM arithmetic:
  `localm config max_resident_models 2` caps how many models stay loaded at
  once (1 restores strict single-resident), and
  `localm config pinned_models a,b` protects named models from being evicted to
  make room for another. Both are off by default, so nothing changes unless you
  set them, and clearing either (`localm config max_resident_models ""`) goes
  back to the automatic behavior.
- **The coder can now keep a task list that outlives its own memory.** Two new
  tools, `set_todos` and `read_todos`, let the model write down the plan for a
  multi-step job ("[x] done", "[>] working on it", "[ ] not started"). The list
  is kept outside the conversation, so it is not lost when a long session
  compacts its history, and it is saved with the session so a paused job
  resumes with its plan intact instead of starting over. In the GUI the tool
  card shows progress and the current step at a glance. Privacy mode keeps the
  list in memory only, like everything else about the session.
- **The coder verifies its work by exit code in interactive sessions too, and
  finds the check itself.** Judging a change by running a command and reading
  its exit code - the harness runs it, not the model, so the model cannot talk
  its way to "done" - used to happen only for a one-shot task with `--until`.
  The REPL and the GUI coder now run the same check at the point the agent
  would otherwise finish a turn that changed files. The command no longer has
  to be typed: localm detects the project's obvious one (`cargo test`,
  `go test ./...`, `npm test` when package.json actually defines a test script,
  or pytest when the project has a pytest setup) and runs without one in a
  project where no check can be found, rather than guessing. Override it with
  `--verify COMMAND`, a `verify = "..."` key in `.localcoder/config.toml`, or
  `/verify` mid-session; `--no-verify` / `/verify off` disables it. A failing
  check is fed back for a fix (with the standing instruction not to edit the
  check to force a pass); when the attempts run out the turn is reported as NOT
  verified instead of as finished. Because a real check now runs, the older
  "verify your work" nudge names that command instead of asking the model to
  re-read its own edits. Sessions opened with a shared, scoped key never run a
  verify command: they have no process execution at all, by design.
- **`localm coder --seed N` for reproducible runs.** Pins the sampler's RNG so
  the same seed, model, prompt and settings reproduce the same output, which is
  what makes it possible to compare two harness or prompt changes without the
  model's own randomness in the way. Measured bit-for-bit on one AMD gfx1030
  box (bundled llama.cpp, Qwen2.5-Coder-7B Q6_K): 5/5
  identical responses with a fixed seed at temperature 0.8, 5/5 different
  without one, and identical again after a full model reload. Different
  hardware, backends, llama.cpp builds and concurrent load were not measured,
  so this is a measurement, not a cross-machine guarantee; `--anthropic`
  ignores the flag and says so, because that API has no seed parameter. A
  `seed = N` key in `.localcoder/config.toml` sets it per project.
- **Multi-GPU split: each card's share is now sized automatically from its
  free VRAM.** With "Split across GPUs" enabled and no manual
  `gpu_split_ratios` pinned, localm no longer divides the model equally: at
  load time it reads every split device's free VRAM and distributes the
  model proportionally, so a card already half-occupied (another model,
  another app) gets a half-sized share instead of an equal one that would
  not fit. Loads that used to be refused with "Not enough VRAM on the
  configured split device(s)" - because one busy card could not hold an
  equal share even though the model fit the cards' combined free space -
  now load with an adapted split; when even the combined space is short,
  the load falls back to partial offload (some layers on CPU) instead of
  refusing, matching single-GPU behavior. On the Vulkan runtime the
  readings come from the runtime's own device registry, so the automatic
  distribution works there too. Explicit `gpu_split_ratios` values are
  honored exactly as before (pinning ratios opts out of the automatic
  distribution), and when per-device free VRAM cannot be measured the
  split falls back to the previous equal shares; the chosen distribution
  is logged either way.

### Changed
- **Coder: `grep` is much faster on real repositories, and its limits are now
  settings.** It streams files line by line instead of reading each one whole,
  and it no longer reads what it cannot use: files under noise directories
  (`.git`, `node_modules`, `__pycache__`, `.venv`, ...), binaries, and files
  above a size cap are skipped before being read. On a 4103-file / 50.6 MB test
  repository a default search went from 6.90s to 0.56s and from 123 MB to 5.7 MB
  of peak memory. Every skip is reported in the result, with the reason and the
  setting that changes it, so a narrower search never looks like a complete one;
  searching a skipped directory on purpose still works by naming it in `path=`
  or `glob=`. The matches-per-file cap (20), the output-line cap (300), and the
  new file-size cap (4 MB) are now the `coder_grep_max_per_file`,
  `coder_grep_max_output_lines`, and `coder_grep_max_file_bytes` settings, each
  overridable per search; 0 means no cap. Line numbers now count line feeds
  only, matching what an editor shows, so a file containing form feeds no longer
  reports shifted numbers - in such a file `^`/`$` anchors now also treat only
  line feeds as line boundaries.
- **NVIDIA on Windows now recommends CUDA.** The setup menu's default backend for
  an NVIDIA GPU on Windows is now `cuda` (peak performance) rather than Vulkan: it
  fetches a self-contained CUDA runtime (no CUDA Toolkit needed) and falls back to
  Vulkan automatically if your driver is too old. Vulkan is still one keypress
  away in the menu, and stays the default for Intel GPUs and for NVIDIA/AMD on
  Linux (where the CUDA build needs a system CUDA runtime).
- **The coder's project memory and user instructions are now bounded.**
  `LOCALCODER.md` (project memory) and `.localcoder/system.md` (user
  instructions) are injected into the coder's system prompt on every turn, and
  both were previously injected whole with no limit, so a file that grew over
  time could crowd out the repo map and the conversation itself. Each is now
  capped at 3000 characters, the same budget the repo map already uses. Normal
  files are unaffected and injected verbatim. Going over is never silent: the
  agent tells you which file was over budget and by how many characters, and the
  prompt itself carries a note so the model knows it is reading a partial file.
  The same cap applies to a `--system` string. A memory file that exists but
  cannot be read is now also reported instead of silently ignored.
- **Corrected the docs on who writes `LOCALCODER.md`.** The CLI docs described it
  as auto-managed project memory that the agent appends to via `/remember` and
  its own reflection. The agent has no tool that writes it: the file changes only
  when you run `/remember` or `/forget` (or edit it yourself), and the agent's
  close-time reflection is stored in the localm data directory, not in your repo.
- **Loading a model could print a scary crash trace even though nothing was
  wrong.** On some Windows + AMD setups, the VRAM check that runs right after a
  model loads could collide with a native library already in memory and print a
  full "Windows fatal exception" stack dump to the console. The load itself
  always succeeded and this was already being caught safely, so nothing was
  actually broken - but the trace looked like a crash. That check now recognizes
  the setup ahead of time and skips it instead of triggering and catching it, so
  the trace no longer appears.

### Fixed
- **The live VRAM meter in the status bar now shows used/total on a GGUF-only
  install, instead of only the card's total.** On a build without the optional
  torch backend, nothing could enumerate the GPU for the status readout (torch
  is absent, and the `nvidia-smi` fallback sees only NVIDIA cards), so the meter
  fell back to the display driver's registry entry, which reports total VRAM but
  no live usage, and showed a bare "VRAM 16.0 GB". It now recovers a
  whole-board free figure from the GPU's own usage counter (AMD's ADL, with the
  operating system's per-adapter WDDM counter as a vendor-neutral fallback), so
  the meter shows live "VRAM X / 16.0 GB". The recovery only applies when the
  pairing is unambiguous (a single AMD adapter for an AMD card, or a single
  reporting adapter instance); a multi-GPU or unrecognised box keeps the
  total-only reading rather than risk attributing the wrong adapter's usage, and
  a probe that is still warming up or has timed out is left untouched so a
  pre-load fit check never acts on a number the driver had not settled.
- **`localm doctor` no longer reports an installed-but-broken package as
  simply "not installed", and no longer goes quiet about the breakage.**
  Doctor shows each key dependency with its version, reading that version from
  the installed package's metadata and falling back to the package's own
  `__version__` when no metadata is present. For a package that resolves its
  attributes lazily (transformers does), asking for `__version__` can itself
  raise a missing-module error, and that error was indistinguishable from the
  package failing to import at all. So a transformers that imported perfectly
  well but was internally broken got listed as "not installed" - and because
  the deeper "is the HF backend actually usable" check only runs when both
  transformers and torch are present, it then skipped silently, saying nothing
  about the fault it exists to report. A version that cannot be read is now
  just a missing version number: the package is still reported as present, and
  the usability check still runs and still names the real root cause.
- **Picking CUDA on a machine `setup-llama` already knows is AMD (or another
  non-NVIDIA vendor) no longer offers only a generic "no NVIDIA driver
  detected" message with Vulkan as the sole fallback.** The vendor mismatch
  was already detected and reported once ("Heads up: picked cuda but detected
  amd"), but that information never reached the CUDA driver-preflight dialogue
  a moment later, which only knew "no NVIDIA GPU" - not what actually IS
  present - and offered a binary continue-or-Vulkan choice regardless of
  hardware. It now names the vendor that was actually detected, recommends
  the real match for it (the same policy `setup.bat`/`setup.sh` use - e.g. the
  self-contained ROCm build for AMD on Windows, not a hardcoded Vulkan), and
  offers a genuine three-way choice: continue with CUDA anyway, switch to the
  recommendation, or quit.
- **A blocked or corrupted `setup-llama` download now says specifically what
  went wrong, instead of naming every possible cause at once.** A downloaded
  runtime archive that turns out too small or not a valid archive used to get
  one generic hedge ("almost certainly an error page or a truncated
  transfer") regardless of which of those it actually was. Confirmed live: a
  corporate network's content filter substituted a small HTML page for a real
  32 MB archive, and there was no way to tell that apart from a merely flaky
  connection from the message alone. The downloaded bytes are now inspected
  directly (does it look like a webpage, a JSON/XML error response, or a real
  archive that was simply cut short?) together with the response's actual
  Content-Type, Content-Length, and final URL after redirects, and the error
  names the specific, evidence-backed cause - or says plainly that the cause
  is not clear, rather than guessing. `setup-llama` also now suggests the
  existing `--from`/`--url` escape hatches when an explicitly-chosen vulkan or
  cpu backend fails to provision, which it previously did not mention at all.
- **The coder's file edits no longer fail just because a snippet's whitespace
  differs from the file.** The `edit_file` and `edit_files` tools matched the
  text to replace byte for byte, so when the model reconstructed a snippet with
  a slightly different indent, a wrapped line collapsed onto one line, or an
  extra trailing space, the edit missed and the model was told to try again,
  sometimes never landing the change at all. The match now falls back to a
  whitespace-tolerant comparison (a run of spaces, tabs, or newlines in the
  snippet matches any run of whitespace in the file), applied ONLY when it
  resolves to a single region so it can never silently edit the wrong one of two
  candidates. An exact match still wins outright, and a genuine miss still shows
  the closest-match hint.
- **Setup no longer fails with a certificate error on a network that intercepts
  HTTPS (a corporate proxy or security product).** `uv`'s default certificate
  verification (its own bundled root list) and localm's own previous certifi-based
  verification for setup-llama, `localm update`, the issues list, and bug-report
  uploads all rejected a TLS-intercepting proxy's re-signed certificate, since
  neither bundle can know about a private, locally-issued root - even though a
  browser on the same machine trusts it fine, because IT already provisions that
  root into the operating system's own certificate store. Every outbound download
  in setup and in the app now verifies against the platform's native certificate
  store first (the same trust a browser already has), falling back to a bundled
  root list only if that specific verification fails - which still covers the
  original case this replaces: a freshly-imaged Windows box whose certificate
  store has not yet cached a legitimate CA. The fallback (when it is ever needed)
  is silent: setup tries the download once, unseen, before ever showing anything.
- **Picking "Portable" during setup no longer silently reuses an already-installed
  `uv`.** Both installers checked only whether any `uv` was reachable on PATH
  before deciding whether to confine it to the clone - not whether it was
  Portable's OWN, previously-confined copy. A `uv` left on PATH by a Shared
  install, a package manager, or even a different clone (Astral's installer adds
  its target directory to the persistent user PATH, so one portable clone's `uv`
  could leak into another's setup) was silently reused, skipping the "install uv
  into this folder" step the Portable prompt promises. Portable now looks
  specifically for its own confined copy and only falls back to the normal
  install-and-confine flow when that copy does not exist yet; Shared mode is
  unaffected.
- **A second localm server sharing the same data directory no longer reports
  the first, healthy one as crashed.** Running more than one server against
  the same data directory is a normal, supported thing to do (`localm ps`,
  `serve --project`, the coder plugin starting its own backing server), but the
  crash-recovery marker used to be a single file per data directory with no
  notion of which server it belonged to. A second instance starting up would
  find the first, perfectly healthy instance's marker, conclude it had died
  hard, and file a bug report about a crash that never happened - and that
  second instance's own later, clean shutdown would then delete whatever
  marker happened to be sitting there, which could by then belong to a third,
  still-running instance, silencing a real crash of that instance for good.
  Each running server now gets its own marker, and a marker is only ever
  treated as evidence of a crash once the process id it recorded is confirmed
  to no longer be running.
- **A chat send that fails before any reply text arrives no longer vanishes
  without a trace.** When the very first request of a send failed - a 400
  because no model was selected, a dropped connection, anything before the
  first streamed token - the error was rendered into the chat, but a guard
  written for a narrower case (a vision-only model rejecting an attached
  image) treated any reply with no content as that same case and immediately
  redrew the chat pane from saved history, which had never seen the failed
  turn. The rendered error was wiped a moment later, leaving only a toast that
  disappears after a few seconds - so a real failure could look like the
  message went nowhere. The error now stays on screen; the vision-reject
  recovery (drop the image, keep chatting) is unchanged. Separately, the chat
  request now falls back to the model actually loaded if the model dropdown is
  ever empty or out of sync, instead of sending a blank model name and letting
  the server reject it.
- **A sub-task the coder gave up on can no longer report back that it succeeded.**
  When `dispatch_parallel` runs sub-tasks side by side and one of them overruns the
  time budget, that sub-task is abandoned - it cannot be stopped, only left to run
  on. It was still able to write its own result afterwards, on top of the verdict
  already recorded for it, so a sub-task that had been given up on could be listed
  as `[ok]` moments later: counted as finished, offered for you to merge, and
  reported as a success to the model that delegated it. The overrun verdict is now
  final, and a late result is refused rather than applied. Nothing is hidden by
  that refusal - the report says the sub-task did eventually finish, how long after
  the deadline, and that whatever it wrote is sitting uncommitted in the working
  copy left behind for it, so you can still go and look. A sub-task that finishes
  within its budget is unaffected and still reports normally.
- **The coder no longer looks a file up in order to refuse it.** With a `--scope`
  set, the check that confines the file tools had the same habit as the shell
  warning below. When a model asked to read or write an absolute path that was
  not inside your working directory, the coder resolved that path - asking the
  disk about that exact file, anywhere on the machine - purely to establish that
  it was out of scope and say no. The refusal was correct; reaching out to make
  it was not, because at the moment of access a routine check and a command gone
  wrong look the same. The decision is now made from the text of the path alone,
  and nothing is touched. Refusals only get stricter, never looser: the single
  case that changes is an absolute path that points outside your working
  directory but leads back inside it through a symlink, which is now refused
  rather than allowed. Traversal out of the working directory is unaffected and
  is still caught when a tool actually runs.
- **The coder no longer touches a file just because a model named it.** With a
  `--scope` set, the coder warns you when a shell command it is about to run
  mentions a path outside that scope. Deciding which words in the command were
  paths involved asking the disk whether each one existed, and for a path written
  with a drive letter or from the filesystem root, that question was asked about
  that exact file, anywhere on the machine. So a model that merely proposed a
  command naming one of your files, or a system file, made the coder look that
  file up - before you had confirmed anything and before the command ran. The
  check now reads only the text of the command and touches nothing at all. It
  still names any path reaching outside your working directory, which is what the
  warning is for; the cost is that a plain relative name that happens to exist
  (`cat notes.txt`) is no longer flagged, because separating those from ordinary
  command words (`npm test` in a project with a `test` folder) is exactly what
  needed the lookup. Nothing that was refused before is allowed now: this has
  always been a warning that never blocked anything, and the scope confinement on
  the file tools is unchanged.
- **The coder no longer reports finished work as failed because its own check
  could not start.** When the coder picks a project's test command for you, it
  now confirms that command can actually run before making it the gate. On
  Windows it could not: npm and yarn ship as `.CMD` shims, which the way the
  coder launches a command cannot start, so any project with a `package.json`
  test script (localm's own repository included) failed verification on every
  turn that changed a file. The coder was then told to fix a defect that was
  never in the code, burned its retries on it, and ended with "this task is NOT
  verified" over work that was fine. The command is now resolved to the runner's
  real location, so it runs; and when a runner genuinely is not installed, no
  check is set up at all rather than one that can only fail. The same applies to
  a Rust project without cargo, a Go project without go, and a Python project
  whose interpreter has no pytest.
- **"The check could not run" is no longer reported as either a pass or a
  failure.** A command that never started is now treated as inconclusive, like a
  test run that collects nothing: the coder says plainly that nothing was
  verified instead of blaming the model, and it does not spend fix attempts on
  it. It is still never called a success. This is decided from what actually
  happened when the command was launched, not from its exit code, so a check
  that ran and failed because something it calls was missing (a script that is
  not there yet, an uninstalled dependency) is still reported as the failure it
  is, and the coder still gets its attempts to fix it. Sessions in the app now
  also carry that third answer, so a finished task whose check never ran is
  labelled "not verified" rather than reading as a clean finish.
- **A background job localm could not stop is no longer left behind in silence.**
  Anything the coder started with `run_shell_background` is stopped when localm
  exits rather than being orphaned, but a process that refuses to die (one stuck
  in uninterruptible I/O, or running at a higher integrity level than localm) was
  counted as stopped anyway and nothing was printed, so the promise held in the
  report and not on the machine. Exit now names any job it could not stop, and
  says up front that it is stopping background jobs, so the pause while it tries
  no longer looks like a hang.
- **A finished background sub-agent's result is no longer discarded because
  unrelated shell commands finished.** localm keeps a bounded list of completed
  background work, and background sub-agents and background shell commands shared
  one budget. Shell results are kept until you ask for them, so in a long session
  they filled that budget and pushed out a sub-agent's finished result before the
  coder collected it, losing its summary, its branch and its diff for good. Each
  kind of background work now has its own budget, so one can never crowd out the
  other, and if a result is ever dropped uncollected, the coder and `/bg` say so
  instead of leaving it looking like nothing had finished.
- **Stopping a background job now says what it could not confirm.** Killing a job
  is meant to take down its whole process tree, but only the job's own process was
  ever checked, so a build tool or dev server that passed the signal to a child
  which then ignored it was reported as fully stopped while that child kept
  running and holding its port. Survivors are now detected and stopped, and
  anything still alive is named in the result; on an install without psutil,
  where they cannot be detected at all, the result says the tree could not be
  verified rather than implying it is clean. On Windows, a failure of the
  underlying `taskkill` was discarded entirely; it is now reported and falls
  through to the backup kill, matching what the other platforms already did.
  Checking a job also no longer reports a job as running and as failed at the same
  time when it happens to finish mid-check.
- **Running two coder sub-tasks at once never actually worked.** The coder
  offers `dispatch_parallel`, a tool that runs up to two sub-tasks side by side,
  each in its own isolated checkout. It was listed in the model's toolbox and it
  asked you to confirm before running, but the call then failed every single
  time, because the running session was never handed to it. Nothing was ever
  dispatched. It works now, and a new check runs every tool the coder offers
  through the real dispatch path, so a tool can no longer ship advertised but
  dead. Fixing it also brought several problems in that same path within reach,
  all addressed here: two malformed calls could permanently use up the machine's
  budget for sub-agents, leaving every later sub-task in that session refused
  with only a generic error; a sub-task that ran out of turns or gave up was
  reported to you, and to the model, as having succeeded (the same was true of
  background sub-agents, and it meant the coder never learned from the failure);
  a sub-task that never got a turn to start was described as having run for ten
  minutes and timed out, and its checkout was left behind on disk; a sub-task
  abandoned for running too long gave its slot back while it was still running,
  so further sub-agents could pile onto an already busy machine; and a tool that
  is meant to run on its own could start while a slow one from the same batch
  was still going.
- **The coder no longer discards your own git worktrees while cleaning up after
  its sub-tasks.** When a sub-task finished, the coder ran a repo-wide `git
  worktree prune`. That command cannot be limited to particular worktrees: it
  drops the record of every registered worktree whose folder is missing at that
  moment, including one of yours that simply lives on an external drive or
  network share you have not mounted. Recovering from that needs `git worktree
  repair`. Cleanup is now limited to the coder's own worktrees, and it skips the
  operation entirely, and says so, whenever anything else would be caught by it.
- **The coder's forgotten-lessons archive no longer reports "nothing here" when
  it simply could not be read.** `localm coder --episodes-archive` printed "No
  dropped episodes archived for this project", and `--restore-episode ID` printed
  "No archived episode with id ID", in two very different situations: nothing was
  ever archived, or the archive exists and could not be opened (another process
  holding the file). The second now says plainly that the archive could not be
  read and the answer would be incomplete, so a recoverable lesson is never
  written off as absent. Restoring a lesson while the archive is unreadable also
  no longer rewrites that archive from the failed read, which would have thrown
  away every remaining recovery copy.
- **Restoring a coder lesson into a full store no longer destroys it.** When
  episodic memory was at its cap, `localm coder --restore-episode ID` could bring
  a lesson back, immediately drop it again because it still ranked lowest, and
  then delete the fresh recovery copy along with the old one - leaving the lesson
  in neither the stored list nor the recoverable archive, gone for good, while the
  command reported "Restored episode ID". The recovery copy is now kept, so the
  lesson stays restorable, and the command says plainly when the store is full and
  the lesson was dropped again rather than claiming an unqualified success.
- **Two coder sessions finishing at once can no longer lose a lesson outright.**
  Recording, forgetting, restoring and consolidating episodic memory each read the
  whole per-project log, changed it, and wrote it back with nothing serialising
  them, so two writers on the same project (two sessions closing together, or a
  `--forget-episode` / `--consolidate-episodes` run alongside a session closing)
  could overwrite each other. A lesson lost that way was gone for good: it never
  passed through the step that files a recovery copy, which is what normally makes
  every dropped lesson restorable. Writes to one project are now serialised, and
  consolidation merges its result onto the current state rather than the snapshot
  it started from, so a lesson recorded while the model was thinking survives.
- **A broken vector index is no longer erased by the next successful indexing
  run.** localm keeps a vector index it has refused to use, and keeps saying the
  collection is degraded, so the problem cannot disappear unnoticed. That held
  only while nothing new was embedded: as soon as one document was re-indexed
  with embeddings on - the normal state of a scheduled re-sync - the new file
  overwrote the refused one and the warning stopped, leaving a collection that
  reported perfect health while most of its documents had quietly lost their
  vectors. The refused index is now set aside before anything can overwrite it,
  every incident keeps its own copy instead of the second one silently replacing
  the first, and the warning now clears only when a rebuild actually covers every
  document (`localm rag repair NAME --embed`), not when a partial re-index papers
  over it. Emptying a collection clears it too, since there is then nothing left
  for those vectors to belong to - previously that left a warning behind that no
  rebuild could ever clear.
- **A hand-run `localm rag` command and a scheduled re-sync can no longer lose
  each other's work.** Writes to a knowledge collection were serialised only
  within one localm process, so `localm rag add|resync|repair|rm` in a terminal
  could overlap the server's own indexing of the *same* collection: both read the
  index, both wrote it back, and one of the two updates vanished (with the
  leftovers occasionally surfacing later as a degraded vector index). Collections
  are now locked across processes as well. A second writer waits for the
  collection, tells you it is waiting, and then stands down with a message naming
  the process that holds it rather than writing anyway - a refused command has
  changed nothing, a refused API call answers 409, and a scheduled job says so in
  its output and picks the folder up on its next run. Holding a collection has no
  time limit, so indexing a large folder for hours is safe; the holder reports in
  while it works, and only a holder that stops reporting (a crash, a killed
  process, a machine that lost power) has its lock reclaimed, about a minute
  later. Nothing changes for the common case of a single writer.
- **The text-to-speech voice you picked never reached the server.** The voice
  picker in the chat parameters saved your choice in the browser only, while
  the tts plugin's own settings (voice, speaking speed, voice model, compute
  device, model precision) lived in a server-side block that nothing could
  write: they could only be changed by hand-editing `config.json`. Picking a
  voice therefore looked like it changed the setting for good and did not.
  Settings now has a "Text-to-speech" section that edits those server-side
  values for real, with validation (an unknown voice, an out-of-range speed or
  a model that is not a Hugging Face repo id are refused with a clear message
  instead of silently falling back). The two stores are now clearly separate
  and both visible: the server value is the default for every browser, the chat
  picker is labelled "this browser", and when this browser has its own pick the
  Settings section says so and offers to clear it. Changing the default voice or
  speed takes effect right away in the browser you saved from (other open
  browsers pick it up on their next load), and a voice remembered in your
  browser that the plugin no longer offers is now ignored rather than handed to
  the voice model (which failed at playback time).
- **A knowledge collection's vector index is never deleted to hide a problem.**
  When localm finds stored embeddings it cannot trust - unreadable, malformed, or
  no longer lining up with the documents - it answers lexically instead and says
  why. It then used to DELETE that file on the next indexing run, including a run
  that indexed nothing at all, so the evidence disappeared with it and the
  collection went on to look like a perfectly healthy lexical-only one. The file
  is now kept where it is, or moved aside as `vectors.json.rejected` when the
  documents around it had to be rewritten, and the reason is repeated by the
  Knowledge page, by `localm rag resync` and by every scheduled re-sync until you
  rebuild the index with `localm rag repair NAME --embed`. Search results are
  unaffected: a collection in that state has always answered lexically.
- **Three media settings were invisible in the GUI.** "ComfyUI launch
  timeout", "Keep ComfyUI headless" (no auto-opened browser tab) and the
  ACE-Step `__func__` crash fix toggle existed as real settings - described,
  validated, changeable from the CLI - but the Settings page never showed
  them anywhere: the Media section only rendered its per-plugin boxes plus
  two hand-picked fields. They now appear in a "Shared" box at the top of
  Settings > Media, and the section renders every media setting by default
  rather than from a hand-maintained list, so a future one cannot silently
  vanish the same way. Also makes the shared `comfy_float_type` fallback (the
  documented default behind the per-plugin "Model weight dtype" setting) a
  real, settable key - previously it could only be created by hand-editing
  config.json.
- **The coder's "do not fail me for having no tests yet" flag now actually
  reaches your test runner.** When the coder picks `npm test` as a project's
  check, it adds `--passWithNoTests` so that a project whose suite is still
  empty is not reported as a failed verification. npm never passed that flag
  along: it reads an option it does not recognise as one of its own settings and
  forwards only plain arguments to your `test` script, so the runner never saw
  it. Against real jest in a project with no test files, the check failed
  exactly as if the flag had never been there. It now goes through the `--`
  separator npm documents for this, so it arrives. yarn is deliberately left
  alone: it already forwarded the flag correctly, and it warns that a future
  version will pass an explicit `--` straight through to your runner, which
  would break a case that works today. The flag is also only sent to a runner
  that has it (jest, vitest); any other project gets a plain `npm test`, because
  a runner that does not know the option can stop outright rather than ignore it
  (`node --test` exits with "bad option"). On Windows none of this was visible
  until the previous fix, because the command could not start at all.
- **Extra arguments you give the coder's test tool now reach an npm project's
  runner too.** Asking it to run the tests with an option of your own - a
  `--watch`, a `--reporter`, a `-t` filter - appended that option to `npm test`,
  where npm read it as one of its own settings and dropped it before your `test`
  script ran. The coder then ran a plain, unfiltered suite and reported it as the
  run that was asked for, with nothing to say the option had been discarded. Such
  arguments now go through the same `--` separator, so they arrive. This is the
  npm behaviour behind the fix above, on the other path into the tool: that one
  covered the flag the coder adds for you, this one covers the arguments you ask
  for yourself. yarn is again deliberately left alone, and pytest, cargo and go
  were never affected.

- **The MCP server's `run_coder_task` and `run_doctor` now always act on the
  server's own install and data.** Their helper processes re-resolved the
  localm data directory (and even which localm code to run) from the working
  directory at every step, and a coder task deliberately runs in YOUR
  project's directory - so when the MCP server itself ran from a source
  checkout (its data home coming from its own folder), the coder chain
  looked in a different, empty data home: a model you had just pulled came
  back "Model not found", and the real error was printed into a console
  window an MCP client never sees. The server now pins its own data home and
  code onto every helper it starts. Relatedly, when the coder auto-starts a
  background server for a caller without a terminal (an MCP client, CI, a
  script), the server's output is now captured and its actual error shown on
  failure, instead of opening a console window nobody can look at.
- **Creating a memory job over the HTTP API no longer needs a dummy prompt.**
  The docs said the `memory` task kind needs no prompt, and the CLI's
  `--memory` flag honoured that, but `POST /api/jobs` still rejected a body
  without a `prompt` field. The prompt requirement now lives in one place (the
  job definition), which accepts a promptless `memory` or `rag` job and still
  refuses a promptless chat or coder one.

- **A coder sub-agent now inherits the scope you set, and two of them can no
  longer run at once.** `spawn_agent` had two independent gaps. It was not
  marked destructive, so when the model asked for several sub-agents in one
  reply they ran in parallel in the same working directory (each free to write
  over the other's files), skipped the confirmation prompt every other
  write/shell tool goes through, and were subject to a batch timeout that
  abandoned a slow child while it was still writing. And a session started with
  `--scope` passed everything to its children except the scope, so a sub-agent
  could read and write files the parent itself was blocked from touching.
  Sub-agents now run one at a time, are confirmed like any other destructive
  action, and are confined to the same scope as the session that spawned them.
- **A self-review that crashed no longer looks like a clean approval.** With
  the optional pre-done review enabled (`coder_review`), a reviewer that failed
  to run - the model unreachable, or its reply unusable - was silently treated
  the same as a review that ran and found nothing: the agent finished and said
  nothing. The run still proceeds (a flaky reviewer must never cost you your
  answer), but it now says plainly that the review did not run, why, and that
  the changes went out unchecked, in the console, the GUI event stream, and the
  session log.
- **`--scope` now tells you that it does not confine shell commands.** The
  scope glob restricts every file tool, but `run_shell` and `run_tests` start a
  real process, which no path check can confine - so a command could always
  read and write outside the scope. That was true before and is still true; it
  was just invisible. A scoped session with shell tools enabled now says so
  once at startup, and a shell command that names a path outside the scope
  prints a best-effort warning before it runs. Nothing is blocked: disable the
  shell tools if you need the scope to be a hard boundary.
- **The coder no longer puts your absolute project path in the model's
  prompt.** The prompt deliberately shows the working directory home-anchored
  (`~/projects/app`) so the absolute machine path and your OS username stay out
  of it, but the codebase map printed immediately below it still carried the raw
  absolute root, undoing that in the same prompt. The map header is now
  home-anchored too, so a model that echoes its context back cannot disclose the
  path.
- **The shell scope warning no longer fires on ordinary commands.** Its
  path-spotting read two things as paths that never are. Any argument with a
  colon in second position counted as a Windows drive path, so an ffmpeg
  offset (`-ss 5:30`), an aspect ratio (`4:3`) and a `sed s:old:new:`
  delimiter were all reported as being outside the scope. And a command word
  that happens to match a directory name counted as a path argument, so `npm
  test` in a repo with a `test/` folder, `make docs` with `docs/`, and `cargo
  build` with `build/` each drew a warning about a path that was never
  referenced. A drive letter must now be a single letter followed by a
  separator or nothing, and the program and subcommand words of a command line
  are no longer guessed at from directory names. Real references are untouched:
  an explicitly written path is still reported everywhere, including in the
  program position (`./build/run.sh`), as is any existing out-of-scope file
  named as an argument.
- **Mid-chat context growth now sees real free VRAM on Windows + AMD.** When a
  long conversation grows the context window mid-generation, the model worker
  decides whether the grown KV cache still fits in VRAM or must move to system
  RAM. On Windows with the AMD (HIP) runtime that decision was broken twice
  over: the worker's VRAM probe helper was started with the wrong interpreter
  and always failed (so the check silently saw "unmeasurable" and always chose
  VRAM), and even the intended reading counts only the calling process's own
  VRAM - blind to the model itself and to every other app - on this platform.
  The probe now starts correctly, and the worker applies the same whole-board
  correction the rest of localm already uses (introduced for load sizing), so
  a grow that genuinely does not fit moves the KV cache to system RAM instead
  of overcommitting VRAM and collapsing generation speed. Failures along this
  path are now named in the debug log instead of being silently swallowed.
- **No more repeated "Windows fatal exception" traces when the native runtime
  and a ROCm torch share a process.** On Windows with the AMD ROCm install,
  once the bundled HIP llama.cpp runtime was loaded into a process, every GPU
  listing in that same process printed a scary `Windows fatal exception: code
  0xc0000139` trace to the console: the two runtimes' same-named DLLs cannot
  coexist, so the lister's torch attempt failed identically on every call, and
  each retry printed a fresh trace. The lister now recognizes that exact
  combination ahead of time and skips straight to its working fallbacks (the
  skip is logged in debug mode). Every other setup keeps torch enumeration
  exactly as before.
- **The GUI can now set up a multi-GPU split on the Vulkan runtime.** On the
  `vulkan` backend build, the Settings "Main GPU" selector and "Split across
  GPUs" checkboxes stayed hidden even on a working multi-GPU box, because they
  listed only the devices torch or nvidia-smi could see, and neither can see
  Vulkan-only devices at all; a split could only be configured by hand-editing
  the config. On that build the selectors now list the devices the Vulkan
  runtime itself registers (read crash-safely out of process), numbered in the
  runtime's own order (the numbering a model load actually uses), with a note
  saying so. Other backends are unchanged.
- **`localm rag add --embed` / `rag query --embed` now actually compute
  embeddings.** The CLI sent a placeholder model name to the embeddings
  endpoint instead of your configured embedding model, so with a dedicated
  embedder set up (the normal case) and no chat model loaded, every `--embed`
  index silently fell back to lexical-only (BM25) and every `--embed` query
  scored lexically, with no vectors ever written. It now sends the configured
  embedding model name, exactly as the GUI does, so CLI indexing/query gets the
  same hybrid (vector + lexical) retrieval. The GUI path was unaffected.
- **Downloads and update checks work over HTTPS on a fresh machine now.**
  Provisioning the native llama.cpp runtime with `setup-llama` (and, from the same
  cause, `localm update`, the issues list, and bug-report upload) could fail on a
  fresh Windows machine with `CERTIFICATE_VERIFY_FAILED` ("unable to get local
  issuer certificate") even though a browser downloaded the same file fine. These
  paths verified against the machine's OS certificate store, which Python's TLS
  does not keep current on Windows; they now verify against a bundled CA set (the
  same one your model downloads already use), so they work regardless of the
  machine's cert-store state.
- **Multi-GPU split: the GGUF loader's own sizing now budgets the whole split,
  not one card.** The earlier "Multi-GPU split fit checks" fix taught the
  pre-load gate and the fit badges to sum capacity across a configured split,
  but the GGUF backend kept a second, deeper preflight that still budgeted the
  whole model against the main GPU alone. On a box where the split devices are
  detectable (for example multiple AMD cards on Windows via the shipped ROCm
  torch, or any install with a working CUDA torch), the split's headline case -
  a model larger than one card that fits combined - was defeated three ways:
  auto GPU-layer sizing silently offloaded only part of the model (a large,
  silent slowdown), a pinned `n_gpu_layers` load was refused with a factually
  wrong "it cannot fit regardless", and the auto context ceiling collapsed to
  the base window despite ample combined headroom. All four sizing checks (auto
  GPU layers, the VRAM preflight, the auto context ceiling, and the
  mid-generation context-grow check) now budget against the split's combined
  free/total, with the same probe-freshness honesty as the admission gate and a
  fall back to the old single-device behavior whenever the combined reading is
  unmeasurable; the refusal message on a split box now names the split's
  combined capacity instead of "this GPU".
- **One failed coder turn no longer marks every later turn as failed.** In a
  coder session with more than one turn (the GUI, or the REPL), a turn that
  ended badly - the turn cap, a circuit breaker, or your own stop - left the
  session flagged as failed for good. Every turn after it was reported failed
  too, however cleanly it finished, so the GUI kept labelling healthy work as a
  failure and `localm coder --ci` could exit non-zero on a task that had
  actually succeeded. Each turn is now judged on its own. Sessions still
  remember that something went wrong earlier, so the close-time lesson the coder
  writes to its episodic memory is unchanged: a session that stumbled and then
  recovered still records what went wrong, and is no longer described as
  unfinished when it did finish.
- **A client plugin reading `window.modelCache` (or another reassigned GUI
  export) got a value frozen at page load, never the current one.** The GUI's
  module loader copies each `app/*` and `pages/*` module's exports onto `window`
  once, right after they evaluate, so a runtime-injected client plugin can reach
  them as `window.X` the same way the pre-module globals worked. That copy was
  only a snapshot: a handful of exports, `modelCache` among them, are
  *reassigned* rather than mutated, and a snapshot never sees a reassignment.
  `window.modelCache` in particular stayed at its initial `{models: [], active:
  ""}` forever, even with a model loaded and chat already working, while the
  module's own internal `import` of the same binding was always current.
  `window.X` is now a live getter into the module's namespace object instead of
  a one-time copy, so it reads the current value the same way the module itself
  does. No plugin reads `window.modelCache` yet, so this had no visible symptom
  before now, but the fix covers every export with this pattern, not just this
  one.

### Security
- **A scoped "settings" key can no longer change plugin settings the plugin's own
  page reserves for you.** Text-to-speech asks for an owner key before it will
  change the script every browser loads, the media backends ask for one before
  they will change a launch command or render target, and enabling a plugin needs
  a plugin-admin key. The general settings endpoint did not ask: it accepted the
  whole per-plugin block and the enabled-plugins list as ordinary settings, so a
  key you had created with permission to change settings but deliberately WITHOUT
  owner rights could write exactly those values through the general route instead
  and skip the check, including pointing text-to-speech at a script hosted
  somewhere else that the browser would then load. Both are now owner-only on that
  route, and the refusal names the endpoint to use instead. Reading settings is
  unchanged. This needed a key you had minted yourself with settings-write but not
  owner rights (no key preset localm ships grants that combination), so a default
  install was not exposed.
- **A scoped "settings" key can no longer redirect where your bug reports are
  sent.** The bug-report upload endpoint and the update endpoint, with their
  optional shared secrets, were treated as ordinary settings, so the same kind of
  key could re-point them. That mattered because the bug-report endpoint is a live
  channel with a real default: "Send to maintainer" posts the collected
  diagnostics and whatever you typed to it, so a redirected endpoint would have
  received your next report. All four are now owner-only to read and to change,
  the same treatment the folder-indexing and private-network settings already had.
  Installing an update was never at risk from this: an update is verified against
  a signing key built into localm, and one that is unsigned or signed by anything
  else is refused before any file is replaced.
- **A malformed grammar could crash the server instead of being rejected.** The
  `grammar` field on chat and completion requests had no limit on size or nesting
  depth before it reached llama.cpp's native grammar parser, so a deeply nested or
  oversized grammar could drive that parser past its native stack limit - crashing
  the whole server process for every request it was handling, not just failing the
  one that sent it. The grammar is now checked in Python first, before any of it
  reaches the native parser, and an oversized or pathologically nested grammar is
  now rejected with an ordinary 400 error instead of being able to crash anything.
- **A scoped "settings" key can no longer widen which browser origins may call
  the API.** The CORS-origins setting was treated as an ordinary setting, like
  chat temperature, so a key you had created with settings-write but
  deliberately WITHOUT owner rights could set it to allow any origin. That
  mattered beyond CORS itself: allowing every origin also opts two
  unauthenticated diagnostic endpoints, /whoami and /debug/stacks, out of their
  own cross-origin refusal, and /whoami's root_dir field is an absolute path
  that names the account localm runs under. The CORS-origins setting is now
  owner-only to change, the same treatment the folder-indexing,
  private-network, and bug-report settings already had. A default install was
  not exposed: this needed a key you had minted yourself with settings-write
  but not owner rights, and no key preset localm ships grants that
  combination.
- **A scoped "read settings" key could see plugin settings only an owner is
  supposed to see.** Media (image/music/video) and text-to-speech settings each
  reserve a couple of fields for an owner key to change: a media backend's
  launch command and API address (a shell command, and where renders are sent),
  and text-to-speech's script and WASM paths (what every browser loads).
  Changing them already required an owner key; reading their current value back
  through the same two settings endpoints did not check the same thing, so a key
  you had created with only "read settings" permission - including the
  ready-made "Full" preset localm itself offers when minting a key - could see
  them anyway. Both endpoints now hide those fields' value from a non-owner
  key, the same treatment the general settings page already gives its own
  owner-only fields. Reading everything else through these endpoints is
  unchanged.
- **A settings change could echo back an owner-only value it was never asked to
  change.** The general settings endpoint already refuses to let a key with
  permission to change settings but deliberately WITHOUT owner rights set an
  owner-only field (for example the update endpoint's shared secret), but the
  response after a successful, otherwise-ordinary change carried the CURRENT
  value of every setting, owner-only ones included, so that key could read a
  value it could never write. This needed a key you had minted yourself with
  settings-write but not owner rights (no key preset localm ships grants that
  combination), so a default install was not exposed by this finding alone. The
  response now hides the same fields the write already refuses.

## [0.1.2] - 2026-07-18

### Added
- **`localm stop`.** `localm run`/`gui`/`serve` start a background server that
  outlives the command; there was no documented way to end it short of reading
  its PID from `ps`/`status` and killing it by hand. `localm stop` (with no
  argument, an id from `ps`, or `--all`) asks it to shut down cleanly and
  force-ends the process if it does not confirm in time.
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
- **HuggingFace search works on every model type, with explicit filters.** The
  Models page "Find models" search used to only work for LLMs; the other model
  types showed a "coming soon" placeholder. It now has explicit, always-visible
  filter checkboxes: which model Types to search for (LLMs, Embedding, Diffusion,
  Encoders, VAEs, LoRAs, Other) and which Format (GGUF or Safetensors). Tick any
  combination and HuggingFace is searched for real, narrowed on both axes where
  its own tagging is reliable, and by format plus your query text where it is not
  (even the most widely-used VAE and text-encoder files on HF carry no tag saying
  so, so a hard type filter there would hide exactly what you want; the format
  filter still applies). Every result shows its detected type, and a model you add
  registers with the type you searched for instead of falling back to a guess.
  These search filters are independent of the Registered-models tabs below, so
  what the search covers is always visible, never inferred silently. The
  Registered models table's Role column and the model-detail view also color-code
  every type consistently.

### Changed
- **The localm data directory is no longer refused for RAG indexing.** Previously
  any file under your data directory (LOCALM_HOME) was hard-blocked from being
  indexed into a knowledge collection, even by you, even explicitly. It is your
  data and your machine, so it is treated exactly like any other folder now (the
  usual whitelist/blacklist rules and consent prompts apply, nothing special).
  Third-party credential folders (`.ssh`, `.aws`, and similar) are still refused.
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

### Removed
- **The `[gguf]` optional extra (`llama-cpp-python`).** GGUF inference has run
  through localm's own ctypes binding to `llama.dll` since day one; this extra
  installed the third-party `llama-cpp-python` package, but nothing in the
  codebase imported it. Removed the dead dependency declaration from
  `pyproject.toml` and regenerated `uv.lock`; `pip install "localm[gguf]"` is
  no longer a recognized extra.
- **The `[qr]` optional extra.** `qrcode` was promoted to a core dependency at
  some point, leaving this extra installing a package that was already there.
  `localm gui --qr` needs no separate install now; `pip install "localm[qr]"`
  is no longer a recognized extra.

### Fixed
- **Semantic search and memory recall no longer silently truncate what they
  can embed at a flat 512 tokens.** The embedder capped every model at the
  smallest bundled default's own window regardless of what it actually
  supports, discarding the back half of anything longer with only a debug
  log line as evidence. It now sizes to whichever is smaller: the loaded
  model's own declared training context, or a generous 2048-token ceiling
  (comfortably above what a knowledge chunk or memory fact ever needs) - so a
  model built for a longer window, such as nomic-embed-text, bge-m3, or
  Qwen3-Embedding, actually uses it.
- **A decoder-based embedding model (Qwen3-Embedding, gte-Qwen2, ...) now
  works correctly with no setting to discover.** These models are trained for
  last-token pooling, but with no `embedding_pooling` chosen, localm forced
  mean pooling on every model - measurably degrading embedding quality for
  this specific class, silently (the vectors still looked normal). Nothing
  explicitly configured now correctly uses each model's own declared pooling
  when it declares last-token specifically; the bundled `bge-small`/`nomic`
  choices are unaffected (still mean, exactly as every existing index built
  with them expects). An explicit `embedding_pooling` choice still always
  wins, as before.
- **Setting up or using an embedding model no longer crashes on some ROCm/HIP
  GPU installs.** `localm setup-llama` was provisioning `rocblas.dll` and
  `hipblaslt.dll` without the GPU kernel data files (`rocblas/library/`,
  `hipblaslt/library/`) they need at runtime - present in the same upstream
  archive the whole time, just dropped by the copy step, since it only copied
  `.dll`/`.exe` files. Without that data, any embedding call that dispatched a
  GEMM through it (the "Set up / apply" test, or real document indexing) hard-
  crashed the isolated embedding worker instead of failing cleanly. Re-run
  `localm setup-llama --force` to pick up the fix on an already-provisioned
  install. As a safety net for installs that have not yet reprovisioned, the
  embedder now automatically retries once on CPU after a GPU crash and says so
  in the Knowledge page instead of just failing - and that retry genuinely
  hides the GPU from the runtime rather than only skipping weight offload, so
  it recovers regardless of which embedding model is selected, not only small
  ones.
- **A collection with no documents no longer shows "reindex needed."** The badge
  meant "this collection predates your embedding model," but an empty,
  freshly-created collection triggered it too, even though there was nothing to
  reindex.
- **A worker crash during embedding no longer surfaces as a bare "Internal
  server error."** `/v1/embeddings` now returns the actual failure reason, so
  indexing and reindexing report why semantic search was skipped for a
  document instead of a meaningless generic message.
- **Knowledge search no longer lets a filler word outrank the better match.** In a
  RAG collection, a document that shared only a common word like "and" or "the" with
  your query could be ranked above the document that actually matched your meaning -
  most visibly on small or narrowly-focused collections, where a filler word is rare
  enough to look significant. Those stopwords are now filtered from the keyword half
  of search (both when indexing and when querying), so a shared filler word alone can
  no longer push the wrong document to the top. Semantic (embedding-based) matching is
  unchanged, and collections do not need re-indexing.
- **Loading a chat model no longer refuses with "Not enough VRAM" just because the
  embedder was still resident.** The embedding model has its own separate lifecycle
  from chat models, so the automatic low-VRAM eviction that frees up space for a new
  load never considered it a candidate. If you had recently indexed or searched a
  RAG collection, the idle embedder could sit in VRAM and starve out a chat model
  load that would otherwise have fit. It is now tried as free-able space before
  localm falls back to asking another running instance to unload, or refusing.
- **A model too big for its own VRAM estimate now actually loads with layers spilled
  to system RAM, instead of being refused outright.** The GUI already tells you a
  "too big" model will still run with some layers offloaded to system RAM - but the
  load path used a cruder, separate size estimate and hard-refused before the
  backend ever got a chance to do that offload. It now attempts the real load, and
  the backend's own accurate sizing (which already supports partial GPU offload and
  CPU spillover) decides whether it truly fits, matching what the badge promises.
- **Loading or embedding on Windows with an AMD GPU no longer risks a blocking
  Windows error dialog.** The first `import torch` in a fresh worker process could
  collide with the native runtime llama.cpp had already loaded into that same
  process, popping a modal dialog that needed a manual click to unstick instead of
  raising a normal, catchable error. localm now recognizes when a process is in that
  risky state and skips torch's VRAM probing entirely rather than triggering the
  collision, and any other native crash in a worker process is also kept from
  surfacing as a blocking dialog.
- **A model load queued behind an in-flight eviction of that same model can no
  longer end up pinned to an engine that gets freed out from under it.**
  Freeing VRAM for one model can require evicting another; if a second request
  for the model being evicted arrived while that eviction's native free was
  still in progress, it could load and register a fresh copy, have it handed
  back and pinned, and then have the still-finishing eviction release it
  anyway - the request believed it held a working model when it did not. That
  request now gets a clean, retryable "currently being freed" response instead
  of a doomed pin.
- **`localcoder` now names the real reason when its auto-started server dies
  fast.** The busy-port refusal above meant an auto-started `localm gui` now
  exits immediately instead of relocating - but `localcoder`'s attach loop
  didn't notice the child had already exited, so it waited out the full
  attach timeout and reported the generic "Failed to attach to the
  auto-started server", which reads like a hang, not a port conflict. An
  explicit `--port` that is busy is now checked before spawning at all, so the
  same "Port N is already in use" message surfaces immediately; any other
  fast, non-zero exit is now caught mid-poll and reported with its exit code
  instead of the misleading generic message.
- **An explicit `--port` that is busy now errors instead of silently moving you
  onto the default port.** `localm gui`/`serve --port N` resolved a busy port by
  scanning localm's range from the start, so a deliberately chosen high port (e.g.
  `--port 8903`, picked to stay clear of other instances) that was in use dropped
  you back onto the shared default 8642 - the opposite of what you asked for. An
  explicit port is now honored exactly or refused with a clear "Port N is already
  in use" message and a non-zero exit; only the default port (when no `--port` is
  given) still auto-bumps through the range. The `--port` help text, which
  promised "auto-bumps when busy", is corrected to match.
- **`localm bug-report` now bundles the server-hang trace too.** The 0.1.2
  automatic server-hang diagnostics attached the captured freeze trace when you
  filed from the app or when a crash was recovered on the next start, but a report
  filed with the `localm bug-report` CLI did not - it left the trace out while
  looking otherwise complete, so a hung server reported that documented way arrived
  with the one diagnostic missing. The CLI runs in a separate process from the
  frozen server, so it now finds the running server through the local instance
  registry and attaches that server's trace, matching the in-app and crash-recovery
  paths.
- **tok/s no longer counts model-load time as slow generation.** The speed readout
  in the CLI and GUI divided the tokens generated by the whole wall time, including
  the seconds spent loading the model on the first request. That made the first
  reply after a load report a rate up to a few hundred times too low (a GPU that
  runs at 64 tok/s showed 0.6 tok/s cold), which looked exactly like the model had
  silently fallen back to the CPU. tok/s is now measured over generation time only,
  matching `localm benchmark`; the load/first-token wait is reported separately (the
  CLI shows a `load` split; the GUI already showed TTFT). A one-token reply, which
  has no generation interval to measure, now omits the rate instead of printing a
  meaningless huge number. Under heavy concurrent GPU load, a delayed-then-caught-up
  first token can also make the opposite mistake - implying tens of thousands of
  tok/s, which is just as physically impossible as the old under-reported rate - so
  an implausible decode window now also omits the rate rather than show a number
  that cannot be true.
- **`localm doctor`'s Python version check.** It reported any 3.10+ interpreter
  as OK, but `pyproject.toml` has required exactly 3.12 for a while (3.10/3.11
  cannot even import `localm.plugins.loader`, and the AMD `[gpu]` wheels are
  cp312-only). The check now matches the actual pin.
- **`localm run <model>` no longer answers you with a different model.** If a
  localm server was already running for that folder with another model loaded,
  `localm run` quietly attached to it and printed a reply generated by THAT
  model - indistinguishable from the model you actually asked for. It now
  refuses with a clear error naming the conflict, and points at `--no-server`
  to run your model in its own process. `localm gui --port`/`--mode` (and the
  other server settings an attach cannot honor) did the same thing: they were
  silently dropped, and now refuse with `--new` as the way out. Re-passing a
  setting the running server already uses is still fine, and it still attaches
  quietly when nothing conflicts.
- **`localm doctor` no longer reports the HuggingFace backend as fine when it
  cannot load a single model.** Doctor only checked that `transformers`
  imported and printed its version. But transformers is a lazy module, so that
  succeeds even when the classes the backend actually loads through are broken
  - which is exactly what happened on Windows + AMD, where every HF model load
  died at "loading processor…" while doctor stayed green. Doctor now resolves
  the real classes it needs, and when they fail it digs down to the underlying
  cause instead of showing the generic "Could not import module" wrapper that
  hid the problem.
- **Garbled startup banner on `localm gui`/`localm serve`.** The background
  model-preload thread and the main thread's own startup prints used separate
  `rich.Console` instances with no shared lock, so their output could
  interleave character-by-character (e.g. "Loading   qwen2.Open the GUI:5
  ..."). All output on that path now goes through one shared console.
- **GPU detection no longer mistakes a slow driver start for "no GPU".** The
  first GPU check after a machine or driver cold start can legitimately take a
  few seconds, but localm gave it only 4 - so on many boxes the first look at
  your hardware "timed out" and everything downstream treated that as a machine
  without a GPU: `localm gpus` and doctor could report nothing found, and the
  Settings GPU controls (Main GPU, Split across GPUs) could silently vanish on
  a multi-GPU box. The time limit was guarding against a server freeze that can
  no longer happen (GPU checks moved off the serving thread long ago), so it
  now simply allows a cold start to finish. If the driver is genuinely stuck,
  the Settings page also no longer concludes "single GPU" from a check that
  never completed - it keeps your GPU controls as they were and retries next
  time, instead of hiding them.
- **A multi-GPU split no longer turns off the VRAM safety check on the Vulkan
  backend.** Before loading a model, localm decides whether it must check each
  card's share of VRAM by asking how many cards your split spans. On the Vulkan
  build that count was measured with a tool that cannot see Vulkan cards, so a
  real, working two-card split was read as "not a split" and the per-card check
  was skipped on exactly the machines that needed it, letting an embedding model
  too large for one card's share reach the loader and, at worst, abort instead of
  failing cleanly. The check now uses the same signal the loader itself uses, so
  it runs whenever a split is actually active. Where per-card free VRAM genuinely
  cannot be read on Vulkan, localm now notes that in the log and leans on the
  isolated loader to fail safely, instead of silently skipping the check. Single-
  GPU machines and the CUDA/ROCm backends are unaffected.
- **The status-bar VRAM figure and the performance-page fit estimate no longer
  present a VRAM number localm cannot stand behind.** The free-VRAM reading behind
  both can be stale (a slow GPU driver makes localm reuse an older figure) or, on
  Windows with an AMD GPU, blind to other processes: it counts only this program's
  own VRAM and misses a model loaded in localm's separate worker (or a game holding
  VRAM), so "free" reads far too high. The status bar could show that stale or
  inflated number, and the performance estimate could call a model a "fit" when it
  would not be. Both now show a used/free figure ONLY when the reading is current
  and whole-GPU; otherwise the status bar shows total VRAM alone (always correct)
  and the estimate reports "free VRAM unknown" rather than guess. The status bar
  also tints the VRAM figure by how full it is.
- **The status-bar CPU figure no longer shows a made-up number on the first
  reading.** The hardware monitor measures CPU use since its previous poll, so the
  very first reading after opening the page had no earlier sample to compare
  against and reported whatever the instant happened to be: 0% on an idle machine,
  or as high as 100% if the page had just loaded a model. It now shows nothing for
  that first reading and a real percentage once it has two samples to compare, so
  the CPU number the status bar shows always reflects actual use.
- **Unloading a model no longer presents a VRAM reading it never took as fact.**
  localm reads free VRAM from the GPU driver with a time limit, and a slow or busy
  driver could not always answer in time. When that happened it quietly reused the
  last reading it had - which on a server that had been running a while was taken
  before your model ever loaded. The before and after figures then came back
  identical, and unloading reported that no VRAM was freed even though your VRAM
  had been freed. localm now tells a reading it actually took apart from a reused
  one: anything it could not measure is reported as unknown and marked as such,
  and it will never claim VRAM went unfreed on the strength of a measurement it
  never took. This covers the per-model Unload button, the embedding model, and
  the VRAM handover that image, music, and video generation do before they run -
  which previously told you "VRAM has not dropped yet" on the strength of a
  reading it could not stand behind. A machine with no GPU telemetry at all is
  unaffected and stays quiet, as before.
- **A multi-GPU split load is no longer refused on the strength of a VRAM reading
  localm never took.** Before placing a model onto a configured GPU split, localm
  checks that each card has room for its share, reading free VRAM from the driver
  with a time limit. A slow or busy driver - a cold GPU right after the server
  starts is the common case - could not always answer in time, and localm quietly
  reused an older reading, then refused the load and quoted a "MB free" figure from
  that stale reading on a machine that in fact had room. It now tells a reading it
  just took apart from a reused one: when it cannot get a fresh per-device reading
  it lets the load proceed rather than refusing on a number it cannot stand behind
  (the load is sandboxed, so a genuine over-commit fails safely on its own) and it
  never shows a stale VRAM figure as the reason. The embedding model's split
  preflight is covered the same way.
- **Your own saved facts come back when you ask about yourself.** Without an
  embedding model installed, memory could only match facts that shared an exact
  word with your question, so asking "what is my name" never surfaced "User is
  called Sam" - it silently answered as if you had saved nothing. When you ask
  about yourself, memory now falls back to a couple of your own saved facts
  instead of staying silent. Questions that are not about you still stay silent,
  and installing an embedding model (`localm setup-embeddings`) is still what
  gives you full meaning-based recall.
- **Your saved memories are recalled again when an API key is set.** With a key
  configured, everything you saved (and everything memory learned on its own) was
  filed under one identity but looked up under a different one, so chat quietly
  recalled none of it. The Memory page still listed every fact, which made it look
  like the model was simply ignoring them. Both paths now use the same identity.
- **The app no longer freezes while memory is distilling in the background.** After
  a turn, memory quietly distils new facts, which takes one model call per candidate
  fact. It held the memory lock for that whole stretch, so your next message - and
  every other request, token stream, and health check - waited for it to finish:
  seconds to minutes on a local model. It now does that thinking without holding the
  lock, so chat stays responsive while memory grows.
- **The first memory distil after an upgrade no longer monopolises the model.** With
  a backlog of past conversations, the first background distil summarised every past
  session, one model generation each, back to back - starving chat for as long as it
  took. It now summarises a few per pass and works through the backlog over time.
- **A conversation is remembered as one entry, even if you come back to it.** An
  in-progress conversation could be summarised while it was still going, and a
  conversation you resumed later could be summarised again, leaving several
  overlapping partial memories of the same session. Each conversation now waits
  until it is finished and keeps a single summary, updated to cover the whole thing.
- **The coder's episodic memory no longer occasionally drops a lesson on
  Windows.** A GUI poll reading past lessons could race a session-close write
  saving a new one, and on Windows a save or read could briefly fail with the
  file busy. The retry that already handled this could still be exhausted by
  real contention (a loaded machine, antivirus scanning the file), causing a
  rare, hard-to-reproduce miss. It now retries for much longer with backoff, so
  it survives realistic delays while still failing loudly if the file is
  genuinely stuck.
- **Installing packages no longer fills up your system drive.** Whenever localm
  installs things for you - setting up its own ComfyUI (including the multi-GB
  PyTorch build), pulling in a plugin's dependencies (such as the voice extra), or
  fetching its own native runtime - it let the installer cache what it downloaded
  in your user profile instead of localm's own data folder: gigabytes landing
  outside localm, on a drive you may not have picked, without being asked or told.
  Those caches now live inside your data directory, so they move with it and are
  removed with it. This covers both installers localm uses, `pip` and `uv`.
  Speech-to-text is fixed the same way: the Whisper model it downloads on first use
  now lands in your data directory rather than the shared HuggingFace cache in your
  home folder. If you already have a Whisper model in that shared cache, it is
  downloaded once more into the data directory; the old copy is left alone and is
  safe to delete. Anything previously cached in your profile is still there - on
  Windows you can reclaim that space with `pip cache purge` (and `uv cache clean`
  if you use uv).
- **Picking a Main GPU no longer stops large transformers models from loading.**
  With a Main GPU selected, localm pinned the whole model onto that one card, so a
  transformers model bigger than its free memory failed with an out-of-memory error
  even though it used to load (more slowly) by keeping part of it in system memory.
  It now still uses only the card you picked, but falls back to system memory for
  whatever does not fit, as it did before. The same fallback was restored for a
  configured multi-GPU split.
- **Image, music and video generation no longer run out of memory on a multi-GPU
  box.** With a GPU split configured, localm added up the free memory across every
  card in the split and concluded a generation would fit, so it kept the chat model
  loaded. But an image, music or video model loads onto a single card, so the
  generation could still fail with an out-of-memory error, quietly fall back to
  system memory and run many times slower, or lock up the display driver. localm now
  picks the card with the most free memory, tells ComfyUI to use that one, and checks
  that card on its own has room before starting - unloading the chat model first if
  it does not. If you have a split configured, generation also now tells you plainly
  that it runs on one card and which one: a single image, music or video model cannot
  be divided across cards the way a chat model can, so your split ratios do not apply
  there. Chat and embeddings still use the full split. Nothing changes on a
  single-GPU machine, or if you have not configured a split or a Main GPU.
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
  with an internal error, permanently. It now reloads itself on the next message,
  and the same applies to the token counts and grammar checks that run alongside
  chat - they fall back to their normal estimate instead of failing.
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
- **An API key with an accented or non-English character no longer locks you out.**
  Setting a key like `pässwort-key` left the server unable to answer any
  authenticated request: your own correct key and a wrong one both failed with
  "Internal server error" rather than letting you in or cleanly refusing. The same
  fault let any caller trigger that error without a key at all, just by sending a
  bearer token containing a non-English character, which filled the log with
  tracebacks. Keys are now compared safely whatever characters they contain, so a
  wrong key is refused cleanly and the right one works.
- **`localm key set` now says which characters a key may use.** An API key travels
  in an HTTP request header, which cannot carry spaces, punctuation, or non-English
  letters reliably, so a key using them left you unable to sign in from most
  clients. Such a key is now refused with a message naming what is allowed: letters,
  numbers, `-` and `_`, the same characters `localm key generate` produces.
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
- **Privacy mode no longer erases your message while you're still reading the
  reply.** localm rechecks whether privacy mode is on roughly every 30 seconds so
  the tab stays in sync with the server, and that recheck used to clear the
  visible conversation every time it ran, not just the first time. If it fired
  while your message was still being answered, or had just landed, the reply
  disappeared from view even though the server had answered correctly, with no
  error shown. The clear now only happens once, the first time privacy mode is
  confirmed after opening the tab; the recurring check no longer touches a
  conversation already in progress.
- **The MCP server's `chat`, `embed`, and `pull_model` tools no longer risk
  corrupting the connection.** MCP tools talk to your client over the same
  channel used for low-level model-loading messages, and triggering a model load
  through `chat`, `embed`, or the load step at the end of `pull_model` could
  print diagnostic text onto that channel - which a strict MCP client reads as a
  malformed response and may disconnect on. These three tools now get the same
  protection the server's other tools already had, so a model load through them
  no longer risks breaking your MCP session.
- **Managed ComfyUI music and audio generation works again.** After `localm comfy
  setup`, generating music (or using any audio-related ComfyUI node) could fail
  outright with a native library load error, because setup installed a mismatched
  build of one required audio library alongside the correctly matched ones. Setup
  now installs a matching build, so music and audio generation load and run
  normally.
- **A rare VRAM-probe race that could crash the server on Windows + AMD ROCm is
  fixed.** Two internal paths could each try to load the GPU driver library for
  the very first time at the same moment - a background VRAM probe that had
  timed out but kept quietly running, and a separate, unrelated VRAM check
  landing right after it. Racing to load that native library could crash the
  whole server. The second path now checks whether that background probe might
  still be mid-load before touching the same library itself, rather than risking
  the crash.
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

[Unreleased]: https://github.com/Matlan1/localm/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/Matlan1/localm/releases/tag/v0.1.5
[0.1.4]: https://github.com/Matlan1/localm/releases/tag/v0.1.4
[0.1.3]: https://github.com/Matlan1/localm/releases/tag/v0.1.3
[0.1.2]: https://github.com/Matlan1/localm/releases/tag/v0.1.2
[0.1.1]: https://github.com/Matlan1/localm/releases/tag/v0.1.1
[0.1.0]: https://github.com/Matlan1/localm/releases/tag/v0.1.0
