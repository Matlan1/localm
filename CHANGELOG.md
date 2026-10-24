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
- **Segment Anything (SAM) checkpoints are recognised as `segmentation` models.** A SAM or SAM 2 `.pth` file found by a ComfyUI or StabilityMatrix scan, or sitting in a `sam`, `sams` or `segmentation` folder, is now listed as a segmentation model instead of `unknown`, and `localm set-type MODEL segmentation` and the Models page type menu offer it. localm does not run segmentation models yet, so naming one for chat is refused with a clear message.

### Changed
- **Adding a folder of models says what it is doing, and starts much faster.** The Add a model card, the Activity list and the terminal now show each step: scanning the folder, how many models were found, which file is being checked, hashed (with a progress bar and speed) or registered and its place in the batch, which files were skipped and why, and a final summary. Before registering anything, localm also read every file in the folder in full to look for an Ollama manifest; on a folder of large models that was most of the wait (about 2 minutes of 3.5 on a 42 GB folder) and is now gone.

### Fixed
- **Importing a folder of models into an install that already has some no longer fails.** With the move or copy option, a model already in the models folder under the same name and with the same content is now used as it is (the original is left where it was) and the rest of the folder imports. A different file under the same name is skipped with a message instead of aborting the whole import.

## [0.3.1] - 2026-10-22

### Added
- **`localm pull owner/repo@revision` pulls a Hugging Face branch, tag or commit.** Append `@revision` to a repo (`localm pull turboderp/Qwen3-0.6B-exl3@4.0bpw -n exl3`) or to a single-file spec (`owner/repo@v1.0:model.gguf`) instead of taking the default branch, which is where repos that keep each quant on its own branch put their files. A revision that does not exist is refused before anything is downloaded, and a network or permission problem is reported as such rather than as a missing revision. Every file is read and downloaded at the commit the revision pointed at when the pull started; `localm list` shows `hf:owner/repo@revision`, and pulling again with the same spec reuses the download. A pull never mixes two revisions in one folder or replaces a same-named file that another pull registered: it stops and says what to remove.
- **A VRAM cache for the MoE experts kept in system RAM.** The new `moe_cache_mib` setting
  (Settings, Engine, or `localm config moe_cache_mib 3072`) keeps the most used experts of a
  Mixture-of-Experts model whose experts stay in system RAM on the GPU, so they are not copied
  there again for every batch, which makes generation faster. It applies when those experts are
  read into memory rather than memory-mapped, sets that much VRAM aside for any model with
  experts, and is off by default (experimental in llama.cpp).
- **More context fits in VRAM with a q8_0 KV cache.** The new `kv_cache_type` setting (Settings,
  Model, Runtime & GPU, or `localm config kv_cache_type q8_0`) stores the KV cache in about half the memory of
  the default `f16`, so roughly 1.9x the context fits before the window stops growing, with
  slightly slower decoding. The automatic context ceiling and the VRAM estimate account for it. It
  applies at the next model load; a model that cannot use flash attention refuses to load under
  `q8_0` and the error names the setting.
- **Inpaint, outpaint, edit with reference images, ControlNet and embeddings with the built-in
  image backend.** The Images page has a mask editor for the input image (brush, eraser, invert,
  and extend the canvas to outpaint), reference images for edit models such as FLUX.1 Kontext,
  Qwen-Image-Edit and FLUX.2, a ControlNet picker with a control image (optionally turned into
  an edge map) for Stable Diffusion 1.x, 2.x and SDXL models, and textual-inversion embeddings
  listed with their trigger words. ControlNets and embeddings are new model types, picked up from
  ComfyUI and StabilityMatrix folders and CivitAI downloads.
- **Workflows for the built-in image, video and music generators.** The Images, Video and Music
  pages have a Workflow card for the native backend: a workflow saves the models (with LoRAs and
  their strengths) and the generation settings, and Apply puts them back. Starter workflows come
  built in (SD-Turbo, FLUX.1 schnell and dev, Wan 2.1 text to video 1.3B, the ACE-Step 1.5 set).
  Applying one offers the download of every model you do not have yet, checked against its sha256
  when the workflow gives one (the built-in ones always do). Workflows can be renamed, deleted,
  exported and imported, and a ComfyUI workflow (API or regular format) imports as one: its
  loaders, sampler settings, size, prompts, guidance and shift are carried over, and every node
  with no native equivalent is listed. The Video page gains a Flow shift setting and the Music
  page a Shift setting for the native backend.
- **`localm coder` shows what it is doing.** A live spinner runs while the model thinks, while
  it writes a file (with the file name and line count) and while a tool runs. `set_todos` shows
  the task list. Type `@path` or `@path:10-40` to attach a file to your message, and a
  multi-line paste arrives as one message.
- **Approval prompts in `localm coder` take more answers.** `a` allows that tool for the rest of
  the session (for a shell command, that exact command only), `f` shows the full diff, and
  `n: <reason>` tells the agent why you said no.
- **Pause, resume and cancel a model download in the GUI.** The Add a model card has Pause (stops
  and keeps what is saved, Resume continues from it) and Cancel (stops and deletes the partial
  download), the download line shows the steady speed and time left, and the Activity list has a
  Stop button for a running operation.
- **The server log says when Windows keeps part of a model in system memory.** On Windows, when
  another program holds VRAM, Windows can place part of a model loaded fully on the GPU in shared
  system memory, and replies run slower until that program releases it. localm now reads the
  model's share of shared memory and logs one line with the amounts when it is large, and
  another when Windows moves it back.
- **Ctrl-C in `localm pull` stops cleanly.** What has been downloaded is kept, and the message
  says to run the same command again to continue from there.
- **`localm coder doctor` and `/doctor`.** One command checks that the model server answers and accepts the key, whether the session uses native tool calls or grammar-constrained ones, the context window against the size of the repo map, the git state, whether the verify command resolves, and that each configured MCP server starts. It exits 1 when a check fails and prints `--output-format json` or `jsonl` for scripts.
- **`--output-format jsonl` for unattended coder runs.** `localm coder TASK --output-format jsonl` writes every agent event (tool calls and results, tokens, turns) to stdout as one JSON object per line as it happens, then a closing `final` line with the outcome; everything else the CLI prints goes to stderr.
- **Plan mode and questions in the `localm coder` terminal.** `/plan` (or the agent itself)
  switches to a read-only mode where the agent cannot write, run or fetch anything until you
  approve its plan: approve it, approve and let it run with the project's check required, or
  send it back with feedback. The agent can also ask you a question with numbered options
  mid-task.
- **The coder shows lint errors right after it edits a Python file.** When `ruff` is on your
  PATH, a write or edit to a `.py` file appends up to 10 `ruff` findings (unused imports,
  undefined names) to the result so the agent can fix them in the same turn. Off with
  `diagnostics = false` in `.localcoder/config.toml`.
- **The coder sees the shape of a large project and of a large file.** A new `outline_file` tool lists a
  source file's classes, functions and methods with their exact line spans (Python is parsed; other
  languages are scanned by pattern and labelled approximate), so the agent can read just the part it
  needs. On a project with more than 300 files the codebase map is now a directory summary of every
  file found plus detail for the shallowest files, instead of stopping after the first 300.
- **Coder compaction keeps the working state.** A compacted coder session now carries the files
  it changed (with added and removed line counts), the verification command's last result and
  the code written since the last test run outside the summary, so they are never lost to
  paraphrase. Before compacting, the agent also keeps durable notes from the condensed turns;
  in `log` and `full` mode, with episodic memory on, they are saved as a lesson, and privacy
  mode writes nothing. `compaction_flush = false` in `.localcoder/config.toml` turns the note
  pass off.
- **Coder tool hooks.** `.localcoder/config.toml` can declare `pre_tool_use` and
  `post_tool_use` commands that run around the agent's tool calls, matched by tool name. A pre
  hook that exits 2 blocks the call and tells the agent why; a post hook that exits 2 sends its
  output to the agent, for example a linter's complaints. A project's hooks run only after you
  trust them, and never in a restricted session.
- **`localm coder` keeps long tool output instead of cutting out the middle.** When a command or
  search prints more than the agent is shown, the full text is kept in memory for the session
  (never written to disk) and the agent can read any part of it back with the new `read_output`
  tool, by line range or by pattern.
- **`localm coder` no longer asks before read-only shell commands.** `ls`, `cat`, `grep`,
  `git status`, `git diff`, `git log` and similar commands run without a prompt when every path
  they name stays inside the project and nothing writes, redirects or runs another program.
  Anything else asks as before. Set `auto_allow_read_only_shell = false` in
  `.localcoder/config.toml` to be asked every time.

### Changed
- **A localm 0.2.0 install that never updated to 0.3.0 cannot update itself to 0.3.1.** 0.3.1 is
  signed only with the release key 0.3.0 introduced, which 0.2.0 does not have. Install 0.3.1
  fresh from the release page instead.
- **Long agent and chat sessions stop re-reading the whole conversation every turn.** On models
  with recurrent layers (Qwen3.5, Qwen3-Next, LFM2 and other hybrid models) each reply now
  continues from a saved state near the end of the previous prompt instead of processing the
  entire conversation again, and a context that has to grow keeps what it already holds when
  there is system RAM for the copy. `localm coder` keeps its system prompt the same for the
  whole session (the codebase map in it is refreshed when the context is compacted), so a file
  edit no longer makes the model re-read everything. The server log shows how many prompt tokens
  each reply reused.
- **localm's managed ComfyUI now installs v0.42.1, up from v0.39.0.** An existing install picks it up with `localm comfy update --reinstall-requirements` (this release's Python requirements changed).
- **The bundled llama.cpp runtime moved from b11541 to b11757.** An existing install picks it up with `localm setup-llama --force`.
- **The bundled AMD (ROCm) llama.cpp runtime moved from build b1342 to b1349** (upstream llama.cpp b11719). An existing install picks it up with `localm setup-llama --force`.
- **The bundled stable-diffusion.cpp runtime moved from master-964-1d0a58a to master-982-0518b34.** An existing install picks it up the next time image generation starts.
- **The bundled koboldcpp runtime (native music generation) moved from v1.123 to v1.123.1.** An existing install picks it up the next time music generation starts.
- **Newer Python libraries.** The Hugging Face backend now needs transformers 5.20 (from 5.15)
  and tokenizers 0.23.3, which running setup again installs. The dependency lock now carries
  Playwright 1.64 and newer server, PDF, image and download libraries; setup keeps a library you
  already have when it still meets localm's minimum version, so an existing install keeps its
  older versions of those until you upgrade them yourself.

- **The coder starts a project's MCP servers only after you trust them.** A cloned repository
  can declare MCP servers in `.localcoder/config.toml`, so they now need the same trust as tool
  hooks: an interactive session lists them and asks, `--trust-hooks`, `/hooks trust` and the
  GUI's hooks button trust them, and changing a server's command, arguments or environment
  asks again. A project that already uses MCP servers asks once.
- **The coder's system prompt is shorter.** Less frequent tools (git commit, branch and push,
  background jobs, notebooks, image generation, Knowledge search, the browser, and MCP and
  plugin tools) are listed by name and fetched with the new `search_tools` tool when needed,
  and with `--native-tools` the prompt no longer repeats the argument schemas the tools API
  already carries. The prompt stays identical from turn to turn, so the server can reuse its
  prompt cache.
- **Download progress and speed follow the bytes received.** The progress bar, the speed and the
  time left now come from the data arriving, so they move steadily instead of stalling for
  seconds and jumping; the same line appears in a terminal `localm pull`, in the GUI and as MCP
  `pull_model` progress. A full-repository download counts the files still downloading, not only
  the finished ones.
- **A reply no longer moves the KV cache to system RAM before it needs the room.** When the
  whole reply budget (`max_tokens`) would not fit in free VRAM but the prompt plus the first 512
  tokens of the reply does, the context is sized for that much with the KV cache in VRAM and
  grows as the reply gets longer, so a short reply on a card with little free VRAM decodes at GPU
  speed. A reply that does grow that long moves to system RAM at that point and is never cut
  short. A KV cache already in system RAM also moves back to VRAM at a smaller size when the
  next request fits in it.
- **Export saves inside the app.** On the computer running localm, Export (the chat's, and the
  coder's session, result and patch exports) asks for a folder in the app's own file browser and
  saves there, without the browser's download prompt. Other devices still download the file.

### Fixed
- **Hires fix on the Images page refines the picture instead of painting a new one.** With the hires fields left empty, the pass used a latent upscale at denoise 0.7, which on a model made for 512-pixel images (such as SD-Turbo) regenerated the image at twice its size: repeated windows and walls and a different composition from the same seed without hires. Every field left empty now takes the selected model's suggestion (the Lanczos upscaler, scale 2 for models made for 512-pixel images and 1.5 for larger ones, denoise 0.35, or 0.55 when you pick a latent upscaler), shown on the page as placeholders that follow the model. Values you type are always kept; **Use suggested values** fills the fields with the suggestion. An API request that leaves `hires` fields out gets the same suggestion.
- **Memory and Knowledge fetch their small embedding model on first use.** Under the default network setting (`net_mode` ask), the missing embedding model (`bge-small-en-v1.5`, 24 MB, checked against its pinned hash) was not downloaded, a log line asked you to start the download yourself, and memory and Knowledge fell back to keyword search until you did. It is now fetched automatically the first time it is needed; `net_mode` off still never downloads it.
- **Adding a folder of models from the GUI no longer fails when one of them is already registered.** The server asked "already registered as ... [a]lias / [c]opy / ..." on its own console window, where nobody answers, so the add aborted and was reported as failed. A model added through the GUI or the API is never asked about now: a duplicate is skipped with a note naming the existing entry, and the other models in the folder are registered.
- **After `localm setup-music`, `localm gui` no longer starts by trying to load a music model as the chat model.** With no model named, localm picked the first registered model whose type was not `unknown`, `embedding` or `tts`, which after music setup was ACE-Step's text encoder: the load failed with an error and the chat page named the wrong model. Only chat models are picked now; component models (text encoders, VAEs, diffusion models, LoRAs, projectors, upscalers) never are. The same default applies to scheduled jobs and the MCP server.
- **The Ollama API accepts the `curl -d` commands from Ollama's own documentation.** `curl http://127.0.0.1:<port>/api/chat -d '{...}'` sends its JSON body labelled `application/x-www-form-urlencoded`, and the Ollama routes that take a body (`/api/chat`, `/api/generate`, `/api/show`, `/api/embed`, `/api/embeddings`, `/api/copy`) refused it with HTTP 400. A body that is a JSON object is now read as JSON whatever its Content-Type says, as Ollama does; a body that is not JSON is still refused with the same 400, and a request from a web page (one that carries an `Origin` header) still needs a JSON Content-Type.
- **A client that leaves during an image or speech generation is no longer logged as a server error.** The job was cancelled correctly, but each such leave logged `ERROR localm: unhandled error: POST /v1/images/generations` with a traceback and counted as HTTP 500 in `/metrics`. It is now answered and counted as 499, with no error in the log; a route that really fails is still a logged 500.
- **An embedding model asked for by name answers with its own vectors.** `/v1/embeddings` and
  Ollama's `/api/embed` and `/api/embeddings` answered a request for a pulled embedding GGUF (for
  example nomic-embed-text) with the configured embedding model's vectors and dimension, labelled
  with the requested name. The named model is now loaded and used, and the configured embedding
  model that Knowledge and memory use is left as it is. An embedding model whose file is missing,
  or whose registry entry has no path, is now refused with an error naming it.
- **An upscaler pulled from Hugging Face is registered as an upscaler, and a model of unknown type is no longer offered as a chat model.** `localm pull Comfy-Org/Real-ESRGAN_repackaged:RealESRGAN_x4plus.safetensors` was registered with type `unknown`, listed by `GET /v1/models`, and a chat naming it failed with HTTP 500. The pull now reads the file's tensor names and types it as an upscaler, as it does for other image and video model files (pulling the same spec again retypes an entry already registered as unknown); `/v1/models` leaves out every model of unknown type; a chat or load request naming an unknown-typed file that no backend can run answers HTTP 422 with a one-line reason; and the refusal for a text-to-speech model names `POST /v1/audio/speech` only when the tts plugin that serves it is installed.
- **A text-to-speech model registered as a chat model is retyped by the folder sync.** `localm list` (and the server's startup sync) fetched the speech projector for such an entry but left it typed `llm`, so `localm speak` refused it until `localm set-type <name> tts`. The sync now gives it the `tts` type.
- **The pull progress shows what is written to disk when it differs from what was received.** A Hugging Face Xet download holds data back from the file, so the bar read 1.2 GB while 269 MB was on disk and a resumed pull started far below the percentage last shown. The terminal line, the plain-text lines, the MCP progress messages and the Models page now add "N MB saved to disk" while the file lags the received figure.
- **Stopping the first download of a Hugging Face file keeps what was received.** A file stored with Xet (most Hugging Face repos) was held in memory until late in the download, so Ctrl+C at 755 MB of 2.5 GB kept 6.5 MB and the next pull started over, while the message said the download was kept. Hugging Face files now download over HTTP from the first byte, writing as they arrive, so a stop, a dropped connection or a cancelled pull on the Models page keeps the bytes received and the next pull continues from there. The stop message says how much is saved to disk (for example "755.3 MB of 2.5 GB is saved to disk"), and says so when nothing was.
- **`localm rm` deletes a model folder that holds read-only files.** A model folder that is a git
  clone (its object files are read-only on Windows) was left half deleted and still registered;
  it is now removed completely.
- **Two pulls of one repo into the same folder no longer mix their files.** A full-repo `localm pull` now holds a lock on its destination folder while it downloads, so a second pull into that folder (another revision with the same `-n` name, or the same pull started twice) is refused straight away with "a download into the folder ... is already running" instead of downloading on top of the first. The lock is released when the pull ends or its process dies. Discarding the partial data of a full-repo pull that is still running is refused the same way.
- **Compiled PyTorch extensions stay in the data folder.** PyTorch keeps the C++/CUDA extensions
  it builds on first import in the user's cache folder (`~/.cache/torch_extensions` on Linux),
  outside the data folder, so "delete saved data" never removed them. They now go to
  `cache/torch_extensions` inside it. A location you set yourself with `TORCH_EXTENSIONS_DIR` is
  used as you set it. Files already in the old location are left where they are.
- **The NVIDIA CUDA kernel cache stays in the data folder.** The CUDA driver keeps kernels it
  compiles on first use in `~/.nv/ComputeCache` (`%APPDATA%\NVIDIA\ComputeCache` on Windows), outside
  the data folder, so "delete saved data" never removed them. They now go to `cache/cuda` inside it.
  A location you set yourself with `CUDA_CACHE_PATH` is used as you set it. Files already in the old
  location are left where they are.
- **Native music works when the model files sit in a folder with non-ASCII characters on Windows.**
  KoboldCpp could not open music model files whose path held characters such as ü, ö or ß (in the
  localm data folder or a user name, for example), and the failure was reported as "the vulkan
  backend did not start" before a retry on the CPU. localm now hands KoboldCpp the file's short
  Windows name. On a drive that keeps no short names, the job stops before starting and says which
  file to move to a folder whose path is ASCII only.
- **Stable Diffusion 1.x, 2.x and SDXL read capital umlauts and other non-ASCII capitals in a
  prompt as their CLIP text encoders were trained to.** "Ölbaum" now reaches the model as
  "ölbaum", as the original CLIP tokenizer lowercases it; before, capital non-ASCII letters were
  split into unrelated byte tokens. Textual-inversion trigger words with such letters still match.
  Models with case-sensitive text encoders (FLUX, SD3, Wan and others) get the prompt unchanged.
- **`localm add <file>.safetensors --type upscaler` registers the upscaler.** It was refused as an
  incomplete model.
- **The built-in image backend decodes on the GPU with the Vulkan runtime.** On a GPU whose Vulkan
  driver limits a single buffer (2 GB on an AMD Radeon RX 6900 XT, for example), the final image decode of a larger
  image (FLUX at 1024x1024, for example) ran on the CPU and took about three minutes. localm now
  reads the GPU's limit and decodes in tiles that fit it, which takes a few seconds; the job log
  says when it tiles. The log also no longer prints "Decoding: done" as the decode starts, and no
  longer repeats "Loading weights: done" for every block of weights streamed in during the first
  step.
- **Videos and Qwen-Image images decode on the GPU with the Vulkan runtime.** The same buffer limit
  put part of the decode of a video (Wan 2.1 at its default 832x480) and of a Qwen-Image or Anima
  image at 1024x1024 on the CPU. localm now decodes them in tiles that fit the limit too, and
  refuses a job that would not fit even in the smallest tile instead of running it on the CPU.
- **A long video no longer makes the graphics driver reset the GPU with the Vulkan runtime.** Each
  sampling step of a video model runs its attention over the whole clip as one job on the GPU, and
  on a GPU whose Vulkan attention is slow (an AMD Radeon RX 6900 XT, for example) that job ran for
  several seconds at 49 or more frames of Wan 2.1 at 832x480, long enough for the driver to reset
  the GPU and fail the job. localm now reads how long the system lets one GPU job run before it
  resets the GPU (on Windows from whether it can pause the GPU in the middle of a job and the TDR
  delay; on Linux from the GPU driver's job timeout), measures how fast the GPU that runs the model
  computes attention, and refuses before it starts a clip whose attention job would run longer
  than half that time. It names the most frames that fit at that size and the other GPU runtimes
  this machine can use. A GPU that the system can pause in the middle of a job gets no such limit,
  and where localm cannot read the limit it keeps a conservative one and says so in the job log.
  HunyuanVideo now also decodes
  in tiles that fit the Vulkan buffer limit, and a Vulkan runtime that does not report its limit
  decodes in tiles under a conservative one and says so, instead of decoding part of the video or
  image on the CPU.
- **Larger images keep their attention on the GPU with the Vulkan runtime.** Without flash
  attention, the attention of a larger image (SD-Turbo at 1024x1024, FLUX above about 1024x1024,
  for example) needed a bigger buffer than the Vulkan driver allows and ran on the CPU. localm now
  reloads the model with flash attention on for such an image when the GPU runs it fast enough,
  turns it off when it would make a single GPU job run longer than the system allows, and
  otherwise refuses the image with the largest size that fits and the other GPU runtimes this
  machine can use. A ControlNet gets the same flash attention setting as the model, and SD3 Medium
  and SD3.5 Medium are sized by their own attention, so they are not reloaded at 1024x1024.
- **The global `localm` command works from a Shared install and uninstalls cleanly.** On Linux and
  macOS setup made `~/.local/bin/localm` a symlink with a relative target that pointed nowhere,
  and uninstall left it behind; on Windows it added the relative entry `bin` to your user PATH.
  Setup now records the full path, and uninstall removes the link, including one an older setup
  made.
- **`localm setup-llama --backend metal` on Windows is refused up front.** It printed "Replacing
  the vulkan build" and then installed Vulkan. A backend with no build for this system is now
  refused before anything changes, and the list of backends that have one includes `amd-rocm`
  where it exists.
- **`localm run ... | cat` returns when the reply is printed on Linux and macOS.** When `run`
  started the background server itself, the server kept the pipe open, so the command waited
  until the server stopped. The server no longer shares the terminal's streams.
- **`setup.bat` works from a folder whose name contains `&`.** The setup and uninstall banners
  split the command at the `&` and printed an error; the folder is now shown in quotes.
- **`setup.bat` works from a folder whose name contains `'` or `^`.** A `'` in the folder name
  broke the desktop shortcut and the checks for a OneDrive folder and for programs still running
  from the environment, and a `^` next to a `!` was dropped from the folders setup used for
  Python and its cache. Setup now passes the folder to PowerShell without quoting it and keeps
  the name exactly. A `^` in a folder already on your `PATH` is also kept when setup adds its
  own `uv` to the front of it.
- **`localm info` names the temp folder that is in use.** After `LOCALM_TMPDIR` pointed at a
  folder that could not be created, it still listed that folder instead of the system temp
  folder. An unreadable `localm-home.cfg` is now reported once, not once per setting read.
- **An embedding or speech model is placed by a current VRAM reading.** When another GPU reading
  was running (the GUI refreshing its stats, or just after a chat model loaded), the embedding and
  speech models skipped their fit check and went onto the GPU even with too little VRAM free, and
  the decision to unload the chat model for them or for an image, video or music job used an
  older figure. They now wait for that reading.
- **Pulling a ControlNet or a textual-inversion embedding no longer fails after the download.** `localm pull` (and the Models page's Add a model, with no type chosen) crashed once the file was on disk, so nothing was registered and a second pull crashed the same way. The file is now typed from its tensors and registered as a ControlNet or an embedding.
- **Cancelling a download on the Models page says what was actually deleted.** The status line used to claim the partial download was deleted every time, including for CivitAI files and whole-repo downloads whose files stay on disk, and when the delete request failed. It now reports the case.
- **Restarting a ComfyUI that localm launched waits for the old one to release its port.** The restart used to be refused with "localm did not start it" while the process that had just been stopped still held the port.
- **`localm pull` prints progress when its output is piped or redirected.** With stdout not a terminal it showed nothing for the whole transfer; it now writes a plain progress line to stderr every few seconds.
- **Native image and video generation works when other programs hold most of the VRAM.** On
  Windows with the Vulkan or ROCm runtime, a model loaded with little free VRAM was given a VRAM
  limit too small for any part of it, and every generation failed at once with "diffusion model
  compute failed". When the runtime refuses to run within that limit, localm now says so, loads
  the model again without the limit, and generates. On the Vulkan runtime the limit also leaves
  more room below the free VRAM, so a model larger than the free VRAM (FLUX.1-dev Q8 with about
  8 GB free) no longer runs out of VRAM part-way through sampling.
- **Export saves in the app only where it can.** A key with access to the computer's files but
  without permission to change settings got a folder picker and then an error instead of a file,
  and the owner on another device (a phone, a second PC) had the file written on the computer
  running localm instead of downloaded. Export now opens the folder picker only for a key that
  may save there and only in a browser on the computer running localm; everywhere else it
  downloads the file.
- **The MCP server no longer picks a speech-to-text model for a request that names no model.**
  Without `--model` or a default model it took the first registered chat model by name, so a
  Qwen3-ASR or Whisper model could answer; those models now come last.
- **The pin-currency check no longer fails on builds localm cannot adopt.** The llama.cpp and
  AMD (ROCm) runtime checks measured how far behind the pin is against the newest upstream
  release even when localm's pin update skips it (a ROCm build with no matching upstream
  release, a llama.cpp tag the Linux CUDA build repository has not released), so the check could
  turn red with nothing to update to. They now measure against the newest build that can be adopted.
- **A scheduled job with no model set no longer picks a speech-to-text model.** The job took the first registered chat model by name, so a Qwen3-ASR or Whisper model could answer a job that named no model; those models now come last, behind every other chat model.
- **`localm config pinned_models '["a"]'` pins `a`.** A JSON list typed on the command line was
  stored as one model name with the brackets and quotes in it, so nothing was pinned. A list
  setting given as JSON text is now read as that list.
- **`localm coder --output-format json` prints only the JSON on stdout.** A warning about an
  unreadable skill, a plugin tool or an MCP server printed ahead of the document; startup
  warnings go to stderr.
- **The coder counts the lines of a file the way `wc -l` does.** A file of 1000 lines was
  reported as 1001 in `read_file`'s header and cut-off notice, and in `write_file`'s "(N lines)".
- **A background sub-agent that ran out of turns is listed as failed.** The session's job list
  showed it as done while the report to the model said it did not complete.
- **The prompt token count follows how fetched text is read.** A request that marks part of its
  text as untrusted reported the count of a prompt in which that text's control tokens were
  parsed, which is not how it is read.
- **A scheduled job does not run a web search written inside a thought that never closes.** The
  chat already ignored it; a reply cut off mid-thought no longer triggers a lookup.
- **Listing the coder's sessions no longer stalls the server.** Closing a session that had been
  idle for a day (its browser, its checkpoint) ran inside the request and froze every other
  request until it finished.
- **The browser names a blocked popup by its address.** A window a page opened to an address that
  did not answer was listed as blocked under an internal error page's address instead of its own.
- **Pulling a whole Hugging Face folder works again.** `localm pull <owner>/<repo>` for a full model folder failed when the download finished, with "'MeterBar' object has no attribute 'set_description_str'", under the huggingface-hub release localm now installs.
- **Memory consolidation no longer stalls the chat you are typing into.** The automatic and scheduled memory passes asked a reasoning model to think through each decision, and on a model that answers one request at a time a chat message sent meanwhile waited behind it (over three minutes in one report). Those calls now run with thinking off and a reply limit, and stop as soon as a chat message is waiting, picking up again shortly after.
- **A ComfyUI that hangs on its port is named at once.** After a GPU crash a ComfyUI that localm did not launch can keep its port and stop answering. `localm comfy restart` and any generation that needs ComfyUI now say right away which process holds the port (its pid and name) and that localm will not stop it, instead of relaunching into the held port and waiting five minutes; `localm comfy status` shows the same line.
- **Closing the connection during image generation stops the job.** A client that went away from `/v1/images/generations` while the job was reporting progress did not cancel it, because the check only ran while the job was silent. The job is now cancelled within a second of the disconnect.
- **Closing the connection during text-to-speech stops the job.** A client that went away while `/v1/audio/speech` was synthesising did not cancel the job, because the check only ran while the job was silent. The job is now cancelled within a second of the disconnect.
- **`localm update` exits non-zero when it fails.** A refused update (bad signature, unreadable trust file, failed apply), a failed update check and a failed `--rollback` printed their message and exited 0, so a script could not tell the update had not happened.
- **`localm doctor` no longer tells you to reinstall the voice extra after a slow first run.** When the recording-decode check does not finish in 60 seconds (common right after an install) doctor says it could not tell and to run it again; the reinstall advice stays for a real decode failure.
- **A model worker that was killed is reported as terminated.** The Windows exit code -1 was printed unsigned as `worker exit 4294967295`; it now reads `-1 (the process was terminated)`.
- **`localm run` no longer prints a traceback when the model process dies mid-reply, and records the failed turn.** The console gets one line (the traceback is in the debug log), and the session log and transcript keep the failed turn with its error and a `finish_reason` of `error`, as the server does.
- **A tiny model no longer answers "I." to plain questions.** The chat plugin's reminder to say "I" instead of "the model" is reworded as a style note after an instruction to answer every question fully.
- **A damaged download that is already on disk is replaced, not kept.** Pulling a Hugging Face
  repo again now re-checks every weight file against the digest the hub publishes and fetches a
  damaged one again (it used to say "Already downloaded"); a damaged vision projector of the same
  repo is replaced instead of being left beside a second copy; a cached Whisper `model.bin` that
  no longer matches its pinned digest is deleted and refused at load; and `localm
  setup-embeddings` (and the Knowledge page's embedding setup) fetches the embedding model again
  when the file on disk does not match its pinned digest.
- **Privacy mode leaves no crash exit file.** After a hard crash on Windows the crash watchdog wrote a `server-crash-exit` file in the data folder even in privacy mode, and nothing removed it. It is no longer written when the run started in privacy mode, and the next start deletes any left over.
- **The launcher writes no error log in privacy mode.** `logs/launcher.log`, which holds error lines that can name your model paths, is now written only when the privacy mode is not `privacy` or "Keep diagnostics for bug reports" is ticked.
- **A rejected grammar's text stays out of the debug log in privacy mode.** When a malformed grammar was only found while the reply was generated, the native parser's error and the grammar source reached the debug log and the recent-activity view of a bug report; both now stay clear of it unless chat content is allowed in the debug log.
- **A pasted log no longer lets chat text into a bug report's log digest.** After a debug record that carries chat content, several pasted localm log lines could end the withholding early and let the chat text after them through. Only log lines stamped after that record end it now.
- **Bug reports keep ordinary words that are also your computer's name.** A machine named `server`, `desktop`, `workstation`, `gpu`, `main` or `build` had every such word in the report, and in the issue title, replaced by `<host>`. Common generic names are left alone; other machine names are still redacted.
- **The automated browser no longer lets a worker reach the network around its gate.** A
  WebSocket opened from a Worker, SharedWorker or service worker, and a `fetch()` from a
  SharedWorker, used to connect directly without passing the network policy. The
  browser now resolves no host on its own, so these fail without connecting anywhere; every
  other request is still decided by the policy as before.
- **Text-to-speech no longer downloads its voice model or voice files when network access is off.**
  With network access set to off, the browser voice (Kokoro) now refuses to fetch the model, or a
  voice it has not used before, instead of offering a one-click download, unless "Allow model
  downloads while network access is off" is on. A model and voices already downloaded still work.
- **Scheduled coder jobs and MCP coder runs no longer let a project's `.localcoder/config.toml`
  write more than your settings allow.** A project mode of `log` or `full` is ignored (with a
  warning) when your configured mode persists less, so a cloned repository cannot turn privacy
  mode into session logs for an unattended run. A project can still pin a run to a stricter mode.
- **Knowledge re-sync now removes documents indexed from a credential folder in every case.**
  Previously a document was kept when the indexed folder itself was a credential folder such as
  `.ssh`, when the file had been deleted, or when its folder was unavailable, and search kept
  returning its text. They are now removed on the next re-sync.
- **`localm coder` no longer compacts after every turn.** With a model that leaves little room for context, the coder used to be summarised again on nearly every request, and reading a codebase file by file triggered a summary per file. It now compacts only when a request would take most of the context window, brings the session down to about half of it, and first drops the output of older tool calls (the model reads a file again when it needs it) before summarising anything. Everything you typed is kept word for word across compactions, and summaries list facts instead of prose. A model server that reports no context window, such as an online API, is no longer compacted at 4096 tokens.
- **A vision projector is no longer offered as a model.** A pulled vision model registers its
  projector (`...-mmproj`) as a separate entry, and `GET /v1/models` listed it as a chat model.
  A request naming it unloaded the working model and then failed to load the projector. The
  projector is now left out of `/v1/models`, and a chat, load or switch request that names it is
  refused with a message naming the model it belongs to, before anything is unloaded.
- **Image, video, music and speech components can no longer be loaded as a chat model.** A VAE,
  text encoder, LoRA, upscaler, image or video checkpoint, or text-to-speech model could be named
  in a chat, load or switch request, which unloaded the working model (or failed with an internal
  server error) before the load failed. Such a request is now refused up front with a message
  saying what the entry is. The image and video components are also left out of `GET /v1/models`;
  text-to-speech models stay listed, since `/v1/audio/speech` takes them by name.
- **"Delete saved data" also removes the coder's hook and MCP trust records** (`coder-hook-trust.json` in the data folder).
- **A short request no longer throws away a running conversation's cached prompt.** A summary, title or other small request sent to the same model between two turns used to take over the conversation's cache, so its next turn processed the whole prompt again. It now uses another slot, and the conversation continues from its cache.
- **A long conversation is no longer summarised again on every message.** When a chat client sends its whole history each turn and the server has to compact it to fit the context window, the server now reuses the summary it already wrote while the conversation still fits, and later only summarises the newly older messages. Each compaction also brings the conversation down to about half the window, so the next one is many turns away instead of one.
- **Long conversations on Mistral-family (tekken) models no longer fail with "Send less text
  per request".** The pre-tokenizer length limit now applies to each message, the way llama.cpp
  tokenizes a conversation, instead of to the whole conversation. When one message the coder built
  (tool output, file contents) is still too long, the coder shortens it and retries; only text you
  typed is reported back to you, with what to do about it.
- **The coder makes the change instead of printing it.** When you ask for a change and the coder
  writes nothing (it puts the code in its reply, says the change is already there, or only runs
  the tests), it is told once that nothing was written; a run that still ends with nothing
  written is reported as not finished instead of "Task finished". A reply of only
  `[no tool needed]` no longer leaves you with an empty answer.
- **Coder edit tools are harder to get wrong.** `edit_file`, `edit_files` and `patch_file` no
  longer apply an edit that would break a Python or TOML file that was valid (the file stays
  unchanged and the agent sees the region its edit would have produced), and a replacement written
  without the indentation of the line it lands on is re-indented when that keeps the file valid.
  Argument names other coding tools use (`old_str`, `new_str`, `file_path`) are accepted, as are
  arguments written next to `args` when the tool takes them; a call with missing arguments gets a
  reply naming them with an example, a missed `old` text says how to add new text, and identical calls in one reply run once.
- **A coder task no longer ends on "let me run the tests".** A reply whose last line announces a
  step it did not take is sent back once; a run that still stops there, or answers with nothing,
  is reported as not finished.
- **A model whose context does not fit no longer crashes the loading process.** When llama.cpp
  could not create the context (for example when a GPU ran out of memory for it), the model was
  freed twice, which on Linux killed the loading process with a segmentation fault ("exit code
  -11") instead of reporting "Failed to create llama context" with advice.
- **Large split models plan their GPU placement correctly.** A split GGUF whose first part holds
  only metadata, the layout big models such as Qwen3.8-Flash-Next ship in, was sized as if every
  byte went to the GPU: the embedding tables that stay in system RAM were counted as VRAM, the
  Mixture-of-Experts experts could not be kept in RAM, the multi-GPU split was not planned and the
  KV cache was over-estimated, so only a few layers were offloaded across every GPU. Such models
  are now placed with every layer on the GPUs and the experts that do not fit kept in system RAM.
- **Multi-GPU loads use all the cards for a Mixture-of-Experts model.** When the experts of some
  layers stay in system RAM, the remaining layers are packed onto the GPUs by size instead of by
  free memory alone, so a small last card no longer forces most experts off the GPUs.
- **A failed load no longer blames low VRAM when the planned load fit,** and the advice names a
  GPU layer count that fits instead of a fixed `-g 24`. A worker crash in privacy mode says that
  privacy mode kept no crash trace and how to keep one.
- **A textual-inversion embedding downloaded from CivitAI now registers as an image embedding**,
  not as a text embedding model, so it no longer appears as an embedding model in the model list
  Ollama clients see. One downloaded earlier is still offered on the Images page; `localm
  set-type <name> textual-inversion` also takes it out of that list.
- **A coder shell command that starts with a quoted program path now runs on Windows.** A command such as `"D:\tools dir\tool.exe" --flag "a b"` used to fail with "is not recognized as an internal or external command", because `cmd` dropped the first and last quote.
- **A shell command's timeout or cancel in `localm coder` is no longer held up by a stray
  background process.** A command that left a detached process holding its output open used to
  keep the timeout, the cancel and even a normal finish waiting until that process exited; they
  now return within a second, with the output read so far.
- **`localm coder` no longer floods the terminal.** A new file shows a short preview instead of
  every line, a reply is no longer printed twice, and a failed tool shows the end of its error
  output without needing `/verbose`. Warnings at the end of a task now appear in the terminal.

- **The Hugging Face token saved in Settings is used for every download.** Model pulls, projector
  and embedding-model downloads, speech-model downloads and the model-type lookup now send it, so
  the "unauthenticated requests to the HF Hub" warning no longer appears when a token is saved,
  and gated models and the higher rate limits apply.
- **An interrupted download resumes from the partial file.** A partial file written by an older
  localm or by Hugging Face's own tools was ignored and the download started again from zero,
  leaving the old file behind; it is now continued, and a leftover from an older upload of the
  file is removed.
- **The coder's shell safety check now sees every line of a multi-line command.** A dangerous
  command on the second line of a shell command (for example `rm -rf /` or `git reset --hard`
  after a line break) was not checked and could run. Each line is now checked like a command
  after `;`.
- **`localm coder TASK --output-format json` prints only the JSON document on stdout.** The reply text and the session record used to be printed on stdout ahead of the document, so parsing stdout failed; they now go to stderr.
- **A chosen llama.cpp backend is no longer replaced without your say.** Choosing CUDA in the
  installer or in Settings could end with Vulkan installed and shown as if it were your pick. Now
  CUDA is installed as asked; when it cannot be used (an old driver, no usable NVIDIA GPU, a
  library that will not load, or a CUDA build that would only run on the CPU), setup says exactly
  why and installs Vulkan only so localm keeps working. Your choice and the reason stay on record:
  `localm doctor` and Settings show "you chose cuda, vulkan is installed", with buttons to retry
  CUDA or keep Vulkan, and `localm update` keeps the record. An old NVIDIA driver now asks before
  Vulkan is installed.
- **Settings and `localm doctor` show the llama.cpp runtime that is actually loaded.** When the
  llama.cpp binary folder setting or `LLAMA_CPP_LIB` points at another build, localm loads that
  build instead of the one setup installed; setup, `localm doctor` and Settings now say so.
- **Image prompts with letters such as ß work with the native image backend on Windows.** A prompt
  containing ß failed with "failed to encode prompt", and letters such as ä, ö and ü reached the
  model garbled; the prompt now reaches stable-diffusion.cpp unchanged.
- **VRAM is given back and used again during a conversation.** A context that had grown for a
  long conversation kept its size and its VRAM after the conversation was compacted; it now
  shrinks once the conversation needs at most half of it. A context whose cache had to move to
  system RAM because VRAM was short, where replies run several times slower, now moves back to
  VRAM once there is room.
- **A model loaded partly on the CPU moves onto the GPU once VRAM frees up.** With automatic GPU
  layers, a model that loaded partly on the CPU because another program held VRAM is reloaded
  on the GPU at a later request once enough VRAM is free. An explicit GPU layer count is never
  changed, and a model that would not fit even in the GPU's total VRAM is not re-checked for
  free VRAM every 30 seconds, so its replies do not wait for those readings.
- **Replies speed up again after Windows moved part of a model out of VRAM.** When another
  program takes VRAM, Windows can move part of a loaded model to system memory, and replies
  stayed several times slower even after that program closed. When a reply decodes far slower
  than the model managed before at the same context length, localm now rebuilds the model's
  context before the next request, which restores the speed without reloading the model. This
  applies on Windows to models loaded fully on the GPU, never while other replies are running,
  and at most once a minute; on other systems a slow reply never rebuilds the context.
- **On Windows, a reply beside another program's VRAM use keeps the model in VRAM.** When other
  programs use VRAM, Windows lets each program keep only part of what is free and moves the rest
  of its memory to system memory, where replies decode about half as fast. When the whole reply
  budget (`max_tokens`) would go past what Windows lets the model keep but the prompt plus the
  first 512 tokens of the reply fits, the reply now starts in the context the model already has,
  so a short reply keeps full GPU speed.
- **The Live tuning VRAM estimate counts the running model's own memory.** The VRAM the running
  model holds was counted as used, so settings that fit read "may not fit"; it is now shown
  beside the free VRAM and counted as available. Apply now reloads the running model with the
  new settings instead of only saving them for the next load.
- **A chat reply that was cut off says so.** A reply that stopped because the model was unloaded or
  replaced while it was writing, or because the reply stream ended before the model finished, used
  to look like a finished reply. It now carries a cut-off note in the chat, and API clients get
  `finish_reason: "error"` with the reason instead of `"stop"`.
- **"continue" carries on with the previous reply.** "continue", "weiter", "continúa" and the same
  word in other languages continue the last reply instead of searching the selected Knowledge
  collection for the word.
- **The MCP server answers a line it cannot parse.** Malformed JSON, a number or id with more than
  4300 digits and JSON nested thousands deep used to get no reply, so a client waited forever. They
  now get a JSON-RPC parse error (-32700) and the server keeps serving.
- **`/v1/models` lists whether each model sees images or hears audio.** The list entries carry the
  same `vision` and `audio_input` fields as `/v1/models/{id}`.
- **The audio transcription error no longer tells you to install Whisper when it is installed.**
  Asking an audio model for `srt`, `vtt` or `verbose_json` now says to send model `whisper-1` when
  faster-whisper is present.
- **`localm setup-embeddings --model` refuses a reranker.** A registered reranker or a reranker
  GGUF path is rejected before anything is saved, and the re-embed reminder still appears when your
  collections have no vectors yet. A collection holding vectors from more than one embedding model
  is marked as mixed in the switch warning.
- **Qwen3 sequence-classification rerankers score the prompt they were trained on.** A Hugging
  Face Qwen3 reranker folder now gets the instruction, query and document template its model card
  specifies instead of a plain sentence pair, which separates relevant from irrelevant passages
  much more clearly.
- **A refused `localm rag add` leaves no empty collection behind.** Pointing a new collection at a
  credential folder or a UNC path removes the collection that the failed add had created.
- **A model folder named like the built-in embedding model no longer replaces it.** `/v1/embeddings`
  for `bge-small-en-v1.5` is answered by the pinned embedding model, and the log says when a
  registered folder with that name is being passed over.
- **Memory recall ignores a word that most facts contain.** A query such as "what database does
  Project Atlas use" no longer brings back every fact that says "use" behind the one that matches.
- **`localm spec-drafts` leaves out a model file cut short**, and the load response and the
  model-switch toast for a model with experts in system RAM say how much of that RAM each
  generated token reads.
- **Saving a token in Settings and opening a Knowledge collection no longer stall the server.**
  The credential write of a config save and the collection open of add, upload, re-embed, repair
  and remove-document run off the event loop. A numpy failure while scoring a collection is now
  logged instead of silently slowing every query.
- **A conversation deleted while the list loads is no longer reported as unreadable.**
- **The same speech seed gives the same audio.** `localm speak --seed N` and the `seed` field of
  `/v1/audio/speech` gave a different recording on every run on a GPU. The speech model's
  projector now draws each code from a fixed order of candidates, so the same seed gives
  identical audio from run to run on the GPU too, at full GPU speed. A seed noted with 0.3.0
  can give different audio than it did there.
- **A FLUX LoRA from XLabs is a LoRA.** It is no longer offered as a FLUX model in the Images
  model picker; it is listed with the other LoRAs.
- **A plugin install that is interrupted can be repeated.** A `localm plugin install` that dies
  mid-copy no longer leaves a half-copied plugin folder that the next install answers with
  "already installed"; the copy is built next to the plugin folder and moved into place once it
  is complete, and a folder without a manifest left by an older interrupted install is replaced.
- **`localm pull` no longer replaces a model you registered in place.** A pull whose name is
  already registered to a file kept outside the models folder (`localm add` without copying)
  stops and says so instead of replacing the registration; pick another name with `--name`.
- **Importing an A1111 image fills Clip skip and says what it left out.** The Images page's
  Import parameters takes the Clip skip value from a Stable Diffusion WebUI image and reports
  Hires fix settings as not applied instead of dropping them silently.
- **Cancelling an image or video job no longer prints the engine's raw failure lines.** The
  "sampling failed" lines the stable-diffusion.cpp engine logs while it stops are kept in the
  debug log. A second out-of-memory retry while decoding says "Still not enough VRAM" instead of
  repeating the first message.
- **`localm image`, `video` and `music` no longer ask "Open it now?" when input is `NUL`.** On
  Windows the null device passed for a terminal.
- **The message for a refused ComfyUI address no longer ends with an internal tag.**
- **A model localm will not load is reported as a refusal, with the real reason.** An
  unsupported format, a model that is not a chat model (an ACE-Step component, a vision
  projector), a draft head or a damaged file now print the reason and exit 1 instead of "Sorry -
  localm hit an unexpected error". A GGUF that llama.cpp cannot read (for example one with an
  invalid alignment) is named as a damaged file instead of blaming the runtime, a GGUF that
  cannot be opened (no read permission, or another program holds it open) says so and names the
  system's reason instead of calling the file damaged, and a runtime
  ABI mismatch is reported as a runtime problem with `localm setup-llama --force`. A GGUF in an
  Ollama store, a llamafile or a Jan folder that is refused for its version says so instead of
  "not a GGUF file", a truncated llamafile is named as one, and `localm list` shows a lone
  `.safetensors` file as `safetensors`. A huge number in a model's config is shortened in the
  load error, and a model that fails to load prints one error line on the console instead of a
  Python traceback (the trace goes to the debug log).
- **Setup recommends the right backend on Linux ARM64 with an older NVIDIA GPU.** Only
  Blackwell-class GPUs have an ARM64 CUDA build, so a Jetson Orin or GH200 is now recommended
  Vulkan instead of CUDA that then installs Vulkan anyway. A refused uv download no longer
  leaves an uninstall note that setup stopped while creating uv.
- **`localm key list` on an unreadable `auth.json` says so.** It no longer prints "No named
  keys" for a keystore it could not read, and exits non-zero.
- **A settings save that cannot get the lock answers "busy".** When another localm process holds
  the settings file, the save returns 503 with a short message and nothing is saved, instead of a
  500 and a traceback in the console.
- **`localm doctor` tells a broken torch from a missing one.** A torch that is installed but
  raises on import is reported as such, and the Hugging Face worker check still runs.
- **`/web` with no model loaded says so before it searches.** It used to run the search and then
  show a raw "Model parameter is required" error; it now gives the same "No model loaded" answer
  as a plain send, and a send made while a model is still loading says which model is loading.
- **A page reloaded while a model is loading follows the load.** The sidebar shows "loading
  <model>" and checks again every 2 seconds, instead of showing "No model loaded" and refusing
  sends for up to 30 seconds.
- **A plugin installed with `localm plugin install` while the GUI is running works at once.**
  Its routes are mounted the next time the page reads the plugin list, so `/web` no longer
  answers "Method Not Allowed" until a restart.
- **A plugin's tab opens right after you install it.** The tab entry used to appear before the
  plugin's page was loaded, and clicking it showed Chat until a reload.
- **`localm key create` warns when no owner key is set.** The new key turns authentication on, so
  the GUI asks for a key and the scoped key is the only one that exists; the note says to run
  `localm key generate` first.
- **`localm serve` and `localm gui` no longer start with a speech-to-text model.** With no model
  named, a registered model whose name or file name has a standalone `asr`, `whisper` or `stt`
  word (Qwen3-ASR) is picked only when no other model is eligible.
- **The usage line explains why MTP is off on a turn with audio.** It used to show the raw reason
  word.
- **A GGUF model loads several seconds faster.** Checking that a model fits and sizing its load
  (GPU layers, the VRAM check, the GPU placement and the context ceiling) now read the GPUs once
  instead of once per step, unless a model has to be unloaded first or another load runs in
  between; on Windows each reading starts a separate process. The VRAM usage line in the server
  log no longer holds up the load: it is printed just after "Model loaded". A small model's
  first reply after loading arrives in about 5.5 seconds instead of about 19 on an RX 6900 XT.
- **Loads that spread over two or more GPUs are no longer sized against one of them when torch
  is loaded in the server.** A model that fits across the GPUs is no longer refused as too large
  for "this GPU" when the combined reading is not available.
- **On an Intel GPU with the SYCL runtime and an XPU build of torch, the GPU list no longer
  shows a made-up adapter** once the runtime is loaded.
- **The automatic context ceiling counts the MTP draft context's attention scores per token.**
  A large automatic context on a model with an MTP draft no longer leaves that memory out.
- **A text-to-speech GGUF registered as a chat model by an older version gets the Q8_0 speech
  projector** when the model folder is synced, not the f16 one the CPU backend cannot run.
- **A low-VRAM warning that names another localm instance's model prints the name as text.** A
  name with square brackets no longer loses them or stops the warning with an error.
- **With a LoRA adapter attached and part of the model on the CPU, the memory-map decision
  counts the weights kept in system memory correctly.**
- **`/v1/embeddings` checks whether a model is a folder off the server's event loop,** and an
  embedding model entry with no path is no longer read as the current folder.
- **`localm coder --url` to a server that reports no context window compacts again.** Ollama,
  LM Studio and similar servers got no compaction at all, so a long session grew until the server
  truncated the prompt. The agent now budgets against the `n_ctx` of your localm config, a
  llama-server reports its real window, and llama-server's "exceeds the available context size"
  refusal is recognised so the request is compacted and sent again.
- **A background sub-agent no longer skews the main agent's compaction.** Both read the token
  count of "the last request" from one shared connection, so one could record the other's.
- **A request to show or find code is no longer reported "not finished".** "Show me the code that
  loads the config" or "Find where the file is created" made the agent write a file nobody asked
  for, or ended the run as failed. Images generated with `generate_image` and work handed to
  sub-agents now count as work done.
- **`localm coder doctor` no longer calls a working API key rejected.** A key that cannot list
  models (HTTP 403 from the model list) is a warning that says only the list was tested.
- **Reviewer, researcher and test-writer sub-agents can read back the cut part of a long output**
  they were told to read back.
- **Deleted files past the project map's detail limit stop being counted** in the directory counts
  and the "not indexed in detail" line without waiting for `/reindex`.
- **`localm coder --output-format json` run from a terminal on Linux or macOS no longer mixes the
  auto-started server's output into stdout,** and the pipe it writes to is no longer held open by
  the server after the run ends.
- **Idle coder sessions and their agent browsers are closed after 24 hours again** when only the
  Browser tab is in use.

### Security

- **A model folder's hub-cache link is refused when its `blobs` folder is itself a link out of the repo.** A sentence-transformers or Hugging Face snapshot whose `blobs` folder pointed elsewhere let a file link inside the snapshot reach files outside the model folder.

- **Removing the last API key no longer opens a network-bound server.** A server reachable
  from the network (`-H 0.0.0.0` or a LAN address) whose key is cleared or revoked while it runs
  now refuses every request until a key is set again, instead of serving chat and model reads to
  anyone on the network. `/health` and `/whoami` still answer, a server started with `--insecure`
  keeps running open, and on such a server the GUI's Clear key and the revoke of the last device
  key are refused with a note to set a new key instead.
- **`localm key clear`, and `localm key rm` of the last key, need you at a terminal.** They always
  ask for confirmation, `--yes` included, and refuse piped input, so a script or another program
  cannot switch the server to open mode.
- **The coder asks before it touches localm's own keys and settings.** With auto-approve on, or in
  an unattended `localm coder` task, a shell command or file tool that reads, writes or deletes
  localm's key files, TLS private keys, session store, `config.json` or instance registry (or
  the data folder as a whole), or that runs `localm key` or `localm config <key> <value>`, now
  needs your approval of that exact call; an "always allow" in the GUI does not cover it, and an
  unattended run refuses it. A project-wide grep or search and replace skips those files and says
  so. Shell commands the coder runs no longer read from its terminal.
- **A Hugging Face model can no longer make localm read files outside its folder.** The `path`
  of each stage in a sentence-transformers model's `modules.json`, and each shard file a model's
  `model.safetensors.index.json` names (read when sizing the model and when loading it as a
  reranker), is now checked before anything is opened: a path with `..`, an absolute or drive
  path, a Windows network share (which could send your Windows sign-in hash to that server) or a
  link leading out of the folder makes the load fail. A sharded model added in place from the
  Hugging Face cache is no longer refused as pointing outside its folder.
- **Another user on the machine can no longer make your localm instance unload its models.** A running instance asked to release its VRAM for a sibling now acts only when the sibling listens under your own OS account, and a request it confirms cannot be reused for anything else. A sibling listening only on `::1` can now ask too. The instance status a sibling reads no longer gives model file paths, digests or the process id to a caller without credentials, and on a key-protected instance it no longer names the loaded models either.
- **A key that is not the owner's no longer sees your project folders.** The instance list
  (Settings, Instances) shows each instance's project folder only to the owner; any other key
  sees the instances without it. The identity check any local program can make (`/whoami`) no
  longer names the project folder at all; another instance of yours still shows it. A
  non-owner key allowed to read model details gets the instance status without the project
  folder and with model file paths shortened like everywhere else.
- **A token typed into config.json by hand is no longer handed out.** localm never reads
  `hf_token` or `civitai_api_key` from config.json (they live in their own store), but a copy
  written there by hand was returned by the settings API and kept on every save. It is now left
  out of every settings response, removed from config.json on the next save, and a warning names
  the key. Set it with `localm config hf_token <value>` or in Settings.
- **Folder and file settings refuse network shares and device paths.** The llama.cpp binary
  folder, the ComfyUI folder and output folder (also per media plugin) and the TLS certificate and
  key paths no longer accept a `\\server\share`, `\\?\` or `\??\` path, which could send your
  Windows sign-in hash to that server; the save fails with a message naming the setting.
- **The coder's `read_env` hides every email address.** One address could be shown as it was.
- **The automated browser's pages can no longer reach the network over WebRTC.** A page's peer
  connection (STUN, TURN over UDP, connectivity checks to an address the page names) could send
  UDP packets past the network gate; the browser now sends none.
- **The automated browser stays off the network in an installed Google Chrome or Microsoft Edge
  too.** With the System engine, Chrome and Edge ignored the WebRTC setting and a page's peer
  connection could still send UDP packets past the network gate; localm now applies the setting
  through the throwaway profile it launches them in, which is deleted when the session stops.
- **A key limited to reading or writing settings can no longer see or change the ComfyUI output
  folder.** `comfy_output_dir`, and the same folder set per media plugin, is now owner-only like
  the other host paths: it is left out of what such a key reads, and saving it needs the owner key.
- **`localm setup-llama` makes no download when `net_mode` is `off`.** `setup.sh` and `setup.bat`
  fetched the llama.cpp runtime with network access switched off. `setup-llama` now refuses to
  download, names `net_mode`, and still installs a runtime from a local build (`--from`) or keeps
  the one already installed. `net_allow_model_downloads` lifts the refusal as it does for model
  downloads.
- **The coder's read-only shell shortcut no longer runs project programs or writes files without
  asking.** A command skips the prompt only when it is a bare `grep`, `cat`, `ls` and so on that no
  project file can stand in for, with no option that runs a program, writes a file, reads a file
  the option names or follows links (`git grep -O`, `rg --hostname-bin`, `sort -o`, `tree -R`,
  `--files0-from`, `grep -f`, also attached, clustered or shortened), with every path in it inside
  the project once links are followed (a `..` after a link or Windows junction included, so a
  path that climbs out through one asks first), and, for git, with no repository setting or hook
  that runs a program (an fsmonitor hook, a filter, a diff driver, a pager, a hook defined in the
  git configuration, or a `post-index-change` hook, the one hook `git status` and `git diff` run,
  also in a `core.hooksPath` folder or in a submodule's repository). Anything else asks as before.
  Hooks those commands never run (git-lfs, pre-commit, husky, lefthook) no longer turn the shortcut
  off, and it now also covers Windows paths written with backslashes (`type D:\app\README.md`),
  real options that only start like a risky one (`grep --exclude=`, `git diff --text`) and git
  options of other subcommands (`git status -uno`, `git log -S`, `git log -M`).
- **The coder's shell safety check reads a command the way both shells do.** A backslash at the
  end of a line joins it to the next, so `git reset \` followed by `--hard` on the next line is
  refused, and quoting that `cmd.exe` and `sh` read differently (single quotes, `\"`, `^"`) can no
  longer hide a second command from the check or from the protection of localm's own files.
- **Trusted project hooks ask before running once file writes are allowed for the session.**
  Answering "a" to a file write in the terminal, or "always allow" in the GUI, now gives every hook
  run the confirmation `run_shell` needs, as auto-approve already did, so a hook cannot run a
  script the model has just rewritten without asking. That prompt (or GUI card, now named
  `run_hook`) approves that one hook run only: it offers no "always allow", so it can no longer allow
  `run_shell` for the session, and hooked tool calls in one reply ask in turn instead of being
  refused. The trust prompt (terminal and GUI) shows
  invisible and direction-changing characters in hook commands as `\uXXXX` escapes, the coder asks
  before touching the hook and MCP trust store, and an "a" answer applies only to the prompt it
  was given at.

## [0.3.0] - 2026-10-14

### Added
- **CUDA on ARM64 Linux.** An ARM64 Linux machine with a Blackwell-class NVIDIA GPU now gets
  the CUDA build and its runtime instead of Vulkan. Older NVIDIA GPUs on ARM64 Linux still get
  Vulkan, because upstream builds ARM64 CUDA for the 13.x line only.
- **Plugins can tell who is calling a route.** The plugin host offers `caller_is_owner(request)`,
  `caller_principal(request)` and `caller_owns(request, owner)`, so a third-party plugin can
  keep a resource to the key that created it, or an action to the owner, without reaching
  into localm's server internals.
- **Sentence-transformers folders embed the way their model defines.** A Hugging Face folder
  with a `modules.json` is registered as an embedding model, and its Pooling (mean, CLS, max,
  last token and the other modes), Dense and Normalize modules are applied, so a CLS-pooled
  model such as bge no longer gets mean-pooled vectors. A folder that lists any other module,
  or keeps its Dense weights in a pickle file, is refused with the module named.
- **Cross-encoder rerankers in Hugging Face format.** A folder whose model is a
  `*ForSequenceClassification` with a single output (for example bge-reranker-v2-m3) is
  registered as a reranker and scores `POST /v1/rerank`, `localm rerank` and Knowledge
  reranking with the head's raw logit, on the CPU or the GPU like the GGUF rerankers. A
  checkpoint saved without its classification head is refused, and neither a classifier with
  several labels nor a single-score model that ships a chat template (a reward model) is
  offered as a reranker. A reranker whose worker stopped waits for a model load in progress
  before it starts again, and one that cannot start again is retried after a minute instead of
  on every query.
  A Qwen3-Reranker sequence-classification conversion, which ships a chat template and
  declares its yes/no token mapping, still counts as a reranker, and a folder stored as a
  reranker that no longer counts as one is filed as a fresh registration of it would file
  it, not as an embedding model.
- **Import models from StabilityMatrix, and type them by content.** Import from ComfyUI also takes a StabilityMatrix install, data or models folder. Every model file found is filed as a diffusion model, text encoder, VAE or LoRA by what is inside it, not only by the folder it sits in, and ComfyUI's `diffusion_models` and `clip_vision` folders and StabilityMatrix's `StableDiffusion`, `DiffusionModels`, `TextEncoders`, `ClipVision`, `Lora` and `VAE` folders are recognised in any letter case.
- **Live preview while a native image or video renders.** The Images and Video pages show a
  small picture of the image taking shape under the job log, refreshed every few sampling
  steps, so a slow render does not run blind and a bad composition can be stopped early.
  The preview is held in memory only and never saved, including in privacy mode.
- **Steps and clip skip on the Images page.** The Advanced fold takes both for the next native
  image, with the model family's step count as the placeholder. ComfyUI jobs that send either
  are refused with a reason, because ComfyUI takes them from its workflow.
- **A negative prompt at CFG 1 says it has no effect.** FLUX and other models that default to
  CFG 1 skip the negative-prompt pass entirely, so a negative prompt changes nothing there. The
  native backend's job log says so and suggests raising CFG.
- **Component pickers list what fits first.** Choosing a model on the Images or Video page
  lists the text encoders and VAEs that fit its family first; files of another family are still
  offered, marked "(other family)".
- **Add models from Jan.** `localm add` on Jan's data folder (or its `llamacpp/models`
  folder) registers every downloaded model under its Jan id, with its vision projector
  attached, and follows a model that Jan only links to a file kept elsewhere. A download
  still in progress is skipped. A copy or move out of Jan's folder is refused, because every
  weights file in it is named `model.gguf`.
- **Models quantized with compressed-tensors load.** A Hugging Face checkpoint whose
  `config.json` declares `quant_method: compressed-tensors` is expanded to full precision
  when it loads, the way FP8 checkpoints are, and the load output says so. The `hf` and `gpu`
  extras now include the compressed-tensors package, and a torch built without
  `torch.distributed` (the ROCm build for Windows) can import it. A checkpoint whose expanded
  weights do not fit the free memory is refused with the size it needs, as an FP8 one is, and
  an NVFP4 or MXFP4 checkpoint also runs when loaded on the CPU.
- **Your own image and video models in the native backend.** The built-in stable-diffusion.cpp
  backend now runs a diffusion model you pick, not only the ones localm downloads for you: a
  FLUX.1, Chroma, SD3, SDXL, Qwen-Image, Z-Image or Wan model file is recognised from its
  contents and gets its text encoders and VAE from your registered models automatically (a CLIP-L
  fine-tune, a T5 or UMT5 encoder, a matching VAE). A choice you make for a component always wins,
  and a component that is missing is offered as a download before generating. Steps, CFG and size
  follow the model family when you leave them blank. A UNet-only Stable Diffusion 1.x or 2.x file in
  ComfyUI or diffusers layout is recognised too, paired with its CLIP-L (SD 1.x) or OpenCLIP-H (SD
  2.x) text encoder and a VAE, and a missing OpenCLIP-H encoder is offered as a verified download.
- **Batch, hires fix, sampler and upscaling in the native image backend.** The Images page can
  make several images in one job (each with the next seed, saved as it finishes), add a hires-fix
  pass (latent, Lanczos, or an ESRGAN upscaler model you registered), and pick a sampler and
  scheduler by name (the Video page gets the same two dropdowns). A gallery image has an Upscale
  action that makes a larger copy with a registered upscaler model, and Reuse settings now restores
  the model, LoRAs, size, sampler, scheduler and hires settings. Upscaler files have their own
  model type, `upscaler`.
- **LoRAs in the native backend.** Images and Video take one or more registered LoRAs with a
  strength each; a LoRA made for another model family is applied with a note saying so.
- **Model pickers on the Images, Video and Music pages.** With the native backend, each page has
  a model dropdown plus the components and LoRAs that model uses, remembered per page and applied
  to the next generation only. In Settings, the native model fields are dropdowns of your
  registered models, with a custom path still possible. An ACE-Step v1 file such as
  `ace_step_v1_3.5b.safetensors` is listed in the Music page's picker with the reason (ComfyUI
  only).
- **Parameters travel with a native image, and come back from any image.** A new Settings >
  Images switch, **Embed generation parameters in the PNG** (off by default, never in privacy
  mode), writes the prompt, seed and settings into each native image. **Import parameters from
  image** on the Images page fills the prompt, seed, size, model and LoRAs from a localm image or
  sidecar, an A1111/Forge image or a ComfyUI image, and names anything it could not apply.
- **Enhance prompt.** The Images, Video and Music pages have a button that asks the loaded chat
  model to expand your prompt for the model that will generate it (tags for Stable Diffusion 1.x,
  2.x and SDXL, sentences for the newer image models, motion and camera wording for video, style
  tags for music). The result stays editable and can be undone; with no chat model loaded the
  button says so. It asks the chat model not to think, and a reply cut off while it was still
  reasoning is reported with a clear message.
- **Register a single image model file.** `localm pull <file>.safetensors` now registers a lone diffusion model, text encoder, VAE or LoRA file, detecting its type,
  instead of asking for a HuggingFace folder.
- **A reply that depends only on the request.** `cache_prompt: false` on
  `/v1/chat/completions`, `/v1/completions`, `/v1/messages` and `/v1/responses` makes a GGUF
  model empty its prompt cache, process the whole prompt and answer on its own, after the
  replies already running. The same request then gets the same reply whatever was asked
  before it, even where the prompt cache would have changed a temperature-0 reply (unless
  speculative drafting is on, or the context window grew in between).
- **Transcription without Whisper.** `POST /v1/audio/transcriptions` now answers with an
  installed GGUF model that hears audio (for example Qwen3-ASR) when faster-whisper is not
  installed, or when `model` names one. Whisper stays the default when it is installed. This
  path returns `json` and `text`; `srt`, `vtt` and `verbose_json` need timestamps it does not
  have and are refused with a message saying so.
- **Attach an audio clip in the chat composer.** The attach button and drag-and-drop now take audio files (WAV, and MP3, FLAC, OGG, M4A or AAC with the voice extra) next to images and documents. The clip appears as its own chip with its length, plays back in the conversation, and is sent as an `input_audio` part, so a model that can hear audio, such as Qwen3-ASR, transcribes it. A file over 50 MB or a WAV over 10 minutes is refused before it is sent, and when the server refuses a clip (for example a model that cannot hear audio) its message is shown and the clip is dropped so the chat stays usable.
- **Audio clips in long chats.** A clip the model cannot decode drops out of the chat with the server's message. Compaction counts and names audio clips like images. Conversations with large clips save to the server up to 96 MB, and a conversation too big even for that tells you it is kept only in this browser. A server without an audio decoder (no voice extra) does not drop clips: the message explaining the missing extra is shown and every clip stays.
- **Token log probabilities.** For GGUF models, `/v1/chat/completions` returns `logprobs` and
  up to 20 `top_logprobs` for each token of the reply's content, streamed or not, and
  `/v1/completions` returns the legacy `logprobs` object. In a chat reply, the tokens of the
  reasoning, of tool calls and of text cut by a stop sequence are left out. A model that
  cannot report them refuses the request with a 400 instead of answering without them.
- **Log probabilities on `/v1/responses`.** A request with
  `include: ["message.output_text.logprobs"]` (and `top_logprobs`, 0 to 20) now returns the
  log probability of each token of the reply's text, with its alternatives, in the `output_text`
  part and, when streamed, in the `response.output_text.delta` and `.done` events. They were a
  400 before; a model that cannot report them still refuses the request with a 400.
- **Send audio to GGUF models that can hear it.** A chat message can carry an OpenAI
  `input_audio` part, and `localm run MODEL --audio clip.wav -p "Transcribe this."` does the
  same from the command line. A GGUF model whose projector has an audio encoder (for example
  Qwen3-ASR) transcribes the clip or answers questions about it. WAV is always read and other
  formats need the voice extra; the clip is converted to the rate the model expects. Audio
  sent to a model that cannot take it is routed to an installed model that can, or refused
  with a message saying which model to use, instead of being dropped. Model listings report
  `audio_input`, and a projector that reads only audio no longer makes a model count as
  able to see images.
- **Speech from a text-to-speech model.** A Qwen3-TTS GGUF (`localm pull
  ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF:Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` fetches it with
  its mmproj) is recognised as a speech model. `localm speak "text" -o out.wav` and the
  OpenAI-compatible `POST /v1/audio/speech` turn text into a 24 kHz WAV (or raw `pcm`),
  optionally in the voice of a short WAV recording, with a seed for a reproducible result.
  It runs in its own worker process, shows its progress, appears as loaded on the Models
  page (Unload releases it), and stops when the client goes away. Pocket TTS files are
  still refused. When a repo ships several projectors, the pull takes the Q8_0 one, then bf16, and
  f16 only when nothing else is there.
- **Images generate without ComfyUI.** A native image backend runs stable-diffusion.cpp in
  a separate worker process on CPU, Vulkan, CUDA, ROCm or Metal, installing its runtime on
  first use. With no model set up, the Images page offers to download the recommended one
  (SD-Turbo, about 2 GB); any model stable-diffusion.cpp supports can be set instead (SD
  1.x/2.x, SDXL, SD3, FLUX, Z-Image and more, as GGUF, safetensors or ckpt, with separate
  text encoders and VAE where the model needs them). The new Generation backend setting picks
  `auto` (ComfyUI when it is set up, otherwise native), `native` or `comfy`, and the Images
  page, `/api/imagine`, `/v1/images/generations`, `localm image`, the chat
  `/generate-image` command, the coder agent and the MCP `generate_image` tool all follow it.
  The Images page and `/api/imagine` take a size, and `localm image` takes `--size`. A GGUF
  image checkpoint without an architecture tag is now registered as an image model instead
  of a chat model. Before each image or video load it measures the VRAM actually free on the card,
  including what other programs hold, and gives the runtime that budget, so weights that do not fit
  are streamed from RAM; the job log shows the free VRAM at each load. On the ROCm runtime native
  images use flash attention, so FLUX steps run about a quarter faster and need less VRAM. A native
  image, upscale or video job waiting for one already running says so in its log, shows the phase
  "queued" in the activity list, and ends at once when cancelled. Per-component GPU placement
  applies to ComfyUI only; a native image job's log says so and each model loads on one GPU.
- **Video generates without ComfyUI.** The same native stable-diffusion.cpp backend makes
  short clips with Wan and the other video models it supports, written as H.264 MP4. With no
  model set up, the Video page offers to download the recommended Wan2.1 T2V 1.3B (three
  files, about 6.3 GB). Settings > Video gets the Generation backend setting (`auto`,
  `native`, `comfy`) and native model fields, and the Video page, `/api/video`,
  `localm video` and the chat `/generate-video` command all follow it.
- **The Anthropic Messages API at `/v1/messages`.** Claude Code, the Anthropic SDKs and other
  tools that speak the Messages API can point their base URL at localm and use any chat model:
  text and image blocks, a system prompt, `stop_sequences`, tool use with `tool_use` and
  `tool_result` blocks, `thinking` (returned as thinking blocks), streaming with the full
  Messages event sequence, and `/v1/messages/count_tokens`. The `x-api-key` header is accepted
  as well as a bearer token, and errors come back in Anthropic's error shape.
- **The OpenAI Responses API at `/v1/responses`.** OpenAI SDKs and tools that speak the
  Responses API can use any localm chat model: input items with text, images and audio,
  `instructions`, function tools with `function_call` and `function_call_output` items,
  `text.format` for structured output, reasoning items, and the streaming event sequence.
  `previous_response_id` continues an earlier response, kept in the server's memory only and
  only for the key that created it.
- **Structured output and more OpenAI fields on `/v1/chat/completions`.**
  `response_format` (`json_object`, or `json_schema` with `strict`) constrains the reply,
  token by token, to JSON or to the schema, also alongside `tools`. `presence_penalty`,
  `frequency_penalty` and `min_p` are applied by the sampler (GGUF and HuggingFace models),
  `max_completion_tokens` is read as the reply cap, `stream_options.include_usage` sends the
  usage in a last chunk, `reasoning_effort: "none"` turns thinking off (on gpt-oss too, whose reply
  then starts in its final channel with no analysis before it), and a `developer`
  message is read as `system`. `/v1/completions` takes the same sampling fields, `echo`, and
  a prompt list of one. A field localm does not serve (`n` above 1, `logprobs`,
  `logit_bias`, the deprecated `functions`, audio output) is now a 400 naming it instead of
  being silently ignored. A grammar, tool choice or response format on an image request to
  a GGUF vision model is now applied instead of dropped, and a GGUF model whose grammar
  sampler faulted earlier refuses grammar requests until localm restarts instead of
  answering without the constraint.
- **`localm setup-sdcpp` installs a native image generation runtime.** It downloads the
  stable-diffusion.cpp build that fits this machine (CPU, Vulkan, CUDA, ROCm or Metal) from
  a pinned upstream release through the network policy, verifies it against a pinned
  checksum, and checks that it loads and finds a compute device, falling back to Vulkan and
  then CPU when the preferred build does not load. `--backend` picks a build and `--status`
  shows what is installed. A runtime that later loses its compute-backend files is refused at model
  load with the reason and `localm setup-sdcpp --force` named.
- **One GGUF model answers several requests at once.** Concurrent chat requests to the
  same model are decoded together instead of waiting for each other, so two or four
  clients share the GPU's throughput rather than queueing (`parallel_slots`, default
  `auto`: 4, or 1 while speculative drafting is on; Settings > Engine > Parallel
  requests). The requests share the model's context window: one that does not fit beside
  the running ones waits for them and shows a waiting status. A reply generated beside
  others can differ slightly from the same request run alone, even at temperature 0.
  Stopping one reply, or a long prompt being read beside a running reply, does not restart the model
  or fail the other replies, and a reply with no token limit waits for the room the others hold. An
  audio clip is decoded before the model is taken, so the other replies keep running meanwhile, and
  stopping the request stops the decode. Changing Parallel requests lets automatic model routing try
  again a model that failed to load with the old value. An image reply, or one sent with
  `cache_prompt: false`, waits for the running replies to finish, and a request cancelled during
  that wait ends at once with no output and no error.
- **`localm setup-music` installs native music generation.** It downloads the KoboldCpp
  build that runs ACE-Step 1.5 on this machine (CUDA, Vulkan, CPU or Metal) from a pinned
  release through the network policy, checks its size and checksum before running it, pulls
  the default ACE-Step 1.5 models, and generates a short test track, falling back to Vulkan
  and then CPU (CPU on Apple Silicon) when the preferred backend does not start. `--backend`
  picks one, `--status` shows what is installed, and `--no-models` / `--no-test` skip those
  steps. Downloading the four ACE-Step 1.5 files fails and deletes the file when its bytes are not
  the expected ones.
- **Music generates without ComfyUI.** With nothing set up, the Music page, the chat `/music`
  command and `localm music` generate ACE-Step 1.5 tracks with the native backend (the
  runtime `localm setup-music` installs), and keep using ComfyUI where one is set up. The
  Music settings pick the backend (`auto`, `native` or `comfy`), the native runtime and the
  model files. The page says which backend runs and offers to download the default models;
  a generation never downloads them by itself. Tracks are WAV files with no embedded
  metadata, and inputs only ComfyUI understands are refused with the reason. A track that
  comes back broken (some GPUs produce one for some prompts) is never saved; on the `auto`
  runtime it is generated again on the CPU.
- **Knowledge results can be reranked.** With a reranker model installed (for example
  `localm pull ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:qwen3-reranker-0.6b-q8_0.gguf`),
  the best 20 matches for a Knowledge question are re-scored by it before they reach the
  chat, so the most relevant excerpt comes first more often. A switch on the Knowledge
  page and the Reranking settings control it, `localm rag query` takes `--rerank` and
  `--no-rerank`, and the query API reports whether a query was reranked. With no
  reranker installed nothing changes; if the reranker fails, the unreranked order is
  kept and the reason is shown.
  With the Q8_0 files of bge-reranker-v2-m3 (gpustack) or Qwen3-Reranker-0.6B (ggml-org),
  the reranker's score also decides which excerpts are relevant enough to reach the
  chat, so a rephrased question finds its answer where the keyword and similarity check alone would
  drop it, and unrelated questions still bring in nothing.
- **Chat follow-ups find what the conversation was about.** With a Knowledge collection
  selected, a message that points back at the previous one ("what does that resync do?",
  "the thing you said about ports") is now searched together with your previous message.
  `localm rag query` takes `--context`, and the query API takes `context` and reports
  `context_used`.
- **Set the reranker's minimum score.** Settings > Knowledge > Reranker minimum score
  (`rag_rerank_min_score`) chooses how high a reranker scores an excerpt before chat adds it, in
  that reranker's own scale, so any reranker can gate Knowledge excerpts by its score and not only
  the two measured Q8_0 files. Blank keeps the measured value, or the keyword and similarity check.
- **A loaded reranker can be unloaded from the Models page.** A reranker that is in
  memory now shows as loaded there, and its Unload button (or `localm unload <name>`)
  frees it unless a rerank is running.
- **Tool calling with any chat model.** `/v1/chat/completions` takes OpenAI `tools`,
  `tool_choice` and `parallel_tool_calls`, and the Ollama API takes `tools` and
  `tool_calls`. The model's calls come back in `message.tool_calls` (a streamed call arrives
  whole in one delta) with `finish_reason` `tool_calls`. `required` or a named function
  constrains the reply to a well-formed call whose arguments follow the function's JSON
  schema; `auto` constrains a call once the model starts one. Earlier calls and tool results
  in the conversation are given back to the model, with tool output treated as untrusted
  text.
- **Reranker models rank documents.** A reranker GGUF (bge-reranker-v2-m3, Qwen3-Reranker,
  a BERT or XLM-R cross-encoder) added with `localm add` or `localm pull` scores documents
  against a query. `POST /v1/rerank` takes `query`, `documents` and `top_n` in the Jina /
  Cohere / llama-server shape, `localm rerank` does the same from the command line, and the
  MCP server has a `rerank` tool. A file without a classifier head is refused instead of
  returning arbitrary scores, and a reranker chosen as the embedding model says what it is
  instead of producing vectors.
- **The Ollama API's `format` takes a JSON schema.** The reply is constrained, token by
  token, to documents that satisfy the schema: objects with required and optional
  properties, arrays with length bounds, enums, integer ranges, `anyOf`, recursive `$ref`
  and more. A keyword that cannot be enforced (`pattern`, `multipleOf`, number bounds)
  is refused with a 400 naming it rather than ignored.
- **`stop` sequences on `/v1/chat/completions` and `/v1/completions`.** A string or a list
  of up to 16: the reply is cut before the first match, the generation ends there instead of
  running to its token budget, and `finish_reason` is `stop`. A stop sequence inside a
  reasoning model's `<think>` block is not applied. The Ollama API's `options.stop` uses it.
- **Prometheus metrics at `/metrics`.** Turn on "Prometheus metrics" in Settings > Server
  (or set `metrics_enabled`) and restart to serve request counts and latency by route,
  prompt and generated tokens, tokens per second, time to first token, queue depth, loaded
  models and GPU memory in Prometheus text format. A streamed reply the client stops partway
  still counts the tokens generated before it stopped. It needs an admin API key, answers
  only on a loopback bind when no key exists, and carries no prompt, reply, model name or
  model path in any label. Off by default.
- **Diffusion language models (Dream, LLaDA, LLaDA-MoE, RND1) run as chat models.**
  A GGUF of one of these architectures now loads and answers through `localm run`, the GUI chat and
  `/v1/chat/completions`. These models write the whole reply at once over a number of denoising
  steps, so the reply appears when it is finished and the status line shows the progress meanwhile.
  Two settings control them: "Diffusion steps" (`diffusion_steps`, more steps give
  better text and a slower reply) and "Diffusion reply length"
  (`diffusion_max_tokens`, 256 by default). Grammar-constrained output, images and
  speculative decoding are not available with these models; a request for a
  grammar is refused with a message saying so.
- **Encoder-decoder (T5) GGUF models now run.** Flan-T5, LaMini-Flan-T5 and other `t5`
  GGUFs register as chat models and answer through `localm run`, the GUI chat and
  `/v1/chat/completions`, where they were refused before.
  These models have no chat template: a single message is read as typed, and a
  conversation is read as one labelled transcript. A prompt can be up to 2,048 tokens
  (fewer when the context window is smaller); a longer one is refused with that limit
  named. They read text only, and speculative decoding does not apply to them.
- **Files and folders localm cannot run now say why.** `localm add`, `localm pull <path>`
  and the GUI add and pull routes name the format and the nearest runnable alternative for
  GGUF version 1 and big-endian GGUF files, legacy GGML `.bin` files, importance-matrix
  GGUFs, Mistral-native folders, `.onnx`, TensorRT, LiteRT and NeMo files, and MLC,
  OpenVINO and EXL2/EXL3 folders. An MLX-quantized, EXL2/EXL3 or OpenVINO folder still
  registers, with a note that it cannot be loaded, and is refused at load with the same
  sentence before any worker starts. Direct-URL and Ollama-blob pulls apply the same
  checks, and the automatic models-folder scan quietly skips the GGUFs it cannot load. A
  Mistral-style folder holding both Hugging Face shards and `consolidated*` files is sized
  by the shards only. Old-format LoRA adapters and llama.cpp session files are recognised and
  explained.
- **Models other tools already downloaded can be added in place.** `localm add` (and
  the Models page's add-path box) now takes the Hugging Face cache folder
  (`~/.cache/huggingface`, or `HF_HOME`), registering one model per repository from
  its current revision and named after the repo; an Ollama model folder
  (`~/.ollama`, or `OLLAMA_MODELS`), registering every GGUF model in it as
  `model-tag`, with a vision model's projector attached; and a `.llamafile`, whose
  model (and projector) is unpacked into the models folder with a progress bar.
  Moving models out of the Hugging Face cache is refused, since it would break the
  cache; copy them instead.
- **OpenAI-compatible speech-to-text and image endpoints.** `POST /v1/audio/transcriptions`
  takes a multipart upload and returns `json`, `text`, `srt`, `vtt` or `verbose_json` (with
  segment and word timestamps), transcribed locally with Whisper; the audio is never written
  to disk. `POST /v1/images/generations` generates through the image plugin (ComfyUI) and
  returns `b64_json`, or a gallery `url` when an API key is configured, honouring `n` and
  `size`; in privacy mode no image is kept on disk. The official `openai` SDK works against
  both, and local apps can call them like `/v1/chat/completions`. See
  docs/server-api.md.
- **An Ollama-compatible API on the same server and port.** `/api/chat`, `/api/generate`,
  `/api/embed`, `/api/embeddings`, `/api/tags`, `/api/show`, `/api/ps` and `/api/version`
  answer in Ollama's format, so a tool that speaks Ollama can use a localm model. Replies
  stream as NDJSON, `format: "json"`, `options.stop`, `think` and images work, and the same
  API keys and scopes apply. Tool calling is not supported yet.
  `/api/copy` makes an alias; pulling, deleting and creating models stay in `localm`. See
  docs/ollama-api.md.
- **Release files carry build provenance and a software bill of materials.** The release
  zip, the sdist, the wheel and a CycloneDX SBOM of the pinned dependencies are attested by
  the release workflow, so `gh attestation verify` proves which workflow built a file and
  from which commit. SECURITY.md has the commands.
- **Docker images for the API server.** Releases publish CPU and Vulkan images to `ghcr.io/matlan1/localm`, and `docker/Dockerfile` builds the same image from a clone. The container keeps its data in a `/data` volume, serves HTTPS, and refuses to start until an API key exists (`docker run --rm -v localm-data:/data ghcr.io/matlan1/localm key generate`). See docs/docker.md.
- **NVIDIA CUDA Docker images.** Releases also publish `cuda` (every NVIDIA architecture before Blackwell) and `cuda13` (Blackwell) images, started with `docker run --gpus all`. On start the container checks that the CUDA runtime loads on the GPU it can see, and exits with the cause instead of serving on the CPU when it does not, for example with no GPU attached or with the wrong tag for the GPU. `localm setup-llama --backend cuda --cuda-line cuda-12` (or `cuda-13`) fetches the CUDA runtime on a machine without a GPU, which is how the images are built. See docs/docker.md.
- **A "Memory-map model files" setting (`use_mmap`: `auto`, `on`, `off`) and a note when
  a model runs from disk.** With `auto`, a model that may not fit in available
  RAM is memory-mapped, so it can run from disk-backed memory instead of failing
  to load; `on` and `off` force it. The load output, the model-switch toast and the
  load response now say when a model runs from disk-backed memory and that first
  tokens are slower.
- **Speculative decoding for any GGUF model, with no draft model.** The new
  Speculative drafting setting (`spec_source`) adds `ngram`: the model drafts the
  tokens that followed the same few tokens earlier in the conversation and checks
  them in one pass, so replies that rewrite a file, quote a passage or repeat
  tool-call JSON come out faster. Replies are the same model's replies. It
  measures your model when it loads and learns while it writes how far copied
  text tends to run, so rewrites draft longer stretches at once, and on models
  where checking drafts is expensive, such as Mixture-of-Experts models, it
  holds back where drafting would not pay instead of slowing replies down. It is
  off by default; `localm bench-spec <model>` measures whether it pays on your
  machine, and the reply's usage line shows how many drafted tokens were accepted. A Speculative
  drafting source other than mtp takes precedence over the Multi-Token Prediction toggle, which
  applies only while Speculative drafting is Inherit; `localm bench-mtp` says so.
- **Speculative decoding with a draft model.** Speculative drafting can now use a
  smaller model of the same family (`spec_source` `draft`, with the Draft model
  setting): it drafts a few tokens and your model checks them in one pass, so
  ordinary replies can come out faster too, not only ones that repeat earlier
  text. Replies are the same model's replies. It measures the draft and your
  model on your hardware when it loads and drafts only as many tokens as pay off,
  so a draft that cannot help is switched off instead of slowing replies down.
  It is off by default; `localm spec-drafts <model>` lists the downloaded models
  that can draft for yours, and `localm bench-spec <model> --source draft`
  measures whether it pays on your machine. When the draft model cannot draft for yours (not found,
  not a GGUF file, a different vocabulary, recurrent layers, failed to load, or slower than decoding
  without it), loading says why.
- **A "Model autoswitch" setting controls when a chat may be answered by a different
  model.** `off` never switches; `image` (the default) switches only for an image
  your model cannot read; `ask` keeps your model and offers the better one on the
  reply, where one click selects it; `loaded` switches only to a model that is
  already loaded, so nothing is loaded or unloaded; `auto` switches to an installed
  model when yours is known to lack what the chat needs; `eager` also switches when
  your model has never been checked. A chat pinned to a model is never switched.
  Turning on Web access in a chat never switches the model by itself.
- **Older vision models whose projector file does not say what kind of projector
  it is now read images.** LLaVA 1.5 projectors (`mmproj-model-f16.gguf` from
  the original LLaVA 1.5 GGUF releases) predate that field, and the bundled
  llama.cpp runtime refused them, so the model stayed text-only. localm now works
  out the projector type from the file's contents and loads a corrected copy
  kept in its cache folder; your own file is never changed.
- **Missing ComfyUI models can be downloaded even when localm has no built-in
  source for them.** When an image, video or music workflow needs a model file
  that is not installed, localm now offers to search Hugging Face for a file with
  exactly that name, shows the repository, path and size it found, and downloads
  it into the right ComfyUI models folder when you confirm. Files published by
  ComfyUI's own Hugging Face organisation are preferred. Only `.safetensors`,
  `.sft` and `.gguf` files are downloaded this way, and a model the workflow
  keeps in a subfolder of its models folder is saved in that subfolder. The
  models used by the built-in music (ACE-Step) and video (Wan 2.2) workflows are
  now offered directly, without a search.
- **Text-to-speech and speech-to-text offer a one-time download prompt instead of a
  hard refusal when network access is off.** Settings also gained direct "download
  the voice/speech model now" controls for both, so first-time setup no longer
  depends on triggering a fetch some other way. If the voice model fails to load,
  playback now falls back to the browser's own built-in voice instead of going
  silent.
- **The MCP `chat` tool takes `images`** (local image files or `data:image/...`
  URIs), and **`localm mcp --share-loaded-models`** lets the MCP server use a model
  another localm instance on this machine already has loaded instead of loading a
  second copy.
- **Switch models in running coder sessions and resume any session with any model.**
  You can now change the model powering an active coder session in place without
  starting a new session or losing history. In the GUI, click the session model
  badge in the session bar or use the Model section in session controls; whenever the
  loaded model changes (from the sidebar, another coder session, a chat or the command
  line) every local coder session that was not started with a chosen model follows it
  (a busy one once its task ends, and only to the model that is still loaded), and a session on
  another backend such as Ollama on localhost never does. A model loaded after the previous one was
  unloaded, by hand or by idle unload, is followed too. A model switch
  made from a coder session now refreshes the sidebar model list straight away.
  When loading the chosen model would evict a model that is in use, or would only
  partly fit in VRAM, the coder asks first, as the sidebar model picker does. If
  the load does not complete, starting a session reports why instead of starting
  on a model that is not loaded, and switching keeps the session on its current
  model.
  In the CLI, switch models in the REPL with `/model <name>` (or inspect with
  `/model`), and resume any past session with a specified model via
  `localm coder --resume [id] --model <name>`. `/model <name>` loads the model before it reports a
  switch: a name the server does not have is refused with the server's explanation, the coder sizes
  its history to the new model's context window, a key without the `models:write` scope is told so,
  and a backend that cannot load models (a `--url` server or an OpenAI or Anthropic provider) says
  it cannot confirm the name instead of reporting a load. `/model` with no name, and the
  active-model tag in session controls, show in the interface's own language. `--resume` followed by
  a task instead of a checkpoint id (`-r "some task"`, which is read as the id, not as the task)
  warns with the two ways to combine a resume with a task. A session resumed with `--resume` stays
  saved until its next turn runs, so leaving the REPL before sending anything keeps it in
  `/sessions` and the GUI's past sessions.
- **`localm setup-browser`** downloads the Chromium build the automated browser
  needs. Previously this required a separate, undocumented-in-app
  `python -m playwright install chromium` step after installing the browser
  extra; it is now a regular localm command that respects the network policy.
- **Live inference status indicator in chat, and a CPU fallback warning in
  chat and the CLI.** Chat shows pulsing stage text (prompt processing,
  vision image encoding, generating) with an elapsed timer during inference.
  Both chat and `localm run` visibly warn when GPU vision encoding fails and
  falls back to CPU; the CLI's warning goes to stderr, so a piped `-p` reply
  stays clean. The status text follows the interface language for the whole reply.
- **The automated browser can now mirror its live view directly inside the
  coder session driving it, in addition to the dedicated Browser tab.** Off by
  default; turn on "Show the agent's browser inline in the coder session" in
  Settings > Network.
- **The Browser tab's live view is now interactive.** Click on the screenshot
  to click the corresponding point in the open browser, scroll with the mouse
  wheel to scroll its page, and type to send keystrokes to it, including Tab,
  Enter, the arrow keys, Page Up/Down, Home, End, Backspace and Delete.
  Clicking the frame gives it keyboard focus, and Tab is forwarded to the
  browser. Esc releases the keyboard from the frame and Shift+Tab moves focus
  back to the controls above it, as a hint under the focused frame says. Clicks, keys and scrolls
  reach the page in the order you make them, and an input that does not reach the browser is
  reported in the status line. `POST /api/browser/click` and `/api/browser/scroll` refuse a position
  that is not a finite number or is beyond a million pixels with a 422, and a click's `button` must
  be `left`, `right` or `middle`.
- **Accelerated RAG vector queries, BM25 inverted index, and chat message queueing.**
  When NumPy is installed, vector search uses a normalized matrix dot product for
  fast cosine similarity (without it, search is unchanged), BM25 uses an inverted postings index for term lookups, and
  recently queried collections stay in memory (at most 256 MiB in total, least
  recently used dropped first) and are re-read whenever their files change.
  Chat now displays an immediate search status indicator with elapsed timer and
  allows queueing follow-up messages while generation or retrieval is active. Stop ends the whole
  turn, including knowledge search and summarising a long chat, and never sends the next queued
  message; queued messages stay listed until you send or cancel them, a message typed during
  Regenerate or `/web` is sent when that reply finishes, and a new message goes after any older
  queued ones.
- **Uninstall from setup.** Running `setup.bat`, `setup.sh` or the graphical
  installer again in a folder where LocaLM is set up now offers to install
  again, uninstall, or cancel (`setup.bat uninstall` / `bash setup.sh
  --uninstall` go straight there). Uninstall lists what it will remove before
  anything happens, stops LocaLM if it is still running, removes what setup
  installed in the folder and what it added elsewhere (desktop shortcut or
  menu entry, the `localm` command and its PATH entry), and asks whether to
  also delete your saved data: chats, settings, downloaded models and
  generated images. `--purge-data` and `--yes` answer those questions up
  front. Setup now records what it creates as it goes, so an install that
  stopped half way can be uninstalled too. When you delete your saved data it
  also names what it cannot remove: the app window's saved login and copies
  of recent chats, kept in a folder pywebview shares with other apps, and the
  copies your web browser keeps. A data folder another LocaLM install still
  uses, on this computer or on another one that shares the folder, is never
  deleted, and neither is a folder `LOCALM_HOME` points LocaLM at (the list
  names it). A LocaLM folder that was moved or renamed after setup is
  uninstalled completely, and a desktop shortcut or menu entry that opens a
  different LocaLM folder is kept. Choosing Install again offers to keep
  using the data folder you already have; when that folder is not available
  (a drive that is not connected, or a network folder that is offline), setup
  says so and offers to stop instead of quietly starting an empty one. The
  graphical installer cannot be closed while it uninstalls, and
  `setup-gui.bat` / `setup-gui.sh` say when something you asked to delete was
  kept or the uninstall did not finish.
- **The graphical installer follows your system's dark or light mode.**
  `setup-gui.bat` / `setup-gui.sh` now open in LocaLM's own dark or light
  colours to match Windows, macOS or GNOME, including a dark title bar on
  Windows. Set `LOCALM_THEME=dark` or `LOCALM_THEME=light` to choose one
  yourself.
- **Secret fields in Settings show their state and can be cleared.** Each API key
  or token field is tagged "(configured)", "(from environment)", "(not set)" or
  "(new value)", its placeholder says what saving will do, and a saved value gets a
  Clear button that removes it when you save, with Undo until then. The stored
  secret itself is never sent to the page. A `HF_TOKEN` or `CIVITAI_API_KEY` environment variable
  that is blank or whitespace-only counts as not set, both here and for downloads.
- **Delete past coder sessions and remove projects from the coder's session
  list.** Each past session in the list has a delete button, and each project
  has a remove button that forgets the project and deletes its saved sessions.
  Both ask first. Removing a project deletes only the coder's saved sessions
  for it, including an older one kept in the project's `.localcoder` folder;
  your project files are left alone. A session that is open, or a project
  with an open session, has to be ended first.
- **A browsable documentation site, with an API reference generated from the
  server's OpenAPI schema.** The `docs/` guides are built into a searchable site,
  and the API reference lists every route, parameter and schema the server
  declares. `scripts/export_openapi.py` writes the schema without loading a model.
  The site is published with each release.
- **GGUF LoRA adapters can be attached to a base model.** A GGUF adapter file (made by
  llama.cpp's `convert_lora_to_gguf.py`, or downloaded) now registers as type `lora`
  instead of a chat model that cannot load. `localm adapter attach ADAPTER BASE
  [--scale S]` ties it to a registered GGUF model, and `localm run BASE`, the server
  and the GUI then load BASE with the adapter applied; `localm adapter detach` and
  `localm adapter list` manage them. An adapter made for a different architecture is
  refused with both architectures named, and a loaded model's load response lists the
  adapters it runs with. Attached adapters count toward the VRAM estimate.
- **LoRA adapters are managed from the Models page.** An adapter row shows what it is
  attached to and its scale, with attach, change and detach controls; a refusal such as a
  mismatched architecture appears inside the dialog. A base model shows the adapters it
  will run with, the ones it is running, and asks for an unload and reload when they
  differ. The model details list the adapters a loaded model runs with.

### Changed
- **The native image and video runtime (stable-diffusion.cpp) moved to master-964-1d0a58a.** SD 1.x, 2.x
  and SDXL images at CFG above 1 now run the prompt and negative-prompt passes together, which makes
  each step faster for the same image and uses more VRAM while sampling. An existing
  install downloads the new runtime the next time a native image or video job starts.
- **The bundled koboldcpp runtime (native music generation) moved from v1.122.1 to v1.123.** An existing install picks it up the next time music generation starts.
- **HF models decode about 3.5x faster on AMD GPUs.** On ROCm, loading an HF model now tunes the
  matrix kernels its one-token decoding uses (PyTorch TunableOp, rocBLAS kernels): gemma-3-4b in
  bf16 on an RX 6900 XT goes from 4.3 to 15 tokens per second with identical output. The first load
  of a model on a card takes a few seconds longer; the results are cached and reused.
- **A native image, video or music job frees the other native runtime when the card is short.** The music runtime and the image/video worker each stay loaded for ten minutes after a job, so a job of the other kind started with that memory still taken. When the job needs more than the free VRAM, the idle runtime is now stopped first and the job log says so in one line. When another localm instance holds VRAM on the same GPU, the log names it (its port, chat model and loaded media runtimes).
- **`localm info` says whether the install is portable.** It prints a `mode` line (`portable`
  or `standard`) and a `temp dir` line naming where temporary files go.
- **The chat's Markdown renderer moved from marked 12 to marked 18.** Messages render the same; the visible differences are a half-typed list marker (`- `, `1. `) showing as an empty list while a reply streams in, and no stray blank line between consecutive HTML blocks.
- **localm's managed ComfyUI now installs v0.39.0, up from v0.31.1.** An existing install picks it up with `localm comfy update --reinstall-requirements` (this release's Python requirements changed).
- **Linux CUDA setup installs fixed NVIDIA runtime libraries.** `localm setup-llama` on Linux now fetches a pinned, checksum-verified version of the CUDA runtime and cuBLAS libraries instead of whichever version PyPI lists as newest.
- **Mixture-of-Experts models that do not fit in VRAM run much faster.** With GPU
  layers auto-sized (the default), such a model now keeps the experts of as few
  layers as needed in system RAM and every layer on the GPU, instead of moving
  whole layers to the CPU (with a configured multi-GPU split, set `n_cpu_moe`
  yourself). On a 16 GB card Qwen3-30B-A3B generates 26 tokens/s
  where moving whole layers gave 19. The load output, the model
  load response (`moe_cpu_layers`) and the GUI's load message say how many
  layers' experts are in RAM, and the load output says how much each token reads
  from RAM, which limits the speed. A `n_cpu_moe` or `n_gpu_layers` you set is
  used as given.
- **Qwen-VL models answer follow-up messages faster.** Qwen2-VL, Qwen2.5-VL and
  Qwen3-VL GGUF models reprocessed the whole conversation on every turn, text-only
  chats included. They now keep the part of the conversation that did not change
  and process only the rest, as other models do.
- **Follow-up messages in a chat with an image answer much faster on GGUF
  models.** Once an image has been sent, each later turn used to encode every
  image in the conversation again and reprocess the whole chat, which could take
  minutes per turn when images are encoded on the CPU. Now an image is encoded
  only once, and on most models only the previous reply and the new message are
  processed; the status line reads "Processing prompt..." when no image needs encoding. Encoded
  images are kept in memory only for the most recent image conversation and are released when the
  model unloads.
- **Follow-up messages in a chat with an image answer faster on HuggingFace-format
  vision models too** (Gemma 3, Gemma 4, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, LLaVA,
  SmolVLM). Each image is encoded once and reused on later turns: a follow-up
  encodes no image, and adding an image encodes only that one. Images are now
  encoded one at a time, so in a chat with several images a LLaVA or Gemma 3
  reply can be worded slightly differently than before. Encoded images are kept
  in memory only for the current image conversation and are released when the
  model unloads.
- **Settings' "Library" section is now called "Model Library & Sources".**
- **Web activity in the chat is its own collapsed card, not a user turn.** A
  web search, a page read, a declined or repeated request and the chat's own
  web notes are now stored in the conversation as tool events rather than as
  messages you typed, and shown as one collapsed "Web" card that opens to the
  source list (id, title, link, whether the page was read or only its snippet)
  and the evidence excerpts, with status and timing. The card can no longer be
  edited or reverted like a message, and a new message cannot be sent while
  a web request is still running. Existing conversations are converted on
  load; exports still label these entries `Web`. What the model reads for a
  completed search or page read is unchanged.
- **Web search now reads the pages, not just the search snippets.** `/web`,
  a model-requested `web_search` in the chat, the coder's `web_search` tool
  and a scheduled chat job's web lookup all run the same retrieval: search,
  read the top three result pages, and hand the model evidence excerpts
  labelled by source (`S1`, `S2`, ...) with a grounding label. The model is
  asked to cite those source IDs rather than URLs it never read, and when no
  page could be read the result is labelled `snippet-only` (or `failed`) in
  the conversation and in the prompt instead of being presented as read
  pages; `/web` also shows a notice in that case. The retrieval is available
  to API clients as `POST /api/web/retrieve`; `/api/web/search` and
  `/api/web/fetch` are unchanged, and `fetch_url` still reads a single page
  on request. `/web` now needs no web toggle to read pages; the network
  policy (`net_mode`, the domain lists, the private-address guard) applies
  exactly as before.
- **With the network policy at `ask` (the default), the chat's "Web access"
  toggle is now on for a new browser.** The model is told the web tools exist
  and every request it makes still shows the approval card before it runs.
  Before, a fresh install told the model it had no internet access until the
  toggle was switched on by hand. `off` still leaves the toggle off, and a
  choice you saved in the browser still wins.
- **The chat's web lookups are no longer capped at three per message.** With web
  access on, the model keeps searching and reading pages until it answers. The
  lookups stop when it repeats a search or page, when two lookups in a row find
  nothing new, when a lookup still cannot be read after one retry, or at the new
  Settings > Chat "Web lookups per message" limit (20 by default, 0 for no
  limit). The chat no longer sends the model a "web search limit reached ... stop
  searching" message, and its own web notes show as a one-line status instead of
  an "Instruction to the model" section.
- **Settings now translates fully into German.** Server controls, other
  running instances, Report a bug, Changelog, Updates (app update, roll back,
  the app launcher and the inference runtime), Issues, Logs and Upload files
  now show in German too, alongside the rest of the Settings page. Every
  Runtime & GPU, Timeouts, Chat, Server, Security, Coder, Desktop app,
  Privacy, Memory, Embeddings, Plugin, Update, Knowledge and Network setting's
  label and help text is now translated as well, along with their section
  headings. Owner key, Keys & devices (including scopes, presets and the QR
  pairing flow), Sign in on this device, Companion app, Diagnostics, Your
  data and Live tuning (including the live VRAM estimate) are now translated
  too, completing the page. The reranker, diffusion, parallel-request and metrics settings, every
  image, music and video generation setting, and the text-to-speech settings are translated as well.
- **Multi-Token Prediction is faster and shows how well it is working.** Each
  speculation step does less extra work, can draft several tokens at once (the
  new "MTP draft tokens" setting, `mtp_draft_tokens`, default 1), and gives the
  draft head the right context for every token. It now also runs on replies
  constrained by a grammar, which includes every chat reply while web access is
  on; those replies never used it before. It measures, while a reply is being
  written, whether drafting is actually faster than generating one token at a
  time, and pauses itself when it is not (common on small models and when
  sampling with a temperature). The chat shows the share of drafted tokens the
  model accepted next to the reply's speed, or that MTP paused, stopped, or was
  off for the reply (it does not run on image turns), the chat API reports it
  as `usage.mtp`, and `localm bench-mtp` prints the acceptance rate and whether
  the replies matched MTP off.
- **The bundled AMD (ROCm) llama.cpp runtime moves to build b1342**, from b1307:
  two months of upstream llama.cpp and a newer ROCm runtime. Run
  `localm setup-llama --force` to pick it up; an existing installation keeps
  working untouched until you do.
- **Text-to-speech runs on a newer speech runtime.** The built-in Kokoro voice now uses
  transformers.js 4.3.1 with onnxruntime-web 1.31, still served from your install so it needs
  no network for the runtime.

### Fixed
- **Embedder GPU layers set to 0 keeps the embedding model on the CPU.** A value typed in
  Settings or with `localm config embedding_gpu_layers` was stored as a decimal number that the
  embedder did not recognise, so it placed the embedding model automatically (usually on the GPU)
  instead of following the setting.
- **`--debug` starts even when the log folder cannot be written.** `localm run`, `localm serve` and
  `localm gui` with `--debug` stopped at startup with "Permission denied" when the data folder's
  `logs` folder could not be created or written. They now start, say the log file could not be
  created, and show the debug output in the console.
- **A damaged settings file no longer replaces its good backup.** When `config.json` (or the
  model registry) could not be read and localm worked from its `.bak` copy, the next save moved the
  damaged file over that backup, leaving the newer settings only in a file nothing could read. The
  damaged file is now kept as `config.json.unreadable-<time>` and the backup stays as it was.
- **`localm unload` no longer reports a model it could not unload as unloaded.** When the
  server keeps a model because a request is still using it, the command now says so and exits
  with an error instead of printing it as unloaded.
- **Engine settings apply when the same model is loaded again.** Settings marked "next load" (context
  size, GPU layers, MoE expert layers, memory-mapping, speculative drafting, parallel requests) took
  effect only for a different model or after a server restart: unloading a model and loading it again
  kept the old values. A reload now uses the current settings; values given on the command line still
  win.
- **Coder edits keep each file's line endings.** An edit, patch or search-and-replace by the coder
  rewrote the line ending of every line in the file: on Windows a file with LF endings came back with
  CRLF, turning a one-line change into a whole-file diff. Untouched lines now keep their bytes, and a
  changed line takes the ending the file mostly uses.
- **FLUX images from ComfyUI no longer end with a false weights warning.** Every FLUX job on the stock
  ComfyUI workflow reported that some CLIP/text-encoder weights were not found and the result may not
  have fully applied, because ComfyUI logs a CLIP encoder's text projection as missing although it
  fills it in itself. That line no longer counts as a warning; other missing weights still do.
- **Pulled image files are typed by what they contain.** `localm pull` of a single image file took
  its type from the repository's tags, so a LoRA such as SDXL-Lightning's registered as a diffusion
  model; it now reads the file the way `localm add` does. A FLUX GGUF and its CLIP, T5 and VAE
  files without an architecture key are recognised by their tensors instead of being registered as
  chat models.
- **Compacting a long chat writes a real summary on a small context window.** The request
  that summarises the older turns could be larger than the model's context window, so the chat
  kept only excerpts of them. Each summary request now fits the window, and a long history is
  summarised in parts, oldest first, so early details survive.
- **A request body that is not JSON gets a clear error, not a server error.** Sending a file or
  other binary data where an API route expects JSON now returns the usual 422 explaining the
  problem, instead of a 500 "Internal server error".
- **A Knowledge collection with a damaged chunk file opens.** A byte that is not valid UTF-8
  in `chunks.jsonl` costs the chunk it sits in, flags the collection as corrupt and offers
  repair, instead of failing every request that opens it.
- **A refused chat message no longer deletes attachments that were not the problem.** Images are now kept when a later message is refused for another reason (for example a rejected grammar), and an audio clip is kept unless the refusal was about it. The server now says which attachment a refusal was about; the chat strips clips or images only for a refusal of that kind.
- **A photo localm cannot read (an SVG, an iPhone HEIC, a cut-off upload) is refused with its own message.** The reply says the attached image could not be read. Remove the message that holds the unreadable image to keep chatting.
- **PDF attachments work on a default install.** `pypdf` is now installed with localm, so a PDF attached in Chat or the Coder is read even when the Knowledge plugin was never enabled.
- **GGUF sizing measures free VRAM again on Windows with two or more GPUs.** With torch
  installed, a box whose llama.cpp runtime sees two GPUs (two cards, or a card beside integrated
  graphics) skipped the VRAM fit check and sized context and layers blind. A load that runs on
  one GPU (a one-GPU split, or one card beside integrated graphics) now reads that GPU's free
  VRAM.
- **Free VRAM on Windows AMD counts other programs while a GPU check runs.** A model load that
  started while the GUI was refreshing its GPU readings used a figure that missed VRAM held by
  other programs, and the GUI could show such a figure as exact.
- **Unloading a model no longer pauses other requests.** The free-VRAM reading taken before an
  unload (a chat, embedding, reranker or speech model, or Unload all) ran on the server's main
  thread and could hold every other request and stream for up to 15 seconds; it now runs in the
  background pool.
- **Closing the agent browser no longer stalls the server.** Shutting a browser down can take
  several seconds, and while it did, every other request and stream waited. The close now runs
  in the background pool like the browser's other actions.
- **Changing the embedding model no longer freezes the server while it checks your collections.** The preview of which collections the switch would affect now reads them off the main thread, so chat and the page keep responding while a large collection is read.
- **The chat model comes back after a failed image, video or music job.** When a generation
  crashed after the chat model had been unloaded to make room, the chat model stayed unloaded
  until your next message. It now reloads on every way a job can end.
- **setup-llama fetches the build for your CPU architecture.** An Intel Mac now gets the macOS
  x64 build instead of the Apple Silicon one, ARM64 and s390x Linux get their own builds
  instead of x64 ones, ARM64 Windows gets its ARM64 builds, and an x64 Windows PC with a
  Blackwell-class NVIDIA GPU gets the x64 CUDA build and runtime instead of the ARM64 ones. A
  backend with no build for your machine is refused before anything is changed, naming the
  backends that have one. The installers no longer recommend it, and `setup.sh` and the
  windowed installer no longer list it.
- **setup.sh on macOS and with `--yes`.** The macOS backend menu lists only `cpu`, `metal` on
  Apple Silicon and your own build, and macOS no longer gets a Linux application-menu entry
  or a launcher warning. `--yes` no longer stops at the plugin picker or the CUDA download
  question, and the hints for provisioning later name the backends your machine has,
  including `metal`.
- **setup.bat lists only the backends your machine has.** The backend menu and the
  provision-later hint no longer offer `amd-rocm` or `sycl` on Windows ARM64, where they have
  no build; the menu numbers stay the same and a number that is not listed returns to the
  recommended backend with a message.
- **A re-sync removes documents indexed from credential folders.** A collection that already
  held files from a folder such as `~/.ssh` kept serving them; `localm rag resync` and the
  scheduled re-sync now drop those entries from the index.
- **`localm rag add` no longer indexes credential folders.** Adding a folder such as `~/.ssh` or
  `~/.aws` from the terminal, or a folder that contains one, used to index its files even though
  the API refuses them. The command now refuses a credential folder, UNC or device path, and (when
  network drives are turned off in Settings) a mapped network drive, skips any file under a
  credential folder while walking a folder, and a re-sync or repair skips a refused source
  instead of failing.
- **A config lock that cannot be read on release is reported.** The lock file left behind is named in a warning instead of lingering silently until it is reclaimed.
- **The MCP server frees VRAM for the embedding model.** When memory recall or a coder task first loads the embedding model next to a resident chat model on a tight card, `localm mcp` now unloads the idle chat model first instead of loading the embedder alongside it. A chat model that is busy or pinned stays loaded, and that, or a process with no chat model to unload, is logged as a warning.
- **A malformed key store or credentials file is no longer overwritten.** Creating or revoking an API key, or saving a Hugging Face or CivitAI credential, now refuses and leaves the file untouched when `auth.json` is not a list or `model_source_credentials.json` is not an object, instead of replacing it with just the new entry.
- **Self-calls to a look-alike loopback URL keep certificate verification.** An HTTPS URL containing a backslash or control character is no longer treated as loopback, so the coder agent does not disable verification for it when the local CA file is missing.
- **Two image, video or music jobs no longer load their models on top of the chat model coming back.** A second job that started while the first job's chat model was reloading loaded its own model beside it, which made the second job several times slower on a full card. A job now waits for a running chat reload, and the chat model reloads once, after the last job.
- **A memory consolidation whose model call fails says so.** When the model raised during
  memory consolidation (an engine error, an eviction, a decode failure), the run reported
  "no new durable facts found" and the scheduled memory job was marked done. Nothing is
  written now, the job is recorded as an error, the Memory panel's synthesize action shows
  the failure, and a session whose summary failed is retried on the next run.
- **A fractional number for a whole-number setting is refused over the API too.**
  `PATCH /v1/config {"max_tokens": 1.5}` used to store 1 while `localm config max_tokens 1.5`
  was refused; both now refuse it with "expected an integer". A whole number written as
  `2048.0` is still accepted.
- **Applying an update no longer says "already up to date" when it cannot compare versions.**
  If the latest release's version is not a recognisable version number, `POST /api/update/apply`
  now says it could not tell whether that release is newer, as the update check already did.
- **A chat answered by another localm instance keeps this instance's session persistence.** With a
  route to another instance's loaded model, a scheduled chat job on a routed model, or
  `localm mcp --share-loaded-models`, that instance used to record the chat in its own mode, so a
  privacy-mode chat could land in another instance's session log. Unless the chat is recorded in
  full mode here, the other instance must now confirm it records it no more than this mode allows.
  If it does not, nothing is sent and the message says which instance and why.
- **Switching models no longer freezes the server while the model folder is measured.** Reading
  the model registry, measuring a model's size on disk and building its engine ran on the
  server's main loop, so a load of a large model folder on a slow or network disk stalled every
  other request, including the status poll, until the disk answered.
- **A scheduled coder job honours its project's privacy setting.** A `mode` set in the project's
  `.localcoder/config.toml` now applies to scheduled runs as it does to interactive coder sessions, so
  a project pinned to privacy mode no longer gets a transcript or audit log from a job. A project
  folder that cannot be checked for that file now means privacy mode too.
- **`localm coder --url` warns in privacy mode when the model server is on another machine**, as
  `--online` and `--anthropic` already did. The warning names the host your prompts and files go to,
  and says when the address came from `LOCALCODER_URL`. These privacy warnings now go to stderr, so
  `--output-format json` output stays clean.
- **On Windows, a file can no longer be renamed to a device name.** Renaming a generated
  image, track or clip, or a file in the folder browser, to a name such as `nul`, `con.png` or
  `com1.txt` is refused there, because a write to such a name goes to the device and its data
  is lost. On Linux and macOS these stay ordinary file names.
- **The coder agent's image tool uses the image plugin's own ComfyUI address.** With a
  ComfyUI URL set for the Images plugin, the coder's image tool sent its jobs to the shared
  default address instead, a different instance from the one the Images page, the chat REPL
  and the MCP server use.
- **Two localm processes signing in with the owner key at once keep one owner identity.** Each
  could record its own, so jobs and sessions stamped by one process stopped counting as the
  owner's after a restart. Signing in and the GUI's sign-in check also no longer hold up other
  requests while the key is checked.
- **Gemma 4 GGUF models are prompted in their own chat format.** Every chat with a Gemma 4
  GGUF used to be sent as a generic ChatML prompt, with a warning that the template was not
  recognised, because llama.cpp's built-in template matcher has no Gemma 4 entry. localm now
  writes the Gemma 4 turn format itself, as the model's own template does with thinking off,
  including the empty thought block the 12B, 26B and 31B templates add. Gemma 4's turn,
  channel, tool and media markers in fetched or tool-returned text are now escaped like other
  models' control tokens.
- **The coder's diff edits leave every line they were not asked to touch exactly as it was.**
  A form feed or Unicode line separator elsewhere in the file no longer turns into a newline, a
  file with mixed line endings keeps each line's own ending, and trailing blank lines survive.
- **Phone share and file upload reject malformed or oversized multipart bodies up front.** Both
  read the form with the same bounded reader as the audio routes, so a body that is not valid
  multipart answers 400 or 415 and a share over its size limit is refused before it is parsed.
- **Signing out every device and disabling a job stick while the server is running.** `localm key
  recover`, `localm key clear` and `localm job disable` could be undone by the running server saving
  a browser session or a job result at the same moment, so revoked sessions kept working or a disabled
  job ran again. Writes to the session and job stores now wait for each other across processes.
- **Disabling the coder plugin ends its open sessions.** A coder session kept running after the plugin
  was disabled, with its stop button gone. Disabling or uninstalling the plugin now stops and closes
  every live coder session, and so does refreshing the plugin to a new version.
- **A background task no longer reloads a model that is being unloaded.** A scheduled job or a memory
  consolidation that started on a model just as it was unloaded or evicted could load it again where
  the Models page and idle unload could not see it. It now fails with a message to retry.
- **A model that fails to load says why, without blaming a healthy runtime.** A load error now keeps
  its error type, so an empty message no longer produces a blank reason, and an unexpected error inside
  localm is reported as such instead of with advice to repair the llama.cpp runtime.
- **A transformers model whose grammar constraint fails mid-reply is no longer reported as constrained.**
  The failure is logged and shown, and later grammar requests to that model are refused, as for GGUF
  models.
- **Grammar-constrained replies from transformers models on a GPU now follow the grammar on Windows.**
  The constraint used to be dropped on the first token on a machine without a C compiler, such as a
  typical Windows install with an AMD ROCm GPU. It is now applied without compiling anything.
- **Ignored GPU and runtime settings are now named.** A device in `gpu_split_indices` whose free memory
  cannot be read, and a `binary_dir` setting that is not a usable path, are logged instead of silently
  dropped. A model worker that does not exit when stopped is killed.
- **Two command-line hints no longer mislead.** `localm memory list` on an empty store says that
  `localm memory add` does not work in privacy mode, and `localm mcp --help` now names the
  `server_activity` tool and each plugin tool.
- **A broken coder config no longer hides your MCP servers or skills silently.** When
  `.localcoder/config.toml` cannot be parsed, or a skill folder or `SKILL.md` cannot be read,
  the coder now prints a warning at start-up naming the file instead of starting with no MCP
  tools or a shorter skill list and saying nothing.
- **An unreadable saved conversation is reported instead of vanishing from the sidebar.** A chat
  file that is corrupt or locked used to be left out of the list with no message. The list now
  names how many were skipped, the Chat page shows a notice, and the log records which one.
- **Model control markers hidden inside other markers no longer reach the reply.** A model
  that wrote a header or turn marker with another control marker inside it (for example a
  Llama 3 role header split by `<|message|>`, or `<start_of_turn>` followed by a marker and
  then `model`) left half of the header or a bare `model` / `assistant` at the start of the
  reply. The whole group is now removed, however the reply is streamed.
- **Plugin problems that used to vanish are now reported.** A first-run copy of the chat plugin
  that fails, an upgrade refresh that fails, a plugin whose teardown raises, and a damaged bundled
  plugin manifest are now logged and shown in the plugin list's errors instead of disappearing. A
  refresh that could not replace the old copy is reported as failed (not "up to date"), and
  `localm plugin refresh` exits with an error for it. If chat was left installed but disabled by a
  failed first-run setup, it is enabled again.
- **Uninstalling a plugin with "delete its data" reports when the plugin's own cleanup failed.**
  The uninstall still completes, but it no longer claims the data is gone.
- **`localm ps` no longer lists a stopped or restarted server.** Stopping a server, or
  restarting it from Settings, used to leave its entry behind, so after a restart `localm ps`
  showed the same process twice and `localm stop <old id>` stopped the running server. The
  entry is now removed on stop and restart, and `localm stop` on an entry a restart left behind
  only removes the entry.
- **Changing Session persistence in Settings takes effect at once.** The server kept the mode it
  started with until a restart, so switching to full or log saved the setting but recorded
  nothing and the chat page still said privacy mode, and switching to privacy kept writing the
  chat transcript. The next exchange now follows the saved mode, and the chat page updates its
  hint and starts or stops saving conversations straight away. Switching to privacy also clears
  the conversation list on the chat page, as a restart into privacy mode does; conversations
  already saved come back when you leave privacy mode.
- **The Chat and Coder persistence overrides now apply to what the server records.** Web chat and
  coder requests went to the server's audit log and transcript under Session persistence alone,
  so a Chat or Coder override of privacy still recorded that chat, or the files the coder sent to
  the model, and the chat page followed Session persistence instead of the Chat override. The
  server now records web chat and coder requests no less privately than their override (and a
  project's own coder mode), never more than Session persistence allows, and the chat page saves
  or keeps conversations by the Chat override.
- **Rotating the owner key with a malformed request no longer replaces it with a random key.** `POST
  /api/auth/key/rotate` with a body that is not empty and not a JSON object (for example a form-encoded
  `key=...`) now answers 400 and leaves the key unchanged, instead of generating a new key the caller
  never learns.
- **Creating a key says when plugin readiness could not be checked.** If checking whether the plugins
  behind the granted scopes are installed fails, the reply now carries a warning saying so instead of
  reading as "all ready".
- **The coder's browser_close stops a browser that is still starting.** Closing the browser
  while it was still starting used to report that no browser was open and leave it to finish
  starting; it now cancels the start, the browser is closed as soon as it is up, and the calls
  waiting for it get an error.
- **A worker process starts even when the debug log it inherits cannot be opened.** In debug
  mode, a model, embedding, speech or image worker whose debug log folder was missing or not
  writable exited before answering, and the load was reported as a crash with no reason; it
  now logs one warning to the console and carries on.
- **The coder no longer crashes on a project config whose `mcp` key is not a table.** A
  `.localcoder/config.toml` with, for example, `mcp = "x"` stopped MCP setup with an error; it now
  means no MCP servers are configured.
- **The FLUX VAE offered for a missing ComfyUI workflow file downloads again.** `ae.safetensors`
  came from a Hugging Face repository that now requires a licence click-through and answers
  401 to an anonymous download. It now comes from an open repository that holds the same file.
- **OpenGL shader caches stay in the data folder.** Mesa and the NVIDIA Linux driver wrote
  compiled shaders to `~/.cache/mesa_shader_cache` and `~/.cache/nvidia`, outside the data
  folder, so "delete saved data" never removed them. They now go to `cache/mesa` and
  `cache/glcache` inside it. A location you set yourself with `MESA_SHADER_CACHE_DIR` or
  `__GL_SHADER_DISK_CACHE_PATH` is used as you set it. Files already in the old location are
  left where they are.
- **SmolVLM2 image chats no longer log a processor warning.** Each image turn logged
  "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict" twice.
- **A half-copied GGUF file is no longer registered by a folder import.** `localm add <folder>` now
  skips a GGUF whose header declares more tensor data than the file holds, and one named
  `incomplete-...` (how GPT4All names a download in progress), and says why. A file named
  `incomplete-...` still registers when it is added by its own path.
- **A Diffusers pipeline or a CTranslate2 folder is named instead of rejected as "not a
  model".** Adding one, or finding one in the Hugging Face cache, now says what it is and
  what to use instead.
- **MTP needs several GB less VRAM.** The MTP draft context now works in 64-token batches. Its
  compute buffer and the buffers of its on-GPU draft sampler grow with the batch size and the
  vocabulary, and at the old 2048-token batch took about 6 GB on Qwen3.5-4B once a reply had
  started; they now take under 0.5 GB. The VRAM estimate charges them from the model's
  vocabulary instead of a flat amount that undercounted them. Drafting and its output are
  unchanged.
- **`localm comfy setup` copies an AMD ROCm or StabilityMatrix ComfyUI.** Replicating a ComfyUI whose PyTorch came from
  AMD's per-GPU ROCm packages (`amd-torch-device-gfx1030` and similar, as StabilityMatrix installs on
  Windows) failed with "pip found no matching version"; those packages now install from AMD's own index.
  The copy also finds the models your ComfyUI finds through its own `extra_model_paths.yaml` (such as
  StabilityMatrix's shared models folder), not only those under its `models` folder.
- **Document attachments without the Knowledge plugin.** Attaching a text document, PDF or code file in
  Chat or the Coder no longer fails with "Method Not Allowed" when the Knowledge plugin is not
  installed: attachments are now converted to text by the built-in chat plugin
  (`POST /api/chat/extract`), still entirely in memory. An API key needs the `chat` scope for
  this, where the old route needed `rag`.
- **A model split between GPU and system RAM says when its RAM part does not fit.** A load that keeps
  layers on the CPU (a partial offload, a machine with no GPU, or a draft model placed on the CPU) whose
  weights in system RAM do not fit available RAM now says in the load output, the GUI load toast and the
  load response (`mmap_cpu_from_disk`) that its CPU layers read from disk-backed memory and every token is
  slower. With `use_mmap off`, such a load warns before it starts that it may fail or swap.
- **One CPU backend, the best match for the CPU, on upstream llama.cpp builds.** With a Vulkan, CUDA or
  CPU build from llama.cpp's releases, localm registered every CPU variant the processor supports and
  ran CPU work on the first one by name; it now lets the build pick the single variant it scores best for
  the processor.
- **gpt-oss replies reach their answer.** A gpt-oss reply stopped at the end of its
  reasoning, so the answer came back empty; it now continues into the answer.
- **gpt-oss tool calls come back as tool calls.** A gpt-oss GGUF model writes a function call
  as a harmony message addressed to `functions.NAME`; that call used to end up in the
  reasoning text with no `tool_calls` in the reply. It is now returned as a `tool_calls` entry
  on the OpenAI, Responses, Anthropic and Ollama routes, streamed or not, and the coder agent
  runs it, while the analysis before it stays reasoning.
- **Small message fixes.** A network bind without a key points at `localm key generate` instead of a
  PowerShell-only command. The coder's "Unattended one-shot" warning goes to stderr, so the JSON on
  stdout stays a lone document.
- **A Hugging Face model whose chat templates have no `default` is refused with a clear `400`.** Before, it
  failed later with a generic worker error.
- **A llama.cpp runtime without lazy grammar support says to run `localm setup-llama`.** A lazy
  grammar request (`grammar_lazy`, which tool calling with `tool_choice` `auto` now uses) used to
  tell a GGUF user to use a GGUF-format model.
- **A TLS certificate or key path too long for the operating system no longer fails the save.** The TLS certificate and key settings report it as a missing file.
- **The Seed help no longer promises an identical reply.** A GGUF model computes only the
  part of a prompt its KV cache does not already hold from earlier requests, and that can
  change the reply, even at temperature 0. The coder's Seed tooltip, `localcoder --seed` and
  the docs said the same seed, model, prompt and settings always give the same output; they
  now say a reply can also depend on what the model answered before, or on n-gram
  speculative drafting when it is on.
- **ACE-Step music model files are no longer listed as chat models.** A pulled ACE-Step
  diffusion model, VAE or text encoder is registered as that kind of model and the planner
  as a non-chat model, and choosing any of them for chat says it is a music generation
  model component.
- **Malformed files and requests are refused with a clear error instead of crashing.** The MCP server no longer exits when a client sends a `tools/call` with the wrong parameter shape, a 4300-digit
  number or deeply nested JSON; a grammar with an enormous repeat count is rejected as an invalid grammar; a plugin
  with a non-UTF-8, over-nested or wrong-typed `plugin.toml` is reported as broken without hiding the other plugins;
  an encrypted or unsupported-compression `.docx` is skipped with a message; and a model folder whose `config.json`,
  `tokenizer.json` or shard index is over-nested, has an enormous number or has the wrong shape is refused or treated
  as having no metadata, and an uploaded workflow, `model_meta.json` or install record that is over-nested is
  rejected or ignored. An over-nested `config.json` no longer stops localm from starting.
- **Over-nested or enormous-number JSON no longer crashes the feature that reads it.** Memory, coder episode, scheduled job, knowledge collection, conversation, prompt library, gallery index and settings files, the reply of a peer or another localm instance, the coder agent's tool calls, reviews and MCP server output, and a model's tool-call arguments are now handled like any other unreadable JSON: skipped or reported, while the rest of the data still loads.
- **`localm doctor` recognises the macOS runtime.** On a Mac it reported the Metal build as "no llama library", skipped the native ABI check and the GPU probe, and ended with "CPU mode only"; it now checks the library like on other systems and names the Metal GPU.
- **A knowledge collection whose `meta.json` is not valid UTF-8 no longer breaks the collection list.** The list, the collection detail view and a model rename now treat that collection as unreadable, flag it as corrupt and carry on with the others, instead of failing for every collection.
- **Bug reports no longer carry email addresses.** Any email address in a report
  (your description, an error, the log, a hang trace, the browser console or the
  settings it lists) is
  replaced with `<redacted-email>` before the report is shown, saved or sent, in the
  app and in the standalone `report-issue` scripts. The maintainer's contact address
  is kept.
- **Bug reports no longer carry machine names.** The server in a network path
  (`\\FILESERVER01\share`), a local-network host name such as `nas.local` or `pc.lan`,
  and this computer's own name are replaced with `<host>` in the app and in the
  standalone `report-issue` scripts.
- **Models you already have on disk are found where LM Studio and llama.cpp keep them.**
  `localm add <folder>` and the models-folder scan now look inside subfolders (as far
  as the Folder import depth setting allows, three levels by default, so LM Studio's
  `publisher/repo/file.gguf` is covered) and register HuggingFace safetensors model
  folders as well as GGUF files; before, a safetensors model was only found when you
  pointed at its own folder, and the automatic scan only saw files sitting directly in
  the models folder. A folder import also reads each GGUF's own header: a vision
  projector is no longer listed as a chat model, an embedding model is listed as one,
  and the model's architecture is recorded. A `.GGUF` file with an upper-case
  extension is recognised.
- **GGUF files that are not chat models are no longer added as chat models.** Draft
  heads for speculative decoding (EAGLE3, DFlash, Gemma 4 assistant) and the WavTokenizer codec are
  added as type `unknown`, text-to-speech models (Qwen3-TTS) as `tts`, and image
  and video GGUFs (Flux, Wan and the other ComfyUI-GGUF checkpoints) as
  `diffusion-unet`. Trying to load one says what it is, instead of failing in the
  runtime; models you added earlier keep the type they were registered with.
  `llama-embed` models and any encoder that declares non-causal attention are
  recognised as embedding models, and openPangu-Embedded, which is a chat model, is
  no longer mistaken for an embedding model. A folder downloaded from a GGUF
  repository (a `config.json` and tokenizer beside the `.gguf` files) registers its
  GGUF files instead of one model that cannot load, and the component folders of a
  diffusers pipeline are no longer listed as models of their own.
- **A model newer than the bundled llama.cpp now says so.** When the runtime does not
  know a model's architecture, the error names the architecture and says the model is
  newer than the runtime (with `localm setup-llama --tag latest` to try a newer build),
  instead of telling you to repair the runtime.
- **A model whose weights in system RAM do not fit free RAM now loads from disk-backed
  memory instead of failing.** With every layer on the GPU, localm read the weights it
  keeps in system RAM (the routed experts of a Mixture-of-Experts model that it moves
  off the GPU, and the token embeddings) fully into memory, and on a computer without a
  GPU it did the same with the whole model, so a model larger than free RAM failed to
  load or pushed the system into swap. localm now checks free RAM first (inside a
  container, the container's memory limit): when those weights fit, nothing changes;
  when they do not, or there is no GPU, the model file is memory-mapped wherever the
  device supports it, so only the parts in use stay in RAM and the rest is read from
  disk as needed.
- **FP8 Hugging Face models (for example `Qwen/Qwen3-0.6B-FP8`) now load and answer
  on CPUs and GPUs without native FP8 support.** They used to load and then stop at
  the first reply with `No module named 'triton'`. Their weights are now expanded to
  bf16 at load time (about 2 bytes per parameter), and the load output says so. A
  model whose expanded size cannot fit in available memory is refused with the sizes
  instead. FP8 runs natively only on an NVIDIA GPU with compute capability 8.9 or newer,
  with triton installed and the FP8 kernel downloaded from the Hugging Face Hub (or
  already downloaded by an earlier load, which then also works offline).
- **Setup can pick up after being interrupted, on every platform.** `setup.sh`,
  `setup.bat` and the graphical installer keep a short journal of the steps they have
  started and finished. Run again after Ctrl+C, a closed window or a crash, setup says
  where the last run stopped and redoes the step that was cut short (a half-made `.venv`
  is no longer mistaken for someone else's and kept). Uninstall also uses the journal, so
  a menu entry or command setup was creating when it stopped is still removed. On
  Windows, `setup.bat` also keeps its scratch files inside the folder instead of the
  system temp folder, and uninstall clears them. Backing out of the "Replace this .venv?" question
  is not treated as a half-built environment, and a second interruption still redoes every step the
  first one cut short.
- **The one-click Linux/macOS install no longer puts uv in your home folder before it asks.**
  `install.sh` used to install uv itself into `~/.local/bin` and edit your shell
  startup files when uv was missing, even if you then chose the Portable setup.
  It now only fetches localm and hands over to `setup.sh`, which installs uv inside
  the folder for Portable and in its usual per-user place only if you pick Shared.
- **Portable mode keeps its temp files in its own root folder, and running instances
  find each other directly.** For a self-contained install, temporary files go to
  `tmp` in the data folder. Settings > General > Temporary files (or the
  `LOCALM_TMPDIR` variable) chooses the data folder, the system temp folder, or any
  folder you name. Instances that share a GPU no longer use a registry file: each
  running instance is found from the operating system's list of listening ports,
  asked directly what model it holds, and asked directly to release it, which it
  does only after the asking instance confirms the request. Instances serving a
  network address are not detected. `LOCALM_PEER_DETECTION=off` turns detection
  off. Separately, when the separate GPU probe cannot answer on Windows, localm no
  longer loads PyTorch inside the server as a fallback, which could stall it for
  tens of seconds; it reads the GPU from `nvidia-smi` and the Windows
  display-adapter registry instead.
- **The installer waits for Enter after you answer "Replace it?".** Typing Y on the
  existing-environment question in `setup.bat` used to continue at once, unlike every
  other question; its other yes/no questions now wait for Enter as well.
- **RAM and VRAM now show right away at startup, and the server no longer freezes
  while a model loads on Windows with an AMD GPU.** Loading a model made localm
  load the ROCm version of PyTorch inside the server itself, which on Windows blocks
  the server from starting any work for 10 to 30 seconds: the status bar stayed empty
  for about a minute and the app reported a frozen server. localm now reads what it
  needs from the installed files and from the separate GPU probe instead, so memory
  figures appear in the first second and a stalled probe can no longer hold up the
  status bar.
- **The launcher can check the models folder again, and a clipped status message can
  be read.** Opening the launcher showed "Could not check the models folder:
  AttributeError" because the launcher runs without a console window and one startup
  step assumed a console. A status line that is cut short is now clickable: it shows the
  full text and copies it, and errors are also written to `logs/launcher.log`.
- **Export in the chat now saves the conversation in the app window.** The button did
  nothing there because the window refused every file download; it now opens a save
  dialog, as do the other save buttons (images, coder results).
- **A crash report now says how the server process ended.** A server that died without a
  fault trace was reported as "most likely an OS kill". The report now includes the
  process exit code (for example an access violation), keeps the end of a long fault
  trace instead of cutting it, and in debug mode keeps the whole trace and logs the
  crash watchdog's own activity to `logs/crash-watchdog.log`.
- **The VRAM estimate now counts the fixed-size state that hybrid models keep.**
  Models that mix linear-attention and full-attention layers (the Qwen3.5 family)
  hold a recurrent state in VRAM beside the context, and with MTP on the context holds
  three or four copies of it. The estimate behind the context ceiling, the GPU-layer
  choice and the low-VRAM warning left it out, so a load that looked like it fit could
  spill layers to the CPU or fail to allocate on a small card.
- **SmolVLM2 models now see images.** Their image processor needs the `num2words`
  package, which the HuggingFace install did not include, so the model loaded as
  text-only and refused every image. It is now installed with the HuggingFace
  stack, and when an image processor does fail to load, the refusal says why
  instead of claiming the model cannot see images.
- **Replies start appearing as soon as the model writes them.** Every reply was
  held back until about 48 characters had been generated, and a reply shorter
  than that appeared only once it was complete. On a HuggingFace-format model
  that added two to three seconds before the first words showed. Text is now
  held back only while it could still turn out to be one of a model's internal
  markers, so the answer after a reasoning block streams at once too.
  A Llama 3 role header that a model writes out as text no longer shows up in
  a longer reply.
- **A model loaded while another one sits idle in VRAM now goes fully on the GPU.**
  localm decided whether a second model fit beside the first from its file size,
  while the loader also needs room for the context's KV cache and its working
  buffers. For small models, models without grouped-query attention (Llama 2
  class) or a large context, the second model was loaded next to the idle one with
  part of it on the CPU, and stayed that slow. localm now checks what the loader
  will actually need and frees the idle model first, in the server and in the MCP
  server. When even an empty card is too small, a model you pick still asks before
  loading partly on the CPU.
- **On AMD (the `amd-rocm` build), work on the CPU runs about twice as fast.**
  That build's CPU backend used no SIMD instructions, so Mixture-of-Experts
  weights in system RAM, layers left on the CPU and CPU-only loads ran far below
  the hardware's speed. Setup now installs the AVX2/AVX-512 CPU backend from the
  same llama.cpp version over it; existing installs are offered it as a runtime
  update. With the Mixture-of-Experts change above, Qwen3-30B-A3B on a 16 GB card
  went from 7 to 26 tokens/s. If that CPU backend cannot be installed, the install is still reported
  as current until a newer CPU backend is released; `setup-llama --backend amd-rocm --force` retries
  it.
- **The "context too large for VRAM" message suggests a context that fits.** It
  suggested `-c 32768` whatever the current context, including when it already
  was 32768; it now names the largest context that fits, leaves the suggestion
  out when no context would help, and for a Mixture-of-Experts model names the
  `n_cpu_moe` value that fits.
- **A chat with web access on now ends on an answer, not on a bare "read page" or
  "web search" line.** Once the lookups stop, the last reply is requested without
  the web tools; if the model still writes a lookup, it is not run and the reply
  says so. A reply that writes lookup after lookup is cut after the first, which
  runs, instead of going on until you press Stop. Earlier lookups are sent back
  to the model in the tool's own format instead of as the chat's "web search: ..."
  lines, which some models copied as text, so a conversation reads the same to the
  model in every interface language; a copied line is treated as a broken lookup
  and the model is asked once to make it properly. After a failed lookup the model
  is told it has no information from it and must not describe or guess what it
  would have found, in the chat and in scheduled chat jobs; it used to be told to
  "answer without the web", which led some models to present invented findings.
- **Chat with a Knowledge collection selected no longer injects excerpts that have
  nothing to do with the question.** The four best-ranked chunks used to be added
  to every question, however weakly they matched, so a question about the
  conversation itself pulled in unrelated passages the model then answered from.
  Only excerpts that clear a relevance floor are added now, and when none do, the
  chat says that no excerpts were relevant. `localm rag query --relevant-only`
  shows the same filtered result. Where the check can only compare words (no embedding model, or a
  passage without a vector), an excerpt needs at least two of the message's words, unless the
  message is a single word.
- **Knowledge excerpts are no longer cut off mid-word.** Each excerpt added to the
  chat is the whole retrieved passage, and the question echoed above them ends on
  a whole word with "…" when it is shortened.
- **Web search and page reads are more reliable, and failures read as plain
  sentences.** A search whose connection is reset or cut is sent again (up to
  three tries). When DuckDuckGo still fails or answers with a bot check, localm
  tries DuckDuckGo's lite page and then Brave Search by itself, asks a
  bot-checked service once more after a short pause, and reports a failure only
  when every service failed, naming each cause. A results page localm cannot
  read is reported as such instead of as "no results". A configured SearXNG
  instance is the only service asked, and localm repairs it rather than
  replacing it: a stray `/search` in its URL is removed, its HTML results are
  read when its JSON format is off, and "no results" while its own search
  engines failed is reported as a failure. A failed search or page read
  now says what happened, for example
  "html.duckduckgo.com closed the connection before answering" or
  "stackoverflow.com refused access, HTTP 403", instead of a Python error.
  Page reads during a search give up after 15 seconds, and a page that keeps trickling in is cut
  off. A GitHub repository link reads the repository's README, a GitHub file link
  reads the file itself, and a Stack Overflow or other Stack Exchange question
  reads the question and its top answers through the Stack Exchange API, so
  these no longer come back as page navigation or a refusal. A site with
  several addresses is reached on the next one when the first cannot be
  connected to, for example when IPv6 does not work on your network.
- **A very large request no longer stays in memory through the activity log.** The 
  recent-activity log (shown in the control window and attached to bug reports) now 
  keeps at most 2000 characters of each entry, marked as truncated, so a multi-megabyte 
  query or prompt cannot pin its full text in memory or slow the window and bug-report 
  saving. A truncated entry that carries a traceback keeps its final exception type and message
  line.
- **Switching the embedding model back to the one a knowledge-base collection was
  built with no longer warns that it needs re-embedding.** The warning is shown
  only for collections the new model would actually invalidate, and the
  collections table drops a stale "re-embed needed" badge as soon as the switch
  or the model download finishes. Opening or deleting a large collection no
  longer stalls the rest of the server while it loads. A collection that was later added to under a
  different model of the same vector size stays listed until it is fully re-embedded.
- **Compacting a long conversation now shows that it is happening.** While older
  messages are being summarised to fit the context window, the chat shows
  "Compacting conversation…" instead of sitting silent, in the browser, in
  `localm run`, and in the coder. Streaming API clients receive a `compacting`
  status before the reply starts.
- **Chats say what localm is doing before the reply starts.** Loading or reloading
  the model, recalling memories, running chat plugins and waiting for another
  request now show as a live status before the reply starts, in the browser chat, the browser coder,
  `localm run`, the coder in the terminal (including one-shot `localm coder "task"` runs, which now
  print each status and tool call), and the MCP `chat` tool for clients that ask for
  progress. A streaming request that is still preparing after a moment opens
  its stream at once and reports each step; a request refused at that point
  ends the stream with the error, and the stream carries the routing and memory
  information otherwise sent as response headers.
- **Long background steps say what they are doing too.** Adding documents to a
  knowledge base names each file as it is read and embedded; the one-time
  embedding-model download reports how much has arrived (also from the Memory
  dialog, which previously showed only "Downloading…"), and a chat whose memory
  recall triggers that download shows "Downloading the embedding model…".
  Image, video and music generation say when they are waiting behind other
  ComfyUI jobs instead of claiming to render, and image generation now shows
  ComfyUI start-up and model-substitution notices that previously went only to
  the server console. MCP clients that ask for progress get the image and
  embedding-download progress too. A coder sub-agent's model statuses and tool
  calls appear in the parent session while it works.
- **`localm run` attached to a server, and the coder talking to a localm server,
  report a failed reply as an error.** A reply the server ended as failed (for
  example a conversation that still does not fit after compacting) used to be
  shown and kept as if the model had written it; it is now reported as an error,
  and `localm run` takes the unanswered message back out of the conversation.
- **The coder answers a question in plain text instead of forcing a tool call.** A
  message such as "why does web search sometimes fail?" used to count as a request
  to act because it contained a word like "search" or "check", and the coder then
  re-prompted the model and finally forced it to emit a tool call, so it ran a
  pointless command and drew a conclusion from it. Only an instruction ("search the
  repo for X", "please run the tests") is now treated as a request to act, and a
  question the model asks you back is accepted as its answer.
- **The `[TOOL_CALLS]` marker that some models write out as text no longer shows up
  in replies.** It is removed from chat and coder output, a reply that repeats it
  hundreds of times is cut short, and a tool call written as
  `[TOOL_CALLS][tool_call]{...}</tool_call>` or `[TOOL_CALLS][{...}]` is run like any
  other. The coder also no longer tells the model that a call it just ran "was NOT
  run" when the call was wrapped that way.
- **A model loaded by an automatic switch no longer stays slow until you reload it.**
  When the model it replaced was still busy answering something else, the new model
  was loaded beside it, partly on the CPU, and kept that slow placement for as long as
  it stayed loaded. An automatic switch now waits for the busy model to finish (up to
  30 seconds, showing "Waiting for another request to finish...") and frees it first;
  a model that still had to load partly on the CPU is reloaded at full speed on its
  next use once the model that held the memory is idle or gone. The reply's "instead
  of" chip now says when the answering model runs partly on the CPU.
- **Small talk no longer ends up in the remembered facts, and one common word no
  longer pulls old session summaries into every chat turn.** A session with
  nothing lasting in it (a greeting, a passing remark, a one-off request) now
  gets no session summary, and a stored session summary is recalled only when a
  message shares at least two content words with it (or is a close paraphrase),
  so unrelated turns carry no remembered-facts block.
- **Common words no longer drag old chat summaries or Coder lessons into
  unrelated turns, and small talk already stored is cleaned up.** A word that
  appears in many of your stored session summaries (or Coder lessons) no longer
  counts as a match, and the Coder's lesson recall needs two shared words, as the
  chat memory now does. Session summaries saved earlier are checked once by the
  model; those that are only small talk are removed, and a removed one can still
  be restored from the memory archive. Greetings and thank-yous no longer count
  as something worth summarising.
- **The microphone button works with every PyAV release again.** Speech-to-text
  failed on any install that had PyAV 19, because faster-whisper's own audio
  decoder passes an option that PyAV 19 removed, so every recording failed. localm
  now decodes the recording itself: the audio is identical to before on PyAV 18 and the transcripts
  are identical on PyAV 19. A failure inside the audio decoder itself is reported as a
  server error naming the library problem instead of "corrupt or unsupported
  audio", and `localm doctor` now checks that a recording can be decoded.
- **Attaching an image in the Coder now says plainly that the Coder cannot take
  images.** It used to ask you to load a vision model, which could not help.
  Attach the image in Chat with a vision model, or paste the text instead.
- **Multi-Token Prediction no longer switches itself off after one long prompt, and no longer stops silently partway through a reply.** The speculative-decoding helper used to be capped at 2048 tokens, so a single longer prompt turned it off for as long as the model stayed loaded, and a reply that crossed 2048 tokens stopped speculating while still being reported as active. It now grows with the conversation's context, keeps working on the next request, and a reply on which it has to stop reports that it did. A permanent stop that comes with a detail suffix, such as a failed helper prefill, is now recognised as a stop.
- **Long chats keep their thread when older messages are compacted.** With a
  reasoning ("thinking") model the summary of older messages was never written,
  so the earlier conversation was replaced by a one-line note and the model lost
  track of what you were talking about. The summary is now written with
  reasoning turned off, and if it still cannot be written the earlier messages
  are kept as short excerpts instead of being dropped. Your latest request is
  always kept word for word, and the conversation no longer shows two assistant
  replies in a row. The
  same applies to the coder, which also keeps your current request when it
  compacts a long session. Closing the request while a server-side compaction is
  running now stops the summary instead of letting it run to the end.
- **Asking a vision model about an image no longer crashes it on the Vulkan
  runtime when the optional GPU extras are installed.** The model process loaded
  a second copy of a runtime library during the request and exited with "The
  model process exited unexpectedly (worker exit 3)"; it now answers.
- **Two machines sharing one data folder no longer take over each other's
  knowledge-base write lock.** Two machines with the same host name (on Windows,
  or on Linux hosts) read each other's lock records as their own process table,
  found the holder's process absent and handed the lock over while the other
  machine was still writing. The lock now tells machines apart by the machine's
  own identifier, as the model download lock already did. A lock written by an
  earlier version reads as another machine's and is released by its heartbeat
  going quiet rather than by its process exiting.
- **A coder session's log file can be deleted after the session ends, even when
  saving the session's lessons failed.** A failure while storing a finished
  session's episode used to skip closing its audit log, which on Windows left the
  file locked until the server stopped.
- **A stopped embedding or reranker worker waits for a model load before it starts again.**
  When the GGUF embedding or reranker worker had stopped, the next Knowledge query or
  embeddings call started it again at once, even while a chat model was loading, so both
  could plan for the same free VRAM. It now waits for that load to finish first.
- **Chatting with a HuggingFace model that has no chat template now says so
  instead of crashing the model process.** Such a model (typically a base,
  non-chat checkpoint) used to load fine, then kill its worker on the first chat
  turn with a generic "exited unexpectedly" message and unload. Chat now returns
  a clear error (HTTP 400, or a message in the CLI) telling you to use an
  instruct or chat variant, and the model stays loaded. When a model process does
  stop on an unexpected error, the message now includes that error instead of
  "no native fault trace was captured". A vision model whose chat template is kept
  only with its tokenizer renders image turns with that template, and an image turn
  the template cannot lay out is refused with a message instead of stopping the
  model.
- **Pulling a model again over a file it already names now updates that entry's
  source and checksum.** After `localm pull civitai:... --redownload` replaced a
  file with different bytes, or a pull recreated a file that had been deleted by
  hand, the model's entry still named the old version and its checksum, which
  peer routing and duplicate detection trust. A CivitAI pull whose entry cannot
  be saved under the requested name now reports that instead of saying it
  finished.
- **Closing a coder session while its browser is still starting no longer
  leaves the browser running.** A browser that finishes starting after its coder
  session has closed is now stopped instead of staying open, with its cookies and
  storage, until localm exits.
- **`localm setup-llama --help` no longer says the newest upstream build is the
  default.** The default installs the llama.cpp build this localm release
  confirmed; the help now says so and describes `--tag <tag>`, `--tag latest`,
  `--tag default` and `--rollback` accurately.
- **The app window keeps its saved login and page data inside your data folder.**
  It used to keep them in a folder shared with other apps, outside your LocaLM
  folder (`%APPDATA%\pywebview` on Windows, `~/.pywebview` on Linux), so deleting
  the data folder, or choosing "delete saved data" when uninstalling, left a
  copy of your recent chats and your sign-in behind. They now live in
  `app-window` inside the data folder and go with it, and on Linux the window's
  web cache moves there too. If `app-window` cannot be used (a read-only data
  folder, or one on a network drive, which the Windows web view does not
  support), the window keeps nothing after it closes and a warning is logged.
  macOS cannot be pointed at a folder, so there the window still keeps its data
  in macOS's own storage outside the data folder. The shared folder from earlier
  installs is left alone and not migrated; the uninstaller still names it.
- **Image, music and video requests with a NaN or infinite number in a
  tuning field are now refused.** Guidance, cfg, denoise, the LoRA strengths,
  lyrics strength and shift each return a 422 instead of starting a
  generation job with garbage sampler values.
- **`localm pull civitai:<version>` no longer takes another version's file as
  already downloaded.** Versions of one CivitAI model often share a filename, and
  when CivitAI lists no SHA256 for the file, a file left by a different version was
  accepted and registered as the one you asked for. Such a file is now reused only
  when localm's own registry records it as that version and its size matches the
  size CivitAI lists, when it lists one; otherwise the pull stops, leaves the
  file as it is and says how to replace it. A SHA256 mismatch says which registered model owns the file, and `--redownload` no
  longer overwrites a file that a model registered from another source points at.
- **A download interrupted at the wrong moment no longer blocks every later pull
  of that file.** A crash while a pull was taking or giving up its download lock
  could leave an empty lock folder behind, and every later pull of that file then
  refused with "already being downloaded by process unknown" until the folder was
  deleted by hand. The lock is now never left without a record of its holder, an
  empty lock folder left by an earlier version is taken over by the next pull,
  and a pull that starts while another is finishing no longer refuses.
- **Two browser calls in one coder turn no longer open two browsers.** With no
  browser open yet, a turn that called `browser_navigate` twice at once started
  a browser for each call. The second replaced the first in the session's
  record, so the first kept running with nothing able to close it. The calls
  now share a single start and drive the same browser; if that start fails,
  every waiting call gets the same error.
- **Model downloads no longer leave Hugging Face files in your user profile.**
  Pulling a model, installing the embedding model and downloading the speech model
  wrote Hugging Face transfer logs, staging folders and cache stamps to
  `~/.cache/huggingface` (or `$XDG_CACHE_HOME/huggingface`), outside the data
  folder, so "delete saved data" never removed them. They now go to
  `cache/huggingface` inside the data folder. A Hugging Face login you saved with
  `huggingface-cli login` keeps working, and a location you set yourself with
  `HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE` or `HF_TOKEN_PATH` is used as you set it.
  Files already in the old location are left where they are. Commands the coder runs (`run_shell`,
  background shells, tests, git) still see your own caches and Hugging Face login.
- **Loading a model on an AMD GPU no longer leaves compiler caches in your user
  profile.** The AMD GPU runtime wrote its compiled-kernel cache to
  `%LOCALAPPDATA%\comgr` and, for convolution models, `~/.miopen`, outside the data
  folder, so "delete saved data" never removed them. They now go to `cache/comgr`
  and `cache/miopen` inside the data folder. A location you set yourself with
  `AMD_COMGR_CACHE_DIR`, `MIOPEN_USER_DB_PATH` or `MIOPEN_CUSTOM_CACHE_DIR` is used
  as you set it. Files already in the old location are left where they are.
- **A browser that starts too slowly is no longer left running.** When the
  automated browser did not come up within its start timeout, the Browser tab
  and the coder's browser tools reported "the browser did not start in time",
  but the launch carried on and, once it finished, left a Chromium and its
  driver running until the server was stopped. That browser is now closed as
  soon as the late launch finishes.
- **Stopping or restarting localm now closes open coder sessions properly.**
  Stopping or restarting from Settings or the tray, or stopping a `localm serve`
  the GUI was attached to, left open coder sessions unclosed. Each one now saves
  its conversation, records its end in its session log, writes its full-mode
  transcript, and closes the automated browser it opened. A session that cannot
  finish closing within a few seconds does not hold up the stop.
- **A change of the system clock no longer lets a second writer into a knowledge
  collection that is being indexed.** After the clock jumped (a laptop waking, a VM
  or WSL resyncing its clock), a `localm rag add`, a scheduled re-sync or a memory
  write could decide that a run still writing the same collection had crashed and
  write alongside it. A run that is still writing now keeps the collection whatever
  the clock does: the waiting command takes a collection over only after it has
  itself watched the other run stay silent. When localm can see that run has
  exited, a clock set back after it stopped no longer keeps its collection locked.
- **An app window that loads slowly now gets its copy and paste shortcuts.** On
  Windows, when the window's first page took longer than about 8 seconds to
  load, its keyboard shortcuts for copy, paste and select all, and its
  right-click menu, were never turned on and did not work in that window. They
  now turn on as soon as the page has loaded, however long that takes, and the
  window comes to the front then (unless you already closed it to the tray).
- **Quitting from an app window that never finished loading no longer opens a
  browser tab.** With "Quit when the app window is closed" on, closing a window
  whose page never loaded opened a browser tab onto the localm that was shutting
  down.
- **A slow-to-load app window no longer makes a stopped localm open a browser
  tab.** When the app window took longer than about 8 seconds to load its first
  page, stopping localm with a termination signal, or closing the window with
  "Quit when the app window is closed" on, opened a browser tab onto the server
  that had just stopped, and closing the window of a `localm gui` attached to an
  already-running instance opened a tab as well. A slow first load now counts as
  loaded, and no tab opens once the server has stopped.
- **A model whose load just failed is no longer loaded again on every request.**
  When a request needed something the selected model lacks (for example tool
  calls in the coder) and the model chosen to answer it crashed or failed while
  loading, localm retried that load on every later request, each time costing the load time and a
  crashed worker, and the request then failed. Such a request is now answered by the selected model
  instead, and routing leaves that model out until its load settings, its model file or the
  llama.cpp runtime change, it loads successfully, or a growing delay (10 minutes, doubling
  to 6 hours) passes. Loading it yourself still tries it. A reply answered by
  the selected model for that reason says so: the chat shows "a capable model was
  not used", the coder session feed and `localm coder` print why, the MCP tools
  add a note, and the `X-Localm-Model-Routing` header and the debug log name the
  skipped model, when its load failed and the reason.
  Requests that were already waiting for the model when its load failed no
  longer repeat the load, and the overlapping failures count as one, so the
  delay no longer grows with the number of requests. A model another localm
  instance already serves still answers through that instance after a failed
  local load, and replacing the runtime with `setup-llama --from` or `--url`
  lets a failed model be tried again.
- **The automated browser can be set up from the GUI, and it starts on Linux.**
  When the bundled browser had not been downloaded, opening it failed with
  a message telling you to run `playwright install`, which localm does not
  give you. The Browser tab and Settings > Server & network now show a
  Download browser button (a one-time download that respects the network
  policy) and errors name localm's own remedy instead. The system engine
  looks for Google Chrome, Chromium, Microsoft Edge and Brave in their
  usual places on Linux, macOS and Windows, not only Chrome at its default
  path; if none is found it lists what it looked for and offers the bundled
  browser. A browser that cannot start for missing system libraries now says
  so and names them. A download that did not finish, or a build missing its
  headless part, is offered for download again instead of reading as
  installed, in the GUI and in `localm setup-browser`. The setting's help
  no longer claims the system engine uses your logged-in sessions: both
  engines start with a fresh, empty profile.
- **A multi-GPU load no longer puts the output layer on a GPU too small for
  it.** Without a configured split, llama.cpp spreads layers over every GPU by
  free memory and puts the output layer, plus a logits buffer that can reach
  several GB for a large vocabulary, on the last GPU in its list, even when
  that card has little free memory. localm now checks what each GPU would
  hold and leaves out a GPU that cannot hold its share, or loads the model on
  one GPU when only that one can hold it, saying which one and why. A GPU
  that would receive nothing is no longer reported as short of memory. The
  check only changes the split when localm can match its GPU numbering to
  llama.cpp's; otherwise llama.cpp's own split is kept. On a computer with
  integrated graphics beside a discrete GPU, the integrated GPU's memory is
  no longer counted when sizing a model, since llama.cpp does not use it
  then. Setting `gpu_split_indices` to a single GPU now loads a GGUF chat
  model on that GPU only, instead of still spreading it over every GPU, when
  localm can match that GPU to llama.cpp's own device list, and an HF model
  now loads on that GPU too; in Settings, ticking one GPU under Split across
  GPUs now does exactly that, and leaving every box unticked keeps automatic
  placement. On such a computer a configured split of several GPUs and the
  main GPU now land on the GPUs you picked instead of being shifted by the
  integrated GPU llama.cpp leaves out, warnings about the main GPU name the
  GPU numbers Settings shows, and a split or single GPU that names the
  integrated GPU is refused with a message saying so, instead of loading
  somewhere else. A llama.cpp build that does use the integrated GPU keeps
  the split as configured. The VRAM shown in the GUI, the memory check before
  a load and the GPU this instance reports to other localm instances now
  describe the GPU a single-GPU load actually runs on, and the automatic
  context limit only counts the GPUs a load uses when GPUs are left out. With
  no split configured, the memory check before loading a GGUF model on
  several GPUs now counts the free memory of every GPU llama.cpp spreads it
  over, instead of GPU 0's alone, so it no longer unloads other models or
  asks for confirmation when the model fits across them, and another localm
  instance using any of those GPUs can be asked to free memory. A crash while
  creating the context is reported as that, with advice
  about the context size and the split, instead of telling you to repair the
  runtime. Bug reports list every GPU with its memory, write "not detected"
  instead of leaving the NVIDIA driver and CUDA version blank, include the
  multi-GPU settings, and keep llama.cpp's out-of-memory lines. The load
  error names this platform's library file instead of always `llama.dll`.
- **The desktop launcher's API server mode starts without a model selected.**
  Choosing "(no model - choose later)" used to stop with "Pick or import a
  model first" even though the server runs fine with nothing loaded. `localm
  serve` likewise no longer requires a model name: `localm serve --no-model`
  starts with nothing loaded, and `localm serve` alone picks a registered chat
  model the same way `localm gui` does.
- **The coder REPL no longer loses a saved session when a turn fails.** Each
  new message, and `/resume`, deleted the saved session before the turn ran,
  so a turn that failed with an error (for example, an unreachable model
  server) left nothing to resume. The last saved state now stays until a turn
  finishes or is interrupted. Ctrl-C also no longer reports progress as saved
  when it could not be written, and in privacy mode no longer says `/resume`
  is unavailable while a resumed session is still saved.
- **The desktop launcher no longer gets stuck on "Checking models folder".**
  When the models folder scan finished before the launcher window was fully
  up, which happens easily with an empty or small models folder, its result
  was lost: the status line kept counting and Launch stayed greyed out. The
  result now always reaches the window, and a scan that cannot run shows why
  instead of reporting the folder as up to date.
- **When `localm gui` cannot open the GUI on an already-running `localm serve`,
  it now says why.** It used to guess "an older instance?" for every failure.
  It now shows the server's own error, a refused attach token, a server that
  did not answer, or an older localm, and the server logs the full error.
- **localm no longer fails to start when an OpenTelemetry endpoint is set in
  your environment.** With FastAPI 0.142, which a fresh install now picks up, a
  set `OTEL_EXPORTER_OTLP_ENDPOINT` stopped the server at startup, because
  FastAPI tried to set up telemetry export on its own. localm now turns
  FastAPI's built-in telemetry off, so nothing is exported whatever `OTEL_*`
  variables are set.
- **LocaLM starts again when its data-folder setting was saved by an older
  `setup.bat`.** A custom data folder path with characters such as é was saved
  in the console's code page instead of UTF-8, and LocaLM stopped at startup
  trying to read it. It now reads the setting as it was written; a setting it
  cannot read at all gives a warning instead of stopping LocaLM.
- **`setup.sh` and `localm make-launcher` no longer delete the `localm` command
  when the LocaLM folder is on a drive that ignores letter case** (a Windows
  drive under WSL, or an NTFS or exFAT disk on Linux). There `LocaLM` and
  `localm` are the same file, and building the app launcher removed it, so
  `localm` was missing after setup. The launcher step now leaves it alone and
  the menu entry starts LocaLM through the environment's Python.
- **Restarting a localm server that had the GUI attached to it no longer leaves
  the window on a "not found" page.** When a second `localm gui` attached to a
  running `localm serve` (or `localm gui --api-mode`), it mounted the GUI on that
  server live, but any restart - the tray Restart, Settings > Restart, applying
  or rolling back an update, or the automatic recovery from a hung server -
  relaunched the server from its original API-only command line. The GUI was
  gone afterwards, and the window that reconnected found nothing to show. The
  restarted server now brings the GUI back, and a server that never had a GUI
  attached still restarts as API-only.
- **On Windows, starting `localm gui` no longer pops a stray console window in front of
  whatever you are doing.** The crash-recovery and update watchdogs, and the coder's
  auto-started server, now run with a hidden console instead of none at all, so their
  helper Python process no longer opens a visible terminal. `localm gui --no-browser`
  also no longer shows the LocaLM status window.
- **A GGUF vision reply can fail with "the vision projector could not evaluate
  this image (mtmd_helper_eval_chunks rc=1)", wrongly blamed on a GPU-specific
  fault, when the live context is too small for the image.** Unlike the text
  path, the vision path never resized the context for what an image+prompt
  actually needs, so llama.cpp's own KV cache ran out of room mid-encode
  ("failed to find a memory slot") - identically on GPU and CPU, since it is a
  capacity limit, not a compute-backend bug. On GPU that failure was
  indistinguishable from the known gfx1030/RDNA2 hipBLAS BF16 fault, so it
  wasted a CPU rebuild-and-retry (also disguising the on_status message as a
  GPU problem) before still failing with the same unhelpful error. The vision
  path now counts the real token cost of the image and prompt up front and
  grows the context first, the same way text generation already does,
  and gives the same graceful "conversation has outgrown the context window"
  error as text does when even a resized context can't fit it.
- **Sending an image to a GGUF vision model with "Max tokens per reply" set to
  unlimited (0) no longer silently returns an empty reply.** Chat showed
  "Encoding image (GPU)..." and then nothing, with no error: the image
  prefilled normally, but the decode loop that turns it into a reply used a
  bounded `range(max_tokens)` that treats 0 (and negative values) as "generate
  nothing" instead of the "no limit" every other generation path already gives
  it. The vision decode loop now honors the same unlimited sentinel text
  generation does, so it keeps generating until the model itself stops
  (or a positive max-tokens budget is actually reached).
- **`localm doctor` no longer reports the HuggingFace backend as working while
  every HuggingFace model fails to load.** Doctor checked that PyTorch and
  transformers load in its own process, but a model loads in a separate worker
  process, where the same import can fail: the Intel GPU `[WinError 126]` failure
  fixed in this release passed that check. `localm doctor` and Settings >
  Diagnostics now run the check in a worker started the way a model load starts
  one, and say which step failed there. A check that does not finish within two
  minutes stops its worker and reports the backend as not verified.
- **A web search or page read in the GUI chat no longer pulls unrelated long-term
  memories into the reply, and the session log no longer records the GUI's web
  notes as something you said.** Search results, page text and notes such as
  "[pending action] ..." reach the model as user-role messages, and their common
  words ("answer", "results", "source") made loosely related facts count as
  relevant; one web turn could go from 1 recalled memory to 6. The GUI now marks
  those messages `origin: "tool"` (a new optional field on a chat message), and
  the server builds the recall query and the logged user line only from what you
  typed. API clients that do not send the field behave exactly as before.
- **On Windows with an Intel GPU, HuggingFace models now load instead of failing
  with `[WinError 126]` and `Error loading "...\torch\lib\c10_xpu.dll" or one of
  its dependencies`.** The process that loads a HuggingFace model could not find
  the Intel GPU runtime that the Intel (XPU) build of PyTorch installs into
  localm's environment, so every HuggingFace model failed to load with that
  build, although PyTorch itself was installed correctly.
- **Compacting a long chat in the GUI no longer recalls memories for the summary
  request or logs it as something you said.** To compact a chat, the GUI asks the
  model to summarise the older turns. The server treated that prompt, and the
  excerpt of earlier turns inside it, as your own message: it could pull in and
  reinforce loosely related memories, and it was written to the session log as
  your line, so memory consolidation later re-learned those turns as if you had
  typed them again. The GUI now marks that message with `origin: "client"` (a
  second value for the optional `origin` message field of `/v1/chat/completions`,
  see `docs/server-api.md`), and the server leaves a marked message out of memory
  recall and the log's user line. Clients that do not send the field are
  unaffected.
- **Setup no longer fails with "Failed to build `tokenizers`" on a computer
  without a Rust compiler.** A new huggingface-hub release (2.0) made setup
  pick an old version of tokenizers that has to be compiled from source.
  Setup now keeps huggingface-hub below 2.0.
- **`localm add`/`pull <folder> --store copy|move` no longer aborts partway through
  and leaves earlier models moved but unregistered when a vision projector's name
  collides with a file already in the models folder.** A byte-identical projector
  is reused in place, a different one lands under a free name, and any other name
  collision is checked for the whole folder before the first file is moved or
  copied, so a genuine refusal never strands an earlier model. Two models in the
  folder that share one projector both record it, and a different projector already
  in the models folder under the same name is never registered in its place.
- **A request that needs something the loaded model lacks is now answered by an
  installed model that has it from every part of localm, not only from API clients
  that leave `model` empty.** An image attached in the GUI chat or `localm run`, web
  access or the coding agent with a model that cannot format structured tool calls,
  a conversation that outgrows the window the model was trained for, a Knowledge
  image being described, and an MCP `chat` or `run_coder_task` call without a model
  are now answered by an installed model that can, and the reply names it. A model
  answering one such request no longer becomes the loaded model for everything
  after it, and only chat models whose file is present are chosen, never an
  embedding, diffusion or other component model. Pinning is always honored: pin a
  GUI chat to a model in its parameters, pass `--pin-model` to `localm run` or
  `localcoder`, name the model in an API request, an MCP call or a coder session,
  or set it on a scheduled job. A scheduled chat job now runs on its own model even
  while a different one is loaded, where it used to run on whichever was loaded.
  When no installed model has everything a message needs, an attached image still
  goes to a model that can read it, for example with web access on and no model
  that both reads images and makes tool calls.
- **`localm run`: an image the model could not read no longer makes every later
  message fail.** That message is withdrawn from the conversation with an
  explanation, as is a message that failed for another reason, instead of being
  sent again ahead of the next one.
- **The message for an image a model cannot read no longer calls any text model
  downloaded from HuggingFace "a vision-capable model" missing its projector.** It
  says the model has no projector recorded, and names a model in your library that
  can read images when there is one.
- **Using a model another localm instance on this machine already has loaded now
  works.** An instance with no API key can be used without inventing a key; a key
  is checked before the route is kept, and one that stops being accepted clears the
  route instead of failing every request; the chosen model no longer snaps back to
  the local one in the sidebar within 30 seconds; models are matched by file, so a
  different model that shares a name is never offered; any model that instance has
  loaded counts, not only its active one; and forwarded requests name the model the
  way that instance does.
- **Deleting saved data while uninstalling no longer deletes a folder that
  existed before LocaLM.** When you had pointed setup at an existing folder
  (a shared models drive, for example), the uninstaller treated the whole
  folder as LocaLM's. It now deletes only the files LocaLM put there, also
  after running setup again on the same folder or copying another LocaLM data
  folder into it, and lists everything it keeps. Setup no longer accepts a
  data folder that could never be deleted safely: a drive root, your home
  folder, the LocaLM folder itself or one that contains it, or a folder inside
  the environment, runtime or tooling folders that uninstall removes.
- **Uninstalling a Portable install on Windows now removes everything.** It
  could not delete the Python runtime it was running from, leaving that half
  deleted, and reported the environment as removed even when it was not.
  Leftovers are also gone: the data-folder setting, the Linux menu launcher
  file, the empty folder of the `localm` command, and the runtime's license
  and marker files.
- **A Portable setup no longer adds its private copy of uv to your PATH or
  shell startup files, or writes a uv install receipt in your user profile.**
  These pointed into the LocaLM folder and were left behind when it was
  deleted; on Linux and macOS every new shell then printed an error.
  Uninstall now also removes the ones earlier setups created.
- **A custom data folder whose path contains characters such as é now works
  after setup.** `setup.bat` saved the path in the console's code page instead
  of UTF-8, which LocaLM could not read. Setup now saves it as UTF-8, also when
  it runs again on an existing install, and the uninstaller and `rollback.bat`
  read a file saved the old way.
- **With a custom data folder, `localm setup-llama --rollback` now finds the
  build setup installed.** Setup downloaded the runtime before asking where
  your data should live, so it recorded that build in a `home` folder inside
  the LocaLM folder instead of the folder you chose, and left that `home`
  folder behind. Every setup now asks where data lives first.
- **The graphical setup (`setup-gui.bat`) now works from a folder whose path
  contains `!`.**
- **Pulling `gemma3-4b` or `gemma3-12b` now checks free disk space for the vision projector too, and the curated shortcuts list says a projector comes with them.** The disk-space preflight for a plain HuggingFace model pull now also probes the same-repo vision projector it is about to auto-attach, so a pull with just enough room for the model alone no longer passes the check and then loses the projector download to insufficient disk space. A projector already in the models folder is not counted again, and the download's progress still counts the model alone.
- **Links that open a new window now work in the Browser tab, and the automated browser no longer leaves hidden windows running.** Clicking a link or button that opens a new window now shows that window in the live view in place of the page, since the automated browser keeps one page open; before, the window opened out of sight and kept running, so the click seemed to do nothing. A window a page opens on its own, without a click, is closed and listed with the refused destinations. Typing that the browser does not finish in time now stops within a few characters instead of carrying on after the error, and a click, key press or typing that times out now says what timed out instead of showing an empty error. `POST /api/browser/type` now refuses more than 4096 characters, and `/api/browser/key` a key name over 64 characters, with a 422. Opening the browser twice in quick succession no longer starts two browsers, and `POST /api/browser/stop` sent while the browser is still starting now stops it once it is up and ends its job.
- **A settings save that fails no longer deletes or replaces your stored Hugging Face or CivitAI token.** Clearing or changing the token in the same save as a setting the server rejected (such as an out-of-range number), an embedding model change still waiting for your confirmation, or a save that timed out used to change the stored token anyway, although the save reported an error. The token now changes only when the whole save succeeds, and if the token alone could not be saved, the error says so and that your other changes were kept.
- **Saving settings while the token file or `config.json` cannot be read now says which file it is and how to fix it, instead of "Internal Server Error".** `localm config` reports the same as an ordinary error instead of treating it as a crash and offering a bug report.
- **Resumed HuggingFace downloads are safe to run alongside another download, resume on Xet-backed repos, and are verified before they are registered.** A pull now writes its own temp file and only continues from a partial whose owning process is confirmed gone, so two pulls of the same file (or another program using the HuggingFace cache) can no longer be stitched into one corrupt file. A partial left by an earlier upload of the same file is no longer counted as "already downloaded" and is cleaned up. "Resuming ... (skipping first N MB)" is now only printed when those bytes are really reused, including on Xet-backed repos where the retry previously started over from zero. Every pull whose sha256 HuggingFace publishes is now hashed against it after download; a mismatch deletes the file and fails instead of registering it. Interrupted direct-URL and CivitAI downloads also continue where they stopped on the next pull of the same URL or file; a partial left by a different URL or file is started over rather than joined onto the new download.
- **A direct-URL download whose connection drops partway now reports a failed download instead of "localm hit an unexpected error".** `localm pull <url>` no longer offers to send a bug report for an ordinary network drop; it says the download was interrupted, keeps the partial file, and the next pull of the same URL continues from it.
- **On Linux, Windows and macOS, a download cut off by a crash no longer blocks the next pull of the same file once its process id belongs to another program.** A pull still refuses while the original download is running, including after the system clock jumps (sleep, NTP, a VM or WSL resync) and when that download runs in another container or pid namespace that shares the data folder.
- **On Linux, killing a coder background job (a shell command or the server/build it started) no longer loses track of its own process after the system clock jumps (sleep, NTP, a VM or WSL resync).** A clock step could make the kill wrongly conclude the job's pid had been recycled by an unrelated program, warn that "the process no longer matches", and kill only the direct child instead of its whole tree - leaving a descendant process (such as a dev server) still running.
- **Two pulls of the same file started together after a crashed download no longer both take over its leftover lock and write into one `.part` file.** Only one continues; the other refuses, as it does for any download already in progress.
- **Curated model download shortcuts for Phi-4-mini and Gemma-3 now resolve to their correct HuggingFace repositories.** The curated shortcuts dropdown in the GUI and `localm pull <alias>` now use the `microsoft_` and `google_` upstream repo prefixes for `phi4-mini`, `gemma3-4b`, and `gemma3-12b`, resolving previous 401 download failures.
- **HuggingFace backend loading no longer emits docstring lint errors, `torch_dtype` deprecation warnings, or offloaded buffer warnings.** The model's data type is now passed under the name modern Transformers expects, layer buffers are offloaded to the CPU alongside parameters during partial offloading (preventing GPU VRAM contention on AWQ models), and stray output from the library's documentation decorator is filtered out.
- **Ctrl+C and closing the console window no longer trigger automatic watchdog restarts on Windows.** The crash-recovery watchdog recognizes a console interrupt exit (`STATUS_CONTROL_C_EXIT`, `0xC000013A`) as an intentional stop, and no longer waits out its grace period after a clean stop.
- **Closing the terminal (Linux, macOS), `kill`, or Ctrl+Break now stop the server cleanly instead of making the crash watchdog relaunch it.** A crash while the console window is closing is still captured and reported on the next start, and app-window mode no longer hangs when the server stops before its window has opened.
- **A native crash while the server is shutting down after Ctrl+C or Stop is now captured and reported on the next start.** The crash guard used to be fully released as soon as the serving loop ended, before the shutdown sequence that unloads models and the embedder had run, so a fault during that unload went unrecorded. It now stays armed through that unload and is only fully released once it has finished.
- **Restarting the server from Settings no longer opens an extra browser tab, in the
  browser or in the standalone app window (`localm[desktop]`).** In the browser, the
  tab you already have reconnects. In app-window mode the native window is reopened
  after the restart; before, the restart opened a second browser tab in either mode
  and could leave the app window closed.
- **A lone vision projector in a folder is no longer attached to a model it was not made for.** A projector beside a model is attached when its name matches the model, or when its name names no model (`mmproj-F16.gguf`, `mmproj-model-f16.gguf`) and no other model in the folder could use it: other quantizations of the same model and its multi-token-prediction head do not count, and a model with a different embedding width is ruled out. A projector named after another model in the folder is left for that model unless that model's embedding width rules it out. `localm pull` applies the same naming rules to the repo's file list, which carries no model headers.
- **Pulling a vision model no longer attaches a different model's vision projector just because the two repos happen to name it the same.** Several HuggingFace vendors ship a vision-language release with an identically named projector file (`mmproj-model-f16.gguf` and similar); a same-named file already in the models folder from an earlier pull is now confirmed to be the repo currently being pulled (its published sha256, or, when that is not published, a matching GGUF embedding width) before it is reused, matching the check the local `--store` import already applies to a filename collision. A confirmed mismatch downloads the new projector under a numbered name instead of attaching the wrong one or touching the earlier model's own file. Covers both the automatic same-repo projector detection and an explicit `--mmproj`.
- **Adding a model with `--store move` or `--store copy` no longer leaves its vision projector behind.** Every projector next to the model that may belong to it now comes along, including generically named ones such as `mmproj-F16.gguf`, instead of only one localm recognises by name. A projector another model in the same folder may also use is copied rather than moved, so that model keeps its vision, and the projector the model is paired with is recorded on its entry, so vision still works in a models folder that holds several projectors. When projectors came along but none is attached, a note names them with the `--mmproj` path to use one. When a single file is added, a same-named projector already in the models folder is reused if identical and otherwise stored under a numbered name, instead of failing the import. A projector path recorded on another model's entry now follows the file when it moves. The same applies when a single file is added with `--on-duplicate copy` / `move`, and importing a folder with `--on-duplicate move` no longer registers a projector at the location it was just moved out of.
- **Grammar-constrained generation no longer floods the console with repetitive "Grammar still awaiting trigger" messages.** The llama.cpp sampler's per-token trigger status lines, which also carried the generated text, are kept off the console, while its other warnings and errors during generation still appear there.
- **Selecting the SYCL llama.cpp backend on an Intel GPU now correctly applies
  a configured Main GPU or GPU split.** The real device numbering on a Vulkan
  or SYCL install lives in that backend's own registry, not in the generic
  probe (torch/nvidia-smi) localm otherwise uses to validate a configured
  device. Vulkan already accounted for this; SYCL did not, so choosing a
  Main GPU or a multi-GPU split on a SYCL install could silently fall back to
  device 0 or drop a valid split device instead of using the GPU you picked.
- **Stopping or restarting the server no longer leaves the coder plugin's
  background shell commands and sub-agents running.** They already stopped
  when the GUI's own background jobs and any localm-launched ComfyUI
  instance did; the coder plugin's own background jobs were missed, so a
  background shell command or sub-agent could keep running after the server
  that started it was gone.
- **Windows setup now looks for what is actually locking `.venv` before
  telling you to close it yourself.** When creating the environment fails
  with "Access is denied" (or a sharing violation), setup clears a leftover
  read-only attribute, finds any localm process from this folder still
  holding it and offers to stop it for you, and waits and retries once for a
  brief lock such as antivirus scanning. If it still cannot create the
  environment, the message now names a OneDrive-synced folder as the likely
  cause when that is what is happening, rather than only guessing at a
  stray process.
- **`localm serve -H` / `localm gui -H` no longer accept a hostname.** The `bind_host`
  setting already required an IP literal or `localhost`, but the command-line flag was passed
  through unchecked, so a server could be started bound by a name that the rest of localm (the
  local-only checks, `localm ps`, attaching to a running instance) never treats as an address
  of this machine. The flag is now refused up front with the same rule the setting uses (an IP
  literal such as `0.0.0.0`, `::1` or one of this machine's interface addresses, or
  `localhost`).
- **Stopping another running instance (the Instances card's Stop button, and `localm stop`)
  now also confirms the target's registered address is one this machine holds before sending
  it a shutdown request.** Stopping one built the shutdown request from the per-install
  registration's recorded host directly, so a tampered registration naming an outside host would
  have had that request, carrying the target's own credential, sent there instead. Such a
  registration is now reported as not an address this machine holds and skips straight to the same
  direct process-stop fallback already used when a target does not answer.
- **GPU VRAM sizing no longer overcounts a model's embedding table.**
  llama.cpp always keeps the input embedding layer on the CPU, regardless of
  how many layers are offloaded, but the preflight sizing used to charge it
  against the VRAM budget anyway. It is now excluded for every load, which
  can mean more layers (or the whole model) now offload to the GPU where
  they previously did not; the effect is small for most models and large for
  ones with a big embedding table relative to their size (Gemma's Per-Layer
  Embeddings architectures in particular). This includes an embedding table stored in a later part
  of a split GGUF.
- **Creating an API key with no capabilities checked no longer freezes the
  page.** The confirmation for a chat-only key used the browser's native
  confirm dialog, which some browsers block entirely and which locks up the
  whole page until dismissed. It now uses the same in-page confirmation
  dialog as every other confirmation in Settings.
- **Setting up localm's own ComfyUI on a machine with no usable GPU now says
  so plainly.** The installer no longer claims it is fetching "PyTorch for
  your GPU" when none was found, and prints an honest note that CPU-only
  image generation is far too slow to be practical for the shipped FLUX
  workflow. The Images page intro and the setup docs carry the same warning.
- **`localm setup-llama --tag latest` (and `update` while tracking it) now
  installs upstream's actual newest build.** It was silently falling back to
  the confirmed pinned build instead, because every real release llama.cpp
  currently publishes upstream is marked as a prerelease and that was being
  excluded; the fallback message also wrongly claimed the release lookup had
  failed. It now resolves the real newest build, and the message no longer
  makes that claim.
- **The auto-offload notice for a partial GPU load now names the real cause
  instead of always blaming the model's size.** It now says when the model
  genuinely exceeds this GPU's total capacity, when another process is
  holding VRAM right now, or when the requested context's KV cache is what
  does not fit - each with the relevant number or a suggestion (lower the
  context, free the other process's VRAM). A Mixture-of-Experts model not
  already using `n_cpu_moe` is also told the option exists. Before, the same
  "model too big for full GPU offload" message showed regardless of the real
  cause.
- **Switching models or unloading one while it is still generating no longer
  gets refused or silently degraded.** The switch/unload now stops that
  generation and completes right away for the common case (press Stop, then
  switch or unload); if something is still genuinely using the model, or the
  load would otherwise fall back to slow CPU offload, you are asked to
  confirm before it proceeds anyway. Before, you could be told a model was
  "still generating" moments after stopping it, or end up with it loaded on
  CPU with no warning.
- **A reply whose generation failed partway (the model reports an inference
  error) is now shown and saved as a failed turn.** It is marked "generation
  failed" in the chat, stays marked after a reload, is not read aloud, and a
  web search it had started to write is not run. Before, it was saved as an
  ordinary reply.
- **Automatic context compaction no longer replaces older messages with an
  error message.** When the summary request fails, the chat falls back to
  trimming the oldest messages and says so, instead of reporting that they
  were summarised. Every message a compaction removes, summarised or trimmed,
  is now kept with the conversation and included in "Export" under "Archived
  compacted messages".
- **A reply that only announces a web lookup ("I will now search ...") and
  then stops is no longer accepted as the final answer.** With web access on,
  the chat asks the model once to either make the call or answer now; this
  happens at most once per message.
  A reply that only offers a lookup ("I can search for current prices if
  that's helpful") or ends on a question is left for you to answer.
- **With web access on, a reply that stops after the model's thoughts no
  longer ends the turn with "(no reply text)".** When a thinking model writes
  its web search inside its thoughts, the search now runs; when it only plans
  one there and stops, the chat asks it once to make the call or answer.
- **Tool calls written the Llama 3 way, with `"parameters"` instead of
  `"args"`, now keep their arguments** in the chat, scheduled jobs and the
  coder. Before, such a web search ran with an empty query and a coder tool
  call arrived with no arguments at all.
- **The coder no longer passes off invented code as a file's content.** When
  a final answer shows a code block for a file in the project and the code
  defines functions or classes that file does not contain, the coder asks
  the model once to read the file and answer from it. If the answer still
  does not match, it is marked `[unverified code: ... not found in ...]`.
- **A tool call that a thinking model starts inside its thoughts now arrives
  as its reply.** With the tool-call grammar on (the chat with web access, and
  the coder by default), such a call used to be delivered as part of the
  thoughts, because the grammar lets nothing follow the call, not even the
  end of the thoughts. The call was never run and the reply was empty, which
  the coder then accepted as its final answer.
- **When the coder reads a file too large to show in full, it is now told
  exactly which lines it did not get** (`lines 103-990 of 1085 are
  missing...`), with the `offset` to read them from. Before, it only saw a
  character count, and a slice read that was cut short carried no notice at
  all.
- **Chat plugins no longer treat a failed generation as a completed turn.**
  Memory consolidation is not scheduled after a failed reply, and the audit
  log and transcript record that the generation ended in an error or a
  token-limit cut-off instead of showing it as a short normal reply.
- **Stopping another running localm instance (Settings > other running
  instances, or `localm stop`) no longer disables this server's own
  native-crash reporting.** Only the stopped instance's own crash marker and
  trace file are cleared; a later hard crash of the still-running server is
  now captured normally instead of producing no trace.
- **Privacy mode no longer leaves a crash-trace file behind after every server
  run, and a hard crash in privacy mode no longer files a bug report on the
  next start.** Crash auto-recovery still works in every mode; the native
  trace and the report are written only outside privacy mode, or with "keep
  diagnostics" on.
- **Loading a chat, embedding, or vision model in privacy mode still wrote a
  native crash-trace file for that model's isolated worker process on every
  load.** It is now written only outside privacy mode, or with "keep
  diagnostics" on, matching the server's own crash guard above.
- **A privacy-mode chat prompt could reach the browser's local storage when
  the server was still starting up (or briefly unreachable) at page load.** The
  web UI now writes nothing to local storage until the server has confirmed
  which mode it runs in.
- **Sharing a photo or text into localm from a phone in privacy mode no longer
  stages the shared content on disk.** It is held in memory until the app
  ingests it. Outside privacy mode, a share the app never picked up is now
  removed after 15 minutes instead of staying on disk indefinitely. A single
  share larger than the upload limit is refused.
- **Knowledge-base indexing no longer records the names and paths of every
  indexed document in the activity log that a bug report carries.** The
  document names still show live in the indexing progress; only the
  embedding-degrade warnings reach the log.
- **Extracting an oversized archive member or an oversized zip/tar bundle during knowledge-base
  indexing no longer records the document's or archive member's name in the activity log that a bug
  report carries.** The warning that a safety limit was hit still shows; only the
  name is withheld.
- **A music or video request whose duration is NaN is now refused instead of starting a job.**
- **Loading a model on some AMD GPUs could repeatedly report the VRAM check as
  inconclusive, trigger an automatic server restart, and occasionally fail to
  come back up on the same port.** Once the GPU runtime is loaded, torch can no
  longer be asked directly for VRAM info; the pre-load VRAM check now also
  trusts the same GPU usage reading the fit badges already rely on for a single
  adapter, so it succeeds instead of exhausting its retries. A restart resuming
  on its own port also now tolerates the brief moment the previous process
  takes to release it, instead of failing immediately.
- **Several chat turns arriving together with memory recall on and the embedding
  model not yet loaded, on a card too small for both models, could stall all
  inference for five minutes.** The chat memory inlet and the `/api/memory/*`
  routes now queue behind the embedding model one at a time, the same way the
  dedicated embeddings endpoint already does.
- **Generated music and video no longer carry your prompt inside the file.**
  ComfyUI embeds the whole workflow it ran, prompt included (for music, the
  lyrics too), as metadata in the FLAC and MP4 it writes, so sharing a track or
  clip shared the prompt with it, in privacy mode as well. Images were already
  cleaned; audio and video now are too, and localm's own ComfyUI is started
  with metadata embedding turned off. If a custom workflow produces a file
  localm cannot clean (an Opus or WebM, for example), the result says so.
- **The native llama.cpp runtime now loads on a fresh Linux install that has no
  system OpenMP library.** Upstream's `cpu`/`vulkan` Linux builds need
  `libgomp.so.1` but do not ship it; `setup.sh`/`localm setup-llama` now
  bundles a copy automatically, so a minimal distro image works without
  installing anything by hand.
- **`setup.sh` no longer throws away the rest of the install when the native
  runtime fails to provision.** It now names the missing piece, offers a
  retry, and - if you decline or it still fails - lets setup finish (PyTorch,
  data directory, launcher, menu entry) so you can provision a runtime later
  with `localm setup-llama`, instead of aborting before any of that runs.
- **The bundled llama.cpp runtime moved from b10375 to b11541.** The newer build carries upstream
  fixes and model support, and when a model is memory-mapped it can read the rows of tensors larger
  than 4 GiB from disk on demand instead of keeping them resident. An existing install picks it up
  with `localm setup-llama --force`.
- **A model whose tokenizer cannot handle ordinary sentences is now refused
  when you try to load it, with an explanation, instead of loading and then
  rejecting most of what you type.** Models declaring the `exaone-moe`
  pre-tokenizer (EXAONE 4.5, K-EXAONE) make the bundled llama.cpp runtime abort
  on a long unpunctuated span of letters and spaces, and the longest span
  localm can safely allow them is 64 characters, shorter than an ordinary
  sentence: they could be loaded but not actually talked to, and their own
  replies would poison the next turn. The refusal comes before the model is
  loaded and says plainly that no setting changes it.
- **An unattended coder run (the MCP server's `run_coder_task`, or
  `localm coder TASK`) could not write files with a model that formats its
  tool calls as ```` ```json ```` fences or bare JSON instead of the
  `<tool_call>` wrapper (Qwen2.5-Coder does), and then reported success
  anyway.** Such a call is now applied when the model's whole reply is nothing
  but exact call objects; a call quoted inside prose, a heading, a wrapper tag
  or a thinking block still needs a confirmation. A call denied for want of a
  confirmation is now named in the result and the run is no longer reported as
  a success: the MCP result is an error, the CLI warns (and exits non-zero
  under `--ci`), `--output-format json` lists it under `denied`, and a
  sub-agent's denied calls reach the parent's outcome.
- **A vision-input failure, or a model load / embedding request that timed out
  and looked hung, pointed you at a debug log even when debug mode was off and
  no such log existed.** Those messages now say how to turn one on
  (`--debug`) when it is off, and name it when it is already on.
- **The MCP server's `run_coder_task` no longer starts a separate localm server
  for every project.** It used to launch `localm coder`, which attached to or
  spawned a server per project directory, each loading its own copy of the
  model: two tasks in two folders meant two model loads, a spare console
  window, and on a card that fits one copy a native fault. The coder now runs
  inside the MCP server on the model it already has loaded, so any number of
  projects share one load, no window opens, and the model stays put while a
  task is running. A task that reaches its timeout is cancelled and reported as
  timed out: the command it was running is stopped, no further file write or
  command runs, and the model stops generating. The timeout covers the whole
  call, including loading the model, and may be at most 3600 seconds. Loading a different model
  while a task is running waits up to 30 seconds for it, then is refused with a message to retry
  instead of loading on top of it.
- **A background coder sub-agent that itself runs turns could take its siblings' results away from
  the session that started them.** Each agent now collects only the background sub-agents it
  started.
- **The coder's separate reviewer model could return a verdict taken from its
  own thinking.** With a thinking model as the reviewer, the `<think>`
  scratchpad was read as the review, so a verdict the model merely considered
  could block or approve the change. The reviewer now reads only the answer.
- **MCP servers configured for the coder were launched again for every task
  and every sub-agent, and never stopped.** One server per configured entry is
  now started and reused; sub-agents share their parent's.
- **The curated model shortcuts picker on the Models page could stay empty for
  the rest of a session.** If its very first load hit a brief server or
  network error, it was never retried; it now loads again the next time the
  page refreshes.
- **A long file path shown by the interactive chat CLI's `/image`, `/images`,
  or `/save` commands could get wrapped onto extra lines partway through a
  word, making it unreadable and impossible to copy-paste correctly.** Such a
  path now always renders on a single line, however long.
- **The model spec field on the Models page could be too narrow to read what
  you pasted into it.** At common desktop widths it shared the "Add a model"
  row equally with the shorter, optional name and sha256 fields; it now gets
  the larger share.
- **The MCP image tool now uses the ComfyUI you configured.** It always talked
  to the default local port instead, so an MCP client could miss a ComfyUI
  running anywhere else, including the one localm manages itself.
- **Music generation no longer unloads the chat model after saying it would
  keep it loaded.** When there was room for both, the job said it was
  keeping the chat model in VRAM and then unloaded it anyway, so the next
  message had to load it again. It now does what it says. With swapping
  turned off, the safety unload that prevents the two models colliding in
  VRAM also works again.
- **Saving an API key that the server rejects now says so.** A wrong or
  expired key was reported as saved and the page reloaded, discarding what
  you typed, leaving you signed out with no explanation. A rejected key now
  shows an error and keeps what you entered so you can correct it. Clearing
  the field to sign out no longer reports that a key was saved.
- **Scheduled coding jobs no longer fail on a server whose API key was set from the terminal.**
  A job could only authenticate with a key supplied through the environment, so a key stored on
  disk was ignored and every scheduled run was rejected as unauthorized.
- **Closing the web-access prompt without choosing now cancels the request properly instead of
  leaving the reply unfinished.** Dismissing the "Allow web access?" dialog with the x or by
  clicking outside it used to leave the chat waiting forever with no error and no response.
- **Restarting on Windows works again when a path in the launch command contains
  a space.** The command line was rebuilt without quotes, so an install folder
  or a model path containing a space was split into pieces and the server did
  not come back. That also broke the restart which finishes an update, so a
  healthy update was rolled back.
- **A model could stay pinned in VRAM forever after certain interrupted chat
  streams.** A rare failure while closing a disconnected stream skipped the
  step that releases the model, so it was never eligible for automatic
  eviction or idle unload until the server restarted.
- **The graphical installer now verifies downloads against your system's certificate store,
  like the console installer.** It previously used only its own bundled certificate list, which
  fails on networks that inspect and re-sign TLS traffic with their own certificate.
- **Typing a folder into the graphical installer's data-location field now selects that
  option.** A typed path used to be silently ignored unless you used the Browse button, and the
  install went to the default location instead.
- **The application menu entry the graphical installer creates on Linux now includes the
  LocaLM icon.** It previously had none.
- **Uninstalling a plugin no longer says it was never installed when the uninstall did not
  complete.** If a plugin's files, or with `--delete-data` its stored data, stay on disk
  because something is holding them, `localm plugin uninstall` now says the plugin was not
  fully uninstalled and exits with an error, instead of reporting that there was nothing to
  uninstall.
- **The plugin management HTTP API and MCP tool had their own versions of that same bug.**
  Uninstalling a plugin that was never installed under the given name over the `/api/plugins/
  {name}/uninstall` HTTP endpoint used to answer with an unrelated 500 error claiming its files
  could not be removed; it now reports that the plugin was not found. And the MCP `uninstall_plugin`
  tool's degraded-removal message used to name the plugin's installed files specifically even when
  only its `delete_data` cleanup had failed and the files were in fact gone; the message no longer
  claims a specific cause it cannot confirm.
- **HuggingFace models: an attached audio clip is no longer silently ignored, and a model that
  can only hear no longer claims it can see.** Audio sent to a HuggingFace audio model was handed
  to it under a name the library does not recognise, so the clip was dropped without a word while
  the prompt still said one was attached. It is now passed correctly together with its sample
  rate, and a clip recorded at a rate the model was not trained on is refused with a message
  naming both rates instead of being ignored, since localm does not resample. Separately, a
  checkpoint that handles only audio used to report that it could accept pictures, so an attached
  image was dropped in silence; image and audio support are tracked separately now, and whichever
  the model cannot take is refused. And a HuggingFace model that fails to load no longer leaves
  its loading process running and holding memory, so retrying after a failed load starts with as
  much free memory as the first attempt had.
- **Dismissing certain confirmation dialogs with the x or by clicking outside them could leave
  the page stuck.** Switching the embedding model, re-embedding or repairing a knowledge
  collection, and adding folders to the knowledge index all left their button disabled and the
  action silently abandoned if the dialog was closed that way instead of with Cancel. Closing it
  that way is now treated the same as Cancel.
- **The desktop launcher failed to start any mode when installed under a folder path containing a
  space or an ampersand.** It now launches correctly regardless of where localm is installed.
- **The console setup script (`setup.bat`) failed at its very first step, with no explanation, when
  run from a folder path containing a literal `!` character.** If that step were skipped, the
  desktop-shortcut option later in the same script could also fail to create a working shortcut,
  or create none at all. Both are fixed.
- **A `!` in the install path could also corrupt the step at the end of `setup.bat` that records
  what was installed, silently writing the wrong location (or none at all) for some of the
  recorded items instead of the real ones.** That step is now insulated from the same handling
  issue the bootstrap and desktop-shortcut steps were.
- **A `!` in the install path could also leave several settings silently recorded with the wrong
  (or no) location instead of the real one: the data directory, the Python/download cache
  directories, uv's own binary directory, and the optional `localm` command's PATH entry.** Each of
  those is now fixed, including the follow-on step that could otherwise fail to find a
  freshly-installed, correctly-placed `uv`.
- **Uninstalling could leave the Desktop shortcut behind on a machine with a redirected Desktop
  folder (for example OneDrive's "Back up your folders").** The installer recorded a guessed path
  instead of the one it actually wrote, so uninstall believed the shortcut was already gone and
  skipped removing it.
- **Relocating a model to a new path now rejects a network (UNC) or device path before touching
  the filesystem, and the relocation itself no longer blocks the server while it runs.** The path
  is validated as text first, so a malformed or unreachable network location is refused instead
  of the server dialing it and stalling every other request for as long as that dial takes.
- **The chat REPL's `/generate-image`, `/generate-music` and `/generate-video` commands now use
  each plugin's own configured ComfyUI address, the same one the Models page and the command line
  already use.** They previously always used the shared address, so a ComfyUI configured
  specifically for one of the three could be bypassed from inside a chat session.
- **A plugin whose installation was interrupted partway through can be installed, reinstalled, or
  removed again.** It previously stayed stuck in a broken, half-installed state that nothing in
  the app could clear, including plugins that install themselves automatically.
- **Removing such a plugin, when the removal itself fails, is no longer reported as a success, or
  over the HTTP API as the plugin never having existed.** `localm plugin uninstall`, the plugin
  management HTTP API, and the MCP `uninstall_plugin` tool can all remove a plugin directory left
  behind by an interrupted install; if a locked file, antivirus, or a permission denial stops that
  removal from completing, all three now report the failure instead of claiming success or that
  there was nothing to remove.
- **A ComfyUI address written in an unusual numeric form (hexadecimal, octal, or a single decimal
  number) is now refused when it points at a link-local or cloud-metadata address, the same as
  the ordinary dotted form already was.** Only that numeric form was previously overlooked.
- **A warning localm needed you to see, such as an admin key being revoked or a network setting
  being forced off, no longer only reaches the bug report.** The always-on activity log that
  feeds bug reports was quietly stopping such warnings from also reaching the console, on every
  ordinary run. They print to the console again.
- **Extracting text from a knowledge-collection PDF is now bounded, instead of being able to run
  for minutes on a document with an extreme number of pages.** Extraction now stops after a page
  count or time limit and leaves a note explaining what was cut, rather than working through the
  whole document unbounded.
- **Setup and the graphical installer now install the same HuggingFace/transformers version
  pyproject specifies, instead of an older version of their own that could go out of sync with it.**
- **The whole server could freeze for up to ten seconds, every request included, while a
  settings or model-registry save waited for another localm process.** A save that found the
  file busy (for example a model download the app itself started, or one interrupted mid-write)
  used to block every read of the settings, including the check on each incoming request, for
  as long as it waited. Readers are no longer held up by a waiting writer, and a lock file a
  waiting reader briefly pinned on Windows is now released instead of being left behind.
- **Several embedding requests arriving together with the embedding model not yet loaded, on a
  card too small for both models, could stall all inference (chat included) for five minutes.**
  Embedding requests on the dedicated embedding model now queue one at a time, and an unload
  the request gave up waiting for no longer runs later on its own.
- **Clicking Remove on the managed ComfyUI while an Update (or Setup) was still running could
  destroy the install.** Remove now refuses while an update or setup is in progress, and the
  `localm comfy remove` command refuses while another localm process is updating the same
  install, naming the process to wait for.
- **Several `setup.bat` warning and error messages on Windows (e.g. "uv is not installed",
  "Install failed") always printed with their `[!]` marker silently missing, on every install.**
  They now show it correctly.
- **A web search/fetch result, knowledge-base excerpt, or attached-document note in chat was
  shown as a right-aligned bubble as if you had typed it yourself, and its "Web" / "Sources" /
  "Doc" label never appeared.** It now renders as a plain, labeled note instead.
- **Fetching a web page that declared a non-UTF-8 encoding (common on older or non-English
  sites) could show garbled text instead of the page's actual content.** The page's declared
  encoding is now used to decode it, falling back to UTF-8 only when none is declared.
- **Installing the PyTorch/HuggingFace stack for a specific vendor (Intel XPU, NVIDIA CUDA, or
  AMD ROCm) from a source checkout could fail outright with an unsatisfiable dependency error
  naming `setuptools`.** The install command routed every dependency, including ordinary ones
  like `setuptools`, through PyTorch's vendor-specific wheel index, which does not mirror a
  recent enough `setuptools` to satisfy this project's own requirement. It now routes only the
  PyTorch packages through that index and leaves the rest on the normal package index.
- **Switching the coder plugin's model in a running GUI session no longer leaves the previous
  model's temperature, token limit, or tool-call-grammar support stuck on the new one.** A small
  model's steadier temperature, or a reasoning model's larger token limit, used to carry over onto
  an unrelated model after a switch instead of applying the new model's own settings; an explicit
  temperature or token-limit choice still survives a switch as before, and a model that had been
  refused constrained tool-call sampling is given a fresh chance to use it after switching away.
- **Pulling a model whose repo ships both an f16 and a bf16 vision projector (such as the
  curated `gemma3-4b` and `gemma3-12b` shortcuts) could silently auto-attach the bf16 build
  instead of f16.** The tie-break between same-repo projector variants matched "f16" as a
  substring, which "bf16" also contains, so a listing ordered bf16-before-f16 picked the
  wrong one. It now matches the precision as a whole token.
- **Installing the PyTorch/HuggingFace stack on Windows with an AMD Radeon RX 7000 or RX 9000
  GPU always failed outright, reporting no compatible PyTorch wheel could be found.** AMD moved
  its official Windows ROCm preview wheels to a new location; the install command now resolves
  the current pinned build there, and a fresh ComfyUI install on this hardware picks up the
  same fix.
- **A GGUF model load that fails while running CPU-only (`-g 0`) no longer pauses for several
  extra seconds before reporting the error.** The failure message computed a "GPU is low on
  memory" hint by probing VRAM even when the load never used the GPU, adding needless latency to
  every CPU-only load failure; the probe is now skipped for a CPU-only load, matching the
  preflight check's own behavior.
- **The `grammar` extra now requires an `apache-tvm-ffi` that xgrammar can load.** Current
  xgrammar releases fail to import with an older `apache-tvm-ffi`, which an existing install
  could keep; `pip install -U 'localm[grammar]'` now upgrades it. If xgrammar still fails to
  load, a grammar request on a HuggingFace-format model is refused with a message saying so
  instead of failing with an unexplained error.
- **Re-downloading a CivitAI file that is already there now works on Windows, and a
  re-download that fails its checksum keeps the copy you had.** `localm pull civitai:...
  --redownload` used to stop with "Cannot create a file when that file already exists" on
  Windows, and on Linux and macOS a re-download that failed checksum verification deleted
  the file that was already in place. A pull of a CivitAI file that another pull finished
  moments earlier now uses that finished file instead of failing or replacing it.
- **Chat says when an answer had no web evidence.** With web access on, an answer that follows a
  web lookup that failed, found nothing, or was refused now carries a visible "no web evidence
  was available for this answer" line, so an answer written from the model's own knowledge is not
  mistaken for one backed by a search.
- **Web calls written as `[tool_call]...[/tool_call]` now run in chat.** A model that wraps its web
  search or page read in bracket tags (also with a mismatched closing tag, or in capitals) gets the
  lookup run, and the raw tag no longer shows in the reply.
- **The Coder no longer forces tools on a status report.** A message like "Build failed." or "Search
  fails with a timeout" is treated as a statement rather than an order, while "fix failed tests"
  still counts as one.
- **Plugins, Models and Settings show what they are waiting for.** The plugin catalog and external
  list show "Loading plugins", install, enable, disable, refresh, uninstall and remove buttons show
  their working state, "Check for updates" says it is checking, removing a model shows "Removing",
  a model's info and relocate buttons show that they are working, the Models page and the
  instances card show a loading line on first open, and Settings shows that its remaining
  sections are still loading.
- **Adding one Ollama model folder keeps its vision projector.** `localm add` on a single
  Ollama manifest folder now attaches the model's vision projector, as adding the whole Ollama
  store already did.
- **Quantizations localm cannot load are named.** A Hugging Face folder quantized with a
  method the loader has no support for (for example `squeezellm`) is registered with a note
  naming the method and refused at load, instead of having its packed weights read as plain
  ones.
- **Linux CUDA setup without the GitHub API.** When the release listing cannot be read
  (offline, or the API rate limit is hit), setup installs the pinned CUDA build and checks it
  against its pinned checksum. On a machine with GPUs of different generations, one Blackwell
  GPU now selects the CUDA 13 build.
- **Stopping or timing out one ComfyUI job leaves the others alone.** Cancelling an image, video or
  music job in the app or over the API interrupted whatever ComfyUI was rendering and
  emptied its whole queue, so another job waiting there was dropped; a generation that timed out kept
  rendering in ComfyUI after localm had given up on it. Both now stop only that job: it is interrupted
  if it is running and removed from the queue if it is waiting. An image may also take up to 30 minutes
  instead of 10, so a slower GPU can finish it, and a cancelled coder run now stops the image its
  `generate_image` tool is rendering instead of letting it run on. Ctrl-C in `localm image`, `music` or `video` still
  stops everything ComfyUI is doing, as it says.
- **A PC with touch input opens straight into the app.** A Windows PC that reports touch points (a
  touchscreen, a pen, or a remote-desktop session from a phone) got the phone onboarding screen, and every
  desktop showed the camera button. Both now follow the browser's own mobile flag, a phone or tablet
  user agent, or a coarse primary pointer.
- **Signing in no longer freezes the server while another localm process is writing.** The
  browser sign-in, the GUI mount and the pairing-QR route verified a key on the server's event
  loop, so while a second localm process (a `pull`, a `key create`, another server) held the
  key store's lock, every other request, log stream and running generation stopped for as long
  as that wait, up to ten seconds. Key verification now runs off the loop, as it already did on
  every other route.
- **Two localm processes can no longer both take the settings lock after a crash.** A lock file
  left behind by a crashed process is reclaimed as before, but the file is re-checked right
  before it is removed, so two processes that notice the same abandoned lock at the same moment
  cannot both delete it and both proceed, which could lose one of their writes to settings, the
  model registry or the key store.
- **Revoking an admin-scoped key says when its browser sessions could not be dropped.** If
  removing the revoked key's sessions failed with an error, the server log said nothing at the
  default level although an admin session cookie for that key could stay usable. It now logs a
  warning naming the key.
### Security
- **Updates are signed under a new key scheme.** A release key signs every build, and three
  offline recovery keys can replace any key through a signed update, with no action from
  you. 0.2.0 updates to 0.3.0 as before.
- **Update and key-listing replies no longer hand a scoped key your file paths.** The backup
  path from `GET /api/update/rollback` and `POST /api/update/apply`, the text of a failed apply,
  and the owner-key-file notice on `GET /v1/keys` named the full location under your data
  directory. A caller other than the owner now gets the scrubbed form (data directory as
  `<data>`, file name kept); the owner and open mode still see the full path.
- **The coder's network reviewer obeys the network policy.** A cloud reviewer (`openai` /
  `anthropic`) or an off-machine reviewer URL sent the cumulative diff even with `net_mode` off or
  the host on `net_deny`. It is now checked like the GUI coder's own model server: when the policy
  refuses it (including a LAN address while `net_allow_private` is off), the review runs on the
  local model with a warning naming the refusal. A reviewer URL on this machine is not affected.
- **Hugging Face pulls are checked against the network policy.** `net_deny`, `net_allow` and the
  private-address guard now apply to the Hugging Face endpoint before any pull, and the size check
  before a single-file download checks every redirect hop and pins its address.
- **Internationalised host names match the domain lists in either spelling.** A `net_deny` or
  `net_allow` entry written in Unicode or in its `xn--` form now matches both spellings of the host.
- **The private-address guard refuses the IPv6 site-local range `fec0::/10`.**
- **Web search reads at most 4 MB of a search service's answer.** A larger answer is refused instead
  of being read whole into memory.
- **A generated image, clip or track no longer loses its owner when the gallery index is
  briefly locked.** If another program held the ownership index while the file was saved, the
  owner was not recorded and the file later became reachable by every key. The read is now
  retried, an owner that still cannot be written is enforced in memory and recorded with the
  next gallery change, and the job output, the rename response and
  `/v1/images/generations` warn until it is on disk. Renames and deletes keep ownership
  correct the same way.
- **A plugin's `requires_extras` can only name pip extras.** An entry that is not an extra name
  (for example one carrying a URL or an environment marker) used to reach the dependency
  installer as part of a requirement string; such a plugin is now refused when it is read.
- **A rejected grammar no longer lands in the debug log in privacy mode.** When a GGUF model's
  grammar check failed, llama.cpp's error message and the whole grammar went to the debug log
  even when privacy mode keeps chat content out of it; it now reaches the log only when chat
  content may.
- **The coder's `read_env` tool no longer passes credentials it called redacted.** It redacted a
  value only when the variable's name looked secret, so a `DATABASE_URL` or `REDIS_URL` with a
  password in it, or an API key under an ordinary name, reached the model as is (an online model
  included) while the terminal said "secrets redacted". It now also removes URL passwords,
  credential query parameters, bearer tokens and common API key and token formats from every value,
  removes the home directory and the account name (as a path segment, and in the variables that name
  the account or the machine) from every value, and its summary says how
  many values it redacted instead of claiming every secret was.
- **`net_mode = off` now stops the issues list.** Opening the issues list in Settings, or running
  `localm issues`, contacted the bug-report proxy even with the network switched off. It now says the
  network is off and sends nothing. The security notes now list everything that still reaches the
  network under `off`: the two admin-only exemptions, the bug-report upload (sent only when you press
  Send), requests to your configured ComfyUI, and the native runtime downloads.
- **A Hugging Face or CivitAI token is no longer resent after a redirect to plain HTTP or another
  port.** A redirect kept the token whenever the host name stayed the same; it is now dropped
  unless the scheme, host and port all match the original request.
- **A SearXNG instance on your network no longer needs the private-address guard turned off.**
  When `net_search_url` names the instance by its address, `localhost`, a dotless name or a
  `.local`/`.lan`-style name, the search reaches its LAN or loopback address on exactly that
  scheme, host and port; link-local and cloud-metadata addresses are still refused, and a public
  host name gets no exemption. The network guide used to tell
  you to set `net_allow_private true` for a LAN instance, which removes the guard from every
  page the model fetches, including cloud metadata and localm's own API; it no longer does.
- **A scoped API key no longer receives this machine's absolute paths.** The memory page, the
  image, music and video galleries (file paths and the saved generation settings, such as the source
  image of an img2img), a gallery move, the uploads list, the managed ComfyUI status,
  a saved bug report and the diagnostics report returned full paths, carrying the account name and
  data directory, to any key that could reach them. A key other than the owner's now gets those paths
  with the data directory, home and install folders replaced by placeholders such as `<data>`; "Use
  as input" and "Reuse settings" accept that form, so img2img from the gallery keeps working.
- **Model routing only goes to a localm run by your own user account.** Any program on the
  computer that answered like a localm instance on a port in localm's range could be offered
  as a peer for a loaded model, and accepting it sent it your chats and the peer's key. A peer
  is now offered, checked and used only when the process listening on its port runs under the
  same user account as this localm.
- **The cross-origin exemptions match exact paths only.** The inference routes that browsers may
  call cross-origin (chat completions, embeddings, rerank, speech, images and the surface and
  instance routes) were matched as path prefixes, so a route added under one of them would have
  skipped the cross-origin refusal and the open-mode shell-token check without anyone listing it.
  Each now matches only its own path. A trailing-slash spelling of one of those paths is no longer
  exempt.
- **An unreadable `config.json` no longer switches the network policy or the sign-in requirement
  off.** When `config.json` (and its backup) exists but cannot be read, localm used to fall back
  to the defaults for everything: `net_mode = off` read as `ask`, the `net_deny` list was empty,
  and an empty `net_allow` list lifted a strict allow-list entirely. Scheduled jobs then got the
  web tool back. Now an unreadable config is treated as `net_mode = off`: every policy-routed
  request is refused and scheduled jobs get no web tool, with a warning in the log. A configured
  `require_auth` is kept in force the same way, and the server asks for a key until the file is
  readable again.
- **A bug report is sent only after you have seen it.** In Settings > Report a bug, "Send to
  maintainer" used to file the report as a public GitHub issue in one click, without showing it.
  It now saves the report and shows its whole text, and nothing is sent until you press "Send this
  report". What is sent is that saved text, apart from its closing note about editing it;
  Retry and the automatic retry after a rate limit send that same report again.
- **Privacy mode no longer saves a bug report on its own.** An unexpected error used to write a
  report (with the error, settings subset and recent activity) to the data folder in every mode.
  In privacy mode without "Keep diagnostics for bug reports", nothing is written: a command that
  fails at a terminal asks whether to save one, and a background error saves nothing.
- **Bug reports remove more from what they carry.** The standalone reporters now remove your
  account name from paths written with doubled backslashes (as in a Python or JSON error message)
  and remove credential values from JSON keys such as `"api_key"`, as the in-app report already
  did. The report's settings section now removes bearer tokens and API keys, and scrubs values
  stored as a list or object, not only plain text.
- **The no-Python bug reporter asks before it sends, and scrubs what it sends.** When
  `report-issue.sh` has no Python and falls back to curl, it used to send the report when it
  could not ask (piped or redirected input) and sent your one-line description unchanged. It now
  sends nothing without an answer typed at a terminal, and removes your home folder, account name,
  credentials and email addresses from the description first; without `perl` it sends nothing.
  The PowerShell reporter now also replaces your home folder itself, and an emailed report has its
  subject and body scrubbed like an uploaded one.
- **One request can no longer tie up the server with an oversized input.** A scheduled job's cron
  expression is limited to 512 characters (a longer one used to stop every scheduled job from firing
  while reporting the scheduler healthy, and the Jobs page now says when the scheduler is stuck); a
  lazy grammar takes at most 64 `grammar_triggers` and stops validating them after 10 seconds; and
  an attached image is refused, before it is decoded, above 50 megapixels, past 64 images or 800
  megapixels in one request (earlier images in the conversation included), or when it is not a PNG,
  JPEG, WebP, GIF, BMP, TIFF or AVIF. These refusals answer 413 with the reason, and the chat page
  keeps the conversation's images.
- **Text localm did not write can no longer drive your terminal.** Model output, server replies,
  model and plugin metadata, search results and error messages printed by the CLI and the coder had
  their terminal control characters passed through, so a crafted reply could clear the screen, move
  the cursor to hide lines of a diff before you approved it, set the window title or write to the
  clipboard. Every console localm prints through now removes those characters, and text interpolated
  into styled output is escaped so it cannot restyle or break it. Streamed chat output is cleaned on a
  terminal and left exact when piped.
- **The default embedding models are downloaded from a fixed version and checked.** The
  first use of semantic search, `localm setup-embeddings` and the Knowledge page fetch
  bge-small (or nomic-embed) from a pinned commit, delete the file and report the failure
  when its sha256 is not the expected one, and register it with that digest.
- **A plain `coder` API key can no longer run commands through the project's coder
  configuration.** A restricted coder session (a shared, non-owner key) started any
  MCP server listed in the project's `.localcoder/config.toml` when it opened, and
  its file tools could create or change that file. So a key meant for reading and
  editing could write a server command and run it by opening a second session, and
  the same file also reached the owner's next session (MCP servers, auto-approve,
  privacy mode). A restricted session now starts no MCP server, loads no plugin
  tools or skills, and refuses to create or change anything inside a `.localcoder`
  directory at any depth; a `search_replace` sweep leaves those files alone and
  lists them. Owner sessions are unchanged.
- **A plain `coder` API key can no longer change the project's git configuration.** A
  restricted coder session's file tools could create or change files under `.git`, and
  could write into a folder that a linked `.localcoder` elsewhere in the project pointed
  at. A restricted session now refuses to write inside a `.git` file or directory at any
  depth, into any folder a linked `.localcoder` or `.git` (or a `.git` file's `gitdir:`
  line) points at, and through a file that has another hard link. Owner sessions are
  unchanged.
- **`setup.sh`, `setup-gui.sh`, `setup.bat` and `setup-gui.bat` install a fixed uv release and check it before running it.**
  They used to run whatever Astral's installer URL returned. They now download the installer
  of one pinned uv release and run it only when its checksum matches; a failed download, a
  checksum mismatch, or (on Linux and macOS) a machine with no `sha256sum`, `shasum` or
  `openssl` stops the uv install with the reason instead of running unchecked code. On
  Windows, installing uv from a PowerShell 7 window no longer fails.
- **Hugging Face models no longer download and run code from the Hugging Face kernel
  hub unless the network policy allows it.** transformers could fetch a compiled kernel
  package from the Hub while a model was loading or replying and import it, even with
  `net_mode off`. The HF backend now blocks those kernels by default and, with
  `net_mode off`, makes no Hub requests at all. Kernels are allowed only for a model
  that needs one (native FP8 on a supported NVIDIA GPU, or MXFP4, EETQ, FBGEMM FP8 or
  Metal quantization) and only when the network policy allows the Hub. With the Hub
  refused, an MXFP4 model loads expanded to full precision and says so, and a model
  that cannot run without its kernel is refused with the reason.
- **Privacy mode no longer leaves the start of tool-enabled replies in the debug
  log.** With `--debug`, `LOCALM_DEBUG` or `keep_diagnostics` on, the text a model
  wrote before a tool call (coder turns, web-enabled chat, jobs) was copied into
  the debug log token by token through llama.cpp's grammar trace, even in privacy
  mode. The log now keeps only the token ids there unless the session mode allows
  chat content in the debug log.
- **A coder session set to privacy by its project's `.localcoder/config.toml` no
  longer leaves model output in the debug log.** With debug logging on, the model
  worker process still wrote the model's raw output (and the grammar trace) to the
  debug log for such a session, because it could not see that the session was
  private. The worker now follows the coder sessions' privacy state.
- **Windows network-share paths written with the `\??\` prefix are refused like
  any other network path.** A path such as `\??\UNC\host\share` got past the
  checks that stop localm from reaching network shares and device paths
  (folder settings, coder and job working folders, model file paths), and
  Windows would then open the share. It is now refused before any file access.
- **A malicious search result or fetched web page could still attempt to forge a model role
  marker.** Web content was already stripped of literal control-token text before reaching the
  model; the chat now also marks exactly which characters of a search result or page came from
  the remote source, so the model backend refuses to parse them as anything but plain text.
- **A parallel coder child's code changes are now handed back to the parent model as untrusted
  text.** Each child's committed diff is included in the report the parent reads, so a control
  token or a fake tool-result marker written into a changed file (by the child, or by a page it
  fetched) reached the parent as if the parent had written it. The diff in that report is now
  defanged and marked so the model backend reads it as plain text only. The `/diff` and
  `/changes` footer still shows each child's diff byte for byte.
- **A downloaded model can no longer make localm read a file from outside its own folder.** A
  model split across several files lists those files in an index, and that list was trusted as
  written. A hostile or tampered model could name a file anywhere on the machine, and loading it
  would open that file, or hang on it. Such a model is now refused before it loads, and the
  refusal names the model.
- **`localm memory` no longer writes to your memory in privacy mode.** Adding a fact, restoring an
  archived one, or resolving a suggested correction from the terminal wrote to disk even in privacy
  mode, where nothing durable is meant to be written. These are now refused, with a message saying
  which setting turns them back on. Listing suggested corrections no longer rewrites their file
  either. Deleting facts and clearing your memory are unchanged and still work in every mode.
- **Removing the owner key could report success even when it only partly worked.** If the server
  could not fully complete the removal, Settings said the server was now open again and reloaded
  the page, discarding the only explanation of what had actually happened. It now shows exactly
  what did not clear and tells you that you have been signed out in this browser and need to sign
  in again with the previous key.
- **The desktop launcher no longer builds a command line from a value containing a literal quote
  character.** A HuggingFace spec or URL typed into the Import-from-URL dialog, or a saved coder
  project folder, could contain a quote positioned so Windows ran a second, unintended command
  alongside the one you launched. Such a value is now refused with an error instead of being used.
- **Creating or revoking an API key, or saving a stored Hugging Face / CivitAI credential, could
  silently wipe out every other key or credential you had if that happened to race the server's own
  use of the same file.** Both stores now refuse to save rather than losing what was already there,
  and no longer report success when that happens.
- **The coder's reviewer model can no longer be tricked into sending your diff off-machine in
  privacy mode.** A malformed reviewer URL is now refused outright, and a genuinely local one is
  classified the same way the rest of localm classifies "on this machine" everywhere else.
- **The coder's URL backend no longer follows a redirect, and an off-machine one now dials the
  address it validated rather than re-resolving the hostname at connect time.**
- **A coder session started with `--scope` can no longer hand a sub-agent a file from outside
  that scope.** The files a `spawn_agent` or `spawn_agent_background` call pre-loads into the
  sub-agent were read without the session's path restriction, so any file in the project could
  reach a sub-agent's context. Every pre-loaded file is now checked against the scope first, and
  an out-of-scope file refuses the whole call, naming the file, before the sub-agent starts.
- **An MCP server or plugin can no longer use its own tool names and descriptions to forge a
  model role marker inside the coder's system prompt.** That text was already stripped of literal
  control-token spellings when the tool was registered; the coder now also marks exactly which
  characters of its tool list came from the external server or plugin, so the model backend
  refuses to parse them as anything but plain text. Built-in tools and the rest of the prompt are
  unchanged.
- **On Llama 3, Gemma and EXAONE models, the marking of fetched web pages, search results and
  external tool output as untrusted text was silently not applied.** Those models' chat templates
  trim the whitespace around each message, and the step that locates the marked text in the
  final prompt required an exact match, so it gave up and the whole request fell back to the
  older text-only protection. It now recognises a trimmed message, so the marking applies on
  those models as it already did on the others.
- **Checking whether one of this install's own instances is still running could send a request
  to an address that instance's registration claimed, rather than to this machine.** The
  instance list (`localm ps`, the Instances card) and every command that attaches to a running
  instance confirm each registration with an identity check at the address it recorded, so a
  tampered registration naming an outside host, or a hostname, made that check dial it. The
  check now dials only an address this machine actually holds: loopback, or an address one of
  its own network interfaces is configured with (an instance bound on `::1` or on a specific
  interface is still found where it listens). A registration naming anything else is reported
  as not running, and nothing is sent to it or looked up. The check is also a single request
  now: a redirect answer from whatever is listening on the recorded port is not followed.
- **Intel GPUs were not always detected, and `sycl` was never offered as a runtime choice.**
  An integrated Intel GPU (Iris Xe, UHD Graphics) could go undetected entirely, since detection
  only recognised Arc-branded names - a box like this fell back to a "no GPU found" CPU-only
  recommendation. Detection now recognises any Intel GPU, integrated or discrete.
  Separately, `sycl` - a real, self-contained-on-Windows runtime that is often faster than
  Vulkan on Intel hardware, including integrated GPUs - was already downloadable via
  `localm setup-llama --backend sycl` but was never listed in either setup wizard's menu. Both
  the console and graphical installers now offer it as an explicit choice, and on Windows an
  Intel GPU is recommended `sycl` by default, with Vulkan as the fallback if it cannot load. On
  Linux the SYCL build needs a separate oneAPI install, so Intel there stays on Vulkan unless
  `sycl` is chosen.
- **`localm setup-embeddings --model` no longer reminds you to re-embed collections when the switch
  found none affected.** The CLI always ended with a reminder to re-embed collections stuck on BM25,
  even when the switch it just ran found nothing affected; it now reflects what the switch actually
  found.
- **Querying or deleting a RAG collection that does not exist, with an API key confined to
  specific folders, now returns 404 like it does for every other caller**, instead of a 403
  that implied the collection exists but is off-limits.
- **The curated ComfyUI models are checked against a fixed sha256.** Downloading one of the eight
  models offered when a ComfyUI workflow is missing a file now fails and deletes
  the file when its bytes are not the expected ones. The weekly pin report lists these digests
  and flags one the hub no longer serves.
- **The Whisper speech models are fetched at a fixed version and checked.** The tiny, base,
  small and medium models are downloaded from a pinned commit of their repository, and the
  downloaded `model.bin` is deleted and refused when its sha256 is not the expected one. A
  model already downloaded keeps loading offline as before.
- **The FP8 kernel that Hugging Face models use is fetched at a fixed version.** Loading an FP8
  checkpoint that needs the `finegrained-fp8` kernel downloads it from a pinned commit instead
  of whatever its version name points at when you load. A transformers release that asks for a
  different kernel version loads no native kernel, so the FP8 weights are expanded to bf16 and
  the load says why.
- **The Kokoro voice model is fetched at a fixed version.** The browser now downloads the
  default Kokoro model and its voices from a pinned commit of their Hugging Face repository,
  not from whatever the repository's default branch holds that day. A model already downloaded
  and cached in the browser is still found. The weekly pin report flags the repository moving
  past the pinned commit.
- **Downloads from Hugging Face are checked more completely.** Every part of a split GGUF is
  checked against the digest the hub publishes, not only the first; a vision projector is
  checked and deleted when it does not match; a file already on disk is checked before it is
  registered under the hub's digest, and refused when it does not match; each weight file of a
  whole-repository download is checked. The file metadata and the bytes are read from one
  commit, so a repository that changes during a pull no longer fails the check. When no digest
  can be read, the pull now says so instead of finishing silently. When a file already on disk is
  registered from another repository, the refusal says the file name is in use, names the model that
  holds it, and leaves the file alone.
- **The starter models are checked against a fixed sha256.** `localm pull llama3.2-1b` and the
  other ten starter models (also from the Add a model list) refuse a download whose bytes are
  not the file this release of localm was built with, and say so when Hugging Face now serves a
  different file: pass `--sha256` with the digest it shows to accept that file. A digest you
  pass yourself is never replaced.

## [0.2.0] - 2026-09-04

### Added
- **The Browser tab can now show the browser the coding agent is driving.**
  It used to open a browser of its own instead, so the agent's own page was
  never visible. A Watch the agent button appears whenever the agent has one
  open. A shared key can only watch sessions it started itself, never yours.
- **Settings can remove the owner key again.** Setting one from Settings >
  Security used to be a one-way door: nothing in the interface could take the
  server back out of protected mode. Removing it asks first, and says plainly
  that the server returns to open mode.
- **localm can now pick a capable model for a request instead of just using
  whichever one happens to be loaded.** Each model is checked for what it can
  actually do - reading images, emitting structured tool calls, how long a
  conversation it was trained to hold - and a request that needs something the
  loaded model does not provide is answered by an installed model that does.
  Sending a picture to a text-only model no longer just fails: if you have a
  model that can see, it answers. **A model you picked yourself is never
  swapped out from under you.** When the model you chose lacks something the
  request needs, localm tells you what is missing and which of your models has
  it, and leaves the choice to you. Where a model has not been inspected yet,
  localm says so rather than claiming it cannot do the job, and a response
  carries a short note saying which model answered and why, so an automatic
  choice is never invisible. Starting a coder session on a model with no
  tool-calling support now points out which of your models are built for it -
  tool calls still work either way.
- **Text-to-speech now speaks Spanish, French, Hindi, Italian and Brazilian
  Portuguese.** The neural voice previously read every language with English
  pronunciation, because the speech engine only ever loaded its English
  pronunciation rules. Those five languages now use their own, so Spanish
  "Jorge" is spoken as "Jorge" rather than "Corque" and Portuguese "Rio" as
  "Rio" rather than "Kio". Thirteen new voices appear in the Settings picker,
  which is now grouped by language, and nothing changes for English. Upstream
  grades these voices lower than the English ones and does not grade the
  Spanish and Portuguese ones at all, so the picker shows a grade only where
  one exists. Japanese and Chinese are deliberately not included: the engine
  cannot pronounce their scripts and produces placeholder text rather than
  speech.
- **Two localm instances on the same machine can now share one loaded model
  instead of each keeping its own copy in VRAM.** When you switch to a
  model, the Models page checks whether another running instance already
  has it loaded; if so, it offers to route your requests there instead of
  loading a second copy. Accepting asks for that instance's API key and
  routes chat requests to it directly. Declining, or when no such instance
  exists, loads locally exactly as before. If the other instance later
  becomes unavailable, the next request tells you plainly and you can load
  a local copy or route again.
- **`localm bench-mtp MODEL` answers whether Multi-Token Prediction is worth
  turning on for a model.** The MTP setting helps some models and slows others
  down, and which is which depends on the model, the quantisation, how much of
  it fits on your GPU, and how fast your machine is - so instead of guessing,
  this loads the model both ways, generates the same prompts through each, and
  tells you which was faster and by how much. It changes nothing: applying the
  answer stays your call. If the model has no MTP head it says so rather than
  reporting a meaningless number, and if MTP loses while part of the model is
  running on the CPU it points that out, since fitting the whole model on the
  GPU can change the answer.
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
- **CivitAI models can now be searched and pulled from the command line**,
  alongside HuggingFace: `localm search --source civitai <query>` and
  `localm pull civitai:<versionId>`. A downloaded file is verified against
  CivitAI's own checksum and lands in the active ComfyUI's matching
  `models/<type>` folder (checkpoints, LoRAs, VAEs, textual inversions) so
  workflows can find it directly, rather than in localm's own flat models
  folder. NSFW results are excluded by default, and content flagged as
  depicting a minor is always excluded. Files in an older, riskier format are
  hidden unless you explicitly ask to see them.
- **CivitAI search is now on the Models page too**, alongside HuggingFace - a
  source picker switches the whole search card between the two. Results show
  CivitAI's own commercial-use, derivatives, and credit permissions (not an
  SPDX license string, since CivitAI does not use one), and each file lists
  its CivitAI safety-scan status. NSFW and legacy-format results stay opt-in
  toggles, off by default, matching the CLI's behavior.
- **The coder can now find where a function or class is called before you
  change it.** A new `find_references` tool searches the project it has
  indexed for call sites of a symbol, so it (and a read-only reviewer
  sub-agent) can check who else calls something before touching its
  signature, without a manual grep round-trip.
- **localm's MCP server now ships as a portable Agent Plugin bundle** (a
  `plugin.json` manifest, `.mcp.json`, a `marketplace.json`, and a
  `skills/localm/` skill at the repo root), instead of pasting the MCP
  server's JSON config by hand. Install it persistently with `claude plugin
  marketplace add Matlan1/localm` followed by `claude plugin install
  localm@localm`, or load it for one session only with `claude --plugin-dir
  <path to this repo>`. The skill gives the assistant operational guidance:
  how to judge model/quant fit before downloading, when to delegate a whole
  task instead of a single chat call, and what image generation needs.
- **A curated picker for the Kokoro voice model, in Settings > Text-to-speech.**
  Switching models used to mean typing a raw Hugging Face repo id into a blank
  box with no discoverability. It now offers the shipped default plus a
  verified alternate with word-level timestamps, with a "Custom" option that
  still accepts any other Kokoro-compatible repo id for power users.
- **The letter grade shown next to each voice (e.g. "Heart (en-us, Female,
  A)") is now explained in Settings.** It rates that voice's training data,
  not a version number: higher means more and cleaner training audio.
- **The interface can now be shown in German, and the language is a setting.**
  Settings > System > Appearance has an "Interface language" picker; the choice
  is saved on the server, so it follows the instance to every browser that
  connects to it, and `localm config language de` sets the same thing from the
  command line. The chat view and the app's shell - the sidebar, its navigation,
  the conversation list, the model box, the message composer and the chat
  parameters drawer - are translated, along with the Settings page's own
  headings, search box, section navigation and Appearance card, the Plugins
  page, the Images page, the Knowledge page, the Models page, and the Coder
  panel. Everything not translated yet stays in English rather than going
  blank, and switching language takes effect immediately without a reload.
  The rest of the app (Music and Video in Studio, and the individual
  settings descriptions) is still English only; adding another language now
  means adding one catalog file rather than changing the app.
- **Optional Hugging Face and CivitAI API tokens.** Set them in Settings
  under Library, or the `HF_TOKEN` / `CIVITAI_API_KEY` environment
  variables, to raise rate limits and reach gated or login-required models
  on search and download. Neither is required: search and downloads keep
  working anonymously with nothing set.
- **Chat messages can now show a background image.** Set it in Settings >
  System > Appearance, next to the logo picker; Clear removes it.
- **You can set your own display name for chat messages**, shown next to your
  messages instead of "You". Set it in Settings > Chat > Avatars.
- **The per-model icon override in Settings > Chat > Avatars now offers a
  dropdown of your installed models** instead of requiring you to type an
  exact model name.
- **LocaLM now has a graphical installer, so you never have to answer setup
  questions in a console.** Double-click `setup-gui.bat` (Windows) or run
  `./setup-gui.sh` (Linux/macOS) and click through four pages, with Back to
  change an earlier answer: which inference runtime to use (your hardware's
  recommendation is already selected, including Apple Silicon and AMD ROCm);
  where models and data live, and whether the Python tooling stays inside the
  folder so nothing is written to your user profile; which optional features
  to install, with the recommended ones already ticked; and whether the GUI
  opens as its own app window, whether you want a desktop shortcut, and
  whether `localm` should run from any terminal. It then installs with a
  progress bar and a live log, says plainly if an optional step did not
  finish, and offers to start LocaLM. It performs the same install as
  `setup.bat` / `setup.sh` and records what it created, so uninstalling
  afterwards removes exactly that. Those console installers are unchanged and
  remain available for anyone who prefers them.
- **The preview canvas can now be turned off, or limited to you.** A reply
  containing a self-contained HTML or SVG block gets a button that renders it
  in a sandboxed frame. Two settings under Chat now govern it: one turns the
  button off everywhere, and one offers it only to your own sessions rather
  than to every API key. Both keep the previous behaviour by default.

- **The coding agent can drive a real web browser.** Off by default. Once you
  install the browser extra, download the browser it drives, and switch it on
  in Settings, the agent can open a page, read it, click, fill in forms, take
  screenshots, and read the page's console and network activity. It reaches
  only what your network policy already allows: every request the page makes is
  checked, including images and scripts the page pulls in on its own and every
  hop of a redirect, and anything refused is reported with the reason.
  WebSocket connections are refused rather than relayed, and say so. An API key
  needs the new
  browser capability, which is separate from the coding agent's shell access,
  so you can grant one without the other.

- **You can watch the browser work.** The Browser tab shows the page live as
  the agent drives it, and lists any address the network policy refused,
  along with the reason. Coder and Browser now share one collapsible group in
  the sidebar, the way the media tabs already do.

### Changed
- **The two installers explain themselves more consistently.** `setup.bat`
  now tells you a backend pick is load-tested and falls back to Vulkan
  rather than swapping silently (matching `setup.sh`), and points a failed
  PyTorch/transformers install at `docs/gpu-setup.md` instead of leaving you
  with no next step. `setup.sh`'s closing summary now also confirms when an
  application-menu entry was created.
- **`setup.bat` now asks whether to use GPU acceleration when one is
  detected, matching `setup.sh`.** Declining makes CPU the pre-selected
  default in the backend menu that follows, so accepting that menu with
  Enter can no longer silently download and load-test a GPU runtime you
  did not want; every backend, including CPU, stays available to pick
  explicitly either way.

### Fixed
- **`rollback.sh` would not start on Linux or macOS.** A fresh copy of the
  project did not mark it runnable, so it refused with a permission error,
  which mattered because it is what you reach for when an update went wrong.
- **`localm image`, `localm music` and `localm video` ignored the ComfyUI
  address set for that plugin.** They always used the shared one, so a plugin
  pointed at its own ComfyUI worked in the interface and quietly went somewhere
  else on the command line.
- **`localm plugin setup` could never offer the browser plugin.** It ships with
  localm and the Plugins page listed it as available, but the command the
  installers themselves run could not see it, and asking for it by name failed.
- **A browser that failed to start said only "Browser closed."** The reason was
  thrown away, so a missing browser download looked like a normal shutdown. It
  now reports what went wrong, and a bundled browser that has not been
  downloaded yet names the command that fetches it.
- **`localm update` no longer tries to replace a pip install.** There it would
  have written a release tree over the directory it shares with every other
  installed package. It now points at pip instead.
- **The coder in a terminal ignored the browser setting.** Turning the browser
  on only affected the coder in the web interface; `localm coder` never got
  the browser tools at all.
- **The default system prompt was applied by the web interface only.** A chat
  started with `localm run` or `localm chat` began with no system prompt,
  whatever the setting said. `--system` still overrides it for one chat.
- **The "Other instances" list (Models page) could take several seconds to
  refresh when more than one localm server was running.** Each running
  instance was checked one at a time, so the wait added up with every extra
  instance. They are now all checked at once, so the list loads about as
  fast as the slowest single instance instead of the sum of all of them.
- **A HuggingFace model's generation failure could leave a reply stuck for
  fifteen minutes instead of failing right away.** If generation hit a real
  error partway through (an out-of-memory condition, a bad setting, or a
  similar failure), the reply just sat there with no new text until a generic
  "stalled" timeout eventually gave up, with no indication of what had
  actually gone wrong. The real error now surfaces within seconds, with its
  actual cause, instead of after the long wait.
- **Repairing a damaged Knowledge collection could silently drop its semantic
  search, with no warning.** Repair recomputes embeddings by default, and is
  meant to ask before proceeding when that will not happen and existing
  embeddings would be lost - but it always assumed an embedding model was
  available, so the warning never showed even when none was installed. Repair
  now checks first, and asks before dropping to keyword-only search when it
  would.
- **A crashed run's native fault trace, when attached to a bug report, was
  checked only for your home folder name, not for stray credentials.** The
  report's other attachments (the recent log tail, an event-loop hang trace)
  were already checked for API keys, tokens, and passwords before being
  included. The native trace now gets that same check, so a credential that
  happened to land in it can no longer ship in an uploaded report.

- **The browser tab notices when localm has restarted.** The offline cache was
  also storing the small status reply the page uses to tell one running localm
  from another, and then serving that stored copy back forever without ever
  asking the server again. Because that reply is exactly how the page spots a
  restart, the "reconnecting" overlay could sit there indefinitely after
  localm came back up. Status replies are no longer cached; only the interface
  files are.

- **The coder's delegated-work summary no longer loses its status and its
  "diff truncated" notice.** In `/changes` and `/diff`, the summary of work done
  on another branch is printed with each entry's state in square brackets. Those
  brackets were being read as formatting and silently removed, so the state
  disappeared, and so did the note saying a long diff had been cut short along
  with the command to see the rest. A truncated diff could therefore read as a
  complete one.

- **The Coder stopped working once you set an owner key from Settings >
  Security.** Starting a session, or asking for an estimate, failed with
  "This server requires an API key" - even though one genuinely was set. The
  Coder's own request back into this same server was using the wrong
  credential in that case; it now uses the key you actually set.
- **Settings > Security now tells you if your owner key silently stopped
  working.** If the key file exists but is (or becomes) empty, localm falls
  back to running without a key - previously visible only in the debug log,
  now shown right in the Owner key section.
- **The Jobs page now tells you if scheduled jobs have stopped firing.** A
  persistent scheduling failure (for example a corrupted job store) used to
  silently halt every scheduled job with no visible sign; it now shows as a
  warning on the Jobs page.
- **Settings > Runtime & GPU now warns you when the llama.cpp runtime could
  not be resolved as configured** - a broken runtime install, or an invalid
  `LLAMA_CPP_LIB` override, used to be reported only to the debug log.
- **Embedding long text no longer fails on runtime builds that ignore the
  shared-cache request.** localm asks its embedding runtime for one shared
  attention cache so several texts can be encoded in a single pass. Some builds
  silently ignore that and hand each text a much smaller private slice instead;
  anything longer than that slice was then rejected, and embedding requests
  (with the RAG indexing built on them) failed with a server error. localm now
  reads back the per-text window the runtime actually granted and trims to it,
  so those builds keep working. Unaffected builds behave exactly as before, and
  the warning that a runtime ignored the request still appears in the log.
- **The GPU load figure now reads the card's own sensor on older AMD boards
  too.** It already showed whole-card load, whichever program was causing it,
  on recent Radeons. On an older board, or an older driver, the sensor it asks
  for is not available, and it quietly fell back to a Windows counter that
  reports whichever part of the card is busiest - so a video call's encoder
  could read as high load while the card was otherwise idle, or generation
  could sit near 100% and barely register. Those boards are now asked through
  the older AMD interfaces they do answer, and the Windows counter is used only
  when the card answers through neither.
- **The list of other running LocaLM servers now really does cover the whole
  machine.** Settings > Server & network said it listed every server running on
  this machine, but it only ever saw servers that share this install's data
  folder. A second LocaLM installed somewhere else stayed invisible while it was
  running, holding the graphics card and answering on its own port, which was
  exactly when you needed to know about it. Those servers are now listed too,
  marked "other install", with their directory and address. They carry no Stop
  button on purpose: this app holds no shutdown credential for another install,
  and ending one outright would skip its model unload and leave it reporting a
  crash that never happened, so it has to be stopped from its own window or the
  terminal it was started in.
- **Knowledge collection names refused spaces and emoji, even though the rest
  of the name was fine.** Trying to create a collection called "my research
  notes" or add an emoji to the name failed with a 400 error. Spaces and
  Unicode (including emoji) are now allowed; a name still can't contain a
  path separator, a period, or other characters that aren't safe on disk.
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
- **If those automatic retries still cannot get a clear reading, localm now
  tries restarting itself before giving up** - the same seamless in-place
  restart Settings' restart button uses, with no interruption beyond a brief
  reconnect. Only when even that is not possible does it ask you to file a
  bug report, and it no longer suggests restarting anything yourself.
- **Restarting the server could take noticeably longer than it needed to,
  even with nothing loaded at the time.** It always checked how much free
  graphics memory was available before restarting, even when there was
  nothing using any and no reason to wait for memory to be freed. That check
  is now skipped whenever there is nothing to release, so a restart with no
  model loaded completes much faster.
- **Clicking an already-open coder session now switches to it instead of
  opening it again.** Clicking a session that was already running started a
  second, independent copy of it every time, each holding its own connection
  open indefinitely. Enough of these open at once could make the rest of the
  app stop responding - most visibly, the Plugins page's list would stop
  loading. Reopening a session you already have open now does nothing more
  than bring it to the front, the same as every other session in the list.
- **Opening a DIFFERENT past coder session for a folder that already had one
  open switched to the open one and said it had worked.** Every past session
  in a folder was treated as the same session, so picking the second one in
  the list quietly left you in the first, reporting "already open". Picking a
  different saved conversation now offers to end the open one and continue the
  one you actually picked, since a folder runs one session at a time. Reopening
  the session that is already open still just brings it to the front.
- **The session that is currently open is no longer also listed under "past
  sessions".** A running session is saved as you go, so it appeared in both
  lists at once and could be "continued" while you were already sitting in it.
- **Reopening a past coder session no longer drops turns that only ran tools.**
  A step that read or wrote a file and said nothing else vanished from the
  restored conversation, so reopening it read as though that work had never
  happened. Those turns now appear, naming the tools that ran.
- **Restarting the server from Settings could leave the page stuck on a
  reconnecting screen instead of coming back.** The page could reload before
  the old server had actually finished shutting down, landing back on the
  server that was already on its way out rather than the freshly restarted
  one. It now waits for confirmation that a genuinely new server process has
  come up before reloading.
- **The first page load right after a server restart could freeze the whole
  server for ten-plus seconds, twice in a row.** The Settings page fetches the
  Companion-app address and the device-pairing QR code as soon as it loads,
  and both of those ran in a way that blocked every other request while they
  worked, including the ordinary "are you back up yet" checks a restart itself
  waits on - so the very first reload after a restart could make an otherwise
  healthy, freshly restarted server look stuck or unreachable. Both now run
  without blocking anything else.
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
- **The coder's "working…" pill could sit motionless for minutes during a
  long generation, with nothing to show it was still going.** It previously
  only changed once new content actually appeared, so a long quiet stretch
  before the first token (or between tool calls) looked identical to the
  page being stuck. It now pulses and counts the seconds since the last
  activity for as long as a task is running, including through a silent
  gap.
- **Settings' "Server port" could show the wrong port.** If the server was
  actually running on a different port than the one saved in Settings (an
  explicit port on the command line, or an automatic move to a free port
  because the saved one was busy), the field only ever showed the saved
  value. It now also shows the port the server is actually running on
  whenever the two differ.
- **"Max tokens per reply" could not be set to unlimited.** Entering 0 was
  rejected, even though 0 already means unlimited for several other similar
  settings. It is now accepted here too, with the same meaning: no
  runaway-guard limit, bounded only by the model's context window.
- **Text in the app window (the standalone desktop window, not a browser tab)
  could not be selected or copied.** Selecting it works, and so does copying:
  Ctrl+C, Ctrl+A and the right-click menu were all switched off in that window,
  so there was no way to get an error message, a reply or a setting out of it.
  All three work now, and the window still does not open developer tools.
- **Clicking an outbound link in the app window, such as the changelog's
  release link, navigated the whole window away to that page instead of
  opening it.** The app window has no address bar or back button, so there
  was no way back short of restarting the app. Outbound links, including
  those in chat replies, now open in your regular browser instead.
- **An explicit WebGPU voice setting could keep producing corrupted, buzzing
  audio instead of stopping.** Kokoro's WebGPU backend has a known audio
  corruption bug on many GPUs, which is why the default already avoided it;
  forcing WebGPU in Settings still hit it. Playback now stops and tells you
  to switch back to WASM the moment real corruption is detected, instead of
  continuing to play the noise. The Settings help text for this option was
  also actively wrong, saying WebGPU is what the default uses when it is not.
- **The chat "speak aloud" button gave no sign it was playing, or that
  clicking it again would stop it.** It now shows while a reply is being
  read aloud, with the icon swapping to a stop button, so it is clear both
  that audio is playing and how to stop it.
- **The first "speak aloud" click after loading the page could take several
  seconds before any audio started.** The voice model was only ever warmed
  up on demand, at that very first click. It now warms up in the background
  as soon as a reply finishes, when that needs no extra download prompt, so
  playback usually starts right away by the time you click.
- **`localm serve`, and `localm gui` when it attaches to an already-running
  instance, said it was "Opening" the address even when nothing was going to
  open.** `serve` never opens a browser, so every `serve` startup that
  attached printed a claim that wasn't true. That line now just states where
  the server can be reached, accurate whether or not a browser actually opens.
- **A model load aborted for an unrelated reason reported a made-up "superseded
  by a newer request: None" instead of what actually happened.** This showed up
  as the error when using the API or the coder plugin's model switcher, if the
  load was cut short by something other than picking a different model. The
  real reason is now shown instead; picking a genuinely different model while
  one is still loading is unaffected and still reports it as superseded.
- **An embedding model that failed to load could leave its worker process
  running in the background.** This worker handles semantic search and agent
  memory, and a load that failed cleanly, such as a broken or misconfigured
  model file, was never cleaned up, so repeated failed attempts each left
  one more stranded process behind, still holding memory. That worker is
  now shut down whenever a load fails this way.
- **A model with Multi-Token Prediction turned on could run out of VRAM, or
  get auto-sized fewer GPU layers or a smaller context than it needed, even
  when the load looked like it should fit.** MTP's own draft context needs
  extra VRAM beyond the model's weights, and that cost was never counted
  before a load, so the low-VRAM preflight check and the automatic
  GPU-layer/context-size sizing were both blind to it. They now reserve room
  for it up front.
- **Network access set to "Ask" left semantic search and memory recall
  silently stuck on lexical-only search, with no hint that a one-time
  download would fix it.** The status message now says the download is
  available right now instead of reading like a hard refusal, and the
  Memory panel offers the same one-click "Download now" action the
  Knowledge page already had, so a Memory-only user has a way to fix it too.
- **`localm comfy restart` reported success without restarting anything, for
  a ComfyUI you started yourself.** It aborted the in-flight render as
  intended, then reported "ComfyUI is running." either way - indistinguishable
  from an actual restart. It now says plainly that nothing was restarted and
  that ComfyUI is left running, so you know to restart it yourself.
- **`localm gui --api-mode` printed startup hints that pointed at pages it
  never serves.** The phone/LAN-access line offered a Settings page
  alternative, and four separate no-model-loaded messages talked about
  "Opening the GUI" or picking a model "on the Models page" - none of which
  `--api-mode` ever mounts. Those lines are now api-mode-aware and describe
  what is actually available instead.
- **localm can now identify its own version in more situations**, instead of
  falling back to "unknown." An install with no readable version file next to
  it and no usable package metadata (uncommon, but possible on a damaged or
  unusual install) previously could not tell what it was running - which also
  meant the update checker could not confirm a downloaded build was genuinely
  newer before installing it. Release builds now carry a fallback copy of
  their own version number, so this case is now rare.
- **A chat request could be rejected as too large for a very small context
  model even though your own message fit comfortably.** Every chat turn adds
  a short style nudge to the model's instructions; on a normal-sized context
  that costs nothing you would notice, but on a tiny one it could push an
  otherwise-small request past the model's context limit by itself. The
  nudge is now skipped when the model's context is too small to absorb it,
  so a request that actually fits succeeds instead of being refused.
- **Models page: the HuggingFace search box was squeezed to a sliver.** The
  source dropdown next to it claimed the whole row, leaving the search field a
  few pixels wide and effectively unusable. It now takes the space the dropdown
  is not using, and the same fix widens the owner/repo field under "Add a model".
- **Coder: the Stop and End buttons could not be reached on a narrower window.**
  The session toolbar ran off the edge of its column, so on a 1280px-wide window
  everything from "memory" rightwards, including Stop, sat under the sessions
  list with no way to scroll to it. The toolbar now wraps onto a second row.
- **localm no longer reports a crash that did not happen.** Starting the
  standalone app window makes Windows raise and immediately handle a harmless
  internal error. localm was treating any such record as proof that the
  previous run had died, so after any unclean stop it announced "localm server
  crashed - native fault captured", saved a bug report and named that harmless
  error as the cause. It now distinguishes errors that actually end a process
  from ones a program raises and handles, and says plainly when the cause is
  simply unknown. Real crashes are still reported, and still name the real
  fault.
- **A failed launch no longer closes its window before you can read the
  error.** Starting localm from the launcher or a shortcut opened a console
  that vanished the instant the process stopped, so a startup refusal (for
  example, a second copy starting in a folder that already has one) flashed
  red and disappeared. The window now stays open when a launch fails; a normal
  exit still closes it.

### Security
- **A fetched web page or external tool result can no longer impersonate you or
  the assistant, whatever model you run.** Text that localm brings in from
  outside - a page the coder fetched, a web search result, output from an MCP
  server - can contain the special markers a model uses to tell one speaker's
  turn from another. Until now those markers were removed by matching against a
  list of the model families localm knew about, so a family that was not on the
  list could smuggle one through, and the model would read the injected text as
  a genuine system or assistant turn and act on it. localm now tracks exactly
  which part of a prompt came from outside and tells the model's tokenizer to
  read that part as plain text, so those markers cannot take effect there no
  matter which family they belong to, while the markers your model's own chat
  format needs keep working normally. This covers both the GGUF and the
  Transformers backends.
- **That same protection now covers what localm's own helpers feed back to the
  model, not just what it fetches from the web.** A sub-agent you delegate to
  reports back in its own words, a past lesson is recalled from memory, a diff
  goes to the reviewer model, and a long session gets summarised - and each of
  those can quote text that came from outside in the first place. Those paths
  were still relying on matching the marker against a list of known model
  families, so one belonging to a family that was not on the list could survive
  and be read as a genuine speaker change. They now carry the same
  came-from-outside marking as a fetched page, so the tokenizer reads that text
  as plain text whatever family the marker belongs to. The web search and page
  fetch endpoints also now say which parts of their answer came from outside, so
  anything built on them can pass that along.
- **Chatting with certain models no longer crashes the server outright.** For a
  handful of tokenizers - among them Llama 4, Mistral's Tekken, EXAONE MoE,
  GPT-4o-compatible and SuperBPE ones - a message containing a long unbroken run
  of letters or digits, such as a pasted identifier, a hash or a DNA sequence,
  made the part of llama.cpp that splits text into words abort. That took the
  whole model process down with it: the reply was lost, the model was unloaded,
  and anything else in flight died with it. No unusual message was needed and
  the same input crashed it every time. Such text is now refused before it
  reaches the tokenizer, with a message saying how long the offending run was
  and how long a run that tokenizer allows, and the model stays loaded and
  keeps serving everything else. Embedding and image prompts are covered too.
  Ordinary writing is unaffected, since punctuation and line breaks end a run,
  and models with unaffected tokenizers are not checked at all.
- **A long run of blank lines crashed the server the same way, and the fix
  above did not cover it.** The check looked for long runs of letters or
  digits, and a line break is neither, so pasting a few thousand consecutive
  blank lines still took the model process down. It affected eleven tokenizers:
  the ten covered above, among them Llama 4, Tekken, GPT-4o-compatible,
  Granite's multilingual embedding model and the Cohere and Aya families, and
  also JAIS-2, which was not previously checked at all. Runs of line breaks are
  now bounded too. Ordinary documents are unaffected, since the limit is far
  above any normal run of blank lines, and the three tokenizers measured to
  handle long runs of them safely are left unchecked.
- **Two tokenizers could make the server crawl rather than crash.** With a
  JAIS-2 model loaded, a message padded with a long stretch of spaces or tabs
  took about a second just to be split into words. With a DeepSeek model, text
  written as one unbroken block with no spaces or line breaks anywhere - which
  is how Chinese, Japanese and Korean are normally written - got slower and
  slower the longer the block ran, reaching over thirty seconds at the largest
  message the server accepts. Neither of those was a crash, so neither was
  caught by the crash check above. Both are now bounded, and the refusal says
  the text is too slow to process rather than claiming it would crash. Ordinary
  use is untouched: a CJK message of a few thousand characters, a long CJK
  document with line breaks in it, and source code or English prose at the full
  length limit all pass.
- **A bug report now hides credentials that were logged as structured data, not
  just ones written as plain settings.** A report already blanked out things
  like `api_key=...` and `Authorization:` headers before you sent it, but a
  credential that reached the report inside a block of JSON, which is the form
  the browser records an error object in, slipped through and was sent in the
  clear. Those are now blanked out too, and the surrounding detail is kept so
  the report stays useful. Two report sections that relied on being cleaned up
  earlier in the process now get the full treatment where they are written, so
  a credential cannot reach them by a route nobody anticipated.
- **A model can no longer take over your terminal through the coder's own
  output.** Text a model produced was printed to the terminal without being
  made inert first, on several paths: the narration it writes alongside a tool
  call, its final answer whenever that looked like Markdown, the reviewer's
  verdict shown by `/review`, the project memory shown by `/memory`, the file
  paths in `/changes`, and the background-job labels in `/bg`. A model could
  therefore embed a raw terminal control sequence and clear your screen or
  overwrite lines already printed, hiding what it had actually done; it could
  render a clickable link whose visible text said one thing and whose
  destination was another; or it could emit one malformed tag and abort the
  turn outright. All of it is now escaped, so it renders as the literal text it
  is. The same applies to the line naming which sub-agent is asking before a
  confirmation prompt: a sub-agent could previously make that line vanish by
  putting a malformed tag in its own name, leaving you approving a command with
  no idea which agent had requested it.

- **A shortcut or linked folder inside your project no longer leads the coder
  out of it.** The coder refuses to read a file outside the project you pointed
  it at, but two things that walk the project - the file tree it shows you and
  the index it builds at startup - followed a linked folder straight out and
  reported back what they found there: the names and sizes of files elsewhere
  on your disk, and for source files the names of the functions inside them.
  On Windows a junction was the more likely way in, because it does not look
  like a link to the tools that were meant to skip one. Both now stop at the
  edge of the project, and the tree tells you when it left a link unfollowed
  rather than quietly omitting it.
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
  downloaded scripts piped straight into a shell; a force push or a remote
  branch deletion aimed at master or main, whether written as a flag or as a
  refspec; and "git reset --hard", which throws away uncommitted work. The
  same push rule covers the coder's built-in push action, not only a push it
  types as a shell command. Each refusal says which rule stopped it and what
  to do instead. Ordinary commands are unaffected, including ones that look
  similar such as deleting a build directory or force-pushing your own
  feature branch.
- **Two coder sessions started close together no longer share one activity
  log.** The file behind a session's log was named only by the second it
  started and the process id, so two sessions begun within the same second on
  the same running server landed on the identical file and their command
  history interleaved into it - including a scoped or restricted session's
  history mixing with an unrelated one. Each session now gets its own file,
  and every entry in it is tagged with the session that wrote it.
- **Two coder sessions in the same project, finishing close together in
  `full` mode, could destroy each other's saved transcript.** The
  human-readable session transcript was named only by the second the session
  ended, with nothing else distinguishing one session from another, so a
  second session closing within the same second silently overwrote the
  first one's file, replacing its entire recorded conversation with the
  second session's own. Each session's transcript now gets its own file.
- **Restarting the server in `full` mode could interleave two sessions'
  conversations into one saved transcript.** The chat and server transcript
  files were named only by the second they were created and the process id,
  both unchanged across an in-place restart, so a restart happening within
  the same second as the previous start reused the exact same file and the
  new conversation was appended into what read as the old one's transcript.
  Each transcript now gets its own file.
- **An MCP client asked to pull a model could point localm at any
  HuggingFace repo, with nothing vetting the source.** Setting up an
  embedding model over MCP was already limited to a short known-good list
  or one you had already registered yourself; pulling a chat model had no
  such limit, so a client steered by injected content could cause an
  arbitrary, unvetted GGUF to be downloaded and loaded. Pulling a model
  over MCP is now held to the same standard: a known repo, or one whose
  source matches a model you already have registered.
- **A HuggingFace or CivitAI token could be sent to a different host than the
  one you configured it for, if that host's API ever issued a redirect.**
  Both integrations attach the token as an Authorization header and follow
  redirects themselves (so each hop can be checked against your network
  policy), but the token was re-attached on every hop regardless of where it
  pointed. It is now dropped the moment a redirect crosses to a different
  host, matching what a browser or a standard HTTP client already does by
  default.
- **A crafted search pattern from the model could corrupt memory in the
  coder's `grep` and `search_replace` tools, rather than just run slowly.**
  These tools already bounded how long a model-supplied pattern is allowed
  to run, since the pattern is attacker-influenced input, but two bugs in
  the bundled regex engine sat underneath that bound: one could write past
  the end of an internal buffer while compiling certain patterns, the other
  could read past the end of one while matching and let the stray byte
  decide whether a search reported a match that was never really there.
  Neither needed a slow or memory-hungry pattern to trigger, so the existing
  time and size limits could not catch them. Upgrading the bundled regex
  engine closes both; the limits themselves are unchanged.
- **A release you download instead of `git clone`-ing can now be verified
  before you trust it.** The auto-updater already checked every in-place
  update against a signature, but a first-time download of a release zip
  (a browser download, a mirror, a forwarded copy) had nothing to check it
  against. `scripts/verify_release.py` now verifies a downloaded
  `localm-<version>.zip` against the same signature, and/or a plain SHA256
  digest, with no new install required. The recommended `git clone` install
  path is unchanged - it never downloads a zip, so it keeps relying on
  git+HTTPS+GitHub the same way any other cloned project does.
- Fixed a flaw in the multi-instance GPU coordination handshake: a request
  asking another local instance to free VRAM could be sent to an address that
  was never verified as belonging to that instance, carrying the one-time
  credential that authorizes the request. The request is now sent only to an
  address and connection type this instance has confirmed is the same
  machine.
- Fixed a flaw in cross-install instance discovery: confirming that a listed
  instance was genuinely running could send a network request to an address
  that instance's own unverified registration claimed, rather than to this
  machine. Discovery now always confirms an instance on this machine,
  regardless of what address a registration entry names.

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

[Unreleased]: https://github.com/Matlan1/localm/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Matlan1/localm/releases/tag/v0.3.0
[0.2.0]: https://github.com/Matlan1/localm/releases/tag/v0.2.0
[0.1.5]: https://github.com/Matlan1/localm/releases/tag/v0.1.5
[0.1.4]: https://github.com/Matlan1/localm/releases/tag/v0.1.4
[0.1.3]: https://github.com/Matlan1/localm/releases/tag/v0.1.3
[0.1.2]: https://github.com/Matlan1/localm/releases/tag/v0.1.2
[0.1.1]: https://github.com/Matlan1/localm/releases/tag/v0.1.1
[0.1.0]: https://github.com/Matlan1/localm/releases/tag/v0.1.0
