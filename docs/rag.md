# Knowledge: chat with your documents

> Knowledge is provided by the `rag` plugin. The GUI Knowledge page and the
> `/api/rag` routes appear only when it is enabled; the `localm rag` CLI below
> remains available for indexing. PDF parsing still needs the `[rag]` pip extra.

localm grounds chat replies in your own files - manuals, notes, code, papers -
fully offline. Two ways in:

1. **Attach a document to a chat** (one-off): click the paperclip, pick a
   PDF/docx/txt/md/code file. It is converted to text **in memory on the
   server and never written to disk**, so this works trace-free in privacy
   mode. The text appears as a dimmed "Doc" message and the model reads it
   before your question.
2. **Knowledge collections** (persistent): index folders or files once, then
   select the collection in the chat parameters drawer. Every question
   retrieves the most relevant excerpts, injects them with `[1]`-style
   citations (file + line), and the model answers from them.

## Collections

From the GUI: **Knowledge** page → create a collection → *add docs* with a
file or folder path (folders are indexed recursively). Indexing streams its
progress; re-adding a path skips unchanged files and re-indexes changed ones.

From the terminal:

```bash
localm rag add manuals D:\docs\printer-manual.pdf
localm rag add project D:\projects\myapp        # folder, recursive
localm rag list
localm rag docs manuals                         # list its documents: path, chunks, status
localm rag query manuals "how do I replace the toner"
localm rag rm-doc manuals D:\docs\printer-manual.pdf   # drop one document (file kept)
localm rag rm manuals                           # delete the whole collection (files kept)
```

`rag rm-doc` matches PATH against the index's stored key exactly, so copy it
from `rag docs`'s own output rather than retyping it - a path that is
otherwise equivalent but spelled differently (forward vs backward slashes, a
different but equal-looking absolute form) is reported as not in the
collection.

`rag docs` flags a document `(missing)` when its source file is no longer on
disk (see [Keeping an index current](#keeping-an-index-current)), and
`(uploaded)` when it was added via the GUI/API upload path rather than a file
path - `rag repair`/`rag resync` cannot re-read an uploaded document from disk,
so only `rag rm-doc` can remove it.

From the coder: two built-in tools let the agent search your Knowledge
collections mid-task - one lists what is available, the other searches a
named collection for matching excerpts. Search asks for confirmation by
default (collection content is not scoped to the project the way file reads
are), and retrieved text goes through the same sanitiser used for any other
untrusted content before it reaches the model. Search is lexical only for
now, not the hybrid/embedding search the Knowledge page itself can do. A
restricted (shared, non-owner) coder session cannot use either tool.

Collections live in `<data dir>/rag/<name>/` - plain JSON, no database.
Deleting a collection removes only the index; your files are untouched.
Creating or indexing a collection is an explicit action, so it writes to disk
in every session mode; the privacy contract governs *automatic* traces, not
things you ask for.

### Where localm may index from

Indexing through the GUI or the HTTP API is confined by a folder policy
(Settings): **whitelist** mode (the default) allows only your home folder, the
working directory, and any folders you add to it; **blacklist** mode allows
everywhere except the folders you deny. In both modes well-known credential
folders (`.ssh`, `.aws`, `.gnupg`, ...) are always refused, wherever they
appear in the path. A Windows drive mapped to a network share (`net use Z:
\\host\share`) is treated as an ordinary local folder unless you turn off
**Settings > Security > "Allow network drives as filesystem locations"**.
The localm data directory (it holds your API key and
registry) is NOT specially excluded from that confinement - if it falls within
an allowed folder, its contents are indexable like any other file, since the
owner already has direct filesystem access to their own data; this is a
deliberate choice for a local, single-user tool, not an oversight. Picking a
folder outside the whitelist offers an "add this folder and continue" prompt
rather than a dead end. The local CLI (`localm rag add`) is unconfined: the
person running it can already read their own files.

### Per-key folder scoping

A named API key's **indexing** reach can be confined further, to a specific
set of folders it may index at all - narrower than the global policy above,
not wider than it. Mint one with:

```bash
localm key create dashboard --scope rag --rag-root D:\docs\manuals --rag-root D:\docs\public
```

Any `--rag-root` grant REPLACES the key's indexing reach with exactly those
folders, instead of falling back to your home dir, working dir, and the
configured allowed roots. Omit `--rag-root` to leave indexing unrestricted
(it still obeys the global folder policy above). **This confines indexing
only.** Querying and listing collections are not scoped by `rag_roots` at
all: a key with the `rag` scope can query or list any existing collection on
the server, including one built from folders outside its granted roots.
Only the owner key may set `rag_roots` on another key (via `key create` or
`POST /v1/keys`); the owner's own reach is never confined by it. See
[cli.md](cli.md#api-keys).

## Supported file types

- Plain text & code (stdlib): `.txt .md .rst .csv .json .yaml .toml .py .js
  .ts .java .c .cpp .go .rs .sh .sql …`
- `.html` (tags stripped), `.docx` (stdlib zip+xml), `.ipynb` (cells)
- `.pdf` - needs the one optional package: `pip install "localm[rag]"`
- **Archives** `.zip` / `.tar` (and `.gz .bz2 .xz .tgz .tbz .txz`) are unpacked in
  memory and each text member is indexed.
- **Images** `.png .jpg .jpeg .webp .gif` are indexed by their description: localm
  asks the active model to describe the image, so a vision-capable model (or a
  chat model with an mmproj projector) must be loaded. Without one, the image is
  skipped with a message telling you to load a vision model.
- A file with an unfamiliar extension is content-sniffed: if its bytes decode as
  text it is indexed as text, otherwise a genuinely binary file is refused rather
  than indexed as mojibake.

### Document format labels

Every chunk is tagged with the format of its source document (`json`, `yaml`,
`python`, `markdown`, `text`, `image`, `archive`, ...). The label is derived
heuristic-first and free: a known file extension is authoritative, then a
structural sniff of the content. Only when both are inconclusive (an odd extension
whose shape is unclear) does localm consult the loaded chat model as a one-off
tie-break, cached per extension so it fires at most once per indexing run, and
never at all when no chat model is loaded. So an embedding-only index never stalls
on a chat call. Turn the tie-break off entirely with `rag_classify_unknown_files
false` (it is on by default); unclassifiable chunks then fall back to the `text`
label.

## How retrieval works (and why it's lexical-first)

Semantic vectors come from a small dedicated embedding model, never from the chat
model. A chat model is a decoder LLM trained to predict the next token, not to
place related texts near each other, so pooling its hidden states yields vectors
that look fine (non-zero, normalised) but barely separate related from unrelated
text: measured on this codebase, Qwen2.5-0.5B's highest cosine among genuinely
*unrelated* pairs (0.7523) came out higher than its lowest cosine among
genuinely *related* pairs (0.7518), so no threshold separates the two, while
`bge-small` leaves a comfortable 0.29 margin. The bundled
GGUF runtime cannot embed a chat model at all; a HuggingFace-format chat model
technically can, and used to, which is why localm now routes it to the dedicated
embedder too rather than quietly returning those vectors.

- **BM25** over ~1200-character paragraph-aware chunks is the always-on
  baseline - pure stdlib, deterministic, fast at home scale.
- **Embeddings** use a small dedicated on-device embedding model (default
  `bge-small-en-v1.5`). Install it with
  `localm setup-embeddings`. When it is present and you index with vectors
  enabled, chunk vectors are stored and queries score as an equal blend of
  normalised BM25 and cosine similarity. The Knowledge page shows `hybrid` vs
  `BM25` per collection. It otherwise loads on demand, on whichever request
  needs it first; Settings has a "Warm up now" button that loads it up front
  instead, with a live status line through resolving, downloading if needed,
  freeing VRAM, and loading (already-loaded shows "Already warm"). Changing
  which model is used (Knowledge page, `PATCH /v1/config`, or
  `localm setup-embeddings --model NAME`) requires an owner (admin-scoped) key;
  a scoped `rag` key can index and query but not switch models. Every one of
  those writers reports which existing collections have vectors and what they
  were built with BEFORE the switch happens (the exact dimension impact is
  only known once the new model is actually loaded); `setup-embeddings` also
  asks you to confirm when it would change the model, and `-y`/`--yes` skips
  that confirmation.
- Embedding failures **degrade, never break**: indexing falls back to
  lexical-only. Indexing (`add`/`upload`) and embedding setup always run as a
  background job, on `localm gui` and a headless `localm serve` alike:
  `POST /api/rag/collections/{name}/add` and `.../upload` return
  `{"job_id": ...}` immediately, and any degrade-to-lexical fallback is noted
  in that job's log - follow it with `GET /api/jobs/{id}/events`, the same way
  the GUI does. A headless caller that needs to confirm a document was
  actually vectored (not just indexed) can compare the collection's
  `has_vectors` stat from `GET /api/rag/collections/{name}` before and after,
  or just read the job's own outcome. A query that cannot use its vectors
  falls back to BM25 and records the reason on the collection's status (a
  corrupt or dimension-mismatched vector sidecar is also logged at WARNING).

By default CLI indexing is lexical-only (no running engine); pass `--embed` to
`localm rag add` / `query` / `resync` / `repair` to compute vectors via a
running localm server (`--url` to point at a specific one), matching the GUI.
`localm rag reembed` always needs an embedder - there is no lexical form of it.

## Keeping an index current

Indexing a folder records the folder itself, not just the files it held at the
time, so the index can be brought back in line with the disk later:

```bash
localm rag resync NAME                     # by hand
localm job add sync-NAME --rag --collection NAME --cron "0 3 * * *"   # scheduled
```

A re-sync re-walks each indexed folder and re-indexes incrementally: new files
are added, changed files re-indexed, unchanged files skipped by content hash.
There is no filesystem watcher and there will not be one - a watcher daemon
would break the self-contained design - so a schedule is how an index stays
fresh.

A document whose source file has vanished is **flagged, not deleted**
(`missing` in the collection's document list, counted as `n_missing` in its
stats): its chunks stay searchable and the flag clears by itself if the file
returns, so a moved file or an unplugged drive cannot silently destroy part of
an index. Pass `--prune-missing` to actually remove those entries. A folder that
is not reachable at run time is reported and skipped whole - nothing under it is
indexed, flagged, or pruned. Scheduled re-syncs run under the same indexing
policy as an interactive add. Full details in
[docs/jobs.md](jobs.md#keeping-an-indexed-folder-current).

## Limits worth knowing

- Retrieval quality is bounded by BM25 unless an embedding model is installed
  (`localm setup-embeddings`) - exact words matter more than synonyms in lexical mode.
- Chunks are capped (4 × ~1200 chars injected per question) to fit small
  context windows; the dynamic context growth and auto-compaction handle the
  rest.
- Each query loads the collection from disk and scores every chunk (there is no
  query-time index cache), so retrieval is brute force by design: fast at home
  scale (thousands of chunks), but query latency grows with collection size and
  it is not built for millions of chunks.
- **One writer per collection at a time, enforced.** Writes to a collection are
  serialised both inside a localm process and BETWEEN processes, so the server's
  API adds, its scheduled re-syncs and a hand-run `localm rag add|resync|repair|rm`
  cannot lose each other's changes. Every file is rewritten atomically too, so a
  concurrent query always sees a consistent snapshot, and queries never wait for a
  writer at all. The two cases wait differently, on purpose. Two writers *inside
  one localm process* (say two indexing jobs started from the Knowledge page)
  simply queue: the one holding the collection is getting on with it, so waiting
  always ends and the work still happens. A writer in *another process* cannot be
  trusted that way, since it may be hung or already gone, so it waits a bounded 30
  seconds, says so while it waits, and then REFUSES with a message naming the
  process that holds the collection rather than interleaving with it. A refused
  command has changed nothing, and the API answers 409. There is no wall-clock
  limit on how long a collection may be HELD, so an hours-long index of a big
  folder is safe: the holder keeps reporting in, and only a holder that stops
  reporting (a crash, a killed process) has its lock reclaimed. Collections are
  independent, so work on a different one never waits.
- `POST .../add` accepts up to 50 paths per request (a path may itself be a
  folder); `POST .../upload` accepts up to 50 files per request, 30 MB each
  and 100 MB total. Split a larger batch across multiple calls.

## Troubleshooting

- **No embeddings? It still works.** Retrieval degrades to lexical-only (BM25)
  automatically (see above). Run `localm setup-embeddings` to install the
  on-device embedding model (default `bge-small-en-v1.5`), then
  `localm rag reembed NAME` to add vectors to what is already indexed, without
  re-reading the source files.
- **A query returns nothing.** No chunk matched: broaden or rephrase the query
  (exact words matter in lexical mode), or confirm the collection
  indexed the files (re-index if a source changed).
- **`rag repair` refuses a collection built entirely from uploads.** An
  uploaded document has no server-side source file to re-read - the bytes were
  never retained - so `repair` cannot rebuild it and says so rather than
  reporting a false "0 re-indexed" success. A mixed collection repairs
  whatever has a real source and names the uploaded documents it left alone;
  re-upload those, or remove them with `rag rm-doc`.
- **Indexing failed.** Indexing is atomic, so a failed index leaves the previous
  snapshot intact. Check the error in the GUI/CLI and `--debug` log; common causes
  are an unreadable/encrypted file or a missing `[rag]` extra for PDF parsing.
- **"Collection X is being written by another localm process".** Something else
  is indexing that collection right now, most often a scheduled re-sync or an add
  from the Knowledge page, and the command you ran waited for it and then stood
  down without changing anything. The message names the process holding it and
  how long it has been running. Let that run finish and repeat the command; a
  scheduled job that hits this says so in its output and picks the folder up on
  its next run. If the holder is gone (the machine lost power mid-index, say), the
  lock is reclaimed by itself about a minute after that process stopped reporting,
  so a crash cannot wedge a collection. The one case that needs you is a lock file
  localm cannot even read the timestamp of, which means something is wrong with
  the file itself (permissions, a damaged filesystem): the message then names the
  file, and deleting it releases the collection once you are sure no localm
  process is using it. Two environment variables tune this for unusual setups:
  `LOCALM_RAG_LOCK_WAIT` (seconds to wait before refusing, default 30) and
  `LOCALM_RAG_LOCK_STALE` (seconds without a heartbeat before a holder counts as
  crashed, default 60; raise it on a very slow or heavily contended disk).
- **"Semantic search is degraded".** localm checked the stored vector index
  against the chunks and refused to use it (unreadable, malformed, or no longer
  lining up), and answered lexically instead. Nothing is deleted to make that go
  away: the vector file is kept as it is, or moved aside to
  `vectors.json.rejected` in the collection folder if the chunks were rewritten
  in the meantime, and the reason is repeated by the Knowledge page, `rag resync`
  and every scheduled run until you clear it with `localm rag reembed NAME`
  (recomputes every chunk's vector from what is already indexed, no source
  files needed) or `localm rag repair NAME --embed` (re-reads from source
  too, needed if the chunks themselves are also suspect). Only a pass that
  covers **every** chunk clears it. Indexing one more document meanwhile does
  not, even with embeddings on: that leaves the older chunks without vectors,
  which is a thinner index than you had, not a repaired one. Each incident keeps its own set-aside copy
  (`vectors.json.rejected`, then `.rejected.2`, and so on), and they stay after a
  successful rebuild as a record of what happened - delete them yourself if you
  want the space back. The one time localm removes them is when the collection
  has no documents left at all, since stored vectors are positional and there is
  then nothing they could ever be matched back up with.
