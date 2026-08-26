# Durable memory: remember facts across chats

> Memory is provided by the `memory` plugin. It is OPT-IN and off by default:
> install and enable it first (see below). While it is off, chat runs normally
> with no recall and no memory growth.

localm can keep a small store of durable facts about you (your name, role,
projects, tools, stable preferences) and quietly recall the relevant ones in
later chats, so you do not repeat yourself every session. Everything stays on
your machine: a plain JSONL store, no cloud, no database.

Memory has two halves:

- **Recall** injects the most relevant remembered facts into the system prompt
  on each chat turn, server-side.
- **Consolidation** distils new durable facts out of your recent sessions and
  folds them into the store, unattended, so memory grows on its own.

## Turn it on

```bash
localm plugin install memory     # installs and enables the memory plugin
```

Or install it from the GUI **Plugins** page. The memory plugin has no extra
Python dependency to install. It adds no tab of its own: it contributes the
`/api/memory*` routes plus the chat recall and consolidation hooks, and its
settings render under **Settings > Privacy & data**, not on the Plugins page.

Memory only *grows* in `log` or `full` session mode. The default mode is
`privacy`, which writes nothing and, by default, recalls nothing (see
[Privacy](#privacy-what-runs-in-each-mode) below). To actually accumulate a
memory, set the mode to `log` or `full` (Settings > Privacy in the app, `--mode
log` on `localm gui` / `serve`, or the `LOCALM_MODE` environment variable). See
[privacy.md](privacy.md) for the full mode reference.

Semantic (meaning-based) recall also needs a small on-device embedding model:

```bash
localm setup-embeddings          # installs bge-small-en-v1.5 for semantic recall
```

Without it, recall still works but is lexical only (exact-word BM25). See
[How recall ranks](#how-recall-ranks-facts).

## From the terminal

`localm memory` manages the same store the app does - it was GUI-only until now,
which mattered because `localm job add --memory` can schedule the consolidation
that produces corrections you then had no way to review. The command reads and
writes the on-disk store directly, so it works even when the memory plugin is
not installed or enabled - installing the plugin only turns on the automatic
chat-turn recall and consolidation hooks described below.

```bash
localm memory list [--all] [--json]                  # what localm has remembered about you
localm memory show ID                                # one fact in full, with its provenance
localm memory add "..." [--kind K] [--importance F]  # save a fact yourself
localm memory forget ID [-y]                         # delete one fact (not recoverable)
localm memory forgotten                              # facts localm archived on its own
localm memory restore ID                             # bring an archived fact back
localm memory corrections [--json]                   # proposals waiting for your review
localm memory accept ID | reject ID                  # resolve one proposal
localm memory clear [-y]                             # erase everything, archive included
```

`memory add` produces the same kind of record `POST /api/memory/append` does:
`--kind` is `semantic` (a standing fact, the default) or `episodic` (something
that happened), and `--importance` (default 0.8) is the weight recall and
prune use. Source is always `user` - there is no `--source` flag - and an
unrecognised `--kind` is silently coerced to `semantic` by the record's own
validation.

`forget` and `forgotten` are not two halves of one thing. `forget` deletes
outright; the archive `forgotten` lists is filled only when localm drops a fact
itself - evicting one at the record cap, or replacing one you accepted a
correction for - and `restore` can only reach those.

`clear` erases everything: live facts, the forgotten archive, and any pending
or dismissed corrections, so no fact text survives it anywhere on disk. It
reads its own result back afterward and refuses to report success if anything
is still there.

## What it stores

Each record is a short, distilled memory, never a raw transcript:

- **Semantic** facts and preferences ("Prefers TypeScript", "Works on a printer
  firmware project").
- **Episodic** one-line summaries of past sessions ("Debugged a flaky upload
  test"), so chat can recall what you worked on, not only standing facts.

A record also carries a source that governs how much it is trusted:

- **user** - you typed it (via `/remember` or the memory editor). Trusted; can
  reach full importance.
- **synth** - the model distilled it from a session. Importance-capped, and a
  synth fact may never overwrite or delete a user-typed one without your
  approval (see [How memory grows](#how-memory-grows-consolidation) below).
- **import** - migrated from the legacy flat `chat-memory.md` file (imported
  once, in `log`/`full` mode).

The store lives under `<data dir>/memory/chat/` as a JSONL file, with an
optional aligned vector sidecar when embeddings are present. It is owner-scoped:
localm is single-user, so all chat turns share one memory namespace.

## Managing memory

In the GUI:

- `/remember <fact>` adds a fact to the store.
- `/memory` opens the memory manager: view and edit your facts (one per line;
  Save replaces the list), click **Synthesize now** to distil facts from your
  recent chats immediately, and review any **suggested corrections** (see
  below) - nothing about them changes until you accept or reject each one.
- The 🧠 toggle in the parameters drawer turns recall on and off (the
  `memory_enabled` setting).

Over the HTTP API (the routes the memory plugin mounts; see
[server-api.md](server-api.md)):

| Route | Purpose |
|---|---|
| `GET /api/memory` | The current facts (text + per-item metadata) and whether writes are allowed. |
| `PUT /api/memory` | Bulk edit. A line matching an existing fact keeps that record; new lines are added; omitted lines are deleted. |
| `POST /api/memory/append` | Add one fact. |
| `PATCH /api/memory/{id}` | Edit one record's text or importance. |
| `DELETE /api/memory/{id}` | Delete one record. |
| `POST /api/memory/consolidate` | Distil durable facts from recent sessions now (needs a loaded model). |
| `GET /api/memory/forgotten` | List archived (evicted or superseded) records. |
| `POST /api/memory/forgotten/{id}/restore` | Recover one archived record back into the store. |
| `POST /api/memory/corrections/{id}/accept` | Apply a suggested correction (see below). |
| `POST /api/memory/corrections/{id}/reject` | Dismiss a suggested correction; keep the fact as-is. |

Writes are refused with `403` in privacy mode, and with `409` when another
localm process is mid-write on the same store (nothing is changed - let it
finish and retry; the terminal commands report the same thing in words). There is no memory search,
export, or import yet, and the per-item `PATCH`/`DELETE` routes have no GUI or
CLI surface: bulk editing goes through the `/memory` textarea, and moving a
store between machines means copying the data directory.

## How memory grows (consolidation)

Consolidation reads your recent session logs and folds durable facts into the
store with an ADD / UPDATE / DELETE / NO_OP loop, plus one episodic summary per
new session. It runs OUT OF BAND, never in the chat hot path, and needs a loaded
model. There are three ways it fires:

1. **Automatically after a chat turn** (the default). A debounced background pass
   runs at most once every 15 minutes (override with the
   `LOCALM_MEMORY_AUTO_INTERVAL` environment variable, in seconds). Controlled by
   the `memory_auto_consolidate` setting - and, since this pass fires from the
   same chat-turn hook as recall, it also stays off while `memory_enabled` (the
   🧠 toggle) is off, even with `memory_auto_consolidate` left on. This is what
   makes memory grow with no manual step: chat normally, and durable facts
   accumulate on their own.
2. **The "Synthesize now" button** in the `/memory` manager, or
   `POST /api/memory/consolidate`, for immediate results.
3. **A scheduled memory job** (needs the `jobs` plugin):

   ```bash
   localm job add distil-memory --memory --every 21600   # every 6 hours
   ```

   See [jobs.md](jobs.md).

Consolidation asks the model for a small JSON object and parses it best-effort
(small local models are not airtight, so a chatty or fenced reply is still
recovered). It is guarded against the failure modes a small local model brings:
session text is treated as DATA, never as instructions to follow; a synth
candidate can never *silently* rewrite or delete a user-typed or imported fact;
an unsure UPDATE keeps both facts rather than risk losing a true one;
near-duplicates collapse without a model call. Episodic capture is watermarked,
so a session is summarised exactly once and re-running consolidation adds
nothing new.

When a high-confidence candidate would update or delete a user-typed/imported
fact, consolidation does not apply it - it stores a **pending correction**
instead and leaves the trusted fact untouched. These surface in the `/memory`
manager as "Suggested corrections", each with the old text struck through and
the proposed change beside it; you **Apply** or **Keep as is** per suggestion,
and accepting one archives the old value to the recoverable forgotten sidecar
before applying the change. A rejected suggestion is remembered so the same
contradiction is not re-proposed on the next pass.

In privacy mode, consolidation returns "skipped" and never calls the model.

## How recall ranks facts

On each chat turn the memory plugin builds a recall query from your most recent
message plus a short window of prior user turns (so an anaphoric follow-up like
"yes, do that" still carries the earlier topic), ranks the store, and injects the
top matches into the system message. Recalled facts are neutralised and wrapped
in a fenced, labelled block marked as data, not instructions, so a memory that
reads like a command is treated as context.

- Ranking blends **relevance** (0.5), **recency** (0.3), and **importance**
  (0.2). Up to 6 facts are injected, within a bounded block.
- Recall is **relevance-gated**: a fact is injected only when it actually relates
  to the query (it shares a content word, or, with embeddings, its similarity
  clears an absolute floor). An off-topic turn injects nothing rather than
  padding the prompt with the top few facts regardless.
- **Semantic vs lexical.** With `localm setup-embeddings` installed, recall uses
  meaning-based cosine similarity (a paraphrase still matches); without it,
  recall falls back to exact-word BM25. Facts stored before you installed the
  embedding model are re-embedded in the background, so semantic recall turns on
  retroactively.

### Seeing what recall did

When memory injects facts, the server reports it so you are never guessing:

- The GUI shows a **"used N memories"** chip on the reply; click it to open the
  memory manager.
- `POST /v1/chat/completions` carries an `X-Localm-Memory` response header with
  the count, the injected items, and a **degrade reason** when the semantic
  signal could not be used: `no_embedder` (no embedding model resolved -
  you have not run `localm setup-embeddings` yet), `no_vectors` (records not
  embedded yet), `low_coverage` (too few records carry a vector), `dim_mismatch`
  (the vector sidecar mixes dimensions, usually from switching embedding
  models), or `query_embed_failed` (the embedder itself failed on this turn's
  query). Any of these means recall fell back to lexical BM25 for that turn.

## Forgetting and bounds

A namespace holds at most 256 records, each capped at 500 characters. Low-value
synthesized memories decay over time and are pruned. User-typed and imported
facts are never dropped by decay; only the size cap can evict them, weakest
first, and every eviction is archived to a recoverable `.forgotten.jsonl`
sidecar next to the store, never silently hard-deleted. `GET
/api/memory/forgotten` lists what is archived and `POST
/api/memory/forgotten/{id}/restore` recovers one record back into the live
store.

## Privacy: what runs in each mode

Memory treats privacy mode more strictly than the rest of the app. The default
mode is `privacy`, and in it memory is fully inert: no facts are written, and
none are recalled, so nothing from a past session reaches the model.

| Mode | Recall | Growth |
|---|---|---|
| **privacy** (default) | Off, unless you opt into read-only recall | Off (no writes) |
| **log** / **full** | On (while `memory_enabled` and the 🧠 toggle are on) | On (while `memory_auto_consolidate` is on; the automatic pass also needs `memory_enabled` - manual **Synthesize now** and a scheduled memory job are unaffected by it) |

You can opt into recalling your existing memories during privacy-mode chats
without ever writing new ones: turn on **Allow memory recall in privacy mode**
(`memory_recall_in_privacy`, with the per-surface `..._chat` switch). Even then,
it only READS: no reinforcement, no consolidation, no migration.

Because the default is privacy, memory does nothing out of the box until you
switch to `log` or `full`. This is deliberate (offline-first, no traces by
default), not a bug.

## The coder's own memory

The coder agent keeps a separate store of past-session lessons (what worked, what
failed) per project, recalled at the start of a coder session. It follows the
same privacy contract (lessons are only written in `log`/`full` mode) and is
distinct from chat memory described here. See [cli.md](cli.md) for the coder.

## See also

- [privacy.md](privacy.md) - session modes and the no-traces contract.
- [gui.md](gui.md) - the chat GUI, slash commands, and the memory manager.
- [jobs.md](jobs.md) - scheduling a memory-synthesis job.
- [server-api.md](server-api.md) - the `/api/memory*` routes and the
  `X-Localm-Memory` header.
- [rag.md](rag.md) - Knowledge collections, the other way to ground replies in
  your own content.
