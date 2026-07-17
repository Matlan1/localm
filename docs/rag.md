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
localm rag query manuals "how do I replace the toner"
localm rag rm manuals
```

Collections live in `<data dir>/rag/<name>/` - plain JSON, no database.
Deleting a collection removes only the index; your files are untouched.
Creating or indexing a collection is an explicit action, so it writes to disk
in every session mode; the privacy contract governs *automatic* traces, not
things you ask for.

### Where localm may index from

Indexing through the GUI or the HTTP API is confined by a folder policy
(Settings): **whitelist** mode (the default) allows only your home folder, the
working directory, and any folders you add to it; **blacklist** mode allows
everywhere except the folders you deny. In both modes the localm data
directory (it holds your API key and registry) and well-known credential
folders (`.ssh`, `.aws`, `.gnupg`, ...) are always refused, wherever they
appear in the path. Picking a folder outside the whitelist offers an
"add this folder and continue" prompt rather than a dead end. The local CLI
(`localm rag add`) is unconfined: the person running it can already read
their own files.

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
  `BM25` per collection.
- Embedding failures **degrade, never break**: indexing falls back to
  lexical-only. Through the GUI, indexing runs as a background job and the
  fallback is noted in that job's log. A headless `localm serve` (no GUI
  attached) runs `/api/rag/collections/{name}/add` and `/upload`
  synchronously with no job to log into, so that note goes to the server log
  instead: it prints when `--debug` / `LOCALM_DEBUG=1` is on, and is always
  captured in the always-on in-memory activity buffer a bug report can
  include, even without `--debug`. A headless caller that needs to confirm a
  document was actually vectored (not just indexed) can compare the
  collection's `has_vectors` stat from `GET /api/rag/collections/{name}`
  before and after. A query that cannot use its vectors falls back to BM25 and
  records the reason on the collection's status (a corrupt or
  dimension-mismatched vector sidecar is also logged at WARNING).

By default CLI indexing is lexical-only (no running engine); pass `--embed` to
`localm rag add` / `localm rag query` to compute vectors via a running localm
server, matching the GUI.

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
- One indexing job per collection at a time is the supported pattern; the
  files are rewritten atomically, so a concurrent query sees a consistent
  snapshot.
- `POST .../add` accepts up to 50 paths per request (a path may itself be a
  folder); `POST .../upload` accepts up to 50 files per request, 30 MB each
  and 100 MB total. Split a larger batch across multiple calls.

## Troubleshooting

- **No embeddings? It still works.** Retrieval degrades to lexical-only (BM25)
  automatically (see above). Run `localm setup-embeddings` to install the
  on-device embedding model (default `bge-small-en-v1.5`), then re-index to get
  vectors blended back in.
- **A query returns nothing.** No chunk matched: broaden or rephrase the query
  (exact words matter in lexical mode), or confirm the collection
  indexed the files (re-index if a source changed).
- **Indexing failed.** Indexing is atomic, so a failed index leaves the previous
  snapshot intact. Check the error in the GUI/CLI and `--debug` log; common causes
  are an unreadable/encrypted file or a missing `[rag]` extra for PDF parsing.
