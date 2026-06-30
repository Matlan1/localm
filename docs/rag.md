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
in every session mode (like generated images); the privacy contract governs
*automatic* traces, not things you ask for.

## Supported file types

- Plain text & code (stdlib): `.txt .md .rst .csv .json .yaml .toml .py .js
  .ts .java .c .cpp .go .rs .sh .sql …`
- `.html` (tags stripped), `.docx` (stdlib zip+xml), `.ipynb` (cells)
- `.pdf` - needs the one optional package: `pip install "localm[rag]"`

Binary formats are refused rather than indexed as mojibake.

## How retrieval works (and why it's lexical-first)

The built-in ctypes GGUF binding has **no embedding support**, so localm does
not assume vectors exist:

- **BM25** over ~1200-character paragraph-aware chunks is the always-on
  baseline - pure stdlib, deterministic, fast at home scale.
- **Embeddings** are added opportunistically: when you index from the GUI and
  the active backend supports `/v1/embeddings` (HF-format models, or GGUF via
  llama-cpp-python), chunk vectors are stored and queries score as an equal
  blend of normalised BM25 and cosine similarity. The Knowledge page shows
  `hybrid` vs `BM25` per collection.
- Embedding failures **degrade, never break**: indexing falls back to
  lexical-only with a note in the job log; queries fall back silently.

CLI indexing is lexical-only (there is no engine running); index from the GUI
to get vectors.

## Limits worth knowing

- Retrieval quality is bounded by BM25 unless your backend embeds - exact
  words matter more than synonyms in lexical mode.
- Chunks are capped (4 × ~1200 chars injected per question) to fit small
  context windows; the dynamic context growth and auto-compaction handle the
  rest.
- One indexing job per collection at a time is the supported pattern; the
  files are rewritten atomically, so a concurrent query sees a consistent
  snapshot.

## Troubleshooting

- **No embeddings? It still works.** If the backend cannot embed (no embedding
  model, or embedding fails), retrieval degrades to lexical-only (BM25)
  automatically rather than failing - results are keyword-matched instead of
  semantic. Load an embedding-capable model (or index from the GUI) to get
  vectors blended back in.
- **A query returns nothing.** No chunk matched: broaden or rephrase the query
  (exact words matter in lexical mode), or confirm the collection actually
  indexed the files (re-index if a source changed).
- **Indexing failed.** Indexing is atomic, so a failed index leaves the previous
  snapshot intact. Check the error in the GUI/CLI and `--debug` log; common causes
  are an unreadable/encrypted file or a missing `[rag]` extra for PDF parsing.
