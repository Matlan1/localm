// SPDX-License-Identifier: AGPL-3.0-or-later
// Doubles for the web retrieval route shared by the GUI web tests.

/** The /api/web/retrieve response (web/plug.py) for *results*, each
 *  {title, url, snippet, page?}: a row with `page` is a page-backed source
 *  whose evidence chunk is that text; a row without one is snippet-only.
 *  prompt_text mirrors EvidenceBundle.to_prompt_text(); the server has
 *  already defanged it and declares it untrusted. */
export function bundleOf(query, results, extra = {}) {
  const sources = results.map((r, i) => ({
    id: `S${i + 1}`, url: r.url, canonical_url: r.url, title: r.title,
    snippet: r.snippet || "", provider_rank: i + 1,
    retrieval_status: r.page ? "fetched" : "failed",
    grounding: r.page ? "page-backed" : (r.snippet ? "snippet-only" : "failed"),
    final_url: r.url, error: r.page ? null : "RuntimeError: HTTP 404",
    region: r.page ? "main" : null, text_chars: r.page ? r.page.length : 0,
    untrusted_fields: r.page ? ["title", "snippet"] : ["title", "snippet", "error"],
  }));
  const chunks = results.filter((r) => r.page || r.snippet).map((r, i) => ({
    source_id: `S${i + 1}`, text: r.page || r.snippet, score: 1, offset: 0,
    kind: r.page ? "page" : "snippet", untrusted_fields: ["text"],
  }));
  const read = sources.filter((x) => x.grounding === "page-backed").length;
  const grounding = read ? "page-backed" : (chunks.length ? "snippet-only" : "failed");
  const summary = grounding === "page-backed"
    ? `page-backed: ${read} of ${sources.length} sources read`
    : grounding === "snippet-only"
      ? `snippet-only: no page was read, ${chunks.length} search snippet${chunks.length === 1 ? "" : "s"} only`
      : "failed: no evidence, no page was read";
  const lines = [`Grounding: ${summary}`];
  if (sources.length) {
    lines.push("Sources:");
    for (const x of sources) {
      lines.push(`[${x.id}] ${x.title || "(untitled)"} - ${x.url} (${x.grounding}` +
                 (x.error ? `, ${x.error}` : "") + ")");
    }
  }
  if (chunks.length) {
    lines.push("", "Evidence:");
    for (const c of chunks) lines.push(`[${c.source_id}${c.kind === "page" ? "" : " snippet"}] ${c.text}`);
  }
  return {
    query, provider: "stub", search_status: "ok", search_error: null,
    grounding, grounding_summary: summary, budget_chars: 12000,
    per_source_cap_chars: 4000, total_chars: chunks.reduce((n, c) => n + c.text.length, 0),
    sources, chunks, prompt_text: lines.join("\n"), untrusted_fields: ["prompt_text"],
    ...extra,
  };
}
