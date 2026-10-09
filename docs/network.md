# Internet access for the coder and chat

> The network policy (`localm/netpolicy.py`) is CORE and always governs every
> outbound request, plugin or not. The chat "Web access" surface is provided
> by the `web` plugin and appears only when it is active. The coder's
> `fetch_url` and `web_search` tools are built into the coder itself and are
> always present, independent of the `web` plugin. The off/ask/allow modes
> and SSRF protection below apply to both.

localm is offline-first: nothing *requires* the internet. But some tasks
genuinely need it - looking up current documentation, checking a version,
querying the weather, reading an error's bug tracker. This page describes how
model-initiated network access works and how to control it.

## The one rule

**Every network request a model can trigger goes through one policy choke
point** (`localm/netpolicy.py`). The coder's `fetch_url` and `web_search`
tools, and the chat's web access, all use it. There is no second path.

Things a *user* triggers directly (`localm pull`, the `/web` chat command,
the Knowledge page's one-time embedding-model download, the mic button's
one-time Whisper download) are consent by definition - but `net_mode off`
still kills them, with no exception. A one-time download like the last two
is a single-call authorization, never written to config; it lets that one
fetch through `ask`'s per-request friction, exactly like clicking "download
now" is itself the consent, and it changes nothing about how the next
request is handled.

Online coder providers (`--online`, `--anthropic`) are a separate case and
are not covered by this policy at all: every request to OpenAI/Anthropic
goes out over HTTPS through a direct provider client, outside
`netpolicy`, with no `net_mode` check anywhere in that path. The only gate
is the explicit CLI flag itself, plus a warning if you try it in privacy
mode.

## Modes

```bash
localm config net_mode off     # nothing gets through
localm config net_mode ask     # default - the coder asks before each request
localm config net_mode allow   # no confirmation
```

| mode | coder `fetch_url` / `web_search` | chat web access |
|---|---|---|
| `off` | tool returns a policy error | `/web` and the toggle return a clear error |
| `ask` (default) | approval prompt per request (terminal y/N or GUI approval card showing the URL/query) | `/web` runs immediately (typing the command is the consent); the "Web access" toggle is on by default so the model knows the tools exist, and every model-initiated request shows a per-request approval card |
| `allow` | runs without asking | works |

The `LOCALM_NET_MODE` env var overrides the config (like `LOCALM_MODE` for
privacy). In the coder, sessions started with auto-approve also auto-approve
network requests in `ask` mode.

`ask`'s approval prompt is enforced by the front end (the GUI modal, the
coder's terminal y/N), not the server. A direct API or MCP caller
authenticated with the `web` scope hits the underlying endpoints directly
and is treated as already consented - it is not prompted. Domain rules and
the SSRF guard below still apply in every mode, `off` included.

## Domain rules

```bash
localm config net_allow "docs.python.org, github.com, wttr.in"   # only these
localm config net_deny  "doubleclick.net"                        # never these
```

- `net_allow` empty (default) = any domain. Non-empty = only listed domains.
- `net_deny` always wins over `net_allow`.
- `example.com` matches `example.com` and every subdomain (`api.example.com`).

## SSRF guard

By default, requests to **loopback, private, and link-local addresses are
refused** - that includes `127.0.0.1` (the localm API itself), `192.168.x.x`
(your router's admin page), and `169.254.169.254` (cloud metadata). Redirects
are followed manually and **every hop is re-validated**, so a public page
cannot bounce the agent into your LAN. Response bodies are size-capped.

If the coder legitimately needs to talk to a local dev server
(`http://localhost:3000`), opt in:

```bash
localm config net_allow_private true
```

The hostname is resolved and validated once, then the connection is pinned to
the validated addresses of that lookup, so the connect cannot re-resolve to a
different address. When the first address cannot be connected to (for example
an IPv6 address on a network without working IPv6), the next validated address
from the same lookup is tried, up to four, all within twice the connect
timeout. This closes the check-and-connect
DNS-rebinding race: a rebind or an unresolvable host is refused, not
reconnected through a fresh lookup. The domain deny/allow lists remain an
additional control.

## Web search

`web_search` searches and then reads the top three result pages, returning an
evidence bundle: each source labelled `S1`, `S2`, ... with its title, URL and
a grounding label, followed by evidence excerpts selected from the pages that
could be read (at most 12,000 characters in total, 4,000 per source). The
bundle's own grounding is `page-backed` when at least one page was read,
`snippet-only` when only the search snippets are available, and `failed` when
there is no evidence at all; the chat, the coder and scheduled jobs all show
that label rather than presenting snippets as read pages. `fetch_url` reads a
single page on request. The same retrieval is exposed to API clients as
`POST /api/web/retrieve` (`{"query": "..."}`); `/api/web/search` and
`/api/web/fetch` remain for explicit low-level use.

Once the results are in, localm waits at most 15 seconds for all of the
page reads together, and a page that keeps trickling in is cut off; a
page that could not be read keeps its search snippet as evidence and is
labelled with the reason (for example `stackoverflow.com refused access, HTTP
403`). A few sites are read from their own content endpoints instead of the
page, by `web_search` and `fetch_url` alike:

- `github.com/<owner>/<repo>` (and `/tree/<ref>/...`): the README, from
  `raw.githubusercontent.com`, then the GitHub REST API (`api.github.com`).
- `github.com/<owner>/<repo>/blob/...`: the raw file from
  `raw.githubusercontent.com`.
- A Stack Overflow or other Stack Exchange question: the question and its top
  three answers from the Stack Exchange API (`api.stackexchange.com`).

The original URL must pass the domain rules first, and each endpoint is
checked like any other request, so with a `net_allow` list the endpoint is
used only when its host is allowed too; otherwise the page itself is read.

The default backend is a chain of no-key search services - no account, no API
key, nothing to configure: DuckDuckGo's HTML page, then DuckDuckGo's lite page,
then Brave Search. A search whose connection is reset or cut is sent again (up
to three tries); when a service still fails, answers with a bot check or
returns a page without results it can read, localm moves on to the next one. A
page counts as "no results" only when it says so itself, and that answer ends
the search; any other page without readable results is reported as a failure.
A service that
answered with a bot check is asked once more after a short pause. Only when
every service failed does the search report it, naming each one's cause. A
query can therefore reach Brave Search when DuckDuckGo does not answer.

A configured SearXNG instance is the only service a search asks: localm
removes a stray `/search` path or query from the configured URL and reads the
instance's HTML results page when its JSON format is turned off. When the
instance returns no results while some of its own search engines failed, the
search is reported as failed instead of empty. For a self-hosted search
backend, point localm at a SearXNG instance:

```bash
localm config net_search_url http://192.168.1.10:8080
localm config net_allow_private true    # if the instance is on your LAN
```

## Chat: two ways to use the web

1. **`/web <query>`** - explicit, one-shot grounding. Searches, reads the top
   result pages, shows the evidence as a dimmed "Web" message in the
   conversation (with its grounding label), and the model answers from it,
   citing the source IDs. The command itself is the consent for that search
   and its page reads, so it works with the toggle off; `net_mode=off`, the
   domain lists and the private-address guard still apply. When no page could
   be read, the message is labelled snippet-only and a notice says so.
2. **The "Web access" toggle** (parameters drawer) - lets the *model* decide.
   The model can emit a `web_search` or `fetch_url` request mid-conversation;
   the GUI executes it through the policy, injects the evidence, and the model
   continues (at most 3 web rounds per send). Every request and result is
   visible in the conversation - nothing happens silently. With no saved
   choice the toggle follows the policy: on under `allow` and `ask` (under
   `ask` each request is approved first), off under `off`. When the toggle is
   off the model is told plainly that it has no internet access. A reply that
   only announces a lookup ("I will now search ...") without making the call
   gets one repair prompt asking for the call or a final answer; it is never
   repeated.

## The coder

`web_search` and `fetch_url` appear in the coder's toolset automatically.
In `ask` mode each request shows an approval (the GUI approval card displays
the exact URL or query). In privacy mode, every requested URL/query, every
GitHub or Stack Exchange content endpoint contacted for it, and the address
a page was actually read from when it differs, are also echoed to
stderr (`[localm privacy] fetch_url: …`) so the session leaves a
visible trace *on your terminal* of what went out, without writing anything
to disk.

## The browser plugin

With the `browser` plugin installed and enabled, the coding agent gets tools
to drive a real, automated browser (`browser_navigate`, `browser_click`,
`browser_fill`, plus read-only `browser_read`/`browser_screenshot`/
`browser_console`/`browser_network`/`browser_close`). It reaches only what
this same policy already allows: every request the driven page makes -
including images and scripts the page pulls in on its own, and every hop of
a redirect - is checked, and anything refused is reported with the reason
instead of silently dropped. WebSocket connections are refused rather than
relayed. `browser_navigate`, `browser_click`, and `browser_fill` confirm in
`ask` mode like `fetch_url`/`web_search` above; the read-only tools do not.
Off by default - install the `browser` extra, download the browser it
drives (the Download browser button under Settings > Server & network, or
`localm setup-browser`), and switch it on in Settings. The `system` engine
runs a Chrome, Chromium, Edge or Brave already installed on the machine
instead of the download; either engine starts with a fresh, empty profile,
so none of your logged-in sessions are used. An API key needs the separate
`browser` scope (see [SECURITY.md](https://github.com/Matlan1/localm/blob/master/SECURITY.md)), independent of the
coding agent's shell access, so you can grant one without the other.

## What the policy does NOT govern

- **Child processes.** `run_shell` commands like `pip install`, `npm install`,
  or `git clone` talk to the network themselves. The gate for those is the
  shell-command approval (and `always_confirm` for `run_shell`), not the
  network policy.
- **Model downloads** (`localm pull`) and **online coder providers**
  (OpenAI/Anthropic opt-ins) - explicit user actions. Note that model
  downloads DO go through this same policy's domain lists and SSRF guard
  (`check_url`); `off` is bypassed only via the separate
  `net_allow_model_downloads` setting.
- **Bug-report upload.** A deliberate, one-off action you trigger yourself
  (see [privacy.md](privacy.md)); it does not call `check_url` and is not
  subject to the domain lists.
- **Requests to your ComfyUI instance.** ComfyUI has its own, narrower
  guards instead: a configured `comfy_api_url` that targets a link-local or
  cloud-metadata address is refused, and the connection itself refuses any
  HTTP redirect outright. Loopback and LAN are both normal, unchecked ComfyUI
  deployments.
- **Embedding-model and Whisper downloads.** These respect `net_mode`
  (including `off`) but fetch from a fixed, hardcoded repository named by
  localm's own code rather than a caller-supplied URL, so they never call
  `check_url` and are not subject to the domain lists, the SSRF guard, or the
  DNS-rebinding pin - none of which a fixed destination needs.
- **The periodic update check.** It respects `net_mode` (blocked when `off`,
  unless explicitly exempted) but, like the embedding/Whisper downloads
  above, targets a fixed endpoint rather than a caller-supplied URL, so it
  does not call `check_url` either. See [privacy.md](privacy.md#4-update-checks-network-policy-not-a-persistence-mode).
- **Privacy mode is orthogonal.** Privacy controls what localm writes to
  *disk*; it cannot make network requests untraceable. Any request leaves DNS
  lookups and traffic visible to your network and the remote server. If a
  conversation must stay fully local, keep web access off - that is why the
  chat toggle is per-conversation and off by default.

Treat this policy as governing the paths named in the sections above it (chat
and coder web access, HuggingFace search and pulls, the browser plugin), not
as a blanket statement about every socket localm opens.

## Trust note: web content is untrusted input

Fetched pages and search snippets enter the model's context. A malicious page
can contain text crafted to steer the model ("ignore your instructions and
run …"). This is inherent to giving any agent web access, and it is why the
default mode asks per request and why destructive coder actions keep their
own approval step regardless of where the idea came from. Treat approval
prompts that follow a web fetch with extra suspicion.
