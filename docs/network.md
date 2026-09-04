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
goes straight out over plain HTTP, with no `net_mode` check anywhere in
that path. The only gate is the explicit CLI flag itself, plus a warning
if you try it in privacy mode.

## Modes

```bash
localm config net_mode off     # nothing gets through
localm config net_mode ask     # default - the coder asks before each request
localm config net_mode allow   # no confirmation
```

| mode | coder `fetch_url` / `web_search` | chat web access |
|---|---|---|
| `off` | tool returns a policy error | `/web` and the toggle return a clear error |
| `ask` (default) | approval prompt per request (terminal y/N or GUI approval card showing the URL/query) | `/web` runs immediately (typing the command is the consent); the toggle still shows a per-request approval card for each model-initiated request |
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
that exact IP, so the connect cannot re-resolve to a different address. This
closes the check-and-connect DNS-rebinding race: a rebind or an unresolvable
host is refused, not reconnected through a fresh lookup. The domain deny/allow
lists remain an additional control.

## Web search

`web_search` returns titles, URLs, and snippets; `fetch_url` reads a page.
The default backend is DuckDuckGo's no-key HTML endpoint - no account, no API
key, nothing to configure. It can rate-limit or change markup; for a sturdier
self-hosted option, point localm at a SearXNG instance (JSON API enabled):

```bash
localm config net_search_url http://192.168.1.10:8080
localm config net_allow_private true    # if the instance is on your LAN
```

## Chat: two ways to use the web

1. **`/web <query>`** - explicit, one-shot grounding. Searches, shows the
   results as a dimmed "Web" message in the conversation, and the model
   answers from them, naming sources.
2. **The "Web access" toggle** (parameters drawer) - lets the *model* decide.
   The model can emit a `web_search` or `fetch_url` request mid-conversation;
   the GUI executes it through the policy, injects the results, and the model
   continues (at most 3 web rounds per send). Every request and result is
   visible in the conversation - nothing happens silently.

## The coder

`web_search` and `fetch_url` appear in the coder's toolset automatically.
In `ask` mode each request shows an approval (the GUI approval card displays
the exact URL or query). In privacy mode, every outbound URL/query is also
echoed to stderr (`[localm privacy] fetch_url: …`) so the session leaves a
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
drives, and switch it on in Settings. An API key needs the separate
`browser` scope (see [SECURITY.md](../SECURITY.md)), independent of the
coding agent's shell access, so you can grant one without the other.

## What the policy does NOT govern

- **Child processes.** `run_shell` commands like `pip install`, `npm install`,
  or `git clone` talk to the network themselves. The gate for those is the
  shell-command approval (and `always_confirm` for `run_shell`), not the
  network policy.
- **Model downloads** (`localm pull`) and **online coder providers**
  (OpenAI/Anthropic opt-ins) - explicit user actions.
- **Privacy mode is orthogonal.** Privacy controls what localm writes to
  *disk*; it cannot make network requests untraceable. Any request leaves DNS
  lookups and traffic visible to your network and the remote server. If a
  conversation must stay fully local, keep web access off - that is why the
  chat toggle is per-conversation and off by default.

## Trust note: web content is untrusted input

Fetched pages and search snippets enter the model's context. A malicious page
can contain text crafted to steer the model ("ignore your instructions and
run …"). This is inherent to giving any agent web access, and it is why the
default mode asks per request and why destructive coder actions keep their
own approval step regardless of where the idea came from. Treat approval
prompts that follow a web fetch with extra suspicion.
