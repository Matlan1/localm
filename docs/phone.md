# Using localm from your phone

localm's web GUI is an installable PWA, so your phone can run it as an app. There
is **no app-store download and no localm cloud**: the app is served by *your*
localm server, and the phone talks straight to your machine. (This is how every
self-hosted app works - the good model stays on your box, the phone is just the
terminal.) So the only real questions are *how the phone reaches your server* and
*how you open the right address*.

## The short version

- **Same Wi-Fi:** run `localm gui -H 0.0.0.0` (set an API key first); open the
  `http://<your-ip>:<port>/` it prints, on your phone; choose *Install app*.
- **Anywhere else:** put the machine and the phone on a [Tailscale](https://tailscale.com)
  network and open the machine's Tailscale address. `tailscale serve` adds HTTPS,
  which also makes the PWA fully installable.

## On the same Wi-Fi (LAN)

1. **Bind to your network.** By default `localm gui` listens only on this machine
   (`127.0.0.1`). To let your phone reach it, bind to all interfaces - and set an
   API key first, because the GUI also exposes the coder agent:
   - set `LOCALM_API_KEY` to something only you know, then
   - `localm gui -H 0.0.0.0`
2. **Open the printed address on the phone.** On startup the console now prints
   the address to use, e.g. `http://192.168.1.50:8642/`. Type that into the phone
   browser.
3. **Install it.** In the phone browser menu choose *Install app* / *Add to Home
   screen*. localm now has its own icon.

> **Install vs. shortcut (the HTTPS catch).** A *true* installed PWA (offline app
> shell, real app icon) needs a "secure context" - HTTPS, or `localhost`. Over a
> plain `http://192.168.x.x` LAN address your phone can still **use** the GUI
> fully, but the browser may only offer a home-screen *shortcut*, not a full
> install. To get the real install on a LAN, serve it over HTTPS (Tailscale
> `serve`, below, or a reverse proxy with a certificate - see
> [tls.md](tls.md)).

## From anywhere (remote): Tailscale (recommended)

Reaching a home server from *outside* your network is the genuinely hard part,
and it is the same for every self-hosted app - no project solves it for casual
users without running a paid cloud relay. The cleanest path that needs no
port-forwarding, no domain, and no certificate wrangling is
[Tailscale](https://tailscale.com) (free for personal use):

1. Install Tailscale on the machine running localm **and** on your phone; sign in
   to the same account on both. They are now on one private network.
2. Run `localm gui -H 0.0.0.0` (with `LOCALM_API_KEY` set).
3. Open the machine's Tailscale address on the phone, e.g. `http://100.x.y.z:8642/`.
4. **For HTTPS** (a nicer name and full PWA install), run `tailscale serve` in
   front of localm; your phone then opens
   `https://<machine>.<tailnet>.ts.net/` from anywhere.

More involved remote options: a reverse proxy (Caddy) + a domain + Let's Encrypt
(see [tls.md](tls.md)), or a Cloudflare Tunnel. **Avoid UPnP / manual
port-forwarding** - it is the classic way to expose a machine you did not mean to.

## Security

Binding past loopback exposes the coder agent (shell + file edits) and the API.
Always set `LOCALM_API_KEY` before `-H 0.0.0.0`; localm refuses to bind to the
network without a key unless you pass `--insecure`. On a trusted home LAN a key is
enough; for anything reachable from the internet use Tailscale or a TLS reverse
proxy. See [network.md](network.md) and [tls.md](tls.md).

## What localm does NOT do (on purpose)

- **No native app-store app.** The PWA covers the phone experience without an
  app-store account or a second codebase to maintain. Most local-LLM tools have
  no phone story at all; the closest peer that does (Open WebUI) also ships a PWA.
- **No localm cloud relay.** Vendor relays (Home Assistant's Nabu Casa, Plex,
  Synology QuickConnect) make remote access one-click, but they require running
  paid infrastructure and routing your traffic through a third party. localm stays
  local-first; Tailscale gives you the same "works from anywhere" without anyone
  else in the data path.
