# Using localm from your phone

localm's web GUI is an installable PWA, so your phone can run it as an app. There
is **no app-store download and no localm cloud**: the app is served by *your*
localm server, and the phone talks straight to your machine. (This is how every
self-hosted app works - the good model stays on your box, the phone is just the
terminal.) So the only real questions are *how the phone reaches your server* and
*how you open the right address*.

## The short version

- **Same Wi-Fi:** run `localm gui -H 0.0.0.0` (set an API key first); open the
  `https://<your-ip>:<port>/` it prints on your phone; trust the certificate once
  (one tap on the key screen); enter your key; choose *Install app*.
- **Anywhere else:** put the machine and the phone on a
  [Tailscale](https://tailscale.com) network and open the machine's Tailscale
  address - localm serves HTTPS there too.

## On the same Wi-Fi (LAN)

1. **Bind to your network.** By default `localm gui` listens only on this machine
   (`127.0.0.1`). To let your phone reach it, bind to all interfaces - and set an
   API key first, because the GUI also exposes the coder agent:
   - set `LOCALM_API_KEY` to something only you know, then
   - `localm gui -H 0.0.0.0`

   localm now serves **HTTPS automatically** on a network bind, so the key and
   your prompts are encrypted in transit. No reverse proxy, no extra tools.
2. **Open the printed address on the phone.** On startup the console prints the
   address to use, e.g. `https://192.168.1.50:8642/` (the port may differ if 8642
   was busy - use what it prints). Type that into the phone browser.
   - *Tip (experimental PoC):* `localm gui -H 0.0.0.0 --qr` prints a scannable QR
     of that address so you can point the phone camera at the terminal instead of
     typing it. Needs `pip install "localm[qr]"` and a terminal that renders block
     glyphs (Windows Terminal is fine). This is a proof-of-concept and may change.
3. **Trust the certificate once.** Because localm signs its certificate with its
   own local CA (not a public one), the first visit shows a one-time "not secure"
   warning - this is a browser rule for any private certificate on a raw IP, not a
   localm bug. Proceed past it to the page, then tap **Install certificate** on the
   key screen (or open `https://<ip>:<port>/localm-ca.crt`) and trust it. After
   that: no warning, and the app installs. See [tls.md](tls.md) for the exact
   per-platform trust step (iOS has an extra toggle).
4. **Enter your key and install.** Type the `LOCALM_API_KEY` into the key screen
   (it is sent over the now-encrypted connection and stored only in that browser).
   Then choose *Install app* / *Add to Home screen* from the browser menu - localm
   gets its own icon.

> **Why the trust step exists.** A *true* installed PWA (offline app shell, real
> app icon) needs a "secure context" - HTTPS or `localhost`. localm gives you the
> HTTPS automatically; the only manual part is trusting its certificate once per
> device, because no public certificate authority will issue a cert for a private
> LAN IP. After the one-time trust, the full install works.

## From anywhere (remote): Tailscale (recommended)

Reaching a home server from *outside* your network is the genuinely hard part,
and it is the same for every self-hosted app - no project solves it for casual
users without running a paid cloud relay. The cleanest path that needs no
port-forwarding, no domain, and no certificate wrangling is
[Tailscale](https://tailscale.com) (free for personal use):

1. Install Tailscale on the machine running localm **and** on your phone; sign in
   to the same account on both. They are now on one private network.
2. Run `localm gui -H 0.0.0.0` (with `LOCALM_API_KEY` set). localm's certificate
   already covers your Tailscale IP, so it serves HTTPS there too.
3. Open the machine's Tailscale address on the phone, e.g.
   `https://100.x.y.z:8642/`, and trust the certificate once (step 3 above).
4. **Optional, for a public-CA cert with no trust step:** run `tailscale serve` in
   front of localm; your phone then opens `https://<machine>.<tailnet>.ts.net/`
   with a certificate browsers already trust, so there is no warning to clear.

More involved remote options: a reverse proxy (Caddy) + a domain + Let's Encrypt
(see [tls.md](tls.md)), or a Cloudflare Tunnel. **Avoid UPnP / manual
port-forwarding** - it is the classic way to expose a machine you did not mean to.

## Security

Binding past loopback exposes the coder agent (shell + file edits) and the API.
Always set `LOCALM_API_KEY` before `-H 0.0.0.0`; localm refuses to bind to the
network without a key unless you pass `--insecure`. Traffic itself is encrypted by
built-in TLS by default (`--no-tls` turns it off for a trusted, isolated LAN). On
a trusted home LAN a key plus the built-in TLS is enough; for anything reachable
from the internet use Tailscale or a TLS reverse proxy with a real certificate.
See [network.md](network.md) and [tls.md](tls.md).

## What localm does NOT do (on purpose)

- **No native app-store app.** The PWA covers the phone experience without an
  app-store account or a second codebase to maintain. Most local-LLM tools have
  no phone story at all; the closest peer that does (Open WebUI) also ships a PWA.
- **No localm cloud relay.** Vendor relays (Home Assistant's Nabu Casa, Plex,
  Synology QuickConnect) make remote access one-click, but they require running
  paid infrastructure and routing your traffic through a third party. localm stays
  local-first; Tailscale gives you the same "works from anywhere" without anyone
  else in the data path.
