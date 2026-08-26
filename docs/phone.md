# Using localm from your phone

localm's web GUI is an installable PWA, so your phone can run it as an app. There
is **no app-store download and no localm cloud**: the app is served by *your*
localm server, and the phone talks straight to your machine. So the only real
questions are *how the phone reaches your server* and *how you open the right
address*.

## The short version

- **Same Wi-Fi:** run `localm gui -H 0.0.0.0` (set an API key first) - or, from
  a GUI that is already open, set *Settings > Server > Bind address* to
  `0.0.0.0` and click *Restart server*. Open the `https://localm.local:<port>/`
  it prints on your phone (or the `https://<your-ip>:<port>/` it prints next to
  it; the Settings *Companion app* card shows that same IP address); trust the
  certificate once (one tap on the key screen); enter your key; choose
  *Install app*.
- **Anywhere else:** put the machine and the phone on a
  [Tailscale](https://tailscale.com) network and open the machine's Tailscale
  address - localm serves HTTPS there too, by name.

## On the same Wi-Fi (LAN)

1. **Bind to your network.** By default `localm gui` listens only on this machine
   (`127.0.0.1`). To let your phone reach it, bind to all interfaces - and set an
   API key first, because the GUI also exposes the coder agent:
   - set an API key: *Settings > Security > Owner key > Generate new key*, or
     `localm key generate`, or set `LOCALM_API_KEY` to something only you
     know - Linux/macOS: `export LOCALM_API_KEY=...`; Windows PowerShell:
     `$env:LOCALM_API_KEY="..."` (or use the desktop launcher's Generate
     button), then
   - `localm gui -H 0.0.0.0`, **or** set *Settings > Server > Bind address* to
     `0.0.0.0` and click *Restart server* - no terminal needed. (Without a
     strong key, `-H 0.0.0.0` on the command line refuses to start at all; the
     Settings-driven bind instead falls back and stays on `127.0.0.1`, and the
     *Companion app* card tells you why.)

   localm now serves **HTTPS automatically** on a network bind, so the key and
   your prompts are encrypted in transit. No reverse proxy, no extra tools.
2. **Open the printed address on the phone.** On startup the console prints the
   addresses to use. The friendliest is the **name**, `https://localm.local:8642/`
   (the port may differ if 8642 was busy - use what it prints); it works because
   localm advertises `localm.local` over mDNS/Bonjour, which phones resolve with no
   setup. The `https://192.168.1.50:8642/` IP address is printed right below it as a
   fallback (some restrictive networks block mDNS). Type either into the phone
   browser.
   - *Rename it:* the name is `localm` by default. Change it with
     `localm config mdns_name studio` (then it is `studio.local`), or turn the whole
     advertisement off with `localm config mdns_enabled false`.
   - *Tip (experimental PoC):* `localm gui -H 0.0.0.0 --qr` prints a scannable QR
     of the address so you can point the phone camera at the terminal instead of
     typing it. No extra install needed (`qrcode` ships with localm); it just
     needs a terminal that renders block glyphs (Windows Terminal is fine). This
     is separate from the key-pairing QR in step 4 below: this one is the URL to
     open, that one copies the API key once you are already in the app somewhere
     else.
3. **Trust the certificate once.** Because localm signs its certificate with its
   own local CA (not a public one), the first visit shows a one-time "not secure"
   warning - this is a browser rule for any private certificate on a raw IP, not a
   localm bug. Proceed past it to the page, then tap **Install certificate** on the
   key screen (or open `https://<ip>:<port>/localm-ca.crt`) and trust it. After
   that: no warning, and the app installs. See [tls.md](tls.md) for the exact
   per-platform trust step (iOS has an extra toggle).
4. **Enter your key and install.** Type the `LOCALM_API_KEY` into the key screen
   (it is sent over the now-encrypted connection and stored only in that browser),
   or skip the typing: tap **Scan QR code** on that same screen, point the phone
   at the computer's **Settings -> Companion app** page (it shows a QR of your
   key once you are signed in there), and the key is copied over automatically.
   Then choose *Install app* / *Add to Home screen* from the browser menu - localm
   gets its own icon.

> **Why the trust step exists.** A *true* installed PWA (offline app shell, real
> app icon) needs a "secure context" - HTTPS or `localhost`. localm serves the
> HTTPS automatically; the one manual part is the one-time certificate trust above.

## From anywhere (remote): Tailscale (recommended)

Remote access needs a private network. The cleanest path that needs no
port-forwarding, no domain, and no certificate wrangling is
[Tailscale](https://tailscale.com) (free for personal use):

1. Install Tailscale on the machine running localm **and** on your phone; sign in
   to the same account on both. They are now on one private network.
2. Run `localm gui -H 0.0.0.0` (with `LOCALM_API_KEY` set). localm detects your
   Tailscale node name and Tailscale IP, covers both in its certificate, and prints
   the reachable URL - so it serves HTTPS there too, by name.
3. Open the machine's Tailscale address on the phone. localm prints the **name**,
   e.g. `https://mybox.tailnet.ts.net:8642/` (Tailscale MagicDNS resolves it on any
   device in your tailnet), with the `https://100.x.y.z:8642/` IP as a fallback.
   Trust the certificate once (step 3 above).
   - *Reach it as `localm` on Tailscale:* Tailscale names the node after your
     machine's hostname. To make it literally `localm.<tailnet>.ts.net`, rename the
     node once with `tailscale up --hostname=localm`. localm prints this exact hint
     at startup when the node is not already named to match; it never renames your
     node for you.
4. **Optional, for a public-CA cert with no trust step:** run `tailscale serve` in
   front of localm; your phone then opens `https://<machine>.<tailnet>.ts.net/`
   with a certificate browsers already trust, so there is no warning to clear.

More involved remote options: a reverse proxy (Caddy) + a domain + Let's Encrypt
(see [tls.md](tls.md)), or a Cloudflare Tunnel. **Avoid UPnP / manual
port-forwarding** - it is the classic way to expose a machine you did not mean to.

## Security

Binding past loopback exposes the coder agent (shell + file edits) and the API.
Set `LOCALM_API_KEY` before `-H 0.0.0.0`; localm refuses to bind to the network
without a key unless you pass `--insecure`. Traffic is encrypted by built-in TLS
by default (`--no-tls` turns it off for a trusted, isolated LAN). On a trusted
home LAN a key plus built-in TLS is enough; for anything reachable from the
internet use Tailscale or a TLS reverse proxy with a real certificate. See
[network.md](network.md).

## What localm does NOT do (on purpose)

- **No native app-store app.** The PWA covers the phone experience without an
  app-store account or a second codebase to maintain.
- **No localm cloud relay.** A vendor relay makes remote access one-click, but it
  requires paid infrastructure and routes your traffic through a third party.
  localm stays local-first; Tailscale gives you the same "works from anywhere"
  without anyone else in the data path.
