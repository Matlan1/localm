# Serving localm over a LAN: built-in TLS and reverse proxies

localm binds to 127.0.0.1 by default and that is the right setting for almost
everyone. When you bind past loopback so other machines (or your phone) can
reach it, localm serves **HTTPS out of the box** - it generates its own
certificate and encrypts every connection. You do not need a reverse proxy or
any external tool for that anymore; the proxy section below is for advanced
setups (a real domain, a public CA).

## Threat model in one paragraph

Anyone who can reach the port can run inference (costs your GPU time), unload
and switch models, and, when the GUI is enabled, start coder sessions that write
files and run shell commands on the host. Bearer auth (`LOCALM_API_KEY`) is the
minimum; TLS stops the key and your prompts from crossing the network in
cleartext. localm now does the TLS itself on a network bind, so both protections
are on by default.

## Always set an API key

```powershell
# Windows PowerShell
$env:LOCALM_API_KEY = Read-Host -MaskInput "localm API key"
localm serve mymodel -H 0.0.0.0
```

```bash
# Linux / macOS
read -rs LOCALM_API_KEY && export LOCALM_API_KEY
localm serve mymodel -H 0.0.0.0
```

localm refuses to bind past loopback without a key of at least 8 characters
(pass `--insecure` to override on a trusted, isolated network; a key that
clears the 8-character floor still is not necessarily hard to guess - prefer
a generated one, `localm key generate`, over something memorable). Clients
send `Authorization: Bearer <key>`. Do not put the key on the command line
(it lands in shell history); use `Read-Host -MaskInput` as above or set it in
your environment manager.

`-H` also accepts an IPv6 literal (`-H ::` for every interface on both
IPv4 and IPv6, or a specific address like `-H ::1`); localm brackets it
correctly in the URLs it prints and adds it as an IP SAN in the certificate.

## Built-in TLS (the default past loopback)

Run either of:

```powershell
localm gui   -H 0.0.0.0     # GUI + API, HTTPS
localm serve mymodel -H 0.0.0.0   # bare API, HTTPS
```

localm prints an `https://<ip>:<port>/` address (the port may differ from 8642
if 8642 was busy - use the address it prints). Under the hood:

- On first run it creates a small **local certificate authority** under
  `<LOCALM_HOME>/tls/` (`ca.crt` + `ca.key`) and a server certificate signed by
  it. The certificate covers 127.0.0.1/::1, this machine's own hostname and LAN
  IP, the mDNS name (`localm.local` by default), and the Tailscale MagicDNS
  name when Tailscale is up - see [phone.md](phone.md) for how to reach localm
  by name instead of a DHCP-assigned IP. The raw Tailscale IP is not reliably
  covered (it depends on hostname resolution turning it up, which can miss it,
  especially on Linux/macOS) - reach localm by its Tailscale name rather than
  its `100.x.y.z` address to avoid a certificate warning. The same cert works
  however a device reaches you, within that set.
- The certificate is reused across restarts, and it is regenerated as it nears
  expiry (about 30 days before) or when its addresses/names change. The CA is
  reused for those regenerations, so a device you trusted once stays trusted
  after your address changes.
- The CA itself is replaced only when it cannot be reused: it has reached its own
  expiry (it is issued for about 10 years), or `ca.crt`/`ca.key` are missing or
  unreadable. localm mints a fresh CA in those cases, and every device that
  trusted the old one has to repeat the trust step below.
- If certificates ever get into a state you cannot explain, stop localm, delete
  `<LOCALM_HOME>/tls/`, and start it again. The CA and the certificate are both
  rebuilt from scratch, so every device has to trust the new CA afterwards.
- The CA private key never leaves your machine.

### The one-time "trust this certificate" step

Because the certificate is signed by *your* CA and not a public one, the first
time a device opens the `https://` address its browser shows a one-time "not
secure" warning, and a phone will not offer **Install app** until the CA is
trusted. This is a browser rule for any private certificate on a raw IP; localm
makes clearing it one tap:

1. Open the printed `https://<ip>:<port>/` address on the device.
2. Proceed through the browser's one-time warning to reach the page.
3. On the key screen tap **Install certificate** (or open
   `https://<ip>:<port>/localm-ca.crt` directly) and trust it when prompted:
   - **Firefox (any OS):** Firefox keeps its OWN certificate store and ignores the
     operating system's, so a Windows/macOS install does not cover it. Open the
     downloaded `localm-ca.crt` and import it, or go to Settings > Privacy &
     Security > Certificates > View Certificates > Authorities > Import and check
     "Trust this CA to identify websites". (Or set
     `security.enterprise_roots.enabled` to `true` in `about:config` to make Firefox
     reuse the OS store.)
   - **Chrome / Edge (Windows, macOS, Android):** open the downloaded `localm-ca.crt`
     and confirm "trust for web sites" (they use the operating system store).
   - **iOS / iPadOS:** the profile downloads, then Settings > General > VPN &
     Device Management installs it, and Settings > General > About > Certificate
     Trust Settings turns it on.
   - **Windows (system store, for Chrome / Edge):** import `localm-ca.crt` into
     "Trusted Root Certification Authorities" for the current user.

After trusting it once: no more warning, and the PWA installs normally (until the
CA is replaced, as above). On a plain
loopback `localm gui` (127.0.0.1, HTTP) there is no certificate and no trust step at
all - this only applies to a network (phone/LAN) bind over HTTPS.

### Turning it off or bringing your own certificate

```powershell
localm gui -H 0.0.0.0 --no-tls          # plain HTTP (key crosses the LAN in cleartext)
localm gui -H 0.0.0.0 --tls-cert D:\certs\localm.crt --tls-key D:\certs\localm.key
```

`--no-tls` serves plain HTTP and is an escape hatch for a trusted, isolated
network only. `--tls-cert`/`--tls-key` let you supply your own certificate (for
example one issued for a real hostname) instead of the built-in local CA.

### From the GUI: Settings > Server

Everything above has a Settings equivalent, so a browser-only user (no
terminal) can bind past loopback too - Settings > Server:

- **Bind address** (`bind_host`) - blank for loopback-only, `0.0.0.0` for
  every IPv4 interface, `::` for every interface on both IPv4 and IPv6, or a
  specific literal address.
- **Encrypt network traffic (TLS)** (`tls_enabled`) - on by default; turning
  it off is the Settings equivalent of `--no-tls`.
- **Custom TLS certificate / private key** (`tls_cert` / `tls_key`) - the
  Settings equivalent of `--tls-cert`/`--tls-key`; leave both blank for the
  built-in CA.

All four are owner-only (Settings hides them, and the API refuses them, for a
non-owner `config:write` key) and apply on the next restart (Settings has a
**Restart server** button). There is **no Settings equivalent of
`--insecure`**, by design: a key check that failed on this route would leave a
browser-only user locked out of a server that refuses to start, with nothing
that could fix it. So instead of exiting, a Settings-driven bind that fails
the check - no key, too short a key, TLS setup failing, or the configured
address simply not being bindable on this machine right now - falls back to
127.0.0.1 with a loud warning (console, log, and a hint in the Companion app
card under Settings > Server) rather than leaving the server unreachable. An
explicit `-H` from a terminal keeps failing hard (exits) exactly as above,
since a terminal can always add `--insecure` or fix the key itself.

## Advanced: terminate TLS with a reverse proxy

You only need this when you want a real domain name and a publicly trusted
certificate (no per-device trust step), for example to reach localm from the
public internet. Keep localm on 127.0.0.1 so the only way in is through the
proxy.

### Caddy (automatic certificates)

`Caddyfile`:

```
llm.example.internal {
    tls internal              # self-signed CA for LAN use, or omit for ACME
    reverse_proxy 127.0.0.1:8642
}
```

```powershell
localm serve mymodel          # binds 127.0.0.1:8642
caddy run
```

### nginx

```nginx
server {
    listen 443 ssl;
    server_name llm.example.internal;

    ssl_certificate     /etc/ssl/localm.crt;
    ssl_certificate_key /etc/ssl/localm.key;

    location / {
        proxy_pass http://127.0.0.1:8642;
        proxy_http_version 1.1;
        # SSE streaming: disable buffering or tokens arrive in bursts
        proxy_buffering off;
        proxy_set_header Connection "";
        proxy_read_timeout 600s;
    }
}
```

The two SSE settings matter: `proxy_buffering off` and a generous
`proxy_read_timeout`, otherwise streaming chat stalls and long generations
get cut off.

## Scope what you expose

- Serve the bare API (`localm serve`) over the LAN, not the GUI, when you do not
  need the coder agent remotely. The GUI's coder sessions execute code on the
  host; that should stay a localhost tool unless you trust every device on the
  network.
- If remote machines need different origins in the browser, set
  `cors_origins` in `<data dir>/config.json` to an explicit list. Never use
  `"*"` at all - not only on an exposed bind. In open (keyless) mode, `"*"`
  lets any website the user's browser visits steal the management shell
  token and take over the instance; the loopback bind does not protect
  against this, since the browser making the request is already local.
- Firewall the localm port (8642-8741) so only the machines you intend can reach
  it.

## What this does not cover

Built-in TLS and a key are right for a trusted LAN or a Tailscale network.
Exposing localm to the public internet needs more: rate limiting, request size
caps, fail2ban-style lockout, and a real certificate authority (the reverse
proxy above). If you need that, treat localm like any internal service behind a
VPN instead.
