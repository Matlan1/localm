# Serving localm over a LAN: TLS and reverse proxies

localm binds to 127.0.0.1 by default and that is the right setting for almost
everyone. If you want other machines on your network to use your models, do
not just bind 0.0.0.0 and move on. This guide covers the safe way.

## Threat model in one paragraph

The HTTP API can run inference (costs your GPU time), unload and switch
models, and, when the GUI is enabled, start coder sessions that write files
and run shell commands on the host. Anyone who can reach the port can do all
of that. Bearer auth (`LOCALM_API_KEY`) is the minimum; TLS stops the key
from crossing the network in cleartext.

## Step 1: always set an API key

```powershell
$env:LOCALM_API_KEY = Read-Host -MaskInput "localm API key"
localm serve mymodel -H 0.0.0.0
```

Without the key, localm prints a warning when binding beyond localhost.
Clients send `Authorization: Bearer <key>`.

Do not put the key on the command line (it lands in shell history); use
`Read-Host -MaskInput` as above or set it in your environment manager.

## Step 2: terminate TLS in front of localm

localm itself speaks plain HTTP. Put a reverse proxy in front for TLS.

### Caddy (simplest, automatic self-signed or ACME certificates)

`Caddyfile`:

```
llm.example.internal {
    tls internal              # self-signed CA for LAN use
    reverse_proxy 127.0.0.1:8642
}
```

Run `caddy run`. Keep localm itself on 127.0.0.1 so the only way in is
through the proxy:

```powershell
localm serve mymodel          # binds 127.0.0.1:8642
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

## Step 3: scope what you expose

- Serve the bare API (`localm serve`) over the LAN, not the GUI. The GUI's
  coder sessions execute code on the host; that should stay a localhost
  tool.
- If remote machines need different origins in the browser, set
  `cors_origins` in `~/.localm/config.json` to an explicit list. Never use
  `"*"` on an exposed bind.
- Firewall the localm port (8642-8741) so only the proxy host can reach it
  when the proxy runs on a different machine.

## What this does not cover

This setup is for a trusted LAN. Exposing localm to the public internet
needs more: rate limiting, request size caps, fail2ban-style lockout, and a
real certificate authority. If you need that, treat localm like any internal
service behind a VPN instead.
