# Reaching localm by name

When you bind localm to your network (`localm gui -H 0.0.0.0` or
`localm serve -H 0.0.0.0`), other devices can reach it by a stable **name**
instead of an IP address that changes with DHCP. There is nothing to install and
nothing to configure: it is on by default.

## On your LAN: `localm.local`

localm advertises itself over **mDNS** (multicast DNS, also called Bonjour /
zeroconf) as `localm.local`. Any device on the same network resolves that name
with no setup:

- Windows 10/11, macOS, iOS, and Android resolve `.local` names natively.
- Linux resolves them when Avahi (`avahi-daemon`) is installed, which most
  desktops ship by default.

So a phone or laptop opens `https://localm.local:8642/` (use the port localm
prints if 8642 was busy) and reaches your server. The name is also folded into
localm's HTTPS certificate, so once you have trusted its local CA
([tls.md](tls.md)), the name-based URL shows no certificate warning.

### Why `.local` and not a bare `localm`

A bare, suffix-less `localm` is **not** reliably resolvable from another machine,
so localm guarantees the portable, cross-platform `.local` (mDNS) name instead.

### Renaming or turning it off

```bash
localm config mdns_name studio     # advertise studio.local instead
localm config mdns_enabled false   # advertise nothing; reach localm by IP only
```

Both settings need a **restart** to take effect: the mDNS advertiser starts once
when the server starts, and the name is also written into the HTTPS
certificate's SANs. Restart the process, or use *Settings > Restart server* in
the GUI.

The name is sanitized to a valid DNS label (lowercase letters, digits, hyphens).
Loopback binds (`127.0.0.1`, the default) never advertise anything; a private
test server started with `--isolated` does not advertise either.

If `localm.local` is already claimed by another host on your network, localm
detects the conflict, logs a note, and falls back to IP-based access rather than
risk a name that resolves to the wrong machine - pick a different `mdns_name` in
that case.

## Over Tailscale: MagicDNS

mDNS does not cross a [Tailscale](https://tailscale.com) network (it is a
different layer-3 overlay), so the LAN `localm.local` name does not apply there.
Tailscale has its own naming, **MagicDNS**, which gives every node a name like
`mybox.tailnet.ts.net` (and a bare `mybox` for devices in the same tailnet).

localm detects your node's MagicDNS name via the `tailscale` CLI, adds it to the
certificate, and prints the reachable URL at startup - so it works by name over
Tailscale automatically, with the Tailscale IP printed alongside as a fallback.

To be reachable as literally `localm` on your tailnet, the Tailscale **node**
must be named `localm`. Tailscale names a node after the machine's hostname, so
rename it once if you want the match:

```bash
tailscale up --hostname=localm
```

localm prints this exact command as a hint at startup when the node is not
already named to match your `mdns_name`. It never runs it for you, since the node
name is your choice and changing it affects how every device addresses this
machine.

## Security notes

- Advertising happens only on a **network bind**, and localm still refuses to
  bind past loopback without an API key (or `--insecure`). The name is a
  convenience on top of the same authenticated, TLS-encrypted server; it grants
  no new access.
- mDNS broadcasts the name and IP on the local segment - that is inherent to LAN
  name discovery and is no more than what you expose by binding to the network in
  the first place. Turn it off with `mdns_enabled false` if you prefer to hand out
  addresses manually.

See also [phone.md](phone.md) (using localm from a phone) and
[network.md](network.md) (model-initiated network access).
