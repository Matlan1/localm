# Security Policy

## Reporting a vulnerability

Please do not open a public issue, discussion or pull request for a security problem.
Report it privately with GitHub's
[Report a vulnerability](https://github.com/Matlan1/localm/security/advisories/new) form
(the repository's *Security* tab -> *Advisories*).

Please include:

- the localm version and your operating system,
- steps to reproduce it, or a proof of concept,
- the impact as you understand it.

We will acknowledge the report and work with you on a fix and a disclosure timeline.

## Supported versions

localm is pre-1.0; security fixes land on the latest `master`.

## Using localm safely

- By default localm listens only on this computer. Set an API key before you make it
  reachable from your network.
- A key with a privileged scope gives access to the host, so give it only to clients you trust.
- Third-party plugins run unsandboxed in-process, so install only code you trust.
- Download models from sources you trust, and keep localm up to date.

[docs/security.md](docs/security.md) describes the whole model: authentication and scopes,
what models and plugins may do, the network policy, TLS, updates, and how to verify a release.
