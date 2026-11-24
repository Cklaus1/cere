# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately through GitHub's private vulnerability
reporting: [Report a vulnerability](https://github.com/Cklaus1/cere/security/advisories/new) (the **Security** tab
of [github.com/Cklaus1/cere](https://github.com/Cklaus1/cere)). Include the CERE version or commit, the
configuration, and steps to reproduce. You will get an acknowledgement within 7 days; a fix or a mitigation plan
follows as soon as the issue is confirmed, and we credit reporters who want to be named.

## Supported versions

Only the latest tagged release (and `main`) receives security fixes.

## Deployment notes

- `cered` authenticates API requests with bearer tokens read from environment variables named in the config
  (`operator_token_env`, `[[auth.scopes]] token_env`). Keep them out of config files and logs. A scope can be
  configured keyless; do that only behind a trusted proxy.
- Bind `cered` to loopback (the shipped profiles use `127.0.0.1`) or put it behind a TLS-terminating reverse proxy
  before exposing it on a network.
- The federation control plane (`cere-worker`, `proto/cere_worker.proto`) authenticates both ends with mutual TLS
  (`ca_file`, `cert_file`, `key_file`, principals) and only listens on loopback or CGNAT (100.64.0.0/10, e.g. a
  Tailscale tailnet) addresses; keep it on a private network.
- A runtime config pins every executable input (library, artifact manifest, tokenizer, template) by SHA-256 and
  `cered` refuses to become ready on a mismatch. Do not disable those pins in production.
- The tokenizer sidecar loads tokenizer files with `trust_remote_code=False`; still, only load model repositories you
  trust and pin their revisions (`scripts/native/model_source.py`).
