# Contributing to CERE

Thanks for helping. CERE is MIT-licensed; by contributing you agree that your contribution is licensed under the
same [MIT License](LICENSE).

## Developer Certificate of Origin (DCO)

Pull requests are welcome. Every commit must be signed off under the
[Developer Certificate of Origin 1.1](https://developercertificate.org/). Signing off is one line at the end of the
commit message, which `git commit -s` adds for you:

```
Signed-off-by: Your Name <you@example.com>
```

With it you certify that you wrote the change, or otherwise have the right to submit it under the project's MIT
license. It is not a copyright assignment: you keep the copyright to your contribution. Use your real name and an
email you can be reached at. Forgot it? `git commit --amend -s` fixes the last commit, and
`git rebase --signoff main` signs off every commit of a branch. Pull requests with unsigned commits cannot be merged.

## Before you open a pull request

- `make ci` passes on your machine (CPU only: build, format, clippy, the Rust, C/C++ host and Python suites).
- New behavior comes with a test. Executor changes that only a GPU can check say which GPU suite you ran
  (`executor/tests/gpu/`, `executor/vulkan/tests/`) and on which card, driver and toolkit.
- A change to a runtime config moves its pinned hash: run `python3 scripts/pin_configs.py --write` and commit the
  result. A rebuilt executor is re-pinned with `scripts/native/lock_native.py` (see [docs/BUILDING.md](docs/BUILDING.md)).
- Performance claims name the model, artifact manifest SHA-256, library SHA-256, config, GPU, driver and the
  measured numbers with their repeat count.
- Third-party code: add it to `executor/THIRD_PARTY.json` and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) with
  its license; only licenses compatible with MIT distribution are accepted. Never commit model weights or tensors
  computed from them.

## Reporting bugs and asking questions

Open an [issue](https://github.com/Cklaus1/cere/issues) with the CERE version or commit, the config you ran, the
GPU, driver and toolkit, the request (with any private content removed) and `cered`'s log around the failure.
Questions and ideas go to [Discussions](https://github.com/Cklaus1/cere/discussions). Security issues: see
[SECURITY.md](SECURITY.md).

## Releases

CERE is developed in a private repository; each release here is a scrubbed export of it, committed on top of this
repository's history and tagged `v<version>` (`v0.6.0` first). A merged pull request stays in this history with its
sign-off; its change is applied to the development repository too, so the next release carries it.
