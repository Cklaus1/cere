"""The Python test package. Hermeticity: no test sees five pieces of host state through the caller's environment.

- The host's GPU1 production marker (`/var/lib/cere/gpu1.production`,
  `_common.PRODUCTION_MARKER`), whose presence would turn every two-card expectation into a production refusal:
  CERE_GPU1_PRODUCTION_MARKER, the marker path every harness reads, names a path that never exists.
- The host's GPU0 burst state (`/var/lib/cere/gpu0.burst`, `_common.BURST_STATE`, ProdBurst),
  which gpu_queue status prints and gpu_watch exempts, and which reads as a burst in state `unknown` wherever it is
  unreadable (a non-root user under $HOME): CERE_GPU0_BURST_STATE names a path that never exists.
- The caller's cargo builds: the harnesses resolve `cered`, `cere-native` and `cere-mock-sglang` under
  `$CARGO_TARGET_DIR/<build>/` (default `target/`), and a report digests the `cered` it finds there, so a test run
  under a worktree's CARGO_TARGET_DIR read that worktree's binaries (unreadable ones failed it). CARGO_TARGET_DIR
  names an empty dir. No Python test needs a built binary; a test that wants one writes its own and patches the
  variable.
- The host's gate slots (`wt/gate-slot-{0,1}.lock`, scripts/gate_cap.py, NightlyGate): a script a test starts that
  takes one (ci_docker.sh, `make ci`, integrate.py gate) would wait for, or hold, a real gate's slot:
  CERE_GATE_SLOT_DIR names an empty dir.
- The host's PLE share directory (`/dev/shm/cere-host`, `native_dev.HOST_SHARE_DIR_DEFAULT`, O47): its segments, their
  `.lock` files and the dev-loop stamp. `native_dev.dev_share_plan` drops a dead creator's `.tmp` under its `.lock`
  there, so a root-owned 0600 lock failed every `test` run of a non-root user (SignalStop, 2026-10-29), and a share
  suite would stamp the host's directory: CERE_HOST_SHARE_DIR names a dir that does not exist yet.

All five point into a fresh temp dir, whatever the caller's environment named, whenever the package is imported
(`python3 -m unittest tests.<module>`, `python3 -m unittest discover -s tests -t .`); the scripts a test starts inherit
them. `scripts/run_python_tests.py` discovers inside tests/ (this package is not imported there) and sets the same
variables for each module it runs.

A sixth piece, the host's locks (`/tmp/cere-*.lock`, `wt/*.lock`), is not reached through the environment: the
audit hook of `tests/hermetic.py` refuses any test's open of one (both entry points install it). The same hook
refuses any open or listing under the host's share directory."""
import atexit
import os
import shutil
import tempfile

from . import hermetic

HERMETIC_DIR = tempfile.mkdtemp(prefix="cere-tests-hermetic-")
atexit.register(shutil.rmtree, HERMETIC_DIR, True)
os.environ["CERE_GPU1_PRODUCTION_MARKER"] = os.path.join(HERMETIC_DIR, "absent-gpu1.production")
os.environ["CERE_GPU0_BURST_STATE"] = os.path.join(HERMETIC_DIR, "absent-gpu0.burst")
os.environ["CARGO_TARGET_DIR"] = os.path.join(HERMETIC_DIR, "target")
os.mkdir(os.environ["CARGO_TARGET_DIR"])
os.environ["CERE_GATE_SLOT_DIR"] = os.path.join(HERMETIC_DIR, "gate-slots")
os.environ["CERE_HOST_SHARE_DIR"] = os.path.join(HERMETIC_DIR, "cere-host")
hermetic.install()
