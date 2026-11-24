"""No test opens a host lock (NonrootLive). The live scripts' run locks (`/tmp/cere-federation.lock`,
`/tmp/cere-bench-g0.lock`, ...: `/tmp/cere-*.lock`) and the GPU queue's locks beside the worktrees (`wt/gpu0-dev.lock`,
`wt/build.lock`, ...: `_common.DEV_LOCKS`' directory) belong to live runs. A test that opened one contended with a live
run holding it, and as a non-root user failed on a lock root created (the fake worlds of test_placement_live,
test_migration_live and test_multimodel_live probed `/tmp/cere-federation.lock`; test_live_backends through them).

`install()` adds an audit hook that refuses such an open with `HostLockOpened` before it happens, whatever the user,
so a regression fails as root and in a clean CI container too, where the lock is absent or writable. A test points the
module's lock at its tmp dir instead. `tests/__init__.py` installs it on import (`python3 -m unittest tests.<module>`)
and `scripts/run_python_tests.py` in each process it starts (its discovery does not import the package).

The same hook refuses (`HostShareReached`) an open, listing or mkdir under the host's PLE share directory
(`/dev/shm/cere-host`, `native_dev.HOST_SHARE_DIR_DEFAULT`): its segments' `.lock` files are root's 0600 ones on the
host, and a test that reached them failed as a non-root user (SignalStop, 2026-10-29). CERE_HOST_SHARE_DIR (set by
both entry points) points the harnesses at a temp dir instead.

DevMemCap (O338): `install()` also puts stand-ins for `systemctl` and `systemd-run` first on PATH (a temp dir), so
no test reads the host's `ceredev.slice` or creates a scope in it, and the tests pass in a container without systemd.
The `systemctl` stand-in answers only `show -p LoadState,MemoryMax ceredev.slice` (`_common.dev_slice_refusal`),
with CERE_FAKE_DEV_SLICE_SHOW when set (`\\n`-separated `Key=value` lines, e.g. `LoadState=loaded\\nMemoryMax=infinity`)
and a capped, loaded slice otherwise, and the same query of `cereshare.slice` (`_common.share_slice_refusal`, with
CERE_FAKE_SHARE_SLICE_SHOW); anything else exits 1. The `systemd-run` stand-in appends its arguments as one
line to CERE_FAKE_SYSTEMD_RUN_LOG when set, drops every word up to `--` and execs the rest (same pid, fds and env,
as `systemd-run --scope` does).

DevShareCap (O354): `install()` points CERE_CGROUP_ROOT and CERE_SHM_ROOT (scripts/live/share_charge.py) at a temp
dir, so no test reads the host's slices' accounting or moves a host segment: a cgroup root whose ceredev.slice and
cereshare.slice have room (gpu_queue's start floor passes) and an empty shm root (nothing to move)."""
import atexit
import os
import shutil
import stat
import sys
import tempfile

# `_common.DEV_LOCKS`' directory (tests/test_host_locks.py checks they agree).
WT = "/var/lib/cere/wt"
# native_dev.HOST_SHARE_DIR_DEFAULT (tests/test_share_hermetic.py checks they agree).
HOST_SHARE = "/dev/shm/cere-host"
SHARE_EVENTS = ("open", "os.listdir", "os.scandir", "os.mkdir")
# The stand-ins `install()` puts first on PATH (module doc, DevMemCap).
FAKE_SYSTEMCTL = """#!/bin/sh
# tests/hermetic.py stand-in: only _common.slice_refusal's query (ceredev.slice, cereshare.slice).
if [ "$#" -eq 4 ] && [ "$1" = show ] && [ "$2" = -p ] && [ "$3" = LoadState,MemoryMax ] && [ "$4" = ceredev.slice ]; then
  printf '%b\\n' "${CERE_FAKE_DEV_SLICE_SHOW:-LoadState=loaded\\nMemoryMax=94489280512}"
  exit 0
fi
if [ "$#" -eq 4 ] && [ "$1" = show ] && [ "$2" = -p ] && [ "$3" = LoadState,MemoryMax ] && [ "$4" = cereshare.slice ]; then
  printf '%b\\n' "${CERE_FAKE_SHARE_SLICE_SHOW:-LoadState=loaded\\nMemoryMax=85899345920}"
  exit 0
fi
echo "systemctl stand-in (tests/hermetic.py): only 'show -p LoadState,MemoryMax ceredev.slice|cereshare.slice', not: $*" >&2
exit 1
"""
FAKE_SYSTEMD_RUN = """#!/bin/sh
# tests/hermetic.py stand-in: logs its words, then execs the command after `--` (as systemd-run --scope does).
[ -z "${CERE_FAKE_SYSTEMD_RUN_LOG:-}" ] || printf '%s\\n' "$*" >>"$CERE_FAKE_SYSTEMD_RUN_LOG"
while [ "$#" -gt 0 ] && [ "$1" != -- ]; do shift; done
[ "$#" -gt 0 ] && shift
exec "$@"
"""
# DevShareCap (module doc): the fake cgroup root's slices, each with room (88 GiB cap, 1 GiB charged).
FAKE_MEMCG = {"memory.max": "94489280512\n", "memory.current": "1073741824\n",
              "memory.stat": "anon 1073741824\nfile 0\nshmem 0\nactive_file 0\ninactive_file 0\n"}


class HostLockOpened(RuntimeError):
    pass


class HostShareReached(RuntimeError):
    pass


def host_lock(path: str) -> bool:
    name = os.path.basename(path)
    parent = os.path.dirname(path)
    return name.endswith(".lock") and (parent == WT or (parent == "/tmp" and name.startswith("cere-")))


def host_share(path: str) -> bool:
    return path == HOST_SHARE or path.startswith(HOST_SHARE + "/")


def refuse_host_locks(event: str, args: tuple) -> None:
    if event not in SHARE_EVENTS or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.path.abspath(os.fsdecode(args[0]))
    if host_share(path):
        raise HostShareReached(f"a test reached the host's PLE share directory ({event} {path}); "
                               "CERE_HOST_SHARE_DIR names the test's share dir")
    if event == "open" and host_lock(path):
        raise HostLockOpened(f"a test opened the host lock {path}; point the module's lock at the test's tmp dir")


_installed = False


def install_systemd_fakes() -> str:
    """Writes the systemctl and systemd-run stand-ins into a fresh temp dir (removed at exit), puts it first on PATH
    and returns it."""
    bindir = tempfile.mkdtemp(prefix="cere-tests-systemd-")
    atexit.register(shutil.rmtree, bindir, True)
    for name, body in (("systemctl", FAKE_SYSTEMCTL), ("systemd-run", FAKE_SYSTEMD_RUN)):
        path = os.path.join(bindir, name)
        with open(path, "w") as f:
            f.write(body)
        os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", os.defpath)
    return bindir


def install_share_fakes() -> str:
    """Points share_charge.py at a fresh temp dir (removed at exit): CERE_CGROUP_ROOT at <dir>/cgroup holding a
    roomy ceredev.slice and cereshare.slice (FAKE_MEMCG), CERE_SHM_ROOT at the empty <dir>/shm. Returns the dir."""
    root = tempfile.mkdtemp(prefix="cere-tests-memcg-")
    atexit.register(shutil.rmtree, root, True)
    for name in ("ceredev.slice", "cereshare.slice"):
        os.makedirs(os.path.join(root, "cgroup", name))
        for leaf, text in FAKE_MEMCG.items():
            with open(os.path.join(root, "cgroup", name, leaf), "w") as f:
                f.write(text)
    os.mkdir(os.path.join(root, "shm"))
    os.environ["CERE_CGROUP_ROOT"] = os.path.join(root, "cgroup")
    os.environ["CERE_SHM_ROOT"] = os.path.join(root, "shm")
    return root


def install() -> None:
    global _installed
    if not _installed:
        sys.addaudithook(refuse_host_locks)
        install_systemd_fakes()
        install_share_fakes()
        _installed = True
