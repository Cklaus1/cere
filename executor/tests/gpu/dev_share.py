"""M2.16 L4d, dev lever 8 (warm load): the dev loop's kept PLE segment for the oracle suites (torch-free:
tests/test_dev_share.py imports it, native_dev.py loads it by path).

`native_dev.py [--gpu 1] test --suite S` (S in native_dev.DEV_SHARE_SUITES, unless `--fresh`) mounts HOST_SHARE_DIR
read-write and sets:
    CERE_DEV_HOST_SHARE=<dir>               the segment directory (absolute, no comma)
    CERE_DEV_ARTIFACT_SHA256=<64 hex>       the pinned artifact's manifest sha256 (the device id's `artifact=`)
    CERE_DEV_FILE_SHA256=<json>             {weights file: sha256} from that artifact's manifest.json, which native_dev
                                            hashed against the pin's manifest_sha256 first
    CERE_DEV_LIBRARY_SHA256=<64 hex>        the build's libcere_executor.so, and
    CERE_DEV_KTEST_SHA256=<64 hex>          its libcere_ktest.so (which creates and attaches the segment): the build
                                            native_dev stamps on the segment it keeps (`check_build`)
    CERE_DEV_HOST_SHARE_ATTACH_ONLY=1       GPU1 (`--gpu 1`, which gets the share only while the stamped segment is
                                            published): attach to it, never create one (a GPU0 run creates it)
With them, a model the oracle creates without an explicit device id from a SHARE_CLASSES class (`creating_class`) is
created on `DevShare.device_id()` (the O47 segment kept after release, `host_share_keep=1`): the first such load
creates and publishes the segment, the next ones attach to it (the executor checks the header at ALLOCATE and verifies
the content digest in the background during UPLOAD, failing `cere_model_seal` on a mismatch; create itself succeeds).
Every other class loads fresh: FRESH_CLASSES names why for the classes whose rows need a fresh load, and a class on
neither list is fresh too. Only the read-only PLE weights are shared: each load is its own model (KV, prefix cache,
drafter, levers and arena are per model), and every attach re-checks the shared bytes. An attached model's
HOST_PINNED tensors of every weights file whose desc tensors are all HOST_PINNED and which has a digest are covered by
`cere_ktest_model_host_resident` instead of a put, and each such file is then checked, as cered's upload does
(`Artifact::check_file_views`): sha256(the file's prefix `[0, 8 + n)` read from /model || the resident views in data
offset order) == its manifest digest, the views tiling the data region exactly. The raw checkpoint files under /model
are byte-identical to the artifact's (every weights file of the manifest has `upstream.lfs_sha256` == `sha256`,
which native_dev checks), so the artifact's digests bind the attached bytes to the files the suites read.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple

ENV_DIR = "CERE_DEV_HOST_SHARE"
ENV_ARTIFACT = "CERE_DEV_ARTIFACT_SHA256"
ENV_FILES = "CERE_DEV_FILE_SHA256"
ENV_ATTACH_ONLY = "CERE_DEV_HOST_SHARE_ATTACH_ONLY"
ENV_LIBRARY = "CERE_DEV_LIBRARY_SHA256"
ENV_KTEST = "CERE_DEV_KTEST_SHA256"
REQUIRED_ENV = (ENV_DIR, ENV_ARTIFACT, ENV_FILES, ENV_LIBRARY, ENV_KTEST)
LOAD_PREFIX = "CERE_ORACLE_LOAD "
MODES = ("fresh", "create", "attach")
# In ENV_DIR next to the segment: the artifact and build it was published for, written by native_dev before a share
# run (`stamp`); a load whose share is not stamped there (`stamped`) attaches nothing and loads fresh.
STAMP_FILE = "cere-dev-loop.json"
PLACEMENT_HOST_PINNED = 2  # cere_executor.h CERE_PLACEMENT_HOST_PINNED
HEX64 = re.compile(r"[0-9a-f]{64}")
FILE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*")

# Dev lever 8: the unittest classes (`module.Class`) whose models the oracle creates on the dev-loop segment. Each
# one's rows are tokens, scores, digests, counters or device bytes of its own model: nothing a load takes on the host
# (MemAvailable, the host ledger, VmLck), no memory floor, no per-PID control, no load or clock row, no segment of its
# own. Only these attach; native_dev.py fails a run whose report shows any other class attached.
_SPEC = "speculation tokens, scores, accept counts and digests of its own model"
_DEVICE_BYTES = "its 80 GiB rows are device bytes, the same with a shared host region"
_K16 = "prefix-cache counters, tokens and scores of its own model (K16Layout reads free device memory only)"
_GRAPH = "decode-graph replay counts, tokens and scores of its own model"
SHARE_CLASSES: dict[str, str] = {
    "test_model_k8_forward.K8Speculation": _SPEC,
    "test_model_k8_forward.K8SpeculationSmallM": _SPEC,
    "test_model_k8_forward.K8DeepChainV19": _SPEC,
    "test_model_k8p4_spec.K8SpeculationP4": _SPEC,
    "test_model_k8p4_spec.K8SpeculationSmallMP4": _SPEC,
    "test_model_p8_rows.P8Rows": "per-row bit identity at R rows",
    "test_model_p8_rows.P8Speculation": "grouped speculation bit identity",
    "test_model_k14_mtp.K14MtpDraft": "MTP draft logits and the planner's computed bytes",
    "test_model_k16_prefix.K16Eviction": _K16,
    "test_model_k16_prefix.K16Spec": _K16,
    "test_model_k16_prefix.K16SpecHost": _K16,
    "test_model_k16_prefix.K16Layout": _K16,
    "test_model_k16_prefix.K16End": _K16,
    "test_graph_decode.GraphRowsK6": _GRAPH,
    "test_graph_decode.GraphRowsQsa": _GRAPH,
    "test_graph_decode.GraphLong": _GRAPH,
    "test_graph_decode.GraphSlotReuse": _GRAPH,
    "test_quality_gate.QualityGate": "quality logprob rows of its own model",
    "test_model_k10_long.K10Long": _DEVICE_BYTES,
    "test_model_k10_long.K10ServedPath": _DEVICE_BYTES,
    "test_model_k11_paged.K11Paged": _DEVICE_BYTES,
    "test_model_k11_paged.K11PoolCapacity": _DEVICE_BYTES,
    "test_kernels_k9_qsa.K9QsaReal": "QSA rows; it records free device memory only",
    # O369: native_dev.py's pre-publish load in cereshare.slice (prepublish_dev_share.py, PREPUBLISH_MODULE)
    "prepublish_dev_share.Prepublish": "no rows: it publishes the kept segment the suite then attaches to",
}
# O369: the module native_dev.py runs alone, under cereshare.slice, before a share run that would create the segment.
PREPUBLISH_MODULE = "prepublish_dev_share.py"
SEGMENT_NAME = re.compile(r"cere-host-[0-9a-f]{64}")  # host_share.h segment_name; native_dev.PUBLISHED_SEGMENT
# The classes whose rows need a fresh load, and why (a class on neither list loads fresh as well).
_K15 = "K15 asserts the executor's host ledger at ALLOCATE (mapped + locked == the host estimate, VmLck growth)"
_CLOCK = "clock and stage-time rows of a fresh load"
_V22 = "it loads the default artifact (the S3 variant at the candidate mount), not the /model files the share names"
FRESH_CLASSES: dict[str, str] = {
    "test_model_k8_forward.K8Forward": "its memory floor (device use after SMOKE within 256 MiB), CR4's per-PID "
                                       "memory control and the merge-base library's digests are taken on a fresh "
                                       "load",
    "test_model_k11_paged.K11MergeBase": "it loads under the merge-base library, which predates the attach",
    "test_model_k16_prefix.K16Prefix": "its D6 device-memory row is taken around a fresh ALLOCATE",
    "test_model_k8cal_costs.K8SpecCostsSmallM": _CLOCK,
    "test_profile_forward.D12Profile": _CLOCK,
    "test_profile_forward.D12Speculation": _CLOCK,
    "test_model_k15_snapshot.K15Snapshot": _K15,
    "test_model_k15_snapshot.K15Image": _K15,
    "test_model_k15_snapshot.K15Memory": _K15,
    "test_model_host_share.O47HostShare": "it creates, publishes and unlinks its own segment",
    "test_twocard_o47_host_share.TwoCardHostShare": "it creates, publishes and unlinks its own segment on two cards",
    "test_twocard_d4_executors.TwoCardExecutors": "its D4 host floor holds two private PLE copies at once",
    "test_kernels_k4_ple.K4PLE": "it asserts MemAvailable >= its pinned PLE table + headroom (no executor model)",
    "test_graph_decode.GraphOption": "explicit device ids (the P1 option parser)",
    "test_model_v22_s3.V22Forward": _V22,
    "test_model_v22_s3.V22Speculation": _V22,
    "test_model_v22_s3.V22End": _V22,
}


class DevShareError(AssertionError):
    """The dev-loop attach failed a check; the load fails (never falls back to a fresh load)."""


class DevShare(NamedTuple):
    directory: str
    artifact_sha256: str
    file_sha256: Mapping[str, str]
    library_sha256: str
    ktest_sha256: str
    attach_only: bool

    def device_id(self) -> str:
        return f"0,host_share={self.directory},artifact={self.artifact_sha256},host_share_keep=1"


def from_env(environ: Mapping[str, str]) -> DevShare | None:
    """The dev-loop share of `environ`; None when none of its variables is set. A partial or malformed set fails."""
    present = [k for k in (*REQUIRED_ENV, ENV_ATTACH_ONLY) if k in environ]
    if not present:
        return None
    missing = [k for k in REQUIRED_ENV if k not in environ]
    if missing:
        raise DevShareError(f"dev-loop share: {', '.join(present)} set without {', '.join(missing)}")
    directory, artifact = environ[ENV_DIR], environ[ENV_ARTIFACT]
    if not directory.startswith("/") or "," in directory:
        raise DevShareError(f"{ENV_DIR}={directory!r}: an absolute directory without a comma")
    if not HEX64.fullmatch(artifact):
        raise DevShareError(f"{ENV_ARTIFACT}: not 64 lowercase hex")
    for key in (ENV_LIBRARY, ENV_KTEST):
        if not HEX64.fullmatch(environ[key]):
            raise DevShareError(f"{key}: not 64 lowercase hex")
    try:
        files = json.loads(environ[ENV_FILES])
    except json.JSONDecodeError as e:
        raise DevShareError(f"{ENV_FILES}: not JSON ({e})") from None
    if not isinstance(files, dict) or not all(
            isinstance(k, str) and FILE_NAME.fullmatch(k) and isinstance(v, str) and HEX64.fullmatch(v)
            for k, v in files.items()):
        raise DevShareError(f"{ENV_FILES}: not an object of file name -> 64 lowercase hex")
    attach_only = environ.get(ENV_ATTACH_ONLY)
    if attach_only not in (None, "1"):
        raise DevShareError(f"{ENV_ATTACH_ONLY}={attach_only!r}: only 1")
    return DevShare(directory, artifact, dict(files), environ[ENV_LIBRARY], environ[ENV_KTEST], attach_only == "1")


def stamp(artifact_sha256: str, library_sha256: str, ktest_sha256: str) -> dict[str, str]:
    """STAMP_FILE's content for a segment of that artifact published by that build."""
    return {"artifact_sha256": artifact_sha256, "ktest_sha256": ktest_sha256, "library_sha256": library_sha256}


def stamped(share: DevShare) -> bool:
    """The share directory's STAMP_FILE names this share's artifact and build: a segment published there by another
    artifact or build (a GPU0 run of another worktree replaced it, or `share drop` removed it) is never attached."""
    try:
        got = json.loads((Path(share.directory) / STAMP_FILE).read_text())
    except (OSError, ValueError):
        return False
    return got == stamp(share.artifact_sha256, share.library_sha256, share.ktest_sha256)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_build(share: DevShare, library: Path, ktest: Path) -> None:
    """The libraries this process loads (`library`, libcere_executor.so; `ktest`, libcere_ktest.so) are the build
    native_dev stamped on the segment (ENV_LIBRARY, ENV_KTEST); a difference, or a missing file, fails every load on
    the share (it never falls back)."""
    for path, want, key in ((library, share.library_sha256, ENV_LIBRARY), (ktest, share.ktest_sha256, ENV_KTEST)):
        try:
            got = file_sha256(path)
        except OSError as e:
            raise DevShareError(f"{path}: cannot hash it ({e.strerror})") from None
        if got != want:
            raise DevShareError(f"{path.name}: sha256 differs from {key} (the build stamped on the dev-loop segment)")


def creating_class(frame, base: type) -> str | None:
    """`module.Class` of the innermost `base` (unittest.TestCase) subclass running at `frame` or one of its callers:
    a classmethod's `cls` (setUpClass) or a method's `self`. None when there is none (the load is fresh)."""
    while frame is not None:
        local = frame.f_locals
        for value in (local.get("cls"), type(local.get("self"))):
            if isinstance(value, type) and issubclass(value, base):
                return f"{value.__module__}.{value.__qualname__}"
        frame = frame.f_back
    return None


def share_for(share: Callable[[], DevShare | None], cls: str | None) -> DevShare | None:
    """The share a load from class `cls` is created on: `share()` for a SHARE_CLASSES class, else None (fresh; `share`
    is not called, so a FRESH class's load never reads the share env, the build or the stamp)."""
    return share() if cls in SHARE_CLASSES else None


class Loads:
    """The loads in flight, handle -> (ALLOCATE start, the share it was created on or None, its class), from create
    to seal or release. Seal and release drop the handle's entry, so no later model (a new handle, or the same handle
    value reused) inherits a share decision or a clock."""

    def __init__(self) -> None:
        self._loads: dict[int, tuple[float, DevShare | None, str | None]] = {}

    def begin(self, handle: int, started: float, share: DevShare | None, cls: str | None) -> None:
        if handle in self._loads:
            raise DevShareError(f"handle {handle} is already loading")
        self._loads[handle] = (started, share, cls)

    def end(self, handle: int, now: float) -> tuple[float, DevShare | None, str | None]:
        return self._loads.pop(handle, (now, None, None))

    def release(self, handle: int) -> None:
        self._loads.pop(handle, None)


def load_mode(share: DevShare | None, covered: bool) -> str:
    """How a sealed load got its PLE: `attach` (covered by the published segment), `create` (on the share, the
    segment created here) or `fresh` (no share)."""
    return "attach" if covered else "create" if share is not None else "fresh"


def load_line(mode: str, ms: float, cls: str | None) -> str:
    """The per-create stderr line native_dev.py collects (`oracle_loads` of the report): the mode, the time and the
    creating class only (`-` for none or a name outside CLASS_NAME)."""
    if mode not in MODES:
        raise ValueError(f"load mode {mode!r}")
    name = cls if cls is not None and CLASS_NAME.fullmatch(cls) else "-"
    return f"{LOAD_PREFIX}attach={int(mode == 'attach')} ms={int(ms)} mode={mode} class={name}"


def resident_views(by_file: Mapping[str, list], digests: Mapping[str, str],
                   resident: Callable[[int, int], memoryview | None], *, attach_only: bool) -> dict[str, dict]:
    """{file: {tensor name: its resident bytes}} for every file of `by_file` ({file: [(tensor index, entry)]}, entry
    with name, placement, nbytes) whose tensors are all HOST_PINNED and which has a digest; `resident(index, nbytes)`
    is `cere_ktest_model_host_resident` (None = UNSUPPORTED). The first answer decides: None means the region is
    being created here (or is private), so nothing is covered and every file is put ({}); a None after a covered
    tensor fails closed (a model's region is one kind). `attach_only` (ENV_ATTACH_ONLY): a None, or no file to cover,
    fails."""
    out: dict[str, dict] = {}
    covered = False
    for file in sorted(by_file):
        items = by_file[file]
        if file not in digests or not all(e.placement == PLACEMENT_HOST_PINNED for _, e in items):
            continue
        views = {}
        for index, entry in items:
            view = resident(index, entry.nbytes)
            if view is None:
                if covered:
                    raise DevShareError(f"{file}: {entry.name} is not resident in the attached segment")
                if attach_only:
                    raise DevShareError("attach only: the published dev-loop segment does not hold this model's "
                                        "region (`native_dev.py share drop`, then a GPU0 run creates it)")
                return {}
            if memoryview(view).nbytes != entry.nbytes:
                raise DevShareError(f"{entry.name}: resident view of {memoryview(view).nbytes} bytes, "
                                    f"the desc says {entry.nbytes}")
            covered = True
            views[entry.name] = view
        out[file] = views
    if attach_only and not out:
        raise DevShareError("attach only: no weights file of the model is covered by the dev-loop segment")
    return out


def check_file(path: Path, digest: str, views: Mapping[str, memoryview]) -> None:
    """sha256(the prefix `[0, 8 + n)` of safetensors file `path` || `views` in data-offset order) == `digest`, where
    `views` ({tensor name: bytes}) holds exactly the header's tensors, each of its header size, tiling the data region
    from 0 with no gap, and the file is 8 + n + the data region long. Hashes the views in place (hashlib releases the
    GIL, so one thread per file runs in parallel)."""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise DevShareError(f"{path}: shorter than a safetensors length prefix")
        (n,) = struct.unpack("<Q", head)
        header_bytes = f.read(n)
        size = os.fstat(f.fileno()).st_size
    if len(header_bytes) != n:
        raise DevShareError(f"{path}: header of {n} bytes truncated")
    header = json.loads(header_bytes)
    spans = sorted((v["data_offsets"][0], v["data_offsets"][1], k) for k, v in header.items() if k != "__metadata__")
    names = {name for _, _, name in spans}
    if names != set(views):
        raise DevShareError(f"{path.name}: views {sorted(set(views) - names)} have no header tensor, header tensors "
                            f"{sorted(names - set(views))} have no view")
    at = 0
    for start, end, name in spans:
        if start != at:
            raise DevShareError(f"{path.name}: the views do not tile the data region ({name} starts at {start}, "
                                f"expected {at})")
        if memoryview(views[name]).nbytes != end - start:
            raise DevShareError(f"{path.name}: {name} view of {memoryview(views[name]).nbytes} bytes, header "
                                f"{end - start}")
        at = end
    if size != 8 + n + at:
        raise DevShareError(f"{path.name}: file size {size} != 8 + {n} header + {at} data bytes")
    h = hashlib.sha256(head + header_bytes)
    for _, _, name in spans:
        h.update(views[name])
    if h.hexdigest() != digest:
        raise DevShareError(f"{path.name}: sha256 of the prefix and the resident views differs from the manifest's")
