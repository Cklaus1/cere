"""O47 (tasks/build-loop-ple.md, maintainer delegate decision 2026-10-12): the shared HOST_PINNED segment stays
unwritten through a full forward pass on the real A4 model, GPU0, container only.

Suite `model` (`native_dev.py test --suite model`; `--suite model_host` runs this module alone). The container mounts
the host's SHARE_DIR (native_dev.py HOST_SHARE_DIR). The model is created with the device-id options
`host_share=<SHARE_DIR>,artifact=<sha256("ktest")>` (a name no deployed artifact has), so this process creates,
fills and publishes its own segment and, as its last mapper, unlinks it at release.

Where `cudaDevAttrHostRegisterReadOnlySupported` is 0 (GPU0 on this host) the region is pinned on the `rw_mprotect`
path: its CPU mapping is PROT_READ after the pin, but the GPU-side write protection of `cudaHostRegisterReadOnly` is
gone, so a kernel that wrote a HOST_PINNED tensor would change the shared bytes under every other lane. This module
pins that none does, and that the check would see it:

* `test_forward_leaves_segment_unchanged`: the segment's content digest (recomputed over the region through this
  process's mapping) after a `cere_ktest_generate` of PROMPT_TOKENS synthetic ids (PREFILL) and up to MAX_NEW tokens
  (DECODE steps; at least 2 committed, so at least one DECODE ran) equals the digest before it, which equals the one
  taken right after the seal (publish).
* `test_digest_catches_device_write`: a test-only device write of one byte through the region's device alias
  (`cere_ktest_segment_poke`) changes the digest; writing the old byte back restores it. On the `read_only` path the
  poke is refused (CERE_ERR_UNSUPPORTED: the registration itself makes a GPU write fault) and the case is recorded as
  such, not run.

Native entry points (k8_ktest.cpp; `oracle.call` appends the stream; 0 = success, otherwise a `cere_status`; "host"
buffers are CPU memory):

    int cere_ktest_model_create_device(const cere_model_desc *desc /*host*/, uint64_t memory_budget_bytes,
                                       const char *device_id /*host, NUL-terminated*/, uint64_t *out_model /*host*/,
                                       cudaStream_t stream);
    int cere_ktest_segment_digest(uint64_t model, uint8_t *out_digest /*host [32]*/, uint64_t *out /*host [2]*/,
                                  cudaStream_t stream);
        out[0] region bytes, out[1] 1 = rw_mprotect, 0 = read_only. CERE_ERR_ARGUMENT without a segment.
    int cere_ktest_segment_poke(uint64_t model, uint64_t offset, uint64_t value, uint64_t *out /*host [2]*/,
                                cudaStream_t stream);
        Test-only: one GPU thread stores the low byte of `value` at region byte `offset` through the device alias;
        out[0] the byte before, out[1] the byte read back through the CPU mapping after. CERE_ERR_UNSUPPORTED on the
        read_only path.

Token IDs stay in memory; the report rows hold digests, byte counts and booleans only.
"""

from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path

import torch

import oracle

ROW = "O47"
SHARE_DIR = Path("/dev/shm/cere-host")
ARTIFACT = hashlib.sha256(b"ktest").hexdigest()
DEVICE_ID = f"0,host_share={SHARE_DIR},artifact={ARTIFACT}"
MAX_CONTEXT = 4_096
PROMPT_TOKENS = 1_024
MAX_NEW = 32
FINISH_LENGTH = 1
FINISH_STOP_TOKEN = 2


def published_segments() -> set[str]:
    """The published segment files in SHARE_DIR (not the `.lock` files, not an unpublished `.tmp`)."""
    return {p.name for p in SHARE_DIR.iterdir()
            if p.name.startswith("cere-host-") and not p.name.endswith((".lock", ".tmp"))}


def segment_digest(handle: int) -> tuple[str, int, bool]:
    """(content digest hex, region bytes, registered on the rw_mprotect path)."""
    digest = torch.zeros(32, dtype=torch.uint8)
    out = torch.zeros(2, dtype=torch.int64)
    oracle.call("cere_ktest_segment_digest", handle, digest, out)
    return bytes(digest.tolist()).hex(), int(out[0]), bool(out[1])


def poke(handle: int, offset: int, value: int) -> tuple[int, int]:
    """(byte before, byte read back after) of the device write of `value` at region byte `offset`."""
    out = torch.full((2,), -1, dtype=torch.int64)
    oracle.call("cere_ktest_segment_poke", handle, offset, value, out)
    return int(out[0]), int(out[1])


def synthetic_ids(n: int) -> list[int]:
    """`n` deterministic in-vocabulary ids (no text, no tokenizer)."""
    return [1_000 + (i * 7_919) % 30_000 for i in range(n)]


class O47HostShare(unittest.TestCase):
    """One shared-mode model; the segment is this process's own (created, published, unlinked at release)."""

    @classmethod
    def setUpClass(cls):
        if not (SHARE_DIR.is_dir() and os.access(SHARE_DIR, os.W_OK)):
            raise oracle.OracleError(f"{SHARE_DIR} is not a writable directory (native_dev.py mounts it)")
        before = published_segments()
        cls.handle = oracle.create_model(MAX_CONTEXT, device_id=DEVICE_ID)
        cls.created = published_segments() - before
        # A class cleanup runs even when the rest of setUpClass fails, so the segment is never left mapped.
        cls.addClassCleanup(cls.release_and_check_unlinked)
        cls.sealed_digest, cls.region_bytes, cls.rw = segment_digest(cls.handle)

    @classmethod
    def release_and_check_unlinked(cls):
        oracle.release_model(cls.handle)
        left = cls.created & published_segments()
        oracle.report_row(ROW, "segment unlinked at release", passed=not left)
        if left:
            raise oracle.OracleError(f"{sorted(left)} left in {SHARE_DIR} after the last mapper's release")

    def test_forward_leaves_segment_unchanged(self):
        self.assertEqual(len(self.created), 1, f"{ROW}: the model did not publish exactly one segment")
        before, region, rw = segment_digest(self.handle)
        prompt = torch.tensor(synthetic_ids(PROMPT_TOKENS), dtype=torch.int32)
        tokens = torch.full((MAX_NEW,), -1, dtype=torch.int32)
        scores = torch.full((MAX_NEW, 4), -1, dtype=torch.int32)
        result = torch.zeros(3, dtype=torch.int64)
        oracle.call("cere_ktest_generate", self.handle, prompt, PROMPT_TOKENS, MAX_NEW, 0, 0, tokens, scores, result)
        committed, finish = int(result[0]), int(result[1])
        after, _, _ = segment_digest(self.handle)
        oracle.report_row(ROW, "forward leaves segment unchanged", region_bytes=region,
                          registration="rw_mprotect" if rw else "read_only", committed=committed, finish=finish,
                          sealed_digest=self.sealed_digest, before=before, after=after,
                          passed=before == after == self.sealed_digest)
        self.assertIn(finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW}: finish {finish}")
        self.assertTrue(committed >= 2, f"{ROW}: {committed} tokens committed, no DECODE step ran")
        self.assertEqual(before, self.sealed_digest, f"{ROW}: the segment changed between the seal and the forward")
        self.assertEqual(after, before, f"{ROW}: the forward pass wrote the shared segment")

    def test_digest_catches_device_write(self):
        offset = self.region_bytes // 2
        if not self.rw:
            with self.assertRaises(oracle.OracleError):
                poke(self.handle, offset, 0)
            oracle.report_row(ROW, "device write caught", registration="read_only", poke_refused=True, passed=True)
            return
        before, _, _ = segment_digest(self.handle)
        old, _ = poke(self.handle, offset, 0)
        try:
            new = old ^ 0xFF
            seen_old, seen_new = poke(self.handle, offset, new)
            poked, _, _ = segment_digest(self.handle)
        finally:
            restored_old, restored = poke(self.handle, offset, old)
        after, _, _ = segment_digest(self.handle)
        oracle.report_row(ROW, "device write caught", registration="rw_mprotect", offset=offset, before=before,
                          poked=poked, after=after, cpu_sees_write=seen_new == new,
                          passed=seen_new == new and poked != before and after == before)
        self.assertEqual(seen_old, 0, f"{ROW}: the first poke's byte did not stick")
        self.assertEqual(seen_new, new, f"{ROW}: the device write is not visible through the CPU mapping")
        self.assertEqual(restored_old, new)
        self.assertEqual(restored, old)
        self.assertNotEqual(poked, before, f"{ROW}: the digest missed a device write to the segment")
        self.assertEqual(after, before, f"{ROW}: restoring the byte did not restore the digest")


if __name__ == "__main__":
    unittest.main()
