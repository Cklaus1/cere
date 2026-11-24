"""M3.2N G-2 (tasks/m32n-spec.md D5, R5; review M32NS-21): one shared PLE host segment, two executors in one process.

Suite `twocard_o47` (`native_dev.py test --suite twocard_o47`, or `twocard` with the D4 module; a G1 window: the
container sees exactly GPU0 then GPU1 as CUDA ordinals 0 and 1, native_dev.TWO_CARD_ATTRIBUTION, with the host's
SHARE_DIR mounted). This is R5's two-executor case of `test_model_host_share.py`; it lives in its own module because
the `model` suite's discovery (`test_model_*.py`) runs in the GPU0-only container. Device ids
`<ordinal>,host_share=<SHARE_DIR>,artifact=<sha256("ktest-twocard-o47")>` (a name no deployed artifact has): the GPU0
executor loads first and creates, fills and publishes the segment at its seal; the GPU1 executor then attaches to it
(its header checked at ALLOCATE, its content digest verified before the seal, every put compared with the segment).
One process thus holds two mappings of one segment, each pinned by its own device (M32NS-21). Inputs are synthetic
seeded token ids (no prompt text).

* `test_one_segment_two_mappings`: exactly one segment was published (by the GPU0 load; none by the GPU1 one), the
  GPU1 executor is attached (between ALLOCATE and the seal, `cere_ktest_model_host_resident` of a HOST_PINNED tensor
  gives its resident bytes, which then cover it; the creator gets UNSUPPORTED; after the seal the ABI refuses the
  call), and both mappings hold the sealed content digest.
* `test_prefill_decode_on_both_leaves_segment_unchanged`: PREFILL + DECODE on both executors at the same time (two
  threads); each card's segment digest is the sealed one before and after.
* `test_k4_reads_the_segment_on_both_cards`: each executor maps and registers the segment itself, and each card
  reads the PLE table through its own alias. On each card K4's FP8 gather of GATHER_ROWS rows spread over the whole
  table, through the segment's device alias, is bit-exact with the same gather over a device copy of those rows read
  through the CPU mapping, and the two cards' outputs are bit-exact with each other (and not all zero):

      int cere_ktest_segment_ple_gather(uint64_t model, const int64_t *ids, uint64_t n, uint16_t *out_alias,
                                        uint16_t *out_copy, int64_t *out_rows, cudaStream_t stream);

  ids host int64 [n] (row ids[i] mod table_rows), out_alias / out_copy host BF16 [n, 160], out_rows host [1].
* Release (class cleanup): the attacher first (the segment stays published while the creator maps it), then the
  creator (the last mapper unlinks it).
MemAvailable before and after the GPU1 load is reported (R5's L1 numbers are the live row's; here only recorded).
"""
from __future__ import annotations

import hashlib
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import oracle

ROW = "R5"
SHARE_DIR = Path("/dev/shm/cere-host")
ARTIFACT = hashlib.sha256(b"ktest-twocard-o47").hexdigest()
ORDINALS = (0, 1)
MAX_CONTEXT = 4_096
PROMPT_TOKENS = 1_024
MAX_NEW = 32
FINISHES = (1, 2)  # length, stop token
# Table rows the two-card K4 check gathers: seeded 62-bit ids (taken mod table_rows by the ktest), plus row 0.
GATHER_ROWS = 4_096
GATHER_SEED = 0x4B34


def device_id(ordinal: int) -> str:
    return f"{ordinal},host_share={SHARE_DIR},artifact={ARTIFACT}"


def published_segments() -> set[str]:
    """The published segment files in SHARE_DIR (not the `.lock` files, not an unpublished `.tmp`)."""
    return {p.name for p in SHARE_DIR.iterdir()
            if p.name.startswith("cere-host-") and not p.name.endswith((".lock", ".tmp"))}


def segment_digest(handle: int) -> tuple[str, int]:
    """(content digest hex over this executor's mapping, region bytes)."""
    digest = torch.zeros(32, dtype=torch.uint8)
    out = torch.zeros(2, dtype=torch.int64)
    oracle.call("cere_ktest_segment_digest", handle, digest, out)
    return bytes(digest.tolist()).hex(), int(out[0])


def host_resident_status(handle: int, index: int) -> int:
    """`cere_ktest_model_host_resident`'s status for tensor `index` (0: resident in an attached segment, and covered:
    never put). Valid between ALLOCATE and the seal only (a sealed model is CERE_ERR_ARGUMENT)."""
    return oracle.call_status("cere_ktest_model_host_resident", handle, index, torch.zeros(1, dtype=torch.int64))


def load_probed(ordinal: int) -> tuple[int, int, int]:
    """`oracle.create_model(MAX_CONTEXT)` on `ordinal` with the first HOST_PINNED tensor's resident status taken
    before the upload (that tensor is not put when it is resident): (ktest handle, status, tensor index)."""
    model_buf, entries, keep = oracle.a4_model_desc(MAX_CONTEXT)
    index = next(i for i, e in enumerate(entries) if e.placement == oracle.PLACEMENT_HOST_PINNED)
    handle = oracle.allocate(model_buf, device_id(ordinal))
    sealed = False
    try:
        status = host_resident_status(handle, index)

        def put(h: int, file: str, items: list) -> None:
            oracle._upload_file(h, file, [(i, e) for i, e in items if status != 0 or i != index])

        oracle.upload_and_seal(handle, entries, put_file=put)
        sealed = True
    finally:
        if not sealed:  # the pair is released before the error propagates (as oracle.create_model)
            oracle.release_model(handle)
    del keep
    return handle, status, index


def mem_available() -> int:
    """MemAvailable in bytes (/proc/meminfo is the host's)."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise oracle.OracleError("/proc/meminfo has no MemAvailable")


def synthetic_ids(n: int) -> list[int]:
    """`n` deterministic in-vocabulary ids (no text, no tokenizer)."""
    return [1_000 + (i * 7_919) % 30_000 for i in range(n)]


def generate(handle: int, ids: list[int]) -> tuple[int, int]:
    """`cere_ktest_generate` (PREFILL, then DECODE): (committed, finish)."""
    prompt = torch.tensor(ids, dtype=torch.int32)
    tokens = torch.full((MAX_NEW,), -1, dtype=torch.int32)
    scores = torch.full((MAX_NEW, 4), -1, dtype=torch.int32)
    result = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_generate", handle, prompt, len(ids), MAX_NEW, 0, 0, tokens, scores, result)
    return int(result[0]), int(result[1])


class TwoCardHostShare(unittest.TestCase):
    """The GPU0 executor creates the segment, the GPU1 executor attaches to it, in this process."""

    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() != len(ORDINALS):
            raise oracle.OracleError(f"{torch.cuda.device_count()} CUDA devices visible, not GPU0 + GPU1")
        if not (SHARE_DIR.is_dir() and os.access(SHARE_DIR, os.W_OK)):
            raise oracle.OracleError(f"{SHARE_DIR} is not a writable directory (native_dev.py mounts it)")
        before = published_segments()
        # Class cleanups run in reverse order and even when setUpClass fails later: the creator is released last.
        cls.creator, creator_status, cls.pinned_index = load_probed(0)
        cls.created = published_segments() - before
        cls.addClassCleanup(cls.release_creator)
        cls.sealed, cls.region_bytes = segment_digest(cls.creator)
        available = mem_available()
        cls.attacher, attacher_status, _ = load_probed(1)
        cls.second_load_bytes = available - mem_available()
        cls.after_attach = published_segments() - before
        cls.addClassCleanup(cls.release_attacher)
        cls.statuses = {0: creator_status, 1: attacher_status}

    @classmethod
    def release_attacher(cls):
        oracle.release_model(cls.attacher)
        kept = cls.created <= published_segments()
        oracle.report_row(ROW, "segment kept while the creator maps it", passed=kept)
        if not kept:
            raise oracle.OracleError(f"{sorted(cls.created)} unlinked at the attacher's release")

    @classmethod
    def release_creator(cls):
        oracle.release_model(cls.creator)
        left = cls.created & published_segments()
        oracle.report_row(ROW, "segment unlinked at the last release", passed=not left)
        if left:
            raise oracle.OracleError(f"{sorted(left)} left in {SHARE_DIR} after the last mapper's release")

    def test_one_segment_two_mappings(self):
        digests = {o: segment_digest(h) for o, h in ((0, self.creator), (1, self.attacher))}
        statuses = self.statuses
        oracle.report_row(ROW, "one segment two mappings", published=len(self.created),
                          published_after_attach=len(self.after_attach), region_bytes=self.region_bytes,
                          creator_resident_status=statuses[0], attacher_resident_status=statuses[1],
                          sealed_digest=self.sealed, digest_gpu0=digests[0][0], digest_gpu1=digests[1][0],
                          second_load_mem_available_drop=self.second_load_bytes,
                          passed=(len(self.created) == 1 and self.after_attach == self.created
                                  and statuses == {0: oracle.CERE_ERR_UNSUPPORTED, 1: 0}
                                  and digests[0] == digests[1] == (self.sealed, self.region_bytes)))
        self.assertEqual(len(self.created), 1, f"{ROW}: the GPU0 load did not publish exactly one segment")
        self.assertEqual(self.after_attach, self.created, f"{ROW}: the GPU1 load published a segment of its own")
        self.assertEqual(statuses[0], oracle.CERE_ERR_UNSUPPORTED, f"{ROW}: the creator reads as attached")
        self.assertEqual(statuses[1], 0, f"{ROW}: the GPU1 executor is not attached to the segment")
        self.assertEqual(digests[0], (self.sealed, self.region_bytes), f"{ROW}: GPU0's mapping changed")
        self.assertEqual(digests[1], (self.sealed, self.region_bytes), f"{ROW}: GPU1's mapping differs")

    def test_prefill_decode_on_both_leaves_segment_unchanged(self):
        handles = {0: self.creator, 1: self.attacher}
        before = {o: segment_digest(h)[0] for o, h in handles.items()}
        start = threading.Barrier(len(handles))

        def run(o: int) -> tuple[int, int]:
            start.wait()
            return generate(handles[o], synthetic_ids(PROMPT_TOKENS))

        with ThreadPoolExecutor(max_workers=len(handles)) as pool:
            futures = {o: pool.submit(run, o) for o in handles}
        results = {o: f.result() for o, f in futures.items()}
        after = {o: segment_digest(h)[0] for o, h in handles.items()}
        unchanged = all(before[o] == after[o] == self.sealed for o in handles)
        oracle.report_row(ROW, "prefill decode on both leaves segment unchanged",
                          **{f"committed{o}": results[o][0] for o in handles},
                          **{f"finish{o}": results[o][1] for o in handles}, passed=unchanged)
        for o in handles:
            committed, finish = results[o]
            self.assertIn(finish, FINISHES, f"{ROW}: card {o} finish {finish}")
            self.assertTrue(committed >= 2, f"{ROW}: card {o}: no DECODE step ran")
            self.assertEqual(before[o], self.sealed, f"{ROW}: card {o}'s mapping changed between seal and forward")
            self.assertEqual(after[o], before[o], f"{ROW}: a forward pass wrote the shared segment (card {o})")

    def test_k4_reads_the_segment_on_both_cards(self):
        g = torch.Generator().manual_seed(GATHER_SEED)
        ids = torch.randint(0, 1 << 62, (GATHER_ROWS,), generator=g, dtype=torch.int64)
        ids[0] = 0
        outs = {}
        for o, h in ((0, self.creator), (1, self.attacher)):
            alias = torch.full((GATHER_ROWS, 160), -1, dtype=torch.int16)
            copy = torch.zeros((GATHER_ROWS, 160), dtype=torch.int16)
            rows = torch.zeros(1, dtype=torch.int64)
            oracle.call("cere_ktest_segment_ple_gather", h, ids, GATHER_ROWS, alias, copy, rows)
            outs[o] = (alias, copy, int(rows[0]))
        per_card = {o: torch.equal(a, c) for o, (a, c, _) in outs.items()}
        across = torch.equal(outs[0][0], outs[1][0]) and outs[0][2] == outs[1][2]
        nonzero = bool(outs[0][0].any())
        oracle.report_row(ROW, "k4 reads the segment on both cards", rows=GATHER_ROWS, table_rows=outs[0][2],
                          alias_equals_copy0=per_card[0], alias_equals_copy1=per_card[1], cards_equal=across,
                          nonzero=nonzero, passed=all(per_card.values()) and across and nonzero)
        for o in outs:
            self.assertTrue(per_card[o], f"{ROW}: card {o}'s K4 gather through its alias differs from the CPU rows")
        self.assertTrue(across, f"{ROW}: the two cards' K4 gathers of the shared table differ")
        self.assertTrue(nonzero, f"{ROW}: the gathered rows are all zero")


if __name__ == "__main__":
    unittest.main()
