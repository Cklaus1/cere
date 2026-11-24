"""P8 (tasks/m23-spec.md P8, D9): rows above 4 on the real A4 model, per-row bit-identity at c = R.

Suite `model_p8` (`native_dev.py [--gpu 1] test --suite model_p8`; pooled, no memory or clock figure is read). The
support table's `max_sessions` is 16 (the n <= 16 rule, `GemmPlan::kMaxRows`), and a speculation round of more rows
than one verify call holds (`verify_rows` = min(16, 4 x max_sessions)) runs as consecutive groups
(`CudaDevice::speculate_rows`). Both are perf-only (D9): every row's tokens and scores equal the row decoded alone.

* `P8Rows.test_decode_rows16_bit_identical` (K3 at R = 16): ROWS16 sessions of seeded synthetic ids (no prompt
  text), each prefilled alone, decode greedily through `cere_ktest_forward_decode_rows` rounds of up to 16 rows
  (sessions join and leave at staggered steps so the rounds run every row count from 1 to 16); every session's
  scores equal the same session decoded alone (S = 1), bit for bit.
* `P8Speculation.test_spec_rows_grouped_bit_identical` (K4 at R in SPEC_ROWS): R drafting sessions decode
  SPEC_DECODE tokens each by speculate_rows rounds with caller drafts (the true next tokens, one made wrong per
  SPEC_WRONG_ROWS), so a round of R > 4 rows needs ceil(4 R / 16) verify groups; every committed token, the accept
  count and every score equal the plain greedy run.
* `P8Speculation.test_spec_rows_mtp_bit_identical`: the same with the pending MTP proposals as drafts (draft-extend
  per group); some drafts are accepted.

Native-vs-native gates only (no SGLang parity, no G1 result).
"""
from __future__ import annotations

import contextlib
import importlib
import unittest

import torch

import oracle

k8 = importlib.import_module("test_model_k8_forward")

ROW = "P8"
ROWS16 = 16
ROWS16_SEED = 20261013
ROWS16_STEPS = 48
# Distinct histories; (join, leave) staggered so the rounds run 1..16 rows.
ROWS16_PROMPT_LENS = (37, 211, 512, 1029, 64, 65, 129, 300, 17, 400, 777, 3, 255, 256, 1000, 90)
ROWS16_JOIN_LEAVE = tuple((s, ROWS16_STEPS - (s % 4) * 3) for s in range(ROWS16))
SPEC_ROWS = (6, 8)  # the -v10 candidates (m23 design note P8): 24 and 32 chain rows per round
SPEC_MAX_SESSIONS = max(SPEC_ROWS)
SPEC_SEED = 20261014
SPEC_DECODE = 96
SPEC_PROMPT_LENS = (37, 211, 512, 1029, 64, 300, 17, 777)
SPEC_WRONG_ROWS = k8.SPEC_WRONG_ROWS


def decode_rows(handle: int, prompts: list, join: list, leave: list) -> torch.Tensor:
    """`cere_ktest_forward_decode_rows` greedy without taps: int32 [S, T + 1, 4] scores, -1 where unwritten (as
    test_model_k8_forward.forward_decode_rows with taps=False)."""
    sessions, rounds = len(prompts), max(leave)
    flat = torch.tensor([t for p in prompts for t in p], dtype=torch.int32)
    lens = torch.tensor([len(p) for p in prompts], dtype=torch.int64)
    scores = torch.full((sessions * (rounds + 1), 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_forward_decode_rows", handle, sessions, flat, lens,
                torch.tensor(join, dtype=torch.int64), torch.tensor(leave, dtype=torch.int64), None, scores,
                None, 0, None, 0, None, 0, counts)
    return scores.view(sessions, rounds + 1, 4)


def speculate(sessions: list, tokens: list, drafts: list, max_commit: list):
    """One greedy speculate_rows call with caller drafts (test_model_k8_forward.session_speculate, greedy)."""
    rows = len(sessions)
    flat = torch.zeros((rows, 3), dtype=torch.int32)
    for r, d in enumerate(drafts):
        flat[r, :len(d)] = torch.tensor(d, dtype=torch.int32)
    status = torch.full((rows,), 1, dtype=torch.int32)
    results = torch.zeros((rows, 2), dtype=torch.int32)
    scores = torch.full((rows, 4, 4), -1, dtype=torch.int32)
    oracle.call("cere_ktest_session_speculate", torch.tensor(sessions, dtype=torch.int64), rows,
                torch.tensor(tokens, dtype=torch.int32), flat,
                torch.tensor([len(d) for d in drafts], dtype=torch.int32),
                torch.tensor(max_commit, dtype=torch.int32), status, results, scores)
    if status.tolist() != [0] * rows:
        raise oracle.OracleError(f"cere_ktest_session_speculate row status {status.tolist()}")
    return k8.Speculated(results[:, 0].tolist(), results[:, 1].tolist(), scores)


def speculate_mtp(sessions: list, tokens: list, draft_counts: list, max_commit: list):
    """One greedy speculate_rows call drafting from the pending MTP proposals (as
    test_model_k8_forward.session_speculate_mtp)."""
    rows = len(sessions)
    status = torch.full((rows,), 1, dtype=torch.int32)
    results = torch.zeros((rows, 2), dtype=torch.int32)
    scores = torch.full((rows, 4, 4), -1, dtype=torch.int32)
    oracle.call("cere_ktest_session_speculate_mtp", torch.tensor(sessions, dtype=torch.int64), rows,
                torch.tensor(tokens, dtype=torch.int32), torch.tensor(draft_counts, dtype=torch.int32),
                torch.tensor(max_commit, dtype=torch.int32), status, results, scores)
    if status.tolist() != [0] * rows:
        raise oracle.OracleError(f"cere_ktest_session_speculate_mtp row status {status.tolist()}")
    return k8.Speculated(results[:, 0].tolist(), results[:, 1].tolist(), scores)



class P8Rows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tc = oracle.text_config()
        cls.hidden = int(cls.tc["hidden_size"])
        cls.layers = int(cls.tc["num_hidden_layers"])
        model = k8.create_model(k8.MAX_CONTEXT, ROWS16)
        cls.addClassCleanup(k8.release_model, model.handle)
        k8.upload_and_seal(model)
        cls.handle = model.handle

    def test_decode_rows16_bit_identical(self):
        """K3 at R = 16: a batched decode step's row r equals session r's decode alone."""
        vocab = int(self.tc["vocab_size"])
        gen = torch.Generator().manual_seed(ROWS16_SEED)
        prompts = [k8.synthetic_ids(gen, vocab, n) for n in ROWS16_PROMPT_LENS]
        join = [j for j, _ in ROWS16_JOIN_LEAVE]
        leave = [lv for _, lv in ROWS16_JOIN_LEAVE]
        rows = decode_rows(self.handle, prompts, join, leave)
        widest = max(sum(j <= k < lv for j, lv in ROWS16_JOIN_LEAVE) for k in range(max(leave)))
        self.assertEqual(widest, ROWS16, f"{ROW} K3 R=16: the widest round holds {widest} rows")
        for s, ids in enumerate(prompts):
            decodes = leave[s] - join[s]
            solo = decode_rows(self.handle, [ids], [0], [decodes])
            case = f"K3 R=16 session {s} (join {join[s]}, leave {leave[s]})"
            oracle.check_bit_exact(self, ROW, f"{case} scores", rows[s, :decodes + 1], solo[0, :decodes + 1])
            self.assertTrue(bool((rows[s, decodes + 1:] == -1).all()),
                            f"{ROW} {case}: a score row past its last decode was written")
        oracle.report_row(ROW, "K3 R=16", sessions=ROWS16, widest=widest)


class P8Speculation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tc = oracle.text_config()
        cls.vocab = int(cls.tc["vocab_size"])
        model = k8.create_model(k8.SPEC_CONTEXT, SPEC_MAX_SESSIONS, drafts=k8.SPEC_DRAFT_TOKENS)
        cls.addClassCleanup(k8.release_model, model.handle)
        k8.upload_and_seal(model)
        cls.handle = model.handle
        gen = torch.Generator().manual_seed(SPEC_SEED)
        cls.prompts = [k8.synthetic_ids(gen, cls.vocab, n) for n in SPEC_PROMPT_LENS]
        cls.truth = [k8.plain_greedy(cls.handle, ids, SPEC_DECODE) for ids in cls.prompts]

    def drafting_sessions(self, stack: contextlib.ExitStack, rows: int, label: str) -> list:
        sessions = []
        for r in range(rows):
            session = stack.enter_context(k8.opened_session(self.handle, draft=True))
            prefill = k8.session_feed(session, self.prompts[r])[0].clone()
            oracle.check_bit_exact(self, ROW, f"{label} R={rows} session {r} prefill", prefill, self.truth[r].prefill)
            sessions.append(session)
        return sessions

    def test_spec_rows_grouped_bit_identical(self):
        """K4 at R > 4 (caller drafts): grouped verify calls commit exactly the plain greedy tokens and scores."""
        for rows in SPEC_ROWS:
            with self.subTest(rows=rows), contextlib.ExitStack() as stack:
                sessions = self.drafting_sessions(stack, rows, "K4")
                at = [0] * rows
                round_ = 0
                while any(i < SPEC_DECODE for i in at):
                    live = [r for r in range(rows) if at[r] < SPEC_DECODE]
                    tokens, drafts, max_commit, expect = [], [], [], []
                    for r in live:
                        i = at[r]
                        commit = min(4, SPEC_DECODE - i)
                        d = self.truth[r].tokens[i + 1:i + commit]
                        w = SPEC_WRONG_ROWS[(round_ + r) % len(SPEC_WRONG_ROWS)]
                        if w is not None and w < len(d):
                            d[w] = k8.wrong(d[w], self.vocab)
                            expect.append(w + 1)
                        else:
                            expect.append(len(d) + 1)
                        tokens.append(self.truth[r].tokens[i])
                        drafts.append(d)
                        max_commit.append(commit)
                    got = speculate([sessions[r] for r in live], tokens, drafts, max_commit)
                    for p, r in enumerate(live):
                        case = f"K4 R={rows} session {r} round {round_}"
                        c = got.committed[p]
                        self.assertEqual((c, got.accepted[p]), (expect[p], expect[p] - 1), f"{ROW} {case}: committed")
                        oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores[p, :c],
                                               self.truth[r].scores[at[r]:at[r] + c])
                        at[r] += c
                    round_ += 1
                oracle.report_row(ROW, f"K4 R={rows}", rounds=round_, committed=rows * SPEC_DECODE)

    def test_spec_rows_mtp_bit_identical(self):
        """K4m at R > 4: MTP-drafted grouped rounds commit exactly the plain greedy tokens and scores."""
        for rows in SPEC_ROWS:
            with self.subTest(rows=rows), contextlib.ExitStack() as stack:
                sessions = self.drafting_sessions(stack, rows, "K4m")
                at = [0] * rows
                round_, accepted = 0, 0
                while any(i < SPEC_DECODE for i in at):
                    live = [r for r in range(rows) if at[r] < SPEC_DECODE]
                    commits = [min(4, SPEC_DECODE - at[r]) for r in live]
                    got = speculate_mtp([sessions[r] for r in live], [self.truth[r].tokens[at[r]] for r in live],
                                        [c - 1 for c in commits], commits)
                    for p, r in enumerate(live):
                        case = f"K4m R={rows} session {r} round {round_}"
                        c, a = got.committed[p], got.accepted[p]
                        self.assertTrue(1 <= c <= commits[p] and a == c - 1,
                                        f"{ROW} {case}: committed {c} accepted {a}")
                        oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores[p, :c],
                                               self.truth[r].scores[at[r]:at[r] + c])
                        at[r] += c
                        accepted += a
                    round_ += 1
                self.assertTrue(accepted > 0, f"{ROW} K4m R={rows}: no MTP draft accepted")
                oracle.report_row(ROW, f"K4m R={rows}", rounds=round_, committed=rows * SPEC_DECODE,
                                  accepted=accepted, accept_length=round(rows * SPEC_DECODE / round_, 4))


if __name__ == "__main__":
    unittest.main()
