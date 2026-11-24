// SelfTune ST4 S3c: the block-diffusion schedule of family sdar (executor/cuda/src/model/sdar/schedule.h) against
// a fake forward that keeps one KV row per position and checks, at every forward, that each row it attends below
// its own positions is a committed row holding the session's fed token there (condition 3: no tail slot or stale
// row is ever read as committed). Covers the plan's DECODE POLICY (prefill of full blocks, one denoise per masked
// row, then one commit), the unmask choice (fewest ties, lowest row), handing out a block one token per call,
// condition 8 (the block's KV is committed in full whatever the host stops at), a cancelled or failed ticket
// mid-block (position unchanged, the next block rewrites the tail before reading it), chunking and capacity.
#include "check.h"
#include "model/sdar/schedule.h"

#include <cstdint>
#include <vector>

namespace {

using cere::sdar::kScheduleBlock;
using cere::sdar::RowPick;
using cere::sdar::Session;

constexpr uint32_t kMask = 999999;

enum class Kind { kEmpty, kCommitted, kTail };

struct FakeForward : cere::sdar::Forward {
    const Session *session = nullptr;
    std::vector<uint32_t> row_token;  // the token whose K/V the row holds
    std::vector<Kind> kind;
    int prefills = 0, denoises = 0, commits = 0;
    int fail_at = -1;  // the forward call (0-based, every kind) that fails
    int calls = 0;
    // ties of row r in a denoise step (cycled by step); the default makes row 1 win, then 3, 2, 0
    std::vector<uint32_t> ties = {3, 1, 2, 2};

    explicit FakeForward(uint64_t rows) : row_token(rows, 0), kind(rows, Kind::kEmpty) {}

    // Every row below `pos` is committed and holds the fed token at that position.
    void check_history(uint64_t pos) {
        for (uint64_t p = 0; p < pos; ++p) {
            CHECK(kind[p] == Kind::kCommitted);
            CHECK(p < session->fed.size() && row_token[p] == session->fed[p]);
        }
    }
    bool fail() { return calls++ == fail_at; }

    cere_status prefill(const uint32_t *tokens, uint64_t n, uint64_t pos) override {
        CHECK(n % kScheduleBlock == 0 && pos % kScheduleBlock == 0 && n > 0);
        check_history(pos);
        if (fail()) return CERE_ERR_INTERNAL;
        for (uint64_t i = 0; i < n; ++i) {
            row_token[pos + i] = tokens[i];
            kind[pos + i] = Kind::kCommitted;
        }
        ++prefills;
        return CERE_OK;
    }
    cere_status denoise(const uint32_t *block, const bool *masked, uint64_t base, RowPick *picks,
                        int32_t *index) override {
        CHECK(base % kScheduleBlock == 0);
        check_history(base);
        if (fail()) return CERE_ERR_INTERNAL;
        uint64_t sum = base;
        for (uint64_t r = 0; r < kScheduleBlock; ++r) {
            row_token[base + r] = block[r];  // the tail slots: written, never committed
            kind[base + r] = Kind::kTail;
            CHECK(masked[r] == (block[r] == kMask));
            sum = sum * 31 + block[r];
        }
        for (uint64_t r = 0; r < kScheduleBlock; ++r) {
            picks[r].id = uint32_t((sum + r * 7) % 1000);
            picks[r].ties = ties[r];
            picks[r].runner_up = picks[r].id + 1;
            picks[r].logit = float(r);
        }
        *index = cere::sdar::select_unmask(picks, masked, int(kScheduleBlock));
        ++denoises;
        return CERE_OK;
    }
    cere_status commit(const uint32_t *block, uint64_t base) override {
        check_history(base);
        if (fail()) return CERE_ERR_INTERNAL;
        for (uint64_t r = 0; r < kScheduleBlock; ++r) {
            CHECK(block[r] != kMask);
            row_token[base + r] = block[r];
            kind[base + r] = Kind::kCommitted;
        }
        ++commits;
        return CERE_OK;
    }
};

struct Fixture {
    Session s;
    FakeForward f;
    explicit Fixture(uint64_t capacity) : f(cere::sdar::kv_rows(capacity)) {
        s.capacity = capacity;
        f.session = &s;
    }
    cere_status step(const std::vector<uint32_t> &tokens, uint64_t chunk, uint64_t *consumed,
                     cere_token_score *out) {
        return cere::sdar::step(f, s, tokens.data(), tokens.size(), chunk, kMask, consumed, out);
    }
};

void test_select_unmask() {
    RowPick p[4] = {};
    bool masked[4] = {true, true, true, true};
    p[0].ties = 2;
    p[1].ties = 1;
    p[2].ties = 1;
    p[3].ties = 3;
    CHECK_EQ(cere::sdar::select_unmask(p, masked, 4), 1);  // fewest ties, the lowest row among equals
    masked[1] = false;
    CHECK_EQ(cere::sdar::select_unmask(p, masked, 4), 2);
    masked[2] = false;
    CHECK_EQ(cere::sdar::select_unmask(p, masked, 4), 0);
    bool none[4] = {};
    CHECK_EQ(cere::sdar::select_unmask(p, none, 4), -1);
}

void test_prompt_block_and_hand_out() {
    Fixture x(64);
    uint64_t consumed = 0;
    cere_token_score out{};
    // 6 prompt tokens: one prefill of the first block, then a decode block at 4 with 2 known rows and 2 masked.
    CHECK_EQ(x.step({10, 11, 12, 13, 14, 15}, 0, &consumed, &out), CERE_OK);
    CHECK_EQ(consumed, 6);
    CHECK_EQ(x.f.prefills, 1);
    CHECK_EQ(x.f.denoises, 2);
    CHECK_EQ(x.f.commits, 1);
    CHECK_EQ(x.s.kv, 8);
    CHECK_EQ(x.s.pending.size(), 2);
    CHECK_EQ(out.id, x.s.pending.front());
    // The committed block holds the prompt rows then the generated ones.
    CHECK_EQ(x.f.row_token[4], 14);
    CHECK_EQ(x.f.row_token[6], x.s.pending[0]);
    CHECK_EQ(x.f.row_token[7], x.s.pending[1]);
    // Feeding the pending front back hands out the next token without a forward.
    const uint32_t second = x.s.pending[1];
    CHECK_EQ(x.step({out.id}, 0, &consumed, &out), CERE_OK);
    CHECK_EQ(consumed, 1);
    CHECK_EQ(out.id, second);
    CHECK_EQ(x.f.denoises, 2);
    // The block is used up: the next fed token opens a fully masked block at 8 (4 denoises + 1 commit).
    CHECK_EQ(x.step({out.id}, 0, &consumed, &out), CERE_OK);
    CHECK_EQ(x.f.denoises, 6);
    CHECK_EQ(x.f.commits, 2);
    CHECK_EQ(x.s.kv, 12);
    CHECK_EQ(x.s.pending.size(), 4);
    CHECK_EQ(out.id, x.s.pending.front());
}

void test_unmask_order() {
    // Ties {3, 1, 2, 2}: row 1 first, then rows 2 and 3 (2 ties, lowest row first), then row 0; the block's tokens are
    // each row's pick at the step that unmasked it, so a schedule that unmasked another row would commit other ids.
    Fixture x(64);
    uint64_t consumed = 0;
    cere_token_score out{};
    CHECK_EQ(x.step({1, 2, 3, 4}, 0, &consumed, &out), CERE_OK);
    CHECK_EQ(x.f.denoises, 4);
    // Replay the fake's picks: step i unmasks order[i] with the pick of the block as it stood.
    const int order[4] = {1, 2, 3, 0};
    std::vector<uint32_t> block(4, kMask);
    for (int i = 0; i < 4; ++i) {
        uint64_t sum = 4;
        for (uint32_t t : block) sum = sum * 31 + t;
        block[order[i]] = uint32_t((sum + uint64_t(order[i]) * 7) % 1000);
    }
    CHECK(x.s.pending == block);
}

void test_stop_commits_the_whole_block() {
    // Condition 8: the host ends the output at a stop token or max_new_tokens (host/api.cpp commit); the device has
    // already committed the block's KV in full, and the output's end never shortens the schedule.
    Fixture x(64);
    uint64_t consumed = 0;
    cere_token_score out{};
    CHECK_EQ(x.step({1, 2, 3, 4, 5}, 0, &consumed, &out), CERE_OK);  // block at 4: 1 known, 3 masked
    CHECK_EQ(x.f.denoises, 3);
    CHECK_EQ(x.f.commits, 1);
    CHECK_EQ(x.s.kv, 8);  // the block's 4 rows committed although the host takes 1 token and stops
    for (uint64_t p = 4; p < 8; ++p) CHECK(x.f.kind[p] == Kind::kCommitted);
}

void test_cancel_mid_block() {
    // Condition 3: a ticket that ends mid-block (cancelled, stopped, or truncated) leaves the host's position (fed)
    // where it was; the session's next call rolls the device back to the block holding the last fed token, and the
    // fake's check_history proves no row past the fed tokens is read as committed before a forward rewrites it.
    Fixture x(64);
    uint64_t consumed = 0;
    cere_token_score out{};
    CHECK_EQ(x.step({1, 2, 3, 4, 5, 6, 7, 8}, 0, &consumed, &out), CERE_OK);  // prefill 8, block at 8 fully masked
    CHECK_EQ(x.s.kv, 12);
    CHECK_EQ(x.step({out.id}, 0, &consumed, &out), CERE_OK);  // the host commits one generated token: fed 9
    CHECK_EQ(x.s.fed.size(), 9);
    CHECK_EQ(x.s.kv, 12);
    // Cancelled here. The session continues with 3 new tokens (a new ticket): fed 12, the block at 8 is prefilled
    // from the fed tokens (rows 9..11 were generated tokens the host never fed: rewritten), then a block at 12.
    const int prefills = x.f.prefills;
    CHECK_EQ(x.step({70, 71, 72}, 0, &consumed, &out), CERE_OK);
    CHECK_EQ(consumed, 3);
    CHECK_EQ(x.f.prefills, prefills + 1);
    CHECK_EQ(x.s.kv, 16);
    for (uint64_t p = 8; p < 12; ++p) CHECK_EQ(x.f.row_token[p], x.s.fed[p]);
    // The host feeds a token other than the pending front: the buffer is dropped and a block at 12 opens with it.
    CHECK_EQ(x.step({555}, 0, &consumed, &out), CERE_OK);
    CHECK_EQ(x.s.fed.size(), 13);
    CHECK_EQ(x.f.row_token[12], 555);
    CHECK_EQ(x.s.kv, 16);
}

void test_failure_mid_block() {
    // A forward that fails mid-block: nothing of the call is consumed (fed and the KV position as at its start),
    // the session is broken (later steps INTERNAL), and no tail row became committed.
    for (int fail_at = 0; fail_at < 5; ++fail_at) {
        Fixture x(64);
        uint64_t consumed = 0;
        cere_token_score out{};
        CHECK_EQ(x.step({1, 2, 3, 4}, 0, &consumed, &out), CERE_OK);  // prefill 0..3, a fully masked block at 4
        const uint64_t fed = x.s.fed.size();
        const uint64_t kv = x.s.kv;
        CHECK_EQ(x.step({out.id}, 0, &consumed, &out), CERE_OK);  // hands out the pending token, no forward
        CHECK_EQ(x.step({out.id}, 0, &consumed, &out), CERE_OK);
        CHECK_EQ(x.step({out.id}, 0, &consumed, &out), CERE_OK);
        const uint64_t fed_before = x.s.fed.size();
        const uint64_t kv_before = x.s.kv;
        CHECK_EQ(fed_before, fed + 3);
        CHECK_EQ(kv_before, kv);
        x.f.fail_at = x.f.calls + fail_at;  // one of the 4 denoises of the next block (at 8), or its commit
        CHECK_EQ(x.step({out.id}, 0, &consumed, &out), CERE_ERR_INTERNAL);
        CHECK(x.s.broken);
        CHECK_EQ(x.s.fed.size(), fed_before);
        CHECK_EQ(x.s.kv, kv_before);
        CHECK(x.s.pending.empty());
        for (uint64_t p = kv_before; p < kv_before + kScheduleBlock; ++p) CHECK(x.f.kind[p] != Kind::kCommitted);
        CHECK_EQ(x.step({1}, 0, &consumed, &out), CERE_ERR_INTERNAL);
    }
    // A failed prefill restores the fed tokens too.
    Fixture x(64);
    uint64_t consumed = 0;
    cere_token_score out{};
    x.f.fail_at = 0;
    CHECK_EQ(x.step({1, 2, 3, 4, 5}, 0, &consumed, &out), CERE_ERR_INTERNAL);
    CHECK_EQ(x.s.fed.size(), 0);
    CHECK_EQ(x.s.kv, 0);
}

void test_chunk_and_capacity() {
    Fixture x(16);
    uint64_t consumed = 0;
    cere_token_score out{};
    out.id = 4242;
    // A prompt longer than the chunk: the first call consumes the chunk and scores nothing.
    CHECK_EQ(x.step({1, 2, 3, 4, 5, 6}, 4, &consumed, &out), CERE_OK);
    CHECK_EQ(consumed, 4);
    CHECK_EQ(out.id, 4242);
    CHECK_EQ(x.s.kv, 4);
    CHECK_EQ(x.f.denoises, 0);
    CHECK_EQ(x.step({5, 6}, 4, &consumed, &out), CERE_OK);
    CHECK_EQ(consumed, 2);
    CHECK_EQ(x.s.kv, 8);
    // Past the session's context: CAPACITY, nothing consumed.
    Fixture y(5);
    CHECK_EQ(y.step({1, 2, 3, 4, 5, 6}, 0, &consumed, &out), CERE_ERR_CAPACITY);
    CHECK_EQ(y.s.fed.size(), 0);
    CHECK(!y.s.broken);
    // The KV rows of a context cover every block that opens below it, whole.
    CHECK_EQ(cere::sdar::kv_rows(5), 8);
    CHECK_EQ(cere::sdar::kv_rows(8), 12);
    // A malformed call.
    CHECK_EQ(x.step({}, 0, &consumed, &out), CERE_ERR_ARGUMENT);
}

}  // namespace

int main() {
    test_select_unmask();
    test_prompt_block_and_hand_out();
    test_unmask_order();
    test_stop_commits_the_whole_block();
    test_cancel_mid_block();
    test_failure_mid_block();
    test_chunk_and_capacity();
    return cere::test::finish("test_sdar_schedule");
}
