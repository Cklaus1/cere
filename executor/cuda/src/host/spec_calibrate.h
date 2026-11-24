// O95 / m23 design note SC-01: the adaptive draft policy's fixed cost tables, measured on the loaded library at
// cere_model_seal instead of compiled in (a compiled table goes stale silently when a library changes the round time,
// DG-01). Host-only, over the Device interface: the rounds run through Device::speculate_rows exactly as the tickets'
// rounds do (the levers' verify graphs, commit, draft-extend and MTP steps) and are timed on the host as spec_step
// times them.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "host/device.h"
#include "host/spec_lookup.h"

namespace cere::host {

// Wall ms of single-row speculation rounds by window t = 1 + drafts. `mtp[t]` (t = 1..kMtpWindows): rounds of t - 1
// pending MTP proposals, greedy, accepted as the model accepts them (model_k8 "A3 window t"). `chain[t]` (t =
// kMtpWindows..the device's longest chain + 1): rounds of t - 1 caller drafts that the round accepted in full (it
// committed t tokens; model_k8 "A3D window t", the dearest round of the window).
struct SpecRoundTimes {
    std::array<std::vector<double>, SpecCosts::kMaxWindow + 1> mtp;
    std::array<std::vector<double>, SpecCosts::kMaxWindow + 1> chain;
};

struct SpecCalibration {
    static constexpr uint32_t kMtpWindows = DraftPolicy::kMaxDrafts + 1;
    static constexpr uint32_t kRounds = 6;  // timed rounds per window and kind (each kind's first round is not timed)
    static constexpr uint32_t kMaxCycles = 3 * (kRounds + 1);  // round-robin cycles before the series must be full
    // Chain cycles the greedy stream recorded for the chain session covers: kRounds + 1 plus slack for rounds that a
    // verify does not accept in full (they are not timed, but still advance the session along the stream).
    static constexpr uint32_t kChainCycles = kRounds + 3;
    // The synthetic prompt: kPeriod seeded ordinary ids repeated to kPromptTokens (a predictable greedy continuation,
    // so the MTP proposals are accepted); generated at load, never on disk.
    static constexpr uint32_t kPeriod = 16;
    static constexpr uint32_t kPromptTokens = 4 * kPeriod;
    static constexpr uint64_t kPromptSeed = 0x9e3779b97f4a7c15ull;
};

// The fixed table of `times` (m29a design note A3 and DraftAccept's scaling): window t <= kMtpWindows costs the
// median of mtp[t]; a deeper window t <= max_window costs the median of chain[t] scaled by the medians' ratio at
// kMtpWindows (mtp / chain), so a tail window is priced on the same footing as window kMtpWindows; a window past
// max_window (never verified on this model) is extrapolated from max_window: by the measured chain windows' per-row
// slope (ms[max_window] - ms[kMtpWindows]) / (max_window - kMtpWindows) when max_window > kMtpWindows and that slope
// is positive (Chains, a long chain past the calibrated depth), else by the prior shape. False (out untouched)
// when max_window is outside [kMtpWindows, kMaxWindow], a needed series is empty or a median is not positive.
bool derive_round_costs(const SpecRoundTimes &times, uint32_t max_window, SpecCosts *out);

// The longest window the seal times: kMtpWindows when the plan's calibrate_windows is CERE_CALIBRATE_MTP (the
// caller's tickets never set CERE_WORK_TAIL/LONG/CHAIN_DRAFTS, so spec_row picks windows <= kMtpWindows only:
// LoadCrit LC-01) or the draft_tokens window (the slot's param, else speculative_draft_tokens) leaves no room for a
// DraftTail tail (window < DraftPolicy::kMaxDrafts + 1); otherwise min(SpecCosts::kMaxWindow,
// min(Device::max_chain_drafts, DraftPolicy::kMaxTailDrafts) + 1).
uint32_t calibration_window(const Device &device, const DeviceModel &model, const ModelPlan &plan);

// Times the rounds of `model` (instantiated, drafting) under the device's current levers on greedy sessions of the
// synthetic prompt, created here one at a time and released before return. The mtp session: kMaxCycles round-robin
// cycles at most over the mtp windows 1..kMtpWindows (a window is skipped in a cycle while fewer than t - 1
// proposals are pending) until each holds kRounds timed rounds; then, when calibration_window > kMtpWindows, untimed
// MTP rounds extend its greedy stream to kChainCycles chain cycles. The chain session replays that stream: round-robin
// over the chain windows kMtpWindows..calibration_window, each round drafting the stream's next t - 1 tokens (a
// lossless greedy verify accepts them all; a round that does not commit t tokens is not kept), until each holds
// kRounds timed rounds or the stream ends. CERE_ERR_CAPACITY when a session does not fit the plan's context or the
// device (no slot, no pages); CERE_ERR_UNSUPPORTED when a series is not full, the device has no window kMtpWindows,
// or the chain session leaves the recorded stream; any other status is the device's.
cere_status measure_round_costs(Device &device, DeviceModel &model, const ModelPlan &plan, SpecRoundTimes *out);

// The K2 GEMM tables of the sealed model ([0] CERE_LEVER_K2_CUBLASLT and CERE_LEVER_K2_TAIL_SPLIT, [1]
// CERE_LEVER_K2_SMALL_M and CERE_LEVER_K2_SMALL_M_A8, whose m <= 16 kernels are small_m's; K2Prefill P3; k2_cost_table):
// only the plan's own
// k2_gemm table is measured (measure_round_costs, derive_round_costs) under the plan's levers with decode_step eager,
// so the load instantiates no CUDA graph (one would keep device memory after it is destroyed); the other table stays
// SpecCosts::untimed() (a later set_levers to it prices by that shape). The plan's levers are then set again. CERE_OK
// writes *out; otherwise *out is untouched and the status is measure_round_costs' (CERE_ERR_UNSUPPORTED also when the
// table does not derive) or set_levers'.
cere_status calibrate_round_costs(Device &device, DeviceModel &model, const ModelPlan &plan,
                                  std::array<SpecCosts, 2> *out);

}  // namespace cere::host
