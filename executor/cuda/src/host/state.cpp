#include "host/state.h"

#include <iterator>

namespace cere::host {

bool TensorLedger::reserve(uint64_t offset, uint64_t len) {
    const uint64_t end = offset + len;
    std::lock_guard<std::mutex> lock(mutex_);
    auto next = ranges_.lower_bound(offset);
    if (next != ranges_.end() && next->first < end) return false;
    if (next != ranges_.begin() && std::prev(next)->second > offset) return false;
    ranges_.emplace_hint(next, offset, end);
    covered_ += len;
    return true;
}

void TensorLedger::unreserve(uint64_t offset) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ranges_.find(offset);
    if (it == ranges_.end()) return;
    covered_ -= it->second - it->first;
    ranges_.erase(it);
}

uint64_t TensorLedger::covered() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return covered_;
}

bool StagingRanges::overlaps(uintptr_t start, uintptr_t end) const {
    auto next = ranges_.lower_bound(start);
    if (next != ranges_.end() && next->first < end) return true;
    return next != ranges_.begin() && std::prev(next)->second.end > start;
}

uintptr_t StagingRanges::acquire(const void *p, uint64_t len) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(p);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ranges_.upper_bound(start);
    if (it == ranges_.begin()) return 0;
    --it;
    if (start + len < start || start + len > it->second.end) return 0;
    ++it->second.puts;
    return it->first;
}

void StagingRanges::release(uintptr_t start) {
    std::lock_guard<std::mutex> lock(mutex_);
    --ranges_.at(start).puts;
}

bool StagingRanges::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ranges_.empty();
}

bool ArenaAllocator::allocate(uint64_t bytes, uint64_t *offset) {
    if (bytes == 0) return false;
    uint64_t cursor = 0;
    auto it = ranges_.begin();
    for (;; ++it) {
        const uint64_t gap_end = it == ranges_.end() ? capacity_ : it->first;
        if (gap_end >= cursor && gap_end - cursor >= bytes) break;
        if (it == ranges_.end()) return false;
        cursor = it->second;
    }
    ranges_.emplace_hint(it, cursor, cursor + bytes);
    used_ += bytes;
    *offset = cursor;
    return true;
}

void ArenaAllocator::free(uint64_t offset) {
    auto it = ranges_.find(offset);
    if (it == ranges_.end()) return;
    used_ -= it->second - it->first;
    ranges_.erase(it);
}

bool ArenaAllocator::fits(uint64_t bytes) const {
    if (bytes == 0) return false;
    uint64_t cursor = 0;
    for (const auto &range : ranges_) {
        if (range.first - cursor >= bytes) return true;
        cursor = range.second;
    }
    return capacity_ - cursor >= bytes;
}

uint64_t ArenaAllocator::range_bytes(uint64_t offset) const {
    const auto it = ranges_.find(offset);
    return it == ranges_.end() ? 0 : it->second - it->first;
}

}  // namespace cere::host
