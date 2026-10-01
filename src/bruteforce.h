#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "metrics.h"
#include "types.h"

namespace vdb {

class BruteForce {
public:
    void insert(const VectorItem& v) { items_.push_back(v); }

    void remove(int id) {
        items_.erase(std::remove_if(items_.begin(), items_.end(),[id](const VectorItem& v) { return v.id == id; }), items_.end());
    }

    void clear() { items_.clear(); }

    std::vector<Neighbor> knn(const std::vector<float>& q, int k, Metric m) const {
        if (k <= 0) throw std::invalid_argument("k must be >= 1");
        if (items_.empty()) return {};

        std::vector<Neighbor> scored;
        scored.reserve(items_.size());
        for (const auto& v : items_)
            scored.push_back({distance(q, v.emb, m), v.id});

        const size_t kk = std::min<size_t>(k, scored.size());
        std::partial_sort(scored.begin(), scored.begin() + kk, scored.end());
        scored.resize(kk);
        return scored;
    }

    size_t size() const { return items_.size(); }
    const std::vector<VectorItem>& items() const { return items_; }

private:
    std::vector<VectorItem> items_;
};

}  // namespace vdb