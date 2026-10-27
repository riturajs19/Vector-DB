#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

#include "metrics.h"
#include "types.h"

namespace vdb {

class KDTree {
public:
    KDTree(int dims = 1, Metric metric = Metric::Cosine)
        : dims_(dims > 0 ? dims : 1),
          metric_(metric),
          searchMetric_(metric == Metric::Manhattan ? Metric::Manhattan : Metric::Euclidean),
          normalized_(metric == Metric::Cosine) {}

    void build(const std::vector<VectorItem>& items) {
        entries_.clear();
        root_.reset();
        if (items.empty()) return;

        dims_ = (int)items[0].emb.size();
        for (const auto& it : items) {
            if ((int)it.emb.size() != dims_)
                throw std::invalid_argument("dimension mismatch: item has " + std::to_string(it.emb.size()) + ", tree expects " + std::to_string(dims_));
            entries_.push_back({it.id, normalized_ ? normalize(it.emb) : it.emb});
        }

        std::vector<int> idx(entries_.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;

        root_ = buildRec(idx, 0, (int)idx.size());
    }

    std::vector<Neighbor> knn(const std::vector<float>& query, int k) const {
        if (k <= 0) throw std::invalid_argument("k must be >= 1");
        if ((int)query.size() != dims_)
            throw std::invalid_argument("dimension mismatch: query has " + std::to_string(query.size()) + ", tree expects " + std::to_string(dims_));
        if (!root_) return {};

        std::vector<float> q = normalized_ ? normalize(query) : query;

        std::priority_queue<Neighbor> heap;
        search(root_.get(), q, k, heap);

        std::vector<Neighbor> out;
        while (!heap.empty()) { out.push_back(heap.top()); heap.pop(); }
        std::sort(out.begin(), out.end());

        if (normalized_) {
            for (auto& n : out) n.first = (n.first * n.first) / 2.0f;
        }
        return out;
    }

    size_t size() const { return entries_.size(); }

private:
    struct Entry {
        int                id;
        std::vector<float> key;
    };

    struct Node {
        int                   entry;
        int                   axis;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    int widestAxis(const std::vector<int>& idx, int lo, int hi) const {
        int   best = 0;
        float bestSpan = -1.0f;
        for (int ax = 0; ax < dims_; ++ax) {
            float mn = entries_[idx[lo]].key[ax], mx = mn;
            for (int i = lo + 1; i < hi; ++i) {
                float v = entries_[idx[i]].key[ax];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            if (mx - mn > bestSpan) { bestSpan = mx - mn; best = ax; }
        }
        return best;
    }

    std::unique_ptr<Node> buildRec(std::vector<int>& idx, int lo, int hi) {
        if (lo >= hi) return nullptr;

        int ax  = widestAxis(idx, lo, hi);
        int mid = lo + (hi - lo) / 2;

        std::nth_element(idx.begin() + lo, idx.begin() + mid, idx.begin() + hi,
                         [&](int a, int b) {
                             return entries_[a].key[ax] < entries_[b].key[ax];
                         });

        auto node   = std::make_unique<Node>();
        node->entry = idx[mid];
        node->axis  = ax;
        node->left  = buildRec(idx, lo, mid);
        node->right = buildRec(idx, mid + 1, hi);
        return node;
    }

    void search(const Node* n, const std::vector<float>& q, int k,
                std::priority_queue<Neighbor>& heap) const {
        if (!n) return;
        const Entry& e = entries_[n->entry];

        float d = distance(q, e.key, searchMetric_);
        if ((int)heap.size() < k) {
            heap.push({d, e.id});
        } else if (d < heap.top().first) {
            heap.pop();
            heap.push({d, e.id});
        }

        float diff    = q[n->axis] - e.key[n->axis];
        const Node* closer  = diff < 0 ? n->left.get()  : n->right.get();
        const Node* farther = diff < 0 ? n->right.get() : n->left.get();

        search(closer, q, k, heap);

        if ((int)heap.size() < k || std::fabs(diff) < heap.top().first)
            search(farther, q, k, heap);
    }

    int    dims_;
    Metric metric_;
    Metric searchMetric_;
    bool   normalized_;

    std::vector<Entry>    entries_;
    std::unique_ptr<Node> root_;
};

}  // namespace vdb
