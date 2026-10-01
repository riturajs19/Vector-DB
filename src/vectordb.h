#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include "bruteforce.h"
#include "kdtree.h"
#include "hnsw.h"
#include "metrics.h"
#include "types.h"

namespace vdb {

class VectorDB {
public:
    VectorDB(int dims, Metric metric)
        : dims_(dims), metric_(metric),
          kdt_(dims, metric), hnsw_(16, metric), nextId_(1) {}

    int insert(const std::string& meta, const std::string& cat,
              const std::vector<float>& emb) {
        std::lock_guard<std::mutex> lk(mu_);
        if ((int)emb.size() != dims_)
            throw std::invalid_argument("dimension mismatch");

        VectorItem v{nextId_++, meta, cat, emb};
        store_[v.id] = v;
        bf_.insert(v);
        hnsw_.insert(v);
        rebuildKDTree();
        return v.id;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!store_.count(id)) return false;
        store_.erase(id);
        bf_.remove(id);
        hnsw_.remove(id);
        rebuildKDTree();
        return true;
    }

    std::vector<Neighbor> search(const std::vector<float>& q, int k,
                                 const std::string& algo) {
        std::lock_guard<std::mutex> lk(mu_);
        if (algo == "bruteforce") return bf_.knn(q, k, metric_);
        if (algo == "kdtree")     return kdt_.knn(q, k);
        return hnsw_.knn(q, k);
    }

    std::vector<VectorItem> all() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<VectorItem> r;
        for (auto& [id, v] : store_) r.push_back(v);
        return r;
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu_);
        return store_.size();
    }

private:
    void rebuildKDTree() {
        std::vector<VectorItem> items;
        for (auto& [id, v] : store_) items.push_back(v);
        kdt_.build(items);
    }

    int    dims_;
    Metric metric_;
    std::unordered_map<int, VectorItem> store_;
    BruteForce bf_;
    KDTree     kdt_;
    HNSW       hnsw_;
    std::mutex mu_;
    int        nextId_;
};

}  // namespace vdb