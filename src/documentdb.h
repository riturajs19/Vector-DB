#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include "bruteforce.h"
#include "hnsw.h"
#include "metrics.h"
#include "types.h"

namespace vdb {

struct DocItem {
    int         id;
    std::string title;
    std::string text;
    std::vector<float> emb;
};

class DocumentDB {
public:
    DocumentDB() : hnsw_(16, Metric::Cosine), dims_(0), nextId_(1) {}

    int insert(const std::string& title, const std::string& text,
              const std::vector<float>& emb) {
        std::lock_guard<std::mutex> lk(mu_);
        if (dims_ == 0) dims_ = (int)emb.size();
        if ((int)emb.size() != dims_)
            throw std::invalid_argument("dimension mismatch");

        DocItem item{nextId_++, title, text, emb};
        store_[item.id] = item;

        VectorItem vi{item.id, title, "doc", emb};
        bf_.insert(vi);
        hnsw_.insert(vi);
        return item.id;
    }

    std::vector<std::pair<float, DocItem>> search(const std::vector<float>& q,
                                                   int k, float maxDist = 0.7f) {
        std::lock_guard<std::mutex> lk(mu_);
        if (store_.empty()) return {};

        auto raw = (store_.size() < 10)
                   ? bf_.knn(q, k, Metric::Cosine)
                   : hnsw_.knn(q, k, 50);

        std::vector<std::pair<float, DocItem>> out;
        for (auto& [d, id] : raw)
            if (store_.count(id) && d <= maxDist)
                out.push_back({d, store_[id]});
        return out;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!store_.count(id)) return false;
        store_.erase(id);
        bf_.remove(id);
        hnsw_.remove(id);
        return true;
    }

    std::vector<DocItem> all() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<DocItem> r;
        for (auto& [id, v] : store_) r.push_back(v);
        return r;
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu_);
        return store_.size();
    }

    int getDims() {
        std::lock_guard<std::mutex> lk(mu_);
        return dims_;
    }

private:
    std::unordered_map<int, DocItem> store_;
    BruteForce bf_;
    HNSW       hnsw_;
    int        dims_;
    std::mutex mu_;
    int        nextId_;
};

}  // namespace vdb