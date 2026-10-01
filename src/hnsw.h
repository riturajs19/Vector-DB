#pragma once

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "metrics.h"
#include "types.h"

namespace vdb {

class HNSW {
    struct Node {
        VectorItem item;
        int        maxLayer;
        std::vector<std::vector<int>> nbrs;
    };

    std::unordered_map<int, Node> graph_;
    int    M_;
    int    M0_;
    int    efConstruct_;
    Metric metric_;
    int    topLayer_ = -1;
    int    entryId_  = -1;
    std::mt19937 rng_;

    int randomLayer() {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        float mult = 1.0f / std::log((float)M_);
        return (int)std::floor(-std::log(u(rng_)) * mult);
    }

    std::vector<Neighbor> searchLayer(const std::vector<float>& q, int entry,
                                      int ef, int layer) const {
        std::unordered_set<int> visited;
        std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<>> candidates;
        std::priority_queue<Neighbor> found;

        float d0 = distance(q, graph_.at(entry).item.emb, metric_);
        visited.insert(entry);
        candidates.push({d0, entry});
        found.push({d0, entry});

        while (!candidates.empty()) {
            auto [cd, cid] = candidates.top();
            candidates.pop();
            if ((int)found.size() >= ef && cd > found.top().first) break;

            const Node& cur = graph_.at(cid);
            if (layer >= (int)cur.nbrs.size()) continue;

            for (int nid : cur.nbrs[layer]) {
                if (visited.count(nid) || !graph_.count(nid)) continue;
                visited.insert(nid);
                float nd = distance(q, graph_.at(nid).item.emb, metric_);
                if ((int)found.size() < ef || nd < found.top().first) {
                    candidates.push({nd, nid});
                    found.push({nd, nid});
                    if ((int)found.size() > ef) found.pop();
                }
            }
        }

        std::vector<Neighbor> result;
        result.reserve(found.size());
        while (!found.empty()) { result.push_back(found.top()); found.pop(); }
        std::sort(result.begin(), result.end());
        return result;
    }

    std::vector<int> selectNeighbors(std::vector<Neighbor>& candidates, int maxM) const {
        std::sort(candidates.begin(), candidates.end());
        std::vector<int> selected;
        std::vector<std::vector<float>> selectedVecs;

        for (auto& [d, id] : candidates) {
            if ((int)selected.size() >= maxM) break;
            const auto& cand = graph_.at(id).item.emb;

            bool goodCandidate = true;
            for (auto& sv : selectedVecs) {
                if (distance(cand, sv, metric_) < d) {
                    goodCandidate = false;
                    break;
                }
            }
            if (goodCandidate) {
                selected.push_back(id);
                selectedVecs.push_back(cand);
            }
        }

        if ((int)selected.size() < maxM) {
            for (auto& [d, id] : candidates) {
                if ((int)selected.size() >= maxM) break;
                if (std::find(selected.begin(), selected.end(), id) == selected.end())
                    selected.push_back(id);
            }
        }
        return selected;
    }

    void connect(int id, int layer, const std::vector<int>& neighbors, int maxM) {
        Node& node = graph_.at(id);
        if ((int)node.nbrs.size() <= layer) node.nbrs.resize(layer + 1);
        node.nbrs[layer] = neighbors;

        for (int nid : neighbors) {
            if (!graph_.count(nid)) continue;
            Node& other = graph_.at(nid);
            if ((int)other.nbrs.size() <= layer) other.nbrs.resize(layer + 1);
            auto& back = other.nbrs[layer];

            if (std::find(back.begin(), back.end(), id) == back.end())
                back.push_back(id);

            if ((int)back.size() > maxM) {
                std::vector<Neighbor> ranked;
                for (int c : back)
                    if (graph_.count(c))
                        ranked.push_back({distance(other.item.emb, graph_.at(c).item.emb, metric_), c});
                back = selectNeighbors(ranked, maxM);
            }
        }
    }

public:
    HNSW(int m, Metric metric, int efConstruct = 200)
        : M_(m), M0_(2 * m), efConstruct_(efConstruct), metric_(metric),
          rng_(42) {
        if (m <= 0) throw std::invalid_argument("M must be >= 1");
    }

    void insert(const VectorItem& item) {
        int id  = item.id;
        int lvl = randomLayer();
        graph_[id] = {item, lvl, std::vector<std::vector<int>>(lvl + 1)};

        if (entryId_ == -1) {
            entryId_  = id;
            topLayer_ = lvl;
            return;
        }

        int ep = entryId_;
        for (int L = topLayer_; L > lvl; --L) {
            auto res = searchLayer(item.emb, ep, 1, L);
            if (!res.empty()) ep = res[0].second;
        }

        for (int L = std::min(topLayer_, lvl); L >= 0; --L) {
            int maxM = (L == 0) ? M0_ : M_;
            auto candidates = searchLayer(item.emb, ep, efConstruct_, L);
            auto selected   = selectNeighbors(candidates, maxM);
            connect(id, L, selected, maxM);
            if (!candidates.empty()) ep = candidates[0].second;
        }

        if (lvl > topLayer_) {
            topLayer_ = lvl;
            entryId_  = id;
        }
    }

    std::vector<Neighbor> knn(const std::vector<float>& query, int k, int ef = 50) const {
        if (k <= 0) throw std::invalid_argument("k must be >= 1");
        if (entryId_ == -1) return {};

        int ep = entryId_;
        for (int L = topLayer_; L > 0; --L) {
            auto res = searchLayer(query, ep, 1, L);
            if (!res.empty()) ep = res[0].second;
        }

        auto result = searchLayer(query, ep, std::max(ef, k), 0);
        if ((int)result.size() > k) result.resize(k);
        return result;
    }

    void remove(int id) {
        if (!graph_.count(id)) return;

        for (auto& [nid, node] : graph_)
            for (auto& layer : node.nbrs)
                layer.erase(std::remove(layer.begin(), layer.end(), id), layer.end());

        if (entryId_ == id) {
            entryId_ = -1;
            for (auto& [nid, node] : graph_)
                if (nid != id) { entryId_ = nid; topLayer_ = node.maxLayer; break; }
        }
        graph_.erase(id);
    }

    size_t size() const { return graph_.size(); }
};

}  // namespace vdb