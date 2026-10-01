#pragma once

#include <string>
#include <utility>
#include <vector>

namespace vdb {

struct VectorItem {
    int                id = 0;
    std::string        metadata;
    std::string        category;
    std::vector<float> emb;
};

using Neighbor = std::pair<float, int>;

}  // namespace vdb