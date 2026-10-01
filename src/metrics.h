#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace vdb {

enum class Metric { Euclidean, Cosine, Manhattan };

inline Metric metricFromString(const std::string& s) {
    if (s == "euclidean") return Metric::Euclidean;
    if (s == "cosine")    return Metric::Cosine;
    if (s == "manhattan") return Metric::Manhattan;
    throw std::invalid_argument("unknown metric: " + s);
}

inline const char* metricToString(Metric m) {
    switch (m) {
        case Metric::Cosine:    return "cosine";
        case Metric::Manhattan: return "manhattan";
        default:                return "euclidean";
    }
}

inline void requireSameDims(const std::vector<float>& a,
                            const std::vector<float>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument(
            "dimension mismatch: " + std::to_string(a.size()) +
            " vs " + std::to_string(b.size()));
}

inline float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    requireSameDims(a, b);
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

inline float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    requireSameDims(a, b);
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) s += std::fabs(a[i] - b[i]);
    return s;
}

inline float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    requireSameDims(a, b);
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na < 1e-12f || nb < 1e-12f) return 1.0f;
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

inline float distance(const std::vector<float>& a, const std::vector<float>& b,
                      Metric m) {
    switch (m) {
        case Metric::Cosine:    return cosine(a, b);
        case Metric::Manhattan: return manhattan(a, b);
        default:                return euclidean(a, b);
    }
}

inline std::vector<float> normalize(const std::vector<float>& v) {
    float n = 0.0f;
    for (float x : v) n += x * x;
    if (n < 1e-12f) return v;
    n = std::sqrt(n);
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] / n;
    return out;
}

}  