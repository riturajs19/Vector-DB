#include "httplib.h"
#include "src/vectordb.h"
#include "src/documentdb.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cctype>

using namespace vdb;

std::string jS(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if      (c == '"')  o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else                o += c;
    }
    return o + '"';
}

std::string jVec(const std::vector<float>& v) {
    std::ostringstream ss; ss << '[';
    for (size_t i = 0; i < v.size(); i++) {
        if (i) ss << ',';
        ss << std::fixed << std::setprecision(4) << v[i];
    }
    return ss.str() + ']';
}

std::vector<float> parseVec(const std::string& s) {
    std::vector<float> v;
    std::istringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) {
        // Trim whitespace
        size_t b = 0;
        while (b < t.size() && std::isspace(static_cast<unsigned char>(t[b]))) b++;
        size_t e = t.size();
        while (e > b && std::isspace(static_cast<unsigned char>(t[e - 1]))) e--;
        if (b < e) {
            try { v.push_back(std::stof(t.substr(b, e - b))); } catch (...) {}
        }
    }
    return v;
}

std::string extractStr(const std::string& body, const std::string& key) {
    size_t p = 0;
    while ((p = body.find('"' + key + '"', p)) != std::string::npos) {
        size_t colon = body.find(':', p + key.size() + 2);
        if (colon == std::string::npos) break;
        size_t v = colon + 1;
        while (v < body.size() && (body[v] == ' ' || body[v] == '\t' || body[v] == '\r' || body[v] == '\n')) v++;
        if (v < body.size() && body[v] == '"') {
            v++;
            std::string result;
            while (v < body.size()) {
                if (body[v] == '"') return result;
                if (body[v] == '\\' && v + 1 < body.size()) {
                    v++;
                    switch (body[v]) {
                        case '"':  result += '"'; break;
                        case '\\': result += '\\'; break;
                        case 'n':  result += '\n'; break;
                        case 'r':  result += '\r'; break;
                        case 't':  result += '\t'; break;
                        default:   result += body[v]; break;
                    }
                } else {
                    result += body[v];
                }
                v++;
            }
            return result;
        }
        p += key.size() + 2;
    }
    return "";
}

int extractInt(const std::string& body, const std::string& key, int def = 0) {
    size_t p = 0;
    while ((p = body.find('"' + key + '"', p)) != std::string::npos) {
        size_t colon = body.find(':', p + key.size() + 2);
        if (colon != std::string::npos) {
            size_t val = colon + 1;
            while (val < body.size() && (body[val] == ' ' || body[val] == '\t' || body[val] == '\r' || body[val] == '\n')) val++;
            if (val < body.size() && (std::isdigit(body[val]) || body[val] == '-')) {
                try { return std::stoi(body.substr(val)); } catch (...) { return def; }
            }
        }
        p += key.size() + 2;
    }
    return def;
}

std::vector<float> extractArr(const std::string& body, const std::string& key) {
    size_t p = 0;
    while ((p = body.find('"' + key + '"', p)) != std::string::npos) {
        size_t colon = body.find(':', p + key.size() + 2);
        if (colon != std::string::npos) {
            size_t bracket = colon + 1;
            while (bracket < body.size() && (body[bracket] == ' ' || body[bracket] == '\t' || body[bracket] == '\r' || body[bracket] == '\n')) bracket++;
            if (bracket < body.size() && body[bracket] == '[') {
                size_t e = bracket + 1;
                int depth = 1;
                while (e < body.size() && depth > 0) {
                    if (body[e] == '[') depth++;
                    else if (body[e] == ']') depth--;
                    e++;
                }
                if (depth == 0) {
                    return parseVec(body.substr(bracket + 1, e - bracket - 2));
                }
            }
        }
        p += key.size() + 2;
    }
    return {};
}

void cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

void seedDemo(VectorDB& db) {
    if (db.size() > 0) return;
    // Pre-load 4D demo vectors matching categories "hr", "finance", "other"
    db.insert("Recruitment and talent acquisition process", "hr", {0.88f, 0.12f, 0.08f, 0.91f});
    db.insert("Annual employee performance and compensation review", "hr", {0.85f, 0.15f, 0.10f, 0.89f});
    db.insert("Company benefits and health insurance packages", "hr", {0.80f, 0.18f, 0.14f, 0.86f});
    db.insert("Corporate quarterly financial statement and profit margins", "finance", {0.12f, 0.92f, 0.88f, 0.15f});
    db.insert("Stock market investment portfolio and risk analysis", "finance", {0.10f, 0.89f, 0.94f, 0.12f});
    db.insert("Tax optimization strategies and capital gains budgeting", "finance", {0.15f, 0.85f, 0.91f, 0.18f});
    db.insert("Deep learning neural networks and transformer models", "other", {0.91f, 0.10f, 0.87f, 0.10f});
    db.insert("KD-Tree and HNSW indexing for approximate nearest neighbors", "other", {0.86f, 0.06f, 0.09f, 0.88f});
    db.insert("Cloud infrastructure distributed microservices deployment", "other", {0.78f, 0.22f, 0.35f, 0.70f});
}

int main() {
    VectorDB   demoDB(0, Metric::Cosine); // Dynamic dimensions
    DocumentDB docDB;

    // Seed demo vectors
    seedDemo(demoDB);

    std::cout << "========================================\n";
    std::cout << "    VectorDB + RAG Server Running\n";
    std::cout << "    Endpoint: http://localhost:8080\n";
    std::cout << "    Demo Vectors: " << demoDB.size() << " items (" << demoDB.getDims() << "D)\n";
    std::cout << "========================================\n" << std::flush;

    httplib::Server svr;

    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        cors(res);
        res.status = 204;
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::string meta = extractStr(req.body, "metadata");
        std::string cat  = extractStr(req.body, "category");
        auto emb = extractArr(req.body, "embedding");
        if (emb.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"Empty or invalid embedding array\"}", "application/json");
            return;
        }
        try {
            int id = demoDB.insert(meta, cat, emb);
            res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
        } catch (std::exception& e) {
            res.status = 400;
            res.set_content("{\"error\":" + jS(e.what()) + "}", "application/json");
        }
    });

    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id = std::stoi(req.matches[1]);
        bool ok = demoDB.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
    });

    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        int k = 5;
        try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        std::string algo = req.get_param_value("algo");
        if (algo.empty()) algo = "hnsw";
        try {
            auto out = demoDB.search(q, k, algo);
            std::ostringstream ss;
            ss << "{\"results\":[";
            for (size_t i = 0; i < out.size(); i++) {
                if (i) ss << ',';
                auto item = demoDB.getItem(out[i].second);
                ss << "{\"id\":" << out[i].second
                   << ",\"distance\":" << out[i].first
                   << ",\"metadata\":" << jS(item.metadata)
                   << ",\"category\":" << jS(item.category)
                   << '}';
            }
            ss << "]}";
            res.set_content(ss.str(), "application/json");
        } catch (std::exception& e) {
            res.status = 400;
            res.set_content("{\"error\":" + jS(e.what()) + "}", "application/json");
        }
    });

    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ostringstream ss;
        ss << '[';
        auto items = demoDB.all();
        for (size_t i = 0; i < items.size(); i++) {
            if (i) ss << ',';
            auto& v = items[i];
            ss << "{\"id\":" << v.id << ",\"metadata\":" << jS(v.metadata)
               << ",\"category\":" << jS(v.category) << ",\"embedding\":" << jVec(v.emb) << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/seed", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        seedDemo(demoDB);
        res.set_content("{\"ok\":true,\"count\":" + std::to_string(demoDB.size()) + "}", "application/json");
    });

    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::string title = extractStr(req.body, "title");
        std::string text  = extractStr(req.body, "text");
        auto emb = extractArr(req.body, "embedding");
        if (emb.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"Empty or invalid embedding array\"}", "application/json");
            return;
        }
        try {
            int id = docDB.insert(title, text, emb);
            res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
        } catch (std::exception& e) {
            res.status = 400;
            res.set_content("{\"error\":" + jS(e.what()) + "}", "application/json");
        }
    });

    svr.Post("/doc/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto emb = extractArr(req.body, "embedding");
        int k = extractInt(req.body, "k", 3);
        float maxDist = 2.0f;
        auto hits = docDB.search(emb, k, maxDist);
        std::ostringstream ss;
        ss << "{\"results\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":" << hits[i].second.id
               << ",\"title\":" << jS(hits[i].second.title)
               << ",\"text\":" << jS(hits[i].second.text)
               << ",\"distance\":" << hits[i].first << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    svr.Delete(R"(/doc/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id = std::stoi(req.matches[1]);
        bool ok = docDB.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
    });

    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ostringstream ss;
        ss << '[';
        auto docs = docDB.all();
        for (size_t i = 0; i < docs.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":" << docs[i].id << ",\"title\":" << jS(docs[i].title)
               << ",\"text\":" << jS(docs[i].text) << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ostringstream ss;
        ss << "{\"demoCount\":" << demoDB.size()
           << ",\"demoDims\":" << demoDB.getDims()
           << ",\"docCount\":" << docDB.size()
           << ",\"docDims\":" << docDB.getDims()
           << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.listen("0.0.0.0", 8080);
    return 0;
}
