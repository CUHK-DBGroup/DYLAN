#include <dylan/index.hpp>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <set>

namespace {
using Args = std::map<std::string, std::string>;
std::string required(const Args &args, const std::string &key) {
    auto it = args.find(key);
    if (it == args.end())
        throw std::invalid_argument("missing --" + key);
    return it->second;
}
int integer(const Args &args, const std::string &key, int fallback) {
    auto it = args.find(key);
    if (it == args.end())
        return fallback;
    std::size_t end = 0;
    const int value = std::stoi(it->second, &end);
    if (end != it->second.size())
        throw std::invalid_argument("invalid --" + key);
    return value;
}
struct Vectors {
    std::vector<float> data;
    int count = 0, dimension = 0;
};
Vectors read_vectors(const std::string &path) {
    // Validate every record here; legacy benchmark I/O remains unchanged.
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open vector file: " + path);
    in.seekg(0, std::ios::end);
    const auto file_size = in.tellg();
    in.seekg(0);
    Vectors result;
    while (in.peek() != std::char_traits<char>::eof()) {
        int dimension = 0;
        in.read(reinterpret_cast<char *>(&dimension), sizeof(dimension));
        if (!in || dimension <= 0 || (result.dimension && dimension != result.dimension))
            throw std::runtime_error("invalid fvecs dimension");
        result.dimension = dimension;
        const auto offset = result.data.size();
        if (file_size - in.tellg() < static_cast<std::streamoff>(dimension) * sizeof(float))
            throw std::runtime_error("truncated fvecs record");
        if (!result.count) {
            const auto record_bytes = (static_cast<std::uint64_t>(dimension) + 1) * sizeof(float);
            if (static_cast<std::uint64_t>(file_size) % record_bytes)
                throw std::runtime_error("fvecs size is not record aligned");
            result.data.reserve(static_cast<std::size_t>(file_size) / record_bytes * dimension);
        }
        result.data.resize(offset + dimension);
        in.read(reinterpret_cast<char *>(result.data.data() + offset), dimension * sizeof(float));
        if (!in)
            throw std::runtime_error("truncated fvecs record");
        ++result.count;
    }
    if (!result.count)
        throw std::runtime_error("empty vector file");
    return result;
}
std::vector<dylan::Labels> read_labels(const std::string &path, int n) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open labels: " + path);
    std::vector<dylan::Labels> result;
    std::string line;
    while (std::getline(in, line)) {
        dylan::Labels row;
        std::stringstream cells(line);
        std::string cell;
        while (std::getline(cells, cell, ',')) {
            std::size_t used = 0;
            int label = std::stoi(cell, &used);
            if (cell.find_first_not_of(" \t\r", used) != std::string::npos)
                throw std::invalid_argument("invalid CSV label");
            row.push_back(label);
        }
        result.push_back(std::move(row));
    }
    if (result.size() != static_cast<std::size_t>(n))
        throw std::invalid_argument("label/vector row count mismatch");
    return result;
}
std::vector<int> read_ids(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open IDs: " + path);
    std::vector<int> result;
    int id;
    while (in >> id)
        result.push_back(id);
    if (!in.eof())
        throw std::invalid_argument("invalid ID file");
    return result;
}
void help() {
    std::cout
        << "dylan build --vectors BASE.fvecs --labels BASE.csv --label-num MAX --output INDEX "
           "[--capacity N] [--ids IDS.txt]\n"
           "            [--m 32] [--ef-construction 100] [--alpha 2] [--enhanced-labels 20] "
           "[--label-seed 37]\n"
           "dylan search --index INDEX --vectors QUERY.fvecs --labels QUERY.csv --match "
           "containment|overlap|equality --output RESULTS.tsv [--k 10] [--ef 100] [--exact 1]\n"
           "dylan insert --index INDEX --vectors NEW.fvecs --labels NEW.csv --ids IDS.txt --output "
           "INDEX [--allow-new-labels 1]\n"
           "dylan delete --index INDEX --ids IDS.txt --output INDEX\n"
           "dylan reorder --index INDEX --output INDEX\n"
           "Old .ind3 imports: --alpha and --enhanced-labels specify missing update settings.\n";
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            help();
            return 0;
        }
        const std::string mode = argv[1];
        const std::map<std::string, std::set<std::string>> allowed{
            {"build",
             {"vectors", "labels", "label-num", "output", "capacity", "ids", "m", "ef-construction",
              "alpha", "enhanced-labels", "label-seed"}},
            {"search",
             {"index", "vectors", "labels", "match", "output", "k", "ef", "exact", "alpha",
              "enhanced-labels"}},
            {"insert",
             {"index", "vectors", "labels", "ids", "output", "alpha", "enhanced-labels",
              "label-num", "allow-new-labels"}},
            {"delete", {"index", "ids", "output", "alpha", "enhanced-labels", "label-num"}},
            {"reorder", {"index", "output", "alpha", "enhanced-labels", "label-num"}}};
        if (!allowed.count(mode))
            throw std::invalid_argument("unknown command: " + mode);
        auto effective_allowed = allowed.at(mode);
        effective_allowed.insert("label-num");
        Args args;
        for (int i = 2; i < argc; i += 2) {
            const std::string option = argv[i];
            if (option.rfind("--", 0) != 0 || i + 1 == argc ||
                !effective_allowed.count(option.substr(2)))
                throw std::invalid_argument("unknown or incomplete option: " + option);
            if (!args.emplace(option.substr(2), argv[i + 1]).second)
                throw std::invalid_argument("duplicate option: " + option);
        }
        const auto output = required(args, "output");
        if (mode == "build") {
            const auto vectors = read_vectors(required(args, "vectors"));
            const auto labels = read_labels(required(args, "labels"), vectors.count);
            std::vector<int> ids(vectors.count);
            std::iota(ids.begin(), ids.end(), 0);
            if (args.count("ids"))
                ids = read_ids(args.at("ids"));
            dylan::BuildOptions options;
            options.dimension = vectors.dimension;
            required(args, "label-num");
            options.label_count = integer(args, "label-num", 0);
            const int capacity = integer(args, "capacity", vectors.count);
            if (capacity < 0)
                throw std::invalid_argument("negative capacity");
            options.capacity = capacity;
            options.max_neighbors = integer(args, "m", 32);
            options.ef_construction = integer(args, "ef-construction", 100);
            options.alpha = integer(args, "alpha", 2);
            options.enhanced_labels = integer(args, "enhanced-labels", 20);
            const int label_seed = integer(args, "label-seed", 37);
            if (label_seed < 0)
                throw std::invalid_argument("negative label seed");
            options.label_seed = static_cast<std::uint32_t>(label_seed);
            auto index = dylan::Index::build(vectors.data, labels, ids, options);
            index.save(output);
        } else {
            auto index = dylan::Index::load(required(args, "index"), integer(args, "alpha", 2),
                                            integer(args, "enhanced-labels", 20),
                                            integer(args, "label-num", 0));
            if (mode == "search" || mode == "insert") {
                const auto vectors = read_vectors(required(args, "vectors"));
                const auto labels = read_labels(required(args, "labels"), vectors.count);
                if (static_cast<std::size_t>(vectors.dimension) != index.options().dimension)
                    throw std::invalid_argument("vector dimension mismatch");
                if (mode == "search") {
                    const auto match = required(args, "match");
                    if (match != "containment" && match != "overlap" && match != "equality")
                        throw std::invalid_argument("unknown match type");
                    const int k = integer(args, "k", 10), ef = integer(args, "ef", 100);
                    if (k < 0 || ef <= 0)
                        throw std::invalid_argument("invalid k or ef");
                    if (std::filesystem::absolute(output).lexically_normal() ==
                        std::filesystem::absolute(required(args, "index")).lexically_normal())
                        throw std::invalid_argument("search output must differ from the index");
                    std::ofstream out(output);
                    if (!out)
                        throw std::runtime_error("cannot open results");
                    out << "query_id\tid\tsquared_l2\n";
                    for (int i = 0; i < vectors.count; ++i) {
                        std::vector<float> q(vectors.data.begin() + i * vectors.dimension,
                                             vectors.data.begin() + (i + 1) * vectors.dimension);
                        const auto f = match == "containment"
                                           ? dylan::Filter::containment(labels[i])
                                       : match == "overlap" ? dylan::Filter::overlap(labels[i])
                                                            : dylan::Filter::equality(labels[i]);
                        const auto result =
                            integer(args, "exact", 0)
                                ? index.scan(q, f, k)
                                : index.search(q, f, {static_cast<std::size_t>(k), ef});
                        for (const auto &hit : result)
                            out << i << '\t' << hit.id << '\t' << hit.distance << '\n';
                    }
                    out.flush();
                    if (!out)
                        throw std::runtime_error("failed to write results");
                } else {
                    const int allow_new = integer(args, "allow-new-labels", 0);
                    if (allow_new != 0 && allow_new != 1)
                        throw std::invalid_argument("allow-new-labels must be 0 or 1");
                    const auto ids = read_ids(required(args, "ids"));
                    if (ids.size() != labels.size())
                        throw std::invalid_argument("ID/vector row count mismatch");
                    for (int i = 0; i < vectors.count; ++i) {
                        std::vector<float> v(vectors.data.begin() + i * vectors.dimension,
                                             vectors.data.begin() + (i + 1) * vectors.dimension);
                        if (allow_new)
                            index.insert_with_new_labels(ids[i], v, labels[i]);
                        else
                            index.insert(ids[i], v, labels[i]);
                    }
                    index.save(output);
                }
            } else {
                if (mode == "delete")
                    for (int id : read_ids(required(args, "ids")))
                        index.erase(id);
                else
                    index.reorder_by_trie();
                index.save(output);
            }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
