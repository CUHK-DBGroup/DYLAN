#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dylan {
using Id = int;
using Labels = std::vector<int>;

// Immutable, normalized label predicate. AND/OR may be nested arbitrarily.
class Filter {
public:
    enum class Kind { Containment, Overlap, Equality, All, Any };
    static Filter containment(Labels labels);
    static Filter overlap(Labels labels);
    static Filter equality(Labels labels);
    static Filter all(std::vector<Filter> children);
    static Filter any(std::vector<Filter> children);
    Kind kind() const noexcept { return kind_; }
    const Labels &labels() const noexcept { return labels_; }
    const std::vector<Filter> &children() const noexcept { return children_; }

private:
    Filter(Kind kind, Labels labels, std::vector<Filter> children);
    Kind kind_;
    Labels labels_;
    std::vector<Filter> children_;
};

struct BuildOptions {
    std::size_t dimension = 128;
    std::size_t capacity = 0; // 0 means the initial object count.
    int label_count = 1;      // Reserved internal label IDs [1, label_count].
    int max_neighbors = 32;
    int ef_construction = 100;
    double alpha = 2; // 1: materialize every node; 2: shared graphs with maintenance.
    int enhanced_labels = 20;
    std::uint32_t label_seed = 37; // Independent RNG for unseen external labels.
};

struct SearchOptions {
    std::size_t k = 10;
    int ef = 100;
};

struct Neighbor {
    Id id;
    float distance; // Squared L2, matching the original implementation.
};

// Owns all storage. Sequential use only: searches modify visit stamps.
// IDs are unique nonnegative application IDs, independent of internal/DFS ordering.
class Index {
public:
    static Index build(const std::vector<float> &vectors, const std::vector<Labels> &labels,
                       const std::vector<Id> &ids, BuildOptions options);
    // Native saves retain update settings, label mapping and RNG states.
    // Old readers only see internal labels and must not resave mapped indexes.
    // For old .ind3 files, supply the original alpha and enhancement setting.
    // legacy_max_label can correct a stored maximum inflated by old repeated loads.
    static Index load(const std::string &path, double legacy_alpha = 2,
                      int legacy_enhanced_labels = 20, int legacy_max_label = 0);
    ~Index();
    Index(Index &&) noexcept;
    Index &operator=(Index &&) noexcept;
    Index(const Index &) = delete;
    Index &operator=(const Index &) = delete;

    std::vector<Neighbor> search(const std::vector<float> &query, const Filter &filter,
                                 SearchOptions options = {});
    // Exact filtered search, useful for oracles and small eligible sets.
    std::vector<Neighbor> scan(const std::vector<float> &query, const Filter &filter,
                               std::size_t k = 10);
    // Uses existing mappings. An unseen label keeps its ID if that reserved ID is free;
    // otherwise use insert_with_new_labels. All public label arguments are external labels.
    void insert(Id id, const std::vector<float> &vector, Labels labels);
    // Assigns each unseen positive external label a uniformly random free internal ID.
    // Throws length_error if reserved label IDs or object capacity are exhausted.
    void insert_with_new_labels(Id id, const std::vector<float> &vector, Labels labels);
    const std::unordered_map<int, int> &label_map() const noexcept;
    std::size_t available_label_count() const noexcept;
    bool erase(Id id); // Lazy deletion; false if absent or already deleted. IDs are not reused.
    void save(const std::string &path) const;
    void reorder_by_trie();
    std::size_t size() const noexcept; // Includes tombstones.
    std::size_t live_size() const noexcept;
    const BuildOptions &options() const noexcept;

private:
    struct Impl;
    explicit Index(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace dylan
