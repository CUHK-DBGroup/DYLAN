#include <dylan/index.hpp>
#include "detail/engine.hpp"
#include "detail/label_registry.hpp"

#include <climits>
#include <cstdint>
#include <cmath>
#include <limits>
#include <sstream>

namespace dylan {
namespace {
Labels normalize(Labels labels) {
    if (std::any_of(labels.begin(), labels.end(), [](int x) { return x <= 0; }))
        throw std::invalid_argument("labels must be positive integers");
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    return labels;
}
void check_options(const BuildOptions &o, std::size_t n) {
    if (!n || !o.dimension || o.dimension > INT_MAX || n > o.capacity || o.capacity > INT_MAX ||
        o.label_count < 1 || o.label_count == INT_MAX || o.max_neighbors < 2 ||
        o.max_neighbors > 32767 || o.ef_construction < 1 || (o.alpha != 1 && o.alpha != 2) ||
        o.enhanced_labels < 0)
        throw std::invalid_argument(
            "invalid dimensions, capacity, graph options or label universe");
}
void check_vector(const std::vector<float> &v, std::size_t dimension) {
    if (v.size() != dimension)
        throw std::invalid_argument("vector dimension mismatch");
    for (float x : v)
        if (!std::isfinite(x))
            throw std::invalid_argument("vectors must be finite");
}
void delete_tree(TrieNode *node) {
    if (!node)
        return;
    for (auto &child : node->child)
        delete_tree(child.second);
    delete node;
}
struct EngineDeleter {
    void operator()(TrieIndex *e) const {
        if (!e)
            return;
        for (auto p : e->linklist)
            std::free(p);
        delete[] e->vecData_;
        delete[] e->keyList_;
        delete[] e->isDeleted;
        delete[] e->visited_array;
        delete_tree(e->root);
        delete e;
    }
};
using Engine = std::unique_ptr<TrieIndex, EngineDeleter>;
using Nodes = std::vector<TrieNode *>;
using Heap = std::priority_queue<std::pair<float, hnswlib::labeltype>>;
std::vector<Neighbor> unpack(Heap heap) {
    std::vector<Neighbor> result(heap.size());
    for (std::size_t i = result.size(); i; --i) {
        result[i - 1] = {static_cast<Id>(heap.top().second), heap.top().first};
        heap.pop();
    }
    return result;
}
bool matches(const Filter &f, const Labels &labels) {
    const auto end = labels.end() - 1; // The final value is the trie terminator.
    switch (f.kind()) {
    case Filter::Kind::Containment:
        return std::includes(labels.begin(), end, f.labels().begin(), f.labels().end());
    case Filter::Kind::Equality:
        return static_cast<std::size_t>(end - labels.begin()) == f.labels().size() &&
               std::equal(labels.begin(), end, f.labels().begin());
    case Filter::Kind::Overlap:
        for (int x : f.labels())
            if (std::binary_search(labels.begin(), end, x))
                return true;
        return false;
    case Filter::Kind::All:
        for (const auto &child : f.children())
            if (!matches(child, labels))
                return false;
        return true;
    case Filter::Kind::Any:
        for (const auto &child : f.children())
            if (matches(child, labels))
                return true;
        return false;
    }
    return false;
}

// Validate lengths before passing trusted native/legacy index data to the old reader.
class Reader {
public:
    explicit Reader(const std::string &path) : in(path, std::ios::binary) {
        if (!in)
            throw std::runtime_error("cannot open index: " + path);
        in.seekg(0, std::ios::end);
        remaining = static_cast<std::uint64_t>(in.tellg());
        in.seekg(0);
    }
    template <class T> T read() {
        T value{};
        require(sizeof(T));
        in.read(reinterpret_cast<char *>(&value), sizeof(T));
        if (!in)
            throw std::runtime_error("truncated index");
        return value;
    }
    void skip(std::uint64_t n) {
        require(n);
        in.seekg(static_cast<std::streamoff>(n), std::ios::cur);
        if (!in)
            throw std::runtime_error("truncated index");
    }
    void require(std::uint64_t n) {
        if (n > remaining)
            throw std::runtime_error("truncated or invalid index");
        remaining -= n;
    }
    std::ifstream in;
    std::uint64_t remaining = 0;
};
void validate_tree(Reader &r, std::size_t n, std::size_t &budget) {
    if (!budget--)
        throw std::runtime_error("invalid trie size");
    const int marker = r.read<int>();
    if (marker == -1)
        throw std::runtime_error("unexpected null trie node");
    if (marker != 1)
        throw std::runtime_error("invalid trie marker");
    r.skip(7 * sizeof(int) + sizeof(bool));
    const auto ids = r.read<std::size_t>();
    if (ids > n)
        throw std::runtime_error("invalid trie ID count");
    for (std::size_t i = 0; i < ids; ++i) {
        int id = r.read<int>();
        if (id < 0 || static_cast<std::size_t>(id) >= n)
            throw std::runtime_error("invalid trie ID");
    }
    const auto children = r.read<std::size_t>();
    if (children > budget)
        throw std::runtime_error("invalid trie child count");
    for (std::size_t i = 0; i < children; ++i) {
        r.read<int>();
        validate_tree(r, n, budget);
    }
}
BuildOptions validate_file(Reader &r, std::size_t &n) {
    BuildOptions o;
    o.max_neighbors = r.read<int>();
    o.ef_construction = r.read<int>();
    o.dimension = r.read<int>();
    o.capacity = r.read<std::size_t>();
    n = r.read<std::size_t>();
    r.read<double>();
    r.read<double>();
    o.label_count = r.read<int>();
    check_options(o, n);
    r.skip(n * o.dimension * sizeof(float));
    r.skip(n * sizeof(int) + o.capacity * sizeof(bool));
    const auto rows = r.read<std::size_t>();
    if (rows < n || rows > o.capacity)
        throw std::runtime_error("invalid label row count");
    for (std::size_t i = 0; i < rows; ++i) {
        const auto size = r.read<std::size_t>();
        if (size > static_cast<std::size_t>(o.label_count) + 1)
            throw std::runtime_error("invalid label row size");
        r.skip(size * sizeof(int));
    }
    r.skip(o.capacity * (2 * sizeof(int) + sizeof(unsigned long long)));
    auto budget = static_cast<std::size_t>(r.remaining / sizeof(int));
    validate_tree(r, n, budget);
    const auto keys = r.read<std::size_t>();
    if (keys > n)
        throw std::runtime_error("invalid ID map size");
    r.skip(keys * 2 * sizeof(int));
    for (std::size_t i = 0; i < n; ++i) {
        const auto id = r.read<std::size_t>();
        const int layer = r.read<int>(), level = r.read<int>();
        if (id != i || layer < 0 || level < 0 || level > 1024)
            throw std::runtime_error("invalid graph record");
        r.skip((static_cast<std::uint64_t>(layer) + 1) * (level + 2) *
               (o.max_neighbors * sizeof(tableint) + sizeof(linklistsizeint)));
    }
    return o;
}
} // namespace

Filter::Filter(Kind kind, Labels labels, std::vector<Filter> children)
    : kind_(kind), labels_(normalize(std::move(labels))), children_(std::move(children)) {}
Filter Filter::containment(Labels x) { return {Kind::Containment, std::move(x), {}}; }
Filter Filter::overlap(Labels x) { return {Kind::Overlap, std::move(x), {}}; }
Filter Filter::equality(Labels x) { return {Kind::Equality, std::move(x), {}}; }
Filter Filter::all(std::vector<Filter> x) { return {Kind::All, {}, std::move(x)}; }
Filter Filter::any(std::vector<Filter> x) { return {Kind::Any, {}, std::move(x)}; }

struct Index::Impl {
    Engine engine;
    BuildOptions options;
    std::size_t live = 0;
    detail::LabelRegistry labels;

    Filter translate(const Filter &f) const {
        if (f.kind() == Filter::Kind::All || f.kind() == Filter::Kind::Any) {
            std::vector<Filter> children;
            children.reserve(f.children().size());
            for (const auto &child : f.children())
                children.push_back(translate(child));
            return f.kind() == Filter::Kind::All ? Filter::all(std::move(children))
                                                 : Filter::any(std::move(children));
        }
        Labels internal;
        internal.reserve(f.labels().size());
        for (int label : f.labels()) {
            const auto found = labels.mapping.find(label);
            if (found != labels.mapping.end())
                internal.push_back(found->second);
            else if (f.kind() != Filter::Kind::Overlap)
                return Filter::any({});
        }
        switch (f.kind()) {
        case Filter::Kind::Containment:
            return Filter::containment(std::move(internal));
        case Filter::Kind::Overlap:
            return Filter::overlap(std::move(internal));
        default:
            return Filter::equality(std::move(internal));
        }
    }

    void insert(Id id, const std::vector<float> &vector, Labels external, bool allow_new);

    void reset_stamps() {
        std::fill_n(engine->visited_array, engine->maxNum, 0u);
        std::vector<TrieNode *> stack{engine->root};
        while (!stack.empty()) {
            auto *node = stack.back();
            stack.pop_back();
            node->visit = node->flag = 0;
            for (auto &c : node->child)
                stack.push_back(c.second);
        }
        engine->tag = engine->tag2 = 0;
    }
    void advance_tags() {
        // TrieNode stores signed stamps. Clear before reaching their limit.
        if (engine->tag >= INT_MAX - 2 || engine->tag2 >= INT_MAX - 2)
            reset_stamps();
        ++engine->tag;
        ++engine->tag2;
    }

    Nodes cover(const Filter &f) {
        Nodes result;
        switch (f.kind()) {
        case Filter::Kind::Containment:
            if (!f.labels().empty() && f.labels().back() > options.label_count)
                return {};
            if (f.labels().empty())
                result.push_back(engine->root);
            else
                engine->cont_search(f.labels(), result);
            break;
        case Filter::Kind::Overlap: {
            std::vector<Nodes> groups;
            for (int x : f.labels())
                if (x <= options.label_count)
                    groups.push_back(engine->label_node[x]);
            engine->or_search(groups, result);
            break;
        }
        case Filter::Kind::Equality: {
            if (!f.labels().empty() && f.labels().back() > options.label_count)
                return {};
            auto *node = engine->root;
            for (int x : f.labels()) {
                const auto it = node->child.find(x);
                if (it == node->child.end())
                    return {};
                node = it->second;
            }
            for (auto &child : node->child)
                if (child.second->child.empty())
                    result.push_back(child.second);
            break;
        }
        case Filter::Kind::All:
        case Filter::Kind::Any: {
            if (f.children().empty()) {
                if (f.kind() == Filter::Kind::All)
                    result.push_back(engine->root);
                break;
            }
            std::vector<Nodes> groups;
            for (const auto &child : f.children())
                groups.push_back(cover(child));
            if (f.kind() == Filter::Kind::All)
                engine->and_search(groups, result);
            else
                engine->or_search(groups, result);
            break;
        }
        }
        return result;
    }
};

Index::Index(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Index::~Index() = default;
Index::Index(Index &&) noexcept = default;
Index &Index::operator=(Index &&) noexcept = default;

Index Index::build(const std::vector<float> &vectors, const std::vector<Labels> &labels,
                   const std::vector<Id> &ids, BuildOptions options) {
    const auto n = ids.size();
    if (!options.capacity)
        options.capacity = n;
    check_options(options, n);
    if (labels.size() != n || vectors.size() != n * options.dimension)
        throw std::invalid_argument("vectors, IDs and labels must have matching counts");
    for (float x : vectors)
        if (!std::isfinite(x))
            throw std::invalid_argument("vectors must be finite");
    {
        std::unordered_set<Id> seen;
        for (Id id : ids)
            if (id < 0 || !seen.insert(id).second)
                throw std::invalid_argument("IDs must be unique and nonnegative");
    }
    bool needs_normalization = false;
    for (const auto &row : labels) {
        for (int label : row)
            if (label <= 0 || label > options.label_count)
                throw std::invalid_argument("label exceeds the fixed positive universe");
        needs_normalization |= !std::is_sorted(row.begin(), row.end()) ||
                               std::adjacent_find(row.begin(), row.end()) != row.end();
    }
    std::vector<Labels> canonical;
    const auto *prepared_labels = &labels;
    if (needs_normalization) {
        canonical = labels;
        for (auto &row : canonical)
            row = normalize(std::move(row));
        prepared_labels = &canonical;
    }
    auto p = std::make_unique<Impl>();
    p->options = options;
    p->live = n;
    p->labels.initialize(options.label_count, options.label_seed);
    for (const auto &row : *prepared_labels)
        for (int label : row)
            p->labels.mark_initial(label);
    p->engine.reset(new TrieIndex(static_cast<int>(options.dimension), n, options.capacity,
                                  const_cast<float *>(vectors.data()), const_cast<Id *>(ids.data()),
                                  const_cast<std::vector<Labels> &>(*prepared_labels),
                                  options.max_neighbors, options.ef_construction,
                                  options.label_count, options.alpha));
    p->engine->item_labels.resize(options.capacity);
    p->engine->trie_label_load(p->engine->root);
    p->engine->label_enhance_num = options.enhanced_labels;
    p->engine->enhance_index(options.enhanced_labels);
    return Index(std::move(p));
}

std::vector<Neighbor> Index::search(const std::vector<float> &query, const Filter &external_filter,
                                    SearchOptions options) {
    check_vector(query, impl_->options.dimension);
    if (options.k > INT_MAX || options.ef <= 0)
        throw std::invalid_argument("invalid k or ef");
    if (!options.k || !impl_->live)
        return {};
    std::optional<Filter> translated;
    if (!impl_->labels.identity)
        translated = impl_->translate(external_filter);
    const auto &filter = translated ? *translated : external_filter;
    auto &e = *impl_->engine;
    impl_->advance_tags();
    const int ef = std::max(options.ef, static_cast<int>(options.k));
    Nodes nodes;
    auto *q = const_cast<float *>(query.data());
    if ((filter.kind() == Filter::Kind::Containment || filter.kind() == Filter::Kind::Overlap) &&
        !filter.labels().empty()) {
        Labels labels;
        for (int x : filter.labels())
            if (x <= impl_->options.label_count)
                labels.push_back(x);
        if (filter.kind() == Filter::Kind::Containment && labels.size() != filter.labels().size())
            return {};
        if (labels.empty())
            return {};
        const std::string type =
            filter.kind() == Filter::Kind::Containment ? "containment" : "overlap";
        if (type == "containment")
            e.superset_search(labels, nodes);
        else
            e.overlap_search(labels, nodes);
        std::sort(nodes.begin(), nodes.end(),
                  [](const TrieNode *a, const TrieNode *b) { return a->size > b->size; });
        if (nodes.size() > static_cast<std::size_t>(ef))
            nodes.resize(ef);
        return unpack(e.searchKnn(q, nodes, static_cast<int>(options.k), ef, labels, type));
    }
    nodes = impl_->cover(filter);
    if (nodes.empty())
        return {};
    Nodes entries;
    std::unordered_set<TrieNode *> seen;
    for (auto *node : nodes)
        if (seen.insert(node->pre_index).second)
            entries.push_back(node->pre_index);
    e.get_key_ancester(entries);
    std::sort(entries.begin(), entries.end(),
              [](const TrieNode *a, const TrieNode *b) { return a->size > b->size; });
    if (entries.size() > static_cast<std::size_t>(ef))
        entries.resize(ef);
    return unpack(e.searchKnnGeneral(q, entries, static_cast<int>(options.k), ef, nodes));
}

std::vector<Neighbor> Index::scan(const std::vector<float> &query, const Filter &external_filter,
                                  std::size_t k) {
    check_vector(query, impl_->options.dimension);
    if (k > INT_MAX)
        throw std::invalid_argument("invalid k");
    Heap result;
    if (!k)
        return {};
    std::optional<Filter> translated;
    if (!impl_->labels.identity)
        translated = impl_->translate(external_filter);
    const auto &filter = translated ? *translated : external_filter;
    auto &e = *impl_->engine;
    for (std::size_t i = 0; i < e.eleCount; ++i) {
        if (e.isDeleted[i] || !matches(filter, e.item_labels[i]))
            continue;
        const float distance =
            e.fstdistfunc_(query.data(), e.getDataByInternalId(i), e.dist_func_param_);
        result.emplace(distance, e.keyList_[i]);
        if (result.size() > k)
            result.pop();
    }
    return unpack(std::move(result));
}

void Index::insert(Id id, const std::vector<float> &vector, Labels labels) {
    impl_->insert(id, vector, std::move(labels), false);
}
void Index::insert_with_new_labels(Id id, const std::vector<float> &vector, Labels labels) {
    impl_->insert(id, vector, std::move(labels), true);
}
void Index::Impl::insert(Id id, const std::vector<float> &vector, Labels external, bool allow_new) {
    check_vector(vector, options.dimension);
    external = normalize(std::move(external));
    auto &e = *engine;
    if (id < 0 || e.key2Id.count(id))
        throw std::invalid_argument("ID is negative or already exists");
    if (e.eleCount == e.maxNum)
        throw std::length_error("index capacity exhausted");
    detail::LabelRegistry::Insertion pending(labels, external, allow_new);
    const auto internal = e.eleCount;
    e.keyList_[internal] = id;
    e.key2Id[id] = static_cast<int>(internal);
    e.insertElement(static_cast<int>(internal), pending.labels,
                    reinterpret_cast<char *>(const_cast<float *>(vector.data())));
    // New label combinations need samples before the next containment query.
    auto *node = e.root;
    for (int x : e.item_labels[internal]) {
        node = node->child.at(x);
        if (node->sample_id < 0)
            node->sample_id = static_cast<int>(internal);
    }
    pending.commit();
    ++live;
}

const std::unordered_map<int, int> &Index::label_map() const noexcept {
    return impl_->labels.mapping;
}
std::size_t Index::available_label_count() const noexcept { return impl_->labels.available(); }

bool Index::erase(Id id) {
    auto &e = *impl_->engine;
    auto it = e.key2Id.find(id);
    if (it == e.key2Id.end() || e.isDeleted[it->second])
        return false;
    e.deleteElement(it->second);
    --impl_->live;
    return true;
}

void Index::save(const std::string &path) const {
    auto &e = *impl_->engine;
    e.saveIndex(path);
    std::ofstream out(path, std::ios::binary | std::ios::app);
    out << "DYLAN_API_V2\n"
        << impl_->options.alpha << ' ' << impl_->options.enhanced_labels << ' '
        << impl_->options.label_seed << '\n'
        << e.level_generator_ << '\n';
    impl_->labels.save(out);
    out.flush();
    if (!out)
        throw std::runtime_error("failed to save index: " + path);
}

Index Index::load(const std::string &path, double legacy_alpha, int legacy_enhanced_labels,
                  int legacy_max_label) {
    Reader reader(path);
    std::size_t n = 0;
    auto p = std::make_unique<Impl>();
    p->options = validate_file(reader, n);
    p->options.alpha = legacy_alpha;
    p->options.enhanced_labels = legacy_enhanced_labels;
    std::default_random_engine rng(100);
    bool has_mapping = false;
    // Legacy loading starts the RNG from seed 100, preserve that behavior for raw files.
    if (reader.remaining) {
        std::string magic;
        std::getline(reader.in, magic);
        has_mapping = magic == "DYLAN_API_V2";
        if ((magic != "DYLAN_API_V1" && !has_mapping) ||
            !(reader.in >> p->options.alpha >> p->options.enhanced_labels) ||
            (has_mapping && !(reader.in >> p->options.label_seed)) ||
            !(reader.in >> std::ws >> rng >> std::skipws))
            throw std::runtime_error("invalid native metadata trailer: " + magic);
    }
    if (legacy_max_label) {
        if (has_mapping && legacy_max_label != p->options.label_count)
            throw std::invalid_argument(
                "cannot change the reserved label maximum of a mapped index");
        if (legacy_max_label < 1 || legacy_max_label > p->options.label_count)
            throw std::invalid_argument("legacy maximum may only correct a drifted stored maximum");
        p->options.label_count = legacy_max_label;
    }
    check_options(p->options, n);
    p->labels.initialize(p->options.label_count, p->options.label_seed);
    if (has_mapping)
        p->labels.load(reader.in);
    p->engine.reset(TrieIndex::loadIndex(path));
    auto &e = *p->engine;
    // The legacy reader increments the universe on every load. Native API keeps it fixed.
    e.label_num = e.max_label = p->options.label_count;
    e.label_node.assign(e.label_num + 2, {});
    e.trie_label_load(e.root);
    e.alpha = p->options.alpha;
    e.label_enhance_num = p->options.enhanced_labels;
    e.level_generator_ = rng;
    e.eng_label.seed(37);
    e.labelhash_generator_ = std::uniform_int_distribution<int>(1, e.label_num);
    for (int i = 1; i <= e.label_num; ++i)
        e.hashedlabel[i] =
            i <= e.label_the
                ? i - 1
                : e.labelhash_generator_(e.eng_label) % (64 - e.label_the) + e.label_the;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j + 1 < e.item_labels[i].size(); ++j) {
            const int label = e.item_labels[i][j];
            if (label <= 0 || label > e.label_num)
                throw std::runtime_error("stored object label exceeds the declared maximum");
            if (has_mapping) {
                if (!p->labels.occupied(label))
                    throw std::runtime_error("stored object label is missing from the label map");
            } else {
                p->labels.mark_initial(label);
            }
        }
        e.labelhash[i] = e.setBits(e.item_labels[i]);
        if (!e.isDeleted[i])
            ++p->live;
    }
    p->reset_stamps();
    return Index(std::move(p));
}

void Index::reorder_by_trie() {
    auto &e = *impl_->engine;
    // Retired graph layers can contain stale storage after shared-index maintenance.
    // They have no navigable edges; clear their counts before the legacy ID remapper.
    for (std::size_t id = 0; id < e.eleCount; ++id)
        for (int layer = 0; layer <= e.max_layer[id]; ++layer)
            if (!e.belong[id][layer])
                for (int level = 0; level <= e.max_layer_i[id]; ++level)
                    e.setListCount(e.get_linklist(id, layer, level), 0);
    e.reorderByTrieDfs();
}
std::size_t Index::size() const noexcept { return impl_->engine->eleCount; }
std::size_t Index::live_size() const noexcept { return impl_->live; }
const BuildOptions &Index::options() const noexcept { return impl_->options; }
} // namespace dylan
