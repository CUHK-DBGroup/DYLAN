# DYLAN

Research implementation for **Dynamic Graph Indexing for Label-Filtered Approximate Nearest Neighbor Search**.

DYLAN combines an ordered label trie with shared HNSW-style graphs. It supports label-filtered approximate nearest neighbor search, nested Boolean filters, insertion, lazy deletion, and index persistence. Distances are **squared L2**.

[C++ API](include/dylan/index.hpp)

## Build

The supported build environment is Linux with GCC, C++17, CMake 3.22 or newer, and OpenMP. Python 3 is needed only for the optional demo data generator. The hnswlib headers are included in `third_party/`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDYLAN_ORIGINAL_SIMD=OFF
cmake --build build -j 4
./build/dylan_example
```

The command above disables manual SIMD for a portable first run. To use the original AVX, AVX2, AVX512F and FMA compiler options on a CPU supporting **all four**, configure with `-DDYLAN_ORIGINAL_SIMD=ON` (the project default). Use a separate build directory when comparing configurations. Scalar and SIMD timings are not interchangeable.

| CMake option | Default | Meaning |
| --- | --- | --- |
| `DYLAN_ORIGINAL_SIMD` | `ON` | Original x86 SIMD and FMA compilation options |
| `DYLAN_SANITIZE` | `OFF` | AddressSanitizer and UndefinedBehaviorSanitizer |
| `DYLAN_BUILD_CLI` | `ON` | Build the `dylan` executable |
| `DYLAN_BUILD_EXAMPLES` | `ON` | Build the C++ example |

For a library-only build, disable CLI and examples. Python is then unnecessary. To embed the source in another CMake project:

```cmake
add_subdirectory(path/to/dylan)
target_link_libraries(your_target PRIVATE dylan::dylan)
```

## Command-line quick start

Run these commands from the repository root. The demo creates 12 two-dimensional base vectors and reserves capacity for 16 objects and 8 internal label IDs.

```bash
python3 examples/generate_demo.py --output data/demo
mkdir -p results

./build/dylan build \
  --vectors data/demo/base.fvecs --labels data/demo/base.csv \
  --label-num 8 --capacity 16 --m 8 --ef-construction 40 \
  --output results/demo.ind3

./build/dylan search \
  --index results/demo.ind3 \
  --vectors data/demo/query.fvecs --labels data/demo/query.csv \
  --match containment --k 3 --ef 64 --output results/hits.tsv

./build/dylan insert \
  --index results/demo.ind3 \
  --vectors data/demo/new.fvecs --labels data/demo/new.csv \
  --ids data/demo/new_ids.txt --allow-new-labels 1 \
  --output results/updated.ind3

./build/dylan search \
  --index results/updated.ind3 \
  --vectors data/demo/query.fvecs --labels data/demo/new_query.csv \
  --match overlap --k 3 --ef 64 --output results/new_label_hits.tsv

./build/dylan delete --index results/updated.ind3 \
  --ids data/demo/delete_ids.txt --output results/deleted.ind3
./build/dylan reorder --index results/deleted.ind3 --output results/reordered.ind3
```

Search accepts `--match containment|overlap|equality`. Add `--exact 1` for an exact filtered scan. Output columns are `query_id`, `id`, and `squared_l2` in a tab-separated file. `./build/dylan --help` lists commands. Nested filters are available through the C++ API.

### Input formats

- **Vectors (`.fvecs`)**: each row contains an `int32` dimension followed by that many `float32` values. All rows must have the same dimension; all values must be finite. The demo writes little-endian files for the supported Linux x86 environment.
- **Labels (`.csv`)**: one row per vector, containing comma-separated positive integer labels without a header. An empty row means an empty label set. Labels are sorted and deduplicated by the API.
- **Object IDs (`.txt`)**: unique nonnegative integers separated by whitespace. IDs are optional at build time, defaulting to `0..n-1`; they are required for insertion.

Vector and label row counts must match. Reserve capacity at construction for later insertions. Deletion does not release capacity or permit object ID reuse.

## C++ API

The public interface is a single header, `dylan/index.hpp`; internal trie nodes and graph storage are private implementation details.

```cpp
#include <dylan/index.hpp>

using dylan::Filter;
dylan::BuildOptions options;
options.dimension = 2;
options.capacity = 100;
options.label_count = 1000;

auto index = dylan::Index::build(
    {0, 0, 1, 1, 2, 2}, {{1}, {1, 2}, {3}}, {10, 20, 30}, options);

index.insert_with_new_labels(40, {0.5f, 0.5f}, {1, 90001});
auto filter = Filter::all({
    Filter::overlap({1, 3}),
    Filter::any({Filter::containment({90001}), Filter::equality({1, 2})})
});
auto approximate = index.search({0, 0}, filter, {10, 100}); // k, ef
auto exact = index.scan({0, 0}, filter, 10);
index.erase(40);
index.save("example.ind3");
auto restored = dylan::Index::load("example.ind3");
```

| Operation | Semantics |
| --- | --- |
| `Filter::containment(Q)` | Every label in `Q` occurs in the object's label set |
| `Filter::overlap(Q)` | At least one label in `Q` occurs in the object's label set |
| `Filter::equality(Q)` | The object's label set equals `Q` |
| `Filter::all(filters)` / `any(filters)` | Recursive AND / OR |
| `search(query, filter, {k, ef})` | ANN results as `{id, distance}`, sorted by distance; effective `ef >= k` |
| `scan(query, filter, k)` | Exact filtered top-k; ties ordered by object ID |
| `insert(id, vector, labels)` | Insert using existing mappings or an unused same-number internal ID |
| `insert_with_new_labels(...)` | Allocate random free internal IDs for unseen labels |
| `erase(id)` | Lazy deletion; returns false for absent or already deleted IDs |
| `size()` / `live_size()` | Total inserted / currently undeleted object counts |
| `reorder_by_trie()` | Reorder internal storage while preserving public IDs and label mappings |

Empty containment and empty `all` match every object; empty overlap and empty `any` match none. Empty equality matches only objects without labels. Queries never register new labels. The index owns its storage, is movable, and cannot be copied.

### Build parameters

`dimension` is required for the input vectors; `capacity=0` uses the initial object count. Other defaults are `label_count=1`, `max_neighbors=32`, `ef_construction=100`, `alpha=2`, `enhanced_labels=20`, and `label_seed=37`. Set `label_count` explicitly for your data. `alpha=1` materializes every trie node's graph; `alpha=2` uses shared graphs with dynamic maintenance. `enhanced_labels=0` disables graph enhancement.

### New labels

`label_count` / `--label-num` reserves internal IDs in `[1, label_count]`. Build-time labels must lie in this range. Only labels actually observed during construction are marked occupied, initially using identity mappings.

`insert_with_new_labels` reuses known mappings and uniformly randomly assigns each unseen positive external label to a free internal ID. External labels may exceed `label_count`. Inspect mappings with `label_map()` and free capacity with `available_label_count()`. All queries and insertions use **external labels**. Mappings are stable and are not reclaimed on deletion.

Ordinary `insert` accepts an unseen label only if its same-number internal ID is free; otherwise use the new-label interface. Exhausted object capacity or label IDs raise `std::length_error`; invalid inputs raise `std::invalid_argument`.


## Appendix

The supplementary appendix is available in [517-Appendix.pdf](517-Appendix.pdf).

## Layout

```text
include/dylan/index.hpp    Public C++ API
src/index.cpp             API implementation and persistence
src/detail/               Trie/graph core and label registry
apps/dylan.cpp            Command-line tool
examples/                 C++ example and demo input generator
third_party/hnswlib/       Bundled C++ dependency headers
```
