# Third-party headers

`hnswlib/` contains the hnswlib headers supplied with the research implementation. They provide distance functions, SIMD helpers and graph-related types. The vendored headers are preserved without changes; the DYLAN trie and shared-graph implementation is in `src/detail/engine.hpp`.
