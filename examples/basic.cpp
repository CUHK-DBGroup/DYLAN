#include <dylan/index.hpp>
#include <iostream>
int main() {
    dylan::BuildOptions options;
    options.dimension = 2;
    options.capacity = 4;
    options.label_count = 8;
    options.enhanced_labels = 0;
    auto index = dylan::Index::build({0, 0, 1, 1, 2, 2}, {{1}, {1, 2}, {3}}, {10, 20, 30}, options);
    index.insert_with_new_labels(40, {0.5f, 0.5f}, {1, 90001});
    std::cout << "90001 -> " << index.label_map().at(90001) << '\n';
    auto predicate =
        dylan::Filter::any({dylan::Filter::containment({1, 90001}), dylan::Filter::overlap({3})});
    for (const auto &hit : index.search({0, 0}, predicate, {2, 32}))
        std::cout << hit.id << " " << hit.distance << '\n';
    index.erase(40);
}
