#pragma once

#include <algorithm>
#include <cstdint>
#include <istream>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dylan::detail {
class LabelRegistry {
public:
    std::unordered_map<int, int> mapping;
    bool identity = true;

    void initialize(int maximum, std::uint32_t seed) {
        rng_.seed(seed);
        free_ids_.resize(maximum);
        std::iota(free_ids_.begin(), free_ids_.end(), 1);
        positions_.resize(static_cast<std::size_t>(maximum) + 1);
        std::iota(positions_.begin(), positions_.end(), -1);
    }
    std::size_t available() const noexcept { return free_ids_.size(); }
    bool occupied(int id) const { return positions_.at(id) == -1; }
    void mark_initial(int label) {
        if (mapping.emplace(label, label).second)
            remove_free(label);
    }

    // Reserve assignments before graph insertion; roll back if insertion does not finish.
    class Insertion {
    public:
        Insertion(LabelRegistry &owner, const std::vector<int> &external, bool allow_new)
            : owner_(owner) {
            std::size_t missing = 0;
            for (int label : external) {
                if (owner.mapping.count(label))
                    continue;
                ++missing;
                if (!allow_new && (static_cast<std::size_t>(label) >= owner.positions_.size() ||
                                   owner.occupied(label)))
                    throw std::invalid_argument(
                        "unseen label has no free identity ID; use insert_with_new_labels");
            }
            if (missing > owner.available())
                throw std::length_error("reserved label IDs exhausted");
            labels.reserve(external.size());
            changes_.reserve(missing);
            if (allow_new && missing)
                next_rng_ = owner.rng_;
            try {
                for (int label : external) {
                    auto found = owner.mapping.find(label);
                    if (found != owner.mapping.end()) {
                        labels.push_back(found->second);
                        continue;
                    }
                    const int internal =
                        allow_new ? owner.free_ids_[std::uniform_int_distribution<std::size_t>(
                                        0, owner.free_ids_.size() - 1)(*next_rng_)]
                                  : label;
                    owner.mapping.emplace(label, internal);
                    changes_.emplace_back(label, owner.positions_[internal]);
                    owner.remove_free(internal);
                    labels.push_back(internal);
                }
                std::sort(labels.begin(), labels.end());
            } catch (...) {
                rollback();
                throw;
            }
        }
        ~Insertion() {
            if (!committed_)
                rollback();
        }
        Insertion(const Insertion &) = delete;
        Insertion &operator=(const Insertion &) = delete;
        void commit() noexcept {
            for (const auto &change : changes_)
                owner_.identity &= owner_.mapping.find(change.first)->second == change.first;
            if (next_rng_)
                owner_.rng_ = *next_rng_;
            committed_ = true;
        }
        std::vector<int> labels;

    private:
        void rollback() noexcept {
            for (auto it = changes_.rbegin(); it != changes_.rend(); ++it) {
                const int internal = owner_.mapping.find(it->first)->second;
                owner_.mapping.erase(it->first);
                owner_.free_ids_.push_back(internal);
                std::swap(owner_.free_ids_[it->second], owner_.free_ids_.back());
                owner_.positions_[owner_.free_ids_.back()] = owner_.free_ids_.size() - 1;
                owner_.positions_[internal] = it->second;
            }
            changes_.clear();
        }
        LabelRegistry &owner_;
        std::vector<std::pair<int, int>> changes_;
        std::optional<std::mt19937> next_rng_;
        bool committed_ = false;
    };

    void save(std::ostream &out) const {
        out << rng_ << '\n' << mapping.size() << '\n';
        std::vector<std::pair<int, int>> ordered(mapping.begin(), mapping.end());
        std::sort(ordered.begin(), ordered.end());
        for (const auto &entry : ordered)
            out << entry.first << ' ' << entry.second << '\n';
        out << free_ids_.size() << '\n';
        for (int id : free_ids_)
            out << id << ' ';
        out << '\n';
    }
    void load(std::istream &in) {
        std::size_t size = 0;
        if (!(in >> std::ws >> rng_ >> std::skipws >> size) || size > available())
            throw std::runtime_error("invalid label map metadata");
        for (std::size_t i = 0; i < size; ++i) {
            int external = 0, internal = 0;
            if (!(in >> external >> internal) || external <= 0 || internal <= 0 ||
                static_cast<std::size_t>(internal) >= positions_.size() || occupied(internal) ||
                !mapping.emplace(external, internal).second)
                throw std::runtime_error("invalid or duplicate label mapping");
            remove_free(internal);
            identity &= external == internal;
        }
        if (!(in >> size) || size != available())
            throw std::runtime_error("invalid free label count");
        for (int id : free_ids_)
            positions_[id] = -2;
        for (std::size_t i = 0; i < size; ++i) {
            int id = 0;
            if (!(in >> id) || id <= 0 || static_cast<std::size_t>(id) >= positions_.size() ||
                positions_[id] != -2)
                throw std::runtime_error("invalid or duplicate free label ID");
            free_ids_[i] = id;
            positions_[id] = static_cast<int>(i);
        }
    }

private:
    void remove_free(int id) {
        const int position = positions_[id];
        const int last = free_ids_.back();
        free_ids_[position] = last;
        positions_[last] = position;
        free_ids_.pop_back();
        positions_[id] = -1;
    }
    std::vector<int> free_ids_;
    std::vector<int> positions_;
    std::mt19937 rng_;
};
} // namespace dylan::detail
