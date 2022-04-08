//
// Created by lepet on 4/6/2022.
//

#include <thrust/find.h>
#include <thrust/zip_function.h>
#include <thrust/binary_search.h>

#include <chrono>

#include "states.h"

using Node = uint64_t;
using Value = uint8_t;
using Path = uint64_t;
enum Edge : uint8_t {
    NORTH = 0,
    EAST = 1,
    SOUTH = 2,
    WEST = 3,
};

using States = empire::States<Node, Path, Value>;
using HostStates = empire::HostStates<Node, Path, Value>;

struct Expansion {
    Node target;

    explicit Expansion(Node target) : target(target) {}

    // Heuristic
    [[nodiscard]]
    __host__ __device__
    Value heuristic(Node node) const {
        Node mask = 0xf;
        Value result = 0;
        for (int i = 0; i < 16; ++i, mask <<= 4) {
            auto x = (mask & node) >> (4 * i);
            auto y = (mask & target) >> (4 * i);
            if (x != y && x != 0) ++result;
        }
        return result;
    }

    // Expansion
    __host__ __device__
    void operator()(
            const Node& node,
            const Value& step,
            const Path& parent,
            const size_t& direction,
            Node& out_node,
            Value& out_step,
            Value& out_score,
            Path& out_parent
    ) const {
        Node mask = 0xf;
        int x = -1, y = -1;
        for (int i = 0; i < 16; ++i, mask <<= 4) {
            if ((node & mask) == 0) {
                x = i / 4;
                y = i % 4;
                break;
            }
        }

        if (direction == NORTH && x > 0) {
            auto selected = node & (mask >> 16);
            out_node = (node | (selected << 16)) ^ selected;
            out_step = step + 1;
            out_score = step + heuristic(node);
            out_parent = parent | (direction << (2 * step));
            return;
        }

        if (direction == EAST && y < 3) {
            auto selected = node & (mask << 4);
            out_node = (node | (selected >> 4)) ^ selected;
            out_step = step + 1;
            out_score = step + heuristic(node);
            out_parent = parent | (direction << (2 * step));
            return;
        }

        if (direction == SOUTH && x < 3) {
            auto selected = node & (mask << 16);
            out_node = (node | (selected >> 16)) ^ selected;
            out_step = step + 1;
            out_score = step + heuristic(node);
            out_parent = parent | (direction << (2 * step));
            return;
        }

        if (direction == WEST && y > 0) {
            auto selected = node & (mask >> 4);
            out_node = (node | (selected << 4)) ^ selected;
            out_step = step + 1;
            out_score = step + heuristic(node);
            out_parent = parent | (direction << (2 * step));
            return;
        }

        out_node = 0;
    }
};

struct NodeComp {
    __host__ __device__
    bool operator()(const Node& lhs, const Node& rhs) {
        if (lhs == 0) return false;
        if (rhs == 0) return true;
        return lhs < rhs;
    }
};

struct StateReduce {
    template<typename Tuple>
    __host__ __device__
    Tuple operator()(const Tuple& lhs, const Tuple& rhs) {
        return thrust::get<1>(lhs) < thrust::get<1>(rhs) ? lhs : rhs;
    }
};

struct StateSelect {
    __host__ __device__
    void operator()(
            const Node& control_node,
            const Value& control_score,
            const Value& test_score,
            Node& out_node
    ) {
        if (control_node == out_node && control_score <= test_score)
            out_node = 0;
    }
};

struct StateRemove {
    template<typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& tuple) {
        return thrust::get<0>(tuple) == 0;
    }
};

auto make_expand_iter(const States& input, States& output, size_t stride, size_t x = 0) {
    using namespace thrust::placeholders;
    auto expand_counter = thrust::make_counting_iterator(x);
    auto stride_counter = thrust::make_transform_iterator(expand_counter, _1 % stride);
    auto direction_iter = thrust::make_transform_iterator(expand_counter, _1 / stride);

    auto input_nodes_iter = thrust::make_permutation_iterator(input.nodes.begin(), stride_counter);
    auto input_steps_iter = thrust::make_permutation_iterator(input.steps.begin(), stride_counter);
    auto input_parents_iter = thrust::make_permutation_iterator(input.parents.begin(), stride_counter);

    return thrust::make_zip_iterator(
            input_nodes_iter,
            input_steps_iter,
            input_parents_iter,
            direction_iter,
            output.nodes.begin() + x,
            output.steps.begin() + x,
            output.scores.begin() + x,
            output.parents.begin() + x
    );
}

auto make_selection_iter(
        const thrust::device_vector<Node>& indices,
        const States& close,
        States& dedup,
        size_t x = 0
) {
    auto indices_iter = indices.begin() + x;
    return thrust::make_zip_iterator(
            thrust::make_permutation_iterator(close.keys(), indices_iter),
            thrust::make_permutation_iterator(close.keys_score(), indices_iter),
            dedup.keys_score(x),
            dedup.keys(x)
    );
}

__host__ __device__
uint32_t hash(Node node) {
    uint64_t hash = node;
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;
    return hash;
}

struct HashStateRemove {
    const uint64_t* table;
    const size_t size;

    HashStateRemove(const uint64_t* table, size_t size) : table(table), size(size) {}

    template<typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& tuple) {
        Node node = thrust::get<0>(tuple);

        auto signature = hash(node);
        auto key = signature % size;
        auto value = table[key];

        if (value != 0 && (value >> 32) == signature) {
            Value score = thrust::get<2>(tuple);
            return Value(value) <= score;
        }

        return false;
    }
};

struct HashInsert {
    uint64_t* table;
    const size_t size;

    HashInsert(uint64_t* table, size_t size) : table(table), size(size) {}

    __device__
    void operator()(const Node& node, const Value& score) const {
        uint64_t signature = hash(node);
        auto key = signature % size;

        uint64_t value = (signature << 32) | score;
        atomicExch(table + key, value);
    }
};

void print_node(Node node) {
    Node mask = 0xf;
    for (auto i = 0u; i < 4; ++i) {
        for (auto j = 0u; j < 4; ++j) {
            auto n = (mask & node) >> ((4 * i + j) * 4);
            std::cout << n << '\t';
            mask <<= 4;
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

int main() {
    Node start = 0xFEDCBA9876543210;
    Node target = 0xAECDF941B8527306;
    Expansion expansion(target);

    std::chrono::high_resolution_clock::duration
            expand_duration(0),
            dedup_duration(0),
            close_duration(0),
            open_duration(0);

    thrust::device_vector<uint64_t> hashtable(16777213);
    HashStateRemove hash_state_remove(thrust::raw_pointer_cast(hashtable.data()), hashtable.size());
    HashInsert hash_insert(thrust::raw_pointer_cast(hashtable.data()), hashtable.size());

    States open, merge, expand, dedup;
    // thrust::device_vector<Node> indices;

    size_t expand_stride = 1024;

    open.reserve(4096);
    // close.reserve(4096);
    merge.reserve(4096);

    expand.resize(expand_stride << 2);
    dedup.reserve(expand_stride << 2);
    // indices.reserve(expand_stride << 2);

    open.push_back(thrust::make_tuple(start, 0, expansion.heuristic(start), NORTH));
    // close = open;

    int iterations;
    for (iterations = 0;; ++iterations) {
        auto stride = std::min(expand_stride, open.size());
        auto expand_size = stride * 4;

        auto tic = std::chrono::high_resolution_clock::now();
        {
            thrust::for_each(
                    make_expand_iter(open, expand, stride),
                    make_expand_iter(open, expand, stride, expand_size),
                    thrust::make_zip_function(expansion)
            );

            // Sort the expanded list by node.
            thrust::sort_by_key(
                    expand.keys(),
                    expand.keys(expand_size),
                    expand.values(),
                    NodeComp()
            );
        }
        auto toc = std::chrono::high_resolution_clock::now();
        expand_duration += toc - tic;

        tic = std::chrono::high_resolution_clock::now();
        {
            /*
            // Search in close list
            indices.resize(dedup.size());
            thrust::lower_bound(
                    close.keys(),
                    close.keys(close.size()),
                    dedup.keys(),
                    dedup.keys(dedup.size()),
                    indices.begin()
            );

            // Exclude suboptimal states
            thrust::for_each(
                    make_selection_iter(indices, close, dedup),
                    make_selection_iter(indices, close, dedup, dedup.size()),
                    thrust::make_zip_function(StateSelect())
            );
            auto end = thrust::remove_if(dedup.iter(), dedup.iter(dedup.size()), StateRemove());
            dedup.resize(end - dedup.iter());
             */

            // Reduce, first pass.
            dedup.resize(expand_size);
            thrust::reduce_by_key(
                    expand.keys(),
                    expand.keys(expand_size),
                    expand.values(),
                    dedup.keys(),
                    dedup.values(),
                    thrust::equal_to<Node>(),
                    StateReduce()
            );
            auto expand_end = thrust::find(dedup.keys(), dedup.keys(expand_size), 0);
            dedup.resize(expand_end - dedup.keys());

            // Remove suboptimal states.
            auto end = thrust::remove_if(dedup.iter(), dedup.iter(dedup.size()), hash_state_remove);
            dedup.resize(end - dedup.iter());
        }
        toc = std::chrono::high_resolution_clock::now();
        dedup_duration += toc - tic;

        tic = std::chrono::high_resolution_clock::now();
        {
            /*
            // Close list is assumed to be sorted by node.
            merge.resize(close.size() + dedup.size());
            thrust::merge_by_key(
                    close.keys(),
                    close.keys(close.size()),
                    dedup.keys(),
                    dedup.keys(dedup.size()),
                    close.values(),
                    dedup.values(),
                    merge.keys(),
                    merge.values()
            );
            close.resize(merge.size());
            auto ends = thrust::reduce_by_key(
                    merge.keys(),
                    merge.keys(merge.size()),
                    merge.values(),
                    close.keys(),
                    close.values(),
                    thrust::equal_to<Node>(),
                    StateReduce()
            );
            close.resize(ends.first - close.keys());
             */

            // Update hash table
            thrust::for_each(
                    thrust::make_zip_iterator(dedup.nodes.begin(), dedup.scores.begin()),
                    thrust::make_zip_iterator(dedup.nodes.end(), dedup.scores.end()),
                    thrust::make_zip_function(hash_insert)
            );
        }
        toc = std::chrono::high_resolution_clock::now();
        close_duration += toc - tic;

        if (thrust::binary_search(dedup.keys(), dedup.keys(dedup.size()), target)) {
            break;
        }

        tic = std::chrono::high_resolution_clock::now();
        {
            // The open list is assumed to be sorted by score.
            // Sort the expanded list by score and merge with open list.
            thrust::sort_by_key(
                    dedup.keys_score(),
                    dedup.keys_score(dedup.size()),
                    dedup.values_score()
            );

            merge.resize(open.size() - stride + dedup.size());
            thrust::merge_by_key(
                    open.keys_score(stride),
                    open.keys_score(open.size()),
                    dedup.keys_score(),
                    dedup.keys_score(dedup.size()),
                    open.values_score(stride),
                    dedup.values_score(),
                    merge.keys_score(),
                    merge.values_score()
            );
            thrust::swap(open, merge);
        }
        toc = std::chrono::high_resolution_clock::now();
        open_duration += toc - tic;
    }

    HostStates host_open, host_dedup;
    host_open.copy_from(open);
    host_dedup.copy_from(dedup);

    auto iter = std::lower_bound(host_dedup.nodes.begin(), host_dedup.nodes.end(), target);
    auto pos = iter - host_dedup.nodes.begin();
    auto path = host_dedup.parents[pos];

    auto step = 0;
    auto node = start;
    while (true) {
        std::cout << (int) step << ' ' << '\n';
        print_node(node);

        if (node == target) break;

        Edge edge = (Edge) (path & 0b11);

        Node mask = 0xf;
        int x = -1, y = -1;
        for (int i = 0; i < 16; ++i, mask <<= 4) {
            if ((node & mask) == 0) {
                x = i / 4;
                y = i % 4;
                break;
            }
        }
        if (edge == NORTH && x > 0) {
            auto selected = node & (mask >> 16);
            node = (node | (selected << 16)) ^ selected;
        }
        if (edge == EAST && y < 3) {
            auto selected = node & (mask << 4);
            node = (node | (selected >> 4)) ^ selected;
        }
        if (edge == SOUTH && x < 3) {
            auto selected = node & (mask << 16);
            node = (node | (selected >> 16)) ^ selected;
        }
        if (edge == WEST && y > 0) {
            auto selected = node & (mask >> 4);
            node = (node | (selected << 4)) ^ selected;
        }

        ++step;
        path >>= 2;
    }

    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Expand duration: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(expand_duration).count()
              << '\n';
    std::cout << "Dedup duration: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(dedup_duration).count()
              << '\n';
    std::cout << "Close duration: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(close_duration).count()
              << '\n';
    std::cout << "Open duration: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(open_duration).count()
              << '\n';

    return 0;
}