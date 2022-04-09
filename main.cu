//
// Created by lepet on 4/6/2022.
//

#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/zip_function.h>
#include <thrust/binary_search.h>

#include <chrono>

#include "states.h"
#include "path.h"

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
        if (node == 0) {
            out_node = 0;
            return;
        }

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

template<size_t INDEX>
struct StateReduce {
    template<typename Tuple>
    __host__ __device__
    Tuple operator()(const Tuple& lhs, const Tuple& rhs) {
        return thrust::get<INDEX>(lhs) < thrust::get<INDEX>(rhs) ? lhs : rhs;
    }
};

template<size_t INDEX>
struct StateRemove {
    template<typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& tuple) {
        return thrust::get<INDEX>(tuple) == 0;
    }
};

struct StateClear {
    __host__ __device__
    void operator()(Node& node, Value& score) {
        node = 0;
        score = ~0;
    }
};

auto make_extract_keys(size_t input_size, size_t stride, size_t x = 0) {
    using namespace thrust::placeholders;
    size_t len = std::ceil((static_cast<double>(input_size)) / stride);
    auto counter = thrust::make_counting_iterator(x);
    return thrust::make_transform_iterator(counter, _1 / len);
}

auto make_extract_values(States& states, size_t x = 0) {
    auto counter = thrust::make_counting_iterator(x);
    return thrust::make_zip_iterator(
            counter,
            states.nodes.begin() + x,
            states.steps.begin() + x,
            states.scores.begin() + x,
            states.parents.begin() + x
    );
}

auto make_extract_values_indices(States& states, thrust::device_vector<int>& indices, size_t x = 0) {
    return thrust::make_zip_iterator(
            indices.begin() + x,
            states.nodes.begin() + x,
            states.steps.begin() + x,
            states.scores.begin() + x,
            states.parents.begin() + x
    );
}

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
        if (node == 0) return true;

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
            extract_duration(0),
            expand_duration(0),
            dedup_duration(0),
            close_duration(0),
            open_duration(0);

    thrust::device_vector<uint64_t> hashtable(16777213);
    HashStateRemove hash_state_remove(thrust::raw_pointer_cast(hashtable.data()), hashtable.size());
    HashInsert hash_insert(thrust::raw_pointer_cast(hashtable.data()), hashtable.size());

    States open, extract, expand, dedup;
    thrust::device_vector<int> keys, indices;

    size_t expand_stride = 1024;

    open.reserve(4096);
    keys.reserve(expand_stride);
    indices.reserve(expand_stride);
    // close.reserve(4096);
    // merge.reserve(4096);

    extract.resize(expand_stride);
    expand.resize(expand_stride << 2);
    dedup.reserve(expand_stride << 2);
    // indices.reserve(expand_stride << 2);

    open.push_back(thrust::make_tuple(start, 0, expansion.heuristic(start), Path()));
    // close = open;

    int iterations;
    for (iterations = 0;; ++iterations) {
        auto stride = std::min(expand_stride, open.size());

        auto tic = std::chrono::high_resolution_clock::now();
        {
            // Extract.
            if (open.size() > expand_stride) {
                keys.resize(stride);
                indices.resize(stride);
                thrust::reduce_by_key(
                        make_extract_keys(open.size(), stride),
                        make_extract_keys(open.size(), stride, open.size()),
                        make_extract_values(open),
                        keys.begin(),
                        make_extract_values_indices(extract, indices),
                        thrust::equal_to<int>(),
                        StateReduce<3>()
                );

                // Remove extracted states from open list.
                auto iter_begin = thrust::make_permutation_iterator(
                        thrust::make_zip_iterator(open.keys(), open.keys_score()),
                        indices.begin()
                );
                auto iter_end = thrust::make_permutation_iterator(
                        thrust::make_zip_iterator(open.keys(), open.keys_score()),
                        indices.end()
                );
                thrust::for_each(iter_begin, iter_end, thrust::make_zip_function(StateClear()));

                /*
                using namespace thrust::placeholders;
                thrust::transform(
                        thrust::make_permutation_iterator(open.keys(), indices.begin()),
                        thrust::make_permutation_iterator(open.keys(), indices.end()),
                        thrust::make_permutation_iterator(open.keys(), indices.begin()),
                        _1 ^ _1
                );
                thrust::transform(
                        thrust::make_permutation_iterator(open.keys_score(), indices.begin()),
                        thrust::make_permutation_iterator(open.keys_score(), indices.end()),
                        thrust::make_permutation_iterator(open.keys_score(), indices.begin()),
                        ~(_1 ^ _1)
                );
                 */

                if (iterations % 200 == 0) {
                    auto end = thrust::remove_if(open.iter(), open.iter(open.size()), StateRemove<0>());
                    open.resize(end - open.iter());
                }

                // auto end = thrust::remove_if(open.iter(), open.iter(open.size()), StateRemove<0>());
                // auto end = thrust::partition(open.iter(), open.iter(open.size()), StatePartition<0>());
                // open.resize(end - open.iter());
            } else {
                extract = open;
                open.resize(0);
            }
        }
        auto toc = std::chrono::high_resolution_clock::now();
        extract_duration += toc - tic;

        tic = std::chrono::high_resolution_clock::now();
        {
            // Expand.
            expand.resize(stride << 2);
            thrust::for_each(
                    make_expand_iter(extract, expand, stride),
                    make_expand_iter(extract, expand, stride, expand.size()),
                    thrust::make_zip_function(expansion)
            );
        }
        toc = std::chrono::high_resolution_clock::now();
        expand_duration += toc - tic;

        tic = std::chrono::high_resolution_clock::now();
        {
            // Remove suboptimal and empty states.
            auto end = thrust::remove_if(
                    expand.iter(),
                    expand.iter(expand.size()),
                    hash_state_remove
            );
            expand.resize(end - expand.iter());

            // Sort the expanded list by node.
            thrust::sort_by_key(
                    expand.keys(),
                    expand.keys(expand.size()),
                    expand.values()
            );

            // Deduplication.
            dedup.resize(expand.size());
            auto dedup_end = thrust::reduce_by_key(
                    expand.keys(),
                    expand.keys(expand.size()),
                    expand.values(),
                    dedup.keys(),
                    dedup.values(),
                    thrust::equal_to<Node>(),
                    StateReduce<1>()
            );
            dedup.resize(dedup_end.first - dedup.keys());
        }
        toc = std::chrono::high_resolution_clock::now();
        dedup_duration += toc - tic;

        tic = std::chrono::high_resolution_clock::now();
        {
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
            auto size = open.size();
            open.resize(size + dedup.size());
            thrust::copy(dedup.iter(), dedup.iter(dedup.size()), open.iter(size));
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
    while (step <= host_dedup.steps[pos]) {
        std::cout << (int) step << ' ' << '\n';
        print_node(node);

        if (node == target) break;

        Edge edge = static_cast<Edge>((path >> (2 * step)) & 0b11);

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
    }

    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Extract duration: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(extract_duration).count()
              << '\n';
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