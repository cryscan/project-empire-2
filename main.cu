//
// Created by lepet on 4/6/2022.
//

#include <thrust/find.h>
#include <thrust/zip_function.h>
#include <thrust/binary_search.h>

#include "states.h"

using Node = uint64_t;
using Value = uint8_t;
enum Edge : uint8_t {
    NORTH = 0,
    EAST = 1,
    SOUTH = 2,
    WEST = 3,
};

using States = empire::States<Node, Edge, Value>;
using HostStates = empire::HostStates<Node, Edge, Value>;

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
            const size_t& direction,
            Node& out_node,
            Value& out_step,
            Value& out_score,
            Edge& out_parent
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
            out_parent = SOUTH;
            return;
        }

        if (direction == EAST && y < 3) {
            auto selected = node & (mask << 4);
            out_node = (node | (selected >> 4)) ^ selected;
            out_step = step + 1;
            out_score = step + heuristic(node);
            out_parent = WEST;
            return;
        }

        if (direction == SOUTH && x < 3) {
            auto selected = node & (mask << 16);
            out_node = (node | (selected >> 16)) ^ selected;
            out_step = step + 1;
            out_score = step + heuristic(node);
            out_parent = NORTH;
            return;
        }

        if (direction == WEST && y > 0) {
            auto selected = node & (mask >> 4);
            out_node = (node | (selected << 4)) ^ selected;
            out_step = step + 1;
            out_score = step + heuristic(node);
            out_parent = EAST;
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

struct StatePartition {
    template<typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& tuple) {
        return thrust::get<0>(tuple) != 0;
    }
};

auto make_expand_iter(const States& input, States& output, size_t stride, size_t x = 0) {
    using namespace thrust::placeholders;
    auto expand_counter = thrust::make_counting_iterator(x);
    auto stride_counter = thrust::make_transform_iterator(expand_counter, _1 % stride);
    auto direction_iter = thrust::make_transform_iterator(expand_counter, _1 / stride);

    auto input_nodes_iter = thrust::make_permutation_iterator(input.nodes.begin(), stride_counter);
    auto input_steps_iter = thrust::make_permutation_iterator(input.steps.begin(), stride_counter);

    return thrust::make_zip_iterator(
            input_nodes_iter,
            input_steps_iter,
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

    States open, close, merge, expand, dedup;
    thrust::device_vector<Node> indices;
    size_t expand_stride = 1024;

    open.reserve(4096);
    close.reserve(4096);
    merge.reserve(4096);

    expand.resize(expand_stride << 2);
    dedup.reserve(expand_stride << 2);
    indices.reserve(expand_stride << 2);

    open.push_back(thrust::make_tuple(start, 0, expansion.heuristic(start), NORTH));
    close = open;

    int iterations = 0;
    for (int i = 0;; ++i) {
        auto stride = std::min(expand_stride, open.size());

        {
            auto expand_size = stride * 4;

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
        }

        {
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
            auto end = thrust::stable_partition(dedup.iter(), dedup.iter(dedup.size()), StatePartition());
            dedup.resize(end - dedup.iter());
        }

        {
            // Update close list.
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
        }

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

        if (thrust::binary_search(close.keys(), close.keys(open.size()), target)) {
            iterations = i;
            break;
        }
    }

    HostStates host_open, host_close;
    host_open.copy_from(open);
    host_close.copy_from(close);

    Node node = target;
    while (true) {
        auto iter = std::lower_bound(host_close.nodes.begin(), host_close.nodes.end(), node);
        auto pos = iter - host_close.nodes.begin();
        auto parent = host_close.parents[pos];
        auto step = host_close.steps[pos];
        auto score = host_close.scores[pos];

        std::cout << (int) step << ' ' << (int) score << '\n';
        print_node(node);

        if (node == start) break;

        Node mask = 0xf;
        int x = -1, y = -1;
        for (int i = 0; i < 16; ++i, mask <<= 4) {
            if ((node & mask) == 0) {
                x = i / 4;
                y = i % 4;
                break;
            }
        }
        if (parent == NORTH && x > 0) {
            auto selected = node & (mask >> 16);
            node = (node | (selected << 16)) ^ selected;
        }
        if (parent == EAST && y < 3) {
            auto selected = node & (mask << 4);
            node = (node | (selected >> 4)) ^ selected;
        }
        if (parent == SOUTH && x < 3) {
            auto selected = node & (mask << 16);
            node = (node | (selected >> 16)) ^ selected;
        }
        if (parent == WEST && y > 0) {
            auto selected = node & (mask >> 4);
            node = (node | (selected << 4)) ^ selected;
        }
    }

    std::cout << "Iterations: " << iterations << std::endl;

    return 0;
}