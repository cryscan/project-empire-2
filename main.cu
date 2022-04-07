//
// Created by lepet on 4/6/2022.
//

#include <thrust/find.h>
#include <thrust/zip_function.h>

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
            out_score = out_step + heuristic(node);
            out_parent = SOUTH;
            return;
        }

        if (direction == EAST && y < 3) {
            auto selected = node & (mask << 4);
            out_node = (node | (selected >> 4)) ^ selected;
            out_step = step + 1;
            out_score = out_step + heuristic(node);
            out_parent = WEST;
            return;
        }

        if (direction == SOUTH && x < 3) {
            auto selected = node & (mask << 16);
            out_node = (node | (selected >> 16)) ^ selected;
            out_step = step + 1;
            out_score = out_step + heuristic(node);
            out_parent = NORTH;
            return;
        }

        if (direction == WEST && y > 0) {
            auto selected = node & (mask >> 4);
            out_node = (node | (selected << 4)) ^ selected;
            out_step = step + 1;
            out_score = out_step + heuristic(node);
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

struct NodeReduce {
    template<typename Tuple>
    __host__ __device__
    Tuple operator()(const Tuple& lhs, const Tuple& rhs) {
        return thrust::get<1>(lhs) < thrust::get<1>(rhs) ? lhs : rhs;
    }
};

int main() {
    Node start = 0xFEDCBA9876543210;
    Node target = 0xAECDF941B8527306;
    Expansion expansion(target);

    States open, merge, expand, dedup;
    size_t expand_stride = 1024;

    open.reserve(4096);
    merge.reserve(4096);
    expand.resize(expand_stride * 4);
    dedup.resize(expand_stride * 4);

    open.push_back(thrust::make_tuple(start, 0, expansion.heuristic(start), NORTH));

    for (int i = 0; i < 10; ++i) {
        // The open list is assumed to be sorted by score.
        auto stride = std::min(expand_stride, open.size());
        auto expand_size = stride * 4;

        auto expand_iter = States::make_expand_iter(open, expand, stride);
        thrust::for_each(
                States::make_expand_iter(open, expand, stride),
                States::make_expand_iter(open, expand, stride, expand_size),
                thrust::make_zip_function(expansion)
        );

        // Sort the expanded list by node
        thrust::sort_by_key(
                expand.keys(),
                expand.keys(expand_size),
                expand.values(),
                NodeComp()
        );
        // Reduce, first pass
        thrust::reduce_by_key(
                expand.keys(),
                expand.keys(expand_size),
                expand.values(),
                dedup.keys(),
                dedup.values(),
                thrust::equal_to<Node>(),
                NodeReduce()
        );

        auto expand_end = thrust::find(dedup.keys(), dedup.keys(expand_size), 0);
        expand_size = expand_end - dedup.keys();

        // TODO: close list

        // Sort the expanded list by score and merge with open list
        thrust::sort_by_key(
                dedup.keys_score(),
                dedup.keys_score(expand_size),
                dedup.values_score()
        );

        merge.resize(open.size() - stride + expand_size);
        thrust::merge_by_key(
                open.keys_score(stride),
                open.keys_score(open.size()),
                dedup.keys_score(),
                dedup.keys_score(expand_size),
                open.values_score(stride),
                dedup.values_score(),
                merge.keys_score(),
                merge.values_score()
        );
        thrust::swap(open, merge);
    }

    HostStates host;
    host.copy_from(open);

    return 0;
}