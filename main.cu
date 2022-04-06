#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/zip_function.h>

using Node = uint64_t;
using Value = uint8_t;

constexpr Node empty_node = 0;
enum Direction {
    UP = 0,
    RIGHT = 1,
    DOWN = 2,
    LEFT = 3,
};

struct States {
    Node* nodes;
    Value* steps;
    Value* values;
    size_t size;
};

struct heuristic_functor {
    Node target;

    explicit heuristic_functor(Node target) : target(target) {}

    __host__ __device__
    Value operator()(Node node) const {
        Node mask = 0xf;
        Value result = 0;
        for (int i = 0; i < 16; ++i, mask <<= 4) {
            auto x = (mask & node) >> (4 * i);
            auto y = (mask & target) >> (4 * i);
            if (x != y && x != 0) ++result;
        }
        return result;
    }
};

template<class T>
auto make_expand_input(const thrust::device_vector<T>& vec, size_t stride, size_t x) {
    using namespace thrust::placeholders;
    auto counter = thrust::make_counting_iterator(x);
    auto iter = thrust::make_permutation_iterator(
            vec.begin(),
            thrust::make_transform_iterator(counter, _1 % stride)
    );
    return iter;
}

auto make_expand_direction_input(size_t stride, size_t x) {
    using namespace thrust::placeholders;
    auto counter = thrust::make_counting_iterator(x);
    auto direction_iter = thrust::make_transform_iterator(counter, _1 / stride);
    return direction_iter;
}

struct expand_functor {
    __host__ __device__
    void operator()(const Node& node, const Value& step, const size_t& direction,
                    Node& expanded_node, Value& expanded_step) {
        Node mask = 0xf;
        int x = -1, y = -1;
        for (int i = 0; i < 16; ++i, mask <<= 4) {
            if ((node & mask) == 0) {
                x = i / 4;
                y = i % 4;
                break;
            }
        }

        if (direction == UP && x > 0) {
            auto selected = node & (mask >> 16);
            expanded_node = (node | (selected << 16)) ^ selected;
            expanded_step = step + 1;
            return;
        }

        if (direction == RIGHT && y < 3) {
            auto selected = node & (mask << 4);
            expanded_node = (node | (selected >> 4)) ^ selected;
            expanded_step = step + 1;
            return;
        }

        if (direction == DOWN && x < 3) {
            auto selected = node & (mask << 16);
            expanded_node = (node | (selected >> 16)) ^ selected;
            expanded_step = step + 1;
            return;
        }

        if (direction == LEFT && y > 0) {
            auto selected = node & (mask >> 4);
            expanded_node = (node | (selected << 4)) ^ selected;
            expanded_step = step + 1;
            return;
        }

        expanded_node = empty_node;
    }
};

struct node_comp {
    __host__ __device__
    bool operator()(const Node& lhs, const Node& rhs) {
        if (lhs == empty_node) return false;
        if (rhs == empty_node) return true;
        return lhs < rhs;
    }
};

int main() {
    Node target = 0xAECDF941B8527306;

    thrust::device_vector<Node> nodes;
    thrust::device_vector<Value> steps;
    thrust::device_vector<Value> values;

    nodes.reserve(1024);
    steps.reserve(1024);
    values.reserve(1024);

    size_t expand_stride = 1024;
    thrust::device_vector<Node> expanded_nodes(4 * expand_stride, empty_node);
    thrust::device_vector<Value> expanded_steps(4 * expand_stride);
    thrust::device_vector<Value> expanded_values(4 * expand_stride);

    nodes.push_back(0xFEDCBA9876543210);
    steps.push_back(0);

    {
        auto stride = std::min(expand_stride, nodes.size());
        auto len = 4 * stride;

        auto nodes_begin = make_expand_input(nodes, stride, 0);
        auto nodes_end = make_expand_input(nodes, stride, len);

        auto steps_begin = make_expand_input(steps, stride, 0);
        auto steps_end = make_expand_input(steps, stride, len);

        auto direction_begin = make_expand_direction_input(stride, 0);
        auto direction_end = make_expand_direction_input(stride, len);

        auto expanded_nodes_begin = expanded_nodes.begin();
        auto expanded_nodes_end = expanded_nodes_begin + len;

        auto expanded_steps_begin = expanded_steps.begin();
        auto expanded_steps_end = expanded_steps_begin + len;

        thrust::for_each(
                thrust::make_zip_iterator(
                        nodes_begin,
                        steps_begin,
                        direction_begin,
                        expanded_nodes_begin,
                        expanded_steps_begin
                ),
                thrust::make_zip_iterator(
                        nodes_end,
                        steps_end,
                        direction_end,
                        expanded_nodes_end,
                        expanded_steps_end
                ),
                thrust::make_zip_function(expand_functor())
        );

        // Sort and remove empty nodes
        thrust::sort_by_key(
                expanded_nodes.begin(),
                expanded_nodes.begin() + len,
                expanded_steps.begin(),
                node_comp()
        );
    }

    std::vector<Node> host_expanded_nodes(4 * expand_stride);
    std::vector<Value> host_expanded_steps(4 * expand_stride);

    thrust::copy(expanded_nodes.begin(), expanded_nodes.end(), host_expanded_nodes.begin());
    thrust::copy(expanded_steps.begin(), expanded_steps.end(), host_expanded_steps.begin());

    return 0;
}
