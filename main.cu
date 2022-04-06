#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/find.h>
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
    Node target;

    explicit expand_functor(Node target) : target(target) {}

    // Heuristic
    [[nodiscard]] __host__ __device__
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
            Node& expanded_node,
            Value& expanded_step,
            Value& expanded_score
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

        if (direction == UP && x > 0) {
            auto selected = node & (mask >> 16);
            expanded_node = (node | (selected << 16)) ^ selected;
            expanded_step = step + 1;
            expanded_score = expanded_step + heuristic(node);
            return;
        }

        if (direction == RIGHT && y < 3) {
            auto selected = node & (mask << 4);
            expanded_node = (node | (selected >> 4)) ^ selected;
            expanded_step = step + 1;
            expanded_score = expanded_step + heuristic(node);
            return;
        }

        if (direction == DOWN && x < 3) {
            auto selected = node & (mask << 16);
            expanded_node = (node | (selected >> 16)) ^ selected;
            expanded_step = step + 1;
            expanded_score = expanded_step + heuristic(node);
            return;
        }

        if (direction == LEFT && y > 0) {
            auto selected = node & (mask >> 4);
            expanded_node = (node | (selected << 4)) ^ selected;
            expanded_step = step + 1;
            expanded_score = expanded_step + heuristic(node);
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
    Node start = 0xFEDCBA9876543210;
    Node target = 0xAECDF941B8527306;

    thrust::device_vector<Node> nodes;
    thrust::device_vector<Value> steps;
    thrust::device_vector<Value> scores;

    thrust::device_vector<Node> merged_nodes;
    thrust::device_vector<Value> merged_steps;
    thrust::device_vector<Value> merged_scores;

    nodes.reserve(4096);
    steps.reserve(4096);
    scores.reserve(4096);

    merged_steps.reserve(4096);
    merged_steps.reserve(4096);

    size_t expand_stride = 1024;
    thrust::device_vector<Node> expanded_nodes(4 * expand_stride, empty_node);
    thrust::device_vector<Value> expanded_steps(4 * expand_stride);
    thrust::device_vector<Value> expanded_scores(4 * expand_stride);
    auto expand_func = expand_functor(target);

    nodes.push_back(start);
    steps.push_back(0);
    scores.push_back(expand_func.heuristic(start));

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

        auto expanded_scores_begin = expanded_scores.begin();
        auto expanded_scores_end = expanded_scores_begin + len;

        thrust::for_each(
                thrust::make_zip_iterator(
                        nodes_begin,
                        steps_begin,
                        direction_begin,
                        expanded_nodes_begin,
                        expanded_steps_begin,
                        expanded_scores_begin
                ),
                thrust::make_zip_iterator(
                        nodes_end,
                        steps_end,
                        direction_end,
                        expanded_nodes_end,
                        expanded_steps_end,
                        expanded_scores_end
                ),
                thrust::make_zip_function(expand_func)
        );

        // Sort and remove empty expanded nodes
        auto expanded_values_begin = thrust::make_zip_iterator(expanded_steps_begin, expanded_scores_begin);
        thrust::sort_by_key(
                expanded_nodes_begin,
                expanded_nodes_end,
                expanded_values_begin,
                node_comp()
        );

        expanded_nodes_end = thrust::find(
                expanded_nodes_begin,
                expanded_nodes_end,
                empty_node
        );
        auto expanded_len = expanded_nodes_end - expanded_nodes_begin;

        // TODO: remove duplications

        // Merge expanded states with remaining open list states
        merged_nodes.resize(nodes.size() - stride + expanded_len);
        merged_steps.resize(nodes.size() - stride + expanded_len);
        merged_scores.resize(nodes.size() - stride + expanded_len);
        auto original_values_begin = thrust::make_zip_iterator(
                steps.begin() + stride,
                scores.begin() + stride
        );
        auto merged_values_begin = thrust::make_zip_iterator(merged_steps.begin(), merged_scores.begin());
        thrust::merge_by_key(
                nodes.begin() + stride,
                nodes.end(),
                expanded_nodes_begin,
                expanded_nodes_end,
                original_values_begin,
                expanded_values_begin,
                merged_nodes.begin(),
                merged_values_begin
        );

        thrust::swap(nodes, merged_nodes);
        thrust::swap(steps, merged_steps);
        thrust::swap(scores, merged_scores);
    }

    std::vector<Node> host_nodes(nodes.size());
    std::vector<Value> host_steps(steps.size());
    std::vector<Value> host_scores(scores.size());

    thrust::copy(nodes.begin(), nodes.end(), host_nodes.begin());
    thrust::copy(steps.begin(), steps.end(), host_steps.begin());
    thrust::copy(scores.begin(), scores.end(), host_scores.begin());

    return 0;
}
