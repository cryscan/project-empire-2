//
// Created by lepet on 4/6/2022.
//

#ifndef PROJECT_EMPIRE_2_STATES_H
#define PROJECT_EMPIRE_2_STATES_H

#include <thrust/device_vector.h>

namespace empire {
    template<typename Node, typename Edge, typename Value>
    struct HostStates;

    template<typename Node, typename Edge, typename Value>
    class States {
        thrust::device_vector<Node> nodes;
        thrust::device_vector<Value> steps;
        thrust::device_vector<Value> scores;
        thrust::device_vector<Edge> parents;

        friend class HostStates<Node, Edge, Value>;

        friend void thrust::swap(States&, States&);

    public:
        [[nodiscard]] size_t size() const { return nodes.size(); }

        auto iter(size_t x = 0) const {
            return thrust::make_zip_iterator(
                    nodes.begin() + x,
                    steps.begin() + x,
                    scores.begin() + x,
                    parents.begin() + x
            );
        }

        auto iter(size_t x = 0) {
            return thrust::make_zip_iterator(
                    nodes.begin() + x,
                    steps.begin() + x,
                    scores.begin() + x,
                    parents.begin() + x
            );
        }

        auto keys(size_t x = 0) const { return nodes.begin() + x; }

        auto values(size_t x = 0) const {
            return thrust::make_zip_iterator(
                    steps.begin() + x,
                    scores.begin() + x,
                    parents.begin() + x
            );
        }

        auto keys(size_t x = 0) { return nodes.begin() + x; }

        auto values(size_t x = 0) {
            return thrust::make_zip_iterator(
                    steps.begin() + x,
                    scores.begin() + x,
                    parents.begin() + x
            );
        }

        auto keys_score(size_t x = 0) const { return scores.begin() + x; }

        auto values_score(size_t x = 0) const {
            return thrust::make_zip_iterator(
                    nodes.begin() + x,
                    steps.begin() + x,
                    parents.begin() + x
            );
        }

        auto keys_score(size_t x = 0) { return scores.begin() + x; }

        auto values_score(size_t x = 0) {
            return thrust::make_zip_iterator(
                    nodes.begin() + x,
                    steps.begin() + x,
                    parents.begin() + x
            );
        }

        static auto make_expand_iter(const States& input, States& output, size_t stride, size_t x = 0) {
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

        static auto make_select_iter(
                const States& control,
                const States& test,
                States& output,
                size_t x = 0
        ) {
            return thrust::make_zip_iterator(
                    control.scores.begin() + x,
                    test.nodes.begin() + x,
                    test.steps.begin() + x,
                    test.scores.begin() + x,
                    test.parents.begin() + x,
                    output.nodes.begin() + x,
                    output.steps.begin() + x,
                    output.scores.begin() + x,
                    output.parents.begin() + x
            );
        }

        void reserve(size_t capacity) {
            nodes.reserve(capacity);
            steps.reserve(capacity);
            scores.reserve(capacity);
            parents.reserve(capacity);
        }

        void resize(size_t size) {
            nodes.resize(size);
            steps.resize(size);
            scores.resize(size);
            parents.resize(size);
        }

        template<typename Tuple>
        void push_back(Tuple&& tuple) {
            nodes.push_back(thrust::get<0>(tuple));
            steps.push_back(thrust::get<1>(tuple));
            scores.push_back(thrust::get<2>(tuple));
            parents.push_back(thrust::get<3>(tuple));
        }
    };

    template<typename Node, typename Edge, typename Value>
    struct HostStates {
        std::vector<Node> nodes;
        std::vector<Value> steps;
        std::vector<Value> scores;
        std::vector<Edge> parents;

        void resize(size_t size) {
            nodes.resize(size);
            steps.resize(size);
            scores.resize(size);
            parents.resize(size);
        }

        void copy_from(States<Node, Edge, Value>& states) {
            resize(states.size());
            thrust::copy(states.nodes.begin(), states.nodes.end(), nodes.begin());
            thrust::copy(states.steps.begin(), states.steps.end(), steps.begin());
            thrust::copy(states.scores.begin(), states.scores.end(), scores.begin());
            thrust::copy(states.parents.begin(), states.parents.end(), parents.begin());
        }
    };
}

namespace thrust {
    template<typename Node, typename Edge, typename Value>
    void swap(empire::States<Node, Edge, Value>& lhs, empire::States<Node, Edge, Value>& rhs) {
        swap(lhs.nodes, rhs.nodes);
        swap(lhs.steps, rhs.steps);
        swap(lhs.scores, rhs.scores);
        swap(lhs.parents, rhs.parents);
    }
}

#endif //PROJECT_EMPIRE_2_STATES_H
