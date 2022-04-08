//
// Created by lepet on 4/6/2022.
//

#ifndef PROJECT_EMPIRE_2_STATES_H
#define PROJECT_EMPIRE_2_STATES_H

#include <thrust/device_vector.h>

namespace empire {
    template<typename Node, typename Path, typename Value>
    struct HostStates;

    template<typename Node, typename Path, typename Value>
    struct States {
        thrust::device_vector<Node> nodes;
        thrust::device_vector<Value> steps;
        thrust::device_vector<Value> scores;
        thrust::device_vector<Path> parents;

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
