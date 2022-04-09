//
// Created by yiqun on 4/8/2022.
//

#ifndef PROJECT_EMPIRE_2_PATH_H
#define PROJECT_EMPIRE_2_PATH_H

namespace empire {
    template<typename Edge, typename Value, size_t PATH_LEN = 64, size_t EDGE_LEN = 2>
    struct Path {
        uint64_t data[PATH_LEN / 64];

        static constexpr uint64_t mask = (1 << EDGE_LEN) - 1;

        __host__ __device__
        Path() : data{0} {}

        __host__ __device__
        Path(const Path& other) : data{0} {
            for (auto i = 0; i < PATH_LEN / 64; ++i)
                data[i] = other.data[i];
        }

        __host__ __device__
        Path& operator=(const Path& other) {
            for (auto i = 0; i < PATH_LEN / 64; ++i)
                data[i] = other.data[i];
            return *this;
        }

        __host__ __device__
        Path& add(Value step, Edge direction) {
            auto loc = step * EDGE_LEN;
            data[loc / 64] |= direction << (loc % 64);
            return *this;
        }

        __host__ __device__
        Path add_copy(Value step, Edge direction) const {
            Path copy = *this;
            return copy.add(step, direction);
        }

        __host__ __device__
        Edge find(Value step) const {
            auto loc = step * EDGE_LEN;
            auto datum = data[loc / 64] >> (loc % 64);
            return static_cast<Edge>(datum & mask);
        }
    };
}

#endif //PROJECT_EMPIRE_2_PATH_H
