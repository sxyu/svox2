// Copyright 2021 Alex Yu
#pragma once
#include <cstdint>
#include <cmath>

// A custom xorshift random generator
// Maybe replace with some CUDA internal stuff?
struct RandomEngine32 {
    uint32_t x, y, z;

    // Inclusive both
    __host__ __device__
    uint32_t randint(uint32_t lo, uint32_t hi) {
        if (hi <= lo) return lo;
        uint32_t z = (*this)();
        return z % (hi - lo + 1) + lo;
    }

    __host__ __device__
    void rand2(float* out1, float* out2) {
        const uint32_t z = (*this)();
        const uint32_t fmax = (1 << 16);
        const uint32_t z1 = z >> 16;
        const uint32_t z2 = z & (fmax - 1);
        const float ifmax = 1.f / fmax;

        *out1 = z1 * ifmax;
        *out2 = z2 * ifmax;
    }

    __host__ __device__
    float rand() {
        uint32_t z = (*this)();
        return float(z) / (1LL << 32);
    }


    __host__ __device__
    void randn2(float* out1, float* out2) {
        rand2(out1, out2);
        // Box-Muller transform
        const float srlog = sqrtf(-2 * logf(*out1 + 1e-32f));
        *out2 *= 2 * M_PI;
        *out1 = srlog * cosf(*out2);
        *out2 = srlog * sinf(*out2);
    }

    __host__ __device__
    float randn() {
        float x, y;
        rand2(&x, &y);
        // Box-Muller transform
        return sqrtf(-2 * logf(x + 1e-32f))* cosf(2 * M_PI * y);
    }

    __host__ __device__
    uint32_t operator()() {
        uint32_t t;
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;
        t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;
        return z;
    }
};
