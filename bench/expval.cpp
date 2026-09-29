// Validates the new floor-split fast_exp against libm over the full range
// of exponents actually produced by the engine (OU decay, phi tail, erf, MT jitter).
#include "../motor_synergy.h"
#include <cstdio>
#include <cmath>
#include <random>
int main(){
    using namespace motor_synergy::detail;
    double max_rel = 0, worst_x = 0;
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(-50.0, 1.0); // engine's real domain
    for (int i = 0; i < 20000000; ++i) {
        double x = u(rng);
        double a = fast_exp(x), r = std::exp(x);
        double rel = std::abs(a - r) / r;
        if (rel > max_rel) { max_rel = rel; worst_x = x; }
    }
    // dense sweep on grid too
    for (double x = -60; x <= 5; x += 1e-4) {
        double a = fast_exp(x), r = std::exp(x);
        double rel = std::abs(a - r) / r;
        if (rel > max_rel) { max_rel = rel; worst_x = x; }
    }
    printf("fast_exp max rel err %.3e at x=%.6f\n", max_rel, worst_x);
    return max_rel < 1e-7 ? 0 : 1;
}
