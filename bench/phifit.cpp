// Validates phi_poly (re-fit AS-26.2.17 coefficients + saturation early-out)
// and the new floor-split fast_exp against libm. Exit 0 = both within budget.
#include "../motor_synergy.h"
#include <cstdio>
#include <cmath>
int main(){
    using namespace motor_synergy::detail;
    double max_err = 0, worst_x = 0;
    for (double z = -9.0; z <= 9.0; z += 1e-5) {
        double exact = 0.5 * std::erfc(-z / std::sqrt(2.0));
        double err = std::abs(phi_poly(z) - exact);
        if (err > max_err) { max_err = err; worst_x = z; }
    }
    printf("phi_poly max abs CDF err %.3e at z=%.4f\n", max_err, worst_x);

    double e_max = 0;
    for (double x = -60; x <= 5; x += 1e-4) {
        double r = std::exp(x);
        e_max = std::max(e_max, std::abs(fast_exp(x) - r) / r);
    }
    printf("fast_exp max rel err %.3e\n", e_max);
    return (max_err < 1e-7 && e_max < 1e-8) ? 0 : 1;
}
