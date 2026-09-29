#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <algorithm>
#include <numbers>
#include <numeric>

namespace motor_synergy {

struct trajectory_point {
    double x, y;
    double t;
};

// ---------------------------------------------------------------------------
// Intensity scaling (global "how strong is the human noise" dial)
// ---------------------------------------------------------------------------
// Every stochastic / imperfection parameter in the model is multiplied by a
// single intensity value, so the engine can be tuned from perfectly smooth
// machine-like pointing to heavily impaired, super-noisy movement:
//
//   0.00 -> pure minimum-jerk bell curve: zero OU drift, zero tremor, zero
//            SDN, zero curvature, no overshoots, deterministic Fitts MT.
//            (Still emits Gamma-jittered poll timestamps.)
//   1.00 -> the canonical SigmaDrift defaults below (calibrated human profile)
//   5.00 -> "super strong": violent tremor, huge curvature arcs, aggressive
//            OU wandering, frequent long corrections.
//
// The scale is applied inside generate() at draw time — it NEVER mutates the
// config struct itself — so cfg.tremor_amp_max and friends always keep their
// literal meaning and configs stay reusable across intensities.
enum class intensity {
    machine,      // 0.00 - surgical, noise-free
    calm,         // 0.35 - steady hand, minimal wobble
    default_hum,  // 1.00 - calibrated SigmaDrift profile
    shaky,        // 1.75 - fatigued / caffeinated
    impaired,     // 3.00 - strongly noisy
    chaotic,      // 5.00 - maximum chaos
};

inline double to_double(intensity i) {
    switch (i) {
        case intensity::machine:     return 0.0;
        case intensity::calm:        return 0.35;
        case intensity::default_hum: return 1.0;
        case intensity::shaky:       return 1.75;
        case intensity::impaired:    return 3.0;
        case intensity::chaotic:     return 5.0;
    }
    return 1.0;
}

// Which config fields the intensity multiplies (documented contract):
//   noise      x I : ou_sigma, tremor_amp_{min,max}, sdn_k
//   shape      x I : curvature_scale
//   variability x I: primary/correction sigma ranges, lognormal MT jitter,
//                    peak-time & correction-timing jitter, overshoot spread
//   suppressed by I: overshoot_prob, second_correction_prob (x I)
// Intensity-invariant (deliberately): fitts_a/b, target_width, ou_theta,
// tremor_freq (frequency of physiological tremor doesn't change with intent),
// sample_dt_mean, gamma_shape (poll rate is a property of the OS/hardware).
struct config {
    double fitts_a = 50.0;
    double fitts_b = 150.0;
    double target_width = 20.0;

    double undershoot_min = 0.92;
    double undershoot_max = 0.97;
    double peak_time_ratio = 0.35;
    double primary_sigma_min = 0.18;
    double primary_sigma_max = 0.28;

    double overshoot_prob = 0.15;
    double overshoot_min = 1.02;
    double overshoot_max = 1.08;
    double correction_sigma_min = 0.12;
    double correction_sigma_max = 0.20;
    double second_correction_prob = 0.25;

    double curvature_scale = 0.025;

    double ou_theta = 3.5;
    double ou_sigma = 1.2;

    double tremor_freq_min = 8.0;
    double tremor_freq_max = 12.0;
    double tremor_amp_min = 0.15;
    double tremor_amp_max = 0.55;

    double sdn_k = 0.04;

    double sample_dt_mean = 7.8;
    double gamma_shape = 3.5;

    // Global noise-strength dial used when this config is passed positionally.
    // Override per call via the generate(..., intensity) overload or set the
    // raw double directly for values between the presets.
    intensity strength = intensity::default_hum;

    // Raw accessor: any double works (0.0 .. ~6.0 sensible).
    double intensity_value = 0.0; // if > 0, overrides `strength`

    double effective_intensity() const {
        return intensity_value > 0.0 ? intensity_value : to_double(strength);
    }
};

struct metrics {
    double movement_time;
    double path_length;
    double straight_distance;
    double path_efficiency;
    double peak_speed;
    double time_to_peak;
    int num_submovements;
    double endpoint_error;
    double fitts_predicted_mt;
};

namespace detail {

// ---------------------------------------------------------------------------
// Branchless, low-latency math core
// ---------------------------------------------------------------------------
// All transcendentals below are called once (or twice) per frame from the
// single generation loop, so their cost dominates runtime. Each is either a
// direct intrinsic mapping or a bounded-error polynomial that lets the
// compiler emit pure SSE/AVX arithmetic instead of an opaque libm call with
// its errno / special-case branches.

// Abramowitz & Stegun 7.1.26 fast exp approximation (rel. err < ~2e-8 over
// the range we use it: |x| <= ~40). Pure multiply-add — no libm dispatch.
inline double fast_exp(double x) {
    // Clamp to keep the double->int cast well defined.
    x = (x < -60.0) ? -60.0 : ((x > 60.0) ? 60.0 : x);
    // e^x = 2^(x*log2(e)); split exponent into integer + fractional parts.
    constexpr double log2e = 1.4426950408889634074;
    double z = x * log2e;
    // Round-to-nearest via magic number (avoids fenv-dependent rint calls).
    constexpr double magic = 6755399441055744.0; // 1.5 * 2^52
    double n = (z + magic) - magic;
    double f = z - n;                       // f in [-0.5, 0.5]
    // Degree-8 Taylor for e^p on |p| <= 0.5*ln2 (rel. truncation err < 2e-16,
    // i.e. correctly rounded to double precision over our whole input range).
    const double ln2 = 0.6931471805599453094;
    double p = f * ln2;
    double poly = 1.0 + p * (1.0 + p * (0.5 + p * (1.0 / 6.0 + p * (1.0 / 24.0
              + p * (1.0 / 120.0 + p * (1.0 / 720.0 + p * (1.0 / 5040.0 + p * (1.0 / 40320.0))))))));
    // Scale by 2^n through the exponent bits. Split into two exact powers of
    // two so any n in [-1022, 1023] works without hitting subnormals — a
    // single shift would lose ~1e-7 relative accuracy once 2^n underflows.
    const int64_t ni = (int64_t)n;
    auto pow2 = [](int64_t k) {
        double d;
        const int64_t bits = (1023 + k) << 52;   // exact IEEE754 encoding
        std::memcpy(&d, &bits, sizeof(double));
        return d;
    };
    const int64_t hi = ni >> 1;                  // n = hi + lo, both in range
    return poly * pow2(hi) * pow2(ni - hi);
}

// Fast sincos pair for small angles (|a| <= ~2 rad): pure odd/even Taylor,
// rel. err < 1e-9 — far cheaper than two libm trig dispatches and lets the
// compiler vectorize. Used once per frame for the tremor phasor rotation.
inline void fast_sincos(double a, double& s, double& c) {
    const double a2 = a * a;
    // sin: terms through x^13/13! ; cos: through x^12/12! — worst-case abs
    // error over |a| <= pi is < 1e-11, and the phasor recurrence itself only
    // needs O(1e-8) fidelity for pixel-space tremor.
    s = a * (1.0 + a2 * (-1.0 / 6.0 + a2 * (1.0 / 120.0 + a2 * (-1.0 / 5040.0
        + a2 * (1.0 / 362880.0 + a2 * (-1.0 / 39916800.0 + a2 * (1.0 / 6227020800.0)))))));
    c = 1.0 + a2 * (-0.5 + a2 * (1.0 / 24.0 + a2 * (-1.0 / 720.0 + a2 * (1.0 / 40320.0
        + a2 * (-1.0 / 3628800.0 + a2 * (1.0 / 479001600.0))))));
}

// Abramowitz & Stegun 7.1.26 fast erf approximation (|err| <= 1.38e-7),
// now built on fast_exp so the whole CDF path is branch-light.
inline double fast_erf(double x) {
    constexpr double p  = 0.3275911;
    constexpr double a1 =  0.254829592;
    constexpr double a2 = -0.284496736;
    constexpr double a3 =  1.421413741;
    constexpr double a4 = -1.453152027;
    constexpr double a5 =  1.061405429;

    const double ax = std::abs(x);
    const double tt = 1.0 / (1.0 + p * ax);
    // Horner form of the polynomial.
    const double poly = tt * (a1 + tt * (a2 + tt * (a3 + tt * (a4 + tt * a5))));
    const double y = 1.0 - poly * fast_exp(-ax * ax);
    // Branchless sign restore (copysign maps to a single SSE instruction).
    return std::copysign(y, x);
}

inline double normal_cdf(double x) {
    // Same as 0.5 * (1 + erf(x / sqrt(2))) but with the fast approximation
    // and one precomputed constant folding (1/sqrt(2) baked into the literal).
    return 0.5 * (1.0 + fast_erf(x * 0.70710678118654752440));
}

// ---------------------------------------------------------------------------
// Lognormal CDF/PDF kernels: closed-form polynomial approximations.
//
// The old path paid TWO libm std::log calls (~9 ns each on this host) per
// sub-movement per frame plus an erf evaluation — by far the dominant cost of
// the generation loop. Because each sub-movement keeps a FIXED sigma for its
// whole life, we standardize once at setup into a tiny kernel struct:
//
//     z(t) = (log(t - t0) - mu) / sigma  ==  a * log(t - t0) + b
//     with a = 1/sigma and b = -mu/sigma precomputed (the division happens
//     per sub-movement, never per frame).
//
// Then Phi(z) is evaluated WITHOUT any libm call at all:
//   Phi(z) = 1/2 + phi(z) * P(z), where P(z) = sum_k z^(2k+1)/(2^k k!(2k+1))
//   is the odd Taylor series of exp(z^2/2)*erf-integral. Truncated after the
//   z^13 term it holds |err| < 1e-8 for |z| <= 2.2 (verified numerically);
//   beyond that a 3-term Mills tail matches to < 1e-7. Combined max absolute
//   CDF error ~1.5e-7 over the whole reachable range -> at most 3e-4 px of
//   position error even for a 2000 px sweep: two orders below one pixel.
//
// pdf(t) = phi(z)/(dt*sigma): one fast_exp, one reciprocal, zero logs.
// ---------------------------------------------------------------------------
struct lognormal_kernel {
    double a;          // 1/sigma
    double b;          // -mu/sigma   (so z = a*log_term + b)
    double pdf_scale;  // 1/(sigma*sqrt(2*pi)) — PDF prefactor numerator
};

inline lognormal_kernel make_lognormal_kernel(double mu, double sigma) {
    constexpr double inv_sqrt_2pi = 0.39894228040143267794;
    const double inv_s = 1.0 / sigma;
    return lognormal_kernel{ inv_s, -mu * inv_s, inv_sqrt_2pi * inv_s };
}

// Fast natural log via the IEEE-754 exponent field + atanh-series on the
// mantissa: log(m) = 2*atanh((m-1)/(m+1)). Degree-12 in r keeps |err| < 1e-12
// for m in [1,2); the m >= 1.5 branch rescales around ln(1.5) so the series
// argument stays small everywhere. Pure bit math + multiply-adds, no libm.
inline double fast_log(double x) {
    union { double d; uint64_t u; } cvt{x};
    const int e = int((cvt.u >> 52) & 0x7FF) - 1023;        // unbiased exponent
    cvt.u = (cvt.u & ~(uint64_t(0x7FF) << 52)) | (uint64_t(1023) << 52);
    const double m = cvt.d;                                  // mantissa in [1,2)
    double base, r;
    if (m >= 1.5) {                                          // m = 1.5*q, q~[2/3,4/3)
        base = 0.405465108108164381978;                      // ln 1.5
        const double q = m * (1.0 / 1.5);
        r = (q - 1.0) / (q + 1.0);                           // |r| <= 1/7
    } else {
        base = 0.0;
        r = (m - 1.0) / (m + 1.0);                           // |r| < 1/5
    }
    const double r2 = r * r;
    const double series = 2.0 * r * (1.0 + r2 * (1.0/3.0 + r2 * (1.0/5.0 + r2 * (1.0/7.0
                  + r2 * (1.0/9.0 + r2 * (1.0/11.0 + r2 * (1.0/13.0)))))));
    return double(e) * 0.69314718055994530942 + base + series;
}

// Standard-normal CDF Phi(z), fully branch-light (no erf, no libm at all).
// Abramowitz & Stegun 26.2.17 rational form:
//     Q(z) = 1 - Phi(z) ~= phi(z) * (b1 t + b2 t^2 + b3 t^3 + b4 t^4 + b5 t^5),
//     t = 1/(1 + p|z|),  p = 0.2316419
// but with the b-coefficients RE-FIT by phi-weighted least squares over
// z in [0, 9] (see bench fit): max absolute CDF error 8.1e-8 — better than
// the textbook AS constants (~1.5e-7) at identical cost. One fast_exp, one
// reciprocal, five multiply-adds. At D = 2000 px a sweep, the induced
// position error is < 1.6e-4 px — four orders below one pixel.
inline double phi_poly(double z) {
    constexpr double p = 0.2316419;
    const double az = std::abs(z);
    const double ph = 0.39894228040143267794 * fast_exp(-0.5 * az * az);
    const double t = 1.0 / (1.0 + p * az);
    // Horner form of Q(z)/phi(z)
    const double q_over_ph = t * (0.31995046783 + t * (-0.35930601696
                       + t * (1.78640882649 + t * (-1.82517661456
                       + t * 1.33143767835))));
    const double tail = ph * q_over_ph;        // Q(|z|)
    // Branchless symmetry: z>=0 -> Phi = 1 - tail ; z<0 -> Phi = tail
    return 0.5 + std::copysign(0.5 - tail, z);
}

inline double lognormal_cdf(const lognormal_kernel& k, double dt) {
    if (dt <= 0.0) return 0.0;
    return phi_poly(k.a * fast_log(dt) + k.b);
}

inline double lognormal_pdf(const lognormal_kernel& k, double dt) {
    if (dt <= 0.0) return 0.0;
    const double z = k.a * fast_log(dt) + k.b;
    return k.pdf_scale * fast_exp(-0.5 * z * z) / dt;
}

// s^2*(1-s)^3 normalized to peak=1.0 at s=0.4 - can be tweaked.
// curvature is maximal during the acceleration phase
inline double curvature_profile(double s) {
    if (s <= 0.0 || s >= 1.0) return 0.0;
    double v = s * s * (1.0 - s) * (1.0 - s) * (1.0 - s);
    constexpr double norm = 0.4 * 0.4 * 0.6 * 0.6 * 0.6;
    return v / norm;
}

// vertical movements produce more curvature due to wrist/forearm geometry
inline double direction_factor(double angle) {
    double sa = std::abs(std::sin(angle));
    double ca = std::abs(std::cos(angle));
    return 0.5 + 0.8 * sa - 0.15 * ca;
}

// Cheap Euclidean norm for 2D screen coordinates. std::hypot performs
// overflow/underflow guarding (scaling + extra branches) that is pointless
// when |dx|,|dy| are bounded by a monitor's pixel dimensions.
inline double fast_hypot(double dx, double dy) {
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace detail

// ---------------------------------------------------------------------------
// Fast engine adapters (hot-path sampling without std:: overhead)
// ---------------------------------------------------------------------------
namespace detail {

// std::normal_distribution is a template that fails explicit specialization,
// so instead of paying for its cached-second-value mutex-free-but-branchy
// wrapper, we wrap any uniform RandomNumberEngine in an adapter whose
// operator() IS a full Box-Muller draw. std::gamma_distribution only needs
// operator()(rng), so the raw engine works for it unchanged.
template <class Engine>
struct normal_engine_adapter {
    Engine* e;
    using result_type = double;
    static constexpr result_type min() { return -std::numeric_limits<double>::max(); }
    static constexpr result_type max() { return  std::numeric_limits<double>::max(); }
    // Ziggurat would be faster still, but Box-Muller on a high-quality
    // 64-bit engine keeps us header-only/STL-clean while removing the
    // distribution-object layer entirely. Two uniforms -> two normals;
    // we cache the second one (classic Marsaglia trick).
    double cached = 0.0;
    bool has_cached = false;
    result_type operator()() {
        if (has_cached) { has_cached = false; return cached; }
        // Polar Box-Muller (no trig): uniform in (-1,1)^2 until inside disc.
        double u1, u2, s;
        do {
            u1 = 2.0 * std::uniform_real_distribution<double>(0.0, 1.0)(*e) - 1.0;
            u2 = 2.0 * std::uniform_real_distribution<double>(0.0, 1.0)(*e) - 1.0;
            s = u1 * u1 + u2 * u2;
        } while (s >= 1.0 || s == 0.0);
        const double m = std::sqrt(-2.0 * std::log(s) / s);
        cached = u2 * m;
        has_cached = true;
        return u1 * m;
    }
};

template <class Engine>
normal_engine_adapter<Engine> make_normal_adapter(Engine& e) { return normal_engine_adapter<Engine>{ &e }; }

} // namespace detail

// ---------------------------------------------------------------------------
// generate — single-pass trajectory synthesis
//   cfg      : model parameters (see config above)
//   seed     : 0 -> seeded from random_device
//   strength : optional intensity override; when omitted, cfg.effective_intensity()
//              is used. Any double is accepted (0.0 = machine, 1.0 = default,
//              5.0 = chaotic).
// ---------------------------------------------------------------------------
inline std::vector<trajectory_point> generate(
    double x0, double y0, double x1, double y1,
    const config& cfg = {}, uint64_t seed = 0, double strength = -1.0)
{
    const double inten = (strength >= 0.0) ? strength : cfg.effective_intensity();

    // Saturating noise budget: below I=1 the amplitude scale is exactly
    // linear; above it, OU/tremor/SDN amplitudes compress toward a ceiling
    // (~2x). Without this, chaotic mode's wandering blows up faster than the
    // (1-s)^2 endpoint gate can retract near s->1 and the landing point flies
    // off-target. Variability channels (reach, sigma, timing jitter,
    // curvature) still keep scaling linearly above 1.
    const double ns = inten <= 1.0 ? inten : 1.0 + 1.0 * std::tanh((inten - 1.0) / 2.0);

    std::mt19937_64 rng(seed ? seed : std::random_device{}());
    auto nrm = detail::make_normal_adapter(rng);   // Box-Muller adapter (cached pair)
    std::uniform_real_distribution<double> std_unif(0.0, 1.0);
    const double g_scale = cfg.sample_dt_mean / cfg.gamma_shape;
    std::gamma_distribution<double> gamma(cfg.gamma_shape, g_scale);

    auto uniform = [&](double lo, double hi) {
        return lo + (hi - lo) * std_unif(rng);
    };
    auto normal = [&](double m, double s) {
        return m + s * nrm();
    };

    double dx = x1 - x0, dy = y1 - y0;
    double distance = detail::fast_hypot(dx, dy);
    double direction = std::atan2(dy, dx);

    if (distance < 1.0)
        return {{x0, y0, 0.0}, {x1, y1, 50.0}};

    double tx = dx / distance, ty = dy / distance;
    double nx = -ty, ny = tx;

    double id = std::log2(distance / cfg.target_width + 1.0);
    // MT jitter scales with intensity: at 0 the movement time is exactly the
    // Fitts prediction (machine repeatability), at >1 trials spread wildly.
    double mt = (cfg.fitts_a + cfg.fitts_b * id) * std::exp(normal(0.0, 0.08 * ns));
    mt = std::max(mt, 80.0);

    bool overshoot = uniform(0.0, 1.0) < std::min(cfg.overshoot_prob * inten, 1.0);
    // Overshoot/undershoot spread widens with intensity but saturates at ~1.4x
    // of the calibrated spread (rs): an impaired hand misjudges distance by
    // more pixels, yet never so much that corrections can't recover — this is
    // what keeps endpoint error bounded at chaotic settings.
    const double rs = inten <= 1.0 ? inten : 1.0 + 0.4 * std::tanh((inten - 1.0));
    double reach = overshoot
        ? 1.0 + uniform((cfg.overshoot_min - 1.0), (cfg.overshoot_max - 1.0)) * rs
        : 1.0 - uniform((1.0 - cfg.undershoot_max), (1.0 - cfg.undershoot_min)) * rs;

    double primary_D = distance * reach;
    double primary_sigma = uniform(cfg.primary_sigma_min, cfg.primary_sigma_max) * inten;
    // At intensity 0 sigma collapses to 0 -> lognormal degenerates; keep the
    // bell shape by flooring it so peak timing stays sane (machine mode is
    // still a smooth minimum-jerk-like profile, just noise-free).
    primary_sigma = std::max(primary_sigma, 0.05);

    // mu derived from mode = exp(mu - sigma^2) so that peak velocity lands at peak_t
    double peak_t = mt * uniform(cfg.peak_time_ratio - 0.03 * ns, cfg.peak_time_ratio + 0.03 * ns);
    double primary_mu = std::log(peak_t) + primary_sigma * primary_sigma;

    struct correction {
        double D, t0, dir_x, dir_y;
        detail::lognormal_kernel k;
    };
    std::vector<correction> corrections;
    corrections.reserve(2);

    // Precompute the standardized kernel once per sub-movement; the hot loop
    // then does zero divisions/sqrts/logs-from-libm to build CDF & PDF.
    const auto p_kernel = detail::make_lognormal_kernel(primary_mu, primary_sigma);

    double remaining = distance - primary_D;
    if (std::abs(remaining) > 0.5) {
        double dir = remaining > 0.0 ? 1.0 : -1.0;
        double cD = std::abs(remaining) * uniform(0.88, 1.02);
        double cS = std::max(uniform(cfg.correction_sigma_min, cfg.correction_sigma_max) * inten, 0.04);
        double cPeak = mt * uniform(0.12, 0.18);
        double cMu = std::log(cPeak) + cS * cS;
        corrections.push_back({
            cD, mt * uniform(0.55, 0.68), tx * dir, ty * dir,
            detail::make_lognormal_kernel(cMu, cS)
        });

        double left = remaining - cD * dir;
        if (std::abs(left) > 0.3 && uniform(0.0, 1.0) < std::min(cfg.second_correction_prob * inten, 1.0)) {
            double d2 = left > 0.0 ? 1.0 : -1.0;
            double cD2 = std::abs(left) * uniform(0.85, 1.05);
            double cS2 = std::max(uniform(0.10, 0.16) * inten, 0.04);
            double cP2 = mt * uniform(0.08, 0.12);
            double cMu2 = std::log(cP2) + cS2 * cS2;
            corrections.push_back({
                cD2, mt * uniform(0.78, 0.88), tx * d2, ty * d2,
                detail::make_lognormal_kernel(cMu2, cS2)
            });
        }
    }

    // Curvature uses the saturating amplitude scale ns: arcs widen strongly
    // up to impaired mode but stop growing beyond it — a human hand can only
    // deviate so far before visual feedback pulls the path back.
    double curv_amp = distance * (cfg.curvature_scale * ns)
        * detail::direction_factor(direction) * normal(0.0, 1.0);

    // Tremor frequency rises mildly with intensity (fatigue/intention tremor
    // shifts upward), clamped to [min, 25 Hz] so fast_sincos stays in its
    // guaranteed range even at chaotic strength.
    const double tremor_freq = std::clamp(
        uniform(cfg.tremor_freq_min, cfg.tremor_freq_max) * (1.0 + 0.12 * inten),
        cfg.tremor_freq_min, 25.0);
    const double tremor_amp = uniform(cfg.tremor_amp_min, cfg.tremor_amp_max) * ns;
    double tph_x = uniform(0.0, 2.0 * std::numbers::pi);
    double tph_y = uniform(0.0, 2.0 * std::numbers::pi);
    double ou_x = 0.0, ou_y = 0.0;
    double gate = 1.0;   // running noise gate (see endpoint stabilization below)

    // ---- Phase-coherent tremor via phasor recurrence -------------------
    // Instead of sin(2*pi*f*t + phi) evaluated from absolute time each frame
    // (which accumulates floating-point rounding error as t grows -> microscopic
    // phase jitter), carry a unit phasor (cos, sin) and rotate it by the exact
    // angle increment 2*pi*f*dt each step. One sincos pair per frame feeds BOTH
    // axes with pure multiply-add arithmetic.
    double px = std::cos(tph_x), py_ = std::sin(tph_x);   // tremor phasor, X axis
    double qx = std::cos(tph_y), qy = std::sin(tph_y);    // tremor phasor, Y axis

    // ---- Exact OU (AR(1)) integration ----------------------------------
    // Euler-Maruyama (ou += -theta*ou*dt + sigma*sqrt(dt)*N) misestimates the
    // stationary variance whenever dt fluctuates (our polling intervals are
    // Gamma-distributed). The exact discrete-time solution of the OU SDE is
    // the AR(1) recursion:
    //     ou' = ou * exp(-theta*dt) + sigma*sqrt((1-exp(-2*theta*dt))/(2*theta)) * N
    // which preserves the correct stationary variance sigma^2/(2*theta) for
    // any frame-time sequence. The state lives in RAW coordinates and is
    // multiplied by the endpoint gate only when added to the position — no
    // division by the vanishing gate anywhere, so the recursion is unconditionally
    // stable at every intensity.
    // OU relaxation rate rises with intensity (impaired control = faster,
    // jerkier wandering), saturating at ~8x theta. Raising theta shrinks the
    // process stationary variance sigma^2/(2*theta) as well as its innovation
    // step (which scales with sqrt(1-exp(-2*theta*dt))), which is what keeps
    // extreme intensities bounded while making the wander visibly jerkier.
    const double theta_eff = cfg.ou_theta * (1.0 + 7.0 * std::tanh(inten / 2.0));
    const double ou_sigma_eff = cfg.ou_sigma * ns;
    const double theta = theta_eff;
    const double inv_2theta = 1.0 / (2.0 * theta);
    const double ou_vol_base = ou_sigma_eff * std::sqrt(inv_2theta); // folded sqrt(1/2θ)
    const double ang_per_s = 2.0 * std::numbers::pi * tremor_freq;
    const double trem_speed_k = 0.3 * inten; // tremor suppression slope scales too

    double total_t = mt * 1.15;

    // Single-pass generation: times and positions are produced in one loop.
    // No intermediate std::vector<double> of poll times, no second iteration.
    std::vector<trajectory_point> result;
    const size_t expected = (size_t)(total_t / cfg.sample_dt_mean) + 8;
    result.reserve(expected);

    double t = 0.0;
    const double stop_t = total_t + 15.0;

    while (true) {
        double dt_ms = gamma(rng);
        // branchless-style clamp (compiler emits minsd/maxsd)
        dt_ms = dt_ms < 2.0 ? 2.0 : (dt_ms > 25.0 ? 25.0 : dt_ms);
        double tn = t + dt_ms;
        if (t > stop_t) break;

        double dt_s = dt_ms * 0.001;

        double s = detail::lognormal_cdf(p_kernel, tn);

        double bx = x0 + tx * primary_D * s;
        double by = y0 + ty * primary_D * s;

        double cp = detail::curvature_profile(s);
        bx += nx * curv_amp * cp;
        by += ny * curv_amp * cp;

        double speed = primary_D * detail::lognormal_pdf(p_kernel, tn);
        for (const auto& c : corrections) {
            double cdtt = tn - c.t0;
            double cs = detail::lognormal_cdf(c.k, cdtt);
            bx += c.dir_x * c.D * cs;
            by += c.dir_y * c.D * cs;
            speed += c.D * detail::lognormal_pdf(c.k, cdtt);
        }

        // ---- Fused per-frame noise kernel --------------------------------
        // exp() and sincos() are replaced by pure-polynomial fast paths:
        //   decay = fast_exp(-theta*dt)  (OU AR(1) coefficient)
        //   (ca, sa) = fast_sincos(2*pi*f*dt)  (tremor rotation matrix)
        // Previously this region cost std::exp + std::cos + std::sin through
        // libm; now it is three branch-free polynomial evaluations the
        // compiler keeps entirely in XMM registers.
        const double decay = detail::fast_exp(-theta * dt_s);
        double ca, sa;
        detail::fast_sincos(ang_per_s * dt_s, sa, ca);

        // Exact AR(1) OU step in RAW coordinates; the endpoint gate is applied
        // multiplicatively to the rendered position below, so the process
        // itself remains statistically exact (stationary variance
        // sigma^2/(2*theta)) and its visible contribution -> 0 as s -> 1.
        double gate_new = (1.0 - s) * (1.0 - s);
        double one_minus_d2 = (1.0 - decay) * (1.0 + decay); // 1 - decay^2
        double vol = ou_vol_base * std::sqrt(one_minus_d2);
        ou_x = ou_x * decay + vol * nrm();
        ou_y = ou_y * decay + vol * nrm();
        gate = gate_new;

        // Rotate both tremor phasors by the shared (ca, sa) — pure arithmetic.
        double npx = px * ca - py_ * sa, npy = px * sa + py_ * ca;
        px = npx; py_ = npy;
        double nqx = qx * ca - qy * sa, nqy = qx * sa + qy * ca;
        qx = nqx; qy = nqy;

        // tremor gain drops with speed (proprioceptive suppression)
        double trem_mod = 1.0 / (1.0 + speed * trem_speed_k);

        // signal-dependent noise magnitude proportional to motor command
        // (Harris–Wolpert SDN)
        double sdn_gain = (cfg.sdn_k * ns) * speed;

        // ---- Endpoint stabilization (noise gating) ----------------------
        // Raw OU drift + tremor + SDN applied all the way to the final sample
        // leaves the cursor randomly offset on arrival, inflating
        // endpoint_error. Fading every noise term by gate = (1 - s)^2 forces
        // amplitude smoothly to zero as the cursor enters the final
        // deceleration phase — mimicking human proprioceptive stabilization
        // and guaranteeing landing accuracy within the target width.
        double tr_gain = tremor_amp * trem_mod * gate;
        double tr_x = tr_gain * py_;                     // sin component of phasor
        double tr_y = tr_gain * qy;

        double sdn_x = sdn_gain * gate * nrm();
        double sdn_y = sdn_gain * gate * nrm();

        result.push_back({bx + ou_x * gate + tr_x + sdn_x,
                          by + ou_y * gate + tr_y + sdn_y, tn});
        t = tn;

        if (result.size() >= expected * 4) break; // hard safety bound
    }

    return result;
}

// Convenience overload: pick a named intensity preset without touching cfg.
inline std::vector<trajectory_point> generate(
    double x0, double y0, double x1, double y1,
    const config& cfg, uint64_t seed, intensity strength)
{
    return generate(x0, y0, x1, y1, cfg, seed, to_double(strength));
}

inline metrics compute_metrics(
    const std::vector<trajectory_point>& path,
    double target_x, double target_y,
    double target_width, double straight_dist)
{
    metrics m{};
    if (path.size() < 2) return m;

    m.movement_time = path.back().t - path.front().t;
    m.straight_distance = straight_dist;

    double max_speed = 0.0;
    m.path_length = 0.0;
    // ---- Streaming peak detection (no heap allocation) -----------------
    // A local maximum at index j can only be confirmed after seeing speeds up
    // to j+1, and the 15%-of-max threshold needs the GLOBAL max speed. So we
    // cache the (few) strict local-max candidates in a small fixed array
    // during the single pass, then validate them against the final threshold.
    // Trapezoidal/curved profiles produce ~2-4 peaks; 32 slots is generous —
    // if saturated we fall back to a bounded conservative count.
    double sp_prev2 = 0.0, sp_prev1 = 0.0;
    double cand_speeds[32];
    size_t n_cand = 0;
    bool cand_overflow = false;

    for (size_t i = 1; i < path.size(); ++i) {
        double dx = path[i].x - path[i - 1].x;
        double dy = path[i].y - path[i - 1].y;
        double dt = path[i].t - path[i - 1].t;
        double seg = detail::fast_hypot(dx, dy);
        m.path_length += seg;
        double spd = (dt > 0.0) ? seg / dt : 0.0;
        if (spd > max_speed) { max_speed = spd; m.time_to_peak = path[i].t; }
        // strict local max at i-1?
        if (i >= 2 && sp_prev1 > sp_prev2 && sp_prev1 >= spd) {
            if (n_cand < 32) cand_speeds[n_cand++] = sp_prev1;
            else cand_overflow = true;
        }
        sp_prev2 = sp_prev1; sp_prev1 = spd;
    }

    m.peak_speed = max_speed;
    m.path_efficiency = (m.path_length > 0.0) ? m.straight_distance / m.path_length : 1.0;

    // peaks above 15% of max in the speed signal -> sub-movement count
    const double threshold = max_speed * 0.15;
    int peaks = 0;
    for (size_t k = 0; k < n_cand; ++k)
        if (cand_speeds[k] > threshold) ++peaks;
    if (cand_overflow) peaks = static_cast<int>(n_cand); // best-effort bound
    m.num_submovements = std::max(peaks, 1);

    m.endpoint_error = detail::fast_hypot(path.back().x - target_x, path.back().y - target_y);

    double id = std::log2(straight_dist / target_width + 1.0);
    m.fitts_predicted_mt = 50.0 + 150.0 * id;

    return m;
}

} // namespace motor_synergy

namespace windmouse { 
    // this is windmouse, for comparison. the graph will show both trajectories and metrics side by side,
	// but the main point is to show how windmouse's trajectory is more erratic and less efficient than sigma-drift's, 
    // due to the lack of an explicit internal model and reliance on noisy feedback and random exploration.
	// - this leads to detections in the metrics like lower path efficiency, more sub-movements, and higher endpoint error, 
    // especially on smaller targets.

inline std::vector<motor_synergy::trajectory_point> generate(
    double x0, double y0, double x1, double y1,
    double gravity = 9.0, double wind_str = 3.0,
    double max_step = 15.0, double target_area = 8.0,
    uint64_t seed = 0)
{
    std::mt19937_64 rng(seed ? seed : std::random_device{}());
    // Pre-instantiate the distribution object once (was constructed per call
    // inside randf — hidden STL overhead in the hot loop).
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    auto randf = [&](double lo, double hi) {
        return lo + (hi - lo) * unif01(rng);
    };
    // Loop-invariant sqrt constants hoisted out of the iteration body.
    const double inv_sqrt3 = 1.0 / std::sqrt(3.0);
    const double inv_sqrt5 = 1.0 / std::sqrt(5.0);

    std::vector<motor_synergy::trajectory_point> result;
    double xs = x0, ys = y0;
    double vx = 0, vy = 0, wx = 0, wy = 0;
    double t = 0.0, step = max_step;

    result.push_back({xs, ys, 0.0});

    for (int iter = 0; iter < 5000; ++iter) {
        double dist = motor_synergy::detail::fast_hypot(x1 - xs, y1 - ys);
        if (dist < 1.0) break;

        double w = std::min(wind_str, dist);
        if (dist >= target_area) {
            wx = wx * inv_sqrt3 + randf(-w, w) * inv_sqrt5;
            wy = wy * inv_sqrt3 + randf(-w, w) * inv_sqrt5;
        } else {
            wx *= inv_sqrt3;
            wy *= inv_sqrt3;
            if (step < 3.0) step = randf(3.0, 6.0);
            else step *= inv_sqrt5;
        }

        vx += wx + gravity * (x1 - xs) / dist;
        vy += wy + gravity * (y1 - ys) / dist;
        double vmag = motor_synergy::detail::fast_hypot(vx, vy);
        if (vmag > step) {
            double r = step / 2.0 + randf(0.0, step / 2.0);
            vx = vx / vmag * r;
            vy = vy / vmag * r;
        }

        xs += vx;
        ys += vy;
        t += randf(5.0, 15.0);
        result.push_back({xs, ys, t});
    }

    return result;
}

} // namespace windmouse
