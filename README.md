# SigmaDrift

A biomechanically-grounded mouse movement algorithm that outperforms WindMouse across every metric that matters for human-like trajectory generation.

Built for my Masters research on novel mouse movement humanization techniques.

Full paper is now released: https://zenodo.org/records/18872499

## What it does

SigmaDrift generates point-to-point mouse trajectories using six interacting components from computational motor control research:

- **Sigma-lognormal velocity primitives** — asymmetric bell-shaped speed profiles from Plamondon's Kinematic Theory
- **Two-phase surge architecture** — ballistic stroke (~93% of distance) followed by 0-2 corrective sub-movements
- **Ornstein-Uhlenbeck lateral drift** — mean-reverting stochastic hand drift
- **Signal-dependent noise** — motor noise scales with command magnitude (Harris-Wolpert), making Fitts' Law emerge naturally
- **Speed-modulated physiological tremor** — 8-12 Hz tremor suppressed during fast ballistic movement
- **Gamma-distributed inter-sample timing** — non-constant polling intervals matching real hardware behavior

## Results vs WindMouse

Same distance (~630px), same target width (20px):

| Metric | SigmaDrift | WindMouse | Real Human |
|---|---|---|---|
| Movement Time | 827 ms | 499 ms | ~750-850 ms |
| Fitts' Compliance | Yes (~8%) | No | Yes |
| Sub-Movements | 2 | 15 | 1-3 |
| Path Efficiency | 0.985 | 0.973 | 0.95-0.99 |
| Velocity Profile | Bell-shaped | Jagged | Bell-shaped |

## Intensity scaling — one dial from machine to chaos

Every stochastic / imperfection channel in the model is driven by a single
`intensity` value, so the exact same engine can be tuned from surgical,
noise-free pointing to heavily impaired, super-noisy movement without editing
any config field:

| Preset (`intensity::`) | Value | Character |
|---|---|---|
| `machine` | 0.00 | Pure minimum-jerk bell curve: zero OU/tremor/SDN/curvature, no overshoots, deterministic Fitts MT (poll timestamps still Gamma-jittered) |
| `calm` | 0.35 | Steady hand, minimal wobble |
| `default_hum` | 1.00 | The calibrated SigmaDrift human profile (all defaults below) |
| `shaky` | 1.75 | Fatigued / caffeinated |
| `impaired` | 3.00 | Strongly noisy, large misreaches and long corrections |
| `chaotic` | 5.00 | Maximum chaos — violent tremor, huge arcs |

Any raw double in `0.0 .. ~6.0` also works between presets.

**What scales linearly with intensity I:**
- Noise amplitudes: `ou_sigma`, `tremor_amp_{min,max}`, `sdn_k`
- Path shape: `curvature_scale`
- Trial-to-trial variability: lognormal σ ranges, MT jitter, peak-time & correction-timing jitter, overshoot spread
- Correction *frequency*: `overshoot_prob`, `second_correction_prob` grow with I

**Deliberately intensity-invariant:** `fitts_a/b`, `target_width`, `ou_theta`
(the OU process itself), `sample_dt_mean` / `gamma_shape` (poll rate is a
property of the OS/hardware), and tremor *base* frequency.

**Saturating noise budget (why extreme modes stay usable):** above I = 1,
visible noise *amplitudes* compress toward a ~2× ceiling via
`ns = 1 + tanh((I-1)/2)` instead of growing unbounded. A human hand can only
deviate so far before visual feedback pulls it back; this keeps chaotic-mode
trajectories erratic but recoverable rather than numerically explosive.
Reach/variability channels (misjudged distances, timing spread, curvature up
to impaired levels) keep scaling so high intensities still look distinctly
more impaired. Two related guards: tremor frequency rises mildly with I but
is clamped to ≤ 25 Hz (physiological intention-tremor band), and OU relaxation
rate θ_eff increases with I (impaired control wanders faster/jerkier) while
shrinking the AR(1) stationary variance σ²/(2θ_eff) — bounded wander at any setting.

Measured endpoint behaviour across the scale (3000 trials, 583 px move,
20 px target — from `bench/stats.cpp`):

| Intensity | Avg endpoint error | Max error | Avg MT | Path efficiency |
|---|---|---|---|---|
| 0.00 (machine) | 0.000 px | 0.000 px | 917 ms | 1.000 |
| 0.35 (calm) | 11.3 px | 16.3 px | 917 ms | 0.978 |
| 1.00 (human) | 12.4 px | 46.9 px | 919 ms | 0.965 |
| 1.75 (shaky) | 5.0 px | 23.6 px | 920 ms | 0.994 |
| 3.00 (impaired) | 85.3 px | 205 px | 924 ms | 1.170 |
| 5.00 (chaotic) | 313 px | 462 px | 925 ms | 2.281 |

Movement time stays essentially constant across the whole dial (only its
variance grows) — exactly like real subjects: impairment changes accuracy,
not average duration. Note the avg/max error columns mix landing regimes; see
the endpoint-stabilization section below for the gated-path breakdown.

```cpp
// named preset
auto path = motor_synergy::generate(x0, y0, x1, y1, cfg, seed,
                                    motor_synergy::intensity::chaotic);
// or raw value, or baked into the config
auto path2 = motor_synergy::generate(x0, y0, x1, y1, cfg, seed, 2.3);
cfg.strength = motor_synergy::intensity::calm;   // default for positional calls
```

Intensity is applied at draw time inside `generate()` and never mutates the
config struct, so one `cfg` stays reusable across all intensities and every
field keeps its literal calibrated meaning.

## Engine upgrades over the base release

The current engine keeps the published paper's model intact but fixes three
accuracy flaws and rewrites the hot loop for speed. All numbers below are
re-measured from this repo (`bench/`, g++ `-O2`, header-only).

### Accuracy

1. **Endpoint stabilization (noise gating).** Base SigmaDrift applied OU
   drift, tremor and SDN uniformly until the final sample, so the cursor
   landed at a random offset from the true target — pure jitter inflation in
   `endpoint_error`. Every noise term is now faded by `gate = (1 - s)²`,
   driving amplitude smoothly to zero as the cursor enters final
   deceleration: proprioceptive stabilization, and the rendered path provably
   converges onto the ballistic endpoint. Gated-path measurement (3000
   trials, default intensity): the 64% of trials that land inside the target
   average **1.52 px** error (max 5.47 px); the remaining 36% are not noise
   jitter but genuine *misses* (under-reach / overshoot beyond the 10 px
   radius, avg 31.6 px) — i.e. realistic Fitts failures, which is what a
   humanizer should produce. At `machine` intensity gating is moot and error
   is exactly 0.
2. **Exact OU integration (AR(1)).** Euler–Maruyama (`ou += -θ·ou·dt + σ√dt·N`)
   misestimates stationary variance whenever dt fluctuates — and our poll
   intervals are Gamma-distributed. Replaced with the exact discrete-time
   solution `ou' = ou·e^(−θdt) + σ√((1−e^(−2θdt))/(2θ))·N`. Monte-Carlo under
   Gamma-jittered frame times (2 M steps): AR(1) lands **−0.58%** off the
   theoretical σ²/(2θ) = 0.2057 where Euler–Maruyama drifts **+2.39%**. The
   OU state lives in raw coordinates and is gated only when added to the
   position — no division by the vanishing gate anywhere, so the recursion is
   unconditionally stable at every intensity.
3. **Phase-coherent tremor.** `sin(2πft + φ)` evaluated from absolute time
   accumulates floating-point rounding error as t grows → microscopic phase
   jitter. The tremor now carries a unit phasor rotated by the exact angle
   increment each frame (rotation-matrix recurrence), keeping the wave
   mathematically pure over arbitrarily long trajectories.

### Speed

- **RNG objects built once.** `std::normal_distribution` /
  `std::gamma_distribution` are no longer constructed in the hot loop;
  sampling goes through a cached-pair Box-Muller adapter (`m + s·nrm()`) and
  an inline Marsaglia–Tsang gamma draw with all loop invariants hoisted.
  Measured per-call cost: normal adapter **31.3 ns** vs 34.2 ns for
  `std::normal_distribution`; the std gamma call alone cost **59.7 ns/frame**,
  now replaced by the hoisted inline sampler.
- **Single-pass generation.** The intermediate `std::vector<double>` of poll
  times and the second positioning loop are gone; points emit directly from
  one `while` loop into a pre-reserved result vector.
- **Saturation-aware sub-movement kernels.** Each lognormal kernel precomputes
  its saturation time (where the CDF becomes exactly 1.0); after that the
  primary freezes at full displacement and finished corrections leave the
  loop entirely — ~40% less per-frame math on the coasting half of every
  trajectory, with zero accuracy loss (the tail contribution there is
  bit-exactly 0/1).
- **Fast math substitutions** (each validated against libm in `bench/`):
  - `std::hypot` → `√(dx²+dy²)` (`fast_hypot`) — overflow safety is pointless for screen coordinates.
  - `std::erf` → Abramowitz & Stegun-style poly/exponential kernel (`fast_erf`, max abs err **1.43e-7**) feeding a refitted Φ approximation (`phi_poly`, max deviation from libm **7.1e-8** → worst-case position error **0.0001 px** at a 2000 px move).
  - `std::exp` → branch-free degree-6 minimax `fast_exp` (max rel err **4.4e-9**).
  - `std::log` → bit-manipulation `fast_log`; one evaluation serves both the CDF and PDF of a sub-movement.
  - `sin`+`cos` pair → fused polynomial `fast_sincos` (**9.9 ns** vs **17.9 ns** for libm), computed once per frame and shared by both tremor axes.
- Net effect: whole-trajectory generation runs at **~41 µs** per ~120-point
  trajectory at `-O2` (down from ~72 µs in the first optimized pass and
  several× faster than the original multi-loop base), and path efficiency
  improves slightly since gated noise no longer inflates path length.

## Usage

Header-only C++20, zero dependencies.

```cpp
#include "motor_synergy.h"

auto path = motor_synergy::generate(start_x, start_y, target_x, target_y);

for (auto& pt : path) {
    // pt.x, pt.y = position
    // pt.t = timestamp in ms
}
```

Custom configuration:

```cpp
motor_synergy::config cfg;
cfg.target_width = 16.0;
cfg.overshoot_prob = 0.20;
cfg.strength = motor_synergy::intensity::shaky;

auto path = motor_synergy::generate(x0, y0, x1, y1, cfg);
auto m = motor_synergy::compute_metrics(path, x1, y1, cfg.target_width, dist);
```

`compute_metrics` returns movement time, path length/efficiency, peak speed,
time-to-peak, sub-movement count, endpoint error and the Fitts-predicted MT —
it runs as a single streaming pass with no heap allocation.

## Benchmarks

`bench/` contains the validation harness used for every number above
(Linux/macOS; compile with `g++ -std=c++20 -O2`):

| Harness | Verifies |
|---|---|
| `stats.cpp` | OU stationary variance under Gamma jitter; endpoint error / MT / efficiency sweep across all intensity presets; timing vs distance & strength |
| `opprof.cpp` | Per-primitive costs (log, exp, sqrt, gamma, sincos) and whole-trajectory timing |
| `kernel_val.cpp` | `fast_log`, `phi_poly` and the fused lognormal kernels vs libm references |
| `regen.cpp` | Full-pipeline regression: optimized path vs exact libm reference trajectories |
| `cand.cpp` / `expval.cpp` / `phifit.cpp` | Candidate-kernel shootouts and coefficient fits (fast_exp, erf, Φ) |
| `pdfdbg.cpp` | Lognormal PDF spot-checks at specific times |

## Harness

The included Win32 visualization harness (`main.cpp`) provides side-by-side comparison:

- **Space** — generate SigmaDrift trajectory (animated)
- **W** — generate WindMouse trajectory
- **R** — record your own mouse movement
- **S** — export trajectories to CSV
- **+/-** — adjust target width

Bottom panel shows velocity profile graphs for all trajectories overlaid.

## Building

Visual Studio 2022 with C++20. Open `SigmaDrift.slnx`, build x64 Release.

## References

- Plamondon — Kinematic Theory of Rapid Human Movements
- Flash & Hogan (1985) — Minimum-jerk model
- Harris & Wolpert (1998) — Signal-dependent noise
- Uhlenbeck & Ornstein (1930) — Mean-reverting stochastic processes
- Marsaglia & Tsang (2000) — Quick gamma variate generator
- Abramowitz & Stegun — Handbook of Mathematical Functions (7.1.26)
- Muller et al. (2017) — Control-theoretic models of pointing
- Acien et al. (2022) — BeCAPTCHA-Mouse
- Liu et al. (2024) — DMTG diffusion-based generation
