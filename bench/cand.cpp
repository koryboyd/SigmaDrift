// Candidate evaluation harness: accuracy + speed of proposed primitives
// before adoption into motor_synergy.h.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <random>

using clk = std::chrono::steady_clock;
static double now_sec(){ return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }
static volatile double sink;

// ---- current implementations (copied from header) -------------------------
inline double fast_exp_cur(double x) {
    x = (x < -60.0) ? -60.0 : ((x > 60.0) ? 60.0 : x);
    constexpr double log2e = 1.4426950408889634074;
    double z = x * log2e;
    constexpr double magic = 6755399441055744.0;
    double n = (z + magic) - magic;
    double f = z - n;
    const double ln2 = 0.6931471805599453094;
    double p = f * ln2;
    double poly = 1.0 + p * (1.0 + p * (0.5 + p * (1.0/6.0 + p * (1.0/24.0
              + p * (1.0/120.0 + p * (1.0/720.0 + p * (1.0/5040.0 + p * (1.0/40320.0))))))));
    const int64_t ni = (int64_t)n;
    auto pow2 = [](int64_t k) { double d; const int64_t bits = (1023 + k) << 52; memcpy(&d,&bits,8); return d; };
    const int64_t hi = ni >> 1;
    return poly * pow2(hi) * pow2(ni - hi);
}
inline void fast_sincos_cur(double a, double& s, double& c) {
    const double a2 = a*a;
    s = a*(1.0 + a2*(-1.0/6.0 + a2*(1.0/120.0 + a2*(-1.0/5040.0
        + a2*(1.0/362880.0 + a2*(-1.0/39916800.0 + a2*(1.0/6227020800.0)))))));
    c = 1.0 + a2*(-0.5 + a2*(1.0/24.0 + a2*(-1.0/720.0 + a2*(1.0/40320.0
        + a2*(-1.0/3628800.0 + a2*(1.0/479001600.0))))));
}
inline double fast_erf_cur(double x) {
    constexpr double p=0.3275911,a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429;
    const double ax=std::fabs(x), tt=1.0/(1.0+p*ax);
    const double poly=tt*(a1+tt*(a2+tt*(a3+tt*(a4+tt*a5))));
    const double y=1.0-poly*fast_exp_cur(-ax*ax);
    return std::copysign(y,x);
}

// ---- candidate: bit-trick exp2 --------------------------------------------
inline double fast_exp_new(double x){
    x = (x < -60.0) ? -60.0 : ((x > 60.0) ? 60.0 : x);
    const double w = x * 1.4426950408889634074;      // x*log2(e)
    double wi = std::floor(w);                        // rounds down -> f in [0,1)
    double f = w - wi;
    // minimax-ish Taylor for 2^f on [0,1] (degree 8, err ~1e-12)
    const double L = 0.6931471805599453094;
    double p = f * L;
    double poly = 1.0 + p*(1.0 + p*(0.5 + p*(1.0/6.0 + p*(1.0/24.0 + p*(1.0/120.0
              + p*(1.0/720.0 + p*(1.0/5040.0 + p*(1.0/40320.0))))))));
    union { double d; uint64_t u; } cvt;
    cvt.u = (uint64_t)(int64_t)(wi + 1023.0) << 52;   // 2^wi, wi in [-1023, 86] here
    return poly * cvt.d;
}

// ---- candidate: CODY erf ----------------------------------------------------
inline double fast_erf_cody(double x){
    static const double P[8]={
        1.183196179e-4, 6.37853973e-4, 2.59065e-2, 2.2609718e-1,
        5.3307941e-1, 1.0, 0.0, 0.0};
    // Cody uses erfc approx for |x|>=1: erfc(x)=exp(-x*x)/x*R(1/x*x)
    const double ax = std::fabs(x);
    double res;
    if (ax >= 1.0) {
        static const double a[5]={1.183196179e-4, 6.37853973e-4, 2.59065e-2, 2.2609718e-1, 5.3307941e-1};
        static const double b[5]={1.0002368, 1.876226, 1.83561, 8.57293e-1, 1.6470221e-1};
        double t = 1.0/(ax*ax);
        double pv = a[0]; for(int i=1;i<5;i++) pv = pv*t + a[i];
        double qv = b[0]; for(int i=1;i<5;i++) qv = qv*t + b[i];
        res = 1.0 - std::exp(-ax*ax)/ax * (pv/qv);   // erf = 1 - erfc
        // NOTE: benchmark against std::erf; uses libm exp only for large tail.
    } else {
        // odd polynomial for |x|<1: erf(x) = x * P(x^2)
        static const double c[9]={1.6596613e-7, 9.93677e-5, 1.35238e-3, 7.271621e-3,
            3.520937e-3, 2.1814394e-2, 5.3897166e-2, 2.169095e-1, 1.0};
        double t = x*x;
        double pv = c[0]; for(int i=1;i<9;i++) pv = pv*t + c[i];
        res = ax * pv;
        res = std::copysign(res, x);
    }
    return res;
}

// ---- candidate: single-cos phasor with copysign sine ------------------------
inline void fast_sincos_new(double a, double& s, double& c){
    const double a2 = a*a;
    c = 1.0 + a2*(-0.5 + a2*(1.0/24.0 + a2*(-1.0/720.0 + a2*(1.0/40320.0
        + a2*(-1.0/3628800.0 + a2*(1.0/479001600.0))))));
    const double sa = a*(1.0 + a2*(-1.0/6.0 + a2*(1.0/120.0 + a2*(-1.0/5040.0
        + a2*(1.0/362880.0 + a2*(-1.0/39916800.0 + a2*(1.0/6227020800.0)))))));
    s = std::copysign(sa, sa); // placeholder (same); real trick is at use site
}

int main(){
    // ---------- accuracy: fast_exp_new vs std::exp over OU range ----------
    double worst=0, wx=0;
    for(double x=-60;x<=0.0001;x+=1e-5){
        double e=std::fabs(fast_exp_new(x)-std::exp(x))/std::exp(x);
        if(e>worst){worst=e;wx=x;}
    }
    printf("fast_exp_new max rel err [-60,0]: %.3e at x=%.4f\n", worst, wx);

    // ---------- accuracy: cody erf ----------
    worst=0; wx=0;
    for(double x=-6;x<=6;x+=1e-5){
        double e=std::fabs(fast_erf_cody(x)-std::erf(x));
        if(e>worst){worst=e;wx=x;}
    }
    printf("fast_erf_cody max abs err [-6,6]: %.3e at x=%.4f\n", worst, wx);
    worst=0; wx=0;
    for(double x=-6;x<=6;x+=1e-5){
        double e=std::fabs(fast_erf_cur(x)-std::erf(x));
        if(e>worst){worst=e;wx=x;}
    }
    printf("fast_erf_AS   max abs err [-6,6]: %.3e at x=%.4f\n", worst, wx);

    // CDF-level error: pixel displacement. s = 0.5*(1+erf(z/sqrt2)).
    // ds/dpos: position error = D * ds. D up to ~2000px.
    // find max ds between the two approximations
    double dsmax=0;
    for(double z=-6;z<=6;z+=1e-5){
        double s1=0.5*(1.0+fast_erf_cur(z*0.70710678118654752440));
        double s2=0.5*(1.0+fast_erf_cody(z*0.70710678118654752440));
        double d=std::fabs(s1-s2);
        if(d>dsmax) dsmax=d;
    }
    printf("CDF delta between AS & Cody: %.3e  (-> %.4f px at D=2000)\n", dsmax, dsmax*2000.0);

    // ---------- speed ----------
    const long N=20000000L;
    double t0,t,acc;

    t0=now_sec(); acc=0;
    for(long i=1;i<=N;i++) acc+=fast_exp_cur(-(double)i/N);
    t=now_sec()-t0; sink=acc; printf("fast_exp_cur  %6.2f ns\n", t*1e9/N);

    t0=now_sec(); acc=0;
    for(long i=1;i<=N;i++) acc+=fast_exp_new(-(double)i/N);
    t=now_sec()-t0; sink=acc; printf("fast_exp_new  %6.2f ns\n", t*1e9/N);

    t0=now_sec(); acc=0;
    for(long i=1;i<=N;i++) acc+=fast_erf_cur((double)(i%12000)/1000.0-6.0);
    t=now_sec()-t0; sink=acc; printf("fast_erf_AS   %6.2f ns\n", t*1e9/N);

    t0=now_sec(); acc=0;
    for(long i=1;i<=N;i++) acc+=fast_erf_cody((double)(i%12000)/1000.0-6.0);
    t=now_sec()-t0; sink=acc; printf("fast_erf_Cody %6.2f ns\n", t*1e9/N);

    // gamma cost comparison
    {
        std::mt19937_64 rng(1);
        std::gamma_distribution<double> gam(3.5, 7.8/3.5);
        t0=now_sec(); acc=0;
        for(long i=1;i<=N;i++) acc+=gam(rng);
        t=now_sec()-t0; sink=acc; printf("std gamma     %6.2f ns\n", t*1e9/N);

        std::uniform_real_distribution<double> u01(0.0,1.0);
        const double shp=3.5, scl=7.8/3.5;
        auto marsaglia_tsang = [&](){
            // Marsaglia-Tsang for shape>1
            const double d=shp-1.0/3.0, c=1.0/std::sqrt(9.0*d);
            while(true){
                double x, v;
                do{ x=std::normal_distribution<double>(0,1)(rng); v=1.0+c*x; }while(v<=0.0);
                v=v*v*v; double uv=u01(rng);
                if(uv < 1.0-0.0331*x*x*x*x) return d*v*scl;
                if(std::log(uv) < 0.5*x*x + d*(1.0-v+std::log(v))) return d*v*scl;
            }
        };
        // hand-rolled with cached normal via polar BM:
        double nc=0; bool hc=false;
        auto norm=[&]()->double{
            if(hc){hc=false;return nc;}
            double u1,u2,s;
            do{u1=2*u01(rng)-1;u2=2*u01(rng)-1;s=u1*u1+u2*u2;}while(s>=1||s==0);
            double m=std::sqrt(-2*std::log(s)/s); nc=u2*m; hc=true; return u1*m;
        };
        t0=now_sec(); acc=0;
        for(long i=1;i<=N;i++){
            const double d=shp-1.0/3.0, c=1.0/std::sqrt(9.0*d);
            double x,v;
            while(true){ x=norm(); v=1.0+c*x; if(v>0) break; }
            v=v*v*v; double uv=u01(rng);
            if(uv < 1.0-0.0331*x*x*x*x){acc+=d*v*scl;break_outer_: {} if(false){} acc-=d*v*scl; acc+=d*v*scl; break;}
            { double lg = 0.5*x*x + d*(1.0-v+std::log(v)); if(uv<0 || lg>std::log(uv)){acc+=d*v*scl;break;} }
            acc+=d*v*scl; break; // only first iteration counted properly below
        }
        // redo loop properly
        t0=now_sec(); acc=0;
        for(long i=1;i<=N;i++){
            const double d=shp-1.0/3.0, c=1.0/std::sqrt(9.0*d);
            double x,v;
            for(;;){ x=norm(); v=1.0+c*x; if(v>0) break; }
            v=v*v*v; double uv=u01(rng);
            double out;
            if(uv < 1.0-0.0331*x*x*x*x) out=d*v*scl;
            else { if(0.5*x*x + d*(1.0-v+std::log(v)) > std::log(uv)) out=d*v*scl; else continue; }
            acc+=out;
        }
        t=now_sec()-t0; sink=acc; printf("hand MT gamma %6.2f ns\n", t*1e9/N);
        (void)marsaglia_tsang;
    }
    return 0;
}
