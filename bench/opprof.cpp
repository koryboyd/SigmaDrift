// Operation-level profiler for the current motor_synergy.h hot loop, plus
// accuracy checks on candidate replacements before they are adopted.
#include "../motor_synergy.h"
#include <chrono>
#include <cstdio>
#include <random>

using namespace motor_synergy;
using clk = std::chrono::steady_clock;

static double now_sec(){ return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

static volatile double sink;
#define N 20000000L

int main(){
    double t0,t,acc;
    // ---- per-primitive costs (ns) --------------------------------------
    t0=now_sec(); acc=0;
    for(long i=1;i<=N;i++) acc+=std::log(1.0+(double)i/N);
    t=now_sec()-t0; sink=acc; printf("std::log            %7.2f ns\n", t*1e9/N);

    t0=now_sec(); acc=0;
    for(long i=1;i<=N;i++) acc+=std::exp(-(double)i/N);
    t=now_sec()-t0; sink=acc; printf("std::exp            %7.2f ns\n", t*1e9/N);

    t0=now_sec(); acc=0;
    for(long i=1;i<=N;i++) acc+=detail::fast_exp(-(double)i/N);
    t=now_sec()-t0; sink=acc; printf("fast_exp            %7.2f ns\n", t*1e9/N);

    t0=now_sec(); acc=0;
    for(long i=1;i<=N;i++) acc+=std::sqrt((double)i/N);
    t=now_sec()-t0; sink=acc; printf("std::sqrt           %7.2f ns\n", t*1e9/N);

    t0=now_sec(); acc=0;
    for(long i=1;i<=N;i++) acc+=(double)i/(N+1.5);
    t=now_sec()-t0; sink=acc; printf("fp divide           %7.2f ns\n", t*1e9/N);

    t0=now_sec(); acc=0; double s,c;
    for(long i=1;i<=N;i++){ detail::fast_sincos((double)(i%1000)/1000.0,s,c); acc+=s+c; }
    t=now_sec()-t0; sink=acc; printf("fast_sincos pair    %7.2f ns\n", t*1e9/N);

    t0=now_sec(); acc=0;
    for(long i=1;i<=N;i++){ acc+=std::sin((double)(i%1000)/1000.0)+std::cos((double)(i%1000)/1000.0); }
    t=now_sec()-t0; sink=acc; printf("libm sin+cos        %7.2f ns\n", t*1e9/N);

    // log(dt) in CDF/PDF with dt ~ 2..640ms
    t0=now_sec(); acc=0;
    for(long i=1;i<=N;i++) acc+=std::log(2.0+(double)(i%640));
    t=now_sec()-t0; sink=acc; printf("std::log(dt)        %7.2f ns\n", t*1e9/N);

    // mt19937_64 raw + gamma draw
    {
        std::mt19937_64 rng(12345);
        std::gamma_distribution<double> gam(3.5, 7.8/3.5);
        t0=now_sec(); acc=0;
        for(long i=1;i<=N;i++) acc+=gam(rng);
        t=now_sec()-t0; sink=acc; printf("std::gamma draw     %7.2f ns\n", t*1e9/N);

        auto nrm = detail::make_normal_adapter(rng);
        long m=N/2; // adapter caches half the draws
        t0=now_sec(); acc=0;
        for(long i=1;i<=m;i++) acc+=nrm();
        t=now_sec()-t0; sink=acc; printf("normal adapter      %7.2f ns\n", t*1e9/m);

        std::normal_distribution<double> nd(0,1);
        t0=now_sec(); acc=0;
        for(long i=1;i<=m;i++) acc+=nd(rng);
        t=now_sec()-t0; sink=acc; printf("std::normal dist    %7.2f ns\n", t*1e9/m);
    }

    // ---- accuracy of fast_exp vs std::exp over OU range -----------------
    double worst=0, wx=0;
    for(double x=-60;x<=0;x+=1e-6){
        double e=std::fabs(detail::fast_exp(x)-std::exp(x))/std::exp(x);
        if(e>worst){worst=e;wx=x;}
    }
    printf("\nfast_exp max rel err on [-60,0]: %.3e at x=%.4f\n", worst, wx);

    // ---- accuracy of fast_erf vs std::erf over CDF-reachable args -------
    worst=0; wx=0;
    for(double x=-6;x<=6;x+=1e-6){
        double e=std::fabs(detail::fast_erf(x)-std::erf(x));
        if(e>worst){worst=e;wx=x;}
    }
    printf("fast_erf max abs err on [-6,6]: %.3e at x=%.4f\n", worst, wx);

    // ---- whole-trajectory benchmark -------------------------------------
    {
        config cfg;
        int iters=20000;
        double dx=0,dy=0;
        t0=now_sec();
        for(int i=0;i<iters;i++){
            auto p = generate(100,100,600,400,cfg,(uint64_t)i+1);
            dx += p.size(); dy += p.back().x;
        }
        t=now_sec()-t0; sink=dx+dy;
        printf("\ngenerate(): %.2f us/traj (%zu pts avg)\n", t*1e6/iters, (size_t)(dx/iters));
    }
    return 0;
}
