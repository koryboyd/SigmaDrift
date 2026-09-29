// Statistical validation harness for motor_synergy.h (no libm hacks).
#include "../motor_synergy.h"
#include <cstdio>
#include <random>
#include <chrono>
#include <numeric>
using namespace motor_synergy;

int main(){
    // ---- 1. stationary variance of the OU kernel under Gamma jitter -------
    {
        std::mt19937_64 rng(42);
        std::gamma_distribution<double> gam(3.5, 7.8/3.5);
        const double theta=3.5, sigma=1.2;
        const double var_theory = sigma*sigma/(2*theta);
        const long steps=4000000;
        double ou=0, sum2=0;
        for(long i=0;i<steps;i++){
            double dt=gam(rng); dt=dt<2?2:(dt>25?25:dt);
            dt*=0.001;
            double decay=std::exp(-theta*dt);
            double vol=sigma*std::sqrt((1-decay*decay)/(2*theta));
            ou = ou*decay + vol*std::normal_distribution<double>(0,1)(rng);
            sum2 += ou*ou;
        }
        printf("OU stationary var: measured %.6f theory %.6f (err %.2f%%)\n",
               sum2/steps, var_theory, 100*(sum2/steps/var_theory-1));
    }

    // ---- 2. lognormal CDF saturation: how many frames sit at s==1 exactly? -
    {
        config cfg; double frac_sat=0; int trials=2000; size_t tot=0, sat=0;
        for(int i=0;i<trials;i++){
            auto p=generate(100,100,900,300,cfg,(uint64_t)i+7);
            double mt_est = p.back().t;
            for(auto&q:p){ tot++; if(q.t> /*beyond MT*/0) {} }
            // count points after ~1.0*MT using primary sigma known? simpler: measure
            // fraction of tail frames where consecutive positions are identical in bell part.
        }
        (void)frac_sat;(void)cfg;(void)trials;
    }

    // ---- 3. endpoint error vs intensity ------------------------------------
    for(double I : {0.0,0.35,1.0,1.75,3.0,5.0}){
        double sum_e=0,max_e=0,sum_mt=0,n=0,sum_pe=0;
        int trials=3000;
        for(int i=0;i<trials;i++){
            auto p=generate(100,100,600,400,{},(uint64_t)(i*131+17),I);
            auto m=compute_metrics(p,600,400,20,std::hypot(500.0,300.0));
            sum_e+=m.endpoint_error; max_e=std::max(max_e,m.endpoint_error);
            sum_mt+=m.movement_time; sum_pe+=m.path_efficiency; n++;
        }
        printf("I=%.2f: avg err %.3f px, max %.3f px, avg MT %.1f ms, eff %.3f\n",
               I, sum_e/n, max_e, sum_mt/n, sum_pe/n);
    }

    // ---- 4. timing scaling with distance & strength ------------------------
    {
        config cfg;
        for(double I:{0.0,1.0,5.0}){
            double t=0; int iters=5000;
            auto t0=std::chrono::steady_clock::now();
            for(int i=0;i<iters;i++){ auto p=generate(0,0,1200,800,cfg,(uint64_t)i+1,I); t+=p.size();}
            auto t1=std::chrono::steady_clock::now();
            printf("I=%.1f long-dist: %.2f us/traj (%.0f pts)\n", I,
                std::chrono::duration<double,std::micro>(t1-t0).count()/iters, t/iters);
        }
    }
    return 0;
}
