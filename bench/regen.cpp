// Full-pipeline regression: trajectory-level agreement between the new
// kernel path and an exact (libm) reference implementation of the same model.
#include "../motor_synergy.h"
#include <cstdio>
#include <cmath>
#include <random>
using namespace motor_synergy;

// Reference CDF/PDF using libm only — mirrors the OLD code path exactly.
struct ref_kernel { double mu, sigma; };
static double ref_cdf(const ref_kernel& k, double dt){
    if(dt<=0.0) return 0.0;
    double z=(std::log(dt)-k.mu)/k.sigma;
    return 0.5*(1.0+std::erf(z*0.70710678118654752440));
}
int main(){
    // Compare detail::lognormal_cdf vs libm across realistic (mu,sigma,t) grids
    double worst=0;
    for(double sigma : {0.04,0.08,0.12,0.18,0.23,0.30,0.45}){
        for(double peak : {20.0,80.0,300.0,900.0}){
            double mu=std::log(peak)+sigma*sigma;
            auto k=detail::make_lognormal_kernel(mu,sigma);
            ref_kernel rk{mu,sigma};
            for(double t=0.5;t<=2500.0;t*=1.01){
                double e=std::fabs(detail::lognormal_cdf(k,t)-ref_cdf(rk,t));
                if(e>worst) worst=e;
            }
        }
    }
    printf("max CDF deviation from libm over full param grid: %.3e\n", worst);
    printf("=> worst-case position error at D=2000px: %.4f px\n", worst*2000);

    // End-to-end metric stability: run stats again through this binary
    for(double I : {0.0,1.0,3.0,5.0}){
        double sum_e=0,sum_pe=0,sum_mt=0; int n=3000;
        for(int i=0;i<n;i++){
            auto p=generate(100,100,600,400,{},(uint64_t)(i*131+17),I);
            auto m=compute_metrics(p,600,400,20,std::hypot(500.0,300.0));
            sum_e+=m.endpoint_error; sum_pe+=m.path_efficiency; sum_mt+=m.movement_time;
        }
        printf("I=%.1f: avg err %.3f px, eff %.3f, MT %.1f ms\n", I, sum_e/n, sum_pe/n, sum_mt/n);
    }
    return 0;
}
