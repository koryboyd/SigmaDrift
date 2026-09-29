#include "../motor_synergy.h"
#include <cstdio>
#include <cmath>
using namespace motor_synergy::detail;
int main(){
    double sigma=0.23, peak_t=300.0;
    double mu=std::log(peak_t)+sigma*sigma;
    auto k=make_lognormal_kernel(mu,sigma);
    for(double t : {1.0, 5.0, 50.0, 150.0, 290.0, 300.0, 400.0, 700.0, 1200.0}){
        double z=(std::log(t)-mu)/sigma;
        double ref=std::exp(-0.5*z*z)/(t*sigma*std::sqrt(2*M_PI));
        double got=lognormal_pdf(k,t);
        printf("t=%7.1f z=%8.3f phi=%11.3e ref=%12.6e got=%12.6e rel=%.3e\n",
               t,z,std::exp(-0.5*z*z),ref,got,std::fabs(got-ref)/ref);
    }
}
