// Validates the new fast_log / phi_poly / lognormal kernels against libm.
#include "../motor_synergy.h"
#include <cstdio>
#include <cmath>
using namespace motor_synergy::detail;

int main(){
    // ---- fast_log accuracy over the realistic dt range [0.5, 1500] -------
    double worst_rel=0, wx=0;
    for(double x=0.5;x<=1500.0;x+= (x<8?1e-4: (x<200?0.005:0.25))){
        double rel = std::fabs(fast_log(x)-std::log(x))/std::fabs(std::log(x)+1e-300);
        double abs_= std::fabs(fast_log(x)-std::log(x));
        if(abs_>worst_rel){worst_rel=abs_;wx=x;}
    }
    printf("fast_log max ABS err on [0.5,1500]: %.3e at x=%.4f\n", worst_rel, wx);

    // ---- phi_poly vs exact Phi (via std::erf) -----------------------------
    double worst=0, wz=0;
    for(double z=-9;z<=9;z+=1e-5){
        double exact=0.5*(1.0+std::erf(z*0.70710678118654752440));
        double e=std::fabs(phi_poly(z)-exact);
        if(e>worst){worst=e;wz=z;}
    }
    printf("phi_poly max abs err [-9,9]: %.3e at z=%.4f\n", worst, wz);

    // tail behaviour sanity: monotone, in [0,1]
    double prev=-1; bool mono=true;
    for(double z=-9;z<=9;z+=1e-4){ double v=phi_poly(z); if(v<prev-1e-12) mono=false; prev=v;
        if(v<-1e-9||v>1+1e-9){printf("OUT OF RANGE z=%f v=%f\n",z,v);mono=false;break;} }
    printf("phi_poly monotone & bounded: %s\n", mono?"yes":"NO");

    // ---- lognormal CDF/PDF vs reference (libm) over a typical trajectory --
    {
        double sigma=0.23, peak_t=300.0;
        double mu=std::log(peak_t)+sigma*sigma;
        auto k=make_lognormal_kernel(mu,sigma);
        double cdf_worst=0,pdf_worst=0, t_at=0;
        for(double t=1;t<=1200;t+=0.01){
            double z=(std::log(t)-mu)/sigma;
            double ref_cdf=0.5*(1.0+std::erf(z*0.70710678118654752440));
            double ref_pdf=std::exp(-0.5*z*z)/(t*sigma*std::sqrt(2*M_PI));
            double e1=std::fabs(lognormal_cdf(k,t)-ref_cdf);
            double e2=std::fabs(lognormal_pdf(k,t)-ref_pdf)/ref_pdf;
            if(e1>cdf_worst){cdf_worst=e1;t_at=t;}
            if(e2>pdf_worst)pdf_worst=e2;
        }
        printf("lognormal CDF max abs err: %.3e (px at D=2000: %.4f) at t=%.1f\n", cdf_worst, cdf_worst*2000, t_at);
        printf("lognormal PDF max rel err: %.3e\n", pdf_worst);
    }
    return 0;
}
