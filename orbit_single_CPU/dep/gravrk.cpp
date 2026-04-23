#include "NumMeth.h"

void gravrk(double x[], double t, double param[], double deriv[]) 
{
double normR = sqrt( x[1]*x[1] + x[2]*x[2] + x[3]*x[3] );
deriv[1] = x[4]; 
deriv[2] = x[5];
deriv[3] = x[6];
deriv[4] = -x[1]/(normR*normR*normR) - ( param[1] + x[5]*param[6] - x[6]*param[5] ) ;  // sum of E from p, imposed E, imposed B. -1 times E + vxB because e-
deriv[5] = -x[2]/(normR*normR*normR) - ( param[2] + x[6]*param[4] - x[4]*param[6] ) ;
deriv[6] = -x[3]/(normR*normR*normR) - ( param[3] + x[4]*param[5] - x[5]*param[4] ) ;
}

// x = x[1,2,3]
// v = x[4,5,6]
// E = param[1,2,3]
// B = param[4,5,6]
