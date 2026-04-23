#include "NumMeth.h"

void rk4(double x[], int nX, double t, double tau, 
  void (*derivsRK)(double x[], double t, double param[], double deriv[]), 
  double param[]);

void rka( double x[], int nX, double& t, double& tau, double err,
  void (*derivsRK)(double x[], double t, double param[], double deriv[]), 
  double param[]) {
// Adaptive Runge-Kutta routine
// Inputs
//   x          Current value of the dependent variable
//   nX         Number of elements in dependent variable x
//   t          Independent variable (usually time)
//   tau        Step size (usually time step)
//   err        Desired fractional local truncation error
//   derivsRK   Right hand side of the ODE; derivsRK is the
//              name of the function which returns dx/dt
//              Calling format derivsRK(x,t,param).
//   param      Extra parameters passed to derivsRK
// Outputs
//   x          New value of the dependent variable
//   t          New value of the independent variable
//   tau        Suggested step size for next call to rka

  // Set initial variables
  double tSave = t;      // Save initial value
  double safe1 = 0.9, safe2 = 1.5;  // Safety factors
  double half_tau, errorRatio, eps, scale, xDiff, ratio, tau_old;

  // Loop over maximum number of attempts to satisfy error bound
  std::array<double, 6+1> xSmall, xBig; //std::vector would let me use nX instead of hard coding 6+1, but takes twice as long to run bc ::vector data goes to heap instead of stack
  int i, iTry, maxTry = 100;  
  for( iTry=1; iTry<=maxTry; iTry++ ) {	
   
    // Take the two small time steps
    half_tau = 0.5 * tau;
    for( i=1; i<=nX; i++ )
      xSmall[i] = x[i];
    rk4(xSmall.data(),nX,tSave,half_tau,derivsRK,param);
    t = tSave + half_tau;
    rk4(xSmall.data(),nX,t,half_tau,derivsRK,param);
  
    // Take the single big time step
    t = tSave + tau;
    for( i=1; i<=nX; i++ )
      xBig[i] = x[i];
    rk4(xBig.data(),nX,tSave,tau,derivsRK,param);
  
    // Compute the estimated truncation error
    errorRatio = 0.0; 
    eps = 1.0e-16;
    for( i=1; i<=nX; i++ ) {
      scale = err * (fabs(xSmall[i]) + fabs(xBig[i]))/2.0;
      xDiff = xSmall[i] - xBig[i];
      ratio = fabs(xDiff)/(scale + eps);
      errorRatio = ( errorRatio > ratio ) ? errorRatio:ratio;
    }
    
    // Estimate new tau value (including safety factors)
    tau_old = tau;
    tau = safe1*tau_old*pow(errorRatio, -0.20);
    tau = (tau > tau_old/safe2) ? tau:tau_old/safe2;
    tau = (tau < safe2*tau_old) ? tau:safe2*tau_old;
  
    // If error is acceptable, return computed values
    if (errorRatio < 1)  {
      for( i=1; i<=nX; i++ )
        x[i] = xSmall[i];
      return; 
    }
   
  }

  // Issue error message if error bound never satisfied
  std::cout << "ERROR: Adaptive Runge-Kutta routine failed" << std::endl;

}
