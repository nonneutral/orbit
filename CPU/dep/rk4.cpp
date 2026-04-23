#include "NumMeth.h"

void rk4(double x[], int nX, double t, double tau, 
  void (*derivsRK)(double x[], double t, double param[], double deriv[]), 
  double param[]) {
// Runge-Kutta integrator (4th order)
// Inputs
//   x          Current value of dependent variable
//   nX         Number of elements in dependent variable x
//   t          Independent variable (usually time)
//   tau        Step size (usually time step)
//   derivsRK   Right hand side of the ODE; derivsRK is the
//              name of the function which returns dx/dt
//              Calling format derivsRK(x,t,param,dxdt).
//   param      Extra parameters passed to derivsRK
// Output
//   x          New value of x after a step of size tau

  std::array<double, 6+1> F1, F2, F3, F4, xtemp;  //std::vector would let me use nX instead of hard coding 6+1, but takes twice as long to run bc ::vector data goes to heap instead of stack

  // Evaluate F1 = f(x,t).
  (*derivsRK)( x, t, param, F1.data() );  

  // Evaluate F2 = f( x+tau*F1/2, t+tau/2 ).
  double half_tau = 0.5*tau;
  double t_half = t + half_tau;
  int i;
  for( i=1; i<=nX; i++ )
    xtemp[i] = x[i] + half_tau*F1[i];
  (*derivsRK)( xtemp.data(), t_half, param, F2.data() );  

  // Evaluate F3 = f( x+tau*F2/2, t+tau/2 ).
  for( i=1; i<=nX; i++ )
    xtemp[i] = x[i] + half_tau*F2[i];
  (*derivsRK)( xtemp.data(), t_half, param, F3.data() );

  // Evaluate F4 = f( x+tau*F3, t+tau ).
  double t_full = t + tau;
  for( i=1; i<=nX; i++ )
    xtemp[i] = x[i] + tau*F3[i];
  (*derivsRK)( xtemp.data(), t_full, param, F4.data() );

  // Return x(t+tau) computed from fourth-order R-K.
  for( i=1; i<=nX; i++ )
    x[i] += tau/6.0*(F1[i] + F4[i] + 2.0*(F2[i]+F3[i]));

}
