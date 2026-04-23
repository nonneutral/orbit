// orbit - Program to compute the orbit of a comet.
#include "./dep/NumMeth.h"
#include "./dep/gravrk.cpp"
#include "./dep/rk4.cpp"
#include "./dep/rka.cpp"
#include <iostream>
#include <string>
#include <fstream>
#define TOOFAR 4

void gravrk( double x[], double t, double param[], double deriv[] );
void rk4( double x[], int nX, double t, double tau,
  void (*derivsRK)(double x[], double t, double param[], double deriv[]),
  double param[]);
void rka( double x[], int nX, double& t, double& tau, double err,
  void (*derivsRK)(double x[], double t, double param[], double deriv[]),
  double param[]);

int main() {

//read input file
double r0, v0, tau, tmax, adaptErr;
double Ex, Ey, Ez, Bx, By, Bz, tramp;
int nP, mz, nStep, method;
std::ifstream infile("in_orbit.txt");
std::string line, varName, defaultValue;
std::string delimiter = "=";
while (std::getline(infile, line)) 
{
  varName = line.substr(0, line.find(delimiter));
  defaultValue = line.substr(line.find(delimiter) + 1);
  if(varName == "principal quantum number n") 
  {
    nP = std::stoi(defaultValue);
    continue;
  } 
  else if(varName == "azimuthal fake quantum number mz") 
  {
    mz = std::stoi(defaultValue);
    continue;
  } 
  else if(varName == "max steps") 
  {
    nStep = std::stoi(defaultValue);
    continue;
  } 
  else if(varName == "Ex (V/m)") 
  {
    Ex = std::stod(defaultValue);
    continue;
  } 
  else if(varName == "Ey (V/m)") 
  {
    Ey = std::stod(defaultValue);
    continue;
  } 
  else if(varName == "Ez (V/m)") 
  {
    Ez = std::stod(defaultValue);
    continue;
  } 
  else if(varName == "Bx (T)") 
  {
    Bx = std::stod(defaultValue);
    continue;
  } 
  else if(varName == "By (T)") 
  {
    By = std::stod(defaultValue);
    continue;
  } 
  else if(varName == "Bz (T)") 
  {
    Bz = std::stod(defaultValue);
    continue;                              
  } 
  else if(varName == "ramp time (s)") 
  {
    tramp = std::stod(defaultValue);
    continue;
  } 
  else if(varName == "time step (s)") 
  {
    tau = std::stod(defaultValue);
    continue;
  } 
  else if(varName == "time limit (s)") 
  {
    tmax = std::stod(defaultValue);
    continue;
  } 
  else if(varName == "RK adaptErr") 
  {
    adaptErr = std::stod(defaultValue);
    continue;
  } 
  else if(varName == "method") 
  {
    method = std::stoi(defaultValue);
  }
}

// Set physical parameters
const double pi = 3.141592654;
const double a0 = 5.292e-11;				// bohr radius					m
const double tref = 2.419e-17;				// recip. of angular freq of e- in H		s
const double vref = 2.188e6;  				// fine structure * speed of light (or a0/tref)	m/s
const double qom = 1e12*1.602/9.109;  			// charge to mass ratio	for e-			C/kg
const double Eref = 5.142e11;				// electric field at bohr radius		V/m
const double Bref = 2.351e5;				// magnetic field at a0 from a bohr magneton?	T
const double Uref = 27.211;				// two rydbergs 				eV
const double Kref = vref * vref / qom;  		// mult by 1/2 v^2 to get KE			eV

//convert to atomic units
double maxparam[6+1]; 
maxparam[1] = Ex/Eref; maxparam[2] = Ey/Eref; maxparam[3] = Ez/Eref;
maxparam[4] = Bx/Bref; maxparam[5] = By/Bref; maxparam[6] = Bz/Bref;
tau = tau/tref;
tmax = tmax/tref;
tramp = tramp/tref;
r0 = nP*nP;  						// e^2/r0 = 1/n^2
v0 = 1.0*mz/r0;  					// mvr = mz hbar. m and hbar = 1
double rSI, vSI, potSI, kinSI, eSI, bSI, dtSI, tmaxSI;
rSI = r0*a0;
vSI = v0*vref;
potSI = 1.602e-19 * 1.602e-19 / 4 / pi / 8.85e-12 / r0 / a0;
kinSI = 0.5 * v0 * v0 * Kref * 1.602e-19;

// debug: print input values in both unit systems
//std::cout << "In SI units, \n r0 = " << rSI << "\n v0 = " << vSI << "\n potential energy = " << potSI << "\n kinetic energy = " << kinSI << "\n tmax = " << (tmax*tref) << "\n dt = " << (tau*tref) << std::endl;
//std::cout << "In atomic units, \n r0 = " << r0 << "\n v0 = " << v0 << "\n potential energy = " << (1.0/r0) << "\n kinetic energy = " << (v0*v0) << "\n tmax = " << tmax << "\n dt = " << tau << std::endl;
//std::cout << "nstep = " << nStep << std::endl;

// Set initial conditions
double time = 0;
double r[3+1], v[3+1], state[6+1], param[6+1];
r[1] = r0; r[2] = 0; r[3] = 0;  v[1] = 0; v[2] = v0; v[3] = 0;
int nState = 6;
int nParam = 6;
int i;
for(i=1; i<4; i++)
{
  state[i] = r[i];
  state[3+i] = v[i];
}
int iParam;
for(iParam=1; iParam<=nParam/2; iParam++)
 param[ nParam/2 + iParam ] = maxparam[ nParam/2 + iParam ];

// Plotting variables
double *xplot, *yplot, *zplot, *tplot, *kinetic, *potential;
xplot = new double [nStep+1];  yplot = new double [nStep+1];  zplot = new double [nStep+1];
tplot = new double [nStep+1];
kinetic = new double [nStep+1]; potential = new double [nStep+1];
double normR, normV;

// Loop over desired number of steps using specified numerical method.
int iStep;
for( iStep=1; iStep<=nStep; iStep++ ) 
{
  if( time > tmax )	
    break ;								// atom survived until tmax
  if( sqrt( r[1]*r[1] + r[2]*r[2] + r[3]*r[3] ) > (TOOFAR * r0) )	
    break;								// electron left orbit

  if( time < tramp )
    for(iParam=1; iParam<=nParam/2; iParam++)
      param[iParam] = maxparam[iParam] * time / tramp ;

  // Record position and energy for plotting.
  normR = sqrt( r[1]*r[1] + r[2]*r[2] + r[3]*r[3] );
  normV = sqrt( v[1]*v[1] + v[2]*v[2] + v[3]*v[3] );
  xplot[iStep] = r[1];               			// Record position for plotting
  yplot[iStep] = r[2];
  zplot[iStep] = r[3];
  tplot[iStep] = time;
  kinetic[iStep] = 0.5*normV*normV * Kref;   		// Record energies in eV
  potential[iStep] = - 1.0/normR * Uref;		// because e^2 / a0 = 2 Rydbergs = Uref

// debug: print current value of variables
//  std::cout << "normR = " << normR << std::endl;
//  std::cout << "normV = " << normV << std::endl;
//  std::cout << "kin = " << kinetic[iStep] << std::endl;
//  std::cout << "pot = " << potential[iStep] << std::endl;

  // Calculate next step
  if( method == 3 ) 
  {
    rk4( state, nState, time, tau, gravrk, param );
    for(i=1; i<4; i++)
    {
      r[i] = state[i];   				// 4th order Runge-Kutta
      v[i] = state[3+i];
    }
    time += tau;
  }
  else 
  {
    rka( state, nState, time, tau, adaptErr, gravrk, param );
    for(i=1; i<4; i++)
    {
      r[i] = state[i];   				// Adaptive Runge-Kutta
      v[i] = state[3+i];
    }
  }
}

// Print out the plotting variables: thplot, rplot, potential, kinetic
std::string cwd = "outfiles/";
std::ofstream xplotOut(cwd + "xplot.txt"), yplotOut(cwd + "yplot.txt"), zplotOut(cwd + "zplot.txt"), rplotOut(cwd + "rplot.txt"),
  tplotOut(cwd + "tplot.txt"), potentialOut(cwd + "potential.txt"), kineticOut(cwd + "kinetic.txt");
for( i=1; i<iStep; i++ ) 
{
  xplotOut << xplot[i] << std::endl;
  yplotOut << yplot[i] << std::endl;
  zplotOut << zplot[i] << std::endl;
  rplotOut << sqrt(xplot[i]*xplot[i]+yplot[i]*yplot[i]) << std::endl;
  tplotOut << tplot[i]*tref << std::endl;
  potentialOut << potential[i] << std::endl;
  kineticOut << kinetic[i] << std::endl;
}

delete [] xplot, yplot, zplot, tplot, kinetic, potential;
return 0;
}
