// Find ionization threshold for hydrogen 
//  with variable principal quantum number, in variable magnetic field
//  averaging over orbital eccentricity and relative angle of electric field
// rka and rk4 functions based on Garcia, Numerical Methods for Physics
#include <cmath>
#include <string>
#include <array>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cuda_runtime.h>

// Physics function
// x = x[1,2,3]
// v = x[4,5,6]
// E = param[1,2,3]
// B = param[4,5,6]
__device__ 
void gravrk( double x[], double param[], double deriv[] )
{
double denom = rsqrt( x[1]*x[1] + x[2]*x[2] + x[3]*x[3] );
denom = denom*denom*denom;
deriv[1] = x[4]; 
deriv[2] = x[5];
deriv[3] = x[6];
deriv[4] = -x[1]*denom - ( param[1] + x[5]*param[6] - x[6]*param[5] ) ;  // sum of E from p, imposed E, imposed B. -1 times E + vxB because e-
deriv[5] = -x[2]*denom - ( param[2] + x[6]*param[4] - x[4]*param[6] ) ;
deriv[6] = -x[3]*denom - ( param[3] + x[4]*param[5] - x[5]*param[4] ) ;
}

// Runge-Kutta integrator (4th order)
__device__ 
void rk4( double x[], double tau, double param[] )
{
  double F1[7], F2[7], F3[7], F4[7], xtemp[7] ;
  double half_tau = 0.5*tau;
  int nX = 6 ;

  // Evaluate F1 = f(x,t)
  gravrk( x, param, F1 ) ;  

  // Evaluate F2 = f( x+tau*F1/2, t+tau/2 )

  for(int i=1; i<=nX; i++ )
    xtemp[i] = x[i] + half_tau*F1[i];
  gravrk( xtemp, param, F2 );  

  // Evaluate F3 = f( x+tau*F2/2, t+tau/2 )
  for(int i=1; i<=nX; i++ )
    xtemp[i] = x[i] + half_tau*F2[i];
  gravrk( xtemp, param, F3 );

  // Evaluate F4 = f( x+tau*F3, t+tau )
  for(int i=1; i<=nX; i++ )
    xtemp[i] = x[i] + tau*F3[i];
  gravrk( xtemp, param, F4 );

  // Return x(t+tau) computed from fourth-order R-K.
  for(int i=1; i<=nX; i++ )
    x[i] += tau/6.0*(F1[i] + F4[i] + 2.0*(F2[i]+F3[i]));
}
 
// Adaptive Runge-Kutta
__device__ 
void rka( double x[], double& t, double& tau, double err, double param[] )
{
  // Set initial variables
  double tSave = t;      // Save initial value
  double safe1 = 0.9, safe2 = 1.5;  // Safety factors
  double half_tau, errorRatio, eps, scale, xDiff, ratio, tau_old;

  // Loop over maximum number of attempts to satisfy error bound
  double xSmall[6+1], xBig[6+1];
  int i, iTry, maxTry = 100;  
  int nX = 6 ;
  for( iTry=1; iTry<=maxTry; iTry++ ) {	
   
    // Take the two small time steps
    half_tau = 0.5 * tau;
    for( i=1; i<=nX; i++ )
      xSmall[i] = x[i];
    rk4(xSmall,half_tau,param);
    t = tSave + half_tau;
    rk4(xSmall,half_tau,param);
  
    // Take the single big time step
    t = tSave + tau;
    for( i=1; i<=nX; i++ )
      xBig[i] = x[i];
    rk4(xBig,tau,param);
  
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
}

// cuda kernal
//  the commands in this function are run independently (in series) on each parallel thread
__global__ 
void rka_kernel(int nphi, int nE, int nScanE, int nStep, double tramp, double tmax, double tau, double  adaptErr, double r0, double v0, double Emag, double Bx, double By, double Bz, double *d_thresh)
{
 double time;
 double state[7]; 
 double phi, Ephi, Etheta, Estart, Estop, Erange, scanfrac, rampfrac;
 double param[7];
 double maxparam[7];
 maxparam[4] = Bx; 
 maxparam[5] = By; 
 maxparam[6] = Bz;
 int iphi = 1 + blockIdx.x * blockDim.x + threadIdx.x;
 int iEth = 1 + blockIdx.y * blockDim.y + threadIdx.y;
 int iEph = 1 + blockIdx.z * blockDim.z + threadIdx.z;

 // Scan eccentricity (using phi) and scan electric field direction in spherical polar coordinates
 if( iphi <= nphi && iEth <= nE && iEph <= nE )
 {
  phi = ( 3.14159 / 2.0 ) * ( (double)iphi / (double)nphi ) ;		// looks like all four quadrants (0 to pi/2, pi/2 to pi, pi to 3pi/2, 3pi/2 to 2pi) same, but should double check
  Etheta = (double)iEth / (double)nE * 3.14159 ;
  Ephi = (double)iEph / (double)nE * 2.0 * 3.14159 ;
  maxparam[1] = Emag * sin(Etheta) * cos(Ephi) ;
  maxparam[2] = Emag * sin(Etheta) * sin(Ephi) ;
  maxparam[3] = Emag * cos(Etheta) ;
  Estart = Emag ;								// Estart will be updated (within each phi loop) based on last scan
  Estop = Emag * 1.0 / (double)nScanE ;
  Erange = Estart - Estop ;
 	  
  // Scan to find ionization threshold
  while( ( Erange / Estop ) > 0.01 )
  {
   time = 0;
   for(int iScan=1; iScan<=nScanE; iScan++ )
   {
    if( ( time * 1.05 ) > tmax )						// threshold found
     break;	
 
    // Initial conditions
    time = 0;
    scanfrac = ( Estart - Erange * (double)iScan / (double)nScanE ) / Emag ;	// start at high E to reduce compute
    state[1] = r0; 		state[2] = 0; 			state[3] = 0;  	// initial position
    state[4] = v0 * cos(phi); 	state[5] = v0 * sin(phi); 	state[6] = 0;	// initial velocity
 
    // Simulate
    for(int iStep=1; iStep<=nStep; iStep++ ) 
    {    
     // Skip all remaining updates if atom survived until tmax or if electron left orbit (r > 10*r0)
     if( (double)time > tmax || ( state[1]*state[1] + state[2]*state[2] + state[3]*state[3] ) > (100 * r0 * r0) )
      break;								
 
     // Ramp E up to nominal over many orbital periods (whereas B is constant)
     if( (double)time < tramp )
     {
      rampfrac = (double)time / tramp ;
      for(int iParam=1; iParam<=3; iParam++)
      { 
       param[iParam] 	= maxparam[iParam] 	* rampfrac * scanfrac ;
       param[iParam+3] 	= maxparam[iParam+3] 	;
      }
     }
 
     // Calculate next step
     rka( state, time, tau, adaptErr, param );  //time += tau; (debug)
    }
   }
     
     // Update E limits for next E scan
     Estop = Emag * scanfrac ;
     Estart = Estop + Erange * ( 3.0 / (double)nScanE ) ;			// go 3 samples past the previous threshold
     Erange = Estart - Estop ; 
  }
  // save result
  int thid = (iphi-1) + (iEth-1) * nphi + (iEph-1) * nphi * nE ;
  d_thresh[thid] = Estop ;
 }
}

// the main program
//  1 standard C++ stuff (on the CPU)
//  2 manage cuda memory, including arrays containing all parallel variables (maybe there's a more efficient way to do this?)
//  3 send the call to the __global__ function, telling the GPU how many threads and blocks to use
int main() {

auto start_time = std::chrono::high_resolution_clock::now();

//read input file
double r0, v0, tau, tmax, adaptErr;
double Emag, Bx, By, Bz, tramp;
int nScanE, nHi, nLo, nE, nphi, nStep;
std::ifstream infile("in_orbit.txt");
std::string line, varName, defaultValue;
std::string delimiter = "=";
while (std::getline(infile, line)) 
{
  varName = line.substr(0, line.find(delimiter));
  defaultValue = line.substr(line.find(delimiter) + 1);
       if(varName == "scan steps for E")  		{ nScanE	= std::stoi(defaultValue);	continue;  } 
  else if(varName == "highest value for n")	  	{ nHi		= std::stoi(defaultValue);	continue;  } 
  else if(varName == "lowest value for n")	  	{ nLo		= std::stoi(defaultValue);	continue;  } 
  else if(varName == "phi steps")  			{ nphi 		= std::stoi(defaultValue);	continue;  } 
  else if(varName == "Esweep steps")  			{ nE 		= std::stoi(defaultValue);	continue;  } 
  else if(varName == "max sim steps")   		{ nStep 	= std::stod(defaultValue);	continue;  } 
  else if(varName == "Bx (T)")  			{ Bx 		= std::stod(defaultValue);	continue;  } 
  else if(varName == "By (T)")   			{ By 		= std::stod(defaultValue);	continue;  } 
  else if(varName == "Bz (T)")   			{ Bz 		= std::stod(defaultValue);	continue;  } 
  else if(varName == "ramp time (s)")   		{ tramp 	= std::stod(defaultValue);	continue;  } 
  else if(varName == "time step (s)")   		{ tau 		= std::stod(defaultValue);	continue;  } 
  else if(varName == "time limit (s)")   		{ tmax 		= std::stod(defaultValue);	continue;  } 
  else if(varName == "RK adaptErr")   			{ adaptErr 	= std::stod(defaultValue);  } 
}

// Set physical parameters
constexpr double tref = 2.419e-17;			// recip. of angular freq of e- in H		s
constexpr double Eref = 5.142e11;			// electric field at bohr radius		V/m
constexpr double Bref = 2.351e5;			// magnetic field at a0 from a bohr magneton?	T

//convert to atomic units
double maxparam[6+1]; 
maxparam[4] = Bx / Bref ; 
maxparam[5] = By / Bref ; 
maxparam[6] = Bz / Bref ;
tau = tau / tref ;
tmax = tmax / tref ;
tramp = tramp / tref ;

// total number of thresholds to be found
int nThreads = nE * nE * nphi ;

// vector to store results
std::vector<double> thresh(nThreads, 0.0); 
std::vector<double> zeros(nThreads, 0.0); 

// Allocate memory on device
double *d_thresh;
cudaDeviceSynchronize();
cudaMalloc((void**)&d_thresh, sizeof(double) * nThreads);
cudaMemcpy(d_thresh, zeros.data(), sizeof(double) * nThreads, cudaMemcpyHostToDevice);   
 
// Choose number of blocks and threads per block
dim3 threadsPerBlock(4,4,4);
dim3 numBlocks((nphi + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (nE + threadsPerBlock.y - 1) / threadsPerBlock.y,
               (nE + threadsPerBlock.z - 1) / threadsPerBlock.z);  

// Scan principal quantum number n
// Initial guess for max E is 1.5 times max expected n, from "Ionization of hydrogen atoms by static and circularly polarized fields: Classical adiabatic theory"
for(int n=nHi; n>=nLo; n-=1)
{
 // Update variables that depend on n
 Emag = 1.5 / 2.6 / n / n / n / n ;			// higher value for higher B. 1.5 for B=0 (fastest). '2.6' is from Rakovic 1998, DOI 10.1088/0953-4075/31/9/014
 r0 = n * n ; 						// e^2/r0 = 1/n^2
 v0 = 1.0 / sqrt(r0) ;  				// 1/2 m v^2 = 1/2 e^2 / 4 pi eps0 r0	1/2 on r.h.s. is virial theorem for 1/r potential

 // SIMULATE
 // Scan Etheta from 0 to pi, Ephi from 0 to 2 pi, and phi from 0 to pi/2
 rka_kernel<<<numBlocks, threadsPerBlock>>>(nphi, nE, nScanE, nStep, tramp, tmax, tau, adaptErr, r0, v0, Emag, maxparam[4], maxparam[5], maxparam[6], d_thresh);
        
 // Check for errors in kernel launch
 cudaDeviceSynchronize();
 cudaError_t err2 = cudaGetLastError();
  if (err2 != cudaSuccess) {  std::cout << "CUDA error: " << cudaGetErrorString(err2) << std::endl; }

 // Copy results back to host
 cudaMemcpy(thresh.data(), d_thresh, sizeof(double) * nThreads, cudaMemcpyDeviceToHost);      

 // Compute sums
 std::sort(thresh.begin(), thresh.end());
 std::cout << n << "," << thresh[0]*Eref;
 for(int q=1; q<=100; q++)
  std::cout << "," << thresh[((q-0.5)/100.0)*nThreads]*Eref;
 std::cout << "," << thresh[nThreads-1]*Eref << std::endl;

 // Reset array for next iteration
 cudaMemset(d_thresh, 0.0, sizeof(double) * nThreads);

// cudaFree(d_thresh);
// cudaMalloc((void**)&d_thresh, sizeof(double) * nThreads);
// cudaMemcpy(d_thresh, zeros.data(), sizeof(double) * nThreads, cudaMemcpyHostToDevice); 
}

 // Free device memory
 cudaFree(d_thresh);

// Print elapsed time
auto end_time = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;

return 0;
}
