input file 'in_orbit.txt'
scan steps: 		scan begins at lowest electric field value and increases by (max-min)/(scan steps) until the atom is ionized. max and min are computed based on n
highest value for n: 	code will run for highest n, iterating over field directions, output to std::out, then move to next lowest n
phi steps: 		angle of initial velocity vector is stepped between 90 and 180 degrees (in the x-y plane)
Esweep steps: 		angle of electric field is stepped from 0 to 180 degrees in \theta and from 0 to 360 degrees in \phi
max sim steps: 		a number higher than the actual sim steps needed
Bx: 			magnetic field in the x direction
ramp time: 		time for electric field to be increased from zero to current scan value
time step: 		initial time step, changed afterward by adaptive runge-kutta
time limit: 		time after which the atom is considered not ionized if the electron is still close to the nucleus
RK adaptErr: 		smaller value --> shorter time steps to satisfy adaptive runge-kutta

cuda file 'orbit.cu'
line 261, threads per block can be scaled up for more powerful gpu (4x4x4 for RTX 3050)
line 271, Emag can be modified depending on magnetic field. See comment there. Higher value increases (max-min) for electric field scan.
compile simply as >nvcc orbit.cu

executable 'a.out'
can be run if Emag is satisfactory
