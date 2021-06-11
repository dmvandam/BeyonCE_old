# BeyonCE

This package is subdivided into 4 modules, which have to be tested and validated. This validation should occur in a separate module.

## Light Curve Simulation

This module covers all the light curve simulation and ring system generation with some basic analytical tools and plotting functionality. -- simulate_lightcurve.py --

### Simulation

The generation of theoretical light curves and how to make them "_noisy_".

 - [x] Simulate a signal-to-noise ratio 1 light curve (this is based off of a modified version of pyPplusS)
 - [x] Be able to add noise to this theoretical baseline (based on different distributions)
 - [x] Remove data from the above light curve (i.e. realistic data collection)
 - [x] It should be able to generate random ring systems

### Analysis

Very basic analysis of the light curve garnered from the light curve gradients.

 - [x] Calculate the slopes given boundaries
 - [x] Determine the minimum transverse velocity according to the Van Werkhoven et al., 2014 method
 - [x] Determine the minimum disk size

### Plotting

To visualise what is occuring it is important that there are plotting functions.

 - [x] Plot vector images of the ring system
 - [x] Plot the simulated light curve
 - [x] Plot a combination of both

### Demo

When the module is run a demo should run to show all the features of this module

 - [x] Demo features

## Parameter Explorer

This module is used to explore and cut down the large parameter space that circumplanetary disk transits incur. 
The most basic parameter set (which is considered in this module) is:

Planet Parameter
 - Planet radius (Rp)

Ring Geometry Parameters
 - Opacities (tau)
 - Inner Radii (Rin) 
 - Outer Radii (Rout) 

Disk Parameters
 - Inclination (i)
 - Tilt (phi) - angle w.r.t. orbital path
 - Transverse Velocity (vt)
 - Impact Parameter (b)

Stellar Parameters
 - Linear Limb-Darkening Parameter (u)

Time Alignment Parameters
 - Time Offset (dt)

This module includes the setting up of a grid based on the (x, y) position of the planet (centre of the ring system) and a stretch factor of the disk.
It includes the appropriate tools for filtering (or cutting away) the parameter space and plotting functionality.

### Ellipse Solving

This means that from an (dx, dy, fy) position in a grid we can determine the disk parameters

 - [x] Determine the semi-major axis
 - [x] Determine the inclination
 - [x] Determine the tilt
 - [x] Time offset and impact parameter are dx and dy respectively
 - [x] Determine the gradients at ellipse edges

### Grid Setup

Speedy analytical equations to quickly setup a 3d grid of variable resolution. Ensure maximum computational efficiency

 - [x] Setting up a grid
 - [x] Imposing reflection symmetry where warranted
 - [ ] Sub-grid methods

### Demo

When the module is run a demo should run to show all the features of this module.

 - [x] Demo features

### TODOs

 - [ ] move get_closest_solution to validation module
 - [ ] finish sub-grid parameter solving

## Ring Fitter 

This module is used to the light curve ring system fitting by subsequently dividing and then merging ringlets. 
This is a precursor to the MCMC fitter, which will free up any number of parameters.

### Fitters

This contains all the fitting routines.

 - [x] ring divider
 - [x] ring merger
 - [x] ringlet_fitter
 - [x] utility functions

### Write Files

This contains all the routines used to write the data to a file such that the results can be saved.

 - [x] write to file
 - [x] read from file
 - [x] helper functions

### Demo

When the module is run a demo should run to show all the features of this module.

 - [ ] Demo features

## MCMC

This module contains the mcmc fitting of a ring system solution to a light curve.
It takes as a starting point the solution of the Ring Fitter module.

### Ring System Model

We need a modified version, because of the way emcee takes parameters.
We also need to define the priors.

 - [x] Defining Ring System Model
 - [x] Setting up Priors
 - [ ] Defining prior bounds

### Likelihood Functions

These are the functions required to lead the walkers in the MCMC function.

 - [x] natural logarithm of likelihood
 - [x] natural logarithm of the lower bound

### Plotting Functions

These are the functions that produce all the relevant plots

 - [x] Triangle Plot
 - [x] Walkers Plot
 - [x] Models Plot
 - [x] Histogram (for priors) plot


### Analysis Functions

These are the functions that further the prime analysis of this module.

 - [x] extract solutions
 - [x] get statistics

### MCMC Functions

These are the functions pertaining with a whole mcmc analysis chain

 - [ ] Runs mcmc formulation

### Other Functions

This is a list of random functionality.

 - [x] Print Parameters

### Demo

When the 
### Demo

When the module is run a demo should run to show all the features of this module.

 - [ ] Demo features


## Validation

This is a whole separate module that is focused on running all the code neccessary to validate the above modules

 - [x] Simulate Light Curve?
 - [ ] Sjalot Explorer (note: find closest data point should be here
