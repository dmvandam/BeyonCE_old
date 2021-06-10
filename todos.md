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

This is module is used to explore and cut down the large parameter space that circumplanetary disk transits incur. 
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
