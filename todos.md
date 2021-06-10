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

This is module is used to explore and cut down the large parameter space that circumplanetary disk transits incur. The most basic parameter set (which is considered in this module) is the size of the planet (rp), the ring geometry (opacity [tau], inner [rin] and outer [rout] radii), the disk inclination (i) and tilt (phi, angle w.r.t. orbital path), the impact parameter (b) and transverse velocity (vt) of the ring system, the linear limb-darkening of the star (u) and finally a time offset (dt) to align the simulated light curve with the data.
