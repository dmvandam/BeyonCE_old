'''
This module is used to create and fit a ring model for a given light curve
based on fixed and known disk parameters (disk radius, impact parameter, 
inclination, tilt, dt, transverse velocity) and the linear limb-darkening 
parameter, u, of the star by generating a disk model and subsequently diving
that disk into equivalent ringlets, which are subsequently fused to form
rings.
'''


###############################################################################
############################# IMPORT MAIN MODULES #############################
###############################################################################

# calculations
import numpy as np
from time import time as time_now
from collections.abc import Iterable
from scipy.optimize import curve_fit
from simulate_lightcurve_v2 import add_noise, simulate_lightcurve


###############################################################################
############################## UTILITY FUNCTIONS ##############################
###############################################################################

def divide_rings(max_radius, known_opacities, num_rings):
    '''
    This functions divides a disk into a set number of rings by equally spacing
    the rings.

    Parameters
    ----------
    max_radius : float
        Size of the largest ring (i.e. the disk) [R*].
    known_opacities : array_like (1-D)
        List of opacities for rings (note that this does not need to be the 
        same length as the number of rings, but it needs to be an integer
        multiple thereof... i.e. you can list 4 opacities and have 8 rings,
        but not 6 rings.
    num_rings : int
        Number of rings to separate the disk into.

    Returns
    -------
    inner_radii : array_like (1-D)
        Inner dimension of the rings [R*].
    outer_radii : array_like (1-D)
        Outer dimension of the rings [R*].
    opacities : array_like (1-D)
        Opacities of each ring [-].
    '''
    # calculate the ring edges
    ring_edges = np.linspace(0, 1, num_rings + 1) * max_radius
    inner_radii = ring_edges[:-1]
    outer_radii = ring_edges[1:]
    # pyPplusS must have a non-zero inner ring edge
    inner_radii[0] = 1e-16
    # determine how the known opacities will be distributed about the opacities
    fill_factor = num_rings // len(known_opacities)
    opacities = np.repeat(known_opacities, fill_factor, 0)
    return inner_radii, outer_radii, opacities

def merge_ringlets(opacities, merge_fraction=0.05):
    '''
    This function merges ringlets that have similar opacities to form thicker
    rings.
    
    Parameters
    ----------
    opacities : array_like (1-D)
        Opacities of each ringlet [-].
    merge_fraction : float
        Allowed standard deviation of the opacities.
        
    Returns
    -------
    merged_opacities : array of floats
        Contains the actual opacities of each ring [-].
    '''
    # intialise lists for new rings
    merged_opacities = -np.ones(len(opacities))
    # setup for-loop
    ringlet_opacities = []
    ringlet_std = 0
    for k in range(len(opacities)):
        # is the opacity within delta_tau of the mean_tau then extend ring_tau
        ringlet_opacities.append(opacities[k])
        ringlet_std = np.std(ringlet_opacities)
        # does the ringlet std exceed the merge fraction then record a ring
        if ringlet_std > merge_fraction:
            # append rings
            ind = np.argmin(merged_opacities)
            merged_opacities[ind:k] = np.mean(ringlet_opacities[:-1])
            # reset ringlet parameters
            ringlet_opacities = [opacities[k]]
            ringlet_std = 0
    # add the final ring parameters
    ind = np.argmin(merged_opacities)
    merged_opacities[ind:] = np.mean(ringlet_opacities)
    return merged_opacities

def compact_ringlets(inner_radii, outer_radii, merged_opacities):
    '''
    This function compacts the ringlets to rings for optimisation of the
    opacities.

    Parameters
    ----------
    inner_radii : array_like (1-D)
        Contains the inner ring edges for each ringlet [R*].
    outer_radii : array_like (1-D)
        Contains the outer ring edges for each ringlet [R*].
    merged_opacities : array_like (1-D)
        Contains the merged opacities of the ringlets [-].

    Returns
    -------
    compact_inner_radii : array of floats
        Compacted inner radii for rings (not ringlets) [R*].
    compact_outer_radii : array of floats
        Compacted outer radii for rings (not ringlets) [R*].
    compact_opacities : array of floats
        Compacted opacities for rings (not ringlets) [-].
    '''
    # extract the unique values, sorting and reconstruction indices
    unique_data = np.unique(merged_opacities, return_index=True,
                            return_inverse=True)
    # mask to get unique opacities
    select_unique = np.sort(unique_data[1])
    # unpack map for expand ringlets
    unpack_map = unique_data[2]
    # compact parameters
    compact_ring_edges = np.append(inner_radii[select_unique], outer_radii[-1])
    compact_inner_radii = compact_ring_edges[:-1]
    compact_outer_radii = compact_ring_edges[1:]
    compact_opacities = merged_opacities[select_unique]
    return (compact_inner_radii, compact_outer_radii, compact_opacities,
            unpack_map)

def expand_ringlets(compact_opacities, unpack_indices):
    '''
    This function uses the unpack_indices to expand the compacted ringlets
    back to their original size (useful for saving into an array).
    
    Parameters
    ----------
    compact_opacities : array_like (1-D)
        Compacted opacities for rings (not ringlets) [-].
    unpack_indices : array_like (1-D)
        Index array to expand compacted inner/outer radii and corresponding
        opacities.

    Returns
    -------
    expanded_opacities : array_like (1-D)
        Contains the merged opacities of the ringlets [-].
    '''
    # expand opacities using the unpack indices
    expanded_opacities = np.sort(compact_opacities)[unpack_indices]
    return expanded_opacities

def merge_rings(inner_radii, outer_radii, opacities, merge_fraction=0.05,
                extend_first=True):
    '''
    This function merges ringlets that have similar opacities to form thicker
    rings.
    
    Parameters
    ----------
    inner_radii : array_like (1-D)
        Contains the inner ring edges for each ring [R*].
    outer_radii : array_like (1-D)
        Contains the outer ring edges for each ring [R*].
    opacities : array_like (1-D)
        Opacities of each ring [-].
    merge_fraction : float
        Allowed standard deviation of the opacities.
    extend_first : bool
        If true then the first ring to transit the star is extended to the
        planet. This is for two reasons: the first is that we could never
        learn about rings before this one (as they do not transit the star)
        and the second is that this could reduce the effect of a numerical
        error during optimisation that causes a wiggle in the light curve.
        If false then no ring extension takes place.
        
    Returns
    -------
    new_inner_radii : array_like (1-D)
        Contains the actual inner ring edges for each ring [R*].
    new_outer_radii : array_like (1-D)
        Contains the actual outer ring edges for each ring [R*].
    new_opacities : array_like (1-D)
        Contains the actual opacities of each ring [-].
    '''
    # intialise lists for new rings
    new_opacities = []
    new_inner_radii = [inner_radii[0]]
    new_outer_radii = []
    # setup for-loop
    ringlet_opacities = []
    ringlet_std = 0
    for k in range(len(opacities)):
        # is the opacity within delta_tau of the mean_tau then extend ring_tau
        ringlet_opacities.append(opacities[k])
        ringlet_std = np.std(ringlet_opacities)
        if ringlet_std > merge_fraction:
            # append rings
            new_opacities.append(np.mean(ringlet_opacities[:-1]))
            new_outer_radii.append(inner_radii[k])
            new_inner_radii.append(inner_radii[k])
            # reset ringlet parameters
            ringlet_opacities = [opacities[k]]
            ringlet_std = 0
    # add the final ring parameters
    new_opacities.append(np.mean(ringlet_opacities))
    new_outer_radii.append(outer_radii[-1])
    # convert to arrays
    new_inner_radii = np.array(new_inner_radii)
    new_outer_radii = np.array(new_outer_radii)
    new_opacities = np.array(new_opacities)
    # extend the innermost ring if it has an opacity of 0
    if (extend_first == True) and (new_opacities[0] == 0):
        new_inner_radii = np.delete(new_inner_radii, 1)
        new_outer_radii = np.delete(new_outer_radii, 0)
        new_opacities   = np.delete(new_opacities, 0)
    return new_inner_radii, new_outer_radii, new_opacities

def optimise_opacities(time, lightcurve, planet_radius, inner_radii,
                       outer_radii, opacities, inclination, tilt,
                       impact_parameter, dt, limb_darkening,
                       transverse_velocity, extend_first=True):
    '''
    Parameters
    ----------
    time : array_like (1-D)
        Time data points at which to the light curve is calculated [day].
    lightcurve : array_like (1-D)
        Normalised flux data points at which the light curve is calculated 
        [L*].
    planet_radius : float
        Size of the planet [R*].
    inner_radii : array_like (1-D)
        Inner dimension of the rings [R*].
    outer_radii : array_like (1-D)
        Outer dimension of the rings [R*].
    opacities : array_like (1-D)
        Opacities of each ring [-].
    inclination : float
        Inclination of the ring system [deg].
    tilt : float
        Tilt of the rings, is the angle between the x-axis and the semi-major
        axis of the projected ellipse [deg].
    impact_parameter : float
        Impact parameter between the centre of the rings w.r.t. centre of the
        star [R*].
    dt : float
        This is a delta time parameter that can be used to shift the light 
        curve left or right [day].
    limb_darkening : float
        Limb-darkening parameter, u,  of the star according to the linear law,
        I(mu)/I(1) = 1 - u * (1 - mu), where mu = cos(y), where y is the angle
        between the line-of-sight and the emergent intensity [-].
    transverse_velocity : float
        The transiting velocity of the ring system across the star [R*/day].
    extend_first : bool
        If true then the first ring to transit the star is extended to the
        planet. This is for two reasons: the first is that we could never
        learn about rings before this one (as they do not transit the star)
        and the second is that this could reduce the effect of a numerical
        error during optimisation that causes a wiggle in the light curve.
        If false then no ring extension takes place.

    Returns
    -------
    optimal_opacities : array_like (1-D)
        Optimised values of the opacity of each ring [-].
    chi2 : float
        chi2 value of the best fit model to the light curve with the given
        ring radii parameters.
    '''
    # create optimal_opacities array
    optimal_opacities = np.zeros(len(inner_radii))
    # determine which rings must be optimised
    tilt_rad = np.deg2rad(tilt)
    mask = np.abs(impact_parameter) - np.abs(outer_radii * np.sin(tilt_rad))
             < 1
    # mask rings
    true_inner_radii = inner_radii[mask]
    true_outer_radii = outer_radii[mask]
    true_opacities   = opacities[mask]
    print('optimise_opacities() is optimising %i ringlets' 
            % len(true_opacities))
    # extend first ring to centre of the planet
    if extend_first == True:
        true_inner_radii[0] = 1e-16
    # determine the lightcurve simulation function
    def fixed_disk(time, *opacities):
        '''
        This function freezes all disk parameters except the relevant
        opacities and sets the equation up such that curve_fit can determine
        the optimal opacities.

        Parameters
        ----------
        time : array_like (1-D)
            Time points at which to calculate the light curve [day].
        *taus : floats (arbitrary number)
            Arbitrary number of taus, should be the same as the number
            of rings [-].
        
        Returns
        -------
        lightcurve : array_like (1-D)
            Simulated theoretical light curve (normalised flux) based on the
            inputs [L*].
        '''
        # make sure that the opacities are recorded in a list
        taus = []
        for opacity in opacities:
            taus.append(opacity)
        # group the parameters
        params = (planet_radius, true_inner_radii, true_outer_radii, taus,
                  inclination, tilt, impact_parameter, dt, limb_darkening,
                  transverse_velocity)
        # calculate the light curve
        lightcurve, _ = simulate_lightcurve(time, *params)
        return lightcurve
    # optimise opacities
    P0, _ = curve_fit(lambda time, *TAUS: fixed_disk(time, *TAUS), time,
                      lightcurve, p0=true_opacities, bounds=(0, 1))
    # determine chi2
    model_lightcurve = fixed_disk(time, *P0)
    chi2 = np.sum((lightcurve - model_lightcurve)**2 / model_lightcurve)
    # fill arrays
    optimal_opacities[-len(P0):] = P0
    # extend opacity to the centre
    if extend_first == True:
        optimal_opacities[:-len(P0)] = P0[0]
    return optimal_opacities, chi2

def determine_num_ringlets(multiplier, max_ringlets=100):
    '''
    This function determines the number of ringlets the disk will divided into
    and how for the ring_divide_fitter function.

    Parameters
    ----------
    multiplier : int or array_like (1-D)
        Contains either a integer which will be repeatedly be multiplied until
        the maximum number of rings has been reached, or an array of values
        with which to multiply the number of ringlets.
    max_ringlets : int
        The maximum number of ringlets allowed [default = 100].

    Returns
    -------
    num_ringlets : array_like (1-D)
        Contains the subsequent number of ringlets that the disk should be
        divided into.
    '''
    num_ringlets = [1]
    # integer -> keep going until you exceed max_ring_number
    if type(multiplier) == int:
        while num_ringlets[-1] < max_ringlets:
            num_ringlets.append(num_ringlets[-1] * multiplier)
    # iterable -> go through each value until end or max_ring_number
    else:
        for m in multiplier:
            if num_ringlets[-1] < max_ringlets:
                num_ringlets.append(num_ringlets[-1] * m)
    # convert to array
    num_ringlets = np.array(num_ringlets)
    # ensure that maximum has not been exceeded
    num_ringlets = num_ringlets[num_ringlets < max_ringlets]
    return num_ringlets


###############################################################################
############################## FITTER FUNCTIONS ###############################
###############################################################################

def division_fitter(time, lightcurve, planet_radius, disk_radius,
                    initial_opacities, inclination, tilt, impact_parameter,
                    dt, limb_darkening, transverse_velocity, num_ringlets,
                    max_num_ringlets, extend_first=True):
    '''
    This function fits the input light curve by splitting a disk with the 
    given input parameters into various ringlets. The opacities of the 
    ringlets are then optimised to fit the lightcurve. Watch out for over-
    fitting.
    
    Parameters
    ----------
    time : array_like (1-D)
        Time data points at which to the light curve is calculated [day].
    lightcurve : array_like (1-D)
        Normalised flux data points at which the light curve is calculated 
        [L*].
    planet_radius : float
        Size of the planet [R*].
    disk_radius : array_like (1-D)
        Maximum disk radii of investigated ellipses [R*].
    initial_opacities : float or array_like (1-D)
        Initial opacity estimate of the single disk structure (use as a first
        guess the approximate depth of the largest ring feature in the light
        curve) [-].
    inclination : array_like (1-D)
        Array of inclinations of the ring system [deg].
    tilt : array_like (1-D)
        Array of tilts of the ring system, this is the angle between the 
        x-axis and the semi-major axis of the projected ellipse [deg].
    impact_parameter : array_like (1-D)
        Array of the impact parameters between the centre of the rings w.r.t.
        centre of the star [R*].
    dt : array_like (1-D)
        Array of delta time parameters that can be used to shift the light 
        curve left or right in time space [day].
    limb_darkening : float
        Limb-darkening parameter, u,  of the star according to the linear law,
        I(mu)/I(1) = 1 - u * (1 - mu), where mu = cos(y), where y is the angle
        between the line-of-sight and the emergent intensity [-].
    transverse_velocity : float
        The transiting velocity of the ring system across the star [R*/day].
    num_ringlets : int
        Number of ringlets the disk has been subdivided into.
    max_num_ringlets : int
        The maximum number of ringlets the disk will be subdivided into.
    extend_first : bool
        If true then the first ring to transit the star is extended to the
        planet. This is for two reasons, the first is that we could never
        learn about rings before this one (as they do not transit the star)
        and the second is that this could reduce the effect of a numerical
        error during optimisation that causes a wiggle in the light curve.
        If false then no ring extension takes place.

    Returns
    -------
    inner_radii : array_like (1-D)
        Contains the values of the inner radii of the rings for each solution
        with a set number of rings [R*].
    outer_radii : array_like (1-D)
        Contains the values of the outer radii of the rings for each solution
        with a set number of rings [R*].
    opacities : array_like (1-D)
        Contains the values of the opacities of each of the rings for each 
        solution with a set number of rings [-].
    chi2 : array_like (1-D)
        Contains the chi2 value of each model for each solution with a set
        number of rings.
    opacities_0 : array_like (1-D)
        This contains the optimal opacity values of the ringlets assuming that
        the ringlets have the widths they should have (e.g. 2 ringlets have
        the thickness of half the disk radius each and there are 2 opacities
        instead of 32 ringlets where the first 16 have the same first opacity 
        and the second 16 have the same second opacity) [-]. This is used for
        the next subdivision of the disk.
    '''
    print('dividing into %i ringlets' % num_ringlets)
    # ensure intial opacities are iterable 
    if not isinstance(initial_opacities, Iterable):
        initial_opacities = np.array([initial_opacities])
    # divide the disk into rings with the same opacities
    divide_data = divide_rings(disk_radius, initial_opacities, num_ringlets)
    compact_inner_radii, compact_outer_radii, extended_opacities = divide_data
    # group parameters for optimise_opacities
    params = (time, lightcurve, planet_radius, compact_inner_radii,
              compact_outer_radii, extended_opacities, inclination, tilt,
              impact_parameter, dt, limb_darkening, transverse_velocity,
              extend_first)
    # optimise the opacities and measure computation time
    start_time = time_now()
    optimal_opacities, chi2 = optimise_opacities(*params)
    end_time = time_now()
    print('  optimisation done in %.2f s' % (end_time - start_time))
    # define inner and outer radii
    ring_edges = np.linspace(0, 1, max_num_ringlets + 1) * disk_radius
    inner_radii = ring_edges[:-1]
    outer_radii = ring_edges[1:]
    # fill out opacities to update array
    fill_ratio = max_num_ringlets / num_ringlets
    opacities = np.repeat(optimal_opacities, fill_ratio, 0)
    # get opacities_0 (which is just optimal_opacities renamed for clarity)
    opacities_0 = optimal_opacities
    return inner_radii, outer_radii, opacities, chi2, opacities_0

def merge_fitter(time, lightcurve, planet_radius, inner_radii, outer_radii,
                 opacities, inclination, tilt, impact_parameter, dt,
                 limb_darkening, transverse_velocity, merge_fraction,
                 unpack_map_in, optimised_chi2_in):
    '''
    This function takes the input ring system model that has been divided into
    ringlets, merges the ringslet via the merge_fraction parameter and then 
    optimises the opacities.

    Parameters
    ----------
    time : array_like (1-D)
        Time data points at which to the light curve is calculated [day].
    lightcurve : array_like (1-D)
        Normalised flux data points at which the light curve is calculated 
        [L*].
    planet_radius : float
        Size of the planet [R*].
    inner_radii : array_like (1-D) 
        Contains the values of the inner radii of the rings for each solution
        with a set number of rings [R*].
    outer_radii : array_like (1-D)
        Contains the values of the outer radii of the rings for each solution
        with a set number of rings [R*].
    opacities : array_like (1-D)
        Contains the values of the opacities of each of the rings for each 
        solution with a set number of rings [-].
    inclination : array_like (1-D)
        Array of inclinations of the ring system [deg].
    tilt : array_like (1-D)
        Array of tilts of the ring system, this is the angle between the 
        x-axis and the semi-major axis of the projected ellipse [deg].
    impact_parameter : array_like (1-D)
        Array of the impact parameters between the centre of the rings w.r.t.
        centre of the star [R*].
    dt : array_like (1-D)
        Array of delta time parameters that can be used to shift the light 
        curve left or right in time space [day].
    limb_darkening : float
        Limb-darkening parameter, u, of the star according to the linear law,
        I(mu)/I(1) = 1 - u * (1 - mu), where mu = cos(y), where y is the angle
        between the line-of-sight and the emergent intensity [-].
    transverse_velocity : float
        The transiting velocity of the ring system across the star [R*/day].
    merge_fraction : float
        Allowed standard deviation of the opacities.
    unpack_map_in : array_like (1-D)
        The unpack map of the inner/outer radii and opacities. This is used to
        determine whether the ring merging has had any effect and whether or
        not this should be repeated.
    optimised_chi2_in : float
        The input optimised chi2 value, this is useful to see if the model is
        getting better or not.

    Returns
    -------
    optimised_opacities : array_like (1-D)
        Contains the values of the opacities of each of the rings for each 
        solution with a set number of rings [-].
    optimised_chi2 : array_like (1-D)
        Contains the chi2 value of each model for each solution with a set
        number of rings.
    '''
    # determine start time
    start_time = time_now()
    # merge ringlets
    merged_opacities = merge_ringlets(opacities, merge_fraction)
    # merge to rings
    compacted = compact_ringlets(inner_radii, outer_radii, merged_opacities)
    compact_rin, compact_rout, compact_taus, unpack_map = compacted
    print('  merged to %i rings' % len(compact_rin))
    if np.all(unpack_map_in == unpack_map):
        print('  no optimisation required')
        return opacities, optimised_chi2_in, unpack_map_in
    # optimise opacities
    optimise_inputs = (time, lightcurve, planet_radius, compact_rin,
                       compact_rout, compact_taus, inclination, tilt,
                       impact_parameter, dt, limb_darkening,
                       transverse_velocity)
    # run optimisation and measure computation time
    start_time = time_now()
    optimised_taus, optimised_chi2 = optimise_opacities(*optimise_inputs)
    end_time = time_now()
    print('  optimisation done in %.2f s' % (end_time - start_time))
    # expand to ringlets
    optimised_opacities = expand_ringlets(optimised_taus, unpack_map)
    return optimised_opacities, optimised_chi2, unpack_map

def ringlet_fitter(time, lightcurve, planet_radius, disk_radius,
                   initial_opacities, inclination, tilt, impact_parameter,
                   dt, limb_darkening, transverse_velocity, num_ringlets,
                   max_num_ringlets, merge_fraction, num_merges=1,
                   extend_first=True):
    '''
    This tries to fit a ring system by fitting a disk that is subsequently
    divided, optimised, merged, optimised, and potentially is recursively
    merged and optimised until the ring bounds no longer change or after a
    certain number of loops (whichever comes first).
    
    Parameters
    ----------
    time : array_like (1-D)
        Time data points at which to the light curve is calculated [day].
    lightcurve : array_like (1-D)
        Normalised flux data points at which the light curve is calculated
        [L*].
    planet_radius : float
        Size of the planet [R*].
    disk_radius : array_like (1-D)
        Maximum disk radii of investigated ellipses [R*].
    initial_opacities : float or array_like (1-D)
        Initial opacity estimate of the single disk structure (use as a first
        guess the approximate depth of the largest ring feature in the light
        curve) [-].
    inclination : array_like (1-D)
        Array of inclinations of the ring system [deg].
    tilt : array_like (1-D)
        Array of tilts of the ring system, this is the angle between the 
        x-axis and the semi-major axis of the projected ellipse [deg].
    impact_parameter : array_like (1-D)
        Array of the impact parameters between the centre of the rings w.r.t.
        centre of the star [R*].
    dt : array_like (1-D)
        Array of delta time parameters that can be used to shift the light 
        curve left or right in time space [day].
    limb_darkening : float
        Limb-darkening parameter, u, of the star according to the linear law,
        I(mu)/I(1) = 1 - u * (1 - mu), where mu = cos(y), where y is the angle
        between the line-of-sight and the emergent intensity [-].
    transverse_velocity : float
        The transiting velocity of the ring system across the star [R*/day].
    num_ringlets : int
        Number of ringlets the disk has been subdivided into.
    max_num_ringlets : int
        The maximum number of ringlets the disk will be subdivided into.
    merge_fraction : float
        Allowed standard deviation of the opacities.
    num_merges : int
        Mumber of times that a ringlet model should going through the merging
        process (note that no extra computational time at convergence of ring
        boundaries).
    extend_first : bool
        If true then the first ring to transit the star is extended to the
        planet. This is for two reasons, the first is that we could never
        learn about rings before this one (as they do not transit the star)
        and the second is that this could reduce the effect of a numerical
        error during optimisation that causes a wiggle in the light curve.
        If false then no ring extension takes place.

    Returns
    -------
    inner_radii : array_like (1-D)   
        Contains the values of the inner radii of the ringlets for each 
        solution with a set number of rings [R*].
    outer_radii : array_like (1-D)
        Contains the values of the outer radii of the ringlets for each 
        solution with a set number of rings [R*].
    model_opacities : array_like (1-D)
        Contains the values of the opacities of each of the rings for each 
        solution with a set number of rings and each of the merge_fitter
        passes [-].
    model_chi2 : array_like (1-D)
        Contains the chi2 value for each model iteration (just division and
        then num_merges merge).
    opacities_0 : array_like (1-D)
        This contains the optimal opacity values of the ringlets assuming that
        the ringlets have the widths they should have (e.g. 2 ringlets have
        the thickness of half the disk radius each and there are 2 opacities
        instead of 32 ringlets where the first 16 have the same first opacity 
        and the second 16 have the same second opacity) [-]. This is used for
        the next subdivision of the disk.
    '''
    # set up array to hold model opacities and chi2
    model_opacities = np.zeros((num_merges + 1, max_num_ringlets))
    model_chi2 = np.zeros(num_merges + 1)
    # divide the disk into the right set of ringlets and optimise the opacity
    # for each ringlet, the outputs are the inner radii/outer radii bounds
    division_data = division_fitter(time, lightcurve, planet_radius,
                                    disk_radius, initial_opacities,
                                    inclination, tilt, impact_parameter,
                                    dt, limb_darkening, transverse_velocity,
                                    num_ringlets, max_num_ringlets,
                                    extend_first)
    inner_radii, outer_radii, opacities, chi2, opacities_0 = division_data
    # set first element of model array (i.e. unmerged)
    model_opacities[0] = opacities
    model_chi2[0] = chi2
    # set-up merge ring recursion
    unpack_map_in = np.array([-1])
    optimised_chi2_in = 0
    # loop over the number of iterations, note that merge_fitter reduces to a
    # variable pass method when the ring boundaries don't change effectively
    # perpetuating the last optimised results over the rest of the array
    for x in range(num_merges):
        merge_data = merge_fitter(time, lightcurve, planet_radius, inner_radii,
                                  outer_radii, model_opacities[x], inclination,
                                  tilt, impact_parameter, dt, limb_darkening,
                                  transverse_velocity, merge_fraction,
                                  unpack_map_in, optimised_chi2_in)
        optimised_opacities, optimised_chi2, unpack_map = merge_data
        # set model opacities and chi2
        model_opacities[x+1] = optimised_opacities
        model_chi2[x+1] = optimised_chi2
        # change the condition parameters
        unpack_map_in = unpack_map
        optimised_chi2_in = optimised_chi2
    return inner_radii, outer_radii, model_opacities, model_chi2, opacities_0

def ring_fitter(noise_fraction, num_ringlet_array, time, lightcurve,
                planet_radius, disk_radius, initial_opacities, inclination,
                tilt, impact_parameter, dt, limb_darkening, transverse_velocity,
                merge_fraction, num_merges, extend_first, filename, title='',
                known_rings=False, known_params=None):
    '''
    This function fits a ring system to a light curve by dividing an input
    disk into successively larger number of ringlets fitting the opacities
    and then merging the ringlets into rings based on a merging factor. It
    further writes the results to a file that is also an output of the file
    
    Parameters 
    ----------
    noise_fraction : float
        The standard deviation of the gaussian distribution added to the
        lightcurve for noise purposes.
    num_ringlet_array : array_like (1-D)
        Contains the integer values into which the disk should be divided
        note that the num_ringlet_array[x+1] must be an integer multiple 
        of num_ringlet_array[x] and that the number of ringlets should be
        ascending.
    time : array_like (1-D)
        Time data points at which to the light curve is calculated [day].
    lightcurve : array_like (1-D)
        Normalised flux data points at which the light curve is calculated 
        [L*].
    planet_radius : float
        Size of the planet [R*].
    disk_radius : array_like (1-D)
        Maximum disk radii of investigated ellipses [R*].
    initial_opacities : float or array_like (1-D)
        Initial opacity estimate of the single disk structure (use as a first
        guess the approximate depth of the largest ring feature in the light
        curve) [-].
    inclination : array_like (1-D)
        Array of inclinations of the ring system [deg].
    tilt : array_like (1-D)
        Array of tilts of the ring system, this is the angle between the 
        x-axis and the semi-major axis of the projected ellipse [deg].
    impact_parameter : array_like (1-D)
        Array of the impact parameters between the centre of the rings w.r.t. 
        centre of the star [R*].
    dt : array_like (1-D)
        Array of delta time parameters that can be used to shift the light 
        curve left or right in time space [day].
    limb_darkening : float
        Limb-darkening parameter, u, of the star according to the linear law,
        I(mu)/I(1) = 1 - u * (1 - mu), where mu = cos(y), where y is the angle
        between the line-of-sight and the emergent intensity [-].
    transverse_velocity : float
        The transiting velocity of the ring system across the star [R*/day].
    merge_fraction : float
        Allowed standard deviation of the opacities.
    num_merges : int
        Number of times that a ringlet model should going through the merging
        process (note that no extra computational time at convergence of ring
        boundaries).
    extend_first : bool
        If true then the first ring to transit the star is extended to the
        planet. This is for two reasons, the first is that we could never
        learn about rings before this one (as they do not transit the star)
        and the second is that this could reduce the effect of a numerical
        error during optimisation that causes a wiggle in the light curve.
        If false then no ring extension takes place
    filename : str
        Name of the file where all the ring_fitter data will be saved.
    title : str
        Title of the file to be saved [default = ''].
    known_rings : bool
        If you know the actual ring system parameters set to True to include
        data in the header of the savefile.
    known_params : tuple of array_like (1-D)
        This is a tuple containing the inner radii of the rings, the outer
        radii of the rings and the opacities of the rings (only used if
        known_rings == True).

    Returns
    -------
    inner_radii : array_like (1-D)
        Contains the values of the inner radii of the ringlets for each 
        solution with a set number of rings [R*].
    outer_radii : array of floats
        Contains the values of the outer radii of the ringlets for each 
        solution with a set number of rings [R*].
    write_data : array of floats
        Contains all the data grouped in one array. the columns are the random
        seed (used to identify the trial), the noise_fraction, the number of 
        ringlets, the opacity of ringlet 1...max, chi2 value of the model,
        with the rows being for the different merge case scenarios (user 
        determined).
    '''
    # determine the maximum number of ringlets
    max_ringlets = num_ringlet_array[-1]
    # setup data file
    if known_rings == False:
        known_params = ([[-1]], [[-1]], [[-1]])
    write_header(time, planet_radius, *known_params, inclination, tilt,
                 impact_parameter, dt, limb_darkening, transverse_velocity,
                 merge_fraction, max_ringlets, title, filename)
    # prepare lightcurve
    seed = np.random.randint(0, 1000000)
    noisy_lightcurve = add_noise(lightcurve, np.random.normal,
                                 (0, noise_fraction), seed)
    # set up loop
    for num_ringlets in num_ringlet_array:
        # run ringlet fitter for num_ringlet input ringlets
        data = ringlet_fitter(time, noisy_lightcurve, planet_radius,
                              disk_radius, initial_opacities, inclination,
                              tilt, impact_parameter, dt, limb_darkening,
                              transverse_velocity, num_ringlets, max_ringlets,
                              merge_fraction, num_merges, extend_first)
        # extract data and refresh intial_opacities
        inner_radii, outer_radii, opacities, chi2, initial_opacities = data
        # prepare write data
        write_data = prepare_write_data(seed, noise_fraction, num_ringlets,
                                        opacities, chi2)
        # save data
        with open(filename, 'a') as f:
            for write_row in write_data:
                str_row = arr_to_str(write_row)
                print(str_row, file=f, flush=True)
    return inner_radii, outer_radii, write_data


###############################################################################
######################### MANIPULATE FILE FUNCTIONS ###########################
###############################################################################

def write_header(time, planet_radius, inner_radii, outer_radii, opacities,
                 inclination, tilt, impact_parameter, dt, limb_darkening,
                 transverse_velocity, merge_fraction, max_ringlets, title,
                 filename):
    '''
    This function writes the header to a text file that will contain all the
    simulation data provided by the ring fitter. It contains the actual model
    data (i.e. parameters for the clean lightcurve) and additionally includes
    the merge factor for the rings in the model and the number of transiting
    or visible rings.

    Parameters
    ----------
    time : array_like (1-D)
        Time data points at which to the light curve is calculated [day].
    planet_radius : float
        Size of the planet [R*].
    inner_radii : array_like (1-D)
        Contains the values of the inner radii of the rings for the model
        with a set number of rings [R*].
    outer_radii : array_like (1-D)
        Contains the values of the outer radii of the rings for the model
        with a set number of rings [R*].
    opacities : array_like (1-D)
        Contains the values of the opacities of each of the rings for the 
        model with a set number of rings [-].
    inclination : float
        Inclination of the ring system [deg].
    tilt : float
        Tilt of the ring system, this is the angle between the x-axis and the
        semi-major axis of the projected ellipse [deg].
    impact_parameter : float
        Impact parameters between the centre of the ring system w.r.t. the
        centre of the star [R*].
    dt : float
        Delta time parameter that can be used to shift the lightcurve left or 
        right in time space [day].
    limb_darkening : float
        Limb-darkening parameter, u, of the star according to the linear law,
        I(mu)/I(1) = 1 - u * (1 - mu), where mu = cos(y), where y is the angle
        between the line-of-sight and the emergent intensity [-].
    transverse_velocity : float
        The transiting velocity of the ring system across the star [R*/day].
    merge_fraction : float
        Allowed standard deviation of the opacities.
    max_ringlets : int
        Maximum number of ringlets used in the file.
    title : str
        String to append to the top of the file.
    filename : str
        Name of the file where all the ring_fitter data will be saved.
    
    Returns
    -------
    None
    '''
    # create file file and populate header
    with open(filename, 'w') as f:
        print('%s created' % filename)
        # print a header title
        print('#### %s ####' % title, file=f)
        # print ring system model parameters
        print('# time = %s' % arr_to_str(time), file=f)
        print('# planet radius [R*] = %.8f' % planet_radius, file=f)
        print('# inner radii [R*] = %s' % arr_to_str(inner_radii), file=f)
        print('# outer radii [R*] = %s' % arr_to_str(outer_radii), file=f)
        print('# opacities [-] = %s' % arr_to_str(opacities), file=f)
        print('# inclination [deg] = %.8f' % inclination, file=f)
        print('# tilt [deg] = %.8f' % tilt, file=f)
        print('# impact parameter [R*] = %.8f' % impact_parameter, file=f)
        print('# dt [day] = %.8f' % dt, file=f)
        print('# limb darkening [-] = %.8f' % limb_darkening, file=f)
        print('# transverse velocity [R*/day] = %.8f' % transverse_velocity,
              file=f)
        print('# merge fraction [-] = %.8f' % merge_fraction, file=f)
        # calculate the number of rings / visible rings
        num_rings = len(inner_radii)
        ring_heights = np.abs(outer_radii * np.sin(np.deg2rad(tilt)))
        transiting = np.abs(impact_parameter) - ring_heights < 1
        num_visible = np.sum(transiting)
        print('# num rings [-] = %i' % num_rings, file=f)
        print('# num visible rings [-] = %i' % num_visible, file=f)
        # write out column header
        col_headers = '# seed, noise_fraction, num_ringlets, num_merges'
        for i in range(max_ringlets):
            col_headers = col_headers + ', tau_%i' % i
        col_headers = col_headers + ', chi2'
        print(col_headers, file=f)
        print('%s header written' % filename)
    return None

def prepare_write_data(seed, noise_fraction, num_ringlets, opacities, chi2):
    '''
    This function prepares the data array from a single trial of the ring
    fitter routine.

    Parameters
    ----------
    seed : int
        The random seed used to add noise to the light curve.
    noise_fraction : float
        The standard deviation of the gaussian distribution added to the
        lightcurve for noise purposes.
    num_ringlets : int
        The number of ringlets that the disk has been divided into for the 
        fit.
    opacities : array_like (1-D)
        Opacities of the ringlets in an (m x n) array where n is the number of 
        ringlets and m is the number of merge runs + 1 (for the unmerged run)
        [m >= 1].
    chi2 : array_like (1-D)
        Contains the chi2 values for each of the model passes (i.e. array of
        length m).

    Returns
    -------
    write_data : array_like (1-D)
        Contains all the data grouped in one array. The columns are the random
        seed (used to identify the trial), the noise_fraction, the number of 
        ringlets, number of merges, the opacity of ringlet 1...max, chi2 value 
        of the model, with the rows being for the different merge case 
        scenarios (user determined).
    '''
    # extending constant parameters
    num_models = len(chi2)
    seeds = seed * np.ones(num_models)
    noise_fractions = noise_fraction * np.ones(num_models)
    nums_ringlets = num_ringlets * np.ones(num_models)
    # creating num_merges column
    num_merges = np.arange(num_models)
    # concatenating to single array
    write_data = np.hstack((seeds[:, None], noise_fractions[:, None],
                            nums_ringlets[:, None], num_merges[:, None],
                            opacities, chi2[:, None]))
    return write_data

def str_to_arr(array_string, delim=','):
    '''
    This function converts a string array converted using arr_to_str() back
    into an array.

    Parameters
    ----------
    array_string : str
        String that has been specially formated by the arr_to_str() function
        that will be converted back into an array.
    delim : str
        The delimiter that separates the array element values [default = ','].
    
    Returns
    -------
    array : array_like (1-D)
        Contains the float data contained by the string.
    '''
    # split the string elements
    str_elements = array_string.split(delim)
    # create the array
    array = np.zeros(len(str_elements))
    # populate the array
    for i, element in enumerate(str_elements):
        array[i] = float(element)
    return array

def arr_to_str(array, fmt='%.8f', delim=','):
    '''
    This function converts an array to a string_array that can be converted
    back into an array using the str_to_arr() function.

    Parameters
    ----------
    array : array_like (1-D)
        Contains float data that should be converted to a string array.
    fmt : str
        formatting string for each individual element of the array 
        [default = '%.8f'].
    delim : str
        str to delimit the element values of the array [default = ','].

    Returns
    -------
    str_array : str
        String version of the input array.
    '''
    # set up str_array
    str_array = ''
    # loop over array elements
    for element in array:
        str_array = str_array + fmt % element + delim
    # remove final delimiter
    str_array = str_array[:-1]
    return str_array

def read_file_header(filename):
    '''
    Reads the header of the filename to extract the actual ring system 
    parameters.

    Parameters
    ----------
    filename : str
        Name of the file to extract ring system information.
    
    Returns
    -------
    time : array_like (1-D)
        Contains the times that the simulated light curve was modelled for
        [day].
    planet_radius : float
        Size of the planet [R*].
    inner radii : array_like (1-D)
        Contains the inner radii of the actual ring system [R*].
    outer_radii : array_like (1-D)
        Contains the outer radii of the actual ring system [R*].
    opacities : array_like (1-D)
        Contains the opacities of the actual ring system [-].
    inclination : float
        Inclination of the ring system [deg].
    tilt : float
        Tilt of the ring system [deg].
    impact_parameter : float
        Impact parameter of the ring system [R*].
    dt : float
        Time offset of the light curve [day].
    limb_darkening : float
        Linear limb-darkening parameter of the star [-].
    transverse_velocity : float
        transverse velocity of the ring system in transit [R*/day].
    merge_fraction : float
        Value used to merge rings.
    num_rings : int
        Total number of rings for the actual ring system.
    num_visible : int
        Number of visible rings for the actual ring system.
    '''
    # open, read, close
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    # extract parameters
    time = str_to_arr(lines[1].split(' = ')[-1])
    planet_radius = float(lines[2].split(' = ')[-1])
    inner_radii = str_to_arr(lines[3].split(' = ')[-1])
    outer_radii = str_to_arr(lines[4].split(' = ')[-1])
    opacities = str_to_arr(lines[5].split(' = ')[-1])
    inclination = float(lines[6].split(' = ')[-1])
    tilt = float(lines[7].split(' = ')[-1])
    impact_parameter = float(lines[8].split(' = ')[-1])
    dt = float(lines[9].split(' = ')[-1])
    limb_darkening = float(lines[10].split(' = ')[-1])
    transverse_velocity = float(lines[11].split(' = ')[-1])
    merge_fraction = float(lines[12].split(' = ')[-1])
    num_rings = int(lines[13].split(' = ')[-1])
    num_visible = int(lines[14].split(' = ')[-1])
    return (time, planet_radius, inner_radii, outer_radii, opacities,
            inclination, tilt, impact_parameter, dt, limb_darkening,
            transverse_velocity, merge_fraction, num_rings, num_visible)



def expand_data(filename, new_num_ringlets, delim=','):
    '''
    This function expands the saved file such that the number of ringlets is
    extended. Note that the new number of ringlets must be an integer multiple
    of the old number of ringlets. This function uses np.repeat to expand the
    already saved opacities and will edit the column headers such that it can
    accomodate the new number of ringlets.

    Parameters
    ----------
    filename : str
        Name of the file to be expanded.
    new_num_ringlets : int
        New number of ringlets (expand to ->). This number must be an integer
        multiple of the old number of ringlets.
    delim : str
        The delimiter that separates the array element values [default = ','].

    Returns
    -------
    None
    '''
    # read file header
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    print('%s read' % filename)
    # prepare header
    header_lines = []
    for line in lines:
        if line[0] == '#':
            header_lines.append(line)
    # update last line
    header_lines = header_lines[:-1]
    col_headers = '# seed, noise_fraction, num_ringlets, num_merges'
    for i in range(new_num_ringlets):
        col_headers = col_headers + ', tau_%i' % i
    col_headers = col_headers + ', chi2'
    header_lines.append(col_headers)
    # create header string
    header = ''.join(header_lines)
    print('  header updated')
    # read data
    data = np.genfromtxt(filename, delimiter=delim)
    pre_data = data[:, :4]
    opacity_data = data[:, 4:-1]
    chi2_data = data[:, -1]
    # expand opacity data
    expand_value = new_num_ringlets // opacity_data.shape[1]
    new_opacity_data = np.repeat(opacity_data, expand_value, 0)
    # create new data array
    new_data = np.hstack((pre_data, new_opacity_data, chi2_data[:, None]))
    # overwrite save file
    print('  data expanded')
    np.savetxt(filename, new_data, header=header, comments='', fmt='%.8f')
    print('%s saved' % filename)
    return None


