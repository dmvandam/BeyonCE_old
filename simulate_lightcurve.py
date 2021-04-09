'''
This module is used to simulate light curves based on the pyPplusS package
developed by Rein & Ofir 2019. It has several functionalities namely: light
curve simulation (generating light curves based on specific parameters, 
generating a random ringsystem, adding noise, removing data), processing the
lightcurve (measuring slopes, gradients), and plotting tools (plotting the
lightcurve, the ringsystem and a combination plot of both with relevant
helper functions for plotting the light curve gradients.
'''


###############################################################################
############################# IMPORT MAIN MODULES #############################
###############################################################################

# calculations
import numpy as np
from pyppluss.segment_models import LC_ringed
# plotting
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Circle, Ellipse, PathPatch


###############################################################################
############################# SIMULATE LIGHT CURVE ############################
###############################################################################

def simulate_lightcurve(time, planet_radius, inner_radii, outer_radii, 
                        opacities, inclination, tilt, impact_parameter, dt, 
                        limb_darkening, transverse_velocity=1):
    '''
    This function simulates a lightcurve based on the input parameters

    Parameters
    ----------
    time : array_like (1-D)
        time points at which to calculate the light curve
    planet_radius : float
        size of the planet [R*]
    inner_radii : array_like (1-D)
        inner dimension of the rings [R*]
    outer_radii : array_like (1-D)
        outer dimension of the rings [R*]
    opacities : array_like (1-D)
        opacities of each ring [-]
    inclination : float
        inclination of the ring system [deg]
    tilt : float
        tilt of the rings, is the angle between the x-axis and the semi-major
        axis of the projected ellipse [deg]
    impact_parameter : float
        impact parameter between the centre of the rings w.r.t. centre of the
        star [R*]
    dt : float
        this is a delta time parameter that can be used to shift the lightcurve
        left or right [days]
    limb_darkening : float
        limb-darkening parameter, u,  of the star according to the linear law,
        I(mu)/I(1) = 1 - u * (1 - mu), where mu = cos(y), where y is the angle
        between the line-of-sight and the emergent intensity [-]
    transverse_velocity : float
        the transiting velocity of the ring system across the star [R*/day]

    Returns
    -------
    lightcurve : array_like (1-D)
        simulated theoretical lightcurve (normalised flux) based on the inputs
        [L*]
    lightcurve_components : list of array_like (1-D)
        list containing the lightcurves produced by each of the components of
        the companion (planet + rings/disks) [L*]
    '''
    # create zeros and ones array
    zero = np.zeros_like(time)
    ones = np.ones_like(time)
    # initialise (p)lanet
    planet_x = (time + dt) * transverse_velocity
    planet_y = impact_parameter * ones
    planet_r = planet_radius * ones
    # inclination and tilt from degrees to radians
    inclination_rad = np.deg2rad(inclination)
    tilt_rad = np.deg2rad(tilt)
    # stellar parameters
    c1 = 0
    c2 = limb_darkening
    c3 = 0
    c4 = 0
    # save lightcurve
    lightcurve = 0
    lightcurve_components = []
    # determine lightcurve components
    # if planet touches the star calculate impact else ones
    if (np.abs(impact_parameter) - planet_radius) < 1:
        r0 = 1e-16 * ones
        r1 = 2e-16 * ones
        planet_lightcurve = LC_ringed(planet_r, r0, r1, planet_x, planet_y,
                                      inclination_rad, tilt_rad, 0, c1, c2,
                                      c3, c4)
    else:
        planet_lightcurve = ones
    # add to lightcurve and lightcurve_components
    lightcurve += planet_lightcurve
    lightcurve_components.append(planet_lightcurve)
    # ensure that first inner radius != 0
    if inner_radii[0] == 0:
        inner_radii[0] = 1e-16
    # loop through rings
    for inner_r, outer_r, opacity in zip(inner_radii, outer_radii, opacities):
        # if ring boundary touches the star calculate impact else ones
        if (np.abs(impact_parameter) - np.abs(outer_r * np.sin(tilt_rad))) < 1:
            # set-up ring radii
            r0 = inner_r * ones
            r1 = outer_r * ones
            # group parameters
            params = (zero, r0, r1, planet_x, planet_y, inclination_rad, 
                      tilt_rad, opacity)
            ring_lightcurve = LC_ringed(*params, c1, c2, c3, c4)
        else:
            ring_lightcurve = ones
        # add to lightcurve and lightcurve_components
        lightcurve += ring_lightcurve - 1
        lightcurve_components.append(ring_lightcurve)
    return lightcurve, lightcurve_components
  
def generate_random_ringsystem(radius_max, ring_num_min=3, ring_num_max=12, 
                              tau_min=0.2, tau_max=1., print_rings=True):
    '''
    This function splits a disk into a ring system with a random number of
    rings with random opacities
    
    Parameters
    ----------
    radius_max : float
        maximum size of the disk [R*]
    ring_num_min : int
        minimum number of rings to separate the disk into
    ring_num_max : int
        maximum number of rings to separate the disk into
    tau_min : float
        minimum opacity of a ring
    tau_max : float
        maximum opacity of a ring
    print_rings : bool
        if true then prints ring stats
    
    Returns
    -------
    inner_radii : array of floats
        contains the inner radii for each ring (ring edges)
    outer_radii : array of floats
        contains the outer radii for each ring (ring edges)
    opacities : array of floats
        contains the opacities of each ring
    '''
    # random number of rings
    num_rings = np.random.randint(ring_num_min, ring_num_max)
    # random ring_edge fractions
    ring_edges = np.random.uniform(0, 1, num_rings - 1) * radius_max
    ring_edges = np.sort(ring_edges)
    # define outer radii
    outer_radii      = np.zeros(num_rings)
    outer_radii[:-1] = ring_edges
    outer_radii[-1]  = radius_max
    # define inner radii
    inner_radii     = np.zeros(num_rings)
    inner_radii[1:] = ring_edges
    inner_radii[0]  = 1e-16
    # random opacities
    opacities = np.random.uniform(tau_min, tau_max, num_rings)
    if print_rings == True:
        print('There are a total of %i rings' % num_rings)
        template = '  ring %s runs from %s to %s [R*] with an opacity of %.4f'
        for n in range(num_rings):
            ring_num = ('%i' % (n+1)).rjust(2)
            ring_in  = ('%.2f' % inner_radii[n]).rjust(6)
            ring_out = ('%.2f' % outer_radii[n]).rjust(6)
            pars = (ring_num, ring_in, ring_out, opacities[n])
            print(template % pars)
    return inner_radii, outer_radii, opacities

def add_noise(lightcurve, noise_func, noise_args, seed=None):
    '''
    this function adds noise to the light curve given a random number function
    and its given inputs. It also then re-normalises the lightcurve.

    Parameters
    ----------
    lightcurve : array_like (1-D)
        simulated theoretical lightcurve (normalised flux) based on the inputs
        [L*]
    noise_func : function
        this function must be one such that it produces random numbers and has
        an argument size (see np.random documentation)
    noise_args : tuple
        this is an ordered tuple containing all the relevant arguments for the
        noise_func, with the exception of size (see np.random documentation)
    seed : int
        this sets the random noise generator so that you can extend noise runs
        performed at an earlier time

    Returns
    -------
    noisy_lightcurve : array_like (1-D)
        simulated theoretical lightcurve (normalised flux) with additional
        noise components defined by this function
    '''
    # determine where the out-of-transit data is
    stellar_flux_mask = (lightcurve == 1)
    # determine the noise
    num_data = len(lightcurve)
    np.random.seed(seed)
    noise = noise_func(*noise_args, size=num_data)
    # add the noise
    noisy_lightcurve = noise + lightcurve
    # renormalise the lightcurve
    median = np.median(noisy_lightcurve[stellar_flux_mask]) - 1
    noisy_lightcurve -= median 
    return noisy_lightcurve

def remove_data(time, lightcurve, remove=None):
    '''
    This function removes data from a light curve to produce holes in the
    data collection to simulate incomplete coverage

    Parameters
    ----------
    time : array_like (1-D)
        time data for the light curve [days]
    lightcurve : array_like (1-D)
        normalised flux data for the light curve [L*]
    remove : int or array of int
        contains either the number of points to removed (chosen at random)
        or an index array for which points to remove

    Returns
    -------
    incomplete_time : array_like (1-D)
        time data for the light curve with data removed [days]
    incomplete_lightcurve : array_like (1-D)
        normalised flux data for the light curve with data removed [days] 
    '''
    if type(remove) == int:
        remove = np.random.randint(0, len(time) - 1, remove)
    incomplete_time = np.delete(time, remove)
    incomplete_lightcurve = np.delete(lightcurve, remove)
    return incomplete_time, incomplete_lightcurve


###############################################################################
################################ CALCULATIONS #################################
###############################################################################

def calculate_slope(time, lightcurve, slope_bounds):
    '''
    This function determines the slope of the lightcurve, between the times
    defined by slope bounds

    Parameters
    ----------
    This function determines the slope of the lightcurve, between the times
    defined by slope bounds

    Parameters
    ----------
    time : array_like (1-D)
        time data for the light curve [days]
    lightcurve : array_like (1-D)
        normalised flux data for the light curve [L*]
    slope_bounds : tuple
        contains the time bounds for the slope calculation

    Returns
    -------
    slope_time : array_like (1-D)
        time at which slope is measured [days]
    slope : array_like (1-D)
        slope measured in the lightcurve [L*/day]
    '''
    # select the relevant section of the lightcurve
    mask = (time >= slope_bounds[0]) * (time <= slope_bounds[1])
    # fit a line to the relevant points
    p0 = np.polyfit(time[mask], lightcurve[mask], 1)
    # get slope_time and slope
    slope_time = 0.5 * (time[mask][0] + time[mask][-1])
    slope = p0[0]
    return slope_time, slope

def calculate_slopes(time, lightcurve, slope_bounds_list):
    '''
    This function bulkifies the calculate slope function by requiring
    a list of slope_bounds

    Parameters
    ----------
    time : array_like (1-D)
        time data for the light curve [days]
    lightcurve : array_like (1-D)
        normalised flux data for the light curve [L*]
    slope_bounds_list : list of tuples
        contains a list of slope bound tuples, which are lower time bound
        and the upper time bound for which the slopes are calculated

    Returns
    -------
    slope_times : array_like (1-D)
        time at which slope is measured [days]
    slopes : array_like (1-D)
        slopes measured in the lightcurve [L*/day]
    '''
    # set up arrays
    num_slopes = len(slope_bounds_list)
    slope_times = np.zeros(num_slopes)
    slopes = np.zeros(num_slopes)
    # loop through slope_bounds_list
    for k, slope_bounds in enumerate(slope_bounds_list):
        # calculate the slopes
        slope_time, slope = calculate_slope(time, lightcurve, slope_bounds)
        slope_times[k] = slope_time
        slopes[k] = slope
    return slope_times, slopes

def slope_to_gradient(slopes):
    '''
    This function converts a lightcurve slope to a normalised projected 
    gradient, using the transformation gradient = np.abs(np.sin(slope)).
    This converts the slope to a gradient that runs from 0 to 1.

    Parameters
    ----------
    slopes : array_like (1-D)
        slopes measured in the light curve [L*/day]
    
    Returns
    -------
    gradients : array_like (1-D)
        gradients measured in the light curve
    '''
    gradients = np.abs(np.sin(slopes))
    return gradients

def get_min_velocity(slopes, limb_darkening):
    '''
    This function determines the minimum transverse velocity of the occulting
    object according to the Van Werkhoven et al. (2014) method. We determine
    this is as a function of the stellar radius.

    Parameters
    ----------
    slopes : array_like (1-D)
        slopes measured in the lightcurve [L*/day]
    limb_darkening : float
        limb-darkening parameter, u,  of the star according to the linear law,
        I(mu)/I(1) = 1 - u * (1 - mu), where mu = cos(y), where y is the angle
        between the line-of-sight and the emergent intensity [-]
    
    Returns
    -------
    min_velocity : float
        the minimum transverse velocity of the occulting object [R*/day]
    '''
    # determine the maximum slope
    dLdt = np.amax(np.abs(slopes))
    # define the minimum velocity (excluding R from van Werkhoven et al. 2014)
    top = 2 * limb_darkening - 6
    bot = 12 - 12 * limb_darkening + 3 * np.pi * limb_darkening
    # calculate the minimum transverse velocity
    min_velocity = np.abs(dLdt * np.pi * (top / bot))
    return min_velocity

def get_min_disk_radius(min_velocity, eclipse_duration):
    '''
    This function converts the minimum transverse velocity of the disk to a 
    minimum disk size based on the duration of the eclipse. This is based on
    the simple assumption that given a velocity and a time, we can find a
    distance, which we can say is the minimum disk diameter.

    Parameters
    ----------
    min_velocity : float
        the minimum transverse velocity of the occulting object [R*/day]
    eclipse_duartion : float
        duration of the eclipse [days]

    Returns
    -------
    min_disk_radius : float
        minimum size of the disk [R*]
    '''
    min_disk_radius = 0.5 * min_velocity * eclipse_duration
    return min_disk_radius 


##############################################################################
############################### PLOT FUNCTIONS ###############################
##############################################################################

def get_slope_line(time, lightcurve, slope_times, slopes, length=0.1):
    '''
    This function produces the (x, y) coordinates of a line that represents
    each slope in the light curve at the correct position on the plot
    (slope_times, lightcurve @ slope_times)

    Parameters
    ----------
    time : array_like (1-D)
        time data for the light curve [days]
    lightcurve : array_like (1-D)
        normalised flux data for the light curve [-]
    slope_times : array_like (1-D)
        time at which slope is measured [days]
    slopes : array_like (1-D)
        slopes measured in the lightcurve [L*/day]
    length : float
        length of the line [days]

    Returns 
    -------
    slope_lines : array_like (3-D)
        x and y coordinates of a line centred at (slope_times, lightcurve @
        slope_times) with input length. the dimensions are as follows: the
        0th dim is the line corresponding to a given slope, 1st dim is either
        the x or the y points, with the 2nd dim being the actual points. This
        allows one to loop over the slopes in the 0th dimension.
    '''
    # calculate the dx and dy values for the line 
    dx = np.sqrt(length / (slopes**2 + 1))
    dy = slopes * dx
    # determine the points on which the lines are centred (x, y)
    x = slope_times
    y = np.interp(x, time, lightcurve)
    # calculate the lines
    slope_lines = np.array([[x - dx, x, x + dx], [y - dy, y, y + dy]])
    # rearrange array such that the dimensions become (slope, x/y, points)
    slope_lines = np.transpose(slope_lines, (2, 0, 1))
    return slope_lines

def get_ring_patch(inner_radius, outer_radius, opacity, inclination, tilt,
                   impact_parameter, dt, facecolor='black'):
    '''
    This function has been edited from a function by Matthew Kenworthy.
    The documentation has been changed, the functionality has not

    Parameters
    ----------
    inner_radius : float
        the inner radius of the ring [R*]
    outer_radius : float
        the outer radius of the ring [R*]
    opacity : float
        the opacity of the ring [-]
    inclination : float
        the inclination of the ring [deg]
    tilt : float
        the counter-clockwise angle of the semi-major axis w.r.t. the x-axis 
        [deg]
    impact_parameter : float
        the y location of the centre of the ring [R*]
    dt : float
        the x location of the centre of the ring [R*]
    facecolor : str
        the color of the ring patch [default = 'black']
    
    Returns
    -------
    ring_patch : matplotlib.patch
        patch of the ring with input parameters
    '''
    # convert inclination and tilt to radians
    inclination_rad = np.deg2rad(inclination)
    tilt_rad = np.deg2rad(tilt)
    # centre location
    dr = np.array([dt, impact_parameter])
    # get an Ellipse patch that has an ellipse defined with eight CURVE4 Bezier
    # curves actual parameters are irrelevant - get_path() returns only a 
    # normalised Bezier curve ellipse which we then subsequently transform
    ellipse = Ellipse((1, 1), 1, 1, 0)
    # get the Path points for the ellipse (8 Bezier curves with 3 additional 
    # control points)
    ellipse_path = ellipse.get_path()
    # define rotation matrix
    rotation_matrix = np.array([[np.cos(tilt_rad),  np.sin(tilt_rad)], 
                                [np.sin(tilt_rad), -np.cos(tilt_rad)]])
    # squeeze the circle to the appropriate ellipse
    annulus1 = ellipse_path.vertices * ([ 1., np.cos(inclination_rad)])
    annulus2 = ellipse_path.vertices * ([-1., np.cos(inclination_rad)])
    # rotate and shift the ellipses
    ellipse_rot1 = np.dot(annulus1 * outer_radius, rotation_matrix) + dr
    ellipse_rot2 = np.dot(annulus2 * inner_radius, rotation_matrix) + dr
    # produce the arrays neccesary to produce a new Path and Patch object
    ring_vertices = np.vstack((ellipse_rot1, ellipse_rot2))
    ring_commands = np.hstack((ellipse_path.codes, ellipse_path.codes))
    # create the Path and Patch objects
    ring_path  = Path(ring_vertices, ring_commands)
    ring_patch = PathPatch(ring_path, facecolor=facecolor, edgecolor='none',
                           alpha=opacity)
    return ring_patch

def get_ringsystem_patches(planet_radius, inner_radii, outer_radii, opacities,
                           inclination, tilt, impact_parameter, dt, 
                           facecolor='black'):
    '''
    This function produces the ring patches for matplotlib defined by the input
    parameters.

    Parameters
    ----------
    planet_radius : float
        size of the planet [R*]
    inner_radii : array_like (1-D)
        the inner radii of the ring system [R*]
    outer_radius : array_like (1-D)
        the outer radii of the ring system [R*]
    opacities : array_like (1-D)
        the opacities of the rings in the ring system [-]
    inclination : float
        the inclination of the ring system [deg]
    tilt : float
        the counter-clockwise angle of the semi-major axis of the ring system
        w.r.t. the x-axis [deg]
    impact_parameter : float
        the y location of the centre of the ring system [R*]
    dt : float
        the x location of the centre of the ring system [R*]
    facecolor : str
        the color of the ring patches [default = 'black']
    
    Returns
    -------
    ringsystem_patches : list of matplotlib.patch
        list containing all the ring patches that make up the ring system
        described by the input parameters
    '''
    # create empty list
    ringsystem_patches = []
    # create planet patch and append to ringsystem_patches
    planet = Circle((dt, impact_parameter), planet_radius, facecolor=facecolor)
    ringsystem_patches.append(planet)
    # bundle ring parameters
    ring_params = (inner_radii, outer_radii, opacities)
    # loop through rings and append to ring_patches
    for inner_radius, outer_radius, opacity in zip(*ring_params):
        ring = get_ring_patch(inner_radius, outer_radius, opacity, inclination,
                              tilt, impact_parameter, dt, facecolor)
        ringsystem_patches.append(ring)
    return ringsystem_patches

def plot_lightcurve(time, lightcurve, lightcurve_components, slope_lines=[], 
                    components=True, xlim=None, ylim=None, ax=None):
    '''
    This function plots the light curve for the provided ringsystem and can
    include the slopes lines provided.

    Parameters
    ----------
    time : array_like (1-D)
        time points at which to calculate the light curve [days]
    lightcurve : array_like (1-D)
        simulated theoretical lightcurve (normalised flux) based on the inputs
        [L*]
    lightcurve_components : list of array_like (1-D)
        list containing the lightcurves produced by each of the components of
        the companion (planet + rings/disks) [L*/day]
    slope_lines : array_like (3-D)
        x and y coordinates of a line centred at (slope_times, lightcurve @
        slope_times) with input length. the dimensions are as follows: the
        0th dim is the line corresponding to a given slope, 1st dim is either
        the x or the y points, with the 2nd dim being the actual points. This
        allows one to loop over the slopes in the 0th dimension.
    components : bool
        determines whether or not the lightcurves of the companion components
        are plotted [default = True]
    xlim : tuple
        x-limits of both subplots
    ylim : tuple
        y-limits of the depiction subplot
    
    Returns
    -------
    ax : matplotlib.axes()
        contains the axes with the lightcurve 
    '''
    # define axes
    if ax == None:
        ax = plt.gca()
    if components == True:
        lbl = 'planet'
        for k, lightcurve_component in enumerate(lightcurve_components):
            mask = lightcurve_component < 1 
            ax.plot(time[mask], lightcurve_component[mask], marker='.', ls='',
                     label=lbl, alpha=0.6)
            lbl = 'ring #%i' % (k+1)
        lbl = 'full lightcurve'
    else:
        lbl = None
    for slope_line in slope_lines:
        ax.plot(slope_line[0], slope_line[1], 'm:', lw=4)
    ax.plot(time, lightcurve, 'k-', lw=2, label=lbl, alpha=0.5)
    ax.legend(bbox_to_anchor=[1.0, 0.0], loc='lower left')
    ax.set_xlabel('Date [days]')
    ax.set_ylabel('Normalised Flux [-]')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    return ax

def plot_ringsystem(ringsystem_patches, xlim=None, ylim=None, ax=None):
    '''
    This function creates the ringsystem cartoon plot

    Parameters
    ----------
    ringsystem_patches : list of matplotlib.patch
        list containing all the ring patches that make up the ring system
        described by the input parameters
    xlim : tuple
        x-limits of both subplots
    ylim : tuple
        y-limits of the depiction subplot
    ax : matplotlib.axes()
        axes to transform [default = None]
    
    Returns
    -------
    ax : matplotlib.axes()
        contains the axes with the ringsystem cartoon
    '''
    # get stellar patch
    star = Circle((0, 0), 1, facecolor='r')
    # create axes
    if ax == None:
        ax = plt.gca()
    ax.set_aspect('equal')
    # add star and ringsystem
    ax.add_patch(star)
    for component in ringsystem_patches:
        ax.add_patch(component)
    # set axes labels
    ax.set_xlabel('x [days]')
    ax.set_ylabel('y [days]')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    return ax

def plot_combined(ringsystem_params, lightcurve_params, savename='test.png', 
                  figsize=(12, 10)):
    '''
    This function creates a figure with two subplots, the ringsystem cartoon
    on the top and the lightcurve on the bottom

    Parameters
    ----------
    ringsystem_params : tuple
        contains all the input parameters for plot_ringsystem(), which are
        time, lightcurve, lightcurve_components, slope_lines=[], 
        components=True, xlim=None, ylim=None, ax=None, where ax should be
        ignored
    lightcurve_params : tuple
        contains all the input parameters for plot_lightcurve(), which are
        ringsystem_patches, xlim=None, ylim=None, ax=None, where ax should
        be ignored
    savename : str
        name of the file to be saved [default = 'test.png']
    figsize : tuple
        size of the plot [default = (12, 10)]
    
    Returns
    -------
    matplotlib.figure()
    '''
    fig = plt.figure(figsize=figsize)
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1 = plt.subplot2grid((3, 1), (2, 0))
    ax0 = plot_ringsystem(*ringsystem_params, ax=ax0)
    ax1 = plot_lightcurve(*lightcurve_params, ax=ax1)
    fig.savefig(savename)
    fig.show()
    return None
