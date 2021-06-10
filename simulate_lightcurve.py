'''
This module is used to simulate light curves and heavily relise on the pyPplusS
package developed by Edan Rein & Aviv Ofir in 2019, which is available via pip
install or the github repository (https://github.com/EdanRein/pyPplusS).

The scientific paper describing the package and astrophysical applications was
published (https://academic.oup.com/mnras/article-abstract/490/1/1111/5568385).

pyPplusS was adapted in the simulate_lightcurve function as such:
    1. where pyPplusS determines the amount of light blocked by the occulter
       in physical space, this module converts to time space as follows

        a. introduces the transverse velocity to convert positions to time
        b. generalises the movement by assuming that the planet transits the
           star in a straight line with fixed impact parameter

    2. where pyPplusS allows a singular ringed planet, here we extend it so
       that an extended ring system can be modelled

        a. rings are defined by their inner and outer radii along with opacity
        b. the rings are deemed to be coplanar and concentric

Further lightcurve simulation tools have been added namely the ability to
    i.      add noise
    ii.     remove data (replicable by using non-uniform time array)
    iii.    generating a random ringsystem (number, dimensions & opacities)

Calculation tools have been added
    i.      calculate slopes in the light curve
    ii.     determine the minimum transverse velocity of the ringsystem. this
            is done via the Van Werkhoven et al. 2014
            (https://academic.oup.com/mnras/article/441/4/2845/1206172)

Plotting functions have been added
    i.      plot ringsystem (image)
    ii.     plot light curve
    iii.    plot combination of both
    iv.     all the relevant helper functions

Finally if the module is run as a script (instead of imported from elsewhere)
a tutorial of each of the functions will be given (i.e. a description will be
printed along with relevant plots to show the working of the functions in this
module).
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
    This function simulates a light curve based on the input parameters.

    Parameters
    ----------
    time : array_like (1-D)
        Time points at which to calculate the light curve.
    planet_radius : float
        Size of the planet [R*].
    inner_radii : array_like (1-D)
        Inner dimensions of the rings [R*].
    outer_radii : array_like (1-D)
        Outer dimensions of the rings [R*].
    opacities : array_like (1-D)
        Opacities of each ring [-].
    inclination : float
        Inclination of the ring system [deg].
    tilt : float
        Tilt of the rings, the angle between the path of orbital motion and
        the semi-major axis of the projected ellipse [deg].
    impact_parameter : float
        Impact parameter between the centre of the rings w.r.t. the centre of 
        the star [R*].
    dt : float
        This is a delta time parameter that can be used to shift the light
        curve left or right in time space [day].
    limb_darkening : float
        Limb-darkening parameter, u,  of the star according to the linear law,
        I(mu)/I(1) = 1 - u * (1 - mu), where mu = cos(y), where y is the angle
        between the line-of-sight and the emergent intensity [-].
    transverse_velocity : float
        The transiting velocity of the ring system across the star [R*/day].

    Returns
    -------
    lightcurve : array_like (1-D)
        Simulated theoretical light curve (normalised flux) based on the 
        inputs [L*].
    lightcurve_components : list of array_like (1-D)
        List containing the lightcurves produced by each of the components of
        the companion (planet + rings/disks) [L*].
    '''
    # create zeros and ones array
    zero = np.zeros_like(time)
    ones = np.ones_like(time)
    # initialise (p)lanet
    planet_x = (time - dt) * transverse_velocity
    planet_y = impact_parameter * ones
    planet_r = planet_radius * ones
    # inclination and tilt from degrees to radians
    inclination_rad = np.deg2rad(inclination)
    tilt_rad = np.deg2rad(tilt)
    # stellar limb darkening parameters
    c1 = 0
    c2 = limb_darkening
    c3 = 0
    c4 = 0
    # light curve variables
    lightcurve = 0
    lightcurve_components = []
    # determine lightcurve components
    # if planet touches the star: calculate transit | else: ones
    if (np.abs(impact_parameter) - planet_radius) < 1:
        r0 = 1e-16 * ones
        r1 = 2e-16 * ones
        planet_lightcurve = LC_ringed(planet_r, r0, r1, planet_x, planet_y,
                                      inclination_rad, tilt_rad, 0, c1, c2,
                                      c3, c4)
    else:
        planet_lightcurve = ones
    # add to lightcurve and append to lightcurve_components
    lightcurve += planet_lightcurve
    lightcurve_components.append(planet_lightcurve)
    # ensure that first inner radius != 0 (requirement of pyPplusS)
    if inner_radii[0] == 0:
        inner_radii[0] = 1e-16
    # loop over rings
    ring_params = (inner_radii, outer_radii, opacities)
    for inner_radius, outer_radius, opacity in zip(*ring_params):
        # if ring boundary touches the star calculate impact else ones
        ring_height = np.abs(outer_radius * np.sin(tilt_rad))
        if (np.abs(impact_parameter) - ring_height) < 1:
            # set-up ring bounds
            r0 = inner_radius * ones
            r1 = outer_radius * ones
            # group parameters
            lightcurve_params = (zero, r0, r1, planet_x, planet_y, inclination_rad, 
                                  tilt_rad, opacity, c1, c2, c3, c4)
            ring_lightcurve = LC_ringed(*lightcurve_params)
        else:
            ring_lightcurve = ones
        # add to lightcurve and append to lightcurve_components
        lightcurve += ring_lightcurve - 1
        lightcurve_components.append(ring_lightcurve)
    return lightcurve, lightcurve_components
  
def generate_random_ringsystem(radius_max, ring_num_min=3, ring_num_max=12,
                               tau_min=0.0, tau_max=1.0, print_rings=True):
    '''
    This function splits a disk into a ring system with a random number of
    rings each with random opacities.
    
    Parameters
    ----------
    radius_max : float
        Maximum size of the disk [R*].
    ring_num_min : int
        Minimum number of rings to separate the disk into.
    ring_num_max : int
        Maximum number of rings to separate the disk into.
    tau_min : float
        Minimum opacity of a ring [default = 0].
    tau_max : float
        Maximum opacity of a ring [default = 1].
    print_rings : bool
        If true then prints ring stats [default = True].
    
    Returns
    -------
    inner_radii : array_like (1-D)
        Inner dimensions of the rings [R*].
    outer_radii : array_like (1-D)
        Outer dimensions of the rings [R*].
    opacities : array_like (1-D)
        Opacities of each ring [-].
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
    This function adds noise to the light curve given a random number function
    and its given inputs. It also then re-normalises the lightcurve with the
    out-of-eclipse data.

    Parameters
    ----------
    lightcurve : array_like (1-D)
        Simulated theoretical light curve (normalised flux) based on the 
        inputs [L*].
    noise_func : function
        This function must be one such that it produces random numbers and has
        an argument size (see np.random documentation).
    noise_args : tuple
        This is an ordered tuple containing all the relevant arguments for the
        noise_func, with the exception of size (see np.random documentation).
    seed : int
        This sets the random noise generator so that you can extend noise runs
        performed at an earlier time.

    Returns
    -------
    noisy_lightcurve : array_like (1-D)
        Simulated theoretical lightcurve (normalised flux) with additional
        noise components defined by this function.
    '''
    # determine where the out-of-transit data is
    stellar_flux_mask = (lightcurve >= 0.999999)
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
    data simulate incomplete coverage.

    Parameters
    ----------
    time : array_like (1-D)
        Time data for the light curve [day].
    lightcurve : array_like (1-D)
        Normalised flux data for the light curve [L*].
    remove : int or array_like (1-D) of int
        Contains either the number of points to removed (chosen at random)
        or an index array for which points to remove.

    Returns
    -------
    incomplete_time : array_like (1-D)
        Time data for the light curve with data removed [day].
    incomplete_lightcurve : array_like (1-D)
        Normalised flux data for the light curve with data removed [L*].
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
    This function determines the slope of the light curve, between the times
    defined by slope bounds.

    Parameters
    ----------
    time : array_like (1-D)
        Time data for the light curve [day].
    lightcurve : array_like (1-D)
        Normalised flux data for the light curve [L*].
    slope_bounds : tuple
        Contains the time bounds for the slope calculation.

    Returns
    -------
    slope_time : array_like (1-D)
        Time at which slope is measured [day].
    slope : array_like (1-D)
        Slope measured in the lightcurve [L*/day].
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
    a list of slope_bounds.

    Parameters
    ----------
    time : array_like (1-D)
        Time data for the light curve [days].
    lightcurve : array_like (1-D)
        Normalised flux data for the light curve [L*].
    slope_bounds_list : list of tuples
        Contains a list of slope bound tuples, which contain the lower 
        and upper time bounds which the slopes are calculated [day].

    Returns
    -------
    slope_times : array_like (1-D)
        Time at which slope is measured [day].
    slopes : array_like (1-D)
        Slopes measured in the lightcurve [L*/day].
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

def get_min_velocity(slopes, limb_darkening):
    '''
    This function determines the minimum transverse velocity of the occulting
    object according to the Van Werkhoven et al. (2014) method. We determine
    this is as a function of the stellar radius.

    Parameters
    ----------
    slopes : array_like (1-D)
        Slopes measured in the light curve [L*/day].
    limb_darkening : float
        Limb-darkening parameter, u,  of the star according to the linear law,
        I(mu)/I(1) = 1 - u * (1 - mu), where mu = cos(y), where y is the angle
        between the line-of-sight and the emergent intensity [-].
    
    Returns
    -------
    min_velocity : float
        The minimum transverse velocity of the occulting object [R*/day].
    '''
    # determine the maximum slope dL/dt
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
        The minimum transverse velocity of the occulting object [R*/day].
    eclipse_duration : float
        Duration of the eclipse [day].

    Returns
    -------
    min_disk_radius : float
        Minimum size of the disk [R*].
    '''
    min_disk_radius = 0.5 * min_velocity * eclipse_duration
    return min_disk_radius 


##############################################################################
############################### PLOT FUNCTIONS ###############################
##############################################################################

def get_slope_lines(time, lightcurve, slope_times, slopes, length=0.1):
    '''
    This function produces the (x, y) coordinates of a line that represents
    each slope in the light curve at the correct position on the plot
    (slope_times, lightcurve @ slope_times).

    Parameters
    ----------
    time : array_like (1-D)
        Time data for the light curve [day].
    lightcurve : array_like (1-D)
        Normalised flux data for the light curve [L*].
    slope_times : array_like (1-D)
        Time at which slope is measured [day].
    slopes : array_like (1-D)
        Slopes measured in the lightcurve [L*/day].
    length : float
        Length of the line [day].

    Returns 
    -------
    slope_lines : array_like (3-D)
        x and y coordinates of a line centred at (slope_times, lightcurve @
        slope_times) with input length. The dimensions are as follows: the
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
    This function has been edited from a function written by Matthew 
    Kenworthy. The variable names, comments and documentation have been 
    changed, but the functionality has not.

    Parameters
    ----------
    inner_radius : float
        The inner radius of the ring [R*].
    outer_radius : float
        The outer radius of the ring [R*].
    opacity : float
        The opacity of the ring [-].
    inclination : float
        The inclination of the ring [deg].
    tilt : float
        Tilt of the rings, the angle between the path of orbital motion and
        the semi-major axis of the projected ellipse [deg].
    impact_parameter : float
        Impact parameter between the centre of the rings w.r.t. the centre of 
        the star [R*].
    dt : float
        This is a delta time parameter that can be used to shift the light
        curve left or right in time space [day]. Note here that as this has no
        effect on the shape it can be different from the actual dt value.
    facecolor : str
        The color of the ring patch [default = 'black'].
    
    Returns
    -------
    ring_patch : matplotlib.patch
        Patch of the ring with input parameters.

    Notes
    -----
    dt here has no effect on the ring system besides a translation along the 
    orbital path. You may want to use a different value than the dt used to 
    calculate the light curve for visualisation purposes.
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
    This function produces all the matplotlib patches necessary to draw the 
    ringsystem defined by the input parameters.

    Parameters
    ----------
    planet_radius : float
        Size of the planet [R*].
    inner_radii : array_like (1-D)
        Inner dimensions of the rings [R*].
    outer_radii : array_like (1-D)
        Outer dimensions of the rings [R*].
    opacities : array_like (1-D)
        Opacities of each ring [-].
    inclination : float
        Inclination of the ring system [deg].
    tilt : float
        Tilt of the rings, the angle between the path of orbital motion and
        the semi-major axis of the projected ellipse [deg].
    impact_parameter : float
        Impact parameter between the centre of the rings w.r.t. the centre of 
        the star [R*].
    dt : float
        This is a delta time parameter that can be used to shift the light
        curve left or right in time space [day].
    facecolor : str
        The color of the ring patches [default = 'black'].
    
    Returns
    -------
    ringsystem_patches : list of matplotlib.patch
        List containing all the ring patches that make up the ring system
        described by the input parameters and a circular patch for the planet.

    Notes
    -----
    dt here has no effect on the ring system besides a translation along the 
    orbital path. You may want to use a different value than the dt used to 
    calculate the light curve for visualisation purposes.
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

def plot_lightcurve(time, lightcurve, lightcurve_components, slope_lines=None, 
                    components=True, xlim=None, ylim=None, ax=None):
    '''
    This function plots the light curve for the provided ringsystem and can
    include the slopes lines provided.

    Parameters
    ----------
    time : array_like (1-D)
        Time data for the light curve [day].
    lightcurve : array_like (1-D)
        Normalised flux data for the light curve [L*].
    lightcurve_components : list of array_like (1-D)
        List containing the lightcurves produced by each of the components of
        the companion (planet + rings/disks) [L*/day].
    slope_lines : array_like (3-D)
        x and y coordinates of a line centred at (slope_times, lightcurve @
        slope_times) with input length. the dimensions are as follows: the
        0th dim is the line corresponding to a given slope, 1st dim is either
        the x or the y points, with the 2nd dim being the actual points. This
        allows one to loop over the slopes in the 0th dimension.
    components : bool
        Determines whether or not the lightcurves of the companion components
        are plotted [default = True].
    xlim : tuple
        x-limits of the plot.
    ylim : tuple
        y-limits of the plot.
    ax : matplotlib.axes()
        Potentially contains an axes object to plot the light curve on to.
    
    Returns
    -------
    ax : matplotlib.axes()
        Contains the axes with the light curve plot.
    '''
    # check slope_lines
    if isinstance(slope_lines, type(None)):
	    slope_lines = []
    # check axes
    if isinstance(ax, type(None)):
        ax = plt.gca()
    # plot components if requested
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
    # plot the slope lines
    for slope_line in slope_lines:
        ax.plot(slope_line[0], slope_line[1], 'm:', lw=4)
    # plot the full light curve
    ax.plot(time, lightcurve, 'k-', lw=2, label=lbl, alpha=0.5)
    # add legend
    if components == True:
        ax.legend(bbox_to_anchor=[1.0, 0.0], loc='lower left')
    # set x/y labels and limits
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Normalised Flux [-]')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    return ax

def plot_ringsystem(ringsystem_patches, xlim=None, ylim=None, ax=None):
    '''
    This function creates the ringsystem cartoon plot.

    Parameters
    ----------
    ringsystem_patches : list of matplotlib.patch
        List containing all the ring patches that make up the ring system
        described by the input parameters and a circular patch for the planet.
    xlim : tuple
        x-limits of the plot.
    ylim : tuple
        y-limits of the plot.
    ax : matplotlib.axes()
        Potentially contains an axes object to plot the light curve on to.
    
    Returns
    -------
    ax : matplotlib.axes()
        Contains the axes with the ringsystem cartoon.
    '''
    # get stellar patch
    star = Circle((0, 0), 1, facecolor='r')
    # create axes
    if ax == None:
        ax = plt.gca()
    ax.set_aspect('equal')
    # add star 
    ax.add_patch(star)
    # add companion (planet + ringsystem)
    for component in ringsystem_patches:
        ax.add_patch(component)
    # set x/y labels and limits
    ax.set_xlabel('x [R*]')
    ax.set_ylabel('y [R*]')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    return ax

def plot_combined(ringsystem_params, lightcurve_params, savename='test.png', 
                  figsize=(12, 10), title=''):
    '''
    This function creates a figure with two subplots, the ringsystem cartoon
    on the top and the lightcurve on the bottom.

    Parameters
    ----------
    ringsystem_params : tuple
        Contains all the input parameters for plot_ringsystem(), which are
        time, lightcurve, lightcurve_components, slope_lines=[], 
        components=True, xlim=None, ylim=None, ax=None, where ax should be
        ignored.
    lightcurve_params : tuple
        Contains all the input parameters for plot_lightcurve(), which are
        ringsystem_patches, xlim=None, ylim=None, ax=None, where ax should
        be ignored.
    savename : str
        Name of the file to be saved [default = 'test.png'].
    figsize : tuple
        Size of the plot [default = (12, 10)].
    title : str
        Title of the figure [default = ''].
    
    Returns
    -------
    matplotlib.figure()
    
    Notes
    -----
    For both ringsystem_params and lightcurve_params the axes object should
    NOT be specified.
    '''
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1 = plt.subplot2grid((3, 1), (2, 0))
    ax0 = plot_ringsystem(*ringsystem_params, ax=ax0)
    ax1 = plot_lightcurve(*lightcurve_params, ax=ax1)
    fig.savefig(savename)
    fig.show()
    return None


##############################################################################
################################# MODULE DEMO ################################
##############################################################################

if __name__ == "__main__":
    # import extra modules for demo
    from matplotlib.patches import Rectangle
    # start demos
    print('')
    print('========================================================')
    print('ALL THE METHODS IN SIMULATE_LIGHTCURVE.PY WILL BE DEMOED')
    print('========================================================')
    print('')
    ### SIMULATE_LIGHTCURVE() ###
    print('1. simulate_lightcurve.simulate_lightcurve()')
    print('--------------------------------------------')
    print('This function simulates the light curve of a transiting ring')
    print('system and each of the ring system\'s components.')
    # initialise parameters
    print('  a. initialising input parameters:')
    time = np.linspace(-85, 85, 301)
    time_pars = (time[0], time[-1], len(time))
    planet_radius = 0.3
    ring_edges = np.linspace(0, 130, 6)
    inner_radii = ring_edges[:-1]
    inner_radii[0] = 1e-16
    outer_radii = ring_edges[1:]
    opacities = np.random.uniform(0, 1, 5)
    inclination = 76
    tilt = 37
    impact_parameter = 12
    dt = 5
    limb_darkening = 0.4
    transverse_velocity = 1
    print('     time: from %.2f to %.2f day in %i equal steps' % time_pars)
    print('     planet_radius: %.2f [R*]' % planet_radius)
    print('     inner_radii: ', inner_radii, ' [R*]')
    print('     outer_radii: ', outer_radii, ' [R*]')
    print('     opacities: ', opacities, ' [-]')
    print('     inclination: %.2f [deg]' % inclination)
    print('     tilt: %.2f [deg]' % tilt)
    print('     impact_parameter: %.2f [R*]' % impact_parameter)
    print('     dt: %.2f [day]' % dt)
    print('     limb_darkening: %.2f' % limb_darkening)
    print('     transverse_velocity: %.2f [R*/day]' % transverse_velocity)
    # list dependencies
    print('  b. demo via:')
    print('      simulate_lightcurve.plot_lightcurve()')
    print('       - slope_lines demoed later')
    # prepare demo
    print('  c. running simulate_lightcurve.simulate_lightcurve() demo')
    sim_args = (time, planet_radius, inner_radii, outer_radii, opacities, 
                inclination, tilt, impact_parameter, dt, limb_darkening,
                transverse_velocity)
    lightcurve, lightcurve_components = simulate_lightcurve(*sim_args)
    fig, ax = plt.subplots(figsize=(12,6))
    fig.suptitle('Demo: simulate_lightcurve.simulate_lightcurve()')
    ax = plot_lightcurve(time, lightcurve, lightcurve_components)
    plt.show()
    print('\n')
    ### GENERATE RANDOM RINGSYSTEM() ###
    print('2. simulate_lightcurve.generate_random_ringsystem()')
    print('---------------------------------------------------')
    print('This function breaks up a circumplanetary disk into a ring system')
    print('by separating the disk into random connected rings with random')
    print('opacities.')
    # intialise input parameters
    print('  a. initialising input parameters:')
    disk_radius = outer_radii[-1]
    ring_num_min = 3
    ring_num_max = 12
    tau_min = 0.0
    tau_max = 1.0
    print_rings = False
    gen_args = (disk_radius, ring_num_min, ring_num_max, tau_min, tau_max, 
                print_rings)
    print('     disk_radius: %.2f [R*]' % disk_radius)
    print('     ring_num_min: %i' % ring_num_min)
    print('     ring_num_max: %i' % ring_num_max)
    print('     tau_min: %.2f [-]' % tau_min)
    print('     tau_min: %.2f [-]' % tau_max)
    print('     print_rings: %r' % print_rings)
    # list dependencies
    print('  b. demo via:')
    print('     simulate_lightcurve.plot_ringsystem()')
    print('       - helper: simulate_lightcurve.get_ringsystem_patches()')
    print('         - helper: simulate_lightcurve.get_ring_patch()')
    # prepare demo
    print('  c. running simulate_lightcurve.generate_random_ringsystem() demo')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Demo simulate_lightcurve.generate_random_ringsystem()')
    rs_xlim = (-120, 120)
    rs_ylim = (-100, 100)
    for i in range(2):
        for j in range(2):
            ax = axes[i,j]
            rin, rout, tau = generate_random_ringsystem(*gen_args)
            rsp_args = (planet_radius, rin, rout, tau, inclination, tilt,
                        impact_parameter, dt)
            rs_patches = get_ringsystem_patches(*rsp_args)
            ax = plot_ringsystem(rs_patches, rs_xlim, rs_ylim, ax)
    plt.show()
    print('\n')
    ### ADD_NOISE() ###
    print('3. simulate_lightcurve.add_noise()')
    print('----------------------------------')
    print('This function adds noise to a light curve given a certain noise')
    print('distribution.')
    # intialise input parameters
    print('  a. intialising input parameters:')
    noise_func = np.random.normal
    mean = np.zeros(4)
    std = np.array([0.00, 0.02, 0.05, 0.10])
    seed = np.random.randint(0, 100000, 1)
    print('     noise_func: np.random.normal')
    print('     noise_args:')
    print('        mean = 0, std = %.2f' % std[0])
    print('        mean = 0, std = %.2f' % std[1])
    print('        mean = 0, std = %.2f' % std[2])
    print('        mean = 0, std = %.2f' % std[3])
    print('     seed: %i (can be None)' % seed)
    # list dependencies
    print('  b. demo via:')
    print('     simulate_lightcurve.plot_lightcurve()')
    print('       - slope_lines demoed later')
    # prepare demo
    print('  c. running simulate_lightcurve.add_noise() demo')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Demo simulate_lightcurve.add_noise()')
    for i in range(2):
        for j in range(2):
            noise_ind = 2 * i + j
            noise_args = (mean[noise_ind], std[noise_ind])
            noisy_lightcurve = add_noise(lightcurve, noise_func, noise_args, 
                                         seed)
            ax = axes[i, j]
            ax = plot_lightcurve(time, noisy_lightcurve, lightcurve, 
                                 components=False, ax=ax)
            ax.set_title('noise = %.2f' % std[noise_ind])
    plt.show()
    print('\n')
    ### REMOVE_DATA() ###
    print('4. simulate_lightcurve.remove_data()')
    print('------------------------------------')
    print('This function removes data from an eclipse in two fashions, either')
    print('by supplying an integer (in which case that many random points will')
    print('be removed) or an index array (removing those particular data')
    print('points.')
    # intialise input parameters
    print('  a. initialising input parameters:')
    remove_int = 200 
    remove_array = np.array([15, 16, 17, 18, 19, 20, 67, 68, 69, 70, 71, 72,
                             73, 74, 75, 76, 77, 78, 79, 80, 100, 101, 102,
                             103, 104, 230, 231, 232])
    remove = [remove_int, remove_array]
    remove_lbl = ['type(remove) = int', 'type(remove) = list/array']
    print('     remove (int) = %i' % remove_int)
    print('     remove (array) = ', remove_array)
    # list dependencies
    print('  b. demo via:')
    print('     simulate_lightcurve.plot_lightcurve()')
    print('       - slope_lines demoed later')
    # prepare demo
    print('  c. running simulate_lightcurve.remove_data() demo')
    fig, axes = plt.subplots(2, 1, figsize= (12, 10))
    fig.suptitle('Demo: simulate_lightcurve.remove_data()')
    for i in range(2):
        itime, ilightcurve = remove_data(time, lightcurve, remove[i])
        axes[i] = plot_lightcurve(time, lightcurve, None, components=False, 
                                  ax=axes[i])
        axes[i].plot(time, lightcurve, 'ko', label='original lightcurve')
        axes[i].plot(itime, ilightcurve, 'go', label='data after removal')
        axes[i].legend()
        axes[i].set_title(remove_lbl[i])
    plt.show()
    print('\n')
    ### CALCULATE_SLOPES ###
    print('5. simulate_lightcurve.calculate_slopes()')
    print('-----------------------------------------')
    print('This function is used to calculate slopes in the light curve that')
    print('can be used for further processing (determining the minimum')
    print('transverse velocity of the ringsystem and to carve out the sjalot')
    print('explorer [separate BeyonCE module].')
    # initialise input parameters
    print('  a. initialising input parameters:')
    slope_bounds_list = [(-42, -39.5), (-32, -28.8), (25.5, 28.5), (46, 49.5)]
    print('     slope_bounds_list:')
    for sb in slope_bounds_list:
        print('        slope_bound = (%.2f, %.2f)' % sb)
    # list dependencies
    print('  b. demo via:')
    print('     helper: simulate_lightcurve.calculate_slope()')
    print('     simulate_lightcurve.plot_lightcurve()')
    print('       - helper: simulate_lightcurve.get_slope_line()')
    # prepare demo
    print('  c. running simulate_lightcurve.calculate_slopes() demo')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Demo: simulate_lightcurve.calculate_slopes()')
    slope_times, slopes = calculate_slopes(time, lightcurve, slope_bounds_list)
    slope_lines = get_slope_lines(time, lightcurve, slope_times, slopes)
    ax = plot_lightcurve(time, lightcurve, None, slope_lines=slope_lines, 
                         ax=ax, components=False)
    for slope_bounds in slope_bounds_list:
        tl, tu = slope_bounds
        bounds = Rectangle((tl, 0), tu-tl, 2, color='g', alpha=0.2)
        ax.add_patch(bounds)
    ax.plot(time, lightcurve, 'kx')
    plt.show()
    print('\n')
    ### GET_MIN_VELOCITY() ###
    print('6. simulate_lightcurve.get_min_velocity()')
    print('-----------------------------------------')
    print('This function follows equation 12 from Van Werkhoven et al. 2014')
    print('(https://academic.oup.com/mnras/article/441/4/2845/1206172).')
    print('The inputs are the measured slopes in the lightcurve, which can be')
    print('measured using simulate_lightcurve.calculate_slopes() and the')
    print('linear limb-darkening parameter of the star. [Demo n/a]')
    print('\n')
    ### GET_MIN_DISK_RADIUS() ###
    print('7. simulate_lightcurve.get_min_disk_radius()')
    print('--------------------------------------------')
    print('This function takes the minimum velocity of the disk provided by')
    print('simulate_lightcurve.get_min_velocity() and the duration of the')
    print('eclipse to determine the minimum disk radius of the transiting')
    print('ring system. [Demo n/a]') 
    print('\n')
    ### PLOT_COMBINED() ###
    print('8. simulate_lightcurve.plot_combined()')
    print('--------------------------------------')
    # initialise parameters
    print('  a. initialising parameters:')
    print('     plot_ringsystem parameters')
    print('     plot_lightcurve parameters')
    rsp_args = (planet_radius, inner_radii, outer_radii, opacities,
                inclination, tilt, impact_parameter, dt)
    ringsystem_patches = get_ringsystem_patches(*rsp_args)
    rs_xlim = (-120, 120)
    rs_ylim = (-100, 100)
    ringsystem_params = (ringsystem_patches, rs_xlim, rs_ylim)
    lightcurve_params = (time, lightcurve, lightcurve_components, slope_lines,
                         True)
    # list dependencies
    print('  b. demo via:')
    print('     helper: get_ringsystem_patches()')
    print('       - helper: get_ring_patch()')
    # prepare demo
    print('  c. running simulate_lightcurve.plot_combined() demo')
    plot_combined(ringsystem_params, lightcurve_params)
    print('     figure saved to \'./test.png\'')
    print('\n')
    print('==========================================================')
    print('ALL THE METHODS IN SIMULATE_LIGHTCURVE.PY HAVE BEEN DEMOED')
    print('==========================================================')
    print('')
