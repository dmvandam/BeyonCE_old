'''
This module is used to explore the vast parameter space that circumplanetary
ring system transits cover. Circumplanetary ring systems transit light curves
span the following parameters:

    1. Disk Shape
        a. Ring Radii
        b. Ring Width
        c. Ring Opacity
    2. Disk Orientation
        a. Inclination
        b. Tilt
    3. Transit Properties
        a. Impact Parameter
        b. Time of Closest Approach
        c. Transverse Velocity
    4. Stellar Properties
        a. Limb-Darkening of the Stellar Disk
    5. Many Other Effects (not considered here)

We need to explore this large parameter space of interconnected properties as
efficiently as possible, and we need to restrict it using the information
encoded in the light curve, specifically:

    i.      The eclipse duration
    ii.     The gradients in the light curve
    iii.    Hill Sphere Stability Considerations

This module explores the space for large circumplanetary disks where the size
of the star w.r.t. the disk is negligible. It does this by creating a 2D grid
of circumplanetary disk centres, in other words (dx, dy), w.r.t. the centre of
the eclipse in time-space and then solving analytically for the family of
ellipses that intersect with the boundaries of the eclipse in time-space. Put
in other words, the width of the ellipse at dy = 0 is equal to the duration of
the eclipse.

This provides a method to represent this convoluted parameter space. Setting up
this grid allows us to make cuts to the parameter space based on two separate
fronts. The first is to limit the radius of the disk such that the disk would
be dynamically stable. This is akin to ensuring that the radius of the disk is
less than 30% of the Hill radius. The second is to convert the measured light
curve slopes to projected gradients, as in Kenworthy & Mamajek 2015, and ensure
that the measured gradients are always less than the theoretical gradients.
(https://iopscience.iop.org/article/10.1088/0004-637X/800/2/126 - for details).
    
    i.      disk_radius < 0.3 * Hill_radius
    ii.     measured_gradients < theoretical_gradients

By doing this we effectively reduce the available grid points in the parameter
space, enabling the further modeling of the light curve.

Finally if the module is run as a script (instead of imported from elsewhere)
a tutorial of each of the functions will be given (i.e. a description will be
printed along with relevant plots to show the working of the functions in this
module).

-----
The module gets its name from the equal-radius surfaces in the parameter grid
plots, which are shaped like sjalots or onions.
'''


###############################################################################
############################ IMPORT MAIN MODULES ##############################
###############################################################################

# calculations
import numpy as np
# timing / progress
from tqdm import tqdm


###############################################################################
############################ ELLIPSE PARAMETERS ###############################
###############################################################################

def determine_fx(te, dy, fy):
    '''
    This function is used to determine the proportionality factor in x given a
    proportionality factor in y. After determining the smallest circle centred
    at (0, dy) that passes through the eclipse points +-(te/2, 0), and this 
    circle is then either stretched or compressed in the y direction  with fy,
    this transformation in the y direction must be compensated by a respective
    compression or stretching of the new ellipse in the x direction (fx).

    Parameters
    ----------
    te : float
        duration of the eclipse [day]
    dy : array_like (1-D)
        contains the y coordinate of ellipse centres [day]
    fy : array_like (2-D)
        contains the original y proportionality factor for the ellipse

    Returns
    -------
    fx : array_like (2-D)
        contains the original x proportionality factor for the ellipse
    '''
    # we find fx for fy != 0
    z1 = fy**2 / (fy**2 - 1)
    z2 = (te/2)**2 / dy[:, None]**2
    z  = z1 * z2
    fx = np.sqrt(z / (1 + z))
    # we replace the nans with 1's as these come from fy**2 - 1 (1-1) = 0
    fx[fy==1] = 1
    # for point (dx, dy) = (0, 0) we know that fx=1
    fx[0, 0] = 1
    return fx

def shear_ellipse_point(Rmin, s, fx, fy, theta):
    '''
    This function finds the x and y coordinate for a sheared ellipse based on 
    the angle provided from the parametric equation of the ellipse.
    
    First we find the smallest circle, centred at (0, dy) that passes through
    the eclipse points +-(te/2, 0), then we transform it to an ellipse by using
    the proportionality factors fy and the determined fx. Finally we shear the
    ellipse to its final form.
    
    The cartesian coordinates for the given angle is returned for each grid
    point calculated.

    Parameters
    ----------
    Rmin : array_like (1-D)
        contains the minimum radius of a circle centred at (0, dy) and passing
        through +-(te/2, 0) [day]
    s : array_like (2-D)
        contains the shear factor based on the location of the grid point
        -dx / dy
    fx : array_like (2-D)
        contains the original x proportionality factor for the ellipse
    fy : array_like (2-D)
        contains the original y proportionality factor for the ellipse
    theta : float
        the angle of the point in the circle that will be transformed
        to the sheared circle [rad]

    Returns
    -------
    xp : array_like (2-D)
        the x-coordinate of the input point in the sheared circle [day]
    yp : array_like (2-D)
        the y-coordinate of the input point in the sheared circle [day]
    '''
    # determine the x and y coordinates of the ellipse
    y = fy * Rmin[:, None] * np.sin(theta)
    x = fx * Rmin[:, None] * np.cos(theta)
    # shear the circle
    yp = y
    xp = x - s * yp
    return xp, yp

def theta_max_min(s, fx, fy):
    '''
    Determines the parametric angle of either the semi-major axis or the 
    semi-minor axis of an ellipse. This is based on the fact that at the 
    vertices and co-vertices of an ellipse: dr/dtheta = 0.

    Parameters
    ----------
    s : array_like (2-D)
        contains the shear factor based on the location of the grid point
        -dx / dy
    fx : array_like (2-D)
        contains the original x proportionality factor for the ellipse
    fy : array_like (2-D)
        contains the original y proportionality factor for the ellipse

    Returns
    -------
    theta_max_min : array_like (2-D) 
        Array containing the angle of either the semi-major or semi-
        minor axis of an ellipse [rad]
    '''
    # determine the parametric angle of either the semi-major or minor axis
    top = 2 * fy * fx * s
    bot = (s**2 + 1) * fy**2 - fx**2
    theta_max_min = 0.5 * np.arctan2(top, bot)
    return theta_max_min

def ellipse_parameters(Rmin, s, fx, fy):
    '''
    This function finds the semi-major axis, a, semi-minor axis, b, the tilt 
    and the inclination of the smallest ellipse that is centred at (dx, dy) 
    w.r.t. the centre of the eclipse with duration te and stretch factors fx 
    and fy.

    Parameters
    ----------
    Rmin : array_like (1-D)
        contains the minimum radius of a circle centred at (0, dy) and passing
        through +-(te/2, 0) [day]
    s : array_like (2-D)
        contains the shear factor based on the location of the grid point
        -dx / dy
    fx : array_like (2-D)
        contains the original x proportionality factor for the ellipse
    fy : array_like (2-D)
        contains the original y proportionality factor for the ellipse

    Returns
    -------
    a : array_like (2-D) 
        semi-major axes of the ellipses investigated [day]
    b : array_like (2-D)
        semi-minor axes of the ellipses investigated [day]
    tilt : array_like (2-D)
        tilt angles of the ellipses investigated. This is the angle of the 
        semi-major axis w.r.t. the x-axis. [deg]
    inclination : array_like (2-D)
        inclination angles of the ellipses investigated. Inclination is based
        on the ratio of semi-minor to semi-major axis. [deg]
    '''
    # find position of (co-) vertices
    theta1 = theta_max_min(s, fx, fy)
    theta2 = theta1 + np.pi/2
    x1, y1 = shear_ellipse_point(Rmin, s, fx, fy, theta1)
    x2, y2 = shear_ellipse_point(Rmin, s, fx, fy, theta2)
    # find the semi-major and semi-minor axes
    R1 = np.hypot(x1, y1)
    R2 = np.hypot(x2, y2)
    a = np.maximum(R1, R2)
    b = np.minimum(R1, R2)
    # determine the inclination
    inclination = np.arccos(b / a)
    # determine the tilt
    tilt = np.arctan2(y1, x1) # assuming R1 > R2
    tilt_mask = R2 > R1 # find where above is not true
    tilt = tilt + tilt_mask * np.pi / 2 # at ^ locations add np.pi/2
    return a, b, np.rad2deg(tilt), np.rad2deg(inclination)

def ellipse_slope(x, dx, dy, s, fx, fy):
    '''
    This function finds the slopes of the tangents to the ellipse defined by 
    te, dx, dy, fy. This particular ellipse is determined by te, so we get 
    slopes at +-(te/2, 0). To determine the slope at (x, 0), where x is between
    +-te/2 we need to find the slope of the tangent for a concentric ellipse 
    that passes through +-(x, 0)

    Parameters
    ----------
    x : float
        time value at which to determine the slope [day]
    te : float
        duration of the eclipse [day]
    dx : array_like (1-D)
        contains the x coordinate of ellipse centres [day]
    dy : array_like (1-D)
        contains the y coordinate of ellipse centres [day]
    fy : array_like (2-D)
        contains the original y proportionality factor for the ellipse
    
    Returns
    -------
    slope : array_like (2-D)
        slopes of the ellipses investigated at (x, 0)
    '''
    # shift coordinates to frame centred on the ellipse centre
    X = x - dx[None, :]
    Y = -dy[:, None]
    # get the slopes
    top = -s * fy**2 * Y - fy**2 * X
    bot = (s**2 * fy**2 + fx**2) * Y + s * fy**2 * X
    slope = top/bot
    return slope

def slope_to_gradient(slope):
    '''
    This function converts a physical slope to a projected angular gradient.
    This is done as follows -> gradient = np.abs(np.sin(np.arctan2(slope, 1)))

    Parameters
    ----------
    slope : array_like (2-D)
        slopes of the ellipses investigated at (x, 0)

    Returns
    -------
    gradient : array_like (2-D)
        gradients of the ellipses investigated at (x, 0)
    '''
    gradient = np.abs(np.sin(np.arctan2(slope, 1)))
    return gradient


###############################################################################
########################### PARAMETER EXPLORATION #############################
###############################################################################

def fill_quadrants(prop, is_tilt=False):
    '''
    This function does the appropriate reflection symmetry and works for the
    semi-major axis, the semi-minor axis and the inclination. To be able to
    fill the quadrants for tilt the is_tilt parameter must be set to True.
    Gradients can not be filled in this way

    Parameters
    ----------
    prop : array_like (2-D)
        contains either the semi-major axis, the semi-minor axis, the tilt or 
        inclination of the investigated ellipses. Note that if the property
        tilt, then the is_tilt parameter should be equal to True, otherwise it
        should be False
    is_tilt : bool
        this parameter should be False unless prop = tilt. This is because
        the tilt parameter depends on which quadrant it is in (Q2 and Q4 are
        180 - Qx) and is not just a simple reflection

    Returns
    -------
    full_prop : array_like (2-D)
        contains the four quadrant version of the property passed in (this
        results in an array that is 4 times the size of the original). It
        also removes the central column/row
    '''
    ny, nx = prop.shape
    full_prop = np.zeros((2*ny, 2*nx))
    # create the quadrants
    Q1 = prop
    Q2 = np.fliplr(prop)
    Q3 = np.flipud(Q2)
    Q4 = np.flipud(prop)
    # fill the new array (watch for python indexing)
    full_prop[ny:, nx:] = Q1
    full_prop[ny:, :nx] = Q2
    full_prop[:ny, :nx] = Q3
    full_prop[:ny, nx:] = Q4
    # adjust quadrants
    if is_tilt == True:
        full_prop[ny:, :nx] = 180 - full_prop[ny:, :nx] # Q2
        full_prop[:ny, nx:] = 180 - full_prop[:ny, nx:] # Q4
    # remove the repeated row/column (i.e. dy=0 and dx=0)
    full_prop = np.delete(full_prop, ny, axis=0)
    full_prop = np.delete(full_prop, nx, axis=1)
    return full_prop

def mask_parameters(a, b, tilt, inclination, gradients, mask):
    '''
    This function applies a mask to the semi-major axis, the semi-minor axis,
    the tilt, the inclination and the gradients.

    Parameters
    ----------
    a : array_like (2-D)
        semi-major axes of the ellipses investigated [day]
    b : array_like (2-D)
        semi-minor axes of the ellipses investigated [day]
    tilt : array_like (2-D)
        tilt angles of the ellipses investigated. This is the angle of the 
        semi-major axis w.r.t. the x-axis. [deg]
    inclination : array_like (2-D)
        inclination angles of the ellipses investigated. Inclination is based
        on the ratio of semi-minor to semi-major axis. [deg]
    gradients : array_like (3-D)
        gradients of the ellipse investigated at each of the measured x values.
        note that the measured x values are w.r.t. the eclipse midpoint i.e.
        they should be between (-te/2, 0) and (te/2, 0).

    Returns
    -------
    a : array_like (2-D)
        semi-major axes of the ellipses investigated with the masked elements
        converted to nan's
    b : array_like (2-D)
        semi-minor axes of the ellipses investigated with the masked elements
        converted to nan's
    tilt : array_like (2-D)
        tilt angles of the ellipses investigated. This is the angle of the 
        semi-major axis w.r.t. the x-axis with the masked elements converted
        to nan's. [deg]
    inclination : array_like (2-D)
        inclination angles of the ellipses investigated. Inclination is based
        on the ratio of semi-minor to semi-major axis with the masked elements
        converted to nan's. [deg]
    gradients : array_like (3-D)
        gradients of the ellipse investigated at each of the measured x values.
        note thtat the measured x values are w.r.t. the eclipse midpoint i.e.
        they should be between (-te/2, 0) and (te/2, 0) with the masked
        elements converted to nan's.
    '''
    # applying the mask to each parameter
    a[mask] = np.nan
    b[mask] = np.nan
    tilt[mask] = np.nan
    inclination[mask] = np.nan
    for k in range(len(gradients)):
        gradients[k][mask] = np.nan
    return a, b, tilt, inclination, gradients

def investigate_ellipses(te, xmax, ymax, fy=1, measured_xs=None, nx=50, ny=50):
    '''
    Investigates the full parameter space for an eclipse of duration te with
    centres at [-xmax, xmax] (2*nx), [-ymax, ymax] (2*ny)

    Parameters
    ----------
    te : float
        duration of the eclipse [day]
    xmax : float
        contains the maximum value of dx [day]
    ymax : float
        contains the maximum value of dy [day]
    fy : float
        contains the original y proportionality factor for the ellipse 
        [default = 1]
    measured_xs : list or array_like (1-D)
        contains the time values where gradients have been measured in the
        light curve [day]
    nx : int
        number of gridpoints in the x direction [default = 50]
    ny : int
        number of gridpoints in the y direction [default = 50]

    Returns
    -------
    a : array_like (2-D)
        semi-major axes of the ellipses investigated [day]
    b : array_like (2-D)
        semi-minor axes of the ellipses investigated [day]
    tilt : array_like (2-D)
        tilt angles of the ellipses investigated. This is the angle of the 
        semi-major axis w.r.t. the x-axis. [deg]
    inclination : array_like (2-D)
        inclination angles of the ellipses investigated. Inclination is based
        on the ratio of semi-minor to semi-major axis. [deg]
    gradients : array_like (3-D)
        gradients of the ellipse investigated at each of the measured x values.
        Note that the measured x values are w.r.t. the eclipse midpoint i.e.
        they should be between (-te/2, 0) and (te/2, 0).
    '''
    # creating grids / phase space
    dy = np.linspace(0, ymax, ny)
    dx = np.linspace(0, xmax, nx)
    # determining sub-parameters
    Fy = fy * np.ones((ny, nx))
    Fx = determine_fx(te, dy, Fy)
    s  = -dx[None, :] / dy[:, None]
    s[np.isnan(s)] = 0 # correct (dx, dy) = (0, 0)
    Rmin = np.hypot(te/2, dy)
    # investigate phase space
    a, b, tilt, inclination = ellipse_parameters(Rmin, s, Fx, Fy)
    # fill quadrants
    a = fill_quadrants(a)
    b = fill_quadrants(b)
    tilt = fill_quadrants(tilt, is_tilt=True)
    inclination = fill_quadrants(inclination)
    # determine slopes for each measured point
    if isinstance(measured_xs, type(None)):
        measured_xs = []
    n_measured = len(measured_xs)
    # if no measured points then gradients = None
    if n_measured == 0:
        gradients = None
    # otherwise create an array and fill with the slopes
    else:
        gradients = np.zeros((n_measured, 2*ny-1, 2*nx-1))
        # get full sub-parameters
        DX = np.linspace(-xmax, xmax, 2 * nx - 1)
        DY = np.linspace(-ymax, ymax, 2 * ny - 1)
        S  = -DX[None, :] / DY[:, None]
        S[np.isnan(S)] = 0
        FY = fy * np.ones((2 * ny - 1, 2 * nx - 1))
        FX = determine_fx(te, DY, FY)
        for k, measured_x in enumerate(measured_xs):
            slope = ellipse_slope(measured_x, DX, DY, S, FX, FY)
            gradients[k] = slope_to_gradient(slope)
    return a, b, tilt, inclination, gradients

def full_investigation(te, xmax, ymax, dfy, Rmax, nx=50, ny=50, measured=None):
    '''
    This function investigates the full parameter space (dx, dy, fy) based
    on a grid size (nx, ny, dfy) dependent on the eclipse geometry (te) and
    the maximum size of the disk (Rmax). It also determines the gradients of
    the theoretical disks at the measured times

    Parameters
    ----------
    te : float
        duration of the eclipse [day]
    xmax : float
        contains the maximum value of dx [day]
    ymax : float
        contains the maximum value of dy [day]
    dfy : float
        contains the step size in fy for the original y proportionality factor
        for the ellipse
    Rmax : float
        the maximum size of the disk (used to apply a mask and determine the
        extent of fy) [day]
    nx : int
        number of gridpoints in the x direction
    ny : int
        number of gridpoints in the y direction
    measured : list of tuples
        contains the measured gradients and respective times as (time, 
        gradient)

    Return
    -------
    ac : array_like (3-D)
        cube of semi-major axes of the ellipses investigated [day]
    bc : array_like (3-D)
        cube of semi-minor axes of the ellipses investigated [day]
    tc : array_like (3-D)
        cube of tilt angles of the ellipses investigated. This is the angle of
        the semi-major axis w.r.t. the x-axis. [deg]
    ic : array_like (3-D)
        cube of inclination angles of the ellipses investigated. Inclination is
        based on the ratio of semi-minor to semi-major axis. [deg]
    gc : array_like (4-D)
        cube of gradients of the ellipse investigated at each of the measured x
        values. Note thtat the measured x values are w.r.t. the eclipse midpoint
        i.e. they should be between (-te/2, 0) and (te/2, 0).
    '''
    # determine fy extent
    fy_max = 2 * Rmax / te
    fys = np.arange(0, fy_max + dfy, dfy)
    nfy = len(fys)
    # extract measured times and gradients
    if isinstance(measured, type(None)):
        measured = []
    measured_times = []
    measured_gradients = []
    for m in measured:
        time, gradient = m
        measured_times.append(time)
        measured_gradients.append(gradient)
    # determine template shapes
    template_shape = (2 * ny - 1, 2 * nx - 1, nfy)
    nm = len(measured)
    # intialise parameter (a, b, t, i, g), (c)ubes
    ac = np.zeros(template_shape)
    bc = np.zeros(template_shape)
    tc = np.zeros(template_shape)
    ic = np.zeros(template_shape)
    gc = np.zeros((nm,) + template_shape)
    # loop over fy
    for k, fy in tqdm(enumerate(fys)):
        # determine the ellipse parameters
        parameters = (te, xmax, ymax, fy, measured_times, nx, ny)
        a, b, t, i, g = investigate_ellipses(*parameters)
        # fill the cubes
        ac[:, :, k] = a
        bc[:, :, k] = b
        tc[:, :, k] = t
        ic[:, :, k] = i
        gc[:, :, :, k] = g
    # remove all solutions where ac == 0
    ac, bc, tc, ic, gc = mask_parameters(ac, bc, tc, ic, gc, ac==0)
    # remove all solutions where ac > Rmax
    ac, bc, tc, ic, gc = mask_parameters(ac, bc, tc, ic, gc, ac>Rmax)
    # per measured gradient remove all solutions where gc[k] < measured[k]
    for k, gradient in enumerate(measured_gradients):
        ac, bc, tc, ic, gc = mask_parameters(ac, bc, tc, ic, gc, gc[k]<gradient)
    return ac, bc, tc, ic, gc

def grid_to_parameters(a_cube, tilt_cube, inclination_cube, xmax, ymax):
    '''
    This function extracts all the acceptable (non-NaN) solutions from the 
    provided cubes and converts the grid points to parameter values.

    Parameters
    ----------
    a_cube : array_like (3-D)
        cube of semi-major axes of the ellipses investigated [day]
    tilt_cube : array_like (3-D)
        cube of tilt angles of the ellipses investigated. This is the angle of
        the semi-major axis w.r.t. the x-axis. [deg]
    inclination_cube : array_like (3-D)
        cube of inclination angles of the ellipses investigated. Inclination is
        based on the ratio of semi-minor to semi-major axis. [deg]
    xmax : float
        contains the maximum value of dx [day]
    ymax : float
        contains the maximum value of dy [day]

    Returns
    -------
    disk_radii : array_like (1-D)
        all the possible disk radii [day]
    disk_tilt : array_like (1-D)
        all the possible disk tilts [deg]
    disk_inclination : array_like (1-D)
        all the possible disk inclinations [deg]
    disk_impact_parameters : array_like (1-D)
        all the possible disk impact parameters [day]
    disk_dts : array_like (1-D)
        all the possible x-offsets of the disk centres [day]
    '''
    # get cube shape
    ny, nx, nf = a_cube.shape
    # set-up dx and dy grids
    yy, xx = np.mgrid[:ny, :nx]
    # normalise grids from -1 to +1
    yy = 2 * (yy / (ny - 1) - 0.5)
    xx = 2 * (xx / (nx - 1) - 0.5)
    # scale to -ymax to +ymax and -xmax to +xmax
    yy = ymax * yy
    xx = xmax * xx
    # add the nf dimension
    yy = np.repeat(yy[:, :, None], nf, 2)
    xx = np.repeat(xx[:, :, None], nf, 2)
    # mask out the bad values
    mask = ~np.isnan(a_cube)
    # apply masks
    disk_radii = a_cube[mask]
    disk_tilts = tilt_cube[mask]
    disk_inclinations = inclination_cube[mask]
    disk_impact_parameters = yy[mask]
    disk_dts = xx[mask]
    return (disk_radii, disk_tilts, disk_inclinations, disk_impact_parameters,
            disk_dts)


###############################################################################
############################ VALIDATION FUNCTIONS #############################
###############################################################################

def get_closest_solution(a_cube, tilt_cube, inclination_cube, xmax, ymax, dfy,
                         disk_radius, disk_tilt, disk_inclination,
                         disk_impact_parameter):
    '''
    This function determines the closest grid point to the actual disk solution
    provided by the disk_* paramaters

    Parameters
    ----------
    a_cube : array_like (3-D)
        cube of semi-major axes of the ellipses investigated
    tilt_cube : array_like (3-D)
        cube of tilt angles of the ellipses investigated. This is the angle of
        the semi-major axis w.r.t. the x-axis. [deg]
    inclination_cube : array_like (3-D)
        cube of inclination angles of the ellipses investigated. Inclination is
        based on the ratio of semi-minor to semi-major axis. [deg]
    xmax : float
        contains the maximum value of dx
    ymax : float
        contains the maximum value of dy
    dfy : float
        contains the step size in fy for the original y proportionality factor
        for the ellipse
    disk_radius : float
        actual size of the disk
    disk_tilt : flaot
        actual tilt of the disk [deg]
    disk_inclination : float
        actual inclination of the disk [deg]
    disk_impact_parameter : float
        actual impact parameter (dy) of the disk

    Returns
    -------
    dx : float
        dx value of the closest solution [day]
    dy : float
        dy value of the closest solution [day]
    fy : float
        stretch factor value of the closest solution
    best_indices : tuple
        contains the indices for the closest solution
    '''
    # extract all the solutions from the provided grids
    sjalot_parameters = grid_to_parameters(a_cube, tilt_cube, inclination_cube,
                                           xmax, ymax)
    radii, tilts, inclinations, impact_parameters, dts = sjalot_parameters
    # determine the minimum distance
    distance = np.sqrt((radii - disk_radius)**2 + (tilts - disk_tilt)**2 + 
                       (inclinations - disk_inclinations)**2 + 
                       (impact_parameters - disk_impact_parameter)**2)
    # convert to sjalot coordinatesdetermine the sjalot parameters (parameters
    # of the closest grid point)
    best_ind = np.argmin(distance)
    dx = dts[best_ind]
    dy = impact_parameters[best_ind]
    # df is more involved
    mask = (a_cube == radii[best_ind]) * (tilt_cube == tilts[best_ind]) * ( 
           inclinations_cube == inclinations[best_ind])
    best_indices = np.unravel_index(np.argmax(mask), mask.shape)    
    fy = dfy * best_indices[2]
    return dx, dy, fy, best_indices


###############################################################################
################################# MODULE DEMO #################################
###############################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # start demos
    print('===========================================')
    print('ALL THE METHODS IN SJALOT.PY WILL BE DEMOED')
    print('===========================================')
    print('')
    ### SHEAR_ELLIPSE_POINT() ###
    print('1. sjalot.shear_ellipse_point()')
    print('-------------------------------')
    print('This function shears a point on a circle to its location on an')
    print('ellipse based on stretch factors in x and y, and a shear factor')
    te = 1
    dy = np.array([0.7])
    dx0 = np.array([0])
    dx1 = np.array([0.3])
    s0 = - dx0[None, :] / dy[:, None]
    s1 = - dx1[None, :] / dy[:, None]
    fy = np.array([[1]])
    fx = determine_fx(te, dy, fy)
    Rmin = np.hypot(te/2, dy)
    theta = np.linspace(0, 2*np.pi, 1001)
    xp0, yp0 = shear_ellipse_point(Rmin, s0, fx, fy, theta)
    xp1, yp1 = shear_ellipse_point(Rmin, s1, fx, fy, theta)
    fig = plt.figure()
    fig.suptitle('Demo: sjalot.shear_ellipse_point() (fy = 1, fx = 1)')
    ax0 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    ax1 = plt.subplot2grid((2, 3), (0, 2))
    ax2 = plt.subplot2grid((2, 3), (1, 2))
    axes = [ax0, ax1, ax2]
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('x [day]')
        ax.set_ylabel('y [day]')
        ax.plot(xp0[0] + dx0[0], yp0[0] + dy[0], 'b-', label='smallest circle')
        ax.plot(dx0[0], dy[0], 'bo')
        ax.plot(xp1[0] + dx1[0], yp1[0] + dy[0], 'r-', label='sheared circle')
        ax.plot(dx1[0], dy[0], 'ro')
        ax.plot([-100, -te/2], [0,0], 'k-', label='eclipse bounds')
        ax.plot([te/2, 100], [0,0], 'k-')
        ax.plot([-te/2, te/2], [0,0], 'ko')
    ax0.set_xlim(-2, 2)
    ax0.set_ylim(-1, 3)
    ax0.legend()
    ax1.set_xlim(-te/2-0.2, -te/2+0.2)
    ax1.set_ylim(-0.2, 0.2)
    ax2.set_xlim(te/2-0.2, te/2+0.2)
    ax2.set_ylim(-0.2, 0.2)
    plt.show()

#def determine_fx(te, dy, fy):
#    return fx
#def shear_ellipse_point(Rmin, s, fx, fy, theta):
#    return xp, yp
#def theta_max_min(s, fx, fy):
#    return theta_max_min
#def ellipse_parameters(Rmin, s, fx, fy):
#    return a, b, np.rad2deg(tilt), np.rad2deg(inclination)
#def ellipse_slope(x, dx, dy, s, fx, fy):
#    return slope
#def slope_to_gradient(slope):
# return gradient
