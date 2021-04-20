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

def determine_fx(te, dx, dy, Fy):
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
        Duration of the eclipse [day].
    dx : array_like (1-D)
        Contains the x coordinate of ellipse centres [day].
    dy : array_like (1-D)
        Contains the y coordinate of ellipse centres [day].
    Fy : float
        Defines the original y proportionality factor for the ellipses.

    Returns
    -------
    fx : array_like (2-D)
        contains the original x proportionality factor for the ellipses.
    fy : array_like (2-D)
        contains the original y proportionality factor for the ellipses.
    '''

    if Fy == 1:
        fx = np.ones((len(dy), len(dx)))
    else:
        # define fx with helper variables
        z1 = Fy**2 / (Fy**2 - 1)
        # preventing divide by 0 error for z2
        z2 =  np.zeros_like(dy)
        z2[dy!=0] = (te/2)**2 / dy[dy!=0]**2
        z  = (z1 * z2) / (z1 * z2 + 1)
        # preventing complex root issues
        fx = np.sqrt(z, where=(z >= 0))
        fx[z<0] = np.nan
        # extending fx to 2D array
        fx = np.repeat(fx[:, None], len(dx), 1)
        # for point (dx, dy) = (0, 0) we know that fx = 1
        origin = (dy[:, None] == 0) * (dx[None, :] == 0)
        fx[origin] = 1
    # extending fy to 2D array
    fy = Fy * np.ones((len(dy), len(dx)))
    return fx, fy

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
        Contains the minimum radius of a circle centred at (0, dy) and passing
        through +-(te/2, 0) [day].
    s : array_like (2-D)
        Contains the shear factor based on the location of the grid point
        -dx / dy.
    fx : array_like (2-D)
        Contains the original x proportionality factor for the ellipse.
    fy : array_like (2-D)
        Contains the original y proportionality factor for the ellipse.
    theta : float
        The angle of the point in the circle that will be transformed
        to the sheared circle [rad].

    Returns
    -------
    xp : array_like (2-D)
        The x-coordinate of the input point in the sheared circle [day].
    yp : array_like (2-D)
        The y-coordinate of the input point in the sheared circle [day].
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
        Contains the shear factor based on the location of the grid point
        -dx / dy.
    fx : array_like (2-D)
        Contains the original x proportionality factor for the ellipse.
    fy : array_like (2-D)
        Contains the original y proportionality factor for the ellipse.

    Returns
    -------
    theta_max_min : array_like (2-D) 
        Array containing the angle of either the semi-major or semi-
        minor axis of an ellipse [rad].
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
        Contains the minimum radius of a circle centred at (0, dy) and passing
        through +-(te/2, 0) [day].
    s : array_like (2-D)
        Contains the shear factor based on the location of the grid point
        -dx / dy.
    fx : array_like (2-D)
        Contains the original x proportionality factor for the ellipse.
    fy : array_like (2-D)
        Contains the original y proportionality factor for the ellipse.

    Returns
    -------
    a : array_like (2-D) 
        Semi-major axes of the ellipses investigated [day].
    b : array_like (2-D)
        Semi-minor axes of the ellipses investigated [day].
    tilt : array_like (2-D)
        Tilt angles of the ellipses investigated. This is the angle of the 
        semi-major axis w.r.t. the x-axis [deg].
    inclination : array_like (2-D)
        Inclination angles of the ellipses investigated. Inclination is based
        on the ratio of semi-minor to semi-major axis [deg].
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
    # determine the inclination avoiding divide by 0/inf/nan
    inclination = np.nan * np.ones_like(a)
    mask = (~np.isnan(a)) * (~np.isnan(b))
    inclination[mask] = np.arccos(b[mask] / a[mask])
    # determine the tilt
    tilt = np.arctan2(y1, x1) # assuming R1 > R2
    tilt_mask = R2[mask] > R1[mask] # find where above is not true
    tilt[mask] = tilt[mask] + tilt_mask * np.pi / 2 # at ^ locations add np.pi/2
    return a, b, np.rad2deg(tilt), np.rad2deg(inclination)

def ellipse_slope(x, dx, dy, s, fx, fy):
    '''
    This function finds the slopes of the tangents to the ellipse defined by 
    te, dx, dy, fy. This particular ellipse is determined by te, so we get 
    slopes at +-(te/2, 0). To determine the slope at (x, 0), where x is between
    +-te/2 we need to find the slope of the tangent for a concentric ellipse 
    that passes through +-(x, 0).

    Parameters
    ----------
    x : float
        Time value at which to determine the slope [day].
    te : float
        Duration of the eclipse [day].
    dx : array_like (1-D)
        Contains the x coordinate of ellipse centres [day].
    dy : array_like (1-D)
        Contains the y coordinate of ellipse centres [day].
    fy : array_like (2-D)
        Contains the original y proportionality factor for the ellipse.
    
    Returns
    -------
    slope : array_like (2-D)
        Slopes of the ellipses investigated at (x, 0).
    '''
    # shift coordinates to frame centred on the ellipse centre
    X = x - dx[None, :]
    Y = -dy[:, None]
    # get the slopes
    top = -s * fy**2 * Y - fy**2 * X
    bot = (s**2 * fy**2 + fx**2) * Y + s * fy**2 * X
    # prevent divide by 0 error
    slope = np.nan * np.ones_like(s)
    slope[bot!=0] = top[bot!=0]/bot[bot!=0]
    return slope

def slope_to_gradient(slope):
    '''
    This function converts a physical slope to a projected angular gradient.
    This is done as follows -> gradient = np.abs(np.sin(np.arctan2(slope, 1)))

    Parameters
    ----------
    slope : array_like (2-D)
        Slopes of the ellipses investigated at (x, 0).

    Returns
    -------
    gradient : array_like (2-D)
        Gradients of the ellipses investigated at (x, 0).
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
    Gradients can not be filled in this way.

    Parameters
    ----------
    prop : array_like (2-D)
        Contains either the semi-major axis, the semi-minor axis, the tilt or 
        inclination of the investigated ellipses. Note that if the property
        tilt, then the is_tilt parameter should be equal to True, otherwise it
        should be False.
    is_tilt : bool
        This parameter should be False unless prop = tilt. This is because
        the tilt parameter depends on which quadrant it is in (Q2 and Q4 are
        180 - Qx) and is not just a simple reflection [default = False].

    Returns
    -------
    full_prop : array_like (2-D)
        Contains the four quadrant version of the property passed in (this
        results in an array that is 4 times the size of the original). It
        also removes the central column/row.
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

def mask_parameters(a, b, tilt, inclination, gradients, mask_arr, operator, value):
    '''
    This function applies a mask to the semi-major axis, the semi-minor axis,
    the tilt, the inclination and the gradients.

    Parameters
    ----------
    a : array_like (2-D)
        Semi-major axes of the ellipses investigated [day].
    b : array_like (2-D)
        Semi-minor axes of the ellipses investigated [day].
    tilt : array_like (2-D)
        Tilt angles of the ellipses investigated. This is the angle of the 
        Semi-major axis w.r.t. the x-axis [deg].
    inclination : array_like (2-D)
        Inclination angles of the ellipses investigated. Inclination is based
        on the ratio of semi-minor to semi-major axis [deg].
    gradients : array_like (3-D)
        Gradients of the ellipse investigated at each of the measured x values.
        note that the measured x values are w.r.t. the eclipse midpoint i.e.
        they should be between (-te/2, 0) and (te/2, 0).
    mask_arr : array_like (2-D)
        Array on which to evaluate a condition (e.g. a > Rmax, where a is the
        mask_arr, > is the operator, Rmax is the value).
    operator : function
        Should be a numpy comparison function like np.less, np.less_equal,
        np.greater, np.greater_equal, np.equal, np.not_equal...
    value : float
        Value for the comparison made for the mask (e.g. a > Rmax, where a is
        the mask_arr, > is the operator, Rmax is the value).

    Returns
    -------
    a : array_like (2-D)
        Semi-major axes of the ellipses investigated with the masked elements
        converted to nan's [day].
    b : array_like (2-D)
        Semi-minor axes of the ellipses investigated with the masked elements
        converted to nan's [day].
    tilt : array_like (2-D)
        Tilt angles of the ellipses investigated. This is the angle of the 
        semi-major axis w.r.t. the x-axis with the masked elements converted
        to nan's [deg].
    inclination : array_like (2-D)
        Inclination angles of the ellipses investigated. Inclination is based
        on the ratio of semi-minor to semi-major axis with the masked elements
        converted to nan's [deg].
    gradients : array_like (3-D)
        Gradients of the ellipse investigated at each of the measured x values.
        note thtat the measured x values are w.r.t. the eclipse midpoint i.e.
        they should be between (-te/2, 0) and (te/2, 0) with the masked
        elements converted to nan's.
    '''
    # generating mask
    mask = np.zeros_like(mask_arr).astype(np.bool)
    # determine where to apply condition (non-nans)
    value_mask = ~np.isnan(mask_arr)
    # for non nan_mask values we want to evalute the condition 
    mask[value_mask] = operator(mask_arr[value_mask], value)
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
    centres at [-xmax, xmax] (2*nx), [-ymax, ymax] (2*ny).

    Parameters
    ----------
    te : float
        Duration of the eclipse [day].
    xmax : float
        Contains the maximum value of dx [day].
    ymax : float
        Contains the maximum value of dy [day].
    fy : float
        Contains the original y proportionality factor for the ellipse 
        [default = 1].
    measured_xs : list or array_like (1-D)
        Contains the time values where gradients have been measured in the
        light curve [day].
    nx : int
        Number of gridpoints in the x direction [default = 50].
    ny : int
        Number of gridpoints in the y direction [default = 50].

    Returns
    -------
    a : array_like (2-D)
        Semi-major axes of the ellipses investigated [day].
    b : array_like (2-D)
        Semi-minor axes of the ellipses investigated [day].
    tilt : array_like (2-D)
        Tilt angles of the ellipses investigated. This is the angle of the 
        semi-major axis w.r.t. the x-axis [deg].
    inclination : array_like (2-D)
        Inclination angles of the ellipses investigated. Inclination is based
        on the ratio of semi-minor to semi-major axis [deg].
    gradients : array_like (3-D)
        Gradients of the ellipse investigated at each of the measured x values.
        Note that the measured x values are w.r.t. the eclipse midpoint i.e.
        they should be between (-te/2, 0) and (te/2, 0).
    '''
    # creating grids / phase space
    dy = np.linspace(0, ymax, ny)
    dx = np.linspace(0, xmax, nx)
    # determining stretch-parameters
    Fx, Fy = determine_fx(te, dx, dy, fy)
    # determine shear avoid divide by 0 errors
    s = np.nan * np.ones((len(dy), len(dy)))
    s[dy!=0, :] = -dx[None, :] / dy[dy!=0][:, None]
    origin = (dy[:, None] == 0) * (dx[None, :] == 0)
    s[origin] = 0
    #s[np.isnan(s)] = 0 # correct (dx, dy) = (0, 0)
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
    # define gradients array
    gradients = np.zeros((n_measured, 2*ny-1, 2*nx-1))
    # if there are measured points determine the gradients
    if n_measured != 0:
        gradients = np.zeros((n_measured, 2*ny-1, 2*nx-1))
        # get full stretch-parameters
        DX = np.linspace(-xmax, xmax, 2 * nx - 1)
        DY = np.linspace(-ymax, ymax, 2 * ny - 1)
        # get full shear parameters avoiding divide by 0 errors
        S = np.nan * np.ones((len(DY), len(DX)))
        S[DY!=0, :] = -DX[None, :] / DY[DY!=0][:, None]
        ORIGIN = (DY[:, None] == 0) * (DX[None, :] == 0)
        S[ORIGIN]  = 0 #-DX[None, :] / DY[:, None]
        #S[np.isnan(S)] = 0
        #FY = fy * np.ones((2 * ny - 1, 2 * nx - 1))
        FX, FY = determine_fx(te, DX, DY, fy)
        for k, measured_x in enumerate(measured_xs):
            slope = ellipse_slope(measured_x, DX, DY, S, FX, FY)
            gradients[k] = slope_to_gradient(slope)
    return a, b, tilt, inclination, gradients

def full_investigation(te, xmax, ymax, dfy, Rmax, nx=50, ny=50, measured=None):
    '''
    This function investigates the full parameter space (dx, dy, fy) based
    on a grid size (nx, ny, dfy) dependent on the eclipse geometry (te) and
    the maximum size of the disk (Rmax). It also determines the gradients of
    the theoretical disks at the measured times.

    Parameters
    ----------
    te : float
        Duration of the eclipse [day].
    xmax : float
        Contains the maximum value of dx [day].
    ymax : float
        Contains the maximum value of dy [day].
    dfy : float
        Contains the step size in fy for the original y proportionality factor
        for the ellipse.
    Rmax : float
        The maximum size of the disk (used to apply a mask and determine the
        extent of fy) [day].
    nx : int
        Number of gridpoints in the x direction.
    ny : int
        Number of gridpoints in the y direction.
    measured : list of tuples
        Contains the measured gradients and respective times as (time, 
        gradient).

    Return
    -------
    ac : array_like (3-D)
        Cube of semi-major axes of the ellipses investigated [day].
    bc : array_like (3-D)
        Cube of semi-minor axes of the ellipses investigated [day].
    tc : array_like (3-D)
        Cube of tilt angles of the ellipses investigated. This is the angle of
        the semi-major axis w.r.t. the x-axis [deg].
    ic : array_like (3-D)
        Cube of inclination angles of the ellipses investigated. Inclination is
        based on the ratio of semi-minor to semi-major axis [deg].
    gc : array_like (4-D)
        Cube of gradients of the ellipse investigated at each of the measured x
        values. Note thtat the measured x values are w.r.t. the eclipse midpoint
        i.e. they should be between (-te/2, 0) and (te/2, 0).
    '''
    # determine fy extent
    fy_max = 2 * Rmax / te
    fys = np.arange(dfy, fy_max + dfy, dfy)
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
    ac, bc, tc, ic, gc = mask_parameters(ac, bc, tc, ic, gc, ac, np.equal, 0)
    # remove all solutions where ac > Rmax
    ac, bc, tc, ic, gc = mask_parameters(ac, bc, tc, ic, gc, ac, np.greater, Rmax)
    # per measured gradient remove all solutions where gc[k] < measured[k]
    for k, gradient in enumerate(measured_gradients):
        ac, bc, tc, ic, gc = mask_parameters(ac, bc, tc, ic, gc, gc[k], np.less, gradient)
    return ac, bc, tc, ic, gc

def grid_to_parameters(a_cube, tilt_cube, inclination_cube, xmax, ymax):
    '''
    This function extracts all the acceptable (non-NaN) solutions from the 
    provided cubes and converts the grid points to parameter values.

    Parameters
    ----------
    a_cube : array_like (3-D)
        Cube of semi-major axes of the ellipses investigated [day].
    tilt_cube : array_like (3-D)
        Cube of tilt angles of the ellipses investigated. This is the angle of
        the semi-major axis w.r.t. the x-axis [deg].
    inclination_cube : array_like (3-D)
        Cube of inclination angles of the ellipses investigated. Inclination is
        based on the ratio of semi-minor to semi-major axis [deg].
    xmax : float
        Contains the maximum value of dx [day].
    ymax : float
        Contains the maximum value of dy [day].

    Returns
    -------
    disk_radii : array_like (1-D)
        All the possible disk radii [day].
    disk_tilt : array_like (1-D)
        All the possible disk tilts [deg].
    disk_inclination : array_like (1-D)
        All the possible disk inclinations [deg].
    disk_impact_parameters : array_like (1-D)
        All the possible disk impact parameters [day].
    disk_dts : array_like (1-D)
        All the possible x-offsets of the disk centres [day].
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
    provided by the disk_* paramaters.

    Parameters
    ----------
    a_cube : array_like (3-D)
        Cube of semi-major axes of the ellipses investigated [day].
    tilt_cube : array_like (3-D)
        Cube of tilt angles of the ellipses investigated. This is the angle of
        the semi-major axis w.r.t. the x-axis [deg].
    inclination_cube : array_like (3-D)
        Cube of inclination angles of the ellipses investigated. Inclination is
        based on the ratio of semi-minor to semi-major axis [deg].
    xmax : float
        Contains the maximum value of dx [day].
    ymax : float
        Contains the maximum value of dy [day].
    dfy : float
        Contains the step size in fy for the original y proportionality factor
        for the ellipse.
    disk_radius : float
        Actual size of the disk [day].
    disk_tilt : flaot
        Actual tilt of the disk [deg].
    disk_inclination : float
        Actual inclination of the disk [deg].
    disk_impact_parameter : float
        Actual impact parameter (dy) of the disk [day].

    Returns
    -------
    dx : float
        dx value of the closest solution [day].
    dy : float
        dy value of the closest solution [day].
    fy : float
        Original y proportionality factor value of the closest solution.
    best_indices : tuple
        Contains the indices for the closest solution.
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
    from simulate_lightcurve import get_slope_lines
    # start demos
    print('===========================================')
    print('ALL THE METHODS IN SJALOT.PY WILL BE DEMOED')
    print('===========================================')
    print('')
    ### DETERMINE_FX() ###
    print('1. sjalot.determine_fx()')
    print('------------------------')
    print('This function compensates for a stretching in the y direction of')
    print('the smallest circle centred at (0, dy) that passes through points')
    print('(-te/2, 0) and (te/2, 0).')
    # initialise input parameters
    print('  a. initialising input parameters')
    te = 1
    dy = np.array([0.7])
    dx = np.array([0.0])
    Rmin = np.hypot(te/2, dy)
    fx0, fy0 = determine_fx(te, dx, dy, 1)
    fx1, fy1 = determine_fx(te, dx, dy, 2)
    fx2, fy2 = determine_fx(te, dx, dy, 0.85)
    print('     te = %.2f' % te)
    print('     dx[0] = %.2f' % dx[0])
    print('     dy[0] = %.2f' % dy[0])
    print('     Rmin[0] = %.2f' % Rmin[0])
    print('     strech_factors:')
    print('       (fx[0,0], fy[0,0]) = (%.2f, %.2f)' % (fx0[0,0], fy0[0,0]))
    print('       (fx[0,0], fy[0,0]) = (%.2f, %.2f)' % (fx1[0,0], fy1[0,0]))
    print('       (fx[0,0], fy[0,0]) = (%.2f, %.2f)' % (fx2[0,0], fy2[0,0]))
    # list dependencies
    print('  b. demo via:')
    print('     sjalot.shear_ellipse_point()')
    print('       - (dx = 0 --> s = 0)')
    # prepare demo
    print('  c. running sjalot.determine_fx() demo')
    # create figure
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Demo: sjalot.determine_fx()')
    # draw circle
    theta = np.linspace(0, 2*np.pi, 1001)
    xc, yc = shear_ellipse_point(Rmin, np.array([[0]]), fx0, fy0, theta)
    s = np.array([[0]])
    # fy = 2 - prepare axes (full diagram, zoom of ingress, zoom of egress)
    ax1a = plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=2)
    ax1b = plt.subplot2grid((3, 4), (2, 0))
    ax1c = plt.subplot2grid((3, 4), (2, 1))
    ax1 = [ax1a, ax1b, ax1c]
    title = 'fy = %.2f   -->   fx = %.2f' % (fy1[0,0], fx1[0,0])
    # x and y coordinates for (b)ad (fx = 1) and (g)ood ellipses (fx = fx)
    x1b, y1b = shear_ellipse_point(Rmin, s, fx0, fy1, theta)
    x1g, y1g = shear_ellipse_point(Rmin, s, fx1, fy1, theta)
    # plot all circles/ellipses into subplots
    for ax in ax1:
        ax.set_aspect('equal')
        ax.set_xlabel('x [day]')
        ax.set_ylabel('y [day]')
        ax.set_title(title)
        ax.plot(xc[0], yc[0] + dy[0], 'b-', label='smallest circle')
        ax.plot(x1b[0], y1b[0] + dy[0], 'r:', label='fy stretch applied')
        ax.plot(x1g[0], y1g[0] + dy[0], 'g-', label='fx compensation applied')
        ax.plot([-100, -te/2], [0, 0], 'k-o', label='eclipse bounds')
        ax.plot([te/2, 100], [0, 0], 'k-o')
        title=''
    # set axes limits
    ax1a.set_xlim(-2.5, 2.5)
    ax1a.set_ylim(-1, 3)
    ax1a.legend()
    ax1b.set_xlim(-te/2 - 0.2, -te/2 + 0.2)
    ax1b.set_ylim(-0.2, 0.2)
    ax1c.set_xlim(te/2 - 0.2, te/2 + 0.2)
    ax1c.set_ylim(-0.2, 0.2)
    # fy = 0.85 - prepare axes (full diagram, zoom of ingress, zoom of egress)
    ax2a = plt.subplot2grid((3, 4), (0, 2), rowspan=2, colspan=2)
    ax2b = plt.subplot2grid((3, 4), (2, 2))
    ax2c = plt.subplot2grid((3, 4), (2, 3))
    ax2 = [ax2a, ax2b, ax2c]
    title = 'fy = %.2f   -->   fx = %.2f' % (fy2[0,0], fx2[0,0])
    # x and y coordinates for (b)ad (fx = 1) and (g)ood ellipses (fx = fx)
    x2b, y2b = shear_ellipse_point(Rmin, s, fx0, fy2, theta)
    x2g, y2g = shear_ellipse_point(Rmin, s, fx2, fy2, theta)
    # plot all circles/ellipses into subplots
    for ax in ax2:
        ax.set_aspect('equal')
        ax.set_xlabel('x [day]')
        ax.set_ylabel('y [day]')
        ax.set_title(title)
        ax.plot(xc[0], yc[0] + dy[0], 'b-', label='smallest circle')
        ax.plot(x2b[0], y2b[0] + dy[0], 'r:', label='fy compression applied')
        ax.plot(x2g[0], y2g[0] + dy[0], 'g-', label='fx compensation applied')
        ax.plot([-100, -te/2], [0, 0], 'k-o', label='eclipse bounds')
        ax.plot([te/2, 100], [0, 0], 'k-o')
        title=''
    # set axes limits
    ax2a.set_xlim(-2.5, 2.5)
    ax2a.set_ylim(-1, 3)
    ax2a.legend()
    ax2b.set_xlim(-te/2 - 0.2, -te/2 + 0.2)
    ax2b.set_ylim(-0.2, 0.2)
    ax2c.set_xlim(te/2 - 0.2, te/2 + 0.2)
    ax2c.set_ylim(-0.2, 0.2)
    # show
    plt.show()
    print('\n')
    ### SHEAR_ELLIPSE_POINT() ###
    print('2. sjalot.shear_ellipse_point()')
    print('-------------------------------')
    print('This function shears a point on a circle to its location on an')
    print('ellipse based on stretch factors in x and y, and a shear factor.')
    # initialise parameters
    print('  a. intialising parameters:')
    dx3 = np.array([0.3])
    s3 = - dx3[None, :] / dy[:, None]
    s3[np.isnan(s3)] == 0
    fx3, fy3 = determine_fx(te, dx3, dy, 1)
    print('     dx[0] = %.2f' % dx3[0])
    print('     s3[0,0] = %.2f' % s3[0, 0])
    # list dependencies 
    print('  b. demo via:')
    print('     -none-')
    # prepare demo
    print('  c. running sjalot.shear_ellipse_point() demo')
    # find sheared ellipse
    xs, ys = shear_ellipse_point(Rmin, s3, fx3, fy3, theta)
    # prepare figure
    fig = plt.figure(figsize=(8, 9))
    fig.suptitle('Demo: sjalot.shear_ellipse_point() (fy = 1, fx = 1)')
    # prepare subplots - full diagram, zoom of ingress, zoom of egress
    ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=2, colspan=2)
    ax1 = plt.subplot2grid((3, 2), (2, 0))
    ax2 = plt.subplot2grid((3, 2), (2, 1))
    axes = [ax0, ax1, ax2]
    # plot all circles/ellipses into subplots
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('x [day]')
        ax.set_ylabel('y [day]')
        ax.plot(xc[0], yc[0] + dy[0], 'b-', label='smallest circle')
        ax.plot(0, dy[0], 'bo')
        ax.plot(xs[0] + dx3[0], ys[0] + dy[0], 'r-', label='sheared circle')
        ax.plot(dx3[0], dy[0], 'ro')
        ax.plot([-100, -te/2], [0,0], 'k-o', label='eclipse bounds')
        ax.plot([te/2, 100], [0,0], 'k-o')
    # draw shearing arrows
    ax0.arrow(xc[0, 0], yc[0, 0] + dy[0], xs[0, 0] + dx3[0] - xc[0,0], 
              ys[0, 0] - yc[0, 0], color='k', length_includes_head=True,
              width=0.01)
    ax0.arrow(xc[0, 250], yc[0, 250] + dy[0], xs[0, 250] + dx3[0] - xc[0, 250], 
              ys[0, 250] - yc[0, 250], color='k', length_includes_head=True,
              width=0.01)
    ax0.arrow(xc[0, 700], yc[0, 700] + dy[0], xs[0, 700] + dx3[0] - xc[0, 700],
              ys[0, 700] - yc[0, 700], color='k', length_includes_head=True,
              width=0.01)
    ax0.arrow(0, dy[0], dx3[0], 0, color='k', length_includes_head=True, 
              width=0.01)
    # set axes limits
    ax0.set_xlim(-2.5, 2.5)
    ax0.set_ylim(-1, 3)
    ax0.legend()
    ax1.set_xlim(-te/2 - 0.2, -te/2 + 0.2)
    ax1.set_ylim(-0.2, 0.2)
    ax2.set_xlim(te/2 - 0.2, te/2 + 0.2)
    ax2.set_ylim(-0.2, 0.2)
    plt.show()
    print('\n')
    ### THETA_MAX_MIN() ###
    print('3. sjalot.theta_max_min()')
    print('-------------------------')
    print('This function finds the angle of either the semi-major or the')
    print('semi-minor axis of an ellipse (adding pi/2 to the angle gives')
    print('the other semi-axis.')
    # initialise parameters
    print('  a. initialising parameters')
    print('     -none-')
    # list dependencies
    print('  b. demo via:')
    print('     sjalot.shear_ellipse_point()')
    print('      - helper: sjalot.determine_fx()')
    # prepare demo
    print('  c. running sjalot.theta_max_min() demo')
    # determine angle and points
    theta_ab = theta_max_min(s3, fx0, fy0)
    xa, ya = shear_ellipse_point(Rmin, s3, fx3, fy3, theta_ab)
    xb, yb = shear_ellipse_point(Rmin, s3, fx3, fy3, theta_ab + np.pi/2)
    xa2, ya2 = shear_ellipse_point(Rmin, s3, fx3, fy3, theta_ab + np.pi)
    xb2, yb2 = shear_ellipse_point(Rmin, s3, fx3, fy3, theta_ab - np.pi/2)
    # prepare figure
    fig = plt.figure(figsize=(8, 6))
    plt.title('Demo: sjalot.theta_max_min()')
    plt.gca().set_aspect('equal')
    # plot ellipse and points
    plt.plot(xs[0] + dx3[0], ys[0] + dy[0], 'r-', label='ellipse')
    plt.plot(dx3[0], dy[0], 'ro')
    plt.plot(xa[0] + dx3[0], ya[0] + dy[0], 'go', label='theta_max_min()')
    plt.plot(xb[0] + dx3[0], yb[0] + dy[0], 'bo', label='+ $\\frac{\pi}{2}$')
    plt.plot(xa2[0] + dx3[0], ya2[0] + dy[0], 'gd', label='+ $\pi$')
    plt.plot(xb2[0] + dx3[0], yb2[0] + dy[0], 'bd', 
             label='+ $\\frac{3 \pi}{2}$')
    # plot semi-major/minor axes
    plt.plot([xa[0] + dx3[0], xa2[0] + dx3[0]], 
             [ya[0] + dy[0], ya2[0] + dy[0]], 'g:')
    plt.plot([xb[0] + dx3[0], xb2[0] + dx3[0]], 
             [yb[0] + dy[0], yb2[0] + dy[0]], 'b:')
    # plot eclipse lines
    plt.plot([-100, -te/2], [0, 0], 'k-o', label='eclipse bounds')
    plt.plot([te/2, 100], [0, 0], 'k-o')
    # set labels and limits
    plt.xlabel('x [day]')
    plt.ylabel('y [day]')
    plt.legend()
    plt.xlim(-2.5, 2.5)
    plt.ylim(-1, 3)
    plt.show()
    print('\n')
    ### ELLIPSE_PARAMETERS() ###
    print('4. sjalot.ellipse_parameters()')
    print('------------------------------')
    print('This function retrieves all the interesting ellipse parameters')
    print('namely the semi-major axis, the semi-minor axis, the tilt and the')
    print('inclination of the ellipse.')
    # intialise parameters
    print('  a. initialising parameters:')
    print('     -none-')
    # list dependencies
    print('  b. demo via:')
    print('     sjalot.shear_ellipse_point()')
    print('      - helper: sjalot.theta_max_min()')
    print('      - helper: sjalot.determine_fx()')
    # prepare demo
    print('  c. running sjalot.ellipse_parameters() demo')
    a, b, tilt, inclination = ellipse_parameters(Rmin, s3, fx3, fy3)
    print('     > ellipse_parameters()')
    print('     >   semi-major axis = %.2f [day]' % a[0, 0])
    print('     >   semi-minor axis = %.2f [day]' % b[0, 0])
    print('     >   tilt = %.2f [deg]' % tilt[0, 0])
    print('     >   inclination = %.2f [deg]' % inclination[0, 0])
    print('     > measuring values from previous plot')
    R1 = np.hypot(xa[0] - xa2[0], ya[0] - ya2[0])
    R2 = np.hypot(xb[0] - xb2[0], yb[0] - yb2[0])
    ra = np.maximum(R1, R2)
    rb = np.minimum(R1, R2)
    if R1 > R2:
        t = np.arctan2(ya[0], xa[0])
    else:
        t = np.arctan2(yb[0], xb[0])
    t = np.rad2deg(t)
    i = np.rad2deg(np.arccos(rb / ra))
    print('     >   semi-major axis = %.2f [day]' % ra[0])
    print('     >   semi-minor axis = %.2f [day]' % rb[0])
    print('     >   tilt = %.2f [deg]' % t[0])
    print('     >   inclination = %.2f [deg]' % i[0])
    print('     > plotting for confirmation')
    # prepare figure
    fig = plt.figure(figsize=(8, 6))
    plt.title('Demo: sjalot.ellipse_parameters()')
    plt.gca().set_aspect('equal')
    # plot ellipse and points
    plt.plot(xs[0] + dx3[0], ys[0] + dy[0], 'r-', label='ellipse')
    plt.plot(dx3[0], dy[0], 'ro')
    # plot semi-major/minor axes
    plt.plot([xa[0] + dx3[0], xa2[0] + dx3[0]], 
             [ya[0] + dy[0], ya2[0] + dy[0]], 'g:', label='a = %.2f [day]' 
             % ra[0])
    plt.plot([xb[0] + dx3[0], xb2[0] + dx3[0]], 
             [yb[0] + dy[0], yb2[0] + dy[0]], 'b:', label='b = %.2f [day]' 
             % rb[0])
    # plot horizontal line
    plt.gca().axhline(y=dy[0], color='k', ls=':')
    plt.text(dx3[0] + 0.2, dy[0] + 0.05, '$\phi$ = %.2f$^\circ$' % t[0])
    plt.text(dx3[0] - 0.3, dy[0] + 0.45, '$i$ = %.2f$^\circ$' % i[0]) 
    # plot eclipse lines
    plt.plot([-100, -te/2], [0, 0], 'k-o', label='eclipse bounds')
    plt.plot([te/2, 100], [0, 0], 'k-o')
    # set labels and limits
    plt.xlabel('x [day]')
    plt.ylabel('y [day]')
    plt.legend()
    plt.xlim(-2.5, 2.5)
    plt.ylim(-1, 3)
    plt.show()
    print('\n')
    ### ELLIPSE_SLOPE() ###
    print('5. sjalot.ellipse_slope()')
    print('-------------------------')
    print('This function determines the slope of a concentric ellipse that')
    print('intersects with the time, x, provided (x, 0).')
    # intialise parameters
    print('  a. initialising parameters:')
    x00 = -0.40
    x01 = -0.25
    x02 = 0.10
    print('     time, x values:')
    print('       x = %.2f' % x00)
    print('       x = %.2f' % x01)
    print('       x = %.2f' % x02)
    # list dependencies
    print('  b. demo via:')
    print('     sjalot.shear_ellipse_point()')
    print('      - helper: sjalot.determine_fx()')
    print('     helper: simulate_lightcurve.get_slope_lines()')
    # prepare demo
    print('  c. running sjalot.ellipse_slope() demo')
    # determine slopes
    slope00 = ellipse_slope(x00, dx3, dy, s3, fx0, fy0)
    slope01 = ellipse_slope(x01, dx3, dy, s3, fx0, fy0)
    slope02 = ellipse_slope(x02, dx3, dy, s3, fx0, fy0)
    # defining get_slope_lines parameters
    slopes = np.array([slope00[0, 0], slope01[0, 0], slope02[0, 0]])
    slope_times = np.array([x00, x01, x02])
    time = np.linspace(-te/2, te/2, 101)
    lightcurve = np.zeros_like(time)
    slope_lines = get_slope_lines(time, lightcurve, slope_times, slopes, 0.1)
    # draw sub-ellipses
    Rmin00 = np.hypot(x00/fx3[0, 0], dy/fy0[0, 0])
    xs00, ys00 = shear_ellipse_point(Rmin00, s3, fx3, fy3, theta)
    Rmin01 = np.hypot(x01/fx3[0, 0], dy/fy0[0, 0])
    xs01, ys01 = shear_ellipse_point(Rmin01, s3, fx3, fy3, theta)
    Rmin02 = np.hypot(x02/fx3[0, 0], dy/fy0[0, 0])
    xs02, ys02 = shear_ellipse_point(Rmin02, s3, fx3, fy3, theta)
    # prepare figure
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('Demo: sjalot.ellipse_slope()')
    plt.gca().set_aspect('equal')
    # plot ellipses and lines
    plt.plot(xs[0] + dx3[0], ys[0] + dy[0], 'r-', label='ellipse')
    plt.plot(xs00[0] + dx3[0], ys00[0] + dy[0], 'r-')
    plt.plot(xs01[0] + dx3[0], ys01[0] + dy[0], 'r-')
    plt.plot(xs02[0] + dx3[0], ys02[0] + dy[0], 'r-')
    # plot lines and points
    plt.plot(x00, 0, 'go', label='x = %.2f [day]' % x00)
    plt.plot(x01, 0, 'bo', label='x = %.2f [day]' % x01)
    plt.plot(x02, 0, 'yo', label='x = %.2f [day]' % x02)
    for slope_line in slope_lines:
        plt.plot(slope_line[0], slope_line[1], 'k:')
    plt.plot([-100, -te/2], [0, 0], 'k-o', label='eclipse bounds')
    plt.plot([te/2, 100], [0, 0], 'k-o')
    plt.xlim(-1, 1.5)
    plt.ylim(-0.25, 1.75)
    plt.legend()
    plt.show()
    print('\n')
    ### SLOPE_TO_GRADIENTS() ###
    print('6. sjalot.slope_to_gradient()')
    print('-----------------------------')
    print('This function converts the light curve slope determined by the')
    print('previous function to a projected angular gradient. This is done')
    print('by taking the absolute value of the sine of the arctangent of the')
    print('slope. This is discussed in greater detail in Kenworthy & Mamajek')
    print('2015 (https://iopscience.iop.org/article/10.1088/0004-637X/800/2/')
    print('126). [Demo n/a]')
    print('\n')
    ### INVESTIGATE_ELLIPSES() ###
    print('7. sjalot.investigate_ellipses()')
    print('--------------------------------')
    print('This is a helper function that reflects quadrants so that you can')
    print('explore the parameter space more effectively.')
    # initialise parameters
    print('  a. initialising parameters:')
    xmax = 50
    nx = 101
    ymax = 50
    ny = 101
    fy = 1.2
    measured_xs = [-0.4, 0.2]
    print('     xmax = %.2f' % xmax)
    print('     nx = %i' % nx)
    print('     ymax = %.2f' % ymax)
    print('     ny = %i' % dy)
    print('     fy = %.2f' % fy)
    print('     measured_xs:')
    for measured_x in measured_xs:
        print('        x = %.2f' % measured_x)
    # list dependencies
    print('  b. demo via:')
    print('     helper: sjalot.fill_quadrants()')
    print('       - will be compared to version without fill_quadrants')
    print('     helper: sjalot.slope_to_gradient()')
    # prepare demo
    print('  c. running sjalot.investigate_ellipses() demo')
    a, b, t, i, g = investigate_ellipses(te, xmax, ymax, fy, measured_xs, nx, ny)
    DXF = np.linspace(-50, 50, 201)
    DYF = np.linspace(-50, 50, 201)
    RMINF = np.hypot(te/2, DYF)
    SF = np.nan * np.ones((len(DYF), len(DXF)))
    SF[DYF!=0, :] = -DXF[None, :] / DYF[DYF!=0, None]
    ORIGIN = (DYF[:, None] == 0) * (DXF[None, :] == 0)
    SF[ORIGIN] = 0
    FXF, FYF = determine_fx(te, DXF, DYF, fy)
    A, B, T, I = ellipse_parameters(RMINF, SF, FXF, FYF)
    G = []
    for measured_x in measured_xs:
        slope = ellipse_slope(measured_x, DXF, DYF, SF, FXF, FYF)
        gradient = slope_to_gradient(slope)
        G.append(gradient)
    G = np.array(G)
    ie_params = (a, b, t, i, g[0])
    ff_params = (A, B, T, I, G[0])
    lbls = ['semi-major axis', 'semi-minor axis', 'tilt', 'inclination', 
            'gradient']
    print('     > validating sjalot.fill_quadrants()')
    for iep, ffp, lbl in zip(ie_params, ff_params, lbls):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Demo: sjalot.investigate_ellipses() -- %s' % lbl)
        # investigate ellipses
        ax0 = axes[0]
        im0 = ax0.imshow(iep, origin='lower', extent=(-xmax, xmax, -ymax, ymax),
                         cmap=plt.cm.get_cmap('viridis', 11))
        ax0.set_title('sjalot.investigate_ellipses()')
        ax0.set_xlabel('x [day]')
        ax0.set_ylabel('y [day]')
        fig.colorbar(im0, ax=ax0)
        ax1 = axes[1]
        im1 = ax1.imshow(ffp, origin='lower', extent=(-xmax, xmax, -ymax, ymax),
                         cmap=plt.cm.get_cmap('viridis', 11))
        ffp_title = 'sjalot.ellipse_parameters() &\n'
        ffp_title = ffp_title + 'sjalot.ellipse_slopes() --> '
        ffp_title = ffp_title + 'sjalot.slope_to_gradient()'
        ax1.set_title(ffp_title)
        ax1.set_xlabel('x [day]')
        ax1.set_ylabel('y [day]')
        fig.colorbar(im1, ax=ax1)
        diff = np.nansum(np.abs(iep-ffp))
        print('     >   %s total difference: %.8f' % (lbl, diff))
        plt.show()
    print('\n')
    ### FULL_INVESTIGATION() ###
    print('8. sjalot.full_investigation()')
    print('------------------------------')
    print('This function does a full scale version of investigate_ellipses.')
    print('The major difference is that it takes into consideration the')
    print('maximum disk size (30% of the Hill radius), and all the measured')
    print('slopes in the light curve. It outputs a semi-major and semi-minor')
    print('axis, tilt, inclination, and the relevant gradient cubes with the')
    print('coordinates of dx, dy, dfy.')
    # initialise parameters
    print('  a. initialising parameters:')
    Rmax = 50 
    measured = [(-0.4, 0.2), (0.2, 0.3)]
    dfy = 0.5
    print('     Rmax = %.2f [day]' % Rmax)
    print('     measured:')
    for point in measured:
        print('       (time, gradient( = (%.2f, %.2f)' % point)
    print('     dfy = %.2f' % dfy)
    # list dependencies
    print('  b. demo via:')
    print('     sjalot.scroll_plotter()')
    # prepare demo
    print('  c. running sjalot.full_investigation() demo')
    ac, bc, tc, ic, gc = full_investigation(te, xmax, ymax, dfy, Rmax, nx, ny,
                                            measured)
    # define scroll function
    def key_event(e):
        '''
        This function allows one to scroll through the cubes produced above
        '''
        global curr_pos
        nf = data_cube.shape[2]
        if e.key == 'right' and curr_pos < nf:
            curr_pos += 1
        elif e.key == 'left' and curr_pos >= 0:
            curr_pos -= 1
        else:
            return
        curr_pos = curr_pos % data_cube.shape[2]
        ax.cla()
        ax.imshow(data_cube[:, :, curr_pos], origin='lower', extent=(-xmax, 
                  xmax, -ymax, ymax), cmap=plt.cm.get_cmap('viridis', 11))
        ax.set_title('fy = %.2f' % fys[curr_pos])
        ax.set_xlabel('x [day]')
        ax.set_ylabel('y [day]')
        fig.canvas.draw()
    # cycle through parameters
    data_cubes = (ac, bc, tc, ic, gc[0], gc[1])
    lbls = ['semi-major axis', 'semi-minor axis', 'tilt', 'inclination',
            'gradient[0]', 'gradient[1]']
    for data_cube, lbl in zip(data_cubes, lbls):
        curr_pos = 40
        fys = np.arange(0, data_cube.shape[2], dfy) 
        fig = plt.figure()
        fig.suptitle('Demo: sjalot.full_investigation() --> %s' % lbl)
        fig.canvas.mpl_connect('key_press_event', key_event)
        ax = fig.add_subplot(111)
        ax.set_title('fy = %.2f' % fys[curr_pos])
        im = ax.imshow(data_cube[:, :, curr_pos], origin='lower', extent=(-xmax, 
                  xmax, -ymax, ymax), cmap=plt.cm.get_cmap('viridis', 11))
        ax.set_xlabel('x [day]')
        ax.set_ylabel('y [day]')
        plt.colorbar(im, ax=ax)
        plt.show()
    print('\n')
    ### GRID_TO_PARAMETERS() ###
    print('8. sjalot.grid_to_parameters()')
    print('------------------------------')
    print('This function takes the data cubes from full_investigation() and')
    print('extracts the relevant disk features, namely the disk radius (which')
    print('is equivalent to the semi-minor axis), the tilt, the inclination,')
    print('the impact parameter (which is the dy grid point), and the time of')
    print('closest approach -dt- (which is the dx grid point), for all points')
    print('which are not NaNs. [Demo n/a]')
    print('\n')
    ### GET_CLOSEST_SOLUTION() ###
    print('9. sjalot.get_closest_solution()')
    print('--------------------------------')
    print('This function takes in the data cubes outputted by the function')
    print('full_investigation() and the values of a simulated disk and finds')
    print('the closest grid point to that solution.')
    # intialise parameters
    print('  a. initialising parameters:')
    te = 10.16
    xmax = 10
    ymax = 10
    dfy = 0.5
    Rmax = 100
    nx = 50
    ny = 50
    time = np.linspace(-15, 15, 301)
    planet_radius = 0
    inner_radii = [1e-16]
    outer_radii = [80]
    opacities = [1.]
    inclination = 83
    tilt = 35
    dt = 1.08
    u = 0.00
    vt = 3.6
    print('     te = %.2f' % te)
    print('     xmax = %.2f' % xmax)
    print('     ymax = %.2f' % ymax)
    print('\n')
    print('=============================================')
    print('ALL THE METHODS IN SJALOT.PY HAVE BEEN DEMOED')
    print('=============================================')
    print('')
