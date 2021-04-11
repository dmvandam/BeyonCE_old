"""
This module is used to find find possible solutions to a light curve based on
a couple of inputs, namely the width of the eclipse (te), a grid of occulter 
centres (dx, dy), initial stretch factor of the occulter (fy), and an input 
list containing a time/location w.r.t. eclipse and the corresponding measured
gradient (G) in the form G = np.abs(np.sin(np.arctan2(slope, 1)))

We want the grid shape to be (len(dy), len(dx))
"""

import numpy as np
from tqdm import tqdm
from time import time as time_now
from scipy.optimize import curve_fit
from simulate_lightcurve import simulate_lightcurve, divide_rings

###############################################################################
############################## ARRAY EQUATIONS ################################
###############################################################################

def determine_fx(te, dy, fy):
    '''
    This function is used to determine the proportionality factor in x given a
    proportionality factor in y. The idea is that when you find the smallest
    circle centred at (0, dy) that passes through the eclipse points ±(te/2, 0)
    and we then either stretch or compress the ellipse in the y direction with 
    fy, we are then forced to compensate by a compression or stretching of the
    ellipse in the x direction (fx)

    Parameters
    ----------
    te : float
        width of the eclipse [time or space]
    dy : array (1-D)
        contains the y coordinate of ellipse centres
    fy : array (2-D)
        contains the original y proportionality factor for the ellipse

    Returns
    -------
    fx : array (2-D)
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
    Finds the x and y coordinate for a sheared ellipse based on the angle in
    parameteric form.
    
    First we find the smallest circle, centred at (0, dy) that passes through
    the eclipse points ±(te/2, 0), then we transform it to an ellipse by using
    the proportionality factors fy and the determined fx. Finally we shear the
    ellipse to its final form.
    
    The cartesian coordinates for the given angle is returned for each grid
    point calculated.

    Parameters
    ----------
    te : float
        width of the eclipse [time or space]
    dx : array (1-D)
        contains the x coordinate of ellipse centres
    dy : array (1-D)
        contains the y coordinate of ellipse centres
    fy : array (2-D)
        contains the original y proportionality factor for the ellipse
    theta : float
        the angle of the point in the circle that will be transformed
        to the sheared circle [rad].

    Returns
    -------
    xp : array_like (2-D)
        the x-coordinate of the input point in the sheared circle.
    yp : array_like (2-D)
        the y-coordinate of the input point in the sheared circle.
    '''
    # determine the x and y coordinates of the ellipse
    y = fy * Rmin[:,None]*np.sin(theta)
    x = fx * Rmin[:,None]*np.cos(theta)
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
    te : float
        width of the eclipse [time or space]
    dx : array (1-D)
        contains the x coordinate of ellipse centres
    dy : array (1-D)
        contains the y coordinate of ellipse centres
    fy : array (2-D)
        contains the original y proportionality factor for the ellipse

    Returns
    -------
    theta_max_min : array_like (2-D) [rad]
        Array containing the angle of either the semi-major or semi-
        minor axis of an ellipse.
    '''
    # determine the parametric angle of either the semi-major or minor axis
    top = 2 * fy * fx * s
    bot = (s**2 + 1) * fy**2 - fx**2
    theta_max_min = 0.5 * np.arctan2(top, bot)
    return theta_max_min

def ellipse_parameters(Rmin, s, fx, fy):
    '''
    Finds the semi-major axis, a, semi-minor axis, b, the tilt and 
    the inclination of the smallest ellipse that is centred at 
    (dx, dy) w.r.t. the centre of the eclipse with duration te.

    Parameters
    ----------
    te : float
        width of the eclipse [time or space]
    dx : array (1-D)
        contains the x coordinate of ellipse centres
    dy : array (1-D)
        contains the y coordinate of ellipse centres
    fy : array (2-D)
        contains the original y proportionality factor for the ellipse

    Returns
    -------
    a : array_like (2-D)
        semi-major axes of the ellipses investigated
    b : array_like (2-D)
        semi-minor axes of the ellipses investigated
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
    Finds the slopes of the tangents to the ellipse defined by te, dx, dy, fy.
    This ellipse is determined by te, so we get slopes at ±(te/2, 0). To
    determine the slope at (x, 0), where x is between ±te/2 we need to find the
    slope of the tangent for a concentric ellipse that passes through ±(x, 0)

    Parameters
    ----------
    x : float
        value at which to determine the slope
    te : float
        width of the eclipse [time or space]
    dx : array (1-D)
        contains the x coordinate of ellipse centres
    dy : array (1-D)
        contains the y coordinate of ellipse centres
    fy : array (2-D)
        contains the original y proportionality factor for the ellipse
    
    Returns
    -------
    slope : array (2-D)
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
    This is done as follows - gradient = np.abs(np.sin(np.arctan2(slope, 1)))

    Parameters
    ----------
    slope : array (2-D)
        slopes of the ellipses investigated at (x, 0)

    Returns
    -------
    gradient : array (2-D)
        gradients of the ellipses investigated at (x, 0)
    '''
    gradient = np.abs(np.sin(np.arctan2(slope, 1)))
    return gradient

def investigate_ellipses(te, xmax, ymax, fy=1, measured_xs=[], nx=50, ny=50):
    '''
    Investigates the full parameter space for an eclipse of duration te with
    centres at [-xmax, xmax] (2*nx), [-ymax, ymax] (2*ny)

    Parameters
    ----------
    te : float
        width of the eclipse [time or space]
    xmax : float
        contains the maximum value of dx
    ymax : float
        contains the maximum value of dy
    fy : float
        contains the original y proportionality factor for the ellipse 
        [default=1]
    nx : int
        number of gridpoints in the x direction
    ny : int
        number of gridpoints in the y direction

    Returns
    -------
    a : array_like (2-D)
        semi-major axes of the ellipses investigated
    b : array_like (2-D)
        semi-minor axes of the ellipses investigated
    tilt : array_like (2-D)
        tilt angles of the ellipses investigated. This is the angle of the 
        semi-major axis w.r.t. the x-axis. [deg]
    inclination : array_like (2-D)
        inclination angles of the ellipses investigated. Inclination is based
        on the ratio of semi-minor to semi-major axis. [deg]
    gradients : array_like (3-D)
        gradients of the ellipse investigated at each of the measured x values.
        note thtat the measured x values are w.r.t. the eclipse midpoint i.e.
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
    n_measured = len(measured_xs)
    # if no measured points then gradients = None
    if n_measured == 0:
        gradients = None
    # otherwise create an array and fill with the slopes
    else:
        gradients = np.zeros((n_measured, 2*ny-1, 2*nx-1))
        # get full sub-parameters
        DX = np.linspace(-xmax, xmax, 2*nx-1)
        DY = np.linspace(-ymax, ymax, 2*ny-1)
        S  = -DX[None, :] / DY[:, None]
        S[np.isnan(S)] = 0
        FY = fy *np.ones((2*ny-1, 2*nx-1))
        FX = determine_fx(te, DY, FY)
        for k, measured_x in enumerate(measured_xs):
            slope = ellipse_slope(measured_x, DX, DY, S, FX, FY)
            gradients[k] = slope_to_gradient(slope)
    return a, b, tilt, inclination, gradients

def fill_quadrants(prop, is_tilt=False):
    '''
    This function does the appropriate reflection symmetry and works for the
    semi-major axis, the semi-minor axis and the inclination. Tilt must be
    done differently and the gradients must be calculated on the four quadrant
    investigation

    Parameters
    ----------
    prop : array_like (2-D)
        contains either the semi-major axis, the semi-minor axis, the tilt or 
        inclination of the investigated ellipses. Note that if the property
        tilt, then the is_tilt parameter should be equal to True, otherwise it
        should be False
    is_tilt : bool
        this parameter should be False unless prop == tilt. This is because
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
    the tilt, the inclination and the gradients

    Parameters
    ----------
    a : array_like (2-D)
        semi-major axes of the ellipses investigated
    b : array_like (2-D)
        semi-minor axes of the ellipses investigated
    tilt : array_like (2-D)
        tilt angles of the ellipses investigated. This is the angle of the 
        semi-major axis w.r.t. the x-axis. [deg]
    inclination : array_like (2-D)
        inclination angles of the ellipses investigated. Inclination is based
        on the ratio of semi-minor to semi-major axis. [deg]
    gradients : array_like (3-D)
        gradients of the ellipse investigated at each of the measured x values.
        note thtat the measured x values are w.r.t. the eclipse midpoint i.e.
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
    a[mask] = np.nan
    b[mask] = np.nan
    tilt[mask] = np.nan
    inclination[mask] = np.nan
    for k in range(len(gradients)):
        gradients[k][mask] = np.nan
    return a, b, tilt, inclination, gradients

def full_investigation(te, xmax, ymax, dfy, Rmax, nx=50, ny=50, measured=[]):
    '''
    This function investigates the full parameter space (dx, dy, fy) based
    on a grid size (nx, ny, dfy) dependent on the eclipse geometry (te) and
    the maximum size of the disk (Rmax). It also determines the gradients of
    the theoretical disks at the measured times

    Parameters
    ----------
    te : float
        width of the eclipse [time or space]
    xmax : float
        contains the maximum value of dx
    ymax : float
        contains the maximum value of dy
    dfy : float
        contains the step size in fy for the original y proportionality factor
        for the ellipse
    Rmax : float
        the maximum size of the disk (used to apply a mask and determine the
        extent of fy)
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
        cube of semi-major axes of the ellipses investigated
    bc : array_like (3-D)
        cube of semi-minor axes of the ellipses investigated
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
    # start with fy = 1 (smallest possible disk)
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

def extract_all_solutions(a_cube, tilt_cube, inclination_cube, xmax, ymax):
    '''
    This function extracts all the acceptable (non-NaN) solutions from the 
    provided cubes

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

    Returns
    -------
    disk_radii : array_like (1-D)
        all the possible disk radii
    disk_tilt : array_like (1-D)
        all the possible disk tilts [deg]
    disk_inclination : array_like (1-D)
        all the possible disk inclinations [deg]
    disk_impact_parameters : array_like (1-D)
        all the possible disk impact parameters [R*]
    disk_dts : array_like (1-D)
        all the possible x-offsets of the disk centres [R*]
    '''
    # get cube shape
    ny, nx, nf = a.shape
    # set-up dx and dy grids
    yy, xx = np.mgrid[:ny, :nx]
    # normalise grids from -1 to +1
    yy = 2 * (yy / (ny-1) - 0.5)
    xx = 2 * (xx / (nx-1) - 0.5)
    # scale to -ymax to +ymax and -xmax to +xmax
    yy = ymax * yy
    xx = xmax * xx
    # mask out the bad values
    mask = ~np.isnan(a)
    # apply masks
    disk_radii = a[mask]
    disk_tilts = tilt[mask]
    disk_inclinations = inclination[mask]
    disk_impact_parameters = yy[mask]
    disk_dts = xx[mask]
    return (disk_radii, disk_tilt, disk_inclinations, disk_impact_parameters,
            disk_dts)

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
        dx value of the closest solution
    dy : float
        dy value of the closest solution (i.e. impact parameter)
    fy : float
        stretch factor value of the closest solution
    '''
    return None
