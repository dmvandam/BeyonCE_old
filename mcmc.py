'''
This module is used to relax the hard ring boundaries determined by the ring
fitter module. It does this by setting up an MCMC model of the ring system,
selecting an appropriate prior and then using the emcee package developed by
Foreman-Macket el al. 2013 (https://iopscience.iop.org/article/10.1086/670067).
It further depends on the simulate_lightcurve.simulate_lightcurve(), which in
turn depends on the pyPplusS pacakage developed by Edan Rein & Aviv Ofir 2019.

simulate_lightcurve.simulate_lightcurve has been modified here to the form
required by emcee, namely:
    1. Parameters tied into one tuple (aptly named P)

    2. Inclination and tilt in P will be in radians instead of degrees (this
       is because walkers move too slowly in degrees

The main tools of this module for MCMC setup include:
    i.      setup the ringsystem model
    ii.     setup the ringsystem prior
    iii.    setup the initial walker position
    iv.     determining the log likelihood
    v.      determining the log probability
    vi.     actually running the MCMC methods

The main plotting tools include:
    i.      plotting the walkers
    ii.     plotting a corner plot
    iii.    plotting samples from the MCMC chain vs the data
    iv.     plotting given models vs the data

The main calculation tools include:
    i.      extracting certain walkers (based on preset conditions)
    ii.     determining the statistics of the walkers

A final utility tool:
    i.      prints out the parameters in P in a human-readable "pretty" way

Finally, if the module is run as a script (instead of imported from elsewhere),
a tutorial of each of the functions will be given (i.e. a description will be
printed along with relevant plots to show the working of the functions in this
module).
'''


###############################################################################
########################### IMPORT MAIN MODULES ###############################
###############################################################################

# calculations
import numpy as np
import emcee
# plotting
import corner
from tqdm import tqdm
import matplotlib.pyplot as plt


###############################################################################
############################### MCMC FUNCTIONS ################################
###############################################################################

def ringsystem_model(P, time)
    '''
    This function is a ring system model that relies on the light curves
    simulated by simulate_lightcurve.simulate_lightcurve(), with three major
    differences. The first is how the arguments are set-up, which is necessary
    for the emcee module, the second is that the inclination and tilt are in
    RADIANS instead of DEGREES. This is to allow the walkers to explore a
    larger parameter space (0.1 rad >> 0.1 deg). The final third difference is
    that the rings are expressed in ring edges, as the assumption is that all
    rings are touching.

    Parameters
    ----------
    P : tuple (2 * num_rings + 8) 
        Contains the internal parameters (re-order for multi ring 
        compatibility).
            redges  - 0:n+1 : inner radii of the rings [R*].
            tau  - n+1:3n+1 : opacities of the rings [-].
            rp : radius of the planet [R*].
            inc : inclination of the disk [rad].
            tilt : tilt of the disk [rad].
            b : impact parameter of the disk [R*].
            dt : time of closest approach [day].
            u : linear limb-darkening parameter [-].
            vt : transverse velocity of the disk [R*/day].
    time : array_like (1-D)
        Time data where ring system light curve should be simulated [day].
    
    Returns
    -------
    lightcurve : array_like (1-D)
        Flux data for the lightcurve given the input parameters.
    '''
    # determine the number of rings
    num_params = len(P)
    num_rings  = (num_params - 8) // 2
    # unpack non-dynamic portion of the parameter tuple P
    rp, inc, tilt, b, dt, u, vt = P[:-7]
    # unpack dynamic portion of the parameter tuple P
    rin  = P[:num_rings]
    rout = P[1:num_rings+1]
    tau  = P[num_rings+1:2*num_rings+1]
    # convert inclination and tilt degrees
    inc_deg = np.rad2deg(inc)
    tilt_deg = np.rad2deg(tilt)
    # get the light curve values
    lightcurve, _ = simulate_lightcurve(time, rp, rin, rout, tau, inc_deg, 
                                        tilt_deg, b, dt, u, vt)
    return lightcurve

def disk_prior(P, redge_bounds, tau_bounds=(0, 1), rp_bounds=(0, 1),
               inc_bounds=(0, np.pi/2), tilt_bounds=(-np.pi, np.pi),
               u_bounds=(0, 1), vt_bounds=(0, 20)):
    '''
    This function determines the bounds of all the input parameters.
    
    Parameters
    ----------
    P : tuple (3 * num_rings + 7) 
        Contains the internal parameters (re-order for multi ring 
        compatibility).
            redges  - 0:n+1 : inner radii of the rings [R*].
            tau  - n+1:2n+1 : opacities of the rings [-].
            rp : radius of the planet [R*].
            inc : inclination of the disk [rad].
            tilt : tilt of the disk [rad].
            b : impact parameter of the disk [R*].
            dt : time of closest approach [day].
            u : linear limb-darkening parameter [-].
            vt : transverse velocity of the disk [R*/day].
    redge_bounds : array_like (2-D)
        Contains the ringlet bounds for the ring edge radii of the modelled 
        ring system. Note that the bounds should be determined by the ringlet
        bounds, which in turn are defined by the size of the disk and the
        number of ringlets used to make the initial model [R*].
    tau_bounds : array_like (2-D) or tuple
        Contains the lower and upper bounds for the opacities. This could be
        for each ring or for the whole system [default = (0, 1)].
    rp_bounds : tuple
        Contains the lower and upper bounds for the radius of the planet 
        [default = (0, 1) R*].
    inc_bounds : tuple
        Contains the lower and upper bounds for the inclination of the ring
        system [default = (0, np.pi/2) rad].
    tilt_bounds : tuple
        Contains the lower and upper bounds for the tilt of the ring system
        [default = (-np.pi, np.pi) rad].
    u_bounds : tuple
        Contains the limb-darkening parameter lower and upper bounds for the
        star [default = (0, 1)].
    vt_bounds : tuple
        Contains the transverse velocity lower and upper bounds for the ring
        system in transity [default = (0, 20) R*/day].

    Returns
    -------
    prior : float
        Either 0 if possible solution or -np.inf if parameters stored in P
        are considered unacceptable.
    '''
    # determine the number of rings
    num_params = len(P)
    num_rings  = (num_params - 8) // 2
    # unpack non-dynamic portion of the parameter tuple P
    rp, inc, tilt, b, dt, u, vt = P[:-7]
    # unpack dynamic portion of the parameter tuple P
    redges  = P[:num_rings+1]
    tau  = P[num_rings+1:2*num_rings+1]
    def within_bounds(parameter, parameter_bounds):
        '''
        This helper function determines whether the parameter is bound by
        the parameter bounds
        
        Parameters
        ----------
        parameter : float
            Parameter value to be checked.
        parameter_bounds : tuple
            Boundaries of the parameter.

        Returns
        -------
        bounded : bool
            Whether or not the parameter is between the parameter_bounds
        '''
        bounded = parameter_bounds[0] <= parameter <= parameter_bounds[1])
        return bounded
    # ensure that all ring opacities are between 0 and 1
    for t in tau:
        if not within_bounds(t, tau_bounds):
            return -np.inf
    # ensure that the inclination (rad) is between 0 and pi/2
    if not within_bounds(inc, inc_bounds):
        return -np.inf
    # ensure that the tilt (rad) is between -pi and pi
    if not within_bounds(tilt, tilt_bounds):
        return -np.inf
    # ensure that the planet is not larger than the star
    if not within_bounds(rp, rp_bounds):
        return -np.inf
    # ensure that the disk transits the star
    disk_height = np.abs(b) - np.abs(rout[-1] * np.sin(tilt))
    if not within_bounds(disk_height, (0, 1)):
        return -np.inf
    # ensure that linear limb-darkening parameter is between 0 and 1
    if not within_bounds(u, u_bounds):
        return -np.inf
    # ensure that transverse velocity is within a given range
    if not within_bounds(vt, vt_bounds):
        return -np.inf
    ### ensure that the inner and outer radii can only shift +- one ringlet
    ### width redge_bounds = [x,2] array containing lower and upper bounds for
    ### rin/rout
    # for inner radii find the closest two ring bounds and prevent cross over
    for r, rb in zip(redges, redge_bounds):
        if not (rb[0] <= r <= rb[1])
            return -np.inf
    # if all conditions are met, then the parameters are allowed
    return 0.

        
###############################################################################
############################ P0 SET UP FUNCTIONS ##############################
###############################################################################

def get_normal(lower_limit, upper_limit, num):
    '''
    This function creates a normal distribution for a parameter, which is more 
    or less bounded between the lower and upper limit. It assumes that the mean
    is between the two limits provided.
    
    Parameterse
    ----------
    lower_limit : float
        Lower limit of the normal distribution.
    upper_limit : float
        Upper limit of the normal distribution.
    num : int
        Number of values to draw from the normal distribution.
        
    Returns
    -------
    parameter : array
        Contains a normal distribution defined by the lower and upper limits.
    '''
    mean  = 0.5 * (lower_limit + upper_limit)
    sigma = (upper_limit - lower_limit) / 6
    parameter = np.random.normal(mean, sigma, num)
    return parameter

def bounded_p0(ndim, nw, bounds, max_trials=20):
    '''
    This function creates a initial walker prior based on the bounds that are
    provided. The walkers will be distributed normally between the bounds
    provided.
    
    Parameters
    ----------
    ndim : int
        Number of parameters.
    nw : int
        Number of walkers.
    bounds : list of tuples
        Contains the upper and lower bound for each parameter.
    max_trials : int
        This is the maximum number of times 
        
    Returns
    -------
    p0 : array
        Contains the initial value for each of the walkers (nw x ndim).
    '''
    # set up the prior
    p0 = np.zeros((0,nw))
    for x in range(ndim):
        # get parameter distribution
        lower_bound, upper_bound = bounds[x]
        p = get_normal(lower_bound, upper_bound, nw)
        # check that values are bound and redraw values that are unbounded
        num_trials = 0
        while np.sum((p < lower_bound) * (p > upper_bound)) != 0:
            # select and redraw values beyond lower boundary
            mask_lower = p < lower_bound
            num_lower  = np.sum(mask_lower)
            p[mask_lower] = get_normal(lower_bound, upper_bound, num_lower)
            # select and redraw values beyond upper boundary
            mask_upper = p > upper_bound
            num_upper  = np.sum(mask_upper)
            p[mask_upper] = get_normal(lower_bound, upper_bound, num_upper)
            # increment the number of trials
            num_trials += 1
            # if max trials exceeded then manually set lower and upper values
            # to the corresponding bounds
            if num_trials == max_trials:
                p[p < lower_bound] = lower_bound
                p[p < upper_bound] = upper_bound
                break
        # add to stack
        p0 = np.vstack((p0, p))
    return p0.T

def ball_p0(P, nw, size, bounds):
    '''
    This function creates a gaussian ball centred on P with a given spread,
    and ensures that none of the parameters are outside of parameter space.

    Parameters
    ----------
    P : list, tuple, array of floats
        contains model parameters
    nw : int
        number of walkers
    size : float
        size of the gaussian ball
    bounds : list of tuples
        contains the upper and lower bound for each parameter

    Returns
    -------
    p0 : array
        contains the initial value for each of the walkers (nw x ndim)
    '''
    ndim = len(P)
    p0 = [np.array(P) + size * np.random.randn(ndim) for i in range(nw)]
    p0 = np.array(p0)
    for x in range(ndim):
        lower_bound, upper_bound = bounds[x]
        p0[:, x][p0[:, x] < lower_bound] = lower_bound
        p0[:, x][p0[:, x] > upper_bound] = upper_bound
    return p0


############################
#%% LIKELIHOOD FUNCTIONS %%#
############################

def lnlike(P, time, flux, error, model):
    '''
    This function returns the natural logarithm of the likelihood function of
    the input model with parameters P, given a time, flux and error.

    Parameters
    ----------
    P : tuple, list, array of float
        Contains the model parameters.
    time : array of float
        Contains time data for the light curve.
    flux : array of float
        Contains flux data for the light curve.
    error : array of float
        Contains error data for the light curve.
    model : function
        Contains the model to be tested.

    Returns
    -------
    like : float
        The natural logarithm of the likelihood function.
    '''
    like = -0.5 * np.sum(((flux - model(P, time))/error)**2 + np.log(error**2))
    return like

def lnprob(P, time, flux, error, model, model_prior, prior_args=None):
    '''
    This function returns the natural logarithm of the probability of the
    likelihood function given the input parameters.

    Parameters
    ----------
    P : tuple, list, array of floats
        Contains the model parameters.
    time : array of float
        Contains time data for the light curve.
    flux : array of float
        Contains flux data for the light curve.
    error : array of float
        Contains error data for the light curve.
    model : function
        Model for the light curve.
    model_prior : function
        Prior to calculate probability.
    prior_args : tuple
        Contains all the values for the arguments of the prior.

    Returns
    -------
    prob : float
        The natural logarithm of the probability of the model.
    '''
    prior = model_prior(P, *prior_args)
    if np.isfinite(prior):
        prob = prior + lnlike(P, time, flux, error, model)
    else:
        prob = - np.inf
    return prob


###############################################################################
############################### Plot Functions ################################
###############################################################################

def plot_hist(samples, lbls=None, ncols=2, bins=20, savename='test.png'):
    '''
    this function plots a histogram of the samples inserted
        
    Parameters
    ----------
    samples : array
        Samples of model parameters. 
    lbls : list of str
        Names for all the parameters.
    ncols : int
        Number of columns to display the subplots in [default = 2].
    bins : int
        Number of bins for the histogram [default = 20].
    savename : str
        name of the saved plot
    
    Returns
    -------
    matplotlib.figure()
    '''
    # setting up
    _, ndim = samples.shape
    if isinstance(lbls, type(None)):
       lbls = [None] * ndim
    nrows = int(ndim / ncols) + int(ndim % ncols != 0)
    # create the figure
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    for i in range(nrows):
        for j in range(ncols):
            ind = ncols * i + j
            if ind < ndim:
                ax[i,j].hist(samples[:, ind], bins=bins)
                ax[i,j].set_title(lbls[ind])
    plt.show()
    return None

def plot_walkers(sampler, cut=0, lbls=None, savename='test.png'):
    '''
    this function plots how the walkers move through the parameter space

    Parameters
    ----------
    sampler : EnsembleSampler
        MCMC object containing all the parameters
    cut : int
        number of links to remove (burn-in period)
    lbls : list of str
        lbls for the parameters
    savename : str
        name of the saved plot

    Returns
    -------
    matplotlib.figure()
    '''
    # extracting samples
    try:
        # ensemble object
        samples = sampler.get_chain()
    except:
        # numpy array
        samples = sampler
    # number of steps, walkers and dimensions
    ns, nw, ndim = samples.shape
    # fix labels if necessary
    if isinstance(lbls, type(None)):
       lbls = [None] * ndim
    # plotting
    fig, ax = plt.subplots(ndim, figsize=(14, ndim * 4), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    ax[0].set_title('%i Walkers (Burn-in = %i)' % (nw, cut), fontsize=24)
    for k in range(ndim):
        ax[k].tick_params(labelsize=18)
        ax[k].plot(samples[cut:, :, k], 'k', alpha=0.3)
        ax[k].set_xlim(0, ns - cut)
        ax[k].set_ylabel(lbls[k], fontsize=24)
        ax[k].yaxis.set_label_coords(-0.07, 0.5)
    ax[-1].set_xlabel('Step Number', fontsize=20)
    fig.savefig(savename)
    plt.show()
    return None

def plot_triangle(sampler, cut=0, lbls=None, bounds=None, savename='test.png'):
    '''
    this function creates a corner plot

    Parameters
    ----------
    sampler : EnsembleSampler
        MCMC object containing all the parameters
    cut : int
        number of links to remove (burn-in period)
    lbls : list of str
        lbls for the parameters
    bounds : list of tuples
        bounds for the histograms in the corner plot
    savename : str
        name of the saved plot

    Returns
    -------
    matplotlib.figure()
    '''
    # extracting samples
    try:
        # ensemble object
        samples = sampler.get_chain(discard=cut, flat=True)
    except:
        # numpy array
        _, _, ndim = sampler.shape
        samples = sampler[cut:, :, :].reshape((-1, ndim))
    # get dimension
    _, ndim = samples.shape
    # fix labels if necessary
    if isinstance(lbls, type(None)):
       lbls = [None] * ndim
    # plotting
    fig = corner.corner(samples, labels=lbls, figsize=(14, ndim * 4),
                        range=bounds)
    fig.savefig(savename)
    plt.show()
    return None

def plot_samples(time, flux, error, model_list, sampler_list, lbls=None, 
                 cuts=0, num=100, plot_lims=None, residual_lims=None, 
                 savename='test.png', alpha=0.1, best_fit=False, dt=0):
    '''
    this function plots various of the solutions found by the MCMC sampling

    Parameters
    ----------
    time : array of float
        contains time data for the light curve
    flux : array of float
        contains flux data for the light curve
    error : array of float
        contains error data for the light curve
    model_list : list of functions
        list of models for the light curve
    sampler_list : list of EnsembleSampler
        list of MCMC objects containing all the parameters
    lbls : list of str
        list containing names of the models / samplers
    cuts : int or list of ints
        number of links to remove (burn-in period), if it is an integer it is
        applied to all samplers
    num : int
        number of models to plot
    plot_lims : tuple
        bounds of the model subplot
    residual_lims : tuple
        bounds of the residual subplot
    savename : str
        name of the saved plot
    alpha : float
        transparency of the model lines [default = 0.1]
    best_fit : bool
        if true, plot the best fit solution [default = False]
    dt : int
        number of days to shift the xlabel by [default = 0]

    Returns
    -------
    plotted_samples : list of arrays
        the model parameters for the lines plotted separated per model/sampler
    '''
    # check whether or not cuts is an iterable
    try:
        test = cuts[0]
    except:
        cuts = cuts * np.ones(len(sampler_list)).astype(np.int)
    # rest of the function
    colors = 2 * ['C1','C2','C3','C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C0']
    plotted_samples = []
    # set up figure
    fig = plt.figure(figsize=(13, 10))
    # ax0 is the flux plot
    ax0 = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=3)
    ax0.set_ylabel('Normalised Flux [-]', fontsize=16)
    ax0.errorbar(time, flux, yerr=error, fmt='o', color='k', label='data')
    ax0.legend(fontsize=14)
    ax0.tick_params(axis='both', labelsize=14)
    ax0.set_ylim(plot_lims)
    # ax1 is the residual plot
    ax1 = plt.subplot2grid((4, 1), (3, 0), colspan=1, rowspan=2, sharex=ax0)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.axhline(y=0, color='k', ls=':')
    ax1.set_ylabel('Residuals [-]', fontsize=16)
    ax1.set_xlabel('Time [BJD - %i]' % (2454833 + dt), fontsize=16)
    for l, sampler, model, c, cut in zip(lbls, sampler_list, model_list, 
                                         colors, cuts):
        # extract samples
        try:
            # ensemble object
            flat_samples = sampler.get_chain(discard=cut, flat=True)
        except:
            # numpy array
            _, _, ndim = sampler.shape
            flat_samples = sampler[cut:, :, :].reshape((-1, ndim))
        # select random samples
        inds = np.random.randint(len(flat_samples), size=num)
        for ind in tqdm(inds):
            # prevent models that do not change
            delta = 0
            while delta < 1e-1:
                sample = flat_samples[ind]
                model_flux = model(sample, time)
                delta = np.sum(np.abs(model_flux[1:] - model_flux[:-1]))
                ind = np.random.randint(len(flat_samples), size=1)[0]
            residuals = flux - model_flux
            ax0.plot(time, model_flux, color=c, label=l, alpha=alpha)
            ax1.plot(time, residuals,  color=c, label=l, alpha=alpha)
            l = None # ensure just one legend entry
        plotted_samples.append(flat_samples[ind])
        if best_fit == True:
            _, pb = stats(sampler, cut=cut)
            best_fit_flux = model(pb, time)
            best_fit_residuals = flux - best_fit_flux
            # black outline photometry
            ax0.plot(time, best_fit_flux, color='k', lw=4)
            ax0.plot(time, best_fit_flux, color=c)
            # black outline residuals
            ax1.plot(time, best_fit_residuals, color='k', lw=3)
            ax1.plot(time, best_fit_residuals, color=c)
    ax1.set_ylim(residual_lims)
    leg = ax0.legend(fontsize=14)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    # figure layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig(savename)
    plt.show()
    return plotted_samples

def plot_models(time, flux, error, model_list, P_list, lbls=None, 
                plot_lims=None, residual_lims=None, savename='test.png',
                flip=False, dt=0, lw=4):
    '''
    this function plots models against each other

    Parameters
    ----------
    time : array of floats
        contains time data for the light curve
    flux : array of floats
        contains flux data for the light curve
    error : array of floats
        contains error data for the light curve
    model_list : list of functions
        models for the light curve
    P_list : list of list, tuple, array of floats
        best fit parameters for each model
    lbls : list of str
        contains the names of the models and parameters for the legend
    plot_lims : tuple
        bounds of the model subplot
    residual_lims : tuple
        bounds of the residual subplot
    savename : str
        name of the saved plot
    flip : bool
        plots the model flipped to measure asymmetry [default = False]
    dt : int
        number of days to shift the xlabel by [default = 0]

    Returns
    -------
    chi2s : array of floats
        contains the chi2 value for each of the models tested according to
        chi2 = sum( ( (flux - model_flux) / error )^2 )
    '''
    colors = 2 * ['r', 'g', 'b', 'y', 'm', 'c']
    chi2s  = []
    # set up figure
    fig = plt.figure(figsize=(13, 10))
    # ax0 is the flux plot
    ax0 = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=3)
    ax0.set_ylabel('Normalised Flux [-]', fontsize=16)
    ax0.errorbar(time, flux, yerr=error, marker='.', color='k', label='data')
    ax0.tick_params(axis='both', labelsize=14)
    ax0.legend(fontsize=14)
    ax0.set_ylim(plot_lims)
    # ax1 is the residual plot
    ax1 = plt.subplot2grid((4, 1), (3, 0), colspan=1, rowspan=2, sharex=ax0)
    for P, model, l, c in zip(P_list, model_list, lbls, colors):
        flux_model = model(P, time)
        residuals = flux - flux_model
        ax0.plot(time, flux_model, label=l, color=c, lw=lw)
        if flip == True:
            ax0.plot(time, np.flip(flux_model), label='%s flipped' % l, lw=lw)
        if len(P_list) == 1:
            c = 'k'
        ax1.plot(time, residuals, marker='.', label=l, color=c)
        # calculate chi2
        chi2 = np.sum((residuals/error)**2)
        chi2s.append(chi2)
    ax1.tick_params(axis='both', labelsize=14)
    ax0.legend(fontsize=14)
    ax1.axhline(y=0, color='k', ls=':')
    ax1.set_ylabel('Residuals [-]', fontsize=16)
    ax1.set_xlabel('Time [BJD - %i]' % (2454833 + dt), fontsize=16)
    ax1.set_ylim(residual_lims)
    # figure layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig(savename)
    plt.show()
    chi2s = np.array(chi2s)
    return chi2s

def extract_solutions(sampler, inds, bounds, cut=0, lbls=None,
                      solution_names=None, savename='test.png'):
    '''
    This function plots how the walkers move, you can also cut some of the data
    to better view the data
    
    Parameters
    ----------
    sampler : EnsembleSampler
        MCMC object containing all the parameters
    inds : list of int
        indices that correspond with the bounds to make masks to extract 
        sub-samples
    bounds : list of tuples
        should be the same length as inds, the tuple should be a lower and an
        upper bound
    cut : int
        number of links to remove (burn-in period)
    lbls : list of str
        contains the names of the parameters
    solution_names : list of str
        contains the names of each extraction
    savename : str
        name of the saved plot
        
    Returns
    -------
    sub_samples : list of arrays
        contains the parameter values of the walkers that have been extracted
        by the inds and bounds
    '''
    # extract samples
    try:
        # ensemble object
        samples = sampler.chain
        samples = np.moveaxis(samples,0,1)
    except:
        # numpy array
        samples = sampler
    ns, nw, ndim = samples.shape
    # fix labels if necessary
    if isinstance(lbls, type(None)):
       lbls = [None] * ndim
    # masks are created based on the final value of the walker
    last_sample = samples[-1, :, :]
    sub_samples = []
    # here we can apply masks
    for ind, bound in zip(inds, bounds):
        lower_mask = last_sample[:, ind] > bound[0] 
        upper_mask = last_sample[:, ind] < bound[1]
        mask = lower_mask * upper_mask
        sub_samples.append(samples[:, mask, :])
    # creating the plot
    colors = ['r','g','c','y','m','b']
    fig, ax = plt.subplots(ndim, figsize=(14, ndim * 4), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    ax[0].set_title('%i Walkers (Burn-in = %i)' % (nw, cut), fontsize=24)
    for k in range(ndim):
        ax[k].tick_params(labelsize=18)
        # plot samples
        ax[k].plot(samples[cut:,:,k],"k",alpha=0.3)
        # plot sub-samples
        lines = []
        for x, sub_sample in enumerate(sub_samples):
            ax[k].plot(sub_sample[cut:, :, k], colors[x % 6], alpha=0.3)
            l, = ax[k].plot(sub_sample[cut:, 0, k], colors[x % 6], alpha=0.01)
            lines.append(l)
        ax[k].set_xlim(0, ns - cut)
        ax[k].set_ylabel(lbls[k], fontsize=24)
        ax[k].yaxis.set_label_coords(-0.07, 0.5)
    leg = ax[0].legend(lines, solution_names, loc='lower right', fontsize=16,
                       frameon=False, bbox_to_anchor=(1.01, 0.97))
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    ax[-1].set_xlabel('Step Number', fontsize=20)
    plt.savefig(savename)
    plt.show()
    return sub_samples


###############################################################################
############################# STATISTICS FUNCTIONS ############################
###############################################################################

def stats(sampler, cut=0):
    '''
    This function returns the percentiles of the parameters and best fit
    parameters.
    
    Parameters
    ----------
    sampler : EnsembleSampler
        MCMC object containing all the parameters.
    cut : int
        Number of links to remove (burn-in period).
        
    Returns
    -------
    statistics : tuple
        Containing the 50th, 16th and 84th percentile of the parameters.
    p_best : list
        Contains just the 50th percentile (the mean).
    '''
    # extracting the samples
    try:
        # ensemble object
        flat_samples = sampler.get_chain(discard=cut, flat=True)
    except:
        # numpy array
        _, _, ndim = sampler.shape
        flat_samples = sampler[cut:, :, :].reshape((-1, ndim))
    # extracting percentiles
    lower, mid, upper = np.percentile(flat_samples, [16,50,84], axis=0)
    statistics = np.array([mid, upper-mid, mid-lower]).T
    # representing best fit
    p_best = mid
    return statistics, p_best


###############################################################################
############################## MAIN MCMC FUNCTION #############################
###############################################################################

def run_mcmc(time, flux, error, model, model_prior, prior_args, P, ns, 
             savename='test.h5', reset=False, 
             moves=[(emcee.moves.StretchMove(), 1)]):
    '''
    this function actually runs the mcmc code

    Parameters
    ----------
    time : array of floats
        Contains time data for the light curve.
    flux : array of floats
        Contains flux data for the light curve.
    error : array of floats
        Contains error data for the light curve.
    model : function
        Model for the light curve.
    model_prior : function
        Prior to calculate model probability.
    prior_args: tuple 
        Contains additional arguments for the model_prior.
    P : list, tuple, array of floats
        Contains model parameters for all the walkers.
    ns : int
        Number of steps for the walkers.
    savename : str
        Name of the backend to save data.
    reset : bool
        If true will reset progress, if false will append to backend
        [default = False].

    Returns
    -------
    p0 : array
        Contains the best fit values for the sampler.
    sampler : EnsembleSampler
        MCMC object containing all the parameters.
    '''
    nw, ndim = P.shape
    # setting up backend
    BE = emcee.backends.HDFBackend(savename)
    if reset == True:
        BE.reset(nw, ndim)
    # setting up the sampler
    args = (time, flux, error, model, model_prior, prior_args)
    sampler = emcee.EnsembleSampler(nw, ndim, lnprob, args=args, backend=BE, 
                                    moves=moves)
    # determine P
    if reset == False:
        try:
            P = sampler.chain[:, -1, :]
        except:
            pass
    p0, _, _ = sampler.run_mcmc(P, ns, progress=True)
    return p0, sampler


###############################################################################
############################# PRINT FUNCTIONS #################################
###############################################################################

def print_parameters(parameters, lbls=None, units=None, digits=6):
    '''
    This function prints the parameters with their units in a user friendly
    way.

    Parameters
    ----------
    parameters : list, tuple, array of floats
        Contains the parameter values to be printed.
    lbls : list of str
        Contains the names of the parameters.
    units : list of str
        Contains the names of the parameter units.
    digits : int
        The number of digits for the formatting str for the parameter values.
    
    Returns
    -------
    None
    '''
    # fix lbls and units if necessary
    if isinstance(lbls, type(None)):
        lbls = [''] * len(parameters)
    if isinstance(units, type(None)):
        lbls = ['-'] * len(parameters)
    # run through parameters to print
    for parameter, lbl, unit in zip(parameters, lbls, units):
        name = lbl.ljust(18)
        digit = digits
        if unit == 'deg':
            parameter = np.rad2deg(parameter)
        if np.abs(parameter) >= 10:
            digit -= 1
            if np.abs(parameter) >= 100:
                digit -= 1
        fmt = '%'+'+.%if' % digit
        print_statement = '%s =     '+fmt+'     [%s]'
        print(print_statement % (name, parameter, unit))
    return None
