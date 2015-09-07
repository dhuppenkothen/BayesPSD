#### THIS WILL DO THE MAXIMUM LIKELIHOOD FITTING
#
# and assorted other things related to that
#
# It's kind of awful code. 
#
# Separate classes for
#   - periodograms (distributed as chi^2_2 or chi^2_2m, for averaged periodograms)
#   - light curves (not really accurate, use scipy.optimize.curve_fit if you can)
#   - Gaussian Processes (for MAP estimates of GPs)
#
# Note: This script has grown over three years. It's not very optimised and doesn't
# necessarily make sense to someone who is not me. Continue to read on your own peril.
#
#
#

#!/usr/bin/env python

import matplotlib.pyplot as plt

#### GENERAL IMPORTS ###
import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import scipy.signal
import copy

from .parametricmodels import FixedCentroidQPO, CombinedModel

try:
    from statsmodels.tools.numdiff import approx_hess
    comp_hessian = True
except ImportError:
    comp_hessian = False


### own imports
from . import posterior
from . import powerspectrum

### global variables ####


#### CLASS THAT FITS POWER SPECTRA USING MLE ##################
#
# This class provides functionality for maximum likelihood fitting
# of periodogram data to a set of models defined above.
#
# It draws heavily on the various optimization routines provided in
# scipy.optimize, and additionally has the option to use R functionality
# via rPy and a given set of functions defined in an R-script.
#
# Note that many different optimization routines are available, and not all
# may be appropriate for a given problem. 
# Constrained optimization is available via the constrained BFGS and TNC routines.
#
#
class MaxLikelihood(object):
    """ Maximum Likelihood Superclass. """
    def __init__(self, x, y, obs=True, fitmethod='L-BFGS-B'):
        """
        Initialize MaximumLikelihood instance.

        Parameters:
        -----------
        x: numpy.ndarray
            The independent variable
        y: numpy.ndarray
            The dependent variable to be modelled
        obs: bool, optional, default True
            If True, then print details about the fit
        fitmethod: string, optional, default "L-BFGS-B"
            Any of the strings allowed in scipy.optimize.minimize in
            the method keyword. Sets the fit method to be used
        """

        self.x= x
        self.y= y
        ### Is this a real observation or a fake data to be fitted?
        self.obs = obs

        self.fitmethod = fitmethod


    def mlest(self, model, ain):
        """
        Do a maximum likelihood fitting with function model and
        initial parameters ain if residuals are to be fit, put a list
        of residuals into keyword 'residuals'

        Parameters:
        -----------
        model: ParametricModel (or subclass) instance
            The model to be used in fitting
        ain : {list | numpy.ndarray}
            List/array with set of initial parameters

        Returns:
        --------
        fitparams: dict
            A dictionary with the fit results
            TODO: Add description of keywords in the dictionary!
        """
        fitparams = self._fitting(model, ain)

        return fitparams


    ### Fitting Routine
    ### optfunc: function to be minimized
    ### ain: initial parameter set
    ### optfuncprime: analytic derivative of optfunc (if required)
    ### neg: bool keyword for MAP estimation (if done):
    ###      if True: compute the negative of the posterior
    def _fitting(self, optfunc, ain, args=None):

        if args is not None:
            if scipy.__version__ < "0.10.0":
                args = [args]
            else:
                args = (args,)
        else:
            args = ()

        ### different commands for different fitting methods,
        ### at least until scipy 0.11 is out
       
        funcval = 100.0
        while funcval == 100 or funcval == 200 or funcval == 0.0 or funcval == np.inf or funcval == -np.inf:
        ## constrained minimization with truncated newton or constrained bfgs
            aopt = scipy.optimize.minimize(optfunc, ain, method=self.fitmethod, args=args)

            ### Newton conjugate gradient, which doesn't work
            #if self.fitmethod == scipy.optimize.fmin_ncg:
            #    aopt = self.fitmethod(optfunc, ain, optfuncprime, disp=0,args=args)

                ### BFGS algorithm
#            elif self.fitmethod == scipy.optimize.fmin_bfgs:
#                aopt = self.fitmethod(optfunc, ain, disp=0,full_output=True, args=args)
#
            print(aopt.message)

            ### all other methods: Simplex, Powell, Gradient
            #else:
            #    aopt = self.fitmethod(optfunc, ain, disp=0,full_output = True, args=args)

            funcval = aopt.fun
            ain = np.array(ain)*((np.random.rand(len(ain))-0.5)*4.0)
 
        ### make a dictionary with best-fit parameters:
        ##  popt: best fit parameters (list)
        ##  result: value of ML function at minimum
        ##  model: the model used

        fitparams = {'popt':aopt.x, 'result':aopt.fun}

        ### compute deviance
        fitparams['deviance'] = -2.0*optfunc.loglikelihood(fitparams['popt'], neg=False)


        ### if this is an observation (not fake data), compute the covariance matrix
            ### for BFGS, get covariance from algorithm output
        if hasattr(aopt, "hess_inv"):
            fitparams["cov"] = aopt.hess_inv
            fitparams["err"] = np.sqrt(np.diag(fitparams["cov"]))
        else:
            ### calculate Hessian approximating with finite differences
            print("Approximating Hessian with finite differences ...")
            if comp_hessian:
                phess = approx_hess(aopt[0], optfunc, neg=args)

                fitparams["cov"] = np.linalg.inv(phess)
                fitparams["err"] = np.sqrt(np.diag(fitparams["cov"]))

        return fitparams


    def compute_statistics(self, fitparams):
        ### figure-of-merit (SSE)
        ### degrees of freedom
        fitparams['dof'] = self.y.shape[0] - float(fitparams['popt'].shape[0])
        ### Akaike Information Criterion
        fitparams['aic'] = fitparams['result']+2.0*fitparams["popt"].shape[0]
        ### Bayesian Information Criterion
        fitparams['bic'] = fitparams['result'] + fitparams["popt"].shape[0]*np.log(self.x.shape[0])

        fitparams['sexp'] = 2.0*len(self.x)*len(fitparams['popt'])
        fitparams['ssd'] = np.sqrt(2.0*fitparams['sexp'])

        fitparams['merit'] = np.sum(((self.y-fitparams['mfit'])/fitparams['mfit'])**2.0)

        ## do a KS test comparing residuals to the exponential distribution
        plrat = self.y[1:]/fitparams["mfit"][1:]
        plks = scipy.stats.kstest(plrat, 'expon', N=len(plrat))
        fitparams['ksp'] = plks[1]

        if self.obs == True:
            self.print_summary(fitparams)

        return fitparams

    def print_summary(self, fitparams):

        print("The best-fit model parameters plus errors are:")
        for i,(x,y) in enumerate(zip(fitparams['popt'], fitparams["err"])):
            print("Parameter " + str(i) + ": " + str(x) + " +/- " + str(y))

        print("Fitting statistics: ")
        print(" -- number of data points: " + str(len(self.x)))
        print(" -- Deviance [-2 log L] D = " + str(fitparams['deviance']))
        print("The Akaike Information Criterion of the model is: "+ str(fitparams['aic']) + ".")
        print("The Bayesian Information Criterion of the model is: "+ str(fitparams['bic']) + ".")

        print("The figure-of-merit function for this model is: " + str(fitparams['merit']) +
              " and the fit for " + str(fitparams['dof']) + " dof is " +
              str(fitparams['merit']/fitparams['dof']) + ".")

        print("Fitting statistics: ")
        print(" -- number of data points: " + str(len(self.x)))
        print(" -- Deviance [-2 log L] D = " + str(fitparams['deviance']))

        print(" -- Summed Residuals S = " + str(fitparams['sobs']))
        print(" -- Expected S ~ " + str(fitparams['sexp']) + " +- " + str(fitparams['ssd']))
        print(" -- KS test p-value (use with caution!) p = " + str(fitparams['ksp']))
        print(" -- merit function (SSE) M = " + str(fitparams['merit']))

        return

    #### This function computes the Likelihood Ratio Test between two nested models
    ### 
    ### mod1: model 1 (simpler model)
    ### ain1: list of input parameters for model 1
    ### mod2: model 2 (more complex model)
    ### ain2: list of input parameters for model 2
    def compute_lrt(self, mod1, ain1, mod2, ain2, noise1 = -1, noise2 = -1, nmax=1):

        ### fit data with both models
        par1 = self.mlest(mod1, ain1)
        par2 = self.mlest(mod2, ain2)


        self.model1fit =  par1
        self.model2fit =  par2
      
        ### compute log likelihood ratio as difference between the deviances
        self.lrt = par1['deviance'] - par2['deviance']

        if self.obs == True: 
            print("The Likelihood Ratio for models %s and %s is: LRT = %.4f"%("1", "2", self.lrt))

        return self.lrt


##########################################################
##########################################################
##########################################################


#### PERIODOGRAM FITTING SUBCLASS ################
#
# Compute Maximum A Posteriori (MAP) parameters
# for periodograms via Maximum Likelihood
# using the 
# posterior class above
#
#
#
#
#
#
#
#


class PerMaxLike(MaxLikelihood):

    ### ps = PowerSpectrum object with periodogram
    ### obs= if True, compute covariances and print summary to screen
    ###    
    ###  fitmethod = choose optimization method
    ###  options are:
    ### 'simplex': use simplex downhill algorithm
    ### 'powell': use modified Powell's algorithm
    ### 'gradient': use nonlinear conjugate gradient
    ### 'bfgs': use BFGS algorithm
    ### 'newton': use Newton CG 
    ### 'leastsq' : use least-squares method
    ### 'constbfgs': constrained BFGS algorithm
    ### 'tnc': constrained optimization via a truncated Newton algorithm
    ### 'nlm': optimization via R's non-linear minimization routine
    ### 'anneal': simulated annealing for convex problems
    def __init__(self, ps, obs=True, fitmethod='L-BFGS-B'):

        self.ps = ps
        MaxLikelihood.__init__(self, ps.freq, ps.ps, obs=obs, fitmethod=fitmethod)


    def mlest(self, model, ain, map=True):

        ## don't want to fit zeroth frequency, so we'll make a temporary
        ## power spectrum object that doesn't have the zeroth frequency in it
        pstemp = powerspectrum.PowerSpectrum()
        pstemp.freq = self.x[1:]
        pstemp.ps = self.y[1:]
        pstemp.df = self.ps.df

        lposterior = posterior.PerPosterior(pstemp, model, m=self.ps.m)

        if not map:
            lpost = lposterior.loglikelihood
        else:
            lpost = lposterior

        fitparams = self._fitting(lpost, ain, args={"neg":True})

        fitparams["model"] = model
        fitparams["mfit"] = model(self.x, *fitparams['popt'])

        return fitparams


    def highest_outlier_smoothed(self, smooth_factor, fitparams, nmax=1):
        smoothed_data = scipy.signal.wiener(self.y, smooth_factor)
        mfit = fitparams["mfit"]

        smaxpow, smaxfreq, smaxind = self._compute_highest_outlier(self.x, smoothed_data, mfit, nmax=nmax)
        if self.obs:
            print(" -- Highest smoothed data/model outlier for smoothing factor" +
                  "[%i] 2I/S = %.3f"%(smooth_factor, smaxpow))
            print("    at frequency f_max =  " + str(smaxfreq))

        return smaxpow, smaxfreq, smaxind


    def highest_outlier_binned(self, bin_factor, fitparams, nmax = 1):

        if bin_factor == 1:
            binps = self.ps
        else:
            binps = self.ps.rebinps(bin_factor*self.ps.df)
        binmodel = fitparams["model"](binps.freq, *fitparams["popt"])
        binmaxpow, binmaxfreq, binmaxind = \
            self._compute_highest_outlier(binps.freq[1:], binps.ps[1:], binmodel[1:], nmax=nmax)

        return binmaxpow, binmaxfreq, binmaxind


    def _compute_highest_outlier(self, xdata, ydata, model, nmax=1):

        ratio = 2.0*ydata/model

        if nmax > 1:
            ratio_sort = copy.copy(ratio)
            ratio_sort.sort()
            max_y = ratio_sort[-nmax:]

            max_x= np.zeros(max_y.shape[0])
            max_ind = np.zeros(max_y.shape[0])

            for i,my in enumerate(max_y):
                max_x[i], max_ind[i] = self._find_outlier(xdata, ratio, my)

        else:
            max_y = np.max(ratio)
            max_x, max_ind = self._find_outlier(xdata, ratio, max_y)

        return max_y, max_x, max_ind

    def _find_outlier(self, xdata, ratio, max_y):
        max_ind = np.where(ratio == max_y)[0]
        if len(max_ind) == 0:
            max_ind = None
            max_x = None
        else:
            if len(max_ind) > 1:
                max_ind = max_ind[0]
            max_x = xdata[max_ind]

        return max_x, max_ind


    def compute_statistics(self, fitparams, nmax=1):

        fitparams["maxpow"], fitparams["maxfreq"], fitparams["maxind"] = \
            self.highest_outlier_binned(1, fitparams, nmax=nmax)

        MaxLikelihood.compute_statistics(self, fitparams)

        return fitparams

    def print_summary(self, fitparams):
        MaxLikelihood.print_summary(self, fitparams)
        print(" -- Highest data/model outlier 2I/S = " + str(fitparams['maxpow']))
        print("    at frequency f_max = " + str(fitparams['maxfreq']))

        return



    #### Fit Lorentzians at each frequency in the spectrum
    #### and return a list of log-likelihoods at each value
    ### fitpars = parameters of broadband noise fit
    ### residuals: if true: divide data by best-fit model in fitpars
    def fitqpo(self, model, ain, hyperpars=None, map=True):

        if map and hyperpars is None:
            raise Exception("MAP fit requested, but hyper-parameters not defined! Need to set"+
                            "hyper-parameters for the priors!")

        ## fit a broadband noise model
        fitparams = self.mlest(model, ain, map=map)
        modelfit = fitparams["mfit"]

        ## for the preliminary fit we'll work with the residuals
        residuals = 2.0*self.y/modelfit

        ### constraint on width of QPO: must be bigger than 2*frequency resolution
        gamma_min = 2.0*(self.x[2]-self.x[1])

        ### empty list for log-likelihoods
        like_rat = []

        ### fit a Lorentzian at every frequency
        for f in self.x[3:-3]:
            ### constraint on width of QPO: must be narrower than the centroid frequency/2
            gamma_max = f/2.0
            norm = np.mean(residuals)+np.var(residuals)
            const = 2.0

            ain = [gamma_min, norm]

            qpo_model = FixedCentroidQPO(f, hyperpars)

            ### fit QPO to data
            pars = self.mlest(qpo_model, ain, map=map)

            ### save fitted frequency and data residuals in parameter dictionary
            pars['fitfreq'] = f
            pars['residuals'] = residuals
            like_rat.append(pars)

        ### returns a list of parameter dictionaries
        return like_rat

    #### Find QPOs in Periodogram data
    ### model = broadband noise model
    ### ain = input parameters for broadband noise model
    ### fitmethod = which method to use for fitting the QPOs
    ### plot = if True, save a plot with log-likelihoods
    ### plotname = string used in filename if plot == True
    ### obs = if True, compute covariances and print out stuff
    def find_qpo(self, model, ain, map=True, hyperpars=None,
                 plot=False,
                 plotname=None,
                 obs = False):

        if map and hyperpars is None:
            raise Exception("MAP fit requested, but hyper-parameters not defined! Need to set"+
                            "hyper-parameters for the priors!")

        #### fit broadband noise model to the data
        optpars = self.mlest(model, ain, map=map)

        ### fit a variable Lorentzian to every frequency and return parameter values
        lrts = self.fitqpo(model, ain, hyperpars=hyperpars, map=map)

        ### list of likelihoods
        deviance_all = np.array([x['deviance'] for x in lrts])


        ### find minimum likelihood ratio
        min_deviance = np.min(deviance_all)
        min_ind = np.where(deviance_all == min_deviance)[0]
        min_ind = min_ind[0]+3

        minfreq = self.x[min_ind]

        print("The frequency of the tentative QPO is: " + str(minfreq))

        ### minimum width of QPO
        gamma_min = np.log((self.x[1]-self.x[0])*2.0)
        ### maximum width of QPO
        gamma_max = minfreq/2.0

        ## Make a callable for the CombinedModel
        qpo_model = lambda: FixedCentroidQPO(minfreq)
        if hyperpars is not None:
            hyperpars["gamma_min"] = gamma_min
            hyperpars["gamma_max"] = gamma_max

        comb_mod = CombinedModel([model, qpo_model], hyperpars)

        ### make a list of input parameters
        noqpo_opt = optpars["popt"]
        qpo_opt = lrts[min_ind-3]["popt"]
        ain_all = np.hstack([noqpo_opt, qpo_opt])


        ### fit broadband QPO + noise model, using best-fit parameters as input
        qpopars = self.mlest(comb_mod, ain_all, map=map)

        ### likelihood ratio of model+QPO to model
        lrt = optpars['deviance'] - qpopars['deviance']


        #if plot:
        #    plt.figure()
        #    axL = plt.subplot(1,1,1)
        #    plt.plot(self.x, self.y, lw=3, c='navy')
        #    plt.plot(self.x, qpopars['mfit'], lw=3, c='MediumOrchid')
        #    plt.xscale("log")
        #    plt.yscale("log")
        #    plt.xlabel('Frequency')
        #    plt.ylabel('variance normalized power')
#
#            axR = plt.twinx()
#            axR.yaxis.tick_right()
#            axR.yaxis.set_label_position("right")
#            plt.plot(self.x[3:-3], like_rat, 'r--', lw=2, c="DeepSkyBlue")
#            plt.ylabel("-2*log-likelihood")
#
#            plt.axis([min(self.x), max(self.x), min(like_rat)-np.var(like_rat), max(like_rat)+np.var(like_rat)])
#
#            plt.savefig(plotname+'.png', format='png')
#            plt.close()

        return lrt, optpars, qpopars


    ### plot two fits against each other
    def plotfits(self, par1, par2 = None, namestr='test', log=False):

        ### make a figure
        f = plt.figure(figsize=(12,10))
        ### adjust subplots such that the space between the top and bottom of each are zero
        plt.subplots_adjust(hspace=0.0, wspace=0.4)


        ### first subplot of the grid, twice as high as the other two
        ### This is the periodogram with the two fitted models overplotted
        s1 = plt.subplot2grid((4,1),(0,0),rowspan=2)

        if log:
            logx = np.log10(self.x)
            logy = np.log10(self.y)
            logpar1 = np.log10(par1['mfit'])
            logpar1s5 = np.log10(par1['smooth5'])

            p1, = s1.plot(logx, logy, color='black', linestyle='steps-mid')
            p1smooth = s1.plot(logx, logpar1s5, lw=3, color='orange')
            p2, = s1.plot(logx, logpar1, color='blue', lw=2)
            s1.set_xlim([min(logx), max(logx)])
            s1.set_ylim([min(logy)-1.0, max(logy)+1])
            s1.set_ylabel('log(Leahy-Normalized Power)', fontsize=18)

        else:
            p1, = s1.plot(self.x, self.y, color='black', linestyle='steps-mid')
            p1smooth = s1.plot(self.x, par1['smooth5'], lw=3, color='orange')
            p2, = s1.plot(self.x, par1['mfit'], color='blue', lw=2)

            s1.set_xscale("log")
            s1.set_yscale("log")

            s1.set_xlim([min(self.x), max(self.x)])
            s1.set_ylim([min(self.y)/10.0, max(self.y)*10.0])
            s1.set_ylabel('Leahy-Normalized Power', fontsize=18)
        if par2:
            if log:
                logpar2 = np.log10(par2['mfit'])
                p3, = s1.plot(logx, logpar2, color='red', lw=2)
            else:
                p3, = s1.plot(self.x, par2['mfit'], color='red', lw=2)
            s1.legend([p1, p2, p3], ["data", "model 1 fit", "model 2 fit"])
        else:
            s1.legend([p1, p2], ["data", "model fit"])

        s1.set_title("Periodogram and fits for data set " + namestr, fontsize=18)

        ### second subplot: power/model for Power law and straight line
        s2 = plt.subplot2grid((4,1),(2,0),rowspan=1)
        pldif = self.y/par1['mfit']
        s2.set_ylabel("Residuals, \n" + par1['model'] + " model", fontsize=18)

        if log:
            s2.plot(logx, pldif, color='black', linestyle='steps-mid')
            s2.plot(logx, np.ones(self.x.shape[0]), color='blue', lw=2)
            s2.set_xlim([min(logx), max(logx)])
            s2.set_ylim([min(pldif), max(pldif)])

        else:
            s2.plot(self.x, pldif, color='black', linestyle='steps-mid')
            s2.plot(self.x, np.ones(self.x.shape[0]), color='blue', lw=2)

            s2.set_xscale("log")
            s2.set_yscale("log")
            s2.set_xlim([min(self.x), max(self.x)])
            s2.set_ylim([min(pldif), max(pldif)])

        if par2:
            bpldif = self.y/par2['mfit']

        ### third subplot: power/model for bent power law and straight line
            s3 = plt.subplot2grid((4,1),(3,0),rowspan=1)

            if log:
                s3.plot(logx, bpldif, color='black', linestyle='steps-mid')
                s3.plot(logx, np.ones(len(self.x)), color='red', lw=2)
                s3.axis([min(logx), max(logx), min(bpldif), max(bpldif)])
                s3.set_xlabel("log(Frequency) [Hz]", fontsize=18)

            else:
                s3.plot(self.x, bpldif, color='black', linestyle='steps-mid')
                s3.plot(self.x, np.ones(len(self.x)), color='red', lw=2)
                s3.set_xscale("log")
                s3.set_yscale("log")
                s3.set_xlim([min(self.x), max(self.x)])
                s3.set_ylim([min(bpldif), max(bpldif)])
                s3.set_xlabel("Frequency [Hz]", fontsize=18)

            s3.set_ylabel("Residuals, \n" + par2['model'] + " model", fontsize=18)

        else:
            if log:
                s2.set_xlabel("log(Frequency) [Hz]", fontsize=18)
            else:
                s2.set_xlabel("Frequency [Hz]", fontsize=18)

        ax = plt.gca()

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(14)

        ### make sure xticks are taken from first plots, but don't appear there
        plt.setp(s1.get_xticklabels(), visible=False)

        ### save figure in png file and close plot device
        plt.savefig(namestr + '_ps_fit.png', format='png')
        plt.close()

        return

