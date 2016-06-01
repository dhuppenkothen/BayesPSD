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

from BayesPSD.parametricmodels import combine_models

try:
    from statsmodels.tools.numdiff import approx_hess
    comp_hessian = True
except ImportError:
    comp_hessian = False


### own imports
from BayesPSD import posterior
from BayesPSD import powerspectrum

### global variables ####
logmin = -100.0


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

    ### x = x-coordinate of data
    ### y = y-coordinate of data
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
    def __init__(self, x, y, obs=True, fitmethod='powell'):

        ### save power spectrum in attributes
        self.x= x
        self.y= y
        ### Is this a real observation or a fake periodogram to be fitted?
        self.obs = obs

        ### smooth data by three different factors
        self.smooth3 = scipy.signal.wiener(self.y, 3)
        self.smooth5 = scipy.signal.wiener(self.y, 5)
        self.smooth11 = scipy.signal.wiener(self.y, 11)

        self._set_fitmethod(self, fitmethod)

    def _set_fitmethod(self, fitmethod):
        ### select fitting method 
        if fitmethod.lower() in ['simplex']:
            self.fitmethod = scipy.optimize.fmin
        elif fitmethod.lower() in ['powell']:
            self.fitmethod = scipy.optimize.fmin_powell
        elif fitmethod.lower() in ['gradient']:
            self.fitmethod = scipy.optimize.fmin_cg
        elif fitmethod.lower() in ['bfgs']:
            self.fitmethod = scipy.optimize.fmin_bfgs
        ### this one breaks because I can't figure out the syntax for fprime
        elif fitmethod.lower() in ['newton']:
            self.fitmethod = scipy.optimize.fmin_ncg
        elif fitmethod.lower() in ['leastsq']:
            self.fitmethod = scipy.optimize.leastsq
        elif fitmethod.lower() in ['constbfgs']:
            self.fitmethod = scipy.optimize.fmin_l_bfgs_b
        elif fitmethod.lower() in ['tnc']:
            self.fitmethod = scipy.optimize.fmin_tnc

        else:
            print("Minimization method not recognized. Using standard (Powell's) method.")
            self.fitmethod = scipy.optimize.fmin_powell


    ### Do a maximum likelihood fitting with function func and 
    ### initial parameters ain
    ### if residuals are to be fit, put a list of residuals into keyword 'residuals'

    ### func = function to be fitted
    ### ain = list with set of initial parameters
    ### obs = if True, compute covariance and print summary to screen
    ### noise = if True, the last parameter in ain is noise and will be renormalized
    ### residuals = put list of residuals here if they should be fit rather than self.y

    def mlest(self, func, ain, obs=True, noise=None, neg=True, functype='posterior'):

        fitparams = self._fitting(func, ain, obs=True)

        if functype in ['p', 'post', 'posterior']:
            fitparams['deviance'] = 2.0*func.loglikelihood(fitparams['popt'], neg=True)
        elif functype in ['l', 'like', 'likelihood']:
            fitparams['deviance'] = -2.0*func(fitparams['popt'])

        print("Fitting statistics: ")
        print(" -- number of frequencies: " + str(len(self.x)))
        print(" -- Deviance [-2 log L] D = " + str(fitparams['deviance']))

        return fitparams


    ### Fitting Routine
    ### optfunc: function to be minimized
    ### ain: initial parameter set
    ### optfuncprime: analytic derivative of optfunc (if required)
    ### neg: bool keyword for MAP estimation (if done):
    ###      if True: compute the negative of the posterior
    def _fitting(self, optfunc, ain, optfuncprime=None, neg = True, obs=True):

        lenpower = float(len(self.y))

        if neg == True:
            if scipy.__version__ < "0.10.0":
                args = [neg]
            else:
                args = (neg,)
        else:
            args = ()

        ### different commands for different fitting methods,
        ### at least until scipy 0.11 is out
       
        funcval = 100.0
        while funcval == 100 or funcval == 200 or funcval == 0.0 or funcval == np.inf or funcval == -np.inf:
        ## constrained minimization with truncated newton or constrained bfgs

            ## Newton conjugate gradient, which doesn't work
            if self.fitmethod == scipy.optimize.fmin_ncg:
                aopt = self.fitmethod(optfunc, ain, optfuncprime, disp=0,args=args)

                ### BFGS algorithm
            elif self.fitmethod == scipy.optimize.fmin_bfgs:
                aopt = self.fitmethod(optfunc, ain, disp=0,full_output=True, args=args)

                warnflag = aopt[6]
                if warnflag == 1 :
                    print("*** ACHTUNG! Maximum number of iterations exceeded! ***")
                #elif warnflag == 2:
                    #print("Gradient and/or function calls not changing!")


            ## all other methods: Simplex, Powell, Gradient
            else:
                aopt = self.fitmethod(optfunc, ain, disp=0,full_output = True, args=args)

            funcval = aopt[1]
            ain = np.array(ain)*((np.random.rand(len(ain))-0.5)*4.0)
 
        ### make a dictionary with best-fit parameters:
        ##  popt: best fit parameters (list)
        ##  result: value of ML function at minimum
        ##  model: the model used

        fitparams = {'popt':aopt[0], 'result':aopt[1]}


        ### calculate model power spectrum from optimal parameters
        #fitparams['mfit'] = func(self.x, *fitparams['popt'])
        ### degrees of freedom
        fitparams['dof'] = lenpower - float(len(fitparams['popt']))
        ### Akaike Information Criterion
        fitparams['aic'] = fitparams['result']+2.0*len(ain)
        ### Bayesian Information Criterion
        fitparams['bic'] = fitparams['result'] + len(ain)*len(self.x)
 
        ### compute deviance
        try:
            fitparams['deviance'] = 2.0*optfunc.loglikelihood(fitparams['popt'])
        except AttributeError:
            fitparams['deviance'] = 2.0*optfunc(fitparams['popt'])

        fitparams['sexp'] = 2.0*len(self.x)*len(fitparams['popt'])
        fitparams['ssd'] = np.sqrt(2.0*fitparams['sexp'])

        fitparams['smooth3'] = scipy.signal.wiener(self.y, 3)
        fitparams['smooth5'] = scipy.signal.wiener(self.y, 5)
        fitparams['smooth11'] = scipy.signal.wiener(self.y, 11)

        ### if this is an observation (not fake data), compute the covariance matrix
        if obs == True:
            ### for BFGS, get covariance from algorithm output
            if self.fitmethod == scipy.optimize.fmin_bfgs:
                print("Approximating covariance from BFGS: ")
                covar = aopt[3]
                stderr = np.sqrt(np.diag(covar))

            else:
                ### calculate Hessian approximating with finite differences
                print("Approximating Hessian with finite differences ...")
                if comp_hessian:
                    phess = approx_hess(aopt[0], optfunc, neg=args)

                    ### covariance is the inverse of the Hessian
                    print("Hessian (empirical): " + str(phess))

                    covar = np.linalg.inv(phess)
                    stderr = np.sqrt(np.diag(covar))

                else:
                    print("Cannot compute hessian! Use BFGS or install statsmodels!")
                    covar = None
                    stderr = None

            print("Covariance (empirical): " + str(covar))

            fitparams['cov'] = covar

            ### errors of parameters are on the diagonal of the covariance
            ### matrix; take square root to get standard deviation
            fitparams['err'] = stderr

            ### Print results to screen
            print("The best-fit model parameters plus errors are:")
            for i,(x,y) in enumerate(zip(fitparams['popt'], stderr)):
                print("Parameter " + str(i) + ": " + str(x) + " +/- " + str(y))
            print("The Akaike Information Criterion of the power law model is: "+ str(fitparams['aic']) + ".")

        return fitparams

    #### This function computes the Likelihood Ratio Test between two nested models
    ### 
    ### mod1: model 1 (simpler model)
    ### ain1: list of input parameters for model 1
    ### mod2: model 2 (more complex model)
    ### ain2: list of input parameters for model 2
    def compute_lrt(self, mod1, ain1, mod2, ain2, noise1 = -1, noise2 = -1, nmax=1):

        print("I AM USING THIS CODE!")
        ### fit data with both models
        par1 = self.mlest(mod1, ain1, obs=self.obs, noise=noise1, nmax=nmax)
        par2 = self.mlest(mod2, ain2, obs=self.obs, noise=noise2, nmax=nmax)

        ### extract dictionaries with parameters for each
        varname1 = "model1fit"
        varname2 = "model2fit"

        self.__setattr__(varname1, par1)
        self.__setattr__(varname2, par2)
      
        ### compute log likelihood ratio as difference between the deviances
        self.lrt = par1['deviance'] - par2['deviance']

        if self.obs == True: 
            print("The Likelihood Ratio for models %s and %s is: LRT = %.4f"%(varname1, varname2, self.lrt))

        return self.lrt


    ### auxiliary function that makes a Lorentzian with a fixed centroid frequency
    ### needed for QPO search algorithm
    def __make_lorentzians(self,x):
       ### loop creates many function definitions lorentz, each differs only by the value
       ### of the centroid frequency f used in computing the spectrum
       for f in x:
           def create_my_func(f):
               def lorentz(x, a, b, e):
                   result = qpo(x, a, b, f, e)
                   return result
               return lorentz
           yield(create_my_func(f))


    #### Fit Lorentzians at each frequency in the spectrum
    #### and return a list of log-likelihoods at each value
    ### fitpars = parameters of broadband noise fit
    ### residuals: if true: divide data by best-fit model in fitpars
    def fitqpo(self, fitpars=None, residuals=False):

        if residuals:
            ### extract model fit
            mfit = fitpars['mfit']

            ### compute residuals: data/model
            residuals = np.array(fitpars["smooth5"])/mfit

        else:
            residuals = np.array(fitpars["smooth5"])

        ### constraint on width of QPO: must be bigger than 2*frequency resolution
        gamma_min = 2.0*(self.x[2]-self.x[1])

        ### empty list for log-likelihoods
        like_rat = []
        
        ### fit a Lorentzian at every frequency
        for f, func, res in zip(self.x[3:-3], self.__make_lorentzians(self.x[3:-3]), residuals[3:-3]):
            ### constraint on width of QPO: must be narrower than the centroid frequency/2
            gamma_max = f/2.0
            norm = np.mean(residuals)+np.var(residuals)
            ain = [gamma_min, norm, 0.0]
            
            ### fit QPO to data 
            #pars = self.mlest(func, ain, noise = True, obs=False, residuals=None)
            pars = self.mlest(func, ain, noise = -1, obs=False, residuals=residuals)

            ### save fitted frequency and data residuals in parameter dictionary
            pars['fitfreq'] = f
            pars['residuals'] = residuals
            like_rat.append(pars)  

        ### returns a list of parameter dictionaries
        return like_rat
            
    #### Find QPOs in Periodogram data
    ### func = broadband noise model
    ### ain = input parameters for broadband noise model
    ### fitmethod = which method to use for fitting the QPOs 
    ### plot = if True, save a plot with log-likelihoods 
    ### plotname = string used in filename if plot == True
    ### obs = if True, compute covariances and print out stuff
    def find_qpo(self, func, ain,
                 fitmethod='nlm',
                 plot=False,
                 plotname=None,
                 obs = False):

        ### fit broadband noise model to the data
        optpars = self.mlest(func, ain, obs=obs, noise=-1)

        ### fit a variable Lorentzian to every frequency and return parameter values
        lrts = self.fitqpo(fitpars=optpars, residuals=True)

        ### list of likelihood ratios
        like_rat = np.array([x['deviance'] for x in lrts])


        ### find minimum likelihood ratio
        minind = np.where(like_rat == min(like_rat))
        minind = minind[0][0]+3
        #print(minind)
        minfreq = self.x[minind] ### ... maybe not! Needs to be +1 because first frequency left out in psfit

        print("The frequency of the tentative QPO is: " + str(minfreq))

        residuals = self.smooth5/optpars['mfit']

        best_lorentz = self.__make_lorentzians([minfreq])       

        noiseind = len(optpars['popt']) - 1


        ### minimum width of QPO
        gamma_min = np.log((self.x[1]-self.x[0])*3.0)
        ### maximum width of QPO
        gamma_max = minfreq/1.5

        print('combmod first component: ' + str(func))
        ### create a combined model of broadband noise model + QPO
        combmod = combine_models((func, len(optpars['popt'])), (qpo, 3), mode='add')

        ### make a list of input parameters
        inpars = list(optpars['popt'].copy())
        inpars.extend(lrts[minind-3]['popt'][:2])
        inpars.extend([minfreq])

        #qpobounds = [[None, None] for x in range(len(inpars)-3)]
        #qpobounds.extend([[gamma_min, gamma_max], [None, None], [None,None]])


        ### fit broadband QPO + noise model, using best-fit parameters as input
        qpopars = self.mlest(combmod, inpars,obs=obs, noise=noiseind, smooth=0)

        ### likelihood ratio of func+QPO to func
        lrt = optpars['deviance'] - qpopars['deviance']

        like_rat_norm = like_rat/np.mean(like_rat)*np.mean(self.y)*100.0

        if plot:
            plt.figure()
            axL = plt.subplot(1,1,1)
            plt.plot(self.x, self.y, lw=3, c='navy')
            plt.plot(self.x, qpopars['mfit'], lw=3, c='MediumOrchid')            
            plt.xscale("log") 
            plt.yscale("log")
            plt.xlabel('Frequency')
            plt.ylabel('variance normalized power')

            axR = plt.twinx()
            axR.yaxis.tick_right()
            axR.yaxis.set_label_position("right")
            plt.plot(self.x[3:-3], like_rat, 'r--', lw=2, c="DeepSkyBlue")
            plt.ylabel("-2*log-likelihood")

            plt.axis([min(self.x), max(self.x), min(like_rat)-np.var(like_rat), max(like_rat)+np.var(like_rat)])

            plt.savefig(plotname+'.png', format='png')
            plt.close()

        return lrt, optpars, qpopars


    ### plot two fits of broadband models against each other
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

            p1, = plt.plot(logx, logy, color='black', linestyle='steps-mid')
            p1smooth = plt.plot(logx, logpar1s5, lw=3, color='orange')
            p2, = plt.plot(logx, logpar1, color='blue', lw=2)

        else:
            p1, = plt.plot(self.x, self.y, color='black', linestyle='steps-mid')
            p1smooth = plt.plot(self.x, par1['smooth5'], lw=3, color='orange')
            p2, = plt.plot(self.x, par1['mfit'], color='blue', lw=2)
        if par2:
            if log:
                logpar2 = np.log10(par2['mfit'])
                p3, = plt.plot(logx, logpar2, color='red', lw=2)
            else:
                p3, = plt.plot(self.x, par2['mfit'], color='red', lw=2)
            plt.legend([p1, p2, p3], ["observed periodogram", par1['model'] + " fit", par2['model'] + " fit"])
        else:
            plt.legend([p1, p2], ["observed periodogram", par1['model'] + " fit"])

        if log:
            plt.axis([min(logx), max(logx), min(logy)-1.0, max(logy)+1])
            plt.ylabel('log(Leahy-Normalized Power)', fontsize=18)

        else:
            plt.xscale("log")
            plt.yscale("log")

            plt.axis([min(self.x), max(self.x), min(self.y)/10.0, max(self.y)*10.0])
            plt.ylabel('Leahy-Normalized Power', fontsize=18)
        plt.title("Periodogram and fits for burst " + namestr, fontsize=18)

        
        ### second subplot: power/model for Power law and straight line
        s2 = plt.subplot2grid((4,1),(2,0),rowspan=1)
        pldif = self.y/par1['mfit']
        if par2:
            bpldif = self.y/par2['mfit']

        if log:
            plt.plot(logx, pldif, color='black', linestyle='steps-mid')
            plt.plot(logx, np.ones(len(self.x)), color='blue', lw=2)

        else:
            plt.plot(self.x, pldif, color='black', linestyle='steps-mid')
            plt.plot(self.x, np.ones(len(self.x)), color='blue', lw=2)
        plt.ylabel("Residuals, \n" + par1['model'] + " model", fontsize=18)

        if log:
            plt.axis([min(logx), max(logx), min(pldif), max(pldif)])

        else:
            plt.xscale("log")
            plt.yscale("log")
            plt.axis([min(self.x), max(self.x), min(pldif), max(pldif)])

        if par2:
            bpldif = self.y/par2['mfit']

        ### third subplot: power/model for bent power law and straight line
            s3 = plt.subplot2grid((4,1),(3,0),rowspan=1)

            if log:
                plt.plot(logx, bpldif, color='black', linestyle='steps-mid')
                plt.plot(logx, np.ones(len(self.x)), color='red', lw=2)
                plt.axis([min(logx), max(logx), min(bpldif), max(bpldif)])

            else:
                plt.plot(self.x, bpldif, color='black', linestyle='steps-mid')
                plt.plot(self.x, np.ones(len(self.x)), color='red', lw=2)
                plt.xscale("log")
                plt.yscale("log")
                plt.axis([min(self.x), max(self.x), min(bpldif), max(bpldif)])

            plt.ylabel("Residuals, \n" + par2['model'] + " model", fontsize=18)
  
        ax = plt.gca()
 
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(14)

        if log:
            plt.xlabel("log(Frequency) [Hz]", fontsize=18)
        else:
            plt.xlabel("Frequency [Hz]", fontsize=18)

        ### make sure xticks are taken from first plots, but don't appear there
        plt.setp(s1.get_xticklabels(), visible=False)
 
        ### save figure in png file and close plot device
        plt.savefig(namestr + '_ps_fit.png', format='png')
        plt.close()

        return



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
    def __init__(self, ps, obs=True, fitmethod='powell'):

        ### ignore first elements in ps.freq and ps.ps (= no. of photons)
        #ps.freq = np.array(ps.freq[1:])
        self.x = ps.freq[1:]
        #ps.ps = np.array(ps.ps[1:])
        self.y = ps.ps[1:]

        self.ps = ps

        ### Is this a real observation or a fake periodogram to be fitted?
        self.obs = obs

        ### set fitmethod
        self._set_fitmethod(fitmethod)


    def mlest(self, func, ain, obs=True, noise=None, nmax=1, residuals = None, smooth=0, m=1, map=True):

        if smooth == 0 :
            power = self.y
        elif smooth == 3:
            power = self.smooth3
        elif smooth == 5:
            power = self.smooth5
        elif smooth == 11:
            power = self.smooth11
        else:
            raise Exception('No valid option for kwarg "smooth". Options are 0,3,5 and 11!')

        if not residuals is None:
            power = residuals

        lenpower = float(len(power))

        ### renormalize normalization so it's in the right range
        varobs = np.sum(power)
        varmod = np.sum(func(self.x, *ain))
        renorm = varobs/varmod

        if len(ain) > 1:
            ain[1] = ain[1] + np.log(renorm)

        ### If last parameter is noise level, renormalize noise level
        ### to something useful:
        if not noise is None:
            #print("Renormalizing noise level ...")
            ### take the last 50 elements of the power spectrum
            noisepower = power[-51:-1]
            meannoise = np.log(np.mean(noisepower))
            ain[noise] = meannoise

        ### set function to be minimized: posterior density for periodograms:

        pstemp = powerspectrum.PowerSpectrum()
        pstemp.freq = self.x
        pstemp.ps = power
        pstemp.df = self.ps.df

        if m == 1:
            lposterior = posterior.PerPosterior(pstemp, func)
        elif m > 1:
            lposterior = posterior.StackPerPosterior(pstemp, func, m)

        else: 
            raise Exception("Number of power spectra is not a valid number!")

        if not map:
            lpost = lposterior.loglikelihood
        else:
            lpost = lposterior

        fitparams = self._fitting(lpost, ain, neg = True, obs=obs)

        fitparams["model"] = str(func).split()[1]
        fitparams["mfit"] = func(self.x, *fitparams['popt']) 

        ### calculate model power spectrum from optimal parameters
        #fitparams['mfit'] = func(self.x, *fitparams['popt'])
        ### figure-of-merit (SSE)
        fitparams['merit'] = np.sum(((power-fitparams['mfit'])/fitparams['mfit'])**2.0)


        ### find highest outlier
        plrat = 2.0*(self.y/fitparams['mfit'])
        #print(plrat)
        fitparams['sobs'] = np.sum(plrat)

        if nmax ==1:
            ### plmaxpow is the maximum of 2*data/model 
            plmaxpow = max(plrat[1:])
            #print('plmaxpow: ' + str(plmaxpow))
            plmaxind = np.where(plrat == plmaxpow)[0]
            #print('plmaxind: ' + str(plmaxind))
            if len(plmaxind) > 1:
                plmaxind = plmaxind[0]
            elif len(plmaxind) == 0:
                plmaxind = -2
            plmaxfreq = self.x[plmaxind]

        else:

            plratsort = copy.copy(plrat)
            plratsort.sort()
            plmaxpow = plratsort[-nmax:]

            plmaxind, plmaxfreq = [], []
            for p in plmaxpow:
                try:
                    plmaxind_temp = np.where(plrat == p)[0]
                    if len(plmaxind_temp) > 1:
                        plmaxind_temp = plmaxind_temp[0]
                    elif len(plmaxind_temp) == 0:
                        plmaxind_temp = -2
                    plmaxind.append(plmaxind_temp)
                    plmaxfreq.append(self.x[plmaxind_temp])

                except TypeError:
                    plmaxind.append(None)
                    plmaxfreq.append(None)



        fitparams['maxpow'] =  plmaxpow
        fitparams['maxind'] = plmaxind
        fitparams['maxfreq'] = plmaxfreq


        s3rat = 2.0*(fitparams['smooth3']/fitparams['mfit'])
        fitparams['s3max'] = max(s3rat[1:])
        try:
            s3maxind = np.where(s3rat == fitparams['s3max'])[0]
            if len(s3maxind) > 1:
                s3maxind = s3maxind[0]
            fitparams['s3maxfreq'] = self.x[s3maxind]
        except TypeError:
            fitparams["s3maxfreq"] = None
        s5rat = 2.0*(fitparams['smooth5']/fitparams['mfit'])
        fitparams['s5max'] = max(s5rat[1:])
        try:
            s5maxind = np.where(s5rat == fitparams['s5max'])[0]
            if len(s5maxind) > 1:
                s5maxind = s5maxind[0]
            fitparams['s5maxfreq'] = self.x[s5maxind]
        except TypeError:
            fitparams['s5maxfreq'] = None

        s11rat = 2.0*(fitparams['smooth11']/fitparams['mfit'])
        fitparams['s11max'] = max(s11rat[1:])
        try:
            s11maxind = np.where(s11rat == fitparams['s11max'])[0]
            if len(s11maxind) > 1:
                s11maxind = s11maxind[0]
            fitparams['s11maxfreq'] = self.x[s11maxind]
        except TypeError:
            fitparams['s11maxfreq'] = None

        ### compute binned periodograms and find highest outlier in those:
        df = (self.x[1]-self.x[0])
        ### first, compute the maximum binning that would even make sense
        bmax = int(self.x[-1]/(2.0*(self.x[1]-self.x[0])))
        #print('bmax: ' + str(bmax))
        bins = [1,3,5,7,10,15,20,30,50,70,100,200,300,500]

        bindict = {}

        for b in bins:
            if b < bmax:
                if b == 1:
                    binps = self.ps
                else:
                    binps = self.ps.rebinps(b*df)
                binpsname = "bin" + str(b)
                bindict[binpsname] = binps
                binpl = func(binps.freq, *fitparams["popt"])
                binratio = 2.0*np.array(binps.ps)/binpl
                maxind = np.where(binratio[1:] == max(binratio[1:]))[0]
                if len(maxind) > 1:
                    maxind = maxind[0]
                elif len(maxind) == 0 :
                    maxind = -2
                binmaxpow = "bmax" + str(b)
                bindict[binmaxpow] = max(binratio[1:])
                binmaxfreq = "bmaxfreq" + str(b)
                bindict[binmaxfreq] = binps.freq[maxind+1]
                bindict['binpl' + str(b)] = binpl

        fitparams["bindict"] = bindict

        ## do a KS test comparing residuals to the exponential distribution
        plks = scipy.stats.kstest(plrat/2.0, 'expon', N=len(plrat))
        fitparams['ksp'] = plks[1]

        if obs == True:
            print("The figure-of-merit function for this model is: " + str(fitparams['merit']) + " and the fit for " + str(fitparams['dof']) + " dof is " + str(fitparams['merit']/fitparams['dof']) + ".")

            print("Fitting statistics: ")
            print(" -- number of frequencies: " + str(len(self.x)))
            print(" -- Deviance [-2 log L] D = " + str(fitparams['deviance']))
            print(" -- Highest data/model outlier 2I/S = " + str(fitparams['maxpow']))
            print("    at frequency f_max = " + str(fitparams['maxfreq']))

            print(" -- Highest smoothed data/model outlier for smoothing factor [3] 2I/S = " + str(fitparams['s3max']))
            print("    at frequency f_max = " + str(fitparams['s3maxfreq']))
            print(" -- Highest smoothed data/model outlier for smoothing factor [5] 2I/S = " + str(fitparams['s5max']))
            print("    at frequency f_max = " + str(fitparams['s5maxfreq']))
            print(" -- Highest smoothed data/model outlier for smoothing factor [11] 2I/S = " + str(fitparams['s11max']))
            print("    at frequency f_max = " + str(fitparams['s11maxfreq']))

            print(" -- Summed Residuals S = " + str(fitparams['sobs']))
            print(" -- Expected S ~ " + str(fitparams['sexp']) + " +- " + str(fitparams['ssd']))
            print(" -- KS test p-value (use with caution!) p = " + str(fitparams['ksp']))
            print(" -- merit function (SSE) M = " + str(fitparams['merit']))

        return fitparams


    def compute_lrt(self, mod1, ain1, mod2, ain2, noise1=-1, noise2=-1, m=1, map=True, nmax=1):


        ### fit data with both models
        par1 = self.mlest(mod1, ain1, obs=self.obs, noise=noise1, m = m, map = map, nmax=nmax)
        par2 = self.mlest(mod2, ain2, obs=self.obs, noise=noise2, m = m, map = map, nmax=nmax)

        ### extract dictionaries with parameters for each
        varname1 = "model1fit"
        varname2 = "model2fit"

        self.__setattr__(varname1, par1)
        self.__setattr__(varname2, par2)

        ### compute log likelihood ratio as difference between the deviances
        self.lrt = par1['deviance'] - par2['deviance']

        if self.obs == True:
            print("The Likelihood Ratio for models %s and %s is: LRT = %.4f"%(varname1, varname2, self.lrt))


        return self.lrt



