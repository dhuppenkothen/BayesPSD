## BAYESIAN ANALYSIS FOR PERIODOGRAMS
#
# 
#
#
# TO DO LIST:
# - add functionality for mixture models/QPOs to mlprior
# - add logging
# - add smoothing to periodograms to pick out narrow signals
# - add proposal distributions to emcee implementation beyond Gaussian
#

#!/usr/bin/env python

from __future__ import print_function
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.ticker import MaxNLocator
import cPickle as pickle
import copy
#import matplotlib
#matplotlib.use('png')


### GENERAL IMPORTS ###
import numpy as np
import scipy.optimize
from scipy.stats.mstats import mquantiles as quantiles
import scipy.stats
import time as tsys
import math


### New: added possibility to use emcee for MCMCs
try:
   import emcee
   import acor
   emcee_import = True
except ImportError:
   print("Emcee and Acor not installed. Using Metropolis-Hastings algorithm for Markov Chain Monte Carlo simulations.")
   emcee_import = False


### OWN SCRIPTS
import generaltools as gt
import lightcurve
import powerspectrum
import mle
import posterior


### Hack for Numpy Choice function ###
#
# Will be slow for large arrays.
#
#
# Input: - data= list to pick from
#        - weights = statistical weight of each element in data
#          if no weights are given, all choices are equally likely
#        - size = number of choices to generate (default: one)
#
# Output: - either single entry from data, chosen according to weights
#         - or a list of choices 
#
# Note that unlike numpy.random.choice, this function has no "replace"
# option! This means that elements picked from data will *always* be
# replaced, i.e. can be picked again!
#
#
def choice_hack(data, weights=None, size=None):



    #print "len(data): " + str(len(data))
    ### if no weights are given, all choices have equal probability
    if weights == None:
        weights = [1.0/float(len(data)) for x in range(len(data))]

    #print("weights: " + str(weights))
    #print "sum of Weights: " + str(sum(weights))
    if not np.sum(weights) == 1.0:
        if np.absolute(weights[0]) > 1.0e7 and sum(weights) == 0:
            weights = [1.0/float(len(data)) for x in range(len(data))]
        else:
            raise Exception("Weights entered do not add up to 1! This must not happen!")


    #print "Sum of weights: " + str(np.sum(weights))

    ### Compute edges of each bin
    edges = []
    etemp = 0.0
    for x,y in zip(data, weights):
       etemp = etemp + y
       edges.append(etemp)

    ### if the np.sum of all weights does not add up to 1, raise an Exception


    ### If no size given, just print one number
    if size == None:
        randno = np.random.rand()    

    ### Else make sure that size is an integer
    ### and make a list of random numbers
    try:
        randno = [np.random.rand() for x in np.arange(size)]
    except TypeError:
        raise TypeError("size should be an integer!")

    choice_index = np.array(edges).searchsorted(randno)
    choice_data = np.array(data)[choice_index]

    return choice_data



### See if cutting-edge numpy is installed so I can use choice
try:
    from numpy.random import choice
     ### if not, use hack
except ImportError:
    choice = choice_hack





### Compute log posterior ###
#
# This function computes the log- posterior
# probability density from the log-likelihood
# and the log-prior
#
# Returns: log-posterior probability density
#
def lpost(t0, func, ps):
    ### log-likelihood at parameter set t0
    mlogl = mle.maxlike(ps.freq, ps.ps, func, t0)
    ### log of prior distribution at t0
    priorval = mlprior(t0, func)
    ### log posterior is log-likelihood + log-prior
    mlpost = mlogl + priorval
    return mlpost


### Compute prior densities ###
#
# This function computes prior densities for
# all parameters and the whole parameter set
# 
# Returns the log-prior density
#
#
# NOTE: so far, this can only do power laws and bent
# power laws! At some point, I need to add functionality
# for mixture models and Lorentzians!
#
#
#
def mlprior(t0, func):

    ### allowed range for PL indices
    alim = [-1.0, 8.0]

    ### power law index always first value
    alpha = t0[0]
    ### palpha is True if it is within bounds and false otherwise
    ### then pr = pbeta*pgamma if palpha = True, 0 otherwise
    palpha = (alpha >= alim[0] and alpha <= alim[1])
    ### normalization always second parameter
    #beta = t0[1]
    pbeta = 1.0
    ### Poisson noise level always last parameter
    #gamma = t0[-1]
    pgamma = 1.0
    pr = palpha*pbeta*pgamma

    ### if we have a power law, we're done
    ### for a bent power law, there are two 
    ### more parameters:
    if func == mle.bpl:
    #    delta = t0[2]
        pdelta = 1.0
    #    eps = t0[3]
        peps = 1.0
        pr = pr*pdelta*peps

    #else:
    #   raise Exception("Function not defined. Will give all parameters a normal distribution!")

    ### compute logarithm of prior density
    if pr > 0:
        mlp = np.log(pr) 
    else:
        ### if prior density is zero, set it to a really small value
        ### to avoid logarithm errors
        mlp = -100.0 

    return mlp


##########################################
#########################################




##########################################
#
# class Bayes: Bayesian data analysis for time series
#
# This class defines a Bayes object that can:
# - pick between two models using likelihood ratio tests
# - find periodicities by picking out the largest power in 
#   an observation/set of fake periodograms
# - search for QPOs via a model selection approach using LRTs
#
#
# TO DO: Need to add smoothing for picking out narrow signals
# 
#
#
class Bayes(object):

    ### initialize Bayes object
    ## ps: power spectrum object
    ## namestr: string to be used for saving output and plots
    ## plot: bool to set whether plots should be made
    ## m: int, number of stacked spectra or averaged frequency bins
    def __init__(self, ps, namestr='test', plot=True, m=1):
        self.ps = ps
        self.namestr = namestr
        self.plot=plot
        self.m = m

    ### THIS IS A METHOD THAT WILL ENABLE TO CHOOSE A MODEL ####
    #
    # Fit two models, compute LRT, then run MCMCs
    # Pick nsim parameter sets from MCMCs, create fake periodograms,
    # fit with both models, compute LRT, compare
    #
    # func1 [function]: simpler model
    # par1 [list]: input parameter set for func1
    # func2 [function]: more complex model
    # par2 [list]: input parameter set for func2
    # bounds1, bounds2 [lists]: bounds for func1 and func2 (constrained optimization only)
    # fitmethod [string]: scipy.optimize optimization routine (see mle.py for options)
    # nchain [int]: number of chains (MH) or walkers (emcee)
    # niter [int]: number of elements in each chain or walker
    # nsim [int]: number of samples to be drawn from MCMC object
    # covfactor [float]: tuning parameter for MCMCs
    # use_emcee [bool]: use emcee package (True) or Metropolis-Hastings (False)?
    # parname [list]: if required, add list of parameter names here (for plotting)
    # noise1, noise2 [int]: index of the white noise parameter in par1 and par2
    #                       (-1 for pl and bpl)
    #
    def choose_noise_model(self, func1, par1, func2, par2, 
                     bounds1=None, 
                     bounds2=None, 
                     fitmethod='constbfgs',
                     nchain = 10,
                     niter = 5000,
                     nsim = 1000,
                     covfactor = 1.0,
                     use_emcee = True,
                     parname = None,
                     noise1 = -1,
                     noise2 = -1,
                     writefile= True):

        tstart = tsys.clock()

        resfilename = self.namestr + "_choosenoisemodel.dat"
        resfile = gt.TwoPrint(resfilename)


        ### make strings for function names from function definition
        func1name = str(func1).split()[1] 
        func2name = str(func2).split()[1]

        ### step 1: fit both models to observation and compute LRT
        psfit = mle.PerMaxLike(self.ps, fitmethod=fitmethod, obs=True)
        obslrt = psfit.compute_lrt(func1, par1, func2, par2, bounds1=bounds1, bounds2=bounds2, noise1=noise1, noise2=noise2, m=self.m)

        ### get out best fit parameters and associated quantities
        fitpars1 = getattr(psfit, func1name+'fit')
        fitpars2 = getattr(psfit, func2name+'fit')

        if self.plot:
            ### plot the periodogram and best fit models
            psfit.plotfits(fitpars1, fitpars2, namestr = self.namestr, log=True)        
   

        if self.m == 1:
            lpost = posterior.PerPosterior(self.ps, func1)
        else:
            lpost = posterior.StackPerPosterior(self.ps, func1, self.m)

        ### Step 2: Set up Markov Chain Monte Carlo Simulations
        ### of model 1:
        mcobs = MarkovChainMonteCarlo(self.ps.freq, self.ps.ps, 
                                      topt = fitpars1['popt'], 
                                      tcov = fitpars1['cov'], 
                                      covfactor = covfactor, 
                                      niter=niter, 
                                      nchain=nchain, 
                                      lpost = lpost,
                                      paraname = parname,
                                      check_conv = True,
                                      namestr = self.namestr,
                                      use_emcee = use_emcee,
                                      plot = self.plot,
                                      printobj = resfile,
                                      m = self.m) 


        ### Step 3: create fake periodograms out of MCMCs
        fakeper = mcobs.simulate_periodogram(func1, nsim = nsim)

        ### empty lists for simulated quantities of interest
        sim_lrt, sim_deviance, sim_ksp, sim_maxpow, sim_merit, sim_fpeak, sim_y0, sim_srat = [], [], [], [], [], [], [], []

        ### Step 4: Fit fake periodograms and read out parameters of interest from each fit:
        for i,x in enumerate(fakeper):
            fitfake = mle.PerMaxLike(x, fitmethod=fitmethod, obs=False)
            try: 
                lrt = fitfake.compute_lrt(func1, par1, func2, par2, bounds1=bounds1, bounds2=bounds2, noise1=noise1, noise2=noise2, m=self.m)

            except:
                resfile('Fitting of fake periodogram ' + str(i) + ' failed! Returning ...')
#                return psfit, fakeper, mcobs
                continue
            sim_pars1 = getattr(fitfake, func1name+'fit')
            sim_pars2 = getattr(fitfake, func2name+'fit')
            #if lrt > 20:
            #    fitfake.plotfits(sim_pars1, sim_pars2, namestr=self.namestr+'_'+str(i))

            sim_lrt.append(lrt)
            sim_deviance.append(sim_pars1['deviance'])
            sim_ksp.append(sim_pars1['ksp'])
            sim_maxpow.append(sim_pars1['maxpow'])
            sim_merit.append(sim_pars1['merit'])
            sim_fpeak.append(sim_pars1['maxfreq'])
            sim_y0.append(sim_pars1['mfit'][sim_pars1['maxind']])
            sim_srat.append(sim_pars1['sobs'])


        if len(sim_maxpow) == 0:
            resfile("Analysis of Burst failed! Returning ...")
            return False, False, False
        else:
 
            ### Step 5: Compute Bayesian posterior probabilities of individual quantities
            p_maxpow = float(len([x for x in sim_maxpow if x > fitpars1['maxpow']]))/float(len(sim_maxpow))
            p_deviance = float(len([x for x in sim_deviance if x > fitpars1['deviance']]))/float(len(sim_deviance))
            p_ksp = float(len([x for x in sim_ksp if x > fitpars1['ksp']]))/float(len(sim_ksp))
            p_merit = float(len([x for x in sim_merit if x > fitpars1['merit']]))/float(len(sim_merit))
            p_lrt = float(len([x for x in sim_lrt if x > obslrt]))/float(len(sim_lrt))
            p_srat = float(len([x for x in sim_srat if x > fitpars1['sobs']]))/float(len(sim_srat))

            resfile('simulated srat: ' + str(sim_srat))
            resfile('observed srat: ' + str(fitpars1['sobs']))
            resfile("p(LRT) = " + str(p_lrt))

            resfile("KSP(obs) = " + str(fitpars1['ksp']))
            resfile("mean(sim_ksp) = " + str(np.mean(sim_ksp)))

            resfile("Merit(obs) = " + str(fitpars1['merit']))
            resfile("mean(sim_merit) = " + str(np.mean(sim_merit)))

            resfile("Srat(obs) = " + str(fitpars1['sobs']))
            resfile("mean(sim_srat) = " + str(np.mean(sim_srat)))


            ### Step 6: Compute errors of Bayesian posterior probabilities
            pmaxpow_err = np.sqrt(p_maxpow*(1.0-p_maxpow)/float(len(sim_ksp)))
            pdeviance_err = np.sqrt(p_deviance*(1.0-p_deviance)/float(len(sim_ksp)))
            pksp_err = np.sqrt(p_ksp*(1.0-p_ksp)/float(len(sim_ksp)))
            pmerit_err = np.sqrt(p_merit*(1.0-p_merit)/float(len(sim_ksp)))
            plrt_err = np.sqrt(p_lrt*(1.0-p_lrt)/float(len(sim_ksp)))
            psrat_err = np.sqrt(p_srat*(1.0-p_srat)/float(len(sim_ksp)))


            ### Display results on screen and make funky plots
            resfile("Bayesian p-value for maximum power P_max =  " + str(p_maxpow) + " +/- " + str(pmaxpow_err))
            resfile("Bayesian p-value for deviance D =  " + str(p_deviance) + " +/- " + str(pdeviance_err))
            resfile("Bayesian p-value for KS test: " + str(p_ksp) + " +/- " + str(pksp_err))
            resfile("Bayesian p-value for Merit function: " + str(p_merit) + " +/- " + str(pmerit_err))
            resfile("Bayesian p-value for the np.sum of residuals: " + str(p_srat) + " +/- " + str(psrat_err))
            resfile("Bayesian p-value for Likelihood Ratio: " + str(p_lrt) + " +/- " + str(plrt_err))

            if self.plot:
                n, bins, patches = plt.hist(sim_lrt, bins=100, normed = True, color="cyan",  histtype='stepfilled')
                plt.vlines(obslrt, 0.0, 0.8*max(n), lw=4, color='navy')
                plt.savefig(self.namestr + '_lrt.png', format='png')
                plt.close()
            tend = tsys.clock()
            resfile("Time for " + str(nsim) + " simulations: " + str(tend-tstart) + " seconds.")

            summary = {"p_lrt":[p_lrt, plrt_err], "p_maxpow":[p_maxpow, pmaxpow_err], "p_deviance":[p_deviance, pdeviance_err], "p_ksp":[p_ksp, pksp_err], "p_merit":[p_merit, pmerit_err], "p_srat":[p_srat, psrat_err], "postmean":mcobs.mean, "posterr":mcobs.std, "postquantiles":mcobs.ci,"rhat":mcobs.rhat, "acor":mcobs.acor, "acceptance":mcobs.acceptance}


            return psfit,fakeper, summary


    ### FIND PERIODICITIES IN OBSERVED DATA ######
    #
    # Find periodicities in observed data and compute
    # significance via MCMCs.
    #
    # func [function]: best-fit broadband noise model
    # par [list]: list of parameters for func
    # bounds [list]: bounds on par (constrained optimization only)
    # fitmethod [string]: scipy.optimize routine (see mle.py for options)
    # nchain [int]: number of walkers/chains
    # niter [int]: number of iterations per chain/walker
    # nsim [int]: number of samples to draw from MCMCs
    # covfactor [float]: tuning parameter for MCMCs
    # parname [list]: list of parameter names (if required, for plotting)
    # noise [int]: index of noise parameter for broadband noise model (-1 for pl and bpl)
    # use_emcee [bool]: use emcee (True) or Metropolis-Hastings (False) for MCMCs?
    def find_periodicity(self, func, par,
                 bounds=None,
                 fitmethod='powell',
                 nchain = 10,
                 niter = 5000,
                 nsim = 1000,
                 covfactor = 1.0,
                 parname = None,
                 noise = -1,
                 use_emcee = True):

        resfilename = self.namestr + "_findperiodicity_results.dat"
        
        resfile = gt.TwoPrint(resfilename)

        tstart = tsys.clock()

        funcname = str(func).split()[1]

        ### step 1: fit model to observation
        psfit = mle.PerMaxLike(self.ps, fitmethod=fitmethod, obs=True)
        fitpars = psfit.mlest(func, par, bounds=bounds, obs=True, noise=noise, m=self.m)
        bindict = fitpars['bindict']
        print('popt: ' + str(fitpars['popt']))


        if self.m == 1:
            lpost = posterior.PerPosterior(self.ps, func)
        else:
            lpost = posterior.StackPerPosterior(self.ps, func, self.m)




        ### Step 2: Set up Markov Chain Monte Carlo Simulations
        ### of model 1:
        mcobs = MarkovChainMonteCarlo(self.ps.freq, self.ps.ps, 
                                      topt = fitpars['popt'],
                                      tcov = fitpars['cov'],
                                      covfactor = covfactor,
                                      niter=niter,
                                      nchain=nchain,
                                      lpost=lpost,
                                      paraname = parname,
                                      check_conv = True,
                                      namestr = self.namestr,
                                      use_emcee = True,
                                      plot=self.plot, 
                                      printobj = resfile,
                                      m = self.m)


        ### Step 3: create fake periodograms out of MCMCs
        fakeper = mcobs.simulate_periodogram(func, nsim = nsim)

        sim_pars_all, sim_deviance, sim_ksp, sim_fpeak, sim_srat, sim_maxpow, sim_merit, sim_y0, sim_s3max, sim_s5max, sim_s11max =[], [], [], [], [], [], [], [], [], [], []
        

        bmax = int(self.ps.freq[-1]/(2.0*(self.ps.freq[1]-self.ps.freq[0])))
        bins = [1,3,5,7,10,15,20,30,50,70,100,200,300,500,700,1000]


        binlist = [r for r in fitpars["bindict"].keys()]
        nbins = len(binlist)/4
        sain = copy.copy(fitpars['popt'])

#        print('popt2: ' + str(fitpars['popt']))
        ### Step 4: Fit fake periodograms:
        for i,x in enumerate(fakeper):
            try:
#            print('popt' + str(i) + 'a : ' + str(fitpars['popt']))
      
                fitfake = mle.PerMaxLike(x, fitmethod=fitmethod, obs=False)
#            print('popt' + str(i) + 'b : ' + str(fitpars['popt']))

                sim_pars = fitfake.mlest(func, sain, bounds=bounds, obs=False, noise=noise, m=self.m)
#            print('popt' + str(i) + 'c : ' + str(fitpars['popt']))
         
                sim_pars_all.append(sim_pars)
 
                sim_deviance.append(sim_pars['deviance'])
                sim_ksp.append(sim_pars['ksp'])
                sim_maxpow.append(sim_pars['maxpow'])
                sim_merit.append(sim_pars['merit'])
                sim_fpeak.append(sim_pars['maxfreq'])
                sim_y0.append(sim_pars['mfit'][sim_pars['maxind']])
                sim_srat.append(sim_pars['sobs'])
                sim_s3max.append(sim_pars['s3max'])
                sim_s5max.append(sim_pars['s5max'])
                sim_s11max.append(sim_pars['s11max'])
        
            except:
                print("Simulation failed! Continuing ...")
                continue 
#               print('popt' + str(i) + 'd : ' + str(fitpars['popt']))

#             print('popt3: ' + str(fitpars['popt']))

        ### upper limit is the power in the sorted array where p_maxpow would be 0.05
        ### i.e. when only 0.05*nsim simulations are higher than this
        ### note: sometimes simulations fail, therefore the 5% limit should be 0.05*len(sims)
        fiveperlim = int(0.05*len(sim_maxpow))
        if fiveperlim == 0: 
            resfile('Warning! Too few simulations to compute five percent limit reliably!')
            fiveperlim = 1
        ninetyfiveperlim = len(sim_maxpow) - fiveperlim


        print('popt4: ' + str(fitpars['popt']))
        bindicts = [x["bindict"] for x in sim_pars_all] 
        ### get out binned powers:

        maxpows_all = {}

        binprob = {}
        for b in bins[:nbins]:
            binps = fitpars['bindict']['bin'+str(b)]
            bmaxpow = np.array([x["bmax" + str(b)] for x in bindicts])

            maxpows_all["bin"+str(b)] = bmaxpow

            bindict['sim_bmaxpow' + str(b)] = bmaxpow
            p_bmaxpow = float(len([x for x in bmaxpow if x > fitpars['bindict']["bmax" + str(b)]]))/float(len(bmaxpow))
            bindict["p_maxpow" + str(b)] = p_bmaxpow
            
            bmaxpow_err = np.sqrt(p_bmaxpow*(1.0-p_bmaxpow)/float(len(bmaxpow)))
            bindict['p_maxpow' + str(b) + 'err'] = bmaxpow_err            
        
            sim_bmaxpow_sort = np.msort(bmaxpow)

            ### note: this is the limit for 2*I/S --> multiply by S to get powers for each frequency 
            ### Like everything else, this is n-trial corrected!
            #print('len(bmaxpow_sort) : ' + str(len(sim_bmaxpow_sort)))
            resfile('ninetyfiveperlim: ' + str(ninetyfiveperlim))
            bmaxpow_ul = sim_bmaxpow_sort[ninetyfiveperlim]
            bindict['bmax' + str(b) + '_ul'] = bmaxpow_ul 
            resfile('The posterior p-value for the maximum residual power for a binning of ' + str(self.ps.df*b) + 'Hz is p = ' + str(p_bmaxpow) + ' +/- ' +  str(bmaxpow_err))
            resfile('The corresponding value of the T_R statistic at frequency f = ' + str(fitpars["bindict"]["bmaxfreq" + str(b)]) + ' is 2I/S = ' + str(fitpars['bindict']["bmax" + str(b)]))

            resfile('The upper limit on the T_R statistic is 2I/S = ' + str(bmaxpow_ul))

            ### now turn upper limit into an rms amplitude:
            ## first compute broadband noise model for binned frequencies
            bintemplate = func(fitpars['bindict']['bin'+str(b)].freq, *fitpars['popt'])
            resfile("bintemplate[0]: " + str(bintemplate[0]))
            ## then compute upper limits for powers I_j depending on frequency
            binpowers = bmaxpow_ul*bintemplate/2.0 - bintemplate
            ## now compute rms amplitude at 40, 70, 100 and 300 Hz

            ## first, convert powers into rms normalization, if they're not already
            if self.ps.norm == 'leahy':
                binpowers = binpowers/(self.ps.df*b * self.ps.nphots)
            elif self.ps.norm == 'variance':
                binpowers = binpowers*self.ps.n**2.0 / (self.ps.df*b*self.ps.nphots**2.0)

            print('len(binps.freq): ' + str(len(binps.freq)))
            print('len(binpowers): ' + str(len(binpowers)))


            ## for 40 Hz: 
            for bc in [40.0, 70.0, 100.0, 300.0, 500.0, 1000.0]:
                if bc > (binps.freq[1] - binps.freq[0]):
                    bind = np.searchsorted(binps.freq, bc) - 1
                    #print('bind :' + str(bind))
                    #print('len(binps.freq): ' + str(len(binps.freq)))
                    #print('len(binpowers): ' + str(len(binpowers)))
                    bpow = binpowers[bind]
                    brms = np.sqrt(bpow*b*self.ps.df)
                    resfile('The upper limit on the power at ' + str(bc) + 'Hz for a binning of ' + str(b) + ' is P = ' + str(bpow*(self.ps.df*b*self.ps.nphots)))
                    resfile('The upper limit on the rms amplitude at ' + str(bc) + 'Hz for a binning of ' + str(b) + ' is rms = ' + str(brms))
                    bindict['bin' + str(b) + '_ul_' + str(int(bc)) + 'Hz'] = brms 
                else:
                    continue


        ### Step 5: Compute Bayesian posterior probabilities of individual quantities
        p_maxpow = float(len([x for x in sim_maxpow if x > fitpars['maxpow']]))/float(len(sim_maxpow))
        p_deviance = float(len([x for x in sim_deviance if x > fitpars['deviance']]))/float(len(sim_deviance))
        p_ksp = float(len([x for x in sim_ksp if x > fitpars['ksp']]))/float(len(sim_ksp))
        p_merit = float(len([x for x in sim_merit if x > fitpars['merit']]))/float(len(sim_merit))
        p_srat = float(len([x for x in sim_srat if x > fitpars['sobs']]))/float(len(sim_srat))
 
        p_s3max = float(len([x for x in sim_s3max if x > fitpars['s3max']]))/float(len(sim_s3max))
        p_s5max = float(len([x for x in sim_s5max if x > fitpars['s5max']]))/float(len(sim_s5max))
        p_s11max = float(len([x for x in sim_s11max if x > fitpars['s11max']]))/float(len(sim_s11max))


        ### sort maximum powers from lowest to highest
        sim_maxpow_sort = np.msort(sim_maxpow)
        sim_s3max_sort = np.msort(sim_s3max)
        sim_s5max_sort = np.msort(sim_s5max)
        sim_s11max_sort = np.msort(sim_s11max)

        ### note: this is the limit for 2*I/S --> multiply by S to get powers for each frequency 
        ### Like everything else, this is n-trial corrected!
        maxpow_ul = sim_maxpow_sort[ninetyfiveperlim]
        #s3max_ul = sim_s3max_sort[ninetyfiveperlim]
        #s5max_ul = sim_s5max_sort[ninetyfiveperlim]
        #s11max_ul = sim_s11max_sort[ninetyfiveperlim]


        ### Step 6: Compute errors of Bayesian posterior probabilities
        pmaxpow_err = np.sqrt(p_maxpow*(1.0-p_maxpow)/float(len(sim_ksp)))
        pdeviance_err = np.sqrt(p_deviance*(1.0-p_deviance)/float(len(sim_ksp)))
        pksp_err = np.sqrt(p_ksp*(1.0-p_ksp)/float(len(sim_ksp)))
        pmerit_err = np.sqrt(p_merit*(1.0-p_merit)/float(len(sim_ksp)))
        #plrt_err = np.sqrt(p_lrt*(1.0-p_lrt)/float(len(sim_ksp)))
        psrat_err = np.sqrt(p_srat*(1.0-p_srat)/float(len(sim_ksp)))

        ps3max_err = np.sqrt(p_s3max*(1.0-p_s3max)/float(len(sim_ksp)))
        ps5max_err = np.sqrt(p_s5max*(1.0-p_s5max)/float(len(sim_ksp)))
        ps11max_err = np.sqrt(p_s11max*(1.0-p_s11max)/float(len(sim_ksp)))


        ### Display results on screen and make funky plots
        resfile("Bayesian p-value for maximum power P_max =  " + str(p_maxpow) + " +/- " + str(pmaxpow_err))
        #resfile('Upper limit on maximum signal power P_max_ul = ' + str(maxpow_ul))

        resfile("Bayesian p-value for maximum power P_max =  " + str(p_s3max) + " +/- " + str(ps3max_err))
        #resfile('Upper limit on maximum signal power P_max_ul = ' + str(s3max_ul))

        resfile("Bayesian p-value for maximum power P_max =  " + str(p_s5max) + " +/- " + str(ps5max_err))
        #resfile('Upper limit on maximum signal power P_max_ul = ' + str(s5max_ul))

        resfile("Bayesian p-value for maximum power P_max =  " + str(p_s11max) + " +/- " + str(ps11max_err))
        #resfile('Upper limit on maximum signal power P_max_ul = ' + str(s11max_ul))


        resfile("Bayesian p-value for deviance D =  " + str(p_deviance) + " +/- " + str(pdeviance_err))
        resfile("Bayesian p-value for KS test: " + str(p_ksp) + " +/- " + str(pksp_err))
        resfile("Bayesian p-value for Merit function: " + str(p_merit) + " +/- " + str(pmerit_err))
        resfile("Bayesian p-value for the np.sum of residuals: " + str(p_srat) + " +/- " + str(psrat_err))

        if self.plot:
            subplot(2,2,1)
            n, bins, patches = plt.hist(sim_maxpow, bins=100, normed = True, color="cyan",  histtype='stepfilled')
            xmin, xmax = min(min(bins), fitpars['maxpow'])/1.2, max(25, fitpars['maxpow']*1.2)
            plt.axis([xmin, xmax, 0.0, max(n)])
            plt.vlines(fitpars['maxpow'], 0.0, max(n), lw=2, color='navy')
            plt.title('unsmoothed data', fontsize=12)
 
            subplot(2,2,2)
            n, bins, patches = plt.hist(sim_s3max, bins=100, normed = True, color="cyan", histtype='stepfilled')
            xmin, xmax = min(min(bins), fitpars['s3max'])/1.2, max(25, fitpars['s3max']*1.2)
            plt.axis([xmin, xmax, 0.0, max(n)])
            plt.vlines(fitpars['s3max'], 0.0, max(n), lw=2, color='navy')
            plt.title('smoothed (3) data', fontsize=12)

            subplot(2,2,3)
            n, bins, patches = plt.hist(sim_s3max, bins=100, normed = True, color="cyan", histtype='stepfilled')
            xmin, xmax = min(min(bins), fitpars['s5max'])/1.2, max(25, fitpars['s5max']*1.2)
            plt.axis([xmin, xmax, 0.0, max(n)])

            plt.vlines(fitpars['s5max'], 0.0, max(n), lw=2, color='navy')
            plt.title('smoothed (5) data/model outlier', fontsize=12)

            subplot(2,2,4)
            n, bins, patches = plt.hist(sim_s3max, bins=100, normed = True, color="cyan",  histtype='stepfilled')
            xmin, xmax = min(min(bins), fitpars['s11max'])/1.2, max(25, fitpars['s3max']*1.2)
            plt.axis([xmin, xmax, 0.0, max(n)])
 
            plt.vlines(fitpars['s11max'], 0.0, max(n), lw=2, color='navy')
            plt.title('smoothed (11) data', fontsize=12)

            plt.savefig(self.namestr + '_maxpow.png', format='png')

        results = {"fitpars":fitpars, 'bindict':bindict, 'maxpows_all':maxpows_all, 'mcobs':mcobs, 'p_maxpow':[sim_maxpow, p_maxpow, pmaxpow_err], 'maxpow_ul':maxpow_ul, 'p_s3max':[sim_s3max, p_s3max, ps3max_err], 'p_s5max':[sim_s5max, p_s5max, ps5max_err], 'p_s11max':[sim_s11max, p_s11max, ps11max_err], 'p_merit':[p_merit, pmerit_err], 'p_srat':[p_srat, psrat_err], 'p_deviance':[p_deviance, pdeviance_err], 'fitpars':fitpars,  "postmean":mcobs.mean, "posterr":mcobs.std, "postquantiles":mcobs.ci, "rhat":mcobs.rhat, "acor":mcobs.acor, "acceptance":mcobs.acceptance}


        return results



    ### FIND QPOS IN OBSERVED PERIODOGRAMS VIA LRT ##########
    #
    #
    #
    #
    #
    #
    #
    #
    # func [function]: best-fit broadband noise model
    # par [list]: list of parameters for func
    # bounds [list]: bounds on par (constrained optimization only)
    # fitmethod [string]: scipy.optimize routine (see mle.py for options)
    # nchain [int]: number of walkers/chains
    # niter [int]: number of iterations per chain/walker
    # nsim [int]: number of samples to draw from MCMCs
    # covfactor [float]: tuning parameter for MCMCs
    # parname [list]: list of parameter names (if required, for plotting)
    # plot [bool]: Make plots?
    # plotstr [string]: if plot == True, set a string for the names of the plots here
    # noise [int]: index of noise parameter for broadband noise model (-1 for pl and bpl)
    # use_emcee [bool]: use emcee (True) or Metropolis-Hastings (False) for MCMCs?

    def find_qpo(self, func, ain,
                 bounds=None,
                 fitmethod='constbfgs',
                 nchain = 10,
                 niter = 5000,
                 nsim = 1000,
                 covfactor = 1.0,
                 parname = None,
                 plotstr=None,
                 use_emcee = True):

        if plotstr == None:
            plotstr = self.namestr

        tstart = tsys.clock()

        funcname = str(func).split()[1]

        print("<< --- len(self.ps beginning): " + str(len(self.ps.ps)))

        ### step 1: fit model to observation
        psfit = mle.PerMaxLike(self.ps, fitmethod=fitmethod, obs=True)
        fitpars = psfit.mlest(func, ain, bounds=bounds, obs=True, noise=-1, m=self.m)

        print("<< --- len(self.ps beginning): " + str(len(self.ps.ps)))


        #obslrt, optpars, qpopars = psfit.find_qpo(func, ain, bounds=bounds, plot=True, obs=True, plotname = self.namestr+'_loglikes')

        print("<< --- len(self.ps beginning): " + str(len(self.ps.ps)))


        if self.m == 1:
            lpost = posterior.PerPosterior(self.ps, func)
        else:
            lpost = posterior.StackPerPosterior(self.ps, func, self.m)


        ### Step 2: Set up Markov Chain Monte Carlo Simulations
        ### of model 1:
        mcobs = MarkovChainMonteCarlo(self.ps.freq, self.ps.ps, 
                                      topt = fitpars['popt'],
                                      tcov = fitpars['cov'],
                                      covfactor = covfactor,
                                      niter=niter,
                                      nchain=nchain,
                                      lpost = lpost,
                                      paraname = parname,
                                      check_conv = True,
                                      namestr = self.namestr,
                                      use_emcee = True,
                                      plot = self.plot,
                                      m = self.m)

        ### fit broadband noise model to the data
#        psfit = mle.PerMaxLike(self.ps, fitmethod=fitmethod, obs=True)

#        fitparams = psfit.mlest(func, ain, bounds=bounds, obs=True)

        ### find optimum QPO values for the real data
        obslrt, optpars, qpopars = psfit.find_qpo(func, ain, bounds=bounds, plot=True, obs=True, plotname = self.namestr+'_loglikes')

#        print("<--- optpars: " + str(optpars['popt']))

        ### Step 2: Make lots of Markov chains to draw the posterior distributions from
#        mc_func = MarkovChainMonteCarlo(self.ps,
#                                      topt = fitparams['popt'],
#                                      tcov = fitparams['cov'],
#                                      covfactor = covfactor,
#                                      niter=niter,
#                                      nchain=nchain,
#                                      func = func,
#                                      paraname = parname,
#                                      check_conv = True,
#                                      namestr = self.namestr+func.func_name,
#                                      use_emcee = use_emcee)

        ### simulate lots of realizations of the broadband noise model from MCMCs
        funcfake = mcobs.simulate_periodogram(func, nsim = nsim)

        ### empty lists to store simulated LRTS and parameters in
        sim_lrt, sim_optpars, sim_qpopars, sim_deviance, sim_ksp, sim_merit, sim_srat = [], [], [], [], [], [], []

        simno = 0

        ### run QPO search on each and return likelihood ratios parameters for each
        for x in funcfake:
            simno = simno + 1
            sim_psfit = mle.PerMaxLike(x, fitmethod='constbfgs',obs=False)
            slrt, soptpars, sqpopars = sim_psfit.find_qpo(func, ain, bounds=bounds, obs=False, plot=True, plotname = plotstr + '_sim' + str(simno) + '_qposearch') 

            sim_lrt.append(slrt)
            sim_optpars.append(soptpars)
            sim_qpopars.append(sqpopars)
            sim_deviance.append(soptpars['deviance'])
            sim_ksp.append(soptpars['ksp'])
            sim_merit.append(soptpars['merit'])
            sim_srat.append(soptpars['sobs'])


        ### Step 5: Compute Bayesian posterior probabilities of individual quantities
        p_deviance = float(len([x for x in sim_deviance if x > optpars['deviance']]))/float(len(sim_deviance))
        p_ksp = float(len([x for x in sim_ksp if x > optpars['ksp']]))/float(len(sim_ksp))
        p_merit = float(len([x for x in sim_merit if x > optpars['merit']]))/float(len(sim_merit))
        p_lrt = float(len([x for x in sim_lrt if x > obslrt]))/float(len(sim_lrt))
        p_srat = float(len([x for x in sim_srat if x > optpars['sobs']]))/float(len(sim_srat))

        print("p(LRT) = " + str(p_lrt))
        #print("LRT(obs) = " + str(obslrt))
        #print("mean(sim_lrt) = " + str(np.mean(sim_lrt)))


        #print("Deviance(obs) = " + str(fitpars1['deviance']))
        #print("mean(sim_deviance) = " + str(np.mean(sim_deviance)))
        print("KSP(obs) = " + str(optpars['ksp']))
        print("mean(sim_ksp) = " + str(np.mean(sim_ksp)))

        print("Merit(obs) = " + str(optpars['merit']))
        print("mean(sim_merit) = " + str(np.mean(sim_merit)))

        print("Srat(obs) = " + str(optpars['sobs']))
        print("mean(sim_srat) = " + str(np.mean(sim_srat)))



        ### Step 6: Compute errors of Bayesian posterior probabilities
        pdeviance_err = np.sqrt(p_deviance*(1.0-p_deviance)/float(len(sim_ksp)))
        pksp_err = np.sqrt(p_ksp*(1.0-p_ksp)/float(len(sim_ksp)))
        pmerit_err = np.sqrt(p_merit*(1.0-p_merit)/float(len(sim_ksp)))
        plrt_err = np.sqrt(p_lrt*(1.0-p_lrt)/float(len(sim_ksp)))
        psrat_err = np.sqrt(p_srat*(1.0-p_srat)/float(len(sim_ksp)))


        ### Display results on screen and make funky plots
        print("Bayesian p-value for deviance D =  " + str(p_deviance) + " +/- " + str(pdeviance_err))
        print("Bayesian p-value for KS test: " + str(p_ksp) + " +/- " + str(pksp_err))
        print("Bayesian p-value for Merit function: " + str(p_merit) + " +/- " + str(pmerit_err))
        print("Bayesian p-value for the np.sum of residuals: " + str(p_srat) + " +/- " + str(psrat_err))
        print("Bayesian p-value for Likelihood Ratio: " + str(p_lrt) + " +/- " + str(plrt_err))

        if self.plot:
            n, bins, patches = plt.hist(sim_lrt, bins=100, normed = True, histtype='stepfilled')
            plt.vlines(obslrt, 0.0, 0.8*max(n), lw=4, color='m')
            plt.savefig(self.namestr + '_qpolrt.png', format='png')
            plt.close()
        tend = tsys.clock()
        print("Time for " + str(nsim) + " simulations: " + str(tend-tstart) + " seconds.")


	summary = {"p_lrt":[p_lrt, plrt_err], "p_deviance":[p_deviance, pdeviance_err], "p_ksp":[p_ksp, pksp_err], "p_merit":[p_merit, pmerit_err], "p_srat":[p_srat, psrat_err], "postmean":mcobs.mean, "posterr":mcobs.std, "postquantiles":mcobs.ci, "rhat":mcobs.rhat, "acor":mcobs.acor, "acceptance":mcobs.acceptance}



        return summary





    def print_summary(self,summary):

        try:
            keys = summary.keys()
        except AttributeError:
            raise Exception("Summary must be a dictionary!")

        probs = dict()
        postpars = dict()

        ### sort out p-values and posterior distribution of parameters
        for x in keys:
            if x[:2] == 'p_':
                probs[x] = summary[x]
            else:
                postpars[x] = summary[x]


        print("The ensemble acceptance rate is " + str(postpars["acceptance"]) + " .")
        try:
            print("The autocorrelation times are: " + str(postpars["acor"]))
        except KeyError:
            print("Module Acor not found. Cannot compute autocorrelation times for the parameters")
        for i,x in enumerate(postpars["rhat"]):
            print("The R_hat value for Parameter " + str(i) + " is " + str(x))


        ### print posterior summary of parameters:
        print("-- Posterior Summary of Parameters: \n")
        print("parameter \t mean \t\t sd \t\t 5% \t\t 95% \n")
        print("---------------------------------------------\n")
        for i in range(len(postpars['postmean'])):
            print("theta[" + str(i) + "] \t " + str(postpars['postmean'][i]) + "\t" + str(postpars['posterr'][i]) + "\t" + str(postpars['postquantiles'][i][0]) + "\t" + str(postpars["postquantiles"][i][1]) + "\n" )


        for x in probs.keys():
            if x == 'p_lrt':
                print("Bayesian p-value for Likelihood Ratio: " + str(probs[x][0]) + " +/- " + str(probs[x][1]))
            elif x == 'p_deviance':
                print("Bayesian p-value for deviance D =  " + str(probs[x][0]) + " +/- " + str(probs[x][1]))
            elif x == 'p_ksp':
                print("Bayesian p-value for KS test: " + str(probs[x][0]) + " +/- " + str(probs[x][1]))
            elif x == 'p_merit':
                print("Bayesian p-value for Merit function: " + str(probs[x][0]) + " +/- " + str(probs[x][1]))
            elif x == 'p_srat':
                print("Bayesian p-value for the sum of residuals: " +  str(probs[x][0]) + " +/- " + str(probs[x][1]))

            elif x == 'p_maxpow':
                if "fitpars" in probs.keys():
                     print("Highest [unsmoothed] data/model outlier at frequency F=" + str(probs["fitpars"]["maxfreq"]) + "Hz with power P=" + str(probs["fitpars"]["maxpow"]))

                print("Bayesian p-value for the highest [unsmoothed] data/model outlier: " +  str(probs[x][0]) + " +/- " + str(probs[x][1]))
            elif x == 'p_s3max':
                if "fitpars" in probs.keys():
                     print("Highest [3 bin smoothed] data/model outlier at frequency F=" + str(probs["fitpars"]["s3maxfreq"]) + "Hz with power P=" + str(probs["fitpars"]["s3max"]))

                print("Bayesian p-value for the highest [3 bin smoothed] data/model outlier: " +  str(probs[x][0]) + " +/- " + str(probs[x][1]))
            elif x == 'p_s5max':
                if "fitpars" in probs.keys():
                     print("Highest [5 bin smoothed] data/model outlier at frequency F=" + str(probs["fitpars"]["s5maxfreq"]) + "Hz with power P=" + str(probs["fitpars"]["s5max"]))

                print("Bayesian p-value for the highest [5 bin smoothed] data/model outlier: " +  str(probs[x][0]) + " +/- " + str(probs[x][1]))
            elif x == 'p_s11max':
                if "fitpars" in probs.keys():
                     print("Highest [11 bin smoothed] data/model outlier at frequency F=" + str(probs["fitpars"]["s11maxfreq"]) + "Hz with power P=" + str(probs["fitpars"]["s11max"]))

                print("Bayesian p-value for the highest [11 bin smoothed] data/model outlier: " +  str(probs[x][0]) + " +/- " + str(probs[x][1]))





        return


    def write_summary(self,summary, namestr=None):

        if not namestr:
            namestr = self.namestr

        try:
            keys = summary.keys()
        except AttributeError:
            raise Exception("Summary must be a dictionary!")

        probs = dict()
        postpars = dict()

        ### sort out p-values and posterior distribution of parameters
        for x in keys:
            if x[:2] == 'p_':
                probs[x] = summary[x]
            else:
                postpars[x] = summary[x]


        picklefile = open(namestr + "_summary_pickle.dat", "w")
        pickle.dump(summary, picklefile)
        picklefile.close()
        

        file = open(namestr + "_summary.dat", "w")


        file.write("The ensemble acceptance rate is " + str(postpars["acceptance"]) + " .\n")
        try:
            file.write("The autocorrelation times are: " + str(postpars["acor"]) + "\n")
        except KeyError:
            file.write("Module Acor not found. Cannot compute autocorrelation times for the parameters \n")
        for i,x in enumerate(postpars["rhat"]):
            file.write("The R_hat value for Parameter " + str(i) + " is " + str(x) + "\n")


        ### print posterior summary of parameters:
        file.write("-- Posterior Summary of Parameters: \n")
        file.write("parameter \t mean \t\t sd \t\t 5% \t\t 95% \n")
        file.write("---------------------------------------------\n")
        for i in range(len(postpars['postmean'])):
            file.write("theta[" + str(i) + "] \t " + str(postpars['postmean'][i]) + "\t" + str(postpars['posterr'][i]) + "\t" + str(postpars['postquantiles'][i][0]) + "\t" + str(postpars["postquantiles"][i][1]) + "\n" )


        for x in probs.keys():
            if x == 'p_lrt':
                file.write("Bayesian p-value for Likelihood Ratio: " + str(probs[x][0]) + " +/- " + str(probs[x][1]) + "\n")
            elif x == 'p_deviance':
                file.write("Bayesian p-value for deviance D =  " + str(probs[x][0]) + " +/- " + str(probs[x][1]) + "\n")
            elif x == 'p_ksp':
                file.write("Bayesian p-value for KS test: " + str(probs[x][0]) + " +/- " + str(probs[x][1]) + "\n")
            elif x == 'p_merit':
                file.write("Bayesian p-value for Merit function: " + str(probs[x][0]) + " +/- " + str(probs[x][1]) + "\n")
            elif x == 'p_srat':
                file.write("Bayesian p-value for the sum of residuals: " +  str(probs[x][0]) + " +/- " + str(probs[x][1]) + "\n")


            elif x == 'p_maxpow':
                file.write("Bayesian p-value for the highest [unsmoothed] data/model outlier: " +  str(probs[x][0]) + " +/- " + str(probs[x][1]) + "\n")
                file.write("Upper limit for highest [unsmoothed] data/model outlier: " + str(summary['maxpow_ul']) + "\n")
            elif x == 'p_s3max':
                file.write("Bayesian p-value for the highest [3 bin smoothed] data/model outlier: " +  str(probs[x][0]) + " +/- " + str(probs[x][1]) + "\n")
                file.write("Upper limit for highest [unsmoothed] data/model outlier: " + str(summary['s3max_ul']) + "\n")

            elif x == 'p_s5max':
                file.write("Bayesian p-value for the highest [5 bin smoothed] data/model outlier: " +  str(probs[x][0]) + " +/- " + str(probs[x][1]) + "\n")
                file.write("Upper limit for highest [unsmoothed] data/model outlier: " + str(summary['s5max_ul']) + "\n")

            elif x == 'p_s11max':
                file.write("Bayesian p-value for the highest [11 bin smoothed] data/model outlier: " +  str(probs[x][0]) + " +/- " + str(probs[x][1]) + "\n")
                file.write("Upper limit for highest [unsmoothed] data/model outlier: " + str(summary['s11max_ul']) + "\n")

        return

        


    ###
    ### arguments must be given with keywords as tuples of (sim_quant, p_quant)
    ###
    def plot_posteriors(namestr='test', **pars):
 
        plotkeys = pars.keys()
        N = len(plotkeys)

        ### number of parameters
        fig = plt.figure(figsize=(2,N/2+1))
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, wspace=0.2, hspace=0.2)
   
        for i in range(N):
            ax = fig.add_subplot(N/2+1,2,i)
            n, bins, patches = ax.hist(pars[plotkeys[i]][0], 30)
            ax.vlines(pars[plotkeys[i]][0], 0.0, 0.8*max(n), lw=4)
            ax.figtext(pars[plotkeys[i]][0]+0.01*pars[plotkeys[i]][0], 0.8*n, "p = " + str(pars[plotkeys[i]][1]))
            ax.title("Posterior for " + plotkeys[i])

        return


    def fitspec(self, fitmethod='tnc'):
        ### fit real periodogram with MLE fitting routine and compute statistics 
        psfit = mle.PerMaxLike(self.ps, fitmethod=fitmethod, obs=True, m=self.m)
        lrt =  psfit.compute_stats()
        ### plot periodogram results
        psfit.plotfits(namestr = self.namestr)
        return psfit



###########################################################
###########################################################
###########################################################


###########################################################
#
# class MarkovChainMonteCarlo: make a sample of MCMCs
#
#
#
# TO DO: Add functionality for proposal distributions beyond a Gaussian
#
#
#
#
#
class MarkovChainMonteCarlo(object):

# ps: power spectrum (data or simulation)
# niter (int): number of iterations in each chain
# nchain (int): number of chains to run
# topt (list): best-fit set of parameters for MLE
# tcov (np.array): covariance matrix from MLE
# discard (int): fraction of chain to discard (standard: niter/2)
# func: function to be fitted (standard functions in mle module)
# paraname (optional): list of parameter names
# check_conv (bool): check for convergence?
# namestr (str): filename root for plots 
# use_emcee (bool): use emcee (True) or Metropolis-Hastings (False) for MCMCs?
#    def __init__(self, ps, topt, tcov, 
#                 covfactor=1.0, 
#                 niter=5000, 
#                 nchain=10, 
#                 discard=None, 
#                 func=mle.pl,  
#                 paraname = None, 
#                 check_conv = True, 
#                 namestr='test', 
#                 use_emcee=True,
#                 plot=True,
#                 printobj = None,
#                 m=1):


    def __init__(self, x, y, topt, tcov, 
                 covfactor=1.0,
                 niter=5000,
                 nchain=10,
                 discard=None,
                 lpost = mle.pl,
                 paraname = None,
                 check_conv = True,
                 namestr='test',
                 use_emcee=True,
                 plot=True,
                 printobj = None,
                 m=1):


        ### Make sure not to include zeroth frequency
#        self.ps = ps
        self.m = m

        self.x = x
        self.y = y

        if printobj:
            print = printobj
        else:
            from __builtin__ import print as print      

        self.plot = plot
        print("<--- self.ps len MCMC: " + str(len(self.x)))
        ### set of optimal parameters from MLE fitting
        self.topt = topt
        print("mcobs topt: " + str(self.topt))
        ### covariances of fitted parameters
        self.tcov = tcov*covfactor
        print("mcobs tcov: " + str(self.tcov))

        ### number of iterations for MCMC algorithm
        self.niter = niter
        ### number of MCMC chains to be computed
        self.nchain = nchain
        ### Error in the fitted parameters
        self.terr = np.sqrt(np.diag(tcov))
        ### function that was fitted
        self.lpost = lpost

        if discard == None:
            discard = math.floor(niter/2.0)


        mcall = []

        ### if emcee package is not installed, enforce Metropolis-Hastings implementation
        if emcee_import == False:
            print("Emcee not installed. Enforcing M-H algorithm!")
            use_emcee = False

        ### if emcee should be used, then use code below 
        if use_emcee:


#            ### define log-Gaussian proposal distribution
#            def lnprob(x, mu, icov):
#                diff = x-mu
#                return -np.dot(diff,np.dot(icov,diff))/2.0

            ### number of walkers is the number of chains
            nwalkers = self.nchain
            ### number of dimensions for the Gaussian (=number of parameters)
            ndim = len(self.topt)
            ### means of the Gaussians are best-fit parameters
            #means = self.topt
            ### inverse of covariance matrix
            #icov = np.linalg.inv(self.tcov)

            ### sample random starting positions for each of the walkers
            p0 = [np.random.multivariate_normal(self.topt,self.tcov) for i in xrange(nwalkers)]

            #if self.m == 1:
            #    lpost = posterior.PerPosterior(self.ps, self.func)
            #else:
            #    lpost = posterior.StackPerPosterior(self.ps, self.func, self.m)

            #posterior = lpost(self.ps, self.func)
            #icov = np.linalg.inv(self.tcov)

            #p0 = [np.random.multivariate_normal(self.topt, icov) for i in xrange(nwalkers)]

            ### initialize sampler
            sampler = emcee.EnsembleSampler(nwalkers,ndim, lpost, args=[False])
 
            ### run burn-in phase and reset sampler
            pos, prob, state = sampler.run_mcmc(p0, 200)
            sampler.reset()

            ### run actual MCMCs
            sampler.run_mcmc(pos, niter, rstate0=state)

            ### list of all samples stored in flatchain
            mcall = sampler.flatchain

            ### print meanacceptance rate for all walkers and autocorrelation times
            print("The ensemble acceptance rate is: " + str(np.mean(sampler.acceptance_fraction)))
            self.L = np.mean(sampler.acceptance_fraction)*len(mcall)
            self.acceptance = np.mean(sampler.acceptance_fraction)
            try:
                self.acor = sampler.acor
                print("The autocorrelation times are: " +  str(sampler.acor))
            except ImportError:
                print("You can install acor: http://github.com/dfm/acor")
                self.acor = None
            except RuntimeError:
                print("D was negative. No clue why that's the case! Not computing autocorrelation time ...")
                self.acor = None
            except:
                print("Autocorrelation time calculation failed due to an unknown error: " + sys.exc_info()[0] + ". Not computing autocorrelation time.")
                self.acor = None

        ### if emcee_use == False, then use MH algorithm as defined in MarkovChain object below
        else:
            ### loop over all chains
            for i in range(nchain):

                #t0 = topt + choice([2.0, 3.0, -3.0, -2.0], size=len(topt))*self.terr
		    
                ### set up MarkovChain object
                mcout = MarkovChain(niter = niter, topt = self.topt, tcov = self.tcov, lpost=self.lpost, paraname = paraname)
                ### create actual chain
                mcout.create_chain(self.x, self.y)
   
                ### make diagnostic plots
                mcout.run_diagnostics(namestr = namestr +"_c"+str(i), paraname=paraname)
 
                mcall.extend(mcout.theta)

            self.L = mcout.L
        mcall = np.array(mcall)

        ### check whether chains/walkers converged
        if check_conv == True:
            self.check_convergence(mcall, namestr, printobj = printobj)            

        ### transpose list of parameter sets so that I have lists with one parameter each
        self.mcall = mcall.transpose()

        ### make inferences from MCMC chain, plot to screen and save plots
        self.mcmc_infer(namestr=namestr, printobj = printobj)


    ### auxiliary function used in check_convergence
    ### computes R_hat, which compares the variance inside chains to the variances between chains
    def _rhat(self, mcall, printobj = None):


        if printobj:
            print = printobj
        else:
            from __builtin__ import print as print 

        print("Computing Rhat. The closer to 1, the better!")

        rh = []

        ### loop over parameters ###
        for i,k in enumerate(self.topt):

            ### pick parameter out of array
            tpar = np.array([t[i] for t in mcall])

            ### reshape back into array of niter*nchain dimensions
            tpar = np.reshape(tpar, (self.nchain, len(tpar)/self.nchain))

            ### compute mean of variance of each chain

            #### THIS DOESN'T WORK FOR SOME REASON! TAKES VARIANCE OF EACH ELEMENT!!!
            ### CHECK THIS!
            sj = map(lambda y: np.var(y), tpar)
            W = np.mean(sj)

            ### compute variance of means of each chain
            mj = map(lambda y: np.mean(y), tpar)
            ### note: this assumes the discards
            B = np.var(mj)*self.L


            ## now compute marginal posterior variance
            mpv = ((float(self.L)-1.0)/float(self.L))*W + B/float(self.L)

            ### compute Rhat
            rh.append(np.sqrt(mpv/W))

            ### print convergence message on screen:
            print("The Rhat value for parameter " + str(i) + " is: " + str(rh[i]) + ".")

            if rh[i] > 1.2:
                print("*** HIGH Rhat! Check results! ***") 
            else:
                print("Good Rhat. Hoorah!")


        return rh


    def _quantiles(self, mcall):

        ### empty lists for quantiles
        ci0, ci1 = [], []

        ### loop over the parameters ###
        for i,k in enumerate(self.topt):

            print("I am on parameter: " + str(i))

            ### pick parameter out of array
            tpar = np.array([t[i] for t in mcall])
            ### reshape back into array of niter*nchain dimensions
            tpar = np.reshape(tpar, (self.nchain, len(tpar)/self.nchain))

            ### compute mean of variance of each chain
            intv = map(lambda y: quantiles(y, prob=[0.1, 0.9]), tpar)

            ### quantiles will return a list with two elements for each
            ### chain: the 0.1 and 0.9 quantiles
            ### need to pick out these for each chain
            c0 = np.array([x[0] for x in intv])
            c1 = np.array([x[1] for x in intv])

            ### now compute the scale
            scale = np.mean(c1-c0)/2.0

            ### compute means of each chain
            mt = map(lambda y: np.mean(y), tpar)
            ### mean of means of all chains
            offset = np.mean(mt)

            ### rescale quantiles (WHY??)
            ci0.append((c0 - offset)/scale)
            ci1.append((c1 - offset)/scale)

        return ci0, ci1


    def check_convergence(self, mcall, namestr, printobj=None, use_emcee = True):


        if printobj:
            print = printobj
        else:
            from __builtin__ import print as print

        ### compute Rhat for all parameters
        rh = self._rhat(mcall, printobj)               
        self.rhat = rh        

        plt.scatter(rh, np.arange(len(rh))+1.0 )
        plt.axis([0.1,2,0.5,0.5+len(rh)])
        plt.xlabel("R_hat")
        plt.ylabel("Parameter")
        plt.title('Rhat')
        plt.savefig(namestr + '_rhat.ps')
        plt.close()


        ### compute 80% quantiles
        ci0, ci1 = self._quantiles(mcall)


        ### set array with colours
        ### make sure there are enough colours available
        colours_basic = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        cneeded = int(math.ceil(len(ci0[0])/7.0))
        colours = []
        for x in range(cneeded):
            colours.extend(colours_basic)


        ### plot 80% quantiles
        if self.plot:
            plt.plot(0,0)
            plt.axis([-2, 2, 0.5, 0.5+len(ci0)])
            for j in range(self.nchain):
                plt.hlines(y=[m+(j)/(4.0*self.nchain) for m in range(len(ci0))], xmin=[x[j] for x in ci0], xmax=[x[j] for x in ci1], color=colours[j])
            #plt.hlines(y=[m+1.0+(1)/(4*self.nchain) for m in np.arange(len(ci0))], xmin=[x[1] for x in ci0], xmax=[x[1] for x in ci1], color=colours[j])

            plt.xlabel("80% region (scaled)")
            plt.ylabel("Parameter") 
            plt.title("80% quantiles")
            plt.savefig(namestr + "_quantiles.ps")
            plt.close()

    def mcmc_infer(self, namestr='test', printobj = None):

        if printobj:
            print = printobj
        else:
            from __builtin__ import print as print


        ### covariance of the parameters from simulations
        covsim = np.cov(self.mcall)

        print("Covariance matrix (after simulations): \n")
        print(str(covsim))

        ### calculate for each parameter its (posterior) mean and equal tail
        ### 90% (credible) interval from the MCMC

        self.mean = map(lambda y: np.mean(y), self.mcall)
        self.std = map(lambda y: np.std(y), self.mcall)
        self.ci = map(lambda y: quantiles(y, prob=[0.05, 0.95]), self.mcall)


        ### print to screen
        print("-- Posterior Summary of Parameters: \n")
        print("parameter \t mean \t\t sd \t\t 5% \t\t 95% \n")
        print("---------------------------------------------\n")
        for i in range(len(self.topt)):
            print("theta[" + str(i) + "] \t " + str(self.mean[i]) + "\t" + str(self.std[i]) + "\t" + str(self.ci[i][0]) + "\t" + str(self.ci[i][1]) + "\n" )

        #np.random.shuffle(self.mcall)

        ### produce matrix scatter plots

        ### number of parameters
        N = len(self.topt)
        print("N: " + str(N))
        n, bins, patches = [], [], []
 
        if self.plot:
            fig = plt.figure(figsize=(15,15))
            plt.subplots_adjust(top=0.925, bottom=0.025, left=0.025, right=0.975, wspace=0.2, hspace=0.2)
            for i in range(N):
                for j in range(N):
                    xmin, xmax = self.mcall[j][:1000].min(), self.mcall[j][:1000].max()
                    ymin, ymax = self.mcall[i][:1000].min(), self.mcall[i][:1000].max()
                    ax = fig.add_subplot(N,N,i*N+j+1)
                    #ax.axis([xmin, xmax, ymin, ymax])
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.ticklabel_format(style="sci", scilimits=(-2,2))

                    #print('parameter ' + str(i) + ' : ' + str(self.topt[i]))
   
                    if i == j:
                        #pass
                        ntemp, binstemp, patchestemp = ax.hist(self.mcall[i][:1000], 30, normed=True, histtype='stepfilled')
                        n.append(ntemp)
                        bins.append(binstemp)
                        patches.append(patchestemp)
                        ax.axis([ymin, ymax, 0, max(ntemp)*1.2])
#                       ax.axis([xmin, xmax, 0, max(ntemp)*1.2])
   
                    else:
                        #ax = fig.add_subplot(N,N,i*N+j+1)

#                        ax.axis([xmin, xmax, ymin, ymax])

                        ax.axis([xmin, xmax, ymin, ymax])
                   #     np.random.shuffle(self.mcall)

                        ### make a scatter plot first
                        ax.scatter(self.mcall[j][:1000], self.mcall[i][:1000], s=7)
                        ### then add contours

#                        np.random.shuffle(self.mcall)

                        xmin, xmax = self.mcall[j][:1000].min(), self.mcall[j][:1000].max()
                        ymin, ymax = self.mcall[i][:1000].min(), self.mcall[i][:1000].max()
                        #print("xmin and xmax: " + str(xmin) + " and " + str(xmax))

                        ### Perform Kernel density estimate on data
                        try:
                            X,Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                            positions = np.vstack([X.ravel(), Y.ravel()])
                            values = np.vstack([self.mcall[j][:1000], self.mcall[i][:1000]])
                            kernel = scipy.stats.gaussian_kde(values)
                            Z = np.reshape(kernel(positions).T, X.shape)
 
                            ax.contour(X,Y,Z,7)
                        except ValueError:
                            print("Not making contours.")

#        for i in range(N):
#            for j in range(N):
#                plt.subplot(N,N,i*N+j+1)
#                ax.xaxis.set_major_locator(MaxNLocator(5))
#                if i == j:
#                    plt.axis([min(bins[i]), max(bins[i]), 0.0, max(n[i]*1.2)])
#                    ax.xaxis.set_major_locator(MaxNLocator(5))
#                else:
#                    plt.axis([min(bins[j]), max(bins[j]), min(bins[i]), max(bins[i])])
#                    ax.xaxis.set_major_locator(MaxNLocator(4))
  


            plt.savefig(namestr + "_scatter.png", format='png')
            plt.close()
        return


#### POSTERIOR PREDICTIVE CHECKS ################
# 
# Note: fpeak is calculated in mle.PerMaxLike.compute_stats
# and can be found in dictionary self.pl_r or self.bpl_r
#
    ## nsim [int] = number of simulations
    ## dist [str] = distribution, one of
    ##      "exp": exponential distribution (=chi2_2), np.random.exponential
    ##       "chisquare": chi^2 distribution with df degrees of freedom
    ## df [int] = degrees of freedom for chi^2 distribution
    def simulate_periodogram(self, func=mle.pl, nsim=5000):

        ### number of simulations is either given by the user,
        ### or defined by the number of MCMCs run!
        nsim = min(nsim,len(self.mcall[0]))

 
        ### define distribution
#        if dist == "exp":
#             print("Distribution is exponential!")
#             noise = np.random.exponential(size = len(self.ps.freq))
#        elif dist == "chisquare":
#             noise = np.random.chisquare(2*df, size=len(self.ps.freq))/(2.0*df)
#        else:
#             raise Exception("Distribution not recognized")

        if self.m == 1:
            noise = np.random.exponential(size=len(self.x))
        else:
            noise = np.random.chisquare(2, size=len(self.x))/(2.0*self.m)

        ### shuffle MCMC parameters
        theta = np.transpose(self.mcall)
        #print "theta: " + str(len(theta))
        np.random.shuffle(theta)


        jump = int(np.floor(nsim/10))

        fper = []
        fps = []
        percount = 1.0

        perlist = [x*100.0 for x in range(10)]
        for x in range(nsim):

            if x in perlist:
                print(str(percount*10) + "% done ...")
                percount += 1.0
            ### extract parameter set
            ain = theta[x]
            ### compute model 'true' spectrum
#            mpower = self.func(self.x, *ain)
            mpower = func(self.x, *ain)
            ### define distribution
#            if dist == "exp":
#                #print("Distribution is exponential!")
#                noise = np.random.exponential(size = len(self.ps.freq))
#            elif dist == "chisquare":
#                noise = np.random.chisquare(2*df, size=len(self.ps.freq))/(2.0*df)
#            else:
#                raiseException("Distribution not recognized")
            if self.m == 1:
                print("m = 1")
                noise = np.random.exponential(size=len(self.x))
            else:
                print("m = " + str(self.m))
                noise = np.random.chisquare(2*self.m, size=len(self.x))/(2.0*self.m)




            ### add random fluctuations
            mpower = mpower*noise

            ### save generated power spectrum in a PowerSpectrum object
            mps = powerspectrum.PowerSpectrum()
            mps.freq = self.x
            mps.ps = mpower
            mps.df = self.x[1] - self.x[0]
            mps.n = 2.0*len(self.x)
            mps.nphots = mpower[0]

            fps.append(mps)
        return fps




############################################
##### CODE BELOW IS A BAD IDEA! IGNORE THIS UNTIL MarkovChain CLASS DEFINITION
###########################################

#### MAKE A PARAMETER SET OBJECT ####
#
#
#
#
class ParameterSet(object):

    ### par = parameter vector (1D)
    ### parname = list with parameter names
    def __init__(par, parname = None):

        if parname == None:
            x = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'iota', 'lappa', 'lambda', 'mu']

        else:
            if not len(par) == len(parname):
                raise Exception("The length of the parameter array and the length of the list with parameter names doesn't match!")
            x = parname
 
        ### set attributes for parameter space
        for i,p in enumerate(par):
            setattr(self, x[i], p)


#### Don't have any methods for now ####


#########################################################
#########################################################
#########################################################

#### MAKE A MARKOV CHAIN OBJECT ###
#
# QUESTION: How can I make an object with variable
# parameters?
#
#
#
#  NEED TO THINK ABOUT HOW TO GET ATTRIBUTES!
#
class MarkovChain(object):


    def __init__(self, mcsuper = None, niter = 5000, topt = None, tcov =None, lpost = None, paraname=None, discard=None):

 
        self.niter = niter
        self.topt = topt
        self.tcov = tcov
        self.terr = np.sqrt(np.diag(tcov))
        self.t0 = topt + choice([2.0, 3.0, -3.0, -2.0], size=len(topt))*self.terr

        self.lpost = lpost
        self.terr = np.sqrt(np.diag(tcov))
        if discard == None:
            self.discard = int(niter/2)
        else:
            self.discard = int(discard)
        if paraname == None:
            self.paraname = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'iota', 'lappa', 'lambda', 'mu']
        else:
            self.paraname = paraname

    ### set up MCMC chain
    ### possible distributions: 
    ###   - 'mvn': multi-variate normal (default)
    ###   - 'stt': student t-test
    def create_chain(self, x, y, topt=None, tcov = None, t0 = None, dist='mvn'):

        if not topt == None:
            self.topt = topt
        if not tcov == None:
            self.tcov = tcov
        if not t0 == None:
            self.t0 = t0


        ### set up distributions
        if dist=='mvn': 
             dist = np.random.multivariate_normal

        ### need to think about multivariate student-t distribution
        ### not available in numpy/scipy
        #elif dist == 'stt':
        #     dist = np.random.standard_t
        #     cov = cov/3.0

        ### set acceptance value to zero
        accept = 0.0

        ### set up array
        ttemp, logp = [], []
        ttemp.append(self.t0)
        #lpost = posterior.PerPosterior(self.ps, self.func)
        logp.append(self.lpost(self.t0, neg=False))       

        #print "self.topt: " + str(self.t0)
        #print "self.tcov: " + str(self.tcov)

        #print("np.arange(self.niter-1)+1" +  str(np.arange(self.niter-1)+1))
        for t in np.arange(self.niter-1)+1:
#            print("cov: " + str(self.tcov))

            tprop = dist(ttemp[t-1], self.tcov)
#            print("tprop: " + str(tprop))

            pprop = self.lpost(tprop)#, neg=False)
            #pprop = lpost(tprop, self.func, ps)
#            print("pprop: " + str(pprop))

            #logr = logp[t-1] - pprop
            logr = pprop - logp[t-1]
            logr = min(logr, 0.0)
            r= np.exp(logr)
            update = choice([True, False], size=1, weights=[r, 1.0-r])
#            print("update: " + str(update))

            if update:
                ttemp.append(tprop)
                logp.append(pprop)
                if t > self.discard:
                     accept = accept + 1
            else:
                ttemp.append(ttemp[t-1])
                logp.append(logp[t-1])
        


            #logr = logp[t-1] - pprop
            #print("logr: " + str(logr))
            #r = logr #np.exp(logr)

            #samplechoice = choice([True, False], size=1, weights=[0.1, 0.9])

            #if r > 0.0:# or not samplechoice:
            #    ttemp.append(tprop)
            #    logp.append(pprop)
            #    if t > self.discard:
            #         accept = accept+1

            #else:
            #    print "r: " + str(r)
            #    newset = choice([[tprop, pprop], [ttemp[t-1], logp[t-1]]], size=1, weights=[np.exp(r), 1.0-np.exp(r)])
            #    ttemp.append(newset[0][0])
            #    logp.append(newset[0][1]) 
            #    if newset[0][0].all() == tprop.all() and t > self.discard:
            #        
            #        accept = accept+ 1
#
#            print "t: " + str(t)
#
#            ### draw value from proposal distribution
#            tprop = dist(ttemp[t-1], self.tcov)
#            #print "tprop: " + str(tprop)
#            ### calculate log posterior density
#            pprop = lpost(tprop, self.func, ps)
#            print "pprop: " + str(pprop)
#            print "old pprop: " + str(logp[t-1])
#            ### compute ratio of posteriors at new and old
#            ### locations in terms of log posteriors
#            logr = logp[t-1] - pprop
#            #logr = pprop - logp[t-1]
#            print "logr: " + str(logr)
#
#            logr = min(logr, 0.0)
#            r = np.exp(logr)
#            print "r: " + str(r)
#            if r > 1.01:
#                print("tprop: " + str(tprop))
#                print("pprop: " + str(pprop))
#                print("log(r): " + str(logr))
#                print("r: " + str(r))
#
#                raise Exception("r > 1! This cannot be true!")
#
#            ### with chance r, the number is updated 
#            update = choice([True, False], size=1, weights=[r,1.0-r])
#            #print "update" + str(update)
#
#
#            if update:
#                ttemp.append(tprop)
#                logp.append(pprop)
#                if t > self.discard:
#                    accept = accept+1
#            else:
#                ttemp.append(ttemp[t-1])
#                logp.append(logp[t-1])
#     
        #print "discard: " + str(self.discard)
        #print "self.discard: " + str(self.discard)

        self.theta = ttemp[self.discard+1:]
        self.logp = logp[self.discard+1:]
        self.L = self.niter - self.discard
        #print "self.niter: " + str(self.niter)
        #print "self.L: " + str(self.L)
        self.accept = accept/self.L
        return

    def run_diagnostics(self, namestr=None, paraname=None, printobj = None):

        if printobj:
            print = printobj
        else:
            from __builtin__ import print as print

        print("Markov Chain acceptance rate: " + str(self.accept) +".")

        if namestr == None:
            print("No file name string given for printing. Setting to 'test' ...")
            namestr = 'test'

        if paraname == None:
           paraname = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'iota', 'lappa', 'lambda', 'mu']


        fig = plt.figure(figsize=(12,10))
        adj =plt.subplots_adjust(hspace=0.4, wspace=0.4)

        for i,th in enumerate(self.theta[0]):
            #print "i: " + str(i)
            ts = np.array([t[i] for t in self.theta]) 
  
            #print "(i*3)+1: " + str((i*3)+1)
            ### plotting time series ###
            p1 = plt.subplot(len(self.topt), 3, (i*3)+1)
            p1 = plt.plot(ts)
            plt.axis([0, len(ts), min(ts), max(ts)])
            plt.xlabel("Number of draws")
            plt.ylabel("parameter value")
            plt.title("Time series for parameter " + str(paraname[i]) + ".")

            ### make a normal distribution to compare to
            #tsnorm = [np.random.normal(self.tcov[i], self.terr[i]) for x in range(len(ts)) ]

            p2 = plt.subplot(len(self.topt), 3, (i*3)+2)


            ### plotting histogram
            p2 = count, bins, ignored = plt.hist(ts, bins=10, normed=True)
            bnew = np.arange(bins[0], bins[-1], (bins[-1]-bins[0])/100.0)
            p2 = plt.plot(bnew, 1.0/(self.terr[i]*np.sqrt(2*np.pi))*np.exp(-(bnew - self.topt[i])**2.0/(2.0*self.terr[i]**2.0)), linewidth=2, color='r')
            plt.xlabel('value of ' + str(paraname[i]))
            plt.ylabel('probability')
            plt.title("Histogram for parameter " + str(paraname[i]) + ".")


            nlags = 30

            p3 = plt.subplot(len(self.topt), 3, (i*3)+3)
            acorr = gt.autocorr(ts,nlags=nlags, norm=True)
            p3 = plt.vlines(range(nlags), np.zeros(nlags), acorr, colors='black', linestyles='solid')
            plt.axis([0.0, nlags, 0.0, 1.0])
        #plt.show()
        plt.savefig(namestr  + "_diag.png", format='png',orientation='landscape')
        plt.close()


##############################################################
    



