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

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import cPickle as pickle
import copy

### GENERAL IMPORTS ###
import numpy as np


### OWN SCRIPTS
import utils
import powerspectrum
import mcmc

import mle
import posterior





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
    """ Bayesian time series analysis

    This class defines a Bayes object that can:
    - pick between two models using likelihood ratio tests
    - find periodicities by picking out the largest power in
      an observation/set of fake periodograms
    - search for QPOs via a model selection approach using LRTs

    Parameters
    ----------
    ps : powerspectrum.Powerspectrum
        A periodogram object that is to be searched for QPOs

    namestr: string, optional, default "test"
        The string that will be used to identify this periodogram when
        saving output (text files and plots)

    plot: boolean, optional, default True
        If True, several diagnostic plots will be saved to disk

    m: integer, optional, default 1
        If the periodogram used is the result of averaging several
        individual periodograms (or bins), this changes the statistical
        distributions. Set m to the number of periodograms
        averaged to be sure to use the right distribution


    Attributes
    ----------

    Examples
    --------

    """

    def __init__(self, ps, namestr='test', plot=True, m=1):
        assert isinstance(ps, powerspectrum.PowerSpectrum), "ps must be of type powerspectrum.PowerSpectrum!"
        self.ps = ps
        self.namestr = namestr
        self.plot=plot
        self.m = m


    def choose_noise_model(self, func1, par1, func2, par2,
                     fitmethod='bfgs',
                     nchain = 10,
                     niter = 5000,
                     nsim = 1000,
                     covfactor = 1.0,
                     use_emcee = True,
                     parname = None,
                     noise1 = -1,
                     noise2 = -1,
                     writefile= True):

        """
        Fit two models func1 and func2, compute the likelihood
        ratio at the maximum-a-posteriori paramters.
        If func1 and func2 differ in complexity, the less complex
        should be func1.

        Then sample the posterior distribution for the the simpler
        model (func1), pick parameter sets from the posterior
        to create fake periodograms.

        Fit each fake periodogram with the same models as the data, and
        compute the likelihood ratios such that it is possible to
        build up a posterior distribution for the likelihood
        ratios and compute a posterior predictive p-value
        that the data can be explained sufficiently with the simpler
        model.

        Parameters
        ----------
        func1 : function
            Parametric model for the periodogram.
            Needs to be a function that takes an array of frequencies and
            k parameters, and returns an array of model powers.
            The function should include a parameter setting a constant background
            level, and this parameter should be last!

        par1 : {list, array-like}
            Input guesses for the MAP fit using func1.
            The number of elements *must* equal the number of parameters k
            taken by func1.

        func2 : function
            Parametric model for the periodogram.
            Needs to be a function that takes an array of frequencies and n
            parameters, and returns an array of model powers
            The function should include a parameter setting a constant background
            level, and this parameter should be last!

        par2 : {list, array-like}
            Input guesses for the MAP fit using func2.
            The number of elements *must* equal the number of parameters n
            taken by func2.


        fitmethod : string, optional, default bfgs
            Allows the choice of different minimization algorithms.
            Default uses BFGS, which is pretty robust for most purposes.


        nchain : int, optional, default 10
            The number of chains or walkers to use in MCMC.
            For Metropolis-Hastings, use ~10-20 and many samples
            For emcee, use as many as you can afford (~500) and fewer samples

        niter : int, optional, default 5000
            Sets the length of the Markov chains.
            For Metropolis-Hastings, this needs to be large (>10000)
            For emcee, this can be smaller, but it's a good idea to
            verify that the chains have mixed.

        nsim : int, optional, default 1000
            The number of simulations to use when computing the
            posterior distribution of the likelihood ratio.
            Note that this also sets the maximum precision of the
            posterior predictive p-value (for 1000 simulations, the
            p-value can be constrained only to 0.001).

        covfactor : float, optional, default 1.0
            A tuning parameter for the MCMC step. Used only in
            Metropolis-Hastings.

        use_emcee : boolean, optional, default True
            If True (STRONGLY RECOMMENDED), use the emcee package
            for running MCMC. If False, use Metropolis-Hastings.

        parname : list, optional, default None
            Include a list of strings here to set parameter names for
            plotting

        noise1, noise2 : int, optional, default -1
            The index for the noise parameter in func1 and func2.
            In the pre-defined models, this index is *always* -1.


        """

        resfilename = self.namestr + "_choosenoisemodel.dat"
        resfile = utils.TwoPrint(resfilename)


        ### make strings for function names from function definition
        func1name = str(func1).split()[1] 
        func2name = str(func2).split()[1]

        ### step 1: fit both models to observation and compute LRT
        psfit = mle.PerMaxLike(self.ps, fitmethod=fitmethod, obs=True)
        obslrt = psfit.compute_lrt(func1, par1, func2, par2, noise1=noise1, noise2=noise2, m=self.m)

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
        mcobs = mcmc.MarkovChainMonteCarlo(self.ps.freq, self.ps.ps, lpost,
                                      topt = fitpars1['popt'], 
                                      tcov = fitpars1['cov'], 
                                      covfactor = covfactor, 
                                      niter=niter, 
                                      nchain=nchain,
                                      parname= parname,
                                      check_conv = True,
                                      namestr = self.namestr,
                                      use_emcee = use_emcee,
                                      plot = self.plot,
                                      printobj = resfile,
                                      m = self.m) 


        ### Step 3: create fake periodograms out of MCMCs
        fakeper = mcobs.simulate_periodogram(func1, nsim = nsim)

        ### empty lists for simulated quantities of interest:
        sim_lrt, sim_deviance, sim_ksp, sim_maxpow, sim_merit, sim_fpeak, sim_y0, sim_srat = [], [], [], [], [], [], [], []

        ### Step 4: Fit fake periodograms and read out parameters of interest from each fit:
        for i,x in enumerate(fakeper):
            fitfake = mle.PerMaxLike(x, fitmethod=fitmethod, obs=False)
            try: 
                lrt = fitfake.compute_lrt(func1, par1, func2, par2, noise1=noise1, noise2=noise2, m=self.m)

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


            summary = {"p_lrt":[p_lrt, plrt_err], "p_maxpow":[p_maxpow, pmaxpow_err], "p_deviance":[p_deviance, pdeviance_err], "p_ksp":[p_ksp, pksp_err], "p_merit":[p_merit, pmerit_err], "p_srat":[p_srat, psrat_err], "postmean":mcobs.mean, "posterr":mcobs.std, "postquantiles":mcobs.ci,"rhat":mcobs.rhat, "acor":mcobs.acor, "acceptance":mcobs.acceptance}


            return psfit,fakeper, summary


    def find_periodicity(self, func, par,
                 fitmethod='bfgs',
                 nchain = 10,
                 niter = 5000,
                 nsim = 1000,
                 covfactor = 1.0,
                 parname = None,
                 noise = -1,
                 use_emcee = True):


        """
        Find periodicities in observed data and compute significance via MCMCs.

        First, fit the periodogram with func and compute the
        maximum-a-posteriori (MAP) estimate.
        Divide the data by the MAP model; for a perfect data-model fit,
        the resulting residuals should follow a chi-square distribution
        with two degrees of freedom.
        Find the highest power in the residuals and its frequency.

        Sample the posterior distribution of parameters for func using MCMC,
        and create fake periodograms from samples of the posterior.
        For each fake periodogram, find the MAP estimate, divide out the
        MAP model and find the highest power in that periodogram.

        Create a posterior distribution of maximum powers and compute
        a posterior predictive p-value of seeing the maximum power
        in the data under the null hypothesis (no QPO).


        Parameters
        ----------

        func : function
            Parametric model for the periodogram.
            Needs to be a function that takes an array of frequencies and
            k parameters, and returns an array of model powers.
            The function should include a parameter setting a constant background
            level, and this parameter should be last!

        par : {list, array-like}
            Input guesses for the parameters taken by func.
            The number of elements in this list or array must match the
            number of parameters k taken by func.

        fitmethod : string, optional, default "bfgs"
            Choose the optimization algorithm used when minimizing the
            -log-likelihood. Choices are listed in mle.py, but the default
            (bfgs) should be sufficient for most applications.

        nchain : int, optional, default 10
            The number of chains or walkers to use in MCMC.
            For Metropolis-Hastings, use ~10-20 and many samples
            For emcee, use as many as you can afford (~500) and fewer samples

        niter : int, optional, default 5000
            Sets the length of the Markov chains.
            For Metropolis-Hastings, this needs to be large (>10000)
            For emcee, this can be smaller, but it's a good idea to
            verify that the chains have mixed.

        nsim : int, optional, default 1000
            The number of simulations to use when computing the
            posterior distribution of the likelihood ratio.
            Note that this also sets the maximum precision of the
            posterior predictive p-value (for 1000 simulations, the
            p-value can be constrained only to 0.001).

        covfactor : float, optional, default 1.0
            A tuning parameter for the MCMC step. Used only in
            Metropolis-Hastings.


        parname : list, optional, default None
            Include a list of strings here to set parameter names for
            plotting

        noise: int, optional, default -1
            The index for the noise parameter in func.
            In the pre-defined models, this index is *always* -1.

        use_emcee : boolean, optional, default True
            If True (STRONGLY RECOMMENDED), use the emcee package
            for running MCMC. If False, use Metropolis-Hastings.


        """


        ## the file name where the output will be stored
        resfilename = self.namestr + "_findperiodicity_results.dat"

        ## open the output log file
        resfile = utils.TwoPrint(resfilename)


        ### step 1: fit model to observation
        psfit = mle.PerMaxLike(self.ps, fitmethod=fitmethod, obs=True)
        fitpars = psfit.mlest(func, par, obs=True, noise=noise, m=self.m)
        bindict = fitpars['bindict']
        #print('popt: ' + str(fitpars['popt']))

        ## which posterior do I need to use?
        if self.m == 1:
            lpost = posterior.PerPosterior(self.ps, func)
        else:
            lpost = posterior.StackPerPosterior(self.ps, func, self.m)



        ### Step 2: Set up Markov Chain Monte Carlo Simulations
        ### of model 1:
        mcobs = mcmc.MarkovChainMonteCarlo(self.ps.freq, self.ps.ps, lpost,
                                      topt = fitpars['popt'],
                                      tcov = fitpars['cov'],
                                      covfactor = covfactor,
                                      niter=niter,
                                      nchain=nchain,
                                      parname= parname,
                                      check_conv = True,
                                      namestr = self.namestr,
                                      use_emcee = True,
                                      plot=self.plot, 
                                      printobj = resfile,
                                      m = self.m)


        ### Step 3: create fake periodograms out of MCMCs
        fakeper = mcobs.simulate_periodogram(func, nsim = nsim)

        sim_pars_all, sim_deviance, sim_ksp, sim_fpeak, sim_srat, \
        sim_maxpow, sim_merit, sim_y0, sim_s3max, sim_s5max, sim_s11max =[], [], [], [], [], [], [], [], [], [], []

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

                sim_pars = fitfake.mlest(func, sain,obs=False, noise=noise, m=self.m)
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


        #print('popt4: ' + str(fitpars['popt']))
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

            #print('len(binps.freq): ' + str(len(binps.freq)))
            #print('len(binpowers): ' + str(len(binpowers)))


            ## for 40 Hz: 
            for bc in [40.0, 70.0, 100.0, 300.0, 500.0, 1000.0]:
                if bc > (binps.freq[1] - binps.freq[0]):
                    bind = np.searchsorted(binps.freq, bc) - 1
                    bpow = binpowers[bind]
                    brms = np.sqrt(bpow*b*self.ps.df)

                    resfile('The upper limit on the power at ' + str(bc) +
                            'Hz for a binning of ' + str(b) + ' is P = ' +
                            str(bpow*(self.ps.df*b*self.ps.nphots)))

                    resfile('The upper limit on the rms amplitude at ' + str(bc) +
                            'Hz for a binning of ' + str(b) + ' is rms = ' + str(brms))

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


        ### Step 6: Compute errors of Bayesian posterior probabilities
        pmaxpow_err = np.sqrt(p_maxpow*(1.0-p_maxpow)/float(len(sim_ksp)))
        pdeviance_err = np.sqrt(p_deviance*(1.0-p_deviance)/float(len(sim_ksp)))
        pksp_err = np.sqrt(p_ksp*(1.0-p_ksp)/float(len(sim_ksp)))
        pmerit_err = np.sqrt(p_merit*(1.0-p_merit)/float(len(sim_ksp)))
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



    def find_qpo(self, func, ain,
                 fitmethod='constbfgs',
                 nchain = 10,
                 niter = 5000,
                 nsim = 1000,
                 covfactor = 1.0,
                 parname = None,
                 plotstr=None,
                 use_emcee = True):

        """
        Find QPOs by fitting a QPO + background model to *every*
        frequency.
        NOTE: I rarely ever use this because it's really computationally
        expensive.

        Parameters
        ----------

        func : function
            Parametric model for the periodogram.
            Needs to be a function that takes an array of frequencies and
            k parameters, and returns an array of model powers.
            The function should include a parameter setting a constant background
            level, and this parameter should be last!

        par : {list, array-like}
            Input guesses for the parameters taken by func.
            The number of elements in this list or array must match the
            number of parameters k taken by func.

        fitmethod : string, optional, default "bfgs"
            Choose the optimization algorithm used when minimizing the
            -log-likelihood. Choices are listed in mle.py, but the default
            (bfgs) should be sufficient for most applications.

        nchain : int, optional, default 10
            The number of chains or walkers to use in MCMC.
            For Metropolis-Hastings, use ~10-20 and many samples
            For emcee, use as many as you can afford (~500) and fewer samples

        niter : int, optional, default 5000
            Sets the length of the Markov chains.
            For Metropolis-Hastings, this needs to be large (>10000)
            For emcee, this can be smaller, but it's a good idea to
            verify that the chains have mixed.

        nsim : int, optional, default 1000
            The number of simulations to use when computing the
            posterior distribution of the likelihood ratio.
            Note that this also sets the maximum precision of the
            posterior predictive p-value (for 1000 simulations, the
            p-value can be constrained only to 0.001).

        covfactor : float, optional, default 1.0
            A tuning parameter for the MCMC step. Used only in
            Metropolis-Hastings.


        parname : list, optional, default None
            Include a list of strings here to set parameter names for
            plotting

        noise: int, optional, default -1
            The index for the noise parameter in func.
            In the pre-defined models, this index is *always* -1.

        use_emcee : boolean, optional, default True
            If True (STRONGLY RECOMMENDED), use the emcee package
            for running MCMC. If False, use Metropolis-Hastings.


        """


        if plotstr == None:
            plotstr = self.namestr


        funcname = str(func).split()[1]

        #print("<< --- len(self.ps beginning): " + str(len(self.ps.ps)))

        ### step 1: fit model to observation
        psfit = mle.PerMaxLike(self.ps, fitmethod=fitmethod, obs=True)
        fitpars = psfit.mlest(func, ain, obs=True, noise=-1, m=self.m)

        #print("<< --- len(self.ps beginning): " + str(len(self.ps.ps)))





        if self.m == 1:
            lpost = posterior.PerPosterior(self.ps, func)
        else:
            lpost = posterior.StackPerPosterior(self.ps, func, self.m)


        ### Step 2: Set up Markov Chain Monte Carlo Simulations
        ### of model 1:
        mcobs = mcmc.MarkovChainMonteCarlo(self.ps.freq, self.ps.ps, lpost,
                                      topt = fitpars['popt'],
                                      tcov = fitpars['cov'],
                                      covfactor = covfactor,
                                      niter=niter,
                                      nchain=nchain,
                                      parname= parname,
                                      check_conv = True,
                                      namestr = self.namestr,
                                      use_emcee = True,
                                      plot = self.plot,
                                      m = self.m)


        ### find optimum QPO values for the real data
        obslrt, optpars, qpopars = psfit.find_qpo(func, ain, plot=True, obs=True, plotname = self.namestr+'_loglikes')


        ### simulate lots of realizations of the broadband noise model from MCMCs
        funcfake = mcobs.simulate_periodogram(func, nsim = nsim)

        ### empty lists to store simulated LRTS and parameters in
        sim_lrt, sim_optpars, sim_qpopars, sim_deviance, sim_ksp, sim_merit, sim_srat = [], [], [], [], [], [], []

        simno = 0

        ### run QPO search on each and return likelihood ratios parameters for each
        for x in funcfake:
            simno = simno + 1
            sim_psfit = mle.PerMaxLike(x, fitmethod='constbfgs',obs=False)
            slrt, soptpars, sqpopars = sim_psfit.find_qpo(func, ain, obs=False, plot=True, plotname = plotstr + '_sim' + str(simno) + '_qposearch')

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

        summary = {"p_lrt":[p_lrt, plrt_err],
                   "p_deviance":[p_deviance, pdeviance_err],
                   "p_ksp":[p_ksp, pksp_err],
                   "p_merit":[p_merit, pmerit_err],
                   "p_srat":[p_srat, psrat_err],
                   "postmean":mcobs.mean,
                   "posterr":mcobs.std,
                   "postquantiles":mcobs.ci,
                   "rhat":mcobs.rhat,
                   "acor":mcobs.acor,
                   "acceptance":mcobs.acceptance}

        return summary



    def print_summary(self,summary):
        """
        Print a summary of the results.

        NOT USED!
        """

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
            print("The $R_hat$ value for Parameter " + str(i) + " is " + str(x))


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

        """
        Write a summary of the analysis to file.

        NOT USED!

        :param summary:
        :param namestr:
        :return:
        """

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
            file.write("The $R_hat$ value for Parameter " + str(i) + " is " + str(x) + "\n")


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




