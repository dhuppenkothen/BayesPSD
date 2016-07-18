
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import math
import sys

import scipy
import scipy.optimize
from scipy.stats.mstats import mquantiles as quantiles
import scipy.stats

### New: added possibility to use emcee for MCMCs
try:
    import emcee
#    import acor
    emcee_import = True
except ImportError:
    print("Emcee and Acor not installed. Using Metropolis-Hastings algorithm for Markov Chain Monte Carlo simulations.")
    emcee_import = False

from BayesPSD import utils
from BayesPSD import powerspectrum



### See if cutting-edge numpy is installed so I can use choice
try:
    from numpy.random import choice
     ### if not, use hack
except ImportError:
    choice = utils.choice_hack





class MarkovChainMonteCarlo(object):
    """
    Markov Chain Monte Carlo for Bayesian QPO searches.

    Either wraps around emcee, or uses the
    Metropolis-Hastings sampler defined in this file.

    Parameters
    ----------

    x : {list, array-like}
        Inependent variable, most likely the frequencies of the
        periodogram in this context.

    y : {list, array-like}
        Dependent variable, most likely the powers of the
        periodogram in this context.

    lpost : Posterior object
        An instance of the class Posterior or one of its subclasses;
        defines the likelihood and priors to be used.
        For periodograms, use
            * posterior.PerPosterior for unbinned periodograms
            * posterior.StackPerPosterior for binned/stacked periodograms

    topt : {list, array-like}
        Starting point for generating an initial set of parameter samples.
        Should be in a region of high posterior, such that the chains
        don't spend a long time exploring regions with low posterior mass.
        If possible, make a MAP fit and use the MAP parameters here.

        The length of topt needs to match the number of parameters used
        in whatever function is stored in lpost.func

    tcov: {array-like}
        The variances and covarianced between parameters used to generate an
        initial set of parameter samples for all chains/walkers.

        There are several options here: you can set large variances and no
        covariances and effectively leave the Markov chains to explore
        the prior mass until they converge. You can also use the inverse
        Fisher information (as for example returned by bfgs) as covariance
        matrix to make an initial guess. This usually works better in the sense
        that it requires fewer steps of the Markov chains.

        popt needs to have dimensions (k,k), where k is the number of parameters
        taken by lpost.func

    covfactor : float, optional, default 1.0
        A tuning parameter for the MCMC step. Used only in
        Metropolis-Hastings.

    niter : int, optional, default 5000
        Sets the length of the Markov chains.
        For Metropolis-Hastings, this needs to be large (>10000)
        For emcee, this can be smaller, but it's a good idea to
        verify that the chains have mixed.

    nchain : int, optional, default 10
        The number of chains or walkers to use in MCMC.
        For Metropolis-Hastings, use ~10-20 and many samples
        For emcee, use as many as you can afford (~500) and fewer samples

    discard : {int, None}, optional, default None
        The number of initial samples to discard from the Markov chain.

        For emcee, the burn-in time is *always* 200 samples (additional to
        whatever is set by niter).

        For the Metropolis-Hastings algorithm, the number of initial samples
        discarded is set by this variable.
        If discard is None, then half of the samples are discarded as default.

    parname : list, optional, default None
        Include a list of strings here to set parameter names for
        plotting

    check_conv : boolean, optional, default True
        If True, check for convergence of the Markov chains using check_convergence
        method below.

        NOTE: This was set up explicitly for Metropolis-Hastings. For emcee,
        this might not necessarily produce easily interpretable results.

    namestr : string, optional, default 'test'
        a string to use for saving plots and output files

    use_emcee : boolean, optional, default True
        If True (STRONGLY RECOMMENDED), use the emcee package
        for running MCMC. If False, use Metropolis-Hastings.

    plot : boolean, optional, default True
        If True, then save some useful plots; in particular,
        convergence plots as well as a triangle plot showing
        the posterior distributions

    printobj : object, optional, default None
        In theory, this allows the use of an alternative
        to the standard print function in order to save
        information to file etc.

        NOTE: CURRENTLY DOESN'T WORK PROPERLY!

    m : int, optional, default 1
        If the input periodogram is the result of stacking
        several individual periodograms, or the result of
        binning adjacent frequencies into a coarser frequency
        resolution, then the distribution to be used in the
        likelihood function is different!
        Set the number of periodograms averaged/stacked here.


    """


    def __init__(self, x, y, lpost, topt, tcov,
                 covfactor=1.0,
                 niter=5000,
                 nchain=10,
                 discard=None,
                 parname = None,
                 check_conv = True,
                 namestr='test',
                 use_emcee=True,
                 plot=True,
                 printobj = None,
                 m=1):


        self.m = m

        self.x = x
        self.y = y

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

            ### number of walkers is the number of chains
            nwalkers = self.nchain
            ### number of dimensions for the Gaussian (=number of parameters)
            ndim = len(self.topt)

            ### sample random starting positions for each of the walkers
            p0 = [np.random.multivariate_normal(self.topt,self.tcov) for i in xrange(nwalkers)]


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
                mcout = MetropolisHastings(topt, tcov, lpost, niter = niter, parname = parname, discard = discard)
                ### create actual chain
                mcout.create_chain(self.x, self.y)

                ### make diagnostic plots
                mcout.run_diagnostics(namestr = namestr +"_c"+str(i), parname=parname)

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


    def check_convergence(self, mcall, namestr, printobj=None, use_emcee = True):

        #if printobj:
        #    print = printobj
        #else:
        #    from __builtin__ import print as print

        ### compute Rhat for all parameters
        rh = self._rhat(mcall, printobj)
        self.rhat = rh

        plt.scatter(rh, np.arange(len(rh))+1.0 )
        plt.axis([0.1,2,0.5,0.5+len(rh)])
        plt.xlabel("$R_hat$")
        plt.ylabel("Parameter")
        plt.title('Rhat')
        plt.savefig(namestr + '_rhat.png', format='png')
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
            plt.savefig(namestr + "_quantiles.png", format="png")
            plt.close()

    ### auxiliary function used in check_convergence
    ### computes R_hat, which compares the variance inside chains to the variances between chains
    def _rhat(self, mcall, printobj = None):

        #if printobj:
        #    print = printobj
        #else:
        #    from __builtin__ import print as print

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


    def mcmc_infer(self, namestr='test', printobj = None):

        #if printobj:
        #    print = printobj
        #else:
        #    from __builtin__ import print as print


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

        ### produce matrix scatter plots

        N = len(self.topt) ### number of parameters
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
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.ticklabel_format(style="sci", scilimits=(-2,2))

                    if i == j:
                        #pass
                        ntemp, binstemp, patchestemp = ax.hist(self.mcall[i][:1000], 30, normed=True, histtype='stepfilled')
                        n.append(ntemp)
                        bins.append(binstemp)
                        patches.append(patchestemp)
                        ax.axis([ymin, ymax, 0, max(ntemp)*1.2])

                    else:

                        ax.axis([xmin, xmax, ymin, ymax])

                        ### make a scatter plot first
                        ax.scatter(self.mcall[j][:1000], self.mcall[i][:1000], s=7)
                        ### then add contours
                        xmin, xmax = self.mcall[j][:1000].min(), self.mcall[j][:1000].max()
                        ymin, ymax = self.mcall[i][:1000].min(), self.mcall[i][:1000].max()

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
    def simulate_periodogram(self, nsim=5000):
        """
        Simulate periodograms from posterior samples of the
        broadband noise model.

        This method uses the results of an MCMC run to
        pick samples from the posterior and use the function
        stored in self.lpost.func to create a power spectral form.

        In order to transform this into a model periodogram,
        it picks for each frequency from an exponential distribution
        with a shape parameter corresponding to the model power
        at that frequency.

        Parameters
        ----------

        nsim : int, optional, default 5000
            The number of periodograms to simulate. This number
            must be smaller than the number of samples generated
            during the MCMC run.

        Returns
        -------
        fps : array-like
            An array of shape (nsim, nfrequencies) with all
            simulated periodograms.

        """

        ## the function to use is stored in lpost:
        func = self.lpost.func

        ### number of simulations is either given by the user,
        ### or defined by the number of MCMCs run!
        nsim = min(nsim,len(self.mcall[0]))

        ### shuffle MCMC parameters
        theta = np.transpose(self.mcall)
        #print "theta: " + str(len(theta))
        np.random.shuffle(theta)

        fps = []
        percount = 1.0

        for x in range(nsim):

            ### extract parameter set
            ain = theta[x]
            ### compute model 'true' spectrum
            mpower = func(self.x, *ain)

            ### define distribution
            if self.m == 1:
                #print("m = 1")
                noise = np.random.exponential(size=len(self.x))
            else:
                #print("m = " + str(self.m))
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
            mps.m = self.m

            fps.append(mps)

        return np.array(fps)





#### MAKE A MARKOV CHAIN OBJECT ###
#
# QUESTION: How can I make an object with variable
# parameters?
#
#
#
#  NEED TO THINK ABOUT HOW TO GET ATTRIBUTES!
#
class MetropolisHastings(object):

    """
    Parameters
    ----------

    topt : {list, array-like}
        Starting point for generating an initial set of parameter samples.
        Should be in a region of high posterior, such that the chains
        don't spend a long time exploring regions with low posterior mass.
        If possible, make a MAP fit and use the MAP parameters here.

        The length of topt needs to match the number of parameters used
        in whatever function is stored in lpost.func

    tcov: {array-like}
        The variances and covarianced between parameters used to generate an
        initial set of parameter samples for all chains/walkers.

        There are several options here: you can set large variances and no
        covariances and effectively leave the Markov chains to explore
        the prior mass until they converge. You can also use the inverse
        Fisher information (as for example returned by bfgs) as covariance
        matrix to make an initial guess. This usually works better in the sense
        that it requires fewer steps of the Markov chains.

        popt needs to have dimensions (k,k), where k is the number of parameters
        taken by lpost.func

    lpost : Posterior object
        An instance of the class Posterior or one of its subclasses;
        defines the likelihood and priors to be used.
        For periodograms, use
            * posterior.PerPosterior for unbinned periodograms
            * posterior.StackPerPosterior for binned/stacked periodograms

    niter : int, optional, default 5000
        Sets the length of the Markov chains.
        For Metropolis-Hastings, this needs to be large (>10000)
        For emcee, this can be smaller, but it's a good idea to
        verify that the chains have mixed.

    parname : list, optional, default None
        Include a list of strings here to set parameter names for
        plotting

    discard : {int, None}, optional, default None
        The number of initial samples to discard from the Markov chain.

        For emcee, the burn-in time is *always* 200 samples (additional to
        whatever is set by niter).

        For the Metropolis-Hastings algorithm, the number of initial samples
        discarded is set by this variable.
        If discard is None, then half of the samples are discarded as default.

    """

    def __init__(self, topt, tcov, lpost, niter = 5000,
                 parname=None, discard=None):

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
        if parname == None:
            self.parname = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'iota', 'lappa', 'lambda', 'mu']
        else:
            self.parname = parname

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


        ### set acceptance value to zero
        accept = 0.0

        ### set up array
        ttemp, logp = [], []
        ttemp.append(self.t0)
        #lpost = posterior.PerPosterior(self.ps, self.func)
        logp.append(self.lpost(self.t0, neg=False))

        for t in np.arange(self.niter-1)+1:

            tprop = dist(ttemp[t-1], self.tcov)

            pprop = self.lpost(tprop)#, neg=False)

            logr = pprop - logp[t-1]
            logr = min(logr, 0.0)
            r= np.exp(logr)
            update = choice([True, False], size=1, p=[r, 1.0-r])

            if update:
                ttemp.append(tprop)
                logp.append(pprop)
                if t > self.discard:
                     accept = accept + 1
            else:
                ttemp.append(ttemp[t-1])
                logp.append(logp[t-1])

        self.theta = ttemp[self.discard+1:]
        self.logp = logp[self.discard+1:]
        self.L = self.niter - self.discard
        self.accept = accept/self.L
        return

    def run_diagnostics(self, namestr=None, parname=None, printobj = None):

        #if printobj:
        #    print = printobj
        #else:
        #    from __builtin__ import print as print

        print("Markov Chain acceptance rate: " + str(self.accept) +".")

        if namestr == None:
            print("No file name string given for printing. Setting to 'test' ...")
            namestr = 'test'

        if parname == None:
           parname = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'iota', 'lappa', 'lambda', 'mu']

        fig = plt.figure(figsize=(12,10))
        adj =plt.subplots_adjust(hspace=0.4, wspace=0.4)

        for i,th in enumerate(self.theta[0]):
            ts = np.array([t[i] for t in self.theta])

            p1 = plt.subplot(len(self.topt), 3, (i*3)+1)
            p1 = plt.plot(ts)
            plt.axis([0, len(ts), min(ts), max(ts)])
            plt.xlabel("Number of draws")
            plt.ylabel("parameter value")
            plt.title("Time series for parameter " + str(parname[i]) + ".")

            p2 = plt.subplot(len(self.topt), 3, (i*3)+2)

            ### plotting histogram
            p2 = count, bins, ignored = plt.hist(ts, bins=10, normed=True)
            bnew = np.arange(bins[0], bins[-1], (bins[-1]-bins[0])/100.0)
            p2 = plt.plot(bnew, 1.0/(self.terr[i]*np.sqrt(2*np.pi))*np.exp(-(bnew - self.topt[i])**2.0/(2.0*self.terr[i]**2.0)), linewidth=2, color='r')
            plt.xlabel('value of ' + str(parname[i]))
            plt.ylabel('probability')
            plt.title("Histogram for parameter " + str(parname[i]) + ".")

            nlags = 30

            p3 = plt.subplot(len(self.topt), 3, (i*3)+3)
            acorr = autocorr(ts,nlags=nlags, norm=True)
            p3 = plt.vlines(range(nlags), np.zeros(nlags), acorr, colors='black', linestyles='solid')
            plt.axis([0.0, nlags, 0.0, 1.0])

        plt.savefig(namestr  + "_diag.png", format='png',orientation='landscape')
        plt.close()


##############################################################





