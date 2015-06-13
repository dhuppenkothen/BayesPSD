import numpy as np
import scipy.misc
 
from .parametricmodels import pl, bpl, qpo, const


logmin = -10000000000000000.0


class Posterior(object):

    def __init__(self,x, y, func):
        self.x = x
        self.y = y

        ### func is a parametric model
        self.func = func

    def _const_prior(self, t0):
        pr = 1.0
        return pr

    ### simplest uninformative log-prior, define alternatives in subclasses!
    def logprior(self, t0):
        ### completely uninformative prior: flat distribution for all parameters
        pr = 1.0 

        return np.log(pr)

    ### use standard definition of the likelihood as the product of all 
    def loglikelihood(self, t0, neg=False):
        loglike = np.sum(np.array([self.func(x, t0) for x in self.x]))
        return loglike

    def __call__(self, t0, neg=False):
        lpost = self.loglikelihood(t0) + self.logprior(t0)

        if neg == True:
            return lpost
        else:
            return -lpost
    

class PerPosterior(Posterior):
    ### initialize posterior object with power spectrum 
    ### and broadband noise model function
    ### note: at the moment, only mle.pl and mle.bpl are supported
    def __init__(self, ps, func):
       self.ps=ps
       #print('I am here!')
       Posterior.__init__(self,ps.freq[1:], ps.ps[1:], func)
       #super(Posterior,self).__init__(ps.freq[1:], ps.ps[1:], func)

    ### prior densities for PL model
    ### choose uninformative priors
    def pl_prior(self,t0):
        alim = [-8.0, 8.0]


        ### power law index always first value
        alpha = t0[0]
        ### palpha is True if it is within bounds and false otherwise
        ### then pr = pbeta*pgamma if palpha = True, 0 otherwise
        palpha = (alpha >= alim[0] and alpha <= alim[1])
        ### normalization always second parameter
        #beta = t0[1]
#        palpha = 1.0
        pbeta = 1.0

        pgamma = 1.0

        #if (alpha < alim[0]) or (alpha > alim[1]):
        #    return np.inf
        #else:
        return palpha*pbeta*pgamma

    ### prior densities for BPL model
    ### choose uninformative priors
    def bpl_prior(self,t0):
        pr0 = self.pl_prior([t0[2], t0[1], t0[-1]])
        #print("pr0: " + str(pr0))
        delta =t0[3]
        pdelta = (delta >= min(np.log(self.ps.freq)))

        peps = 1.0
        return pr0*pdelta*peps


    def plqpo_prior(self, t0):

        noise = t0[2]
        pnoise = scipy.stats.norm.pdf(noise, 0.6932, 0.2)/2.0
        if pnoise < 0.01:
            pnoise = 0.0

        gamma = t0[3]

        gamma_min = np.log(self.ps.df)
        gamma_max = np.log(nu0/2.0)

        pgamma = (gamma >= gamma_min and gamma <= gamma_max)

        norm = t0[4]
        pnorm = ( 20.0 >= norm >= -10.0)
     
        pr0 = 1.0

        try:
            nu = t0[5]
            pnu = scipy.stats.norm.pdf(nu, 93.0, 5)/0.08
            if pnu < 0.01:
                pnu = 0.0
            return pr0*pnoise*pgamma*pnorm*pnu
        except IndexError:
            return pr0*pnoise*pgamma*pnorm      

    def bplqpo_prior(self, t0):


        noise = t0[4]
        pnoise = scipy.stats.norm.pdf(noise, 0.6932, 0.2)/2.0
        if pnoise < 0.01:
            pnoise = 0.0

        gamma = t0[5]

        gamma_min = np.log(self.ps.df)
        gamma_max = np.log(self.ps.freq[-1]/2.0)

        pgamma = (gamma >= gamma_min and gamma <= gamma_max)

        norm = t0[6]
        pnorm = ( 50.0 >= norm >= -5.0)


        pr0 = 1.0
        return pr0*pgamma*pnorm


    def qpofixprior(self, t0):
        print("Using QPO fix prior")
        pr0 = self.pl_prior(t0[:3])
        pnorm = 1.0
        return pnorm*pr0


    def qpo_prior(self,t0):

        gamma = t0[0]
        nu0 = t0[2]
        norm = t0[1]

        gamma_min = np.log(self.ps.df)
        gamma_max = np.log(nu0/2.0)

        pgamma = (gamma >= gamma_min and gamma <= gamma_max)
        pnu0 = 1.0
        pnorm = 1.0

        return pgamma*pnu0*pnorm


    ### For now, assume combmod is broadband + QPO
    def combmod_prior(self, t0):
        if len(t0) in [5,6]:
            ### PL + QPO (+ noise)
            pr = self.pl_prior(t0[:3])*self.qpo_prior(t0[3:])
        elif len(t0) in [3,4]:
            pr = self._const_prior(t0[0])*self.qpo_prior(t0[1:])

        else:
            pr = self.bpl_prior(t0[:5])*self.qpo_prior(t0[5:])

        return pr
           

    def const_prior(self, t0):
        noise = t0[0]
        pmean = np.log(np.mean(self.ps.ps[1:]))
        pnoise = scipy.stats.norm.pdf(noise, pmean, pmean/2.0)

        return pnoise




    ### log of the prior
    ### actually, this is -log(prior)
    ### useful so that we can compute -log_posterior
    def logprior(self, t0):

        if self.func == pl:
           mlp = self.pl_prior(t0)
        elif self.func == bpl:
           mlp = self.bpl_prior(t0)
        elif self.func == qpo or self.func.func_name == "lorentz":
           mlp = self.qpo_prior(t0)
        elif self.func.func_name == "combmod":
           mlp = self.combmod_prior(t0)
        elif self.func.func_name == "qpofix":
           mlp = self.qpofixprior(t0)
        elif self.func.func_name == "plqpo":
           mlp = self.plqpo_prior(t0)
        elif self.func.func_name == "bplqpo":
           mlp = self.bplqpo_prior(t0)
        elif self.func == const:
           mlp = self.const_prior(t0)

        else:
           mlp = 1.0

        if mlp > 0:
           return -np.log(mlp)
        else:
           return -logmin

    ### LOG - LIKELIHOOD
    ### actually, this is -log-likelihood (!)
    ### minimizing -logL is the same as maximizing logL
    ### so it all works out okay
    def loglikelihood(self,t0, neg=False):
        funcval = self.func(self.ps.freq, *t0)

        res = np.sum(np.log(funcval))+ np.sum(self.ps.ps/funcval)
        if np.isnan(res):
            #print("res is nan")
            res = -logmin
        elif res == np.inf or np.isfinite(res) == False:
            #print("res is infinite!")
            res = -logmin

        return res



class StackPerPosterior(PerPosterior, object):

    def __init__(self, ps, func, m):
        self.m = m
        PerPosterior.__init__(self, ps, func)


    def loglikelihood(self,t0, neg=False):

        funcval = self.func(self.ps.freq, *t0)

#        res = np.sum(np.log(funcval))+ np.sum(self.ps.ps/funcval)

        res = 2.0*self.m*(np.sum(np.log(funcval))+ np.sum(self.ps.ps/funcval) +
                          np.sum((2.0/float(2*self.m) - 1.0)*np.log(self.ps.ps)))
#        print('res: ' + str(res))
#        print("type res: " + str(type(res)))
        if np.isnan(res):
            #print("res is nan")
            res = -logmin
        elif res == np.inf:
            #print("res is infinite!")
            res = -logmin

        return res

