import numpy as np
import scipy.stats

from BayesPSD import posterior, powerspectrum, parametricmodels

np.random.seed(20150907)



class PosteriorClassDummy(posterior.Posterior):
    """
    This is a test class that tests the basic functionality of the
    Posterior superclass.
    """
    def __init__(self, x, y, model):
        posterior.Posterior.__init__(self, x, y, model)


    def  loglikelihood(self, t0, neg=False):
        loglike = 1.0
        return loglike

    def logprior(self, t0):
        lp = 2.0
        return lp


class TestPosterior(object):

    def setUp(self):
        self.x = np.arange(100)
        self.y = np.ones(self.x.shape[0])
        self.model = parametricmodels.Const(hyperpars={"a_mean":2.0, "a_var":1.0})
        self.p = PosteriorClassDummy(self.x,self.y,self.model)


    def test_inputs(self):
        assert np.allclose(self.p.x, self.x)
        assert np.allclose(self.p.y, self.y)
        assert isinstance(self.p.model, parametricmodels.Const)

    def test_call_method_positive(self):
        t0 = [1,2,3]
        post = self.p(t0, neg=False)
        assert post == 3.0

    def test_call_method_negative(self):
        t0 = [1,2,3]
        post = self.p(t0, neg=True)
        assert post == -3.0




class TestPerPosterior(object):

    def setUp(self):
        m = 1
        nfreq = 1000000
        freq = np.arange(nfreq)
        noise = np.random.exponential(size=nfreq)
        power = noise*2.0

        ps = powerspectrum.PowerSpectrum()
        ps.freq = freq
        ps.ps = power
        ps.m = m
        ps.df = freq[1]-freq[0]
        ps.norm = "leahy"

        self.ps = ps
        self.a_mean, self.a_var = 2.0, 1.0

        self.model = parametricmodels.Const(hyperpars={"a_mean":self.a_mean, "a_var":self.a_var})


    def test_making_posterior(self):
        lpost = posterior.PerPosterior(self.ps, self.model)
        print(lpost.x)
        print(self.ps.freq)
        assert lpost.x.all() == self.ps.freq[1:].all()
        assert lpost.y.all() == self.ps.ps[1:].all()

    def test_logprior(self):
        t0 = [2.0]

        lpost = posterior.PerPosterior(self.ps, self.model)
        lp_test = lpost.logprior(t0)
        lp = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        assert lp == lp_test

    def test_loglikelihood(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        loglike = -np.sum(self.ps.ps[1:]/m + np.log(m))

        lpost = posterior.PerPosterior(self.ps, self.model)
        loglike_test = lpost.loglikelihood(t0, neg=False)

        assert np.isclose(loglike, loglike_test)

    def test_negative_loglikelihood(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        loglike = np.sum(self.ps.ps[1:]/m + np.log(m))

        lpost = posterior.PerPosterior(self.ps, self.model)
        loglike_test = lpost.loglikelihood(t0, neg=True)

        assert np.isclose(loglike, loglike_test)


    def test_posterior(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        lpost = posterior.PerPosterior(self.ps, self.model)
        post_test = lpost(t0, neg=False)

        loglike = -np.sum(self.ps.ps[1:]/m + np.log(m))
        logprior = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        post = loglike + logprior

        assert np.isclose(post_test, post, atol=1.e-10)

    def test_negative_posterior(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        lpost = posterior.PerPosterior(self.ps, self.model)
        post_test = lpost(t0, neg=True)

        loglike = -np.sum(self.ps.ps[1:]/m + np.log(m))
        logprior = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        post = -loglike - logprior

        assert np.isclose(post_test, post, atol=1.e-10)

class TestPerPosteriorAveragedPeriodogram(object):

    def setUp(self):
        m = 10
        nfreq = 1000000
        freq = np.arange(nfreq)
        noise = scipy.stats.chi2(2.*m).rvs(size=nfreq)/np.float(m)
        power = noise

        ps = powerspectrum.PowerSpectrum()
        ps.freq = freq
        ps.ps = power
        ps.m = m
        ps.df = freq[1]-freq[0]
        ps.norm = "leahy"

        self.ps = ps
        self.a_mean, self.a_var = 2.0, 1.0

        self.model = parametricmodels.Const(hyperpars={"a_mean":self.a_mean, "a_var":self.a_var})

    def test_likelihood(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)


