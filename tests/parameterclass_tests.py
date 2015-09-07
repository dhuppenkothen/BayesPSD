
#import matplotlib.pyplot as plt
from nose.tools import eq_

import numpy as np
from BayesPSD import parametricmodels

logmin = parametricmodels.logmin





"""



def test_bpl():
    x = np.arange(1000)

    alpha1 = 1.0
    amplitude = 3.0
    alpha2 = 3.0
    x_break = 5.0


    c = parametricmodels.BentPowerLaw()(x, alpha1, amplitude, alpha2, x_break)

    plt.figure()
    plt.loglog(x,c)
    plt.savefig("bpl_model_test.png", format="png")
    plt.close()

    return


def test_qpo():
    x = np.arange(1000)

    gamma = 1.0
    norm = 2.0
    x0 = 200.0

    c = parametricmodels.QPO()(x, gamma, norm, x0)

    plt.figure()
    plt.plot(x,c)
    plt.savefig("qpo_model_test.png", format="png")
    plt.close()

    return


def test_combmodel1():
    models = [parametricmodels.QPO, parametricmodels.Const]

    cmod = parametricmodels.CombinedModel(models, hyperpars=None)
    print("The combined number of parameters is %i (should be 4)"%cmod.npar)
    print("The name of the combined model is %s"%cmod.name)

    x = np.arange(1000)

    gamma = 0.5
    norm = 3.0
    x0 = 200.0
    a = 2.0

    c = cmod(x, gamma, norm, x0, a)
    plt.figure()
    plt.plot(x,c)
    plt.ylim(0.0, 5.0)
    plt.savefig("comb_model_test1.png", format="png")
    plt.close()

    return

def test_combmodel2():
    models = [parametricmodels.PowerLaw, parametricmodels.QPO, parametricmodels.Const]

    cmod = parametricmodels.CombinedModel(models, hyperpars=None)
    print("The combined number of parameters is %i (should be 6)"%cmod.npar)
    print("The name of the combined model is %s"%cmod.name)

    x = np.arange(1000)

    alpha = 1.0
    pl_norm = 3.0
    gamma = 0.5
    qpo_norm = 3.0
    x0 = 200.0
    a = 2.0

    c = cmod(x, alpha, pl_norm, gamma, qpo_norm, x0, a)
    plt.figure()
    plt.loglog(x,c)
    plt.ylim(0.0, 5.0)
    plt.savefig("comb_model_test2.png", format="png")
    plt.close()

    return
    """

class TestConstModel(object):
    def setUp(self):
        self.x = np.arange(1000)
        self.const = parametricmodels.Const()

    def test_length(self):
        a = 2.0
        assert self.const(self.x,a).shape == self.x.shape

    def test_value(self):
        a = 2.0
        all(self.const(self.x, a)) == a


class TestPowerLawModel(object):

    def setUp(self):
        self.x = np.arange(1000)
        self.pl = parametricmodels.PowerLaw()

    def test_shape(self):
        alpha = 2.0
        amplitude = 3.0

        assert self.pl(self.x, alpha, amplitude).shape == self.x.shape


    def test_value(self):
        pl_eqn = lambda x, i, a: np.exp(-i*np.log(x) + a)

        alpha = 2.0
        amplitude = 3.0

        for x in xrange(1,10):
            eq_(pl_eqn(x, alpha, amplitude), self.pl(x, alpha, amplitude))



class TestConstPrior(object):

    def setUp(self):
        self.hyperpars = {"a_mean": 2.0, "a_var": 0.1}
        self.const = parametricmodels.Const(self.hyperpars)

    def test_prior_nonzero(self):
        a = 2.0
        assert self.const.logprior(a) > logmin

    def test_prior_zero(self):
        a = 100.0
        assert self.const.logprior(a) == logmin



class TestPowerlawPrior(object):
    def setUp(self):
        self.hyperpars = {"alpha_min":-8.0, "alpha_max":5.0,
                          "amplitude_min": -10.0, "amplitude_max":10.0}

        alpha_norm = 1.0/(self.hyperpars["alpha_max"]-self.hyperpars["alpha_min"])
        amplitude_norm = 1.0/(self.hyperpars["amplitude_max"]-self.hyperpars["amplitude_min"])
        self.prior_norm = np.log(alpha_norm*amplitude_norm)

        self.pl = parametricmodels.PowerLaw(self.hyperpars)

    def test_prior_nonzero(self):
        alpha = 1.0
        amplitude = 2.0
        print(self.pl)
        assert self.pl.logprior(alpha, amplitude) == self.prior_norm

    def prior_zero(self, alpha, amplitude):
        assert self.pl.logprior(alpha, amplitude) == logmin

    def generate_prior_zero_tests(self):
        alpha_all = [1.0, 10.0]
        amplitude_all = [-20.0, 2.0]
        for alpha, amplitude in zip(alpha_all, amplitude_all):
            yield self.prior_zero, alpha, amplitude



class TestBentPowerLawPrior(object):

    def setUp(self):

        self.hyperpars = {"alpha1_min": -8.0, "alpha1_max":5.0,
                 "amplitude_min": -10., "amplitude_max":10.0,
                 "alpha2_min":-8.0, "alpha2_max":4.0,
                 "x_break_min":np.log(0.1), "x_break_max":np.log(500)}

        alpha1_norm = 1.0/(self.hyperpars["alpha1_max"]-self.hyperpars["alpha1_min"])
        alpha2_norm = 1.0/(self.hyperpars["alpha2_max"]-self.hyperpars["alpha2_min"])
        amplitude_norm = 1.0/(self.hyperpars["amplitude_max"]-self.hyperpars["amplitude_min"])
        x_break_norm = 1.0/(self.hyperpars["x_break_max"]-self.hyperpars["x_break_min"])
        self.prior_norm = np.log(alpha1_norm*alpha2_norm*amplitude_norm*x_break_norm)
        self.bpl = parametricmodels.BentPowerLaw(self.hyperpars)


    def zero_prior(self, alpha1, amplitude, alpha2, x_break):
        assert self.bpl.logprior(alpha1, amplitude, alpha2, x_break) == logmin

    def nonzero_prior(self, alpha1, amplitude, alpha2, x_break):
        assert self.bpl.logprior(alpha1, amplitude, alpha2, x_break) == self.prior_norm


    def test_prior(self):

        alpha1 = [1.0, 10.0]
        alpha2 = [1.0, 10.0]
        amplitude = [2.0, -20.0]
        x_break = [np.log(50.0), np.log(1000.0)]

        for i, a1 in enumerate(alpha1):
            for j, amp in enumerate(amplitude):
                for k, a2 in enumerate(alpha2):
                    for l, br in enumerate(x_break):
                        if i == 1 or j == 1 or k == 1 or l == 1:
                            yield self.zero_prior, a1, amp, a2, br
                        else:
                            yield self.nonzero_prior, a1, amp, a2, br




class TestQPOPrior(object):

    def setUp(self):

        self.hyperpars = {"gamma_min":-1.0, "gamma_max":5.0,
                     "amplitude_min":-10.0, "amplitude_max":10.0,
                     "x0_min":0.0, "x0_max":100.0}

        gamma_norm = 1.0/(self.hyperpars["gamma_max"]-self.hyperpars["gamma_min"])
        amplitude_norm = 1.0/(self.hyperpars["amplitude_max"]-self.hyperpars["amplitude_min"])
        x0_norm = 1.0/(self.hyperpars["x0_max"]-self.hyperpars["x0_min"])
        self.prior_norm = np.log(gamma_norm*amplitude_norm*x0_norm)
        self.qpo = parametricmodels.QPO(self.hyperpars)


    def zero_prior(self, gamma, amplitude, x0):
        assert self.qpo.logprior(gamma, amplitude, x0) == logmin

    def nonzero_prior(self, gamma, amplitude, x0):
        assert self.qpo.logprior(gamma, amplitude, x0) == self.prior_norm


    def test_prior(self):

        gamma = [2.0, -10.0]
        amplitude = [5.0, -20.0]
        x0 = [10.0, -5.0]

        for i,g in enumerate(gamma):
            for j,a in enumerate(amplitude):
                for k, x in enumerate(x0):
                    pars = [g, a, x]
                    if i == 1 or j == 1 or k == 1:
                        yield self.zero_prior, g, a, x
                    else:
                        yield self.nonzero_prior, g, a, x



