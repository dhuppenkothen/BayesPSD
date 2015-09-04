
### Parametric models for use in fitting periodograms

import numpy as np
import math



class ParametricModel(object):

    def __init__(self, npar, name):
        self.npar = npar
        self.name = name

    def __call__(self, freq, *pars):
        return self.func(freq, *pars)




class Const(ParametricModel):

    def __init__(self):
        npar = 1
        name = "const"
        ParametricModel.__init__(self, npar, name)

    def func(self, x, a):
        return np.exp(np.ones(x.shape[0])*a)

    def prior(self):
        return



class PowerLaw(ParametricModel):

    def __init__(self):
        npar = 2 ## number of parameters in the model
        name = "powerlaw" ## model name
        ParametricModel.__init__(self, npar, name)

    def func(self, x, alpha, amplitude):
        """
        Power law model.

        Parameters:
        -----------
        x: numpy.ndarray
            The independent variable
        alpha: float
            The  power law index
        amplitude: float
            The *logarithm* of the normalization or amplitude of the power law

        Returns:
        --------
        model: numpy.ndarray
            The power law model for all values in x.
        """
        res = -alpha*np.log(x) + amplitude
        return np.exp(res)


class BentPowerLaw(ParametricModel):

    def __init__(self):
        npar = 4
        name = "bentpowerlaw"
        ParametricModel.__init__(self, npar, name)

    def func(self, x, alpha1, amplitude, alpha2, x_break):
        """
        A bent power law with a bending factor of 1.

        Parameters:
        -----------
        x: numpy.ndarray
            The independent variable
        alpha1: float
            The  power law index at small x
        amplitude: float
            The normalization or amplitude of the bent power law
        alpha2: float
            The power law index at large x
        x_break: float
            The position in x where the break between alpha1 and alpha2 occurs

        """
        ### compute bending factor
        logz = (alpha2 - alpha1)*(np.log(x) - x_break)

        ### be careful with very large or very small values
        logqsum = sum(np.where(logz<-100, 1.0, 0.0))
        if logqsum > 0.0:
            logq = np.where(logz<-100, 1.0, logz)
        else:
            logq = logz
        logqsum = np.sum(np.where((-100<=logz) & (logz<=100.0), np.log(1.0 + np.exp(logz)), 0.0))
        if logqsum > 0.0:
            logqnew = np.where((-100<=logz) & (logz<=100.0), np.log(1.0 + np.exp(logz)), logq)
        else:
            logqnew = logq

        logy = -alpha1*np.log(x) - logqnew + amplitude
        return np.exp(logy)






class QPO(ParametricModel):

    def __init__(self):
        npar = 3
        name = "qpo"
        ParametricModel.__init__(self, npar, name)

    def func(self, x, gamma, amplitude, x0):
        """
        Lorentzian profile for fitting QPOs.

        Parameters:
        -----------
        x: numpy.ndarray
            The independent variable
        gamma: float
            The width of the Lorentzian profile
        amplitude: float
            The height or amplitude of the Lorentzian profile
        x0: float
            The position of the centroid of the Lorentzian profile
        """
        gamma = np.exp(gamma)
        amplitude = np.exp(amplitude)

        alpha = amplitude*gamma/(math.pi*2.0)
        y = alpha/((x - x0)**2.0 + gamma**2.0)
        return y







"""


### auxiliary function that makes a Lorentzian with a fixed centroid frequency
### needed for QPO search algorithm
def make_lorentzians(x):
   ### loop creates many function definitions lorentz, each differs only by the value
   ### of the centroid frequency f used in computing the spectrum
   for f in x:
       def create_my_func(f):
           def lorentz(x, a, b, e):
               result = qpo(x, a, b, f, e)
               return result
           return lorentz
       yield(create_my_func(f))


def plqpo(freq, plind, beta, noise,a, b, c, d=None):
#def plqpo(freq, plind, beta, noise,a, b, d=None):

    #c = 93.0943061934
    powerlaw = pl(freq, plind, beta, noise)
    quasiper = qpo(freq, a, b, c, d)
    return powerlaw+quasiper

def bplqpo(freq, lplind, beta, hplind, fbreak, noise, a, b, c, d=None):
#def bplqpo(freq, lplind, beta, hplind, fbreak, noise, a, b,d=None):

    #c = 93.0943061934
    powerlaw = bpl(freq, lplind, beta, hplind, fbreak, noise)
    quasiper = qpo(freq, a, b, c, d)
    return powerlaw+quasiper








#### COMBINE FUNCTIONS INTO A NEW FUNCTION
#
# This function will return a newly created function,
# combining all functions giving in *funcs
#
# *funcs should be tuples of (function name, no. of parameters)
# where the number of parameters is that of the function minus the
# x-coordinate ('freq').
#
#
# **kwargs should only really have one keyword:
# mode = 'add'
# which defines whether the model components will be added ('add')
# or multiplied ('multiply').
# By default, if nothing is given, components will be added
#
#
# NOTE: When calling combmod, make sure you put in the RIGHT NUMBER OF
#       PARAMETERS and IN THE RIGHT ORDER!
#
#
# Example:
# - make a combined power law and QPO model, multiplying components together.
#   The power law includes white noise (3 parameters, otherwise two), the
#   QPO model doesn't (3 parameters, otherwise 4):
#   >>> combmod = combine_models((pl, 3), qpo(3), mode='multiply')
#
#
def combine_models(*funcs, **kwargs):

    ### assert that keyword 'mode' is given in function call
    try:
        assert kwargs.has_key('mode')
    ### if that's not true, catch Assertion error and manually set mode = 'add'
    except AssertionError:
        kwargs["mode"] = 'add'

    ### tell the user what mode the code is using, exit if mode not recognized
    if kwargs["mode"]  == 'add':
        print("Model components will be added.")
    elif kwargs['mode'] == 'multiply':
        print('Model components will be multiplied.')
    else:
        raise Exception("Operation on model components not recognized.")

    ### this is the combined function returned by combined_models
    ### 'freq': x-coordinate of the model
    ### '*args': model parameters
    def combmod(freq, *args):
        ### create empty list for result of the model
        res = np.zeros(len(freq))

        ### initialize the parameter count to make sure the right parameters
        ### go into the right function
        parcount = 0

        ### for each function, compute f(x) and add or multiply the result with the previous iteration
        for i,x in enumerate(funcs):
           funcargs = args[parcount:parcount+int(x[1])]
           if kwargs['mode'] == 'add':
               res = res + x[0](freq, *funcargs)
           elif kwargs['mode'] == 'multiply':
               res = res * x[0](freq, *funcargs)
           parcount = parcount + x[1]
        return res
    return combmod



"""
