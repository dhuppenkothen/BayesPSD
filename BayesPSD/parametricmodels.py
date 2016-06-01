
### Parametric models for use in fitting periodograms

import numpy as np
import math

def const(freq, a):
    return np.array([np.exp(a) for x in freq])



#### POWER LAW ########
#
# f(x) = b*x**a + c
# a = power law index
# b = normalization (LOG)
# c = white noise level (LOG), optional
#
def pl(freq, a, b, c=None):
    res = -a*np.log(freq) + b
    if c:
        return (np.exp(res) + np.exp(c))
    else:
        return np.exp(res)



#### Lorentzian Profile for QPOs
#
# f(x) = (a*b/(2*pi))/((x-c)**2 + a**2)
#
# a = full width half maximum
# b = log(normalization)
# c = centroid frequency
# d = log(noise level) (optional)
#
def qpo(freq, a, b, c, d=None):

    gamma = np.exp(a)
    norm = np.exp(b)
    nu0 = c

    alpha = norm*gamma/(math.pi*2.0)
    y = alpha/((freq - nu0)**2.0 + gamma**2.0)

    if d is not None:
        y = y + np.exp(d)

    return y


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


###  BENT POWER LAW
# f(x) = (a[1]*x**a[0])/(1.0 + (x/a[3])**(a[2]-a[0]))+a[4])
# a = low-frequency index, usually between 0 and 1
# b = log(normalization)
# c = high-frequency index, usually between 1 and 4
# d = log(frequency where model bends)
# e = log(white noise level) (optional)
# f = smoothness parameter

#def bpl(freq, a, b, c, d, f=-1.0, e=None):
def bpl(freq, a, b, c, d, e=None):

    ### compute bending factor
    logz = (c - a)*(np.log(freq) - d)

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

    logy = -a*np.log(freq) - logqnew + b

#    logy = -a*np.log(freq) + f*logqnew + b
    if e:
        y = np.exp(logy) + np.exp(e)
    else:
        y = np.exp(logy)
    return y







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




