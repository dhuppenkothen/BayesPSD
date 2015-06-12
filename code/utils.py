
import numpy as np
import scipy

def choice_hack(data, p=None, size=1):

    """ Hack for Numpy Choice function.

    Because old versions of numpy do not have the
    numpy.random.choice function, I've defined a hack
    below that can do the same thing.

    Will be slow for large arrays!

    Note that unlike numpy.random.choice, this function has no "replace"
    option! This means that elements picked from data will *always* be
    replaced, i.e. can be picked again!

    Parameters
    ----------

    data : {list, array-like}
        List to pick elements from

    p : {list, None}, optional, default None
        the weights for the elements in data.
        Needs to be of the same shape as data.
        If None, the weights will be 1./size(data)

    size : int, optional, default 1
        The number of samples to generate.


    Returns
    -------
    choice_data : {object, list}
        Either a single object of the same type as the elements
        of the input data, drawn from that input data according to
        the weights; or a list of objects with length equal to
        the size parameter set above.

    """

    weights = p

    ### if no weights are given, all choices have equal probability
    if weights == None:
        weights = [1.0/float(len(data)) for x in range(len(data))]

    if not np.sum(weights) == 1.0:
        if np.absolute(weights[0]) > 1.0e7 and sum(weights) == 0:
            weights = [1.0/float(len(data)) for x in range(len(data))]
        else:
            raise Exception("Weights entered do not add up to 1! This must not happen!")


    ### Compute edges of each bin
    edges = []
    etemp = 0.0
    for x,y in zip(data, weights):
       etemp = etemp + y
       edges.append(etemp)

    ### if the np.sum of all weights does not add up to 1, raise an Exception
    if size == 1:
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


################################################################################################

class TwoPrint(object):
    """
    Print both to a screen and to a file.

    Parameters
    ----------
    filename : string
        The name of the file to save to.

    """

    def __init__(self,filename):
        self.file = open(filename, "w")
        self.filename = filename
        self.file.write("##\n")
        self.close()
        return

    def __call__(self, printstr):
        """
        Print to a the screen and a file at the
        same time.

        Parameters
        ----------
        printstr : string
            The string to be printed to file and screen.

        """
        print(printstr)
        self.file = open(self.filename, "a")
        self.file.write(printstr + "\n")
        self.close()
        return

    def close(self):
        self.file.close()
        return

#####################################################################################

def autocorr(x, nlags = 100, fourier=False, norm = True):

    """ Computes the autocorrelation function,
    i.e. the correlation of a data set with itself.
    To do this, shift data set by one bin each time and compute correlation for
    the data set with itself, shifted by i bins

    If the data is _not_ correlated, then the autocorrelation function is the delta
    function at lag = 0

    The autocorrelation function can be computed explicitly, or it can be computed
    via the Fourier transform (via the Wiener-Kinchin theorem, I think)

    Parameters
    ----------
    x : {list, array-like}
        The input data to autocorrelate.

    nlags : int, optional, default 100
        The number of lags to compute,

    fourier: boolean, optional, default False
        If True, use the Fourier transform to compute the ACF (True),
        otherwise don't.

    norm : boolean, optional, default True
        If True, normalize the the ACF to 1


    Returns
    -------
    rnew : array-like
        The autocorrelation function of the data in x
    """

    ### empty list for the ACF
    r = []
    ### length of the data set
    xlen = len(x)

    ### shift data set to a mean=0 (otherwise it comes out wrong)
    x1 = np.copy(x) - np.mean(x)
    x1 = list(x1)

    ### add xlen zeros to the array of the second time series (to be able to shift it)
    x1.extend(np.zeros(xlen))

    ### if not fourier == True, compute explicitly
    if not fourier:
        ### loop over all lags
        for a in range(nlags):
            ### make a np.array of 2*xlen zeros to store the data set in
            x2 = np.zeros(len(x1))
            ### put data set in list, starting at lag a
            x2[a:a+xlen] = x-np.mean(x)
            ### compute autocorrelation function for a, append to list r
            r.append(sum(x1*x2)/((xlen - a)*np.var(x)))

    ### else compute autocorrelation via Fourier transform
    else:
        ### Fourier transform of time series
        fourier = scipy.fft(x-np.mean(x))
        ### take conjugate of Fourier transform
        f2 = fourier.conjugate()
        ### multiply both together to get the power spectral density
        ff = f2*fourier
        ### extract real part
        fr = np.array([b.real for b in ff])
        ps = fr
        ### autocorrelation function is the inverse Fourier transform
        ### of the power spectral density
        r = scipy.ifft(ps)
        r = r[:nlags+1]
    ### if norm == True, normalize everything to 1
    if norm:
        rnew = r/(max(r))
    else:
        rnew = r
    return rnew

