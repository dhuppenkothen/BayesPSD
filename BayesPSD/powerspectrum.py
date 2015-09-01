import numpy as np
import math
import scipy
import scipy.optimize

from . import lightcurve


def add_ps(psall, method='avg'):

    pssum = np.zeros(len(psall[0].ps))
    for x in psall:
        pssum = pssum + x.ps

    if method.lower() in ['average', 'avg', 'mean']:
        pssum = pssum/len(psall)

    psnew = PowerSpectrum()
    psnew.freq = psall[0].freq
    psnew.ps = pssum
    psnew.n = psall[0].n
    psnew.df = psall[0].df
    psnew.norm = psall[0].norm
    return psnew 

### REDOING THIS IN CLASSES ####
class PowerSpectrum(lightcurve.Lightcurve):

    def __init__(self, lc = None, counts = None, nphot=None, norm='leahy', verbose=False):

        self.norm = norm

        if isinstance(lc, lightcurve.Lightcurve) and counts is None:             
            pass

        elif not lc is None and not counts is None:
            if verbose == True:
                print "You put in a standard light curve (I hope). Converting to object of type Lightcurve"
            lc = lightcurve.Lightcurve(lc, counts, verbose=verbose)
        else:
            self.freq = None
            self.ps = None
            self.df = None
            return

        if nphot is None:
            nphots = np.sum(lc.counts)
        else:
            nphots = nphot
        nel = np.round(lc.tseg/lc.res)

        df = 1.0/lc.tseg

        fourier= scipy.fft(lc.counts) ### do Fourier transform
        f2 = fourier.conjugate() ### do conjugate
        ff = f2*fourier   ### multiply both together
        fr = np.array([x.real for x in ff]) ### get out the real part of ff
        ps = 2.0*fr[0: int(nel/2)]/nphots
        freq = np.arange(len(ps))*df

        if norm.lower() in ['leahy']:
            self.ps = ps
            
        elif norm.lower() in ['rms']:
            self.ps = ps/(df*nphots)

        elif norm.lower() in ['variance', 'var']:
            self.ps = ps*nphots/len(lc.counts)**2.0

        self.freq = [f+(freq[1]-freq[0])/2.0 for f in freq]
        self.df = self.freq[1] - self.freq[0]
        self.nphots = nphots
        self.n = len(lc.counts)

    def rebinps(self, res, verbose=False):
        ### frequency range of power spectrum
        flen = (self.freq[-1] - self.freq[0])
        ### calculate number of new bins in rebinned spectrum
        bins = np.floor(flen/res) 
        ### calculate *actual* new resolution
        self.bindf = flen/bins
        ### rebin power spectrum to new resolution
        binfreq, binps, dt = self._rebin_new(self.freq, self.ps, res, method='mean')
        newps = PowerSpectrum()
        newps.freq = binfreq
        newps.ps = binps
        newps.df = dt
        newps.nphots = binps[0]
        newps.n = 2*len(binps)
        return newps


    def rebin_log(self, f=0.01):
        """
        Logarithmic rebin of the periodogram.
        The new frequency depends on the previous frequency
        modified by a factor f:

        dnu_j = dnu_{j-1}*(1+f)

        Parameters:
        -----------
        f: float, optional, default 0.01
            parameter that steers the frequency resolution


        Returns:
            binfreq: numpy.ndarray
                the binned frequencies
            binps: numpy.ndarray
                the binned powers
        """

        minfreq = self.freq[1]*0.5
        maxfreq = self.freq[-1]
        binfreq = [minfreq]
        df = self.freq[1]
        while binfreq[-1] <= maxfreq:
            binfreq.append(binfreq[-1] + df*(1.+f))
            df = binfreq[-1]-binfreq[-2]
        binps, bin_edges, binno = scipy.stats.binned_statistic(self.freq, self.ps, statistic="mean", bins=binfreq)

        nsamples = np.array([len(binno[np.where(binno == i)[0]]) for i in xrange(np.max(binno))])
        df = np.diff(binfreq)
        binfreq = binfreq[:-1]+df/2.
        return binfreq, binps, nsamples



    def findmaxpower(self):
        psfiltered = filter(lambda x: x >= 100.0, self.ps)
        maxpow = max(psfiltered)
        return maxpow

    def checknormal(self, freq, ps):
        ### checks the normalization of a power spectrum above fnyquist/10 Hz
        fmin = max(freq)/10.0
        minind = np.array(freq).searchsorted(fmin)
        psnew = ps[minind:-1]
        normlevel = np.average(psnew)
        normvar = np.var(psnew)

        return normlevel, normvar


