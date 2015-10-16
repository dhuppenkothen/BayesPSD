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

    def __init__(self, lc = None, counts = None, nphot=None, norm='leahy', m=1, verbose=False):

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
        
        #nel = np.round(lc.tseg/lc.res)
        nel = len(lc.counts)

        df = 1.0/lc.tseg
        fnyquist = 0.5/(lc.time[1]-lc.time[0])

        fourier= scipy.fftpack.fft(lc.counts) ### do Fourier transform
        #f2 = fourier.conjugate() ### do conjugate
        #ff = f2*fourier   ### multiply both together
        #fr = np.array([x.real for x in ff]) ### get out the real part of ff
        fr = np.abs(fourier)**2.#/np.float(len(lc.counts))**2.


        if norm.lower() in ['leahy']:
            #self.ps = 2.0*fr[0: int(nel/2)]/nphots
            p = np.abs(fourier[:nel/2])**2.
            self.ps = 2.*p/np.sum(lc.counts)
        
        elif norm.lower() in ['rms']:
            #self.ps = 2.0*lc.tseg*fr/(np.mean(lc.countrate)**2.0)
            p = fr[:nel/2+1]/np.float(nel**2.)
            self.ps = p*2.*lc.tseg/(np.mean(lc.counts)**2.0)

        elif norm.lower() in ['variance', 'var']:
            self.ps = ps*nphots/len(lc.counts)**2.0

        self.df = df
        self.freq = np.arange(len(ps))*df + df/2.
        self.nphots = nphots
        self.n = len(lc.counts)
        self.m = m


    def compute_fractional_rms(minfreq, maxfreq):
        minind = self.freq.searchsorted(minfreq)
        maxind = self.freq.searchsorted(maxfreq)
        powers = self.ps[minind:maxind]
        if self.norm == "leahy":
            rms = np.sqrt(np.sum(powers)/(self.nphots))       
        elif self.norm == "rms":
            rms = np.sqrt(np.sum(powers*self.df))

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

        df = self.df
        minfreq = self.freq[0] - 0.5*df
        maxfreq = self.freq[-1]
        binfreq = [minfreq, minfreq + df]
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



