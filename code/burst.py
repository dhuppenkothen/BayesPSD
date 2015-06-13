import numpy as np
import cPickle as pickle

from . import utils
from . import lightcurve
from . import powerspectrum
from . import mle
from . import bayes
from . import mcmc


#### CLASS BURST ####################
#
# Makes a Burst object.
# Borrows filterenergy function from class Data, hence
# a subclass of utils.Data
#
#
class Burst(utils.Data,object):

    def __init__(self, bstart, blength, 
                 energies=None, 
                 photons=None,
                 events = None, 
                 filename=None, 
                 instrument="gbm",
                 fnyquist = 4096.0,
                 norm='leahy',
                 fluence = None,
                 epeak = None,
                 ttrig = None ):

        ### which instrument was used to record this data
        ### note: this makes a difference in terms of file formats
        ### and general data structure
        self.instrument = instrument

        if ":" in str(bstart):
            self.bst = self.convert_time(bstart) - 0.2*blength
        else:
            self.bst = bstart - 0.2*blength

        ### assume burst length is in seconds
        self.blen = blength + 0.4*blength
        self.bend = self.bst + 1.2*blength
        self.fluence = fluence
        self.epeak = epeak
        self.ttrig = ttrig

 
        ### data is in form of Photon objects such that time/energy filtering
        ### becomes easy
        if photons is None and filename:
            self.read_data(filename)

        elif not photons is None:
            self.photons = photons
 
        else:
            raise Exception("Data missing! You must specify either a photon object or a file name from which to read the data!")


        if not events is None:
            self.energies = events
        startind = photons.searchsorted(self.bst)
        endind = photons.searchsorted(self.bend)
        self.photons = self.photons[startind:endind]
        self.energies = self.energies[startind:endind]


        ### filter for energy selection, if this is specified
        #if energies:
        #    gt.Data.filterenergy(self, energies[0], energies[1])
        if energies:
            self.photons = np.array([s for s,e in zip(self.photons, self.energies) if energies[0] <= e <=energies[1]])
            self.energies = np.array([e for s,e in zip(self.photons, self.energies) if energies[0] <= e <=energies[1]])

        #### filter for burst times
        #gt.Data.filterburst([self.bst-0.1*self.blen, self.bend+0.1*self.blen])

        ### make a light curve
        self.time = self.photons
        print("length time: " + str(len(self.time)))
        print("tseg: " + str(self.time[-1] - self.time[0]))
 
        #self.time = np.array([s.time for s in self.photons])
        self.lc = lightcurve.Lightcurve(self.time, timestep=0.5/fnyquist, tseg=self.blen, tstart=self.bst)

        print("length lc: " + str(len(self.lc.time)))

        ### make a periodogram
        self.ps = powerspectrum.PowerSpectrum(self.lc, norm=norm)

        return

    #### CONVERT TIME ##############
    #
    # convert from HH:MM:SS to seconds
    #
    def convert_time(self, time):
        hours = float(time[:2])*3600.0
        minutes = float(time[3:5])*60.0
        seconds = float(time[6:])
        tnew = hours + minutes + seconds
        return tnew
 

    #### READ DATA FROM FILE
    #
    # read time-tagged event data from file
    # keyword 'type' can be either one of
    #                - "ascii" = read ascii data
    #                - "pickle" = read data from python pickle file
    #                   that contains a list of photon objects
    def read_data(self, filename, type="ascii"):
        if type in ["a", "ascii"]:
            data = utils.conversion(filename)
            time = np.array([float(t) for t in data[0]])
            events = np.array([float(e) for e in data[1]])
            self.photons = [utils.Photon(t,e) for t,e in zip(time, events)]
        elif type in ["p", "pickle"]:
            self.photons = utils.getpickle(filename)
        else:
            raise Exception("File type not recognized! Must be one of 'pickle' or 'ascii'!")
        return 


            
    def bayesian_analysis(self, namestr='test', nchain=500, niter=100, nsim=1000, m=1, fitmethod='bfgs'):

        btest = bayes.Bayes(self.ps, namestr=namestr, m=m)
        psfit, fakeper, self.model_summary = btest.choose_noise_model(mle.pl, [2,3,0.5], mle.bpl, [1,3,2,3,0.5], nchain=nchain, niter=niter, nsim=nsim, fitmethod=fitmethod)

        if not psfit:
            print("Analysis of burst " + str(namestr) + " failed. Returning ...")
            return

        else:
            if self.model_summary["p_lrt"][0] < 0.05:
                print("Model not adequately fit by a power law! Using broken power law instead!")
                self.model = mle.bpl
                self.psfit = getattr(psfit, str(self.model).split()[1]+"fit")
            else:
                self.model = mle.pl
                self.psfit = getattr(psfit, str(self.model).split()[1]+"fit")


            self.per_summary = btest.find_periodicity(self.model, self.psfit["popt"], nchain=nchain, niter=niter, nsim=nsim, fitmethod=fitmethod)

            self.mcmc = self.per_summary["mcobs"]
            return



    #### MAKE AN MCMC SAMPLE #######################
    #
    # Runs the MarkovChainMonteCarlo code to make
    # an MCMC sample independent of the Bayes routines.
    #
    def mcmc_sample(self, func, pars, cov, nchain=500, niter=100, nsim=1000):

        mcobs = mcmc.MarkovChainMonteCarlo(self.ps, func=func, topt=pars, tcov=cov, nchain=nchain, niter=niter, nsim=nsim)

        return mcobs


    def save_burst(self, filename):

        burstfile = open(filename, 'w')
        pickle.dump(self, burstfile)
        burstfile.close()
        return

#####################################################
#####################################################
#####################################################


#### BURST SUBCLASS FOR GBM DATA ###################
#
# Subclass of class Burst to deal with GBM data
# and GBM specific issues
#
#
class GBMBurst(Burst, object):


    def __init__(self, bid, bstart, blength,
                 energies=None,
                 photons=None,
                 events = None,
                 filename=None,
                 instrument="gbm",
                 fnyquist = 4096.0,
                 norm='leahy',
                 fluence = None,
                 epeak = None,
                 ttrig = None):

        ### set burst ID
        self.bid = bid

        ### if photons and filename aren't given, then data comes from procdata file
        if photons ==None and not filename:
           filename = "tte_bn" + str(bid) + "_procdata.dat"
       
        Burst.__init__(self, bstart, blength,
                 energies,
                 photons,
                 events,
                 filename,
                 instrument,
                 fnyquist,
                 norm,
                 fluence = fluence,
                 epeak = epeak,
                 ttrig = ttrig)
        
        return


    def read_data(self, filename, filetype='ascii', det="combined"):

        Burst.read_data(self, filename, type=filetype)

        evt = self.photons[det]
        self.photons = np.array([x.time for x in evt.photons])
        self.energies = np.array([x.energy for x in evt.photons])
        return


 

