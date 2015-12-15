#!/usr/bin/env python
#####################
#
# Class definition for the light curve class. 
# Used to create light curves out of photon counting data
# or to save existing light curves in a class that's easy to use.
#
#

import matplotlib.pyplot as plt

import numpy
import math
import numpy as np

import scipy.optimize


dayseconds = 60.*60.*24.

class Lightcurve(object):
    def __init__(self, time, counts = None, timestep=1.0, tseg=None, verbose = False, tstart = None, format="seconds"):

        self.format = format # time format
        if counts is None:
            if verbose == True:
                print "You put in time of arrivals."
                print "Time resolution of light curve: " + str(timestep)
            ### TOA has a list of photon times of arrival
            self.toa = time
            self.ncounts = len(self.toa)
            self.tstart = tstart
            self.makeLightcurve(timestep, tseg = tseg,verbose=verbose)
            
        else:
            self.time = np.array(time)
            self.counts = np.array(counts)
            self.res = time[1] - time[0]
            if self.format == "seconds":
                self.countrate = self.counts/self.res
            else:
                #print("I am here!")
                self.countrate = self.counts/(self.res*dayseconds)
            self.tseg = self.time[-1] - self.time[0] + self.res

    def makeLightcurve(self, timestep, tseg=None, verbose=False):

        ### if self.counts exists, this is already a light curve, so abort
        try:
            self.counts
            raise Exception("You can't make a light curve out of a light curve! Use rebinLightcurve for rebinning.")
        except AttributeError:

            ## tstart is an optional parameter to set a starting time for the light curve
            ## in case this does not coincide with the first photon
            if self.tstart is None:
                ## if tstart is not set, assume light curve starts with first photon
                tstart = self.toa[0]
            else:
                tstart = self.tstart
            ### number of bins in light curve

            ## compute the number of bins in the light curve
            ## for cases where tseg/timestep are not integer, computer one
            ## last time bin more that we have to subtract in the end
            if tseg:
                timebin = np.ceil(tseg/timestep)
                frac = (tseg/timestep) - int(timebin - 1)
            else:
                timebin = np.ceil((self.toa[-1] - self.toa[0])/timestep)
                frac = (self.toa[-1] - self.toa[0])/timestep - int(timebin - 1)
            #print('tstart: ' + str(tstart))

            tend = tstart + timebin*timestep

            ### make histogram
            ## if there are no counts in the light curve, make empty bins
            if self.ncounts == 0:
                print("No counts in light curve!")
                timebins = np.arange(timebin+1)*timestep + tstart
                counts = np.zeros(len(timebins)-1)
                histbins = timebins
                self.res = timebins[1] - timebins[0]
            else:
                timebins = np.arange(timebin+1)*timestep + tstart
                counts, histbins = np.histogram(self.toa, bins=timebin, range = [tstart, tend])
                self.res = histbins[1] - histbins[0]

            #print("len timebins: " + str(len(timebins)))
            if frac > 0.0:
                self.counts = np.array(counts[:-1])
            else:
                self.counts = np.array(counts) 
            ### time resolution of light curve
            if verbose == True:
                print "Please note: "
                print "You specified the time resolution as: " + str(timestep)+ "."
                print "The actual time resolution of the light curve is: " + str(self.res) +"."

            if self.format == "seconds":
                self.countrate = self.counts/self.res
            else:
                self.countrate = self.counts/(self.res*dayseconds)
            self.time = np.array([histbins[0] + 0.5*self.res + n*self.res for n in range(int(timebin))])
            if frac > 0.0:
                self.time = np.array(self.time[:-1])
            else:
                self.time = self.time
            self.tseg = self.time[-1] - self.time[0] + self.res

    def saveLightcurve(self, filename):
        """ This method saves a light curve to file. """
        lfile = open(filename, 'w')
        lfile.write("# time \t counts \t countrate \n")
        for t,c,cr in zip(self.time, self.counts, self.countrate):
            lfile.write(str(t) + "\t" + str(c) + "\t" + str(cr) + "\n")
        lfile.close()

    def plot(self, filename, plottype='counts'):
        if plottype in ['counts']:
            plt.plot(self.time, self.counts, lw=3, color='navy', linestyle='steps-mid')
            plt.ylabel('counts', fontsize=18)
        elif plottype in ['countrate']:
            plt.plot(self.time, self.countrate)
            plt.ylabel('countrate', fontsize=18)
        plt.xlabel('time [s]', fontsize=18)
        plt.title('Light curve for observation ' + filename)
        plt.savefig(str(filename) + '.ps')
        plt.close()

    def rebinLightcurve(self, newres, method='sum', verbose = False, implementation="new"):
        ### calculate number of bins in new light curve
        nbins = math.floor(self.tseg/newres)+1
        self.binres = self.tseg/nbins
        print "New time resolution is: " + str(self.binres)

        if implementation in ["o", "old"]:
            self.bintime, self.bincounts, self.binres = self._rebin(self.time, self.counts, nbins, method, verbose=verbose)
        else:
            #print("I am here")
            self.bintime, self.bincounts, self.binres = self._rebin_new(self.time, self.counts, newres, method)

    def bkgestimate(self, tseg, loc='both'):
       
        tmin = np.array(self.time).searchsorted(self.time[0]+tseg)
        tmax = np.array(self.time).searchsorted(self.time[-1]-tseg)
        cmin = np.mean(self.counts[:tmin])
        cmax = np.mean(self.counts[tmax:])
        if loc == 'both':
            print("The mean counts/bin before the burst is: " + str(cmin))
            print("The mean counts/bin after the burst is: " + str(cmax))
            print("The combined mean counts/bin is : " + str(np.mean([cmin, cmax])))
            self.meanbkg = np.mean([cmin, cmax])
        elif loc == 'before':
            print("The mean counts/bin before the burst is: " + str(cmin))
            self.meanbkg = cmin
        elif loc == 'after':
            print("The mean counts/bin after the burst is: " + str(cmax))
            self.meanbkg = cmax
        return


    def removebkg(self, tseg, loc='both'):
        self.bkgestimate(tseg, loc=loc)
        counts = self.counts - self.meanbkg
        zeroinds = np.where(counts <= 0.0)[0]
        time = np.array([t for i,t in enumerate(self.time) if not i in zeroinds ])
        counts = np.array([c for i,c in enumerate(counts) if not i in zeroinds ])

        self.ctime = time
        self.ccounts = counts
        return

    ### add Poisson noise to a light curve
    ### this is of some use for artificially created light curves
    def addpoisson(self):
        pcounts = np.array([np.random.poisson for x in self.ctype])
        pcountrate = pcounts/self.res
        self.counts = pcounts
        self.countrate = pcountrate


    ### chop up light curve in pieces and save each piece in
    ### a separate light curve object
    ## len [float]: length of segment (in seconds)
    ## overlap [float, < 1.0]: overlap between segments, in seconds
    def moving_bins(self, timestep=1.0, length=1.0, overlap=0.1):

        #print('self.toa' + str(len(self.toa)))
        ### number of light curves
        nbins = int(math.floor((self.tseg-2.0*overlap)/length))
        print("<<< nbins: " + str(nbins)) 
        try: 
            tstart = self.toa[0]
        except AttributeError:
            raise Exception('No time of arrivals given! Cannot chop up light curve!')

        lcs = []
        tend = 0.0

        while tend <= self.toa[-1] :
            tend = tstart + length 
            stind = self.toa.searchsorted(tstart)
            #print("<<<--- start index : " + str(stind))
            eind = self.toa.searchsorted(tend)
            #print("<<<--- end index: " + str(eind))
            tnew = self.toa[stind:eind]
            #print("<<<--- self.toa: " + str(self.toa[-1]))
            #print("<<<--- tend: " + str(tend))
            if len(tnew) == 0:
                if self.toa[-1] - tend > 0.0:
                    print("tend smaller than end of light curve. Continuing ...")
                    tstart = tend - overlap
                    continue
                else:
                    break
            lcs.append(Lightcurve(tnew, timestep=timestep, tseg=length)) 
            tstart = tend - overlap

        return lcs

    def fitprofile(self, func, p0=None):
        if not p0:
            
            p0 = [10.0, 0.01, 0.01]


        popt, pcov = scipy.optimize.curve_fit(func, self.time, self.counts, p0=p0, maxfev = 50000)
        stderr = np.sqrt(np.diag(pcov))

        print("The best-fit parameters for the FRED model are: \n")

        bestfit = func(self.time, *popt)
        newfit = np.where(np.log(bestfit) > 100.0, 1.0e-100, bestfit)
        fitparams = {"popt":popt, "cov":pcov, "err":stderr, "mfit":newfit}

        return fitparams

    def _rebin_new(self, time, counts, dtnew, method='sum'):


        try:
            step_size = float(dtnew)/float(self.res)
        except AttributeError:
            step_size = float(dtnew)/float(self.df)

        output = []
        for i in numpy.arange(0, len(counts), step_size):
            total = 0
            #print "Bin is " + str(i)

            prev_frac = int(i+1) - i
            prev_bin = int(i)
            #print "Fractional part of bin %d is %f"  %(prev_bin, prev_frac)
            total += prev_frac * counts[prev_bin]

            if i + step_size < len(time):
                # Fractional part of next bin:
                next_frac = i+step_size - int(i+step_size)
                next_bin = int(i+step_size)
                #print "Fractional part of bin %d is %f"  %(next_bin, next_frac)
                total += next_frac * counts[next_bin]

            #print "Fully included bins: %d to %d" % (int(i+1), int(i+step_size)-1)
            total += sum(counts[int(i+1):int(i+step_size)])
            output.append(total)

        tnew = np.arange(len(output))*dtnew + time[0]
        if method in ['mean', 'avg', 'average', 'arithmetic mean']:
            cbinnew = output
            cbin = np.array(cbinnew)/float(step_size)
        elif method not in ['sum']:
            raise Exception("Method for summing or averaging not recognized. Please enter either 'sum' or 'mean'.")
        else:
            cbin = output

        return tnew, cbin, dtnew

    def convert_seconds_to_mjd(self, mjdobs):
        """
        Convert time format of the light curve from seconds (MET or otherwise)
        to MJD. 

        Parameter:
        ----------
            mjdobs: the start time of the light curve in MJD format.

        """
        assert self.format == "seconds", "Time format of the light curve must be seconds to"\
                                         "be able to convert to MJD"

        time_mjd = self.time/dayseconds + mjdobs
        tseg_mjd = self.tseg/dayseconds
        res_mjd = self.res/dayseconds

        lc_mjd = Lightcurve(time=time_mjd, counts=self.counts, format="mjd")
        return lc_mjd


    def convert_mjd_to_seconds(self):
        """
        Convert time format of the light curve from MJD to (MET) seconds.
        """

        time_sec = self.time*dayseconds
        tseg_sec = self.time*dayseconds
        res_sec = self.time*dayseconds

        lc_sec = Lightcurve(time=time_sec, counts=self.counts, format="seconds")
        return lc_sec
