
import numpy as np
import argparse
import textwrap

from BayesPSD.code.burst import GBMBurst


def read_burstdata(filename, datadir="./"):
    """
    Run Bayesian QPO search on all bursts in file filename.

    Parameters
    ----------
    filename: string
        Name of a file with minimal burst data. Needs to have columns:
        1. ObsID
        2. MET trigger time
        3. Seconds since trigger
        4. Burst duration in seconds
        Note that this is the way my file is currently set up. You can change
        this by changing the indices of the columns read out below.

    datadir: string
        Directory where the data (including the file in filename) is located.
    """

    ## read in data
    ## type needs to be string, otherwise code fails on ObsID column,
    ## which doesn't purely consist of numbers
    data  = np.loadtxt(datadir+filename, dtype=np.string_)

    ## ObsIDs are in first column, need to remain string
    obsids = data[:,0]

    ## trigger time is in second column, should be float
    trigtime = data[:,1].astype("float64")

    ## start time in seconds since trigger is in third column,
    ## should be float
    bstart = data[:,2].astype("float64")

    ## burst duration in seconds is in fourth column,
    ## should be float
    blength = data[:,3].astype("float64")


    return obsids, trigtime, bstart, blength

def run_bursts(filename, datadir="./", nchain=500, niter=200, nsim=1000):
    """
    Run the Bayesian QPO search on all bursts.

    Parameters
    ----------
    filename: string
        Name of a file with minimal burst data. Needs to have columns:
        1. ObsID
        2. MET trigger time
        3. Seconds since trigger
        4. Burst duration in seconds
        Note that this is the way my file is currently set up. You can change
        this by changing the indices of the columns read out below.

    datadir: string, optional, default "./" (current directory)
        Directory where the data (including the file in filename) is located.

    nchain: int, optional, default 500
        number of chains/walkers for MCMC

    niter: int, optional, default 200
        number of iterations for each MCMC chain

    nsim: int, optional, default 1000
        Number of fake periodograms to simulate to build posterior distributions

    """

    ## first load the ObsIDs, start times etc.
    obsids, trigtime, bstart, blength = read_burstdata(filename, datadir=datadir)

    ## empty list to store all burst objects in
    allbursts = []

    ## which energy range do we want to run over?
    energies = [8.,200.]

    ## what's the Nyquist frequency supposed to be?
    ## This depends on the time resolution of your instrument
    ## and the frequencies where you expect QPOs to appear
    fnyquist = 4096.

    ## get the unique set of ObsIDs
    obsid_set = np.unique(obsids)

    ## loop over all ObsIDs
    for o in obsid_set:
        ## this filename structure should reflect what your data files look like
        ## mine all look like ObsID_tte_combined.dat
        ## and contain TTE data (seconds since trigger) and photon energies
        datafile = datadir+"%s_tte_combined.dat"%o
        data = np.loadtxt(datafile)
        times = data[:,0]
        events = data[:,1]

        ## find all bursts in this observation
        bst = bstart[obsids == o]
        blen = blength[obsids == o]
        ttrig = trigtime[obsids == o]
        print(len(bst))

        fitmethod = "bfgs"

        ## loop over all bursts
        for s,l in zip(bst, blen):
            burst = GBMBurst(bid=o, bstart=s, blength=l,
                            energies=energies, photons=times, events=events,
                            instrument="gbm", fnyquist=fnyquist)

            namestr = "%s_%.3f"%(o,s)

            burst.bayesian_analysis(namestr = namestr,
                       nchain = nchain,
                       niter = niter,
                       nsim = nsim,
                       m = 1, fitmethod = fitmethod)


            allbursts.append(burst)

    return allbursts


def main():
    run_bursts(filename, datadir, nchain, niter, nsim)


    return


if __name__ == "__main__":
    ### DEFINE PARSER FOR COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Bayesian QPO searches for burst light curves.",
                                     epilog=textwrap.dedent("""
    Examples
    --------

    Print this help message:

            $> python run_bursts.py --help

    Run this script from anywhere on your system:

            $> python /absolute/path/to/BayesPSD/code/run_bursts.py --help


    Run on example data in the data directory:

            $> python /absolute/path/to/BayesPSD/code/run_bursts.py -f "sgr1550_burstdata.dat"
                    -d "absolute/path/to/BayesPSD/data/"

    Run on example data (from example data directory) with more walkers, more iterations, more simulations, just MORE!

            $> python ../code/run_bursts.py -f "sgr1550_burstdata.dat" -d "./" -c 750 -i 600 -s 10000


    """))

    ### other arguments
    parser.add_argument('-f', '--filename', action="store", dest="filename",
                        required=True, help="Data file with ObsIDs, trigger times, start times and burst durations.")
    parser.add_argument('-d', '--datadir', action="store", dest="datadir", required=False, default="./",
                        help="Directory with the data (default: current directory).")
    parser.add_argument('-c', '--nchain', action="store", dest="nchain", required=False, type=int, default=500,
                        help="The number of walkers/chains for the MCMC run (default: 500).")
    parser.add_argument('-i', '--niter', action="store", dest="niter", required=False, type=int, default=200,
                        help="The number of iterations per chain/walker in the MCC run (default: 200).")
    parser.add_argument('-s', '--nsim', action="store", dest="nsim", required=False, type=int, default=1000,
                        help="The number of fake periodograms to simulate (default: 1000).")

    clargs = parser.parse_args()

    filename = clargs.filename
    datadir = clargs.datadir
    nchain = clargs.nchain
    niter = clargs.niter
    nsim = clargs.nsim

    main()