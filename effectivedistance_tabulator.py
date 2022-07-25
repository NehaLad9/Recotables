#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.0/icetray-start
#METAPROJECT /cvmfs/icecube.opensciencegrid.org/users/tyuan/icetray/tarballs/icetray.tabulator.r30f6a2f4.Linux-x86_64.gcc-9.3.0
# -*- coding: utf-8 -*-
################################################################
#
# effective distance tables
#
# 1. step: create effective distance photo table
#
# Author: Marcel Usner
#
# Modified: Colton Hill
################################################################

# imports
import sys
import os
import datetime
import numpy
from scipy import optimize
import tqdm

from optparse import OptionParser

from photospline import SplineTable

from icecube.dataclasses import I3Constants, I3Particle
from icecube.clsim.tablemaker.photonics import FITSTable, Parity, Efficiency, Geometry

# parse arguments
usage = "usage: %prog [options] flatinputfile.abs.fits fullinputfile.abs.fits phototableoutputfile.fits"
optparser = OptionParser(usage)
(opts, args) = optparser.parse_args()

# check arguments
if len(args) < 3:
    optparser.print_usage()
    sys.exit()

flatinfile = args[0]
fullinfile = args[1]
phototableoutfile = args[2]

if (not os.path.exists(flatinfile)):
    optparser.error("Input flat spline table %s doesn't exist!" % flatinfile)
if (not os.path.exists(fullinfile)):
    optparser.error("Input full spline table %s doesn't exist!" % fullinfile)


def check_exists(file):
    if os.path.exists(file):
        os.unlink(file)

# real code
print("\n Starting ... \n")
starttime = datetime.datetime.now()

# scan properties for efficient calculatiion of effective distance
r_range = 0.5 # # relative scale to radius bin

def r_stepsize(r):
    if r < 10 : return 0.01
    elif r < 50 : return 0.1
    elif r < 100 : return 0.25
    elif r < 200 : return 0.5
    elif r < 300 : return 0.75
    elif r < 400 : return 1.


def centers(x):
    return (x[:-1]+x[1:])*0.5


def edges(x):
    """returns bin edges with centers that approximately match x. approx
    "inverse" of center(x). Note that it is impossible to ensure that
    centers(edges(x)) == x for all x using the functions defined in
    this module as the returned values of centers(e) are subject to
    constraints that do not necessarily exist for arbitrary center
    points.
    """
    c = centers(x)
    return numpy.concatenate(([2*x[0]-c[0]], c, [2*x[-1]-c[-1]]))

 # photonics light yield for a 1 GeV cascade
lightfactor = 32582*5.21

# extents
nbins=(283, 72, 100, 105)
extents = ((0., 1000.), (0., 360.), (-1., 1.), (0., 7000.))
binedges = [
    numpy.linspace(numpy.sqrt(extents[0][0]), numpy.sqrt(extents[0][1]), nbins[0]+1)**2,
    numpy.linspace(extents[1][0], extents[1][1], nbins[1]+1),
    numpy.linspace(extents[2][0], extents[2][1], nbins[2]+1),
    numpy.linspace(numpy.sqrt(extents[3][0]), numpy.sqrt(extents[3][1]), nbins[3]+1)**2
]
bincenters = [centers(_) for _ in binedges]

tcoord=bincenters[3]
twidth=numpy.ediff1d(binedges[3])
print(twidth)
def get_yield(spt, r, phi, costheta):
    if spt.ndim==3:
        return lightfactor*numpy.exp(spt.evaluate_simple([r, phi, costheta]))
    elif spt.ndim==4:
        # return spt.evaluate_simple([r, phi, costheta, tcoord])*twidth # cdf bin area
        return numpy.argmax(spt.evaluate_simple([r, phi, costheta, tcoord], 1<<3)) # max t of pdf
    else:
        raise RuntimeError

# use those bin centers that are also used for the cascade spline fit
bincenters[0] = bincenters[0][2:-1] # r_firstbin and r_lastbin from singletable_splinefit
bincenters[1] = numpy.concatenate((bincenters[1][-4:]-360,bincenters[1],360+bincenters[1][:4])) # full phi patch
bincenters[2] = numpy.concatenate(([-1.],bincenters[2],[1.])) # pole patch
print(bincenters)
nbins_new = [len(_) for _ in bincenters[:3]]

# update binedges
binedges=[binedges[0][2:-1], edges(bincenters[1]), numpy.concatenate(([-1.], binedges[2], [1.]))]

# load spline table
tablemap = {'flat' : SplineTable(flatinfile), 'full': SplineTable(fullinfile)}
assert tablemap['flat'].ndim==tablemap['full'].ndim
if tablemap['full'].ndim == 4:
    r_range=0.25

print("Loaded spline tables ...")
print("Calculating effective distances ...")

values = numpy.empty(nbins_new)
for i in tqdm.tqdm(numpy.arange(len(bincenters[0]))):
    r = bincenters[0][i]
    # if r < 60:
    #     continue
    for j in numpy.arange(len(bincenters[1])):
        phi = bincenters[1][j]
        # if phi < 100 or phi > 140:
        #     continue
        for k in numpy.arange(len(bincenters[2])):
            costheta = bincenters[2][k]
            # if abs(costheta) > 0.1:
            #     continue
            if r > 1:
                yield_full = get_yield(tablemap['full'], r, phi, costheta)
                test_environment = []
                for r_t in numpy.arange(r*(1-r_range), r*(1+r_range), r_stepsize(r)): # only scan nearby environment
                    if extents[0][0] < r_t < extents[0][1]:  # bail if distance happens to be outside of table range
                        yield_flat = get_yield(tablemap['flat'], r_t, phi, costheta)
                        diff=(numpy.sum(numpy.abs(yield_full-yield_flat)), r_t)
                        test_environment.append(diff)
                        # print(r,j,k, diff, r_t)
                yield_diff, r_eff = min(test_environment)
                # print(r,phi,costheta, yield_diff, r_eff)
            else:
                yield_diff, r_eff = 0., r
            delta = r_eff*numpy.random.uniform(-0.005, 0.005) # 0.5% jitter
            values[i,j,k] = r_eff+delta
            #if j == 7 and k == 49:
            #    print("\t r = %.1f, r_eff = %.1f (at phi = %i, theta = %i)" % 
            #            (r, r_eff, int(phi), int(numpy.arccos(costheta)*180/numpy.pi)))
weights = numpy.ones(values.shape)

# empty header (not needed)
header = {
    'n_photons':         0,
    'efficiency':        Efficiency.NONE,
    'geometry':          Geometry.SPHERICAL,
    'parity':            Parity.EVEN,
    'zenith':            0.,
    'azimuth':           0.,
    'z':                 0.,
    'energy':            0.,
    'lightscale':        1.,
    'type':              int(I3Particle.ParticleType.unknown),
    'level':             1,
    'n_group':           I3Constants.n_ice_group,
    'n_phase':           I3Constants.n_ice_phase,
}

table = FITSTable(binedges, values, weights, header)

print(f"Generated histogram with dimensions {table.shape}")
print(f"Saving photo table to {phototableoutfile}")

check_exists(phototableoutfile)
table.save(phototableoutfile)

print("\n done.")
print(f"this took {datetime.datetime.now() - starttime}")
