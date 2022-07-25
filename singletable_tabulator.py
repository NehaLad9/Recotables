#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.0/icetray-start
#METAPROJECT /cvmfs/icecube.opensciencegrid.org/users/tyuan/icetray/tarballs/icetray.tabulator.r30f6a2f4.Linux-x86_64.gcc-9.3.0

##METAPROJECT /scratch/tyuan/metaprojects/combo/build/
# -*- coding: utf-8 -*-
################################################################
#
# single photo tables
# (edit: was previously rev146968)
# 1. step: tabulate photon flux from a single cascade
#
# Author: Marcel Usner (adapted from Jakob van Santen)
#
################################################################

print("Script has begun")

import sys
print(sys.argv)
from optparse import OptionParser
from icecube.icetray import I3Units
from os import path, unlink
import datetime
import numpy

from icecube import icetray, clsim
from icecube.clsim.tablemaker.tabulator import TabulatePhotonsFromSource, generate_seed
from icecube.clsim.tablemaker.photonics import FITSTable

import os
import subprocess
import threading
import time

# parse arguments
usage = "usage: %prog [options] outputfile"
parser = OptionParser(usage, description=__doc__)
parser.add_option("--seed", dest="seed", type="int", default=None, help="Seed for random number generators; harvested from /dev/random if unspecified.")
parser.add_option("--nevents", dest="nevents", type="int", default=100, help="Number of light sources to inject [%default]")
parser.add_option("--z", dest="z", type="float", default=0., help="IceCube z-coordinate of light source, in meters [%default]")
parser.add_option("--zenith", dest="zenith", type="float", default=0., help="Zenith angle of source, in IceCube convention and degrees [%default]")
parser.add_option("--azimuth", dest="azimuth", type="float", default=0., help="Azimuth angle of source, in IceCube convention and degrees [%default]")
parser.add_option("--energy", dest="energy", type="float", default=1, help="Energy of light source, in GeV [%default]")
parser.add_option("--light-source", choices=('cascade', 'flasher', 'infinite-muon'), default='cascade', help="Type of light source. If 'infinite-muon', Z will be ignored, and tracks sampled over all depths. [%default]")
parser.add_option("--flasherwidth", dest="flasherwidth", type="float", default=127, help="Width of the flasher source [%default]")
parser.add_option("--flasherbrightness", dest="flasherbrightness", type="float", default=127, help="Brightness of the flasher source [%default]")
parser.add_option("--icemodel", choices=('spice_mie', 'spice_lea_full', 'spice_lea_flat', 'spice_lea_tilt', 'spice_lea_anisotropy', 'spice_3.2.1_full', 'spice_3.2.1_flat', 'spice_3.2.1_exagg', 'spice_3.2.1_noexagg', 'spice_3.2.2_full', 'spice_3.2.2_flat', 'spice_bfr-v2_flat', 'spice_bfr-v2_full', 'spice_bfr-v2_anisotropy'), default='spice_bfr-v2_flat', help="Ice model [%default]")
parser.add_option("--holeice", choices=('as.h2-50cm', 'as.flasher_p1_0.30_p2_0', 'as.flasher_p1_0.35_p2_0', 'as.nominal'), default='as.h2-50cm', help="DOM angular sensitivity model [%default]")
parser.add_option("--tablesize", choices=('half', 'full'), default='half', help="Table size in azimuth extension [%default]")
parser.add_option("--tabulate-impact-angle", default=False, action="store_true", help="Tabulate the impact angle on the DOM instead of weighting by the angular acceptance")
parser.add_option("--prescale", dest="prescale", type="float", default=1, help="Only propagate 1/PRESCALE of photons. This is useful for controlling how many photons are simulated per source, e.g. for infinite muons where multiple trajectories need to be sampled [%default]")
parser.add_option("--step", dest="steplength", type="float", default=1, help="Sampling step length in meters [%default]")
parser.add_option("--overwrite", dest="overwrite", action="store_true", default=False, help="Overwrite output file if it already exists [%default]")
parser.add_option("--errors", dest="errors", action="store_true", default=False, help="Write the errors to the fits table in addition to values [%default]")
parser.add_option("--extend", dest="extend", action="store_true", default=False, help="Include cascade extension [%default]")
parser.add_option("--rrange", dest="rrange", type="float", default=1000., help="maximum radial range for binning of table")
parser.add_option("--rbins", dest="rbins", type="int", default=283, help="number of radial bins")
parser.add_option("--tbins", dest="tbins", type="int", default=105, help="number of time bins")
parser.add_option("--mergeonly", action="store_true", default=False, dest="mergeonly", help="just merge parallel files")
parser.add_option("--verbose", action="store_true", default=False, dest="verbose", help="Enable verbose logging")
opts, args = parser.parse_args()

print("Options Loaded")

print("Arguments: {}".format(args))
print("Options: {}".format(opts))

if opts.mergeonly:
    outfile = args[-1]
    FITSTable.stack(outfile, *args[:-1])
    sys.exit()

###
if opts.verbose:
    icetray.set_log_level(icetray.I3LogLevel.LOG_INFO)
    icetray.logging.set_level_for_unit('I3CLSimStepToTableConverter', 'TRACE')
    icetray.logging.set_level_for_unit('I3CLSimTabulatorModule', 'DEBUG')
    icetray.logging.set_level_for_unit('I3CLSimLightSourceToStepConverterGeant4', 'TRACE')
    icetray.logging.set_level_for_unit('I3CLSimLightSourceToStepConverterFlasher', 'TRACE')
else:
    icetray.set_log_level(icetray.I3LogLevel.LOG_WARN)

###
if len(args) != 1:
	parser.error("You must specify an output file!")
outfile = args[0]
if path.exists(outfile):
	if opts.overwrite:
		unlink(outfile)
	else:
		parser.error("Output file exists! Pass --overwrite to overwrite it.")

###
if opts.seed is None:
	opts.seed = generate_seed()

### Punnet Square of yes/no for anis/tilt
if opts.icemodel.lower() == "spice_lea_full":
    disabletilt = False
    opts.icemodel = "spice_lea"
elif opts.icemodel.lower() == "spice_lea_anisotropy":
    disabletilt = True
    opts.icemodel = "spice_lea"
elif opts.icemodel.lower() == "spice_lea_tilt":
    disabletilt = False
    opts.icemodel = "spice_lea_flat"
elif opts.icemodel.lower() == "spice_lea_flat":
    disabletilt = True
    opts.icemodel = "spice_lea_flat"

#NEED TO COMPLETE PUNNET SQUARE? no anisotropy only or tilt only.                                 
elif opts.icemodel.lower() == "spice_3.2.1_flat":
    disabletilt = True
    opts.icemodel = "spice_3.2.1_flat"
elif opts.icemodel.lower() == "spice_3.2.1_full":
    disabletilt = False
    opts.icemodel = "spice_3.2.1"

#TEST - "EXAGGerated tilt" (+100m to all non-origin tilt values, see tilt.dat)
elif opts.icemodel.lower() == "spice_3.2.1_noexagg":
    disabletilt = True
    opts.icemodel = "spice_3.2.1_exagg"
elif opts.icemodel.lower() == "spice_3.2.1_exagg":
    disabletilt = False
    opts.icemodel = "spice_3.2.1_exagg"

#Spice 3.2.2 (vertically variable anisotropy) may not be supported yet - using Spice 3.2.1 as most recent model!)
elif opts.icemodel.lower() == "spice_3.2.2_flat":
    disabletilt = True
    opts.icemodel = "spice_3.2.2_flat"
elif opts.icemodel.lower() == "spice_3.2.2_full":
    disabletilt = False
    opts.icemodel = "spice_3.2.2"

elif opts.icemodel.lower() == "spice_bfr-v2_flat":
    disabletilt = True
    opts.icemodel = "spice_bfr-v2_flat"
elif opts.icemodel.lower() == "spice_bfr-v2_full":
    disabletilt = False
    opts.icemodel = "spice_bfr-v2"

else:
    disabletilt = False

#changed so print MUST correspond to set options!
print("Selected ice model is {0}, with setting disableTilt = {1}".format(opts.icemodel,disabletilt))

from I3Tray import I3Tray

print("")
print("starting tray...")
print("")
starttime = datetime.datetime.now()

tray = I3Tray()

rrange = opts.rrange #default 1000
print("rrange set to {}".format(rrange))
rbins = opts.rbins #default 283
print("rbins set to {}".format(rbins))
tbins = opts.tbins
print("tbins set to {}".format(tbins))

if opts.tablesize == "half":
    nbins=(rbins, 36, 100, tbins)
    extents = ((0, rrange), (0, 180), (-1, 1), (0, 7e3))
elif opts.tablesize == "full":
    nbins=(rbins, 72, 100, tbins)
    extents = ((0, rrange), (0, 360), (-1, 1), (0, 7e3))
dims = [
    clsim.tabulator.PowerAxis(extents[0][0], extents[0][1], nbins[0], 2),
    clsim.tabulator.LinearAxis(extents[1][0], extents[1][1], nbins[1]),
    clsim.tabulator.LinearAxis(extents[2][0], extents[2][1], nbins[2]),
    clsim.tabulator.PowerAxis(extents[3][0], extents[3][1], nbins[3], 2),
]
geo = clsim.tabulator.SphericalAxes
axes = geo(dims)

table_params = { "PhotonSource": opts.light_source,
                "Zenith": opts.zenith*I3Units.degree,
                "Azimuth": opts.azimuth*I3Units.degree,
                "ZCoordinate": opts.z*I3Units.m,
                "Energy": opts.energy*I3Units.GeV,
                "Seed": opts.seed,
                "NEvents": opts.nevents,
                "IceModel": opts.icemodel,
                "DisableTilt": disabletilt,
                # "AngularAcceptance": hic,
                "Axes": axes}
                #"RecordErrors": opts.errors
                #"CascadeExtension": opts.extend}

if opts.light_source == "flasher":
    table_params["FlasherWidth"] = opts.flasherwidth
    table_params["FlasherBrightness"] = opts.flasherbrightness

tray.AddSegment(TabulatePhotonsFromSource, Filename=outfile, TabulateImpactAngle=opts.tabulate_impact_angle, PhotonPrescale=opts.prescale, **table_params)
    
tray.AddModule('TrashCan')
tray.Execute()
tray.Finish()

print("done.")
print("this took", datetime.datetime.now() - starttime)
