#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.0/icetray-start
#METAPROJECT /cvmfs/icecube.opensciencegrid.org/users/tyuan/icetray/tarballs/icetray.tabulator.r30f6a2f4.Linux-x86_64.gcc-9.3.0

# -*- coding: utf-8 -*-
################################################################
#
# effective distance tables
#
# 2. step: spline fit for effective distance photo table
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
import json

from optparse import OptionParser

from photospline import glam_fit, ndsparse
from icecube.clsim.tablemaker import splinefitstable, splinetable
from icecube.clsim.tablemaker.photonics import FITSTable, Parity, Efficiency, Geometry


def icecubeify(spline):
    icspl = splinetable.SplineTable()
    icspl.order = spline.order
    icspl.knots = spline.knots
    icspl.coefficients = spline.coefficients
    icspl.extents = spline.extents
    return icspl


def construct_knots(nknots, extents):
    coreknots = [None]*3
    # It's tempting to use some version of the bin centers as knot positions, but this should be avoided. Data points exactly at the
    # knot locations are not fully supported, leading to genuine wierdness in the fit.
    # radius
    coreknots[0] = numpy.append(numpy.linspace(1, 10**0.5, 10)**2, 
                                numpy.linspace(12**0.5, extents[0][1]**0.5, nknots[0])**2)
    # phi
    coreknots[1] = numpy.linspace(extents[1][0], extents[1][1], nknots[1])
    # costheta
    coreknots[2] = numpy.cos(numpy.linspace(numpy.pi-1e-2, 1e-2, nknots[2]))

    # r
    max_r = extents[0][-1]
    rknots = numpy.append(numpy.append([-1, -0.5, -0.1], coreknots[0]), [max_r + 50, max_r + 100, max_r + 150])
    rknots = numpy.append(rknots, [0.74, 1.44, 1.99, 2,30, 2.61, 3.01, 3.78, 24.81, 29.06, 34.24, 44.44, 49.09, 57.69, 62.18])
    rknots = numpy.sort(rknots)
    # phi
    _ = coreknots[1][1]-coreknots[1][0]
    npad = numpy.ceil(17.5/_+3) # pad 3 knots past last reflected bin center
    thetaknots = numpy.append(numpy.append(coreknots[1][0]-numpy.arange(npad,0,-1)*_, coreknots[1]),
                              coreknots[1][-1]+numpy.arange(1, npad+1)*_)
    # costheta
    zknots = numpy.concatenate((-2-coreknots[2][:3][::-1],
                                 coreknots[2],
                                 2+coreknots[2][:3]))
    
    return [rknots, thetaknots, zknots]

def spline_spec(ndim, opts, extents):
    rsmooth = float(opts['rsmooth'])
    fsmooth = float(opts['fsmooth'])
    zsmooth = float(opts['zsmooth'])
    rknots = int(opts['rknots'])
    fknots = int(opts['fknots'])
    zknots = int(opts['zknots'])
    order = [2,2,2] # quadric splines to get smooth derivatives
    penalties = {'smooth':[rsmooth, fsmooth, zsmooth],
                 'order':[2,2,2]} # penalize curvature
    if opts['tablesize'] == 'full':
        fknots *= 2
    knots = construct_knots([rknots, fknots, zknots], extents)
    print(f"Number of Knots: {len(knots)}")
    return order, penalties, knots

# Rescale all axes to have a maximum value of ~ 10
def rescale_axes(knots, bin_centers, bin_widths):
    axis_scale = []
    for i in range(0,len(bin_centers)):
        scale = 2**numpy.floor(numpy.log(numpy.max(bin_centers[i])/10.)/numpy.log(2))
        axis_scale.append(scale)
        bin_centers[i] /= scale
        knots[i] /= scale
        bin_widths[i] /= scale
    return axis_scale

def check_exists(opts, outputfile):
    if os.path.exists(outputfile):
        if opts['force'] is True:
            os.unlink(outputfile)
            print(f"Overwritting {outputfile} - force is True")
            return
        choice = input(f"File {outputfile} exists! Overwrite? (y/n)")
        if choice.lower() in ['y', 'yes']:
            os.unlink(outputfile)
            print(f"Overwritting {outputfile}")
            return
        else:
            sys.exit(1)

def load_dict(f):
    if not os.path.isfile(f):
        print("Could not find dict file")
        return
    if os.path.isfile(f):
        with open(f, 'r') as open_file:
            current_dict = json.load(open_file)
            return current_dict

# parse arguments
usage = "usage: %prog [options] phototable.fits splinetable.eff.fits"
optparser = OptionParser(usage)
optparser.add_option("--rknots", dest="rknots", type="int", help="number of knots in r", default=25)
optparser.add_option("--fknots", dest="fknots", type="int", help="number of knots in phi", default=25)
optparser.add_option("--zknots", dest="zknots", type="int", help="number of knots in cos(theta)", default=25)
optparser.add_option("--rsmooth", dest="rsmooth", type="float", help="smoothness coefficient in radial dimension", default=1e-6)
optparser.add_option("--fsmooth", dest="fsmooth", type="float", help="smoothness coefficient in angular dimension", default=1e-6)
optparser.add_option("--zsmooth", dest="zsmooth", type="float", help="smoothness coefficient in cos(theta)", default=1e-6)
optparser.add_option("--force", dest="force", action="store_true", help="Overwrite existing fits files", default=False)
if __name__ == "__main__":
    (opts, args) = optparser.parse_args()
    try:
        json_file = args[2]
        options = load_dict(json_file)
        use_opt = False
    except:
        print("No valid json dictionary given - using option parser")
        use_opt = True

    # check arguments
    if len(args) < 3:
        optparser.print_usage()
        sys.exit()

    if not os.path.exists(args[0]):
        optparser.error("Input table %s doesn't exist!" % args[0])

    if use_opt == False:
        check_exists(options, args[1])
    
    starttime = datetime.datetime.now()

    # load and normalize clsim photo table
    table = FITSTable.load(args[0])
    # choose extents and only go up to r < 996.5 m to have full support for single tables with r < 1000 m
    extents = [(1.0, 996.0), (0.0, 360.0), (-1.0, 1.0)] # r, phi, cos(polar)
    print(f"Loaded table with dimensions {table.shape}")

    # the data
    z = table.values
    bin_centers = [b.copy() for b in table.bin_centers]
    bin_widths = [b.copy() for b in table.bin_widths]

    # add some numerical stability sauce
    w = 1000*numpy.ones(z.shape)
    w[numpy.logical_not(numpy.isfinite(z))] = 0
    z[numpy.logical_not(numpy.isfinite(z))] = 0

    order, penalties, knots = spline_spec(3, options, extents)
    axis_scale = rescale_axes(knots, bin_centers, bin_widths)

    # penalties are specified, so no need for global smooth parameter
    print("Beginning spline fit...")
    _data, w = ndsparse.from_data(z, w)
    spline = icecubeify(glam_fit(_data,w,bin_centers,knots,order,
                                 penalties['smooth'],penaltyOrder=penalties['order']))
    spline.geometry = table.header['geometry']
    spline.extents = extents
    # set ngroup to 0 if this is for bfr ice. in this case, the effective distance correction is for amplitude only
    spline.ngroup = 0 if options['bfr'] else table.header['n_group']
    spline.parity = table.header['parity']
    spline.knots = [spline.knots[i] * axis_scale[i] for i in range(0, len(spline.knots))]

    print(f"Saving spline table to {args[1]}")
    splinefitstable.write(spline, args[1])

    # clean up
    del(w,z,bin_centers,bin_widths,order,penalties,knots,spline)

    print(f"Run Time: {datetime.datetime.now() - starttime}")

##end
