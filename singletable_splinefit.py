#!/usr/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.0/icetray-start
#METAPROJECT /cvmfs/icecube.opensciencegrid.org/users/tyuan/icetray/tarballs/icetray.tabulator.r30f6a2f4.Linux-x86_64.gcc-9.3.0

##!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.0.1/icetray-start
##METAPROJECT /data/user/chill/ehe/I3_BASE/build
# -*- coding: utf-8 -*-
################################################################
#
# single photo tables
#
# 3. step: run spline fit on clsim photo tables for a single cascade
#
# Author: Marcel Usner (adapted from Jakob van Santen)
#
################################################################

# imports
from optparse import OptionParser

from photospline import glam_fit, ndsparse
from icecube.clsim.tablemaker import splinetable, splinefitstable
from icecube.clsim.tablemaker.photonics import FITSTable, Parity, Efficiency, Geometry

import sys
import os
import datetime
import numpy
import json
import glob
import re
import pickle

#for easier reading
# from termcolor import colored

##trying for memory profiling
import tracemalloc
import linecache
from datetime import datetime
from queue import Queue, Empty
from resource import getrusage, RUSAGE_SELF
from threading import Thread
from time import sleep

#try:
from table_patch import get_patched, patch_poles, patch_half_phi, patch_full_phi
#except ImportError:
#    print("Make sure you have the patch!")
#    exit(1)

def icecubeify(spline):
    icspl = splinetable.SplineTable()
    icspl.order = spline.order
    icspl.knots = spline.knots
    icspl.coefficients = spline.coefficients
    icspl.extents = spline.extents
    return icspl

def memory_monitor(command_queue: Queue, poll_interval=0.5):
    tracemalloc.start()
    old_max = 0
    snapshot = None
    while True:
        try:
            command_queue.get(timeout=poll_interval)
            if snapshot is not None:
                # print(colored(datetime.now(), 'green'))
                display_top(snapshot)
            return
        except Empty:
            max_rss = getrusage(RUSAGE_SELF).ru_maxrss
            if max_rss > old_max:
                old_max = max_rss
                snapshot = tracemalloc.take_snapshot()
                # print(colored(f"{datetime.now()} max RSS {max_rss}", 'green'))

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
               tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
               tracemalloc.Filter(False, "<unknown>"),
               ))
    top_stats = snapshot.statistics(key_type)
 
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)
 
    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

##end profiling code

def construct_knots(nknots, extents, verbose):
    if verbose is True:
        print(f"core knots: {nknots}")
    coreknots = [None]*4
    # It's tempting to use some version of the bin centers as knot positions, but this should be avoided. Data points exactly at the
    # knot locations are not fully supported, leading to genuine wierdness in the fit.
    # radius
    coreknots[0] = numpy.append(numpy.linspace(0, 10**0.5, 8)**2, numpy.linspace(12**0.5, extents[0][1]**0.5, nknots[0])**2)
    # phi
    coreknots[1] = numpy.linspace(extents[1][0], extents[1][1], nknots[1])
    # costheta
    coreknots[2] = numpy.cos(numpy.linspace(numpy.pi-1e-2, 1e-2, nknots[2]))
    # We're fitting the CDF in time, so we need tightly-spaced knots at early times to be able to represent the potentially steep slope.
    # time
    coreknots[3] = numpy.logspace(0, numpy.log10(extents[3][1]), nknots[3])
    
    ###### ##### trying to improve knots at small t
    #coreknots[3] = numpy.logspace(0, numpy.log10(extents[3][1])*0.75, nknots[3]-5)
    #extra_knots = numpy.logspace(numpy.log10(extents[3][1])*0.80, numpy.log10(extents[3][1]), 5)
    #for knot in extra_knots:
    #    numpy.append(coreknots[3], knot)
    ##### ######

    # Now append the extra knots off both ends of the axis in order to provide full support at the boundaries
    # radius
    #rknots = numpy.append(numpy.append([-1, -0.5, -0.1], coreknots[0]), [120, 140, 160])
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
    # zknots = numpy.append(numpy.append([-1.04, -1.02, -1.005], numpy.cos(numpy.radians(numpy.arange(5,180,5)))), [1.005, 1.02, 1.04])
    # add more knots around cherenkov peak
    zknots = numpy.concatenate(([0.64941176, 0.70352941, 0.81176471, 0.86588235], zknots))
    zknots.sort()
    # NB: we can get away with partial support in time, since we know that F(0) is identically zero.
    # time
    tknots = numpy.append(numpy.append([0., 0.15, 0.3, 0.6], coreknots[3]), [7100, 7200, 7300])
    
    return [rknots, thetaknots, zknots, tknots]

def spline_spec(ndim, opts, extents, verbose):

    rsmooth = float(opts['rsmooth'])
    fsmooth = float(opts['fsmooth'])
    zsmooth = float(opts['zsmooth'])
    tsmooth = float(opts['tsmooth'])

    rknots = int(opts['rknots'])
    fknots = int(opts['fknots'])
    zknots = int(opts['zknots'])
    tknots = int(opts['tknots'])

    print(f"Running spline setup for {ndim} dimensions")

    if ndim > 3:
        if opts['zenith']:
            order = [2,2,2,2] 
            penalties = {'order':[2]*4,
                         'smooth':[rsmooth, fsmooth, zsmooth, tsmooth]} # penalize curvature in rho,z,phi
            knots = construct_knots([rknots, fknots, zknots, tknots], extents, verbose)
            # knots[3] = numpy.arange(-27., 230, 13)
            # knots[3] = numpy.arange(-25., 206, 10)
            knots[3] = numpy.arange(-12.5,192.6,5)
        else:
            order = [2,2,2,3] # quadric splines for t to get smooth derivatives
            penalties = {'order':[2,2,2,3],  # penalize curvature in rho,z,phi
                         'smooth':[rsmooth, fsmooth, zsmooth, tsmooth]} # order 3 in time CDF => order 2 in time PDF
            knots = construct_knots([rknots, fknots, zknots, tknots], extents, verbose)
    else:
        order = [2,2,2] # quadric splines to get smooth derivatives
        penalties = {'order':[2]*3,
                     'smooth':[rsmooth, fsmooth, zsmooth]} # penalize curvature
        knots = construct_knots([rknots, fknots, zknots, tknots], extents, verbose)[:3]
    if verbose is True:
        print(f"Knot sizes: {len(knots[0])}, {len(knots[1])}, {len(knots[2])}")
        print(f"Radius Knots: {knots[0]}")
        print(f"Phi Knots: {knots[1]}")
        print(f"Zenith Knots: {knots[2]}")
        if ndim > 3:
            print(f"Time Knots: {knots[3]}")
    
    return order, penalties, knots

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

def default_path(input):
    file_path = os.path.abspath(input)
    base = os.path.basename(file_path)
    name, extension = os.path.splitext(base)
    return os.path.dirname(file_path) + '/' + name + '.abs.fits', os.path.dirname(file_path) + '/' + name + '.prob.fits'

# Rescale all axes to have a maximum value of ~ 10
def rescale_axes(knots, bin_centers, bin_widths):
    axis_scale = []
    for i in range(0,len(bin_centers)):
        scale = 2**numpy.floor(numpy.log(numpy.max(bin_centers[i])/10.)/numpy.log(2))
        axis_scale.append(scale)
        bin_centers[i] =bin_centers[i] /scale
        knots[i] = knots[i]/scale
        bin_widths[i] = bin_widths[i]/scale
    return axis_scale

def load_table(opts, table_file, r_firstbin, r_lastbin):

    # load and normalize clsim photo table
    table = FITSTable.load(table_file)
    table.remove_nans_and_infinites() 
    table.normalize()

    print(f"Load & Normalise {table_file}")
    print("table.values shape:", table.values.shape)

    # check for a sane normalization and geometry
    eff = Efficiency.RECEIVER | Efficiency.WAVELENGTH | Efficiency.N_PHOTON
    #int2bin = lambda num, count: "".join([str((num >> y) & 1) for y in range(count-1, -1, -1)])
    header = table.header
    if (header['efficiency'] != eff):
        temp_num = bin(header['efficiency'])
        temp_str = str(temp_num[2:])
        default_num = bin(eff)
        default_str = str(default_num[2:])
        err = f"Unknown normalization {temp_str} (expected {default_str})"
        raise ValueError(err)
    if (header['geometry'] is not Geometry.SPHERICAL):
        raise ValueError("This table does not have spherical geometry")

    #values = table.values[r_firstbin:r_lastbin]

    bin_edges = table.bin_edges
    bin_centers = table.bin_centers
    bin_widths = table.bin_widths
   
    ##then slice in r
    bin_edges[0] = bin_edges[0][r_firstbin:r_lastbin]
    bin_widths[0] = bin_widths[0][r_firstbin:r_lastbin]

    weights = table.weights
    if not table.weights is None: # new photo tables do not contain the weights
        weights = table.weights[r_firstbin:r_lastbin]

    ## apply Samuel's patch to table
    ## mask is used when table value is 0
    mask = (bin_centers[0] > 40.) #m

    if opts['abs']:
        ##get patched values
        values, unpatched = get_patched(table, bin_centers[0], mask)
    if opts['prob']:
        values = table.values
        norm = numpy.sum(values, axis=3)
        values = numpy.nan_to_num(values / norm.reshape(norm.shape + (1,)))
    bin_centers, bin_widths, values, weights = patch_poles(bin_centers, bin_widths, values, weights)
    if opts['prob']:
        # extrapolation of pdf can go below zero so cap it
        values[values<0.] = 0
        values = numpy.cumsum(values, axis=3)
        # extrapolized normalization can become greater than 1 so renorm
        norm = values[:,:,:,-1]
        values = numpy.nan_to_num(values / norm.reshape(norm.shape + (1,)))
    if opts['tablesize'] == 'half':
        bin_centers, bin_widths, values, weights = patch_half_phi(bin_centers, bin_widths, values, weights)
    elif opts['tablesize'] == 'full':
        bin_centers, bin_widths, values, weights = patch_full_phi(bin_centers, bin_widths, values, weights)
    bin_centers[0] = bin_centers[0][r_firstbin:r_lastbin]
    values = values[r_firstbin:r_lastbin]
    print(f"values.shape: {values.shape}")

    return header, bin_edges, bin_centers, bin_widths, values, weights

def calc_params_amp(norm):
    z = norm

    # add some numerical stability sauce
    w = 1000*numpy.ones(norm.shape)
    w[numpy.logical_not(numpy.isfinite(z))] = 0
    z[numpy.logical_not(numpy.isfinite(z))] = 0
    return z, w

def amplitude_spline(bin_centers, bin_widths, extents, opts):
    verbose = opts['verbose']
    if opts['zenith']:
        order, penalties, knots = spline_spec(4, opts, extents, verbose)
    else:
        order, penalties, knots = spline_spec(3, opts, extents, verbose)
    #bin_centers = [b.copy() for b in bin_centers[:3]]
    #bin_widths = [b.copy() for b in bin_widths[:3]]
    new_bin_centers = bin_centers[:3]
    new_bin_widths = bin_widths[:3]
    if opts['zenith']:
        new_bin_centers.append(bin_centers[4])
        new_bin_widths.append(bin_widths[4])

    axis_scale = rescale_axes(knots, new_bin_centers, new_bin_widths)

    if verbose is True:
        print(f"New bin centers size: {len(new_bin_centers)}")
        print(f"New bin centers[0] size: {len(new_bin_centers[0])}")
        print(f"Knots size: {len(knots)}")
        #print(f"Knots[0] size: {len(knots[0])}")
        print(f"Order size: {len(order)}")
        print(f"Data (z) size: {z.shape}")

    return new_bin_centers, knots, order, penalties, axis_scale

def amplitude_fit(z, w, header, new_bin_centers, knots, order, penalties, axis_scale, abs_outputfile):
    print("Beginning spline fit for abs table...")
    print("--- Shapes ---")
    print(f"z shape: {z.shape}")
    print(f"w shape: {w.shape}")
    print(f"bin_centers: {new_bin_centers}")
    print(f"knots: {knots}")
    print(f"order: {order}")
    print(f"penalties shape: {penalties}")
    # penalties are specified, so no need for global smooth parameter
    # import pdb
    # pdb.set_trace()
    # penalties[2] = penalties[2][:-1]
    # spline = icecubeify(glam_fit(z[...,0],w[...,0],new_bin_centers[:-1],knots[:-1],order[:-1],0.,penalties=penalties))
    _data, w = ndsparse.from_data(z, w)
    spline = icecubeify(glam_fit(_data,w,new_bin_centers,knots,order,penalties['smooth'],penaltyOrder=penalties['order']))
   
    #import IPython
    #IPython.embed()

    spline.geometry = Geometry.SPHERICAL
    spline.extents = extents[:3]+[extents[-1]] #hacky way to set extents with zenith (-1 dim)
    spline.ngroup = header['n_group']
    if 'parity' in header:
        spline.parity = header['parity']
    else:
        spline.parity = Parity.EVEN

    print(f"Saving table to {abs_outputfile}")
    spline.knots = [spline.knots[i] * axis_scale[i] for i in range(0, len(spline.knots))]
    splinefitstable.write(spline, abs_outputfile)

    ##function goes out of scope - should handle automatically
    # clean up
    #del(w,z,bin_centers,bin_widths,order,penalties,knots,spline)

def calc_params_prob(values):
    #z = numpy.nan_to_num(values / norm.reshape(norm.shape + (1,)))
    z = values
    # XXX HACK: ignore weights for normalized timing
    w = 1000*numpy.ones(values.shape)

    #del(table, norm, values)
    return z, w

def probability_spline(z, w, header, bin_centers, bin_widths, extents, opts, prob_outputfile):
    verbose = opts['verbose']
    order, penalties, knots = spline_spec(4, opts, extents, verbose)
    new_bin_centers = bin_centers
    new_bin_widths = bin_widths
    axis_scale = rescale_axes(knots, new_bin_centers, new_bin_widths)
    
    if verbose is True:
        print("Beginning spline fit for timing table...")
        print("--- Shapes ---")
        print(f"z shape: {z.shape}")
        print(f"w shape: {w.shape}")
        print(f"bin_centers: {new_bin_centers}")
        print(f"knots shape: {knots}")
        print(f"order shape: {order}")
        print(f"penalties shape: {penalties}")

    _data, w = ndsparse.from_data(z, w)
    spline = icecubeify(glam_fit(_data,w,new_bin_centers,knots,order,penalties['smooth'],penaltyOrder=penalties['order'],monodim=3)) # penalties are specified, so no need for global smooth parameter
    spline.geometry = Geometry.SPHERICAL
    spline.extents = extents
    spline.ngroup = header['n_group']
    spline.parity = header['parity']
    if 'parity' in header:
        parity = header['parity']
    else:
        parity = Parity.EVEN
    
    print(f"Saving timing spline to {prob_outputfile}")
    spline.knots = [spline.knots[i] * axis_scale[i] for i in range(0, len(spline.knots))]
    splinefitstable.write(spline, prob_outputfile)

    # clean up
    #del(w,z,bin_centers,bin_widths,order,penalties,knots,spline)

def get_extents(opts, r_start, r_end):
    extents = [(r_start, r_end), (0.0, 180.0), (-1.0, 1.0), (0.0, 7000.)] # r, phi, cos(polar), t
    if opts['tablesize'] == 'full':
        extents[1] = (0.0, 360.0)
        opts['fknots'] *= 2
        print("WARNING - FULL TABLE CONSUMES TOO MUCH MEMORY!")
        # if opts['abs'] is True:
        #     opts['fknots'] *= 2
        # if opts['prob'] is True:
        #     opts['fknots'] *= 1
            # due to a limited memory, currently it doesn't seem to be possible to fit the full range for prob table
            # it is however not neccessary and only interesting for testing purposes
            # so instead just take half of the table to fit and later only evaluate that half
            # az_firstbin = 0
            # az_lastbin = 36 
            # # 72 phi bins from 0...360deg, aimed table extension here is 0...180deg.
            # values = values[:, az_firstbin:az_lastbin, :, :]
            # if not weights is None: # new photo tables do not contain the weights
            #     weights = weights[:, az_firstbin:az_lastbin, :, :]
    return extents

def spline_main(opts, table_file, r_firstbin, r_lastbin):
    #open file & get vals
    header, bin_edges, bin_centers, bin_widths, values, weights = load_table(opts, table_file, r_firstbin, r_lastbin)
    
    #get params for fitting
    ##extents are table size to use in r, phi, costheta
    ##values are the table values
    extents = get_extents(opts, bin_edges[0][0], bin_edges[0][-1])
    print("Loaded histogram with dimensions ", len(values.shape))
    if weights is not None:
        weights = numpy.cumsum(weights, axis=3)
        weights = weights[:,:,:,-1]
    

    if opts['abs'] is True:
        z, w = calc_params_amp(values)
    if opts['prob'] is True:
        #new_bin_centers[3] += new_bin_widths[3]/2.
        z, w = calc_params_prob(values)
        
    return z, w, header, bin_centers, bin_widths, extents
    
def handle_inputs(args, opts, use_opt, table_file, out_file_prefix):
    # check arguments
    if len(args) < 1:
        optparser.print_usage()
        sys.exit(1)
    
    abs_outputfile, prob_outputfile = default_path(out_file_prefix)

    if opts['abs'] is True and opts['prob'] is True:
        print("WARN - Only run one spline at once")
        sys.exit(1)

    if (not os.path.exists(table_file)):
        optparser.error(f"Input table {table_file} doesn't exist!")

    if os.stat(table_file).st_size == 0:
        optparser.error(f"Input table {table_file} has zero size! Did photomc finish properly?")

    if opts['prob'] is False and opts['abs'] is False:
        print("Neither Amplitude nor Timing fit enabled")
        exit(1)

    #else:
    #    if opts.prob and opts.abs:
    #        abs_outputfile, prob_outputfile = default_path(out_file)
    #    else:
    #        abs_outputfile = prob_outputfile = out_file

    check_exists(opts, abs_outputfile)
    check_exists(opts, prob_outputfile)

    return abs_outputfile, prob_outputfile

##load an individual D-Egg dict
def load_dict(f):
    if not os.path.isfile(f):
        return
    if os.path.isfile(f):
        with open(f, 'r') as open_file:
            current_dict = json.load(open_file)
    return current_dict

# parse arguments
usage = "usage: %prog [options] table.pt [output.fits]"
optparser = OptionParser(usage=usage)
optparser.add_option("--abs", action="store_true", help="fit only the total amplitude in each cell", default=False)
optparser.add_option("--prob", action="store_true", help="fit only the normalized CDFs", default=False)
optparser.add_option("--rknots", type="int", help="number of knots in radial dimension", default=25)
optparser.add_option("--fknots", type="int", help="number of knots in angular dimension", default=25)
optparser.add_option("--zknots", type="int", help="number of knots in cos(zenith)", default=25)
optparser.add_option("--tknots", type="int", help="number of knots in time dimension", default=25)
optparser.add_option("--rsmooth", type="float", help="smoothness coefficient in radial dimension", default=1e-6)
optparser.add_option("--fsmooth", type="float", help="smoothness coefficient in angular dimension", default=1e-6)
optparser.add_option("--zsmooth", type="float", help="smoothness coefficient in cos(zenith)", default=1e-6)
optparser.add_option("--tsmooth", type="float", help="smoothness coefficient in time dimension", default=1e-6)
optparser.add_option("--tablesize", choices=('half', 'full'), help="table size in azimuth (either half or full, i.e. 0...180deg or 0...360deg)", default='half')
optparser.add_option("--force", action="store_true", help="overwrite existing fits files", default=False)
optparser.add_option("--zenith", action="store_true", help="fit over the generated zenith", default=False)
if __name__ == "__main__":
    ##where possible functions are used to remove
    ##unused params from memory

    (opts, args) = optparser.parse_args()
    table_file = args[0]
    out_file = args[1]
    try: 
        json_file = args[2]
        print(f'Trying to load {json_file}')
        options = load_dict(json_file)
        use_opt = False
    except:
        print("No json dictionary given - using option parser")
        use_opt = True

    abs_outputfile, prob_outputfile = handle_inputs(args, options, use_opt, table_file, out_file)
    # exclude the first and last radial bins for numerical stability and to save time
    r_firstbin = 2
    r_lastbin = 282 
    # 283 radial bins from 0m...1000m
    
    print("Starting ...")
    starttime = datetime.now()
    
    #tracemalloc.start()
    queue = Queue()
    poll_interval = 0.1
    monitor_thread = Thread(target=memory_monitor, args=(queue, poll_interval))
    monitor_thread.start()

    try:
        if options['zenith']:
            if options['debug']:
                z,w,header,bin_centers,bin_widths,extents = pickle.load(open('pkl/zenith_fit.pkl', 'rb'))
            else:
                table_file = re.sub(r'zen[0-9]+', 'zen0', table_file)
                z, w, header, bin_centers, bin_widths, extents = spline_main(options, table_file, r_firstbin, r_lastbin)
                zeniths = range(5, 181,5)
                z = z[..., None]
                w = w[..., None]
                bin_centers.append(numpy.asarray([0,]+list(zeniths)))
                bin_widths.append(numpy.asarray([10,]*len(bin_centers[-1])))
                extents.append((0, zeniths[-1]))
                for _ in zeniths:
                    table_file = re.sub(r'zen[0-9]+', f'zen{_}', table_file)
                    _z, _w, _header, _bin_centers, _bin_widths, _extents = spline_main(options, table_file, r_firstbin, r_lastbin)
                    z =numpy.append(z, _z[...,None], axis=-1)
                    w =numpy.append(w, _w[...,None], axis=-1)
                if options['dump']:
                    pickle.dump((z,w,header,bin_centers,bin_widths,extents), open('pkl/zenith_fit.pkl', 'wb'))
            # z = numpy.log(numpy.cumsum(numpy.exp(z), axis=3))
        else:
            if options['debug']:
                z,w,header,bin_centers,bin_widths,extents = pickle.load(open('pkl/debug.pkl', 'rb'))
            else:
                z, w, header, bin_centers, bin_widths, extents = spline_main(options, table_file, r_firstbin, r_lastbin)
                if options['dump']:
                    pickle.dump((z,w,header,bin_centers,bin_widths,extents), open('pkl/debug.pkl', 'wb'))
        if options['abs'] is True:
            new_bin_centers, knots, order, penalties, axis_scale = amplitude_spline(bin_centers, 
                                                                    bin_widths, extents, options)
            amplitude_fit(z, w, header, new_bin_centers, knots, order, penalties, axis_scale, abs_outputfile)
        if options['prob'] is True:
            probability_spline(z, w, header, bin_centers, bin_widths, extents, options, prob_outputfile)
    finally:
        queue.put('stop')
        monitor_thread.join()
    
    #snapshot = tracemalloc.take_snapshot()
    #display_top(snapshot)

    print(f"This took: {datetime.now() - starttime}")
##end
