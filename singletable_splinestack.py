#!/usr/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.0/icetray-start
#METAPROJECT /cvmfs/icecube.opensciencegrid.org/users/tyuan/icetray/tarballs/icetray.tabulator.r30f6a2f4.Linux-x86_64.gcc-9.3.0

# -*- coding: utf-8 -*-
################################################################
#
# single photo tables
#
# 5. step: stack individual spline tables for all single cascade sources
#
# Author: Marcel Usner (adapted from Jakob van Santen)
#
################################################################

# imports
from glob import glob
import re
import os
import sys
import copy
from optparse import OptionParser

import numpy

from icecube.clsim.tablemaker import splinetable, splinefitstable


def stack_tables(tablist, order=2):
    # We expect an array of (splinetable, coordinate) tuples

    bigtab = None

    for table in tablist:
        slice = table[0]
        position = table[1]

        slice.coefficients = slice.coefficients.reshape(
            slice.coefficients.shape + (1,))
        ndim = slice.coefficients.ndim

        if bigtab is None:
            bigtab = slice
            bigtab.knots.append([position])
            bigtab.periods.append(0)
            bigtab.order.append(order)
        else:
            # if position==185 or position==-5:
            #     import pdb
            #     pdb.set_trace()
            bigtab.knots[ndim - 1].append(position)
            bigtab.coefficients = numpy.concatenate(
                (bigtab.coefficients, slice.coefficients),
                ndim - 1)

    # Shift the knots (see bsplineinterp.py)
    baseknots = bigtab.knots[ndim - 1]
    baseknots = numpy.asarray(baseknots)+ \
        (numpy.max(baseknots) - numpy.min(baseknots)) / \
        (2.0 * (len(baseknots)-1)) * (order - 1)
    interpknots = []
    for i in range(order, 0, -1):
        interpknots.append(baseknots[0] - i * (baseknots[1] - baseknots[0]))
        # interpknots.append(baseknots[0]) #TY:clamp knots testing
    interpknots.extend(baseknots)
    interpknots.append(interpknots[len(interpknots) - 1]  + (
        interpknots[len(interpknots) - 1] - interpknots[len(interpknots) - 2]))
    bigtab.knots[ndim - 1] = numpy.asarray(interpknots)

    return bigtab


def unique(seq, idfun=None):
    """Order-preserving uniquification, cribbed from http://www.peterbe.com/plog/uniqifiers-benchmark"""
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


if __name__ == "__main__":
    # Parse command line options
    parser = OptionParser(usage="%prog [options] [source dir] [fits output]")
    parser.add_option("-z", "--zstep",
                      type="int",
                      dest="zstep",
                      metavar="STEP",
                      default=0,
                      help="Increment between source depths (in meters)")
    parser.add_option("-a", "--astep",
                      type="int",
                      dest="astep",
                      metavar="STEP",
                      default=0,
                      help="Increment between source azimuth angles (in degrees)")
    parser.add_option("-f", "--filter",
                      dest="file_type",
                      metavar="EXT",
                      default="fits",
                      help="File extension filter to use (e.g. '.diff.fits')")
    parser.add_option("-o", "--order",
                      dest="order",
                      metavar="ORDER",
                      type="int",
                      default=2,
                      help="BSpline order to use for zenith")

    if len(sys.argv) < 2:
        print(sys.argv)
        sys.argv.append("-h")

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print("Please supply a source directory name")
        sys.exit(0)
    if len(args) < 2:
        print("Please supply an output file name")
        sys.exit(0)

    if os.path.exists(args[1]):
        if input("File %s exists. Overwrite (y/n)? " % args[1]) == 'y':
            os.unlink(args[1])
        else:
            sys.exit()

    sourcedir = args[0] + "/"
    tables = [(i, re.match(".*_z(-?\d+)_zen(\d+).*", i).groups())
              for i in glob("%s*%s" % (sourcedir, options.file_type))]

    # Convert tables to a useful form and sort it
    tables = sorted([(i[0], (int(i[1][0]), int(i[1][1])))
                     for i in tables], key=lambda tab: tab[1])

    # Read in all the actual tables
    print('Table list acquired, reading in tables...',)
    tables = [(splinefitstable.read(i[0]), i[1]) for i in tables]
    print('done')

    depths = unique([tab[1][0] for tab in tables])
    angles = unique([tab[1][1] for tab in tables])

    extents = []
    if len(angles) > 1:
        extents.append((angles[0], angles[-1]))
    if len(depths) > 1:
        extents.append((depths[0], depths[-1]))
    print('extents:',extents)

    # XXX HACK: provide full support above and below by cloning the end tables
    print(f"HACK: cloning tables at {depths[0]} and {depths[-1]}")
    gap = depths[0] - depths[1]
    bottom = [(copy.deepcopy(tab[0]), (tab[1][0] + gap, tab[1][1]))
              for tab in tables if tab[1][0] == tables[0][1][0]]
    gap = depths[-1] - depths[-2]
    top = [(copy.deepcopy(tab[0]), (tab[1][0] + gap, tab[1][1]))
           for tab in tables if tab[1][0] == tables[-1][1][0]]
    tables = bottom + tables + top

    zpos = numpy.unique([i[1][0] for i in tables])
    if options.zstep:
        zpos = [i
                for i in range(zpos[0],
                               zpos[len(zpos) - 1] + options.zstep,
                               options.zstep)
                if i in zpos
                ]

        if len(zpos) < (zpos[len(zpos) - 1] - zpos[0]) / options.zstep + 1:
            print("Error: Some depth steps are missing in table directory.")
            sys.exit(1)

    intermedtables = []

    for z in zpos:
        print(f'Stacking tables at z = {z}')
        # Select all the tables at this z
        sublist = filter(lambda tab: tab[1][0] == z, tables)
        # Reformat to just one coordinate for stacking
        sublist = [(tab[0], tab[1][1]) for tab in sublist]
        if options.astep:
            sublist = [i
                       for i in sublist
                       if i[1] in range(sublist[0][1],
                                        sublist[len(sublist) - 1][1] +
                                        options.astep,
                                        options.astep)
                       ]
            if len(sublist) < (sublist[len(sublist) - 1][1] - sublist[0][1]) / options.astep + 1:
                print("Error: Some azimuth steps are missing in table directory.")
                print(f"Just stopping at z={z}")
                break
                sys.exit(1)

        # extend angular range by mirroring next-to-last
        # angle bins (e.g. 170 and 10 deg) to the outside
        # (e.g. 190 and -10) so that 0 and 180 will have
        # support
        print('\t Extending angular range...')
        lowmirror = [(copy.deepcopy(sublist[1][0]), -sublist[1][1])]
        highmirror = [(copy.deepcopy(sublist[-2][0]),  sublist[-1]
                       [1] + (sublist[-1][1] - sublist[-2][1]))]
        sublist = lowmirror + sublist + highmirror
        print('done')

        print('\t Stacking...')
        print(f'order set to: {options.order}')
        subtbl = stack_tables(sublist, options.order)
        # splinefitstable.write(subtbl, f'z{z}_{args[1]}')
        intermedtables.append((subtbl, z))
        print('done')

    # We no longer need to original tables
    del tables

    if len(zpos) > 1:
        finaltab = stack_tables(intermedtables)
    else:
        finaltab = intermedtables[0][0]

    try:
        targetfile = args[1]
    except IndexError:
        targetfile = os.path.normpath(os.path.join(os.path.abspath(
            sourcedir), '..', os.path.basename(os.path.abspath(sourcedir)) + '.fits'))

    finaltab.extents += extents

    splinefitstable.write(finaltab, targetfile)
    print(f"Output written to {targetfile}")

# end
