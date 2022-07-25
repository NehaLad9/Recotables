#!/usr/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.0/icetray-start
#METAPROJECT /cvmfs/icecube.opensciencegrid.org/users/tyuan/icetray/tarballs/icetray.tabulator.r30f6a2f4.Linux-x86_64.gcc-9.3.0

# -*- coding: utf-8 -*-
################################################################
#
# effective distance tables
#
# 4. step: stack individual spline tables for all effective distance tables
#
# Author: Marcel Usner
#
# Modified: Colton Hill
################################################################

# imports
from glob import glob
import re, os, sys
import copy
from optparse import OptionParser

import numpy

from icecube.clsim.tablemaker import splinetable, splinefitstable

# Parse command line options
parser = OptionParser(usage="%prog [options] [source dir] [fits output]")
parser.add_option("--zstep", type="int", dest="zstep", metavar="STEP", default=0, help="Increment between source depths (in meters)")
parser.add_option("--force", dest="force", action="store_true", help="Overwrite existing fits files", default=False)
parser.add_option("--prob", action="store_const", const="prob", default="abs", help="Stack shape effective distances")
options, args = parser.parse_args()

if len(args) < 1:
    print("Please supply a source directory name")
    sys.exit(0)
if len(args) < 2:
    print("Please supply an output file name")
    sys.exit(0)

def check_exists(file):
    if os.path.exists(file):
        if options.force:
            os.unlink(file)
        else:
            sys.exit()

def stack_tables(tablist, order = 2):
    # We expect an array of (splinetable, coordinate) tuples

    bigtab = None

    for table in tablist:
        slice = table[0]
        position = table[1]

        slice.coefficients = slice.coefficients.reshape( \
            slice.coefficients.shape + (1,))
        ndim = slice.coefficients.ndim

        if bigtab is None:
            bigtab = slice
            bigtab.knots.append([position])
            bigtab.periods.append(0)
            bigtab.order.append(order)
        else:
            bigtab.knots[ndim - 1].append(position)
            bigtab.coefficients = numpy.concatenate(
                (bigtab.coefficients, slice.coefficients),
                ndim - 1)

    # Shift the knots (see bsplineinterp.py)
    baseknots = bigtab.knots[ndim - 1]
    baseknots = baseknots + (numpy.max(baseknots)-numpy.min(baseknots))/(2.0*len(baseknots))*(order-1)
    interpknots = []
    for i in range (order,0,-1):
        interpknots.append(baseknots[0] - i*(baseknots[1] - baseknots[0]))
    interpknots.extend(baseknots)
    interpknots.append(interpknots[len(interpknots)-1] + (interpknots[len(interpknots)-1] - interpknots[len(interpknots)-2]))
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
        if marker in seen: continue 
        seen[marker] = 1 
        result.append(item)
    return result

# Parse tables
sourcedir = args[0] + "/"
outfile = args[1]
tables_list = [(i, re.match(".*_z(-?\d+).*", i).groups()) for i in glob(f"{sourcedir}*{options.prob}.fits")]

# Convert tables to a useful form and sort it
tables_list = sorted([(i[0], int(i[1][0])) for i in tables_list], key=lambda tab: tab[1])

# Read in all the actual tables
print('Table list acquired, reading in tables...')
tables = [(splinefitstable.read(i[0]), i[1]) for i in tables_list]
print('done')

depths = unique([tab[1] for tab in tables])

extents = []
if len(depths) > 1:
    extents.append((depths[0], depths[-1]))

# XXX HACK: provide full support above and below by cloning the end tables
print("HACK: cloning tables at %.2f and %.2f" % (depths[0], depths[-1]))
gap = depths[0] - depths[1]
bottom = [(copy.deepcopy(tab[0]), (tab[1] + gap)) for tab in tables if tab[1] == tables[0][1]]
gap = depths[-1] - depths[-2]
top = [(copy.deepcopy(tab[0]), (tab[1] + gap)) for tab in tables if tab[1] == tables[-1][1]]
tables = bottom + tables + top

zpos = numpy.unique([i[1] for i in tables])
if options.zstep:
    zpos = [i for i in range(zpos[0], zpos[len(zpos)-1] + options.zstep, options.zstep) if i in zpos]

if len(zpos) < (zpos[len(zpos)-1] - zpos[0]) / options.zstep + 1:
    print("Error: Some depth steps are missing in table directory.")
    sys.exit(1)

if len(zpos) > 1:
    finaltab = stack_tables(tables)
else:
    finaltab = tables[0][0]

try:
    targetfile = outfile
except IndexError:
    targetfile = os.path.normpath(os.path.join(os.path.abspath(sourcedir), '..', os.path.basename(os.path.abspath(sourcedir)) + '.fits'))

finaltab.extents += extents

check_exists(targetfile)
splinefitstable.write(finaltab, targetfile)
print(f"Output written to {targetfile}")

##end
