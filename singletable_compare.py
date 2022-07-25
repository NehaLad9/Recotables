#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
#METAPROJECT /data/user/musner/software/meta-projects/tarballs/py2-v1/simulation.openblas.2015-08-05
# -*- coding: utf-8 -*-
################################################################
#
# single photo tables
#
# 4. step: generate plots to check the quality of the spline fit for a single cascade
#
# Author: Marcel Usner
#
################################################################

# imports
import matplotlib
matplotlib.use('pdf') # use non-interactive backend
TEXMODE = matplotlib.rcParams['text.usetex']
if TEXMODE:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Avenir Next']
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 15
matplotlib.rcParams['axes.titlesize'] = 15
import matplotlib.pyplot as p
import matplotlib.gridspec as gridspec
import numpy as n
#from scipy.misc import factorial

import sys
import os
from optparse import OptionParser

from icecube.photospline.photonics import FITSTable
from icecube.photospline import splinefitstable
from icecube.photonics_service import I3PhotoSplineTable

from scipy.stats import distributions

from tqdm import tqdm
from termcolor import colored

#from singletable_quality import spline_quality

# parse arguments
#usage = "usage: %prog --outdir"
#optparser = OptionParser(usage=usage)
#optparser.add_option("--mode", choices=('exact', 'fine', 'coarse', 'snapshot', 'test'), default='snapshot', help="Scan mode to make plots [%default]")
#optparser.add_option("--outdir", type="string", default="", help="Output directory for plots [%default]")
#optparser.add_option("--cores", type="int", default=1, help="Number of parallel cores")

if __name__ == "__main__":
    #opts, args = optparser.parse_args()
    #args = [None] * 2

    table_path = '/home/colton.hill/tablemaker/table_lists/z-280_zen180_prob.txt'
    spline_path = '/home/colton.hill/tablemaker/spline_lists/z-280_zen180_prob.txt'

    table_file = open(table_path, 'r')
    spline_file = open(spline_path, 'r')

    outdir_1 = 'compare_splines/'

    for f in table_file:
        table = f[:-1]

    for spline in tqdm(spline_file):
        fileid = spline[:-1].rsplit('/')[-1].rsplit('.fits')[0]
        spline = spline[:-1]

        spline_name = spline.split('/')
        spline_name = spline_name[-1]

        spline_name = spline_name.split('.prob.prob.fits')
        spline_name = spline_name[0]

        outdir = os.path.join(outdir_1, spline_name)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)


        #args[0] = table
        #args[1] = spline[:-1]
        #splinetable = I3PhotoSplineTable()
        #splinetable.SetupTable(args[1], -1)
        #spline_quality(opts, args, splinetable)

        os.system(f'python3 singletable_quality.py {table} {spline} --outdir {outdir} --mode exact --cores 30')

##end
