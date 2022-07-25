#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.0/icetray-start
#METAPROJECT /cvmfs/icecube.opensciencegrid.org/users/tyuan/icetray/tarballs/icetray.tabulator.r30f6a2f4.Linux-x86_64.gcc-9.3.0

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
from matplotlib.ticker import FormatStrFormatter
import numpy as n
#from scipy.misc import factorial

import sys
import os
from optparse import OptionParser

from icecube.clsim.tablemaker.photonics import FITSTable
from icecube.clsim.tablemaker import splinefitstable
from photospline import SplineTable

from scipy.stats import distributions

from termcolor import colored
from concurrent.futures import ProcessPoolExecutor, wait
from tqdm import tqdm

# photonics light scale
LIGHTSCALE = 32582*5.21

# amplitude plots
def absplot(bins, val, bin_centers, dim_table_centers, dim_table_values, dim_spline_centers, dim_spline_values,
        r_range, fileid, filetype, base_path, xlabels, ylabels, xlimits):
    
    # the figure
    fig = p.figure()
    fig.set_size_inches(14, 8)
    fig.subplots_adjust(hspace=0.3, left=0.09, right=0.96)
    p.rcParams.update({'font.size': 14})

    # the formatter
    formatter = p.ScalarFormatter(useMathText=False, useOffset=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))

    # plot constants
    r_bin, phi_bin, costheta_bin = bins
    #r, phi, costheta = bin_centers[0][r_bin], bin_centers[1][phi_bin], bin_centers[2][costheta_bin]
    r, phi, costheta = val
    if TEXMODE:
        title = 'Bin: $\\rm r=%.1f m$, $\\upphi=%.1f^\\circ$, $\\cos\\uptheta=%.2f$' % (r, phi, costheta)
    else:
        title = 'Bin: r=%.1f m, phi=%.1f deg, cos(theta)=%.2f' % (r, phi, costheta)
    outfile = '%s_abs_%i_%i_%i.%s' % (fileid, r_bin+1, phi_bin+1, costheta_bin+1, filetype)
    print(f"outfile (amplitude): {outfile}")
    
    # the subplots
    for subplot in range(4):

        if subplot < 2:
            dim = 0
        elif subplot < 3:
            dim = 1
        elif subplot < 4:
            dim = 2

        # the axis
        ax = fig.add_subplot(221+subplot)

        # the data
        if dim==2:
            ax.plot(n.degrees(n.arccos(dim_table_centers[dim])), dim_table_values[dim], marker='x', ms=3, linewidth=0, label='photo table')
            ax.plot(n.degrees(n.arccos(dim_spline_centers[dim])), dim_spline_values[dim], label='spline fit')
            xlimits[dim] = (0,180)
        else:
            ax.plot(dim_table_centers[dim], dim_table_values[dim], marker='x', ms=3, linewidth=0, label='photo table')
            ax.plot(dim_spline_centers[dim], dim_spline_values[dim], label='spline fit')
        
        # the labels
        ax.set_xlabel(xlabels[dim], labelpad=8, fontsize=16)
        ax.set_ylabel(ylabels[dim], labelpad=7, fontsize=16)
        
        # deal with specific subplot properties
        if subplot == 0:
            ax.set_xlim(*xlimits[dim])
            ax.set_yscale('log')
            if n.sum(dim_table_values[dim]) > 0:
                yrange_table = dim_table_values[dim][(dim_table_centers[dim] >= xlimits[dim][0]) & 
                                                     (dim_table_centers[dim] <= xlimits[dim][1])]
                yrange_spline = dim_spline_values[dim][(dim_spline_centers[dim] >= xlimits[dim][0]) & 
                                                       (dim_spline_centers[dim] <= xlimits[dim][1])]
                ymin = min([yrange_table[n.nonzero(yrange_table)].min(), yrange_spline[n.nonzero(yrange_spline)].min()])
                ymax = max([yrange_table.max(), yrange_spline.max()])
                ymin = n.nan_to_num(ymin)
                ymax = n.nan_to_num(ymax)
                if ymin > 10e-18:
                    try:
                        ax.set_ylim(ymin=0.7*ymin, ymax=1.3*ymax)
                    except ValueError:
                        print(colored("Ymax is already inf - cannot multiply by 1.3", 'yellow'))
                        try:
                            ax.set_ylim(ymin=0.7*ymin, ymax=ymax)
                        except:
                            print(colored("Ymin - cannot multiply by 0.7", 'yellow'))
                            ax.set_ylim(ymin=ymin, ymax=ymax)
                else:
                    ax.set_ylim(ymin=10e-18, ymax=1.3*ymax)

            ax.text(1.1, 1.1, title, horizontalalignment='center', fontsize=20, transform=ax.transAxes)
            ax.legend(loc=0, prop={'size': 14})
        elif subplot == 1:
            rangeindex = int(n.argwhere(r_range == r_bin))
            if rangeindex == 0:
                xmin = xlimits[dim][0]
                xmax = n.round(bin_centers[0][r_range[0]]+(bin_centers[0][r_range[1]]-bin_centers[0][r_range[0]])/2., 0)
            elif rangeindex == len(r_range)-1:
                xmin = n.round(bin_centers[0][r_range[-1]]-(bin_centers[0][r_range[-1]]-bin_centers[0][r_range[-2]])/2., 0)
                xmax = xlimits[dim][1]
            else:
                xmin = n.round(bin_centers[0][r_range[rangeindex]]-(bin_centers[0][r_range[rangeindex]]-bin_centers[0][r_range[rangeindex-1]])/2., 0)
                xmax = n.round(bin_centers[0][r_range[rangeindex]]+(bin_centers[0][r_range[rangeindex+1]]-bin_centers[0][r_range[rangeindex]])/2., 0)
            xmin = xmin - xmin % 10
            xmax = xmax + xmax % 10
            ax.set_xlim(xmin, xmax)
            if n.sum(dim_table_values[dim]) > 0:
                yrange_table = dim_table_values[dim][(dim_table_centers[dim] >= xmin) &
                                                     (dim_table_centers[dim] <= xmax)]
                yrange_spline = dim_spline_values[dim][(dim_spline_centers[dim] >= xmin) & 
                                                       (dim_spline_centers[dim] <= xmax)]
                if n.sum(yrange_table) > 0 and n.sum(yrange_spline) > 0:
                    ymin = min([yrange_table[n.nonzero(yrange_table)].min(), 
                                yrange_spline[n.nonzero(yrange_spline)].min()])
                    ymax = max([yrange_table.max(), yrange_spline.max()])
                elif n.sum(yrange_table) == 0 and n.sum(yrange_spline) > 0:
                    ymin = yrange_spline[n.nonzero(yrange_spline)].min()
                    ymax = yrange_spline.max()
                else:
                    if n.sum(yrange_spline) == 0:
                        print(colored("yrange_spline sum is 0!", 'red'))
                    if n.sum(yrange_table) == 0:
                        print(colored("yrange_table sum is 0!", 'red'))
                    if n.sum(yrange_spline) == 0 and n.sum(yrange_table) == 0:
                        continue
                    else:
                        ymin = yrange_table[n.nonzero(yrange_table)].min()
                        ymax = yrange_table.max()

                
                ax.set_yscale('log')
                if ymin > 10e-18:
                    try:
                        ymin = n.nan_to_num(ymin)
                        ymax = n.nan_to_num(ymax)
                        ax.set_ylim(ymin=0.7*ymin, ymax=1.3*ymax)
                    except ValueError:
                        print(colored("Ymax is already inf - cannot multiply by 1.3", 'yellow'))
                        try:
                            ax.set_ylim(ymin=0.7*ymin, ymax=ymax)
                        except:
                            print(colored("Ymin - cannot multiply by 0.7", 'yellow'))
                            ax.set_ylim(ymin=ymin, ymax=ymax)
                else:
                    ax.set_ylim(ymin=10e-18, ymax=1.3*ymax)
            ax.legend(loc=0, prop={'size': 14})
        elif subplot > 1:
            ax.set_yscale('log')
            if n.sum(dim_table_values[dim]) > 0:
                chargemean = dim_table_values[dim].mean()
                xmean = (xlimits[dim][1]+xlimits[dim][0])/2.
                percentage = 10
                ax.axhline(chargemean*(1.-percentage/100.), xmin=0., xmax=0.45, color='blue', ls='--')
                ax.axhline(chargemean*(1.-percentage/100.), xmin=0.545, xmax=1., color='blue', ls='--')
                ax.axhline(chargemean, xmin=0., xmax=1., color='blue', ls='--')
                ax.axhline(chargemean*(1.+percentage/100.), xmin=0., xmax=0.45, color='blue', ls='--')
                ax.axhline(chargemean*(1.+percentage/100.), xmin=0.545, xmax=1., color='blue', ls='--')
                if TEXMODE:
                    ax.annotate('$+%i\%%$' % percentage, xy=(xmean*0.99, chargemean*(1.+percentage/100.)), 
                                  fontsize=10, color='blue', va='center', ha='center')
                    ax.annotate('$-%i\%%$' % percentage, xy=(xmean*0.99, chargemean*(1.-percentage/100.)), 
                                  fontsize=10, color='blue', va='center', ha='center')
                else:
                    ax.annotate('+%i\%%' % percentage, xy=(xmean*0.99, chargemean*(1.+percentage/100.)),
                                  fontsize=10, color='blue', va='center', ha='center')
                    ax.annotate('--%i\%%' % percentage, xy=(xmean*0.99, chargemean*(1.-percentage/100.)),
                                  fontsize=10, color='blue', va='center', ha='center')
                ax.set_xlim(*xlimits[dim])
                #ax.yaxis.set_major_formatter(formatter)
                ymin = min([dim_table_values[dim].min(), dim_spline_values[dim].min(), chargemean*(1.-percentage/100.)])
                ymax = max([dim_table_values[dim].max(), dim_spline_values[dim].max(), chargemean*(1.+percentage/100.)])
                
                ymin_ok = False
                ymax_ok = False
                if not n.isinf(ymin) and not n.isnan(ymin) and ymin > 10e-18:
                    ymin_ok = True
                if not n.isinf(ymax) and not n.isnan(ymax):
                    ymax_ok = True
                if ymin_ok == True and ymax_ok == True:
                    ax.set_ylim(ymin=0.1*ymin, ymax=10*ymax)
                if ymin_ok == False and ymax_ok == True:
                    ax.set_ylim(ymin=10e-18, ymax=10e2)
        else:
            ax.set_xlim(*xlimits[dim])
            ax.yaxis.set_major_formatter(formatter)

    # save the fig
    #base_path
    #p.savefig(os.path.join(opts.outdir,  name, outfile))
    p.savefig(os.path.join(base_path, outfile))
    p.close()

    # be chatty
    #print(f"saved: {os.path.join(opts.outdir, name, outfile)}")
    print(f"saved: {os.path.join(base_path, outfile)}")


# time plots
def probplot(bins, val, dim_table_centers, dim_table_values, 
        dim_spline_centers, dim_spline_values, dim_spline_vcomp, 
        dim_table_widths, dim_spline_edges, fileid, filetype,
        base_path, xlabels, ylabels, xlimits, p_val):
    
    # the figure
    fig = p.figure()
    fig.set_size_inches(12, 6)
    outer=gridspec.GridSpec(1,2,top=0.87, bottom=0.13, left=0.08, right=0.96)
    #fig.subplots_adjust(top=0.87, bottom=0.13, left=0.08, right=0.96)
    p.rcParams.update({'font.size': 14})
    
    # plot constants
    r_bin, phi_bin, costheta_bin = bins
    #r, phi, costheta = bin_centers[0][r_bin], bin_centers[1][phi_bin], bin_centers[2][costheta_bin]
    r, phi, costheta = val
    if TEXMODE:
        title = 'Bin: $\\rm r=%.1f m$, $\\upphi=%.1f^\\circ$, $\\cos\\uptheta=%.2f$' % (r, phi, costheta)
    else:
        title = 'Bin: r=%.1f m, phi=%.1f deg, cos(theta)=%.2f' % (r, phi, costheta)
    outfile = '%s_prob_%i_%i_%i.%s' % (fileid, r_bin+1, phi_bin+1, costheta_bin+1, filetype)
    print(f"outfile (time): {outfile}")

    # the subplots
    for subplot in range(2):
        inner=gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=outer[subplot],wspace=0.1, hspace=0.1,height_ratios=[3,1])
        # the axis
        #ax = fig.add_subplot(121+subplot)
        for ssplot in range(2):
            ax = p.subplot(inner[ssplot])
            if ssplot == 0:
                # the data
                ax.plot(dim_table_centers, dim_table_values, marker='x', ms=3, linewidth=0, label='PhotoTable')
                ax.plot(dim_spline_centers, dim_spline_values, label='Spline')

                # the labels
                #ax.set_xlabel(xlabels[3], labelpad=8, fontsize=16)
                #ax.set_ylabel(ylabels[3], labelpad=7, fontsize=16)
                ax.set_ylabel(ylabels[3], fontsize=16)
        
                # the legend
                ax.legend(loc=4, prop={'size': 14}, title=f'{p_val}')
                ax.set_ylim(ymin=0., ymax=1.1)
                ax.set_xticklabels([])
                
                if subplot == 0:
                    ax.text(1.1, 1.07, title, horizontalalignment='center', fontsize=20, transform=ax.transAxes)
 
                # deal with specific subplot properties
            if ssplot == 1:
                # the ratio data
                diff = dim_spline_vcomp-dim_table_values
                ax.plot(dim_table_centers,diff, marker='x', ms=3, mfc='r', mec='r', linewidth=0, label='Diff.')
                ax.axhline(0,ls='--',color='k')
                ax.set_ylim(ymin=-0.1, ymax=0.1)
                ax.set_xlabel(xlabels[3], fontsize=16)
                ax.set_ylabel("Spl. - Tbl.")
            if subplot == 0:
                ax.set_xlim(*xlimits[3])
            elif subplot == 1:
                if n.sum(dim_table_values) > 0:
                    rangeindex = n.argwhere(dim_table_values > 0.95)[0]
                    if rangeindex < 10 : rangeindex = 10
                    xmax = dim_table_centers[rangeindex]
                    xmax = xmax + xmax % 1000
                else:
                    xmax=xlimits[3][1]
                ax.set_xlim(xmin=0., xmax=xmax)
                #ax.set_ylim(ymin=0., ymax=1.1)

    # save the fig
    #p.savefig(os.path.join(opts.outdir, name, outfile))
    p.tight_layout()
    p.savefig(os.path.join(base_path, outfile))
    p.close()

    ##hist the diffs
    diff = dim_spline_vcomp-dim_table_values
    fig_d, ax_d = p.subplots()
    ax_d.hist(diff, bins=20)
    ax_d.set_xlabel('Spline - Values')
    ax_d.set_ylabel('Entries')
    outfile_diff = '%s_prob_diff_%i_%i_%i.%s' % (fileid, r_bin+1, phi_bin+1, costheta_bin+1, filetype)
    fig_d.savefig(os.path.join(base_path, outfile_diff))
    
    diff2 = (dim_spline_vcomp-dim_table_values)**2
    fig_d2, ax_d2 = p.subplots()
    ax_d2.hist(diff2, bins=20)
    ax_d2.set_xlabel(r'(Spline - Values)$^2$')
    ax_d2.set_ylabel('Entries')
    outfile_diff2 = '%s_prob_diff2_%i_%i_%i.%s' % (fileid, r_bin+1, phi_bin+1, costheta_bin+1, filetype)
    fig_d2.savefig(os.path.join(base_path, outfile_diff2))


    table_cdf = dim_table_values
    test_table_pdf = []
    i = 0
    for i in range(len(table_cdf)):
        if i == 0:
            cdf_diff_i = table_cdf[i] - 0.0
            x_diff_i = abs(dim_table_centers[i] - dim_table_widths[0])
        else:
            cdf_diff_i = table_cdf[i] - table_cdf[i-1]
            cdf_diff_i = dim_table_centers[i] - dim_table_centers[i-1]
        pdf_i = cdf_diff_i / x_diff_i
        test_table_pdf.append(pdf_i)

    table_pdf = n.diff(table_cdf)
    table_pdf = n.append(0, table_pdf)
    table_pdf = table_pdf / dim_table_widths
    
    spline_cdf = dim_spline_values
    spline_pdf = n.diff(spline_cdf)
    spline_pdf = n.append(0, spline_pdf)
    spline_pdf = spline_pdf / n.diff(dim_spline_edges)

    fig_pdf, (ax_pdf, ax_pdf_zoom) = p.subplots(1,2)
    fig_pdf.set_size_inches(12, 4)
    t_cent = dim_table_centers
    #t_cent = t_cent
    s_cent = dim_spline_centers
    #s_cent = s_cent

    #ax_pdf.plot(t_cent, test_table_pdf, label='Other Table PDF', drawstyle='steps')
    ax_pdf.plot(t_cent, table_pdf, label='PhotoTable PDF', drawstyle='steps')
    ax_pdf.plot(s_cent, spline_pdf, label='Spline PDF', drawstyle='steps')
    ax_pdf.set_xlabel(xlabels[3], fontsize=16)
    ax_pdf.set_ylabel('PDF')
    ax_pdf.legend(loc=0)
    ax_pdf.set_xlim(xlimits[3][0] - 10, xlimits[3][-1] + 10)
    
    ax_pdf_zoom.plot(t_cent, table_pdf, label='PhotoTable PDF', drawstyle='steps')
    ax_pdf_zoom.plot(s_cent, spline_pdf, label='Spline PDF', drawstyle='steps')
    ax_pdf_zoom.set_xlabel(xlabels[3], fontsize=16)
    ax_pdf_zoom.set_ylabel('PDF')
    ax_pdf_zoom.legend(loc=0)
    table_pdf_max = n.max(table_pdf)
    ind = 0
    max_ind = 0
    for val in table_pdf:
        if val == table_pdf_max:
            max_ind = ind
            break
        ind += 1
    ax_pdf_zoom.set_xlim(t_cent[max_ind] - 150, t_cent[max_ind] + 150)
   
    if n.min(table_pdf) < 0 or n.min(spline_pdf) < 0:
        ax_pdf.set_ylim(-0.02, n.max(spline_pdf)*1.05)
        ax_pdf_zoom.set_ylim(-0.02, n.max(spline_pdf)*1.05)

    fig_pdf.suptitle(title)
    outfile_pdf = '%s_prob_pdf%i_%i_%i.%s' % (fileid, r_bin+1, phi_bin+1, costheta_bin+1, filetype)
    p.tight_layout()
    fig_pdf.savefig(os.path.join(base_path, outfile_pdf))

    # be chatty
    #print(f"saved {os.path.join(opts.outdir, outfile)}")
    print(f"saved {os.path.join(base_path, outfile)}")

def get_ranges(mode, phototable):
    # select table bins to check
    #firstbin, lastbin = 10, 180 # 1m...400m
    firstbin, lastbin = 10, phototable.shape[0] #1m...1000m
    if mode == 'exact':
        r_range = n.arange(0, phototable.shape[0], 1)
        phi_range = n.arange(0, phototable.shape[1], 1)
        costheta_range = n.arange(0, phototable.shape[2], 1)
    elif mode == 'fine':
        r_range = n.arange(firstbin, lastbin, 8)
        phi_range = n.arange(0, phototable.shape[1], 2)
        costheta_range = n.arange(0, phototable.shape[2], 4)
    elif mode == 'coarse':
        r_range = n.arange(firstbin, lastbin, 10)
        phi_range = n.arange(0, phototable.shape[1], 4)
        costheta_range = n.arange(0, phototable.shape[2], 10)
    elif mode == 'snapshot':
        r_range = n.array([firstbin, 30, 50, 100, 140, 160, (lastbin-1), lastbin])-1
        if phototable.shape[1] == 36:
            phi_range = n.array([1, 10, 20, 36])-1
        elif phototable.shape[1] == 72:
            phi_range = n.array([1, 20, 40, 60, 72])-1
        costheta_range = n.array([1, 30, 50, 70, 100])-1
    elif mode == 'test':
        r_range = n.array([firstbin, 50, 140, lastbin])-1
        if phototable.shape[1] == 36:
            phi_range = n.array([1, 10, 36])-1
        elif phototable.shape[1] == 72:
            phi_range = n.array([1, 40, 72])-1
        costheta_range = n.array([1, 50, 100])-1

    return r_range, phi_range, costheta_range

def ks_calc(dim_table_values, dim_spline_vcomp):
                
    diff = abs(dim_spline_vcomp-dim_table_values)
    n_points = len(dim_table_values)

    max_diff = n.max(diff)

    #if mode == 'exact':
    prob = distributions.kstwo.sf(max_diff, n_points)
    #elif mode == 'asymp':
    #    prob = distributions.kstwobign.sf(D * np.sqrt(N))
    #else:
        # mode == 'approx'
    #    prob = 2 * distributions.ksone.sf(D, N)
    return prob

def rising_edge_test(edge, table_centers, table_values, spline_centers, spline_values):
    ##get times where CDF exceeds threshold
    index = 0
    for val in table_values:
        if val > edge:
            table_edge_t = table_centers[index]
            break
        else:
            table_edge_t = -1
        index += 1
    index = 0
    for val in spline_values:
        if val > edge:
            spline_edge_t = spline_centers[index]
            break
        else:
            spline_edge_t = -1
        index += 1

    return table_edge_t, spline_edge_t

def spatial_dims(bin_centers, norm, r, r_bin, r_range, phi, phi_bin, costheta, 
                 costheta_bin, xlabels, ylabels, xlimits, fileid, filetype, base_path, *args):
    # initialize data containers
    dim_table_centers, dim_table_values = {}, {}
    dim_spline_centers, dim_spline_values, dim_spline_vcomp = {}, {}, {}
    
    for dim in range(3):
        dim_spline_vcomp[dim] = []
        dim_spline_values[dim] = []
        dim_table_centers[dim] = bin_centers[dim]
        if dim == 0:
            dim_spline_centers[dim] = n.linspace(xlimits[dim][0]**0.5, xlimits[dim][1]**0.5, 200)**2
        elif dim == 2:
            dim_spline_centers[dim] = n.linspace(-1, 1, 200) #TODO: remove
        else:
            dim_spline_centers[dim] = n.linspace(xlimits[dim][0], xlimits[dim][1], 200)
                    
        if dim == 0:
            dim_table_values[dim] = LIGHTSCALE*norm[:, phi_bin, costheta_bin]
            for dim_table_center in dim_table_centers[dim]:
                dim_spline_vcomp[dim].append(LIGHTSCALE*
                                n.exp(splinetable.evaluate_simple([dim_table_center, phi, costheta, *args])))
        elif dim == 1:
            dim_table_values[dim] = LIGHTSCALE*norm[r_bin, :, costheta_bin]
            for dim_table_center in dim_table_centers[dim]:
                dim_spline_vcomp[dim].append(LIGHTSCALE*
                                n.exp(splinetable.evaluate_simple([r, dim_table_center, costheta, *args])))
        elif dim == 2:
             dim_table_values[dim] = LIGHTSCALE*norm[r_bin, phi_bin, :]
             for dim_table_center in dim_table_centers[dim]:
                 dim_spline_vcomp[dim].append(LIGHTSCALE*
                                n.exp(splinetable.evaluate_simple([r, phi, dim_table_center, *args])))
        dim_spline_vcomp[dim] = n.asarray(dim_spline_vcomp[dim])
        for dim_spline_center in dim_spline_centers[dim]:
            ##some problems with np.exp overflowing
            if n.isfinite(dim_spline_center) is False:
                n.nan_to_num(dim_spline_center)
            if dim == 0:
                dim_spline_value = LIGHTSCALE*n.exp(splinetable.evaluate_simple([dim_spline_center, phi, costheta, *args]))
            elif dim == 1:
                dim_spline_value = LIGHTSCALE*n.exp(splinetable.evaluate_simple([r, dim_spline_center, costheta, *args]))
            elif dim == 2:
                dim_spline_value = LIGHTSCALE*n.exp(splinetable.evaluate_simple([r, phi, dim_spline_center, *args]))
            dim_spline_values[dim].append(dim_spline_value)
        dim_spline_values[dim] = n.asarray(dim_spline_values[dim])

    absplot([r_bin, phi_bin, costheta_bin], [r, phi, costheta], bin_centers, dim_table_centers, 
             dim_table_values, dim_spline_centers, dim_spline_values, r_range,
             fileid, filetype, base_path, xlabels, ylabels, xlimits)

def timing_dims(t_bin_centers, dim_table_widths, dim_spline_edges, cdf, 
                r, r_bin, phi, phi_bin, costheta, costheta_bin,
                xlabels, ylabels, xlimits, fileid, filetype,
                base_path, make_plots=False, verbose=False):
    dim_table_centers = t_bin_centers
    dim_spline_centers = n.append(n.linspace(0., 0.9, 10), n.logspace(0., n.log10(xlimits[3][1]),200))
    dim_table_values = cdf[r_bin, phi_bin, costheta_bin, :]            

    dim_spline_values = []
    dim_spline_vcomp = []
    for dim_spline_center in dim_spline_centers:
        dim_spline_values.append(splinetable.evaluate_simple([r, phi, costheta, dim_spline_center]))
    for dim_table_center in dim_table_centers:
        dim_spline_vcomp.append(splinetable.evaluate_simple([r, phi, costheta, dim_table_center]))
    dim_spline_values = n.asarray(dim_spline_values)
    dim_spline_vcomp = n.asarray(dim_spline_vcomp)                    

    p_val = ks_calc(dim_table_values, dim_spline_vcomp)                    
    ##extract information from KS test
    if verbose == True:
        if p_val > 0.95:
            print(f"----- Good P-val!: {p_val} -----")
            print(f'{r_bin+1}, {phi_bin+1}, {costheta_bin+1}')
            print("---------------------------------")
        if p_val < 0.05:
            print(f"------ Low P-val!: {p_val} -----")
            print(f'{r_bin+1}, {phi_bin+1}, {costheta_bin+1}')
            print("---------------------------------")

    table_edge_1, spline_edge_1 = rising_edge_test(0.1, dim_table_centers, 
                        dim_table_values, dim_spline_centers, dim_spline_values)
    table_edge_2, spline_edge_2 = rising_edge_test(0.2, dim_table_centers, 
                        dim_table_values, dim_spline_centers, dim_spline_values)
    edges = [table_edge_1, spline_edge_1, table_edge_2, spline_edge_2]

    if make_plots == True:
        probplot([r_bin, phi_bin, costheta_bin], [r, phi, costheta], 
                dim_table_centers, dim_table_values, dim_spline_centers, dim_spline_values, 
                dim_spline_vcomp, dim_table_widths, dim_spline_edges, fileid, filetype,
                base_path, xlabels, ylabels, xlimits, p_val)
    return p_val, edges

def eval_p(result, low_r, med_r, high_r, 
        low_phi, med_phi, high_phi, 
        low_costheta, med_costheta, high_costheta, 
        low_rc, med_rc, high_rc, low_pc, med_pc, high_pc,
        p_val_list):
    
    p_val, r, r_bin, phi, phi_bin, costheta, costheta_bin = result

    p_val_list.append(p_val)

    if p_val > 0.98:
        #high_r.append(r)
        high_r.append(r_bin)
        high_phi.append(phi)
        high_costheta.append(costheta)
        high_rc.append((r_bin, costheta))
        high_pc.append((phi, costheta))
    elif p_val < 0.02:
        #low_r.append(r)
        low_r.append(r_bin)
        low_phi.append(phi)
        low_costheta.append(costheta)
        low_rc.append((r_bin, costheta))
        low_pc.append((phi, costheta))
    else:
        #med_r.append(r)
        med_r.append(r_bin)
        med_phi.append(phi)
        med_costheta.append(costheta)
        med_rc.append((r_bin, costheta))
        med_pc.append((phi, costheta))

def eval_edges(edge_slices, table_edge_1, spline_edge_1, table_edge_2, spline_edge_2):
    t1 = edge_slices[0]
    s1 = edge_slices[1]
    t2 = edge_slices[2]
    s2 = edge_slices[3]

    table_edge_1.append(t1)
    spline_edge_1.append(s1)
    table_edge_2.append(t2)
    spline_edge_2.append(s2)

def dim_wrapper(bin_centers, bin_widths, dim_spline_edges, norm, cdf, r_bin, r_range, phi_range, costheta_range,
                xlabels, ylabels, xlimits, fileid, filetype, base_path, ndim, make_plots=False):
    r = bin_centers[0][r_bin]
    r_width = bin_widths[0][r_bin]
    p_info_list = []
    edges_list = []
    for phi_bin in phi_range:
        phi = bin_centers[1][phi_bin]
        phi_width = bin_widths[1][phi_bin]
        for costheta_bin in costheta_range:
            costheta = bin_centers[2][costheta_bin]
            costheta_width = bin_widths[2][costheta_bin]

            # iterate over all space dimensions
            if ndim == 3:
                spatial_dims(bin_centers, norm, r, r_bin, r_range, phi, phi_bin, costheta, costheta_bin, 
                            xlabels, ylabels, xlimits, fileid, filetype, base_path)
                edges = []
            elif ndim == 4 and 'test' in args[1] and 'prob' not in args[1]:
                zen = float(fileid.split('_')[6][3:])
                z = float(fileid.split('_')[5][1:])
                spatial_dims(bin_centers, norm, r, r_bin, r_range, phi, phi_bin, costheta, costheta_bin, 
                             xlabels, ylabels, xlimits, fileid, filetype, base_path, zen)
                edges = []
            elif ndim == 4:
                p_val, edges  = timing_dims(bin_centers[3], bin_widths[3], dim_spline_edges, cdf, 
                                        r, r_bin, phi, phi_bin, 
                                        costheta, costheta_bin, xlabels, ylabels, 
                                        xlimits, fileid, filetype, base_path, 
                                        make_plots=make_plots, verbose=False)
                p_info_list.append((p_val, r, r_bin, phi, phi_bin, costheta, costheta_bin))
                edges_list.append(edges)
            elif ndim == 5:
                zen = float(fileid.split('_')[6][3:])
                z = float(fileid.split('_')[5][1:])
                spatial_dims(bin_centers, norm, r, r_bin, r_range, phi, phi_bin, costheta, costheta_bin, 
                             xlabels, ylabels, xlimits, fileid, filetype, base_path, zen, z)
                edges = []

    return (p_info_list, edges_list)

def spline_quality(opts, args, splinetable):
    num_cores = opts.cores
    do_plot = opts.make_plots
    if do_plot == "True":
        make_plots = True
    else:
        make_plots = False

    # check files
    if len(args) != 2:
        print(usage)
        sys.exit(0)
    print(f"args[0]: {args[0]}")
    print(f"args[1]: {args[1]}")
    fileid = args[0].rsplit('/')[-1].rsplit('.fits')[0]
    print(f"fileid: {fileid}")
    filetype = 'pdf'

    file_string = os.path.basename(args[1])
    file_type = file_string.split(".")

    if file_type[-2] not in ['abs', 'prob']:
       raise TypeError("Input file format not matching default - abs or prob")

    base = os.path.basename(args[0])
    name, extension = os.path.splitext(base)
    #base_path = os.path.join(opts.outdir, name)
    base_path = os.path.join(opts.outdir, base)
    base_path = base_path + "_" + file_type[-2]
    if not os.path.exists(os.path.abspath(base_path)):
        os.mkdir(os.path.abspath(base_path))

    # load and normalize photo table
    phototable = FITSTable.load(args[0])
    phototable.normalize()
    values = n.cumsum(phototable.values, axis=3)
    bin_centers = phototable.bin_centers
    bin_widths = phototable.bin_widths
    bin_centers[3] += phototable.bin_widths[3]/2.
    norm = values[:,:,:,-1]
    cdf = n.nan_to_num(values / norm.reshape(norm.shape + (1,)))
    shape = phototable.shape
    print(f"Phototable {shape}")

    # load spline fits table
    fitstable = splinefitstable.read(args[1])
    extents = fitstable.extents

    # global header values
    if TEXMODE:
        xlabels = ['$\\rm r~[m]$', '$\\upphi~[^\\circ]$', '$\\cos\\uptheta$', '$\\rm t~[ns]$']
    else:
        xlabels = ['r [m]', 'phi [deg]', 'cos(theta)', 't [ns]']
    ylabels = 3*['mean pe'] + ['cdf']
    xlimits = extents
    ndim = len(extents)

    mode = opts.mode.lower()
    r_range, phi_range, costheta_range = get_ranges(mode, phototable)
    
    ##p_value lists
    p_val_list = []
    low_r = []
    high_r = []
    low_phi = []
    high_phi = []
    low_costheta = []
    high_costheta = []
    med_r = []
    med_phi = []
    med_costheta = []
    low_rc = [] 
    med_rc = []
    high_rc = []
    low_pc = []
    med_pc = []
    high_pc = []
                    
    table_edge_1 = []
    table_edge_2 = []
    spline_edge_1 = []
    spline_edge_2 = []

    lin_edges = []
    log_edges = []
    dim_spline_edges = []
    
    if file_type[-2] == 'prob':
        lin_edges.append(n.diff(n.linspace(0., 0.9, 10))/2 + n.linspace(0., 0.9, 10)[1:])
        log_cen = n.logspace(0., n.log10(xlimits[3][1]),200)
        log_edges.append(n.diff(log_cen)/2 + log_cen[1:])
        lin_edges = n.array(lin_edges)
        log_edges = n.array(log_edges)
        first_log_edge = log_cen[0] - (log_edges[0][0] - log_cen[0])
        edges = n.append(lin_edges, first_log_edge)
        edges = n.append(edges, log_edges)
        dim_spline_edges = n.append(-0.05, 0.05)
        dim_spline_edges = n.append(dim_spline_edges, edges)

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        #with tqdm(total=len(r_range)) as progress:
        futures = []

        # table and spline parsing
        for r_bin in r_range:
            future = executor.submit(
                        dim_wrapper,
                        bin_centers, bin_widths, dim_spline_edges, norm, cdf,
                        r_bin, r_range, phi_range, costheta_range,
                        xlabels, ylabels, xlimits, fileid, filetype, base_path, ndim, make_plots)
            #future.add_done_callback(lambda p: progress.update())
            futures.append(future)
        results = wait(futures)
        ##end loop over dimensions

    for result in results.done:
        for p_slice in result.result()[0]:
            eval_p(p_slice, low_r, med_r, high_r, low_phi, med_phi, high_phi,
                    low_costheta, med_costheta, high_costheta, 
                    low_rc, med_rc, high_rc, low_pc, med_pc, high_pc,
                    p_val_list) 
        for edge_slice in result.result()[1]:
            eval_edges(edge_slice, table_edge_1, spline_edge_1, table_edge_2, spline_edge_2)

    #for now, only do P-value tests for timing
    if ndim != 4:
        exit(0)

    ##make p val plot from KS test
    par = fileid.split("_flat_")
    par = par[1]
    par = par.split("_")
    depth = par[0]
    depth = depth[1:]
    zen = par[1]

    font = {'family': 'serif',
            'weight': 'normal',
            'size': 18,
           }

    fig_p, ax_p = p.subplots()
    vals, bins, opts = ax_p.hist(p_val_list, bins=100, range=[0.0,1.0],
                                             color='royalblue', label='Full Range', alpha=0.5)
    fig_p.text(bins[0] * 0.1, n.max(vals) * 0.9, f"0.0:{vals[0]} \n 1.0:{vals[-1]}", fontdict=font)
    ax_p.legend(loc=0)
    ax_p.set_xlabel('P-value (From KS Test)')
    ax_p.set_ylabel('Entries')
    ax_p.set_title(f'Z{depth}, {zen}, ' + r'azi180  - All r$\theta$$\phi$')
    outfile = '%s_p_val.%s' % (fileid, filetype)
    fig_p.savefig(os.path.join(base_path, outfile))
    print(f"File save to: {base_path + outfile}")

    r_centers = bin_centers[0]
    r_widths = bin_widths[0]
    r_centers = n.array(r_centers)
    r_widths = n.array(r_widths)
    r_edges = r_centers[0] - r_widths[0]
    r_edges = n.append(r_edges, (r_centers + r_widths))
    colors = ['royalblue', 'turquoise', 'goldenrod']
    labels = [f'High P: {len(high_r)}', f'Med P: {len(med_r)}', f'Low P: {len(low_r)}']
    pr_set = n.arange(0, r_range[-1]+4, 2)
    r_vals = (n.linspace(n.sqrt(extents[0][0]), n.sqrt(extents[0][-1]), r_range[-1]))**2
    r_ticks = r_vals[0::55]

    fig_pr, ax_pr = p.subplots()
    vals_r, bins_r, opts_r = ax_pr.hist([high_r, med_r, low_r], bins=pr_set,
            #range=[r_range[0], r_range[-1]],
            edgecolor='black', linewidth=1.0, stacked=True,
            color=colors, label=labels)
    ax_pr.set_xlim(0, pr_set[-1])
    ax_tw = ax_pr.twiny()
    # Move twinned axis ticks and label from top to bottom
    ax_tw.xaxis.set_ticks_position("bottom")
    ax_tw.xaxis.set_label_position("bottom")
    ax_tw.spines["bottom"].set_position(("axes", -0.25))
    ax_tw.set_frame_on(True)
    ax_tw.patch.set_visible(False)
    for sp in ax_tw.spines.values():
        sp.set_visible(False)
    ax_tw.spines["bottom"].set_visible(True)
    ax_tw.set_xbound(0, extents[0][-1])
    #ax_tw.set_xticks(r_tick_labelss)
    ax_tw.set_xlabel('Distance from Cascade [m]')
    ax_tw.set_xticks(r_ticks)
    #ax_tw.set_xticklabels(bin_centers[0])
    ax_tw.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax_pr.legend(bbox_to_anchor=(0., 1.04, 1., .102),
                       ncol=3, mode="expand", borderaxespad=0., fontsize='small')
    ax_pr.set_xlabel('r Bin Number')
    ax_pr.set_ylabel('Entries')
    ax_pr.set_title(f'Z{depth}, {zen}, ' + r'azi180  - All $\theta$$\phi$', y=1.2)
    outfile = '%s_p_val_r.%s' % (fileid, filetype)
    fig_pr.tight_layout()
    fig_pr.savefig(os.path.join(base_path, outfile))
    print(f"File saved to: {base_path + outfile}")
    ##from IPython import embed; embed()

    fig_pp, ax_pp = p.subplots()
    vals_p, bins_p, opts_p = ax_pp.hist([high_phi, med_phi, low_phi],
            bins=len(phi_range), range=[bin_centers[1][0], bin_centers[1][-1]],
            edgecolor='black', linewidth=1.0, stacked=True,
            color=colors, label=labels)
    ax_pp.legend(loc=0)
    ax_pp.set_xlabel(r'$\phi$ from Cascade Dir. [$^{\circ}$]')
    ax_pp.set_ylabel('Entries')
    ax_pp.set_title(f'Z{depth}, {zen}, ' + r'azi180  - All r$\theta$')
    outfile = '%s_p_val_phi.%s' % (fileid, filetype)
    fig_pp.tight_layout()
    fig_pp.savefig(os.path.join(base_path, outfile))
    print(f"File saved to: {base_path + outfile}")

    fig_pt, ax_pt = p.subplots()
    vals_t, bins_t, opts_t = ax_pt.hist([high_costheta, med_costheta, low_costheta],
            bins=len(costheta_range), range=[bin_centers[2][0], bin_centers[2][-1]],
            edgecolor='black', linewidth=1.0, stacked=True,
            color=colors, label=labels)
    ax_pt.legend(loc=0)
    ax_pt.set_xlabel(r'cos($\theta$) from Cascade Dir.')
    ax_pt.set_ylabel('Entries')
    ax_pt.set_title(f'Z{depth}, {zen}, ' + r'azi180  - All r$\phi$')
    outfile = '%s_p_val_costheta.%s' % (fileid, filetype)
    fig_pt.tight_layout()
    fig_pt.savefig(os.path.join(base_path, outfile))
    print(f"File saved to: {base_path + outfile}")

    fig_prt, ax_prt = p.subplots()
    h = ax_prt.hist2d(high_r, high_costheta, bins=[len(r_range), len(costheta_range)])
    cbar = p.colorbar(h[3])
    cbar.set_label('# of Entries: P-Val > 0.98', labelpad=18, rotation=270)
    ax_prt.set_xlabel('r Bin Number')
    ax_prt.set_ylabel(r'cos($\theta$) from Cascade Dir.')
    outfile = '%s_p_val_high_r_costheta.%s' % (fileid, filetype)
    fig_prt.tight_layout()
    fig_prt.savefig(os.path.join(base_path, outfile))
    print(f"File saved to: {base_path + outfile}")
    
    fig_prt2, ax_prt2 = p.subplots()
    h = ax_prt2.hist2d(low_r, low_costheta, bins=[len(r_range), len(costheta_range)])
    cbar = p.colorbar(h[3])
    cbar.set_label('# of Entries: P-Val < 0.02', labelpad=18, rotation=270)
    ax_prt2.set_xlabel('r Bin Number')
    ax_prt2.set_ylabel(r'cos($\theta$) from Cascade Dir.')
    outfile = '%s_p_val_low_r_costheta.%s' % (fileid, filetype)
    fig_prt2.tight_layout()
    fig_prt2.savefig(os.path.join(base_path, outfile))
    print(f"File saved to: {base_path + outfile}")
    
    fig_prt3, ax_prt3 = p.subplots()
    h = ax_prt3.hist2d(med_r, med_costheta, bins=[len(r_range), len(costheta_range)])
    cbar = p.colorbar(h[3])
    cbar.set_label('# of Entries: P-Val > 0.02 & < 0.98', labelpad=18, rotation=270)
    ax_prt3.set_xlabel('r Bin Number')
    ax_prt3.set_ylabel(r'cos($\theta$) from Cascade Dir.')
    outfile = '%s_p_val_med_r_costheta.%s' % (fileid, filetype)
    fig_prt3.tight_layout()
    fig_prt3.savefig(os.path.join(base_path, outfile))
    print(f"File saved to: {base_path + outfile}")
            
    if n.min(table_edge_1) < n.min(spline_edge_1):
        min_e_1 = n.min(table_edge_1)
    elif n.min(table_edge_1) >= n.min(spline_edge_1):
        min_e_1 = n.min(spline_edge_1)
    if n.max(table_edge_1) > n.max(spline_edge_1):
        max_e_1 = n.max(table_edge_1)
    elif n.max(table_edge_1) <= n.max(spline_edge_1):
        max_e_1 = n.max(spline_edge_1)
    
    if n.min(table_edge_2) < n.min(spline_edge_2):
        min_e_2 = n.min(table_edge_2)
    elif n.min(table_edge_2) >= n.min(spline_edge_2):
        min_e_2 = n.min(spline_edge_2)
    if n.max(table_edge_2) > n.max(spline_edge_2):
        max_e_2 = n.max(table_edge_2)
    elif n.max(table_edge_2) <= n.max(spline_edge_2):
        max_e_2 = n.max(spline_edge_2)

    fig_e1, ax_e1 = p.subplots()
    vals, bins, sets = ax_e1.hist(table_edge_1, bins=200, range=[min_e_1, max_e_1], 
            color='royalblue', alpha=0.4, label='PhotoTable')
    vals, bins, sets = ax_e1.hist(spline_edge_1, bins=200, range=[min_e_1, max_e_1],
            color='goldenrod', alpha=0.4, label='Spline')
    fig_e1.suptitle('Rising Edge Time (10%)')
    ax_e1.set_xlabel('Rising Edge Time [ns]')
    ax_e1.set_ylabel('Entries')
    ax_e1.legend(loc=0)
    fig_e1.tight_layout()
    outfile = '%s_edge_1.%s' % (fileid, filetype)
    fig_e1.savefig(os.path.join(base_path, outfile))
    print(f"File saved to: {base_path + outfile}")
   
    table_edge_1 = n.array(table_edge_1)
    spline_edge_1 = n.array(spline_edge_1)
    edge_diff_1 = table_edge_1 - spline_edge_1
    fig_e1b, ax_e1b = p.subplots()
    vals, bins, sets = ax_e1b.hist(edge_diff_1, bins=200, range=[n.min(edge_diff_1), n.max(edge_diff_1)], 
            color='royalblue')
    fig_e1b.suptitle(r'$\Delta$ Rising Edge Time (10%)')
    ax_e1b.set_xlabel(r'$\Delta$ Rising Edge Time [ns]')
    ax_e1b.set_ylabel('Entries')
    fig_e1b.tight_layout()
    outfile = '%s_edge_1_delta.%s' % (fileid, filetype)
    fig_e1b.savefig(os.path.join(base_path, outfile))
    print(f"File saved to: {base_path + outfile}")

    fig_e2, ax_e2 = p.subplots()
    vals, bins, sets = ax_e2.hist(table_edge_2, bins=200, range=[min_e_2, max_e_2], 
            color='royalblue', alpha=0.4, label='PhotoTable')
    vals, bins, sets = ax_e2.hist(spline_edge_2, bins=200, range=[min_e_2, max_e_2],
            color='goldenrod', alpha=0.4, label='Spline')
    fig_e2.suptitle('Rising Edge Time (20%)')
    ax_e2.set_xlabel('Rising Edge Time [ns]')
    ax_e2.set_ylabel('Entries')
    ax_e2.legend(loc=0)
    fig_e2.tight_layout()
    outfile = '%s_edge_2.%s' % (fileid, filetype)
    fig_e2.savefig(os.path.join(base_path, outfile))
    print(f"File saved to: {base_path + outfile}")
    
    table_edge_2 = n.array(table_edge_2)
    spline_edge_2 = n.array(spline_edge_2)
    edge_diff_2 = table_edge_2 - spline_edge_2
    fig_e2b, ax_e2b = p.subplots()
    vals, bins, sets = ax_e2b.hist(edge_diff_2, bins=200, range=[n.min(edge_diff_2), n.max(edge_diff_2)], 
            color='royalblue')
    fig_e2b.suptitle(r'$\Delta$ Rising Edge Time (20%)')
    ax_e2b.set_xlabel(r'$\Delta$ Rising Edge Time [ns]')
    ax_e2b.set_ylabel('Entries')
    fig_e2b.tight_layout()
    outfile = '%s_edge_2_delta.%s' % (fileid, filetype)
    fig_e2b.savefig(os.path.join(base_path, outfile))
    print(f"File saved to: {base_path + outfile}")
    

# parse arguments
usage = "usage: %prog [options] phototable.fits splinetable.fits"
optparser = OptionParser(usage=usage)
optparser.add_option("--outdir", type="string", default="", help="Output directory for plots [%default]")
optparser.add_option("--mode", choices=('exact', 'fine', 'coarse', 'snapshot', 'test'), default='snapshot', help="Scan mode to make plots [%default]")
optparser.add_option("--cores", type="int", default=1, help="Number of parallel cores")
optparser.add_option("--make_plots", choices=('True', 'False'), default=False, help="make plots, it's slow")
if __name__ == "__main__":
    opts, args = optparser.parse_args()
    # load photo spline table
    
    splinetable = SplineTable(args[1])
    
    spline_quality(opts, args, splinetable)

##end
