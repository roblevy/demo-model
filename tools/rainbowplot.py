# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 18:15:46 2014

@author: rob
"""
import numpy as np
from matplotlib import cm
from matplotlib.pyplot import legend
from matplotlib.colors import LinearSegmentedColormap

__alpha__ = 0.6 # Transparency
__s__ = 120 # Marker size
__cmap__ = cm.get_cmap('Spectral') # Colour map

def _init(ax, x, y, df, keep_lims, draw_x_equals_y):
    if x is None or y is None:
        x = df.icol(0)
        y = df.icol(1)
    x, y = _data_init(x, y)
    if not keep_lims:
        _axis_init(ax, x, y)
    if draw_x_equals_y:
        _draw_x_equals_y(ax)
    return x, y
    
def _data_init(x, y):
    x = x.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    # If x and/or y are pandas objects, convert them
    # to numpy matrices first
    try:
        x = x.as_matrix()
    except AttributeError:
        pass
    try:
        y = y.as_matrix()
    except AttributeError:
        pass
    return x, y
    
def _axis_init(ax, x, y, draw_x_equals_y=False, upper_tol=5):
    """
    Set the axis object to just fit the data,
    with an upper tolerance of `upper_tol` percent
    """
    xlim = _expand_upper_limit([np.nanmin(x), np.nanmax(x)], upper_tol)
    ylim = _expand_upper_limit([np.nanmin(y), np.nanmax(y)], upper_tol)
    # Expand the upper limit by upper_tol percent
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def _expand_upper_limit(lim, upper_tol):
    """
    Increase the upper limit of lim = [lower, upper]
    by `upper_tol` percent
    """
    expanded = list(lim)
    expanded[1] += (lim[1] - lim[0]) * upper_tol / 100
    return expanded


def _draw_x_equals_y(ax):
    ax.plot([max(ax.get_xlim()[0], ax.get_ylim()[0]), 
             min(ax.get_xlim()[1], ax.get_ylim()[1])],
            [max(ax.get_xlim()[0], ax.get_ylim()[0]), 
             min(ax.get_xlim()[1], ax.get_ylim()[1])], '0.25')        
    
    
def rainbow_scatter(ax, x=None, y=None, df=None,
                    labels=None, marker='o', 
                    keep_lims=False, draw_x_equals_y=False):
    x, y = _init(ax=ax, x=x, y=y, df=df,
                 keep_lims=keep_lims, 
                 draw_x_equals_y=draw_x_equals_y)
    ax.scatter(x, y,
               c=np.linspace(0, 1, len(x)),
               alpha=__alpha__, s=__s__, marker=marker,
               edgecolor='none', 
               cmap=__cmap__)
    if labels is not None:
        for i in range(len(labels)):
            ax.annotate(labels[i], [x[i], y[i]],
            fontname='Droid sans', size=14, weight=400,
            color='0.01', alpha=0.6)
            
def group_scatter(ax, x=None, y=None, df=None, 
                  groupby_args=None, plot_args=None, legend_args=None,
                  labels=None, marker='o', 
                  keep_lims=False, draw_x_equals_y=False):
    x, y = _init(ax=ax, x=x, y=y, df=df,
                 keep_lims=keep_lims, 
                 draw_x_equals_y=draw_x_equals_y)
    c = np.linspace
    n = df.groupby(**groupby_args).ngroups
    legend_entries = []
    for i, (k, gb) in enumerate(df.groupby(**groupby_args)):
        x = gb.icol(0)
        y = gb.icol(1)
        points = ax.scatter(x, y,
                   c=cm.get_cmap('Spectral')(float(i) / n),
                   alpha=__alpha__, s=__s__, marker=marker,
                   edgecolor='none')
        points.set_label(k)
        legend_entries.append(points)
    ax.legend(legend_entries, 
              [le.get_label() for le in legend_entries], 
              scatterpoints=1, **legend_args)

#%%    
def _get_new_colormap(end_rgb={'red':0, 'green':0, 'blue':0}):
    """
    Create a new version of the Spectral palette that ends in end_rgb

    end_rgb should be a dict of RGB values keyed on colour name
    """
    cmap = cm.get_cmap('Spectral_r')
    rgb = {}
    rgb['red'] = cmap._segmentdata['red']
    rgb['green'] = cmap._segmentdata['green']
    rgb['blue'] = cmap._segmentdata['blue']
    for key, col in rgb.iteritems():
        # Rescale to allow addition of another colour
        new_scale = [float(x + 1) / (len(col)) for x in range(len(col))]
        new_col = [(new_scale[i], old[1], old[2]) for i, old in enumerate(col)]
        new_col = [(0, end_rgb[key], end_rgb[key])] + new_col
        rgb[key] = new_col
    return LinearSegmentedColormap('SpectralBlack', rgb)

SpectralBlack = _get_new_colormap()
SpectralWhite = _get_new_colormap({'red':1, 'green':1, 'blue':1})
