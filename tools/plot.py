import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
# For seriation and other R functions
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()

__plot_dims__ = [24, 24] # inches
__verbose__ = False

def set_verbose(verbose=True):
    """
    Makes functions report progress if `verbose` is True
    """
    global __verbose__
    __verbose__ = verbose

def _report(message):
    if __verbose__:
        print message

def _set_fig_size(fig):
    fig.set_figwidth(__plot_dims__[0])
    fig.set_figheight(__plot_dims__[1])   

def _squarify(df, fill_value=0):
    """
    Make sure df is square, regardless of whether it is initially
    long or wide. Set new values to `fill_value`
    """
    _report("Squarifying matrix")
    new = 0 * (df + df.T)
    return (new.fillna(0) + df).fillna(fill_value)


def heatmap(df, ax, xrotation=90, rescale=True, labels=True, **kwargs):
    """
    A heatmap of the DataFrame `df` with columns
    along the bottom and rows down the side

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame to plot from. Should work directly with `pcolor`
    ax: matplotlib.axes
        The axes to plot to
    xrotation: int
        The rotation of the x tick labels
    """
    _report("Generating heatmap")
    df = _squarify(df)
    if rescale:
        df = df / df.max().max()
    if 'cmap' not in kwargs.keys():
        kwargs['cmap'] = plt.cm.Greys
    pc = ax.pcolor(df, **kwargs)
    ax.set_xlim([0, df.shape[1] + .5])
    ax.set_ylim([0, df.shape[0] + .5])
    ax.xaxis.tick_top()
    if labels:
        ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
        ax.set_xticklabels(df.index, minor=False, rotation=xrotation, ha='left')
        ax.set_yticklabels(df.columns, minor=False)
    ax.invert_yaxis()
    return pc

def _seriation_order(df):
    # R imports
    _report("Getting R seriation order")
    seriation = importr('seriation')
    stats = importr('stats')
    base = importr('base')

    names = np.array(base.names(df))
    d = stats.dist(df, method='euclidean')
    o = seriation.seriate(d)
    # R is 1-indexed! Python is 0-indexed, so adjust for this
    order = [x - 1 for x in np.array(seriation.get_order(o))]
    return names[order]

def _seriate(df):
    """
    Reorder the rows and columns of `df` according to
    `_seriation_order()`

    Note that rows and columns must have the same set of labels
    and that those labels cannot contain spaces as R inserts
    a `.` for each space.
    """
    order = _seriation_order(df)
    with open('order_%s.txt' % time.time(), 'w') as f:
        for x in order:
            f.write('%s\n' % x)
    return df.loc[order, order], order

def seriated_heatmap(df, ax=None, order=None, **kwargs):
    """
    Reorder the rows and columns of `df` and draw a heatmap to create a visual
    clustering

    The DataFrame will only be reordered (using seriation) if no `order` is passed.
    kwargs are passed to `pcolor`, the heatmap function in matplotlib.
    See https://cran.r-project.org/web/packages/seriation/index.html for
    details of the seriation algorithm which requires an R installation and the
    seriation package
    """
    df = _flatten_multiindex(df)
    if order is None:
        df, order = _seriate(df)
    else:
        df = df.copy().loc[order, order]
    fig, ax = _fig_ax(ax)
    pcolor = heatmap(df.fillna(0), ax, rescale=False, **kwargs)
    fig.tight_layout()
    return fig, ax, order

def _fig_ax(ax):
    """
    Get the figure and the axis, or create new ones if required
    """
    if ax is None:
        fig, ax = plt.subplots()
        _set_fig_size(fig)
    else:
        fig = ax.get_figure()
    return fig, ax

def _flatten_multiindex(df, sep='_', flatten_columns=True):
    """
    Flatten a multiindexed DataFrame to a single-level index
    with values separated by `sep`
    """
    df = df.copy()
    idx_values = [sep.join(i).replace(' ', '_') for i in df.index.values]
    idx_name = sep.join(df.index.names)
    idx = pd.Index(idx_values, name=idx_name)
    df.index = idx
    if flatten_columns:
        cols = [sep.join(i).replace(' ', '_') for i in df.columns.values]
        df.columns = cols
    return df

def density_plot(df, ax=None, **kwargs):
    """
    Convert df to a Series and draw a Gaussian kernel density plot
    """    
    try:
        df = df.stack()
    except AttributeError:
        pass
    fig, ax = _fig_ax(ax)
    density = gaussian_kde(df)
    xs = np.linspace(df.min(),df.max(),200)
    colour = next(ax._get_lines.color_cycle)
    x_density = density(xs)
    ax.hist(df, alpha=0.2, normed=True, color=colour)
    ax.plot(xs,x_density, color=colour, **kwargs)

def heat_map(df, cmap_name='Blues'):
    n = len(df)
    fig, ax = plt.subplots()
    _set_fig_size(fig)
    ax.imshow(df, interpolation='nearest', cmap=cmap_name)
    ax.set_xticks(np.linspace(0, n-1, n))
    ax.set_xticklabels(df.columns, rotation=90)
    ax.set_yticks(np.linspace(0, n-1, n))
    ax.set_yticklabels(df.index)

def scatter_plot(df, xcol=0, ycol=1, labelled=True, regression_line=False,
                 ymin=0, ymax=None, xmin=0, xmax=None, label_join=' ', **kwargs):
    """
    Create a new labelled scatter plot with data from `df`.
    Keyword arguments are passed to df.plot
    
    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The DataFrame to plot from
    xcol: str, int
        The name (or index position) of the x column
    ycol: str, int
        The name (or index position) of the y column
    labelled: bool
        Add labels using the index of `df`
    regression_line: bool
        Add a regression line
    label_join: string
        Character to join index levels by for label if index has multiple
        levels
    """
    fig, ax = plt.subplots()
    _set_fig_size(fig)
    df.plot(xcol, ycol, kind='scatter',
        ax=ax, c=range(len(df)),
        colormap='Spectral', colorbar=False,
        s=120, alpha=0.8,
        edgecolor='None', 
        ylim=[ymin, ymax], xlim=[xmin, xmax])
    if labelled:
        ax = annotate_axes(ax, df, xcol, ycol, label_join)
    if regression_line:
        add_regression_line(ax, x=df[[xcol]].squeeze(), y=df[[ycol]].squeeze())
    return fig, ax

def annotate_axes(ax, df, xcol=0, ycol=1, label_join=' '):
    """
    Annotate the axes `ax` with the DataFrame `df`
    
    `df` must have a single-level index with the text of the
    annotation, and the x-values must be stored in a column called
    `xcol` and similar for the y-values
    
    Parameters
    ----------
    ax: matplotlib.axes.AxesSubplot
        The axes containg the graph to be annotated
    df: pandas.core.frame.DataFrame
        The dataframe containing the x, y data, indexed with
        the values to use as the annotation
    xcol: str
        The name of the x column
    ycol: str
        The name of the y column
    """
    for k, v in df.iterrows():
        if isinstance(k, (tuple, list)):
            k = label_join.join(k)
        ax.annotate(k, [v[xcol], v[ycol]],
                    xytext=(10,-5), textcoords='offset points',
                    family='sans-serif', fontsize=14, 
                    color='darkslategrey')
    return ax

def add_regression_line(ax, x, y):
    from pandas.stats.api import ols
    # For some reason, ols doesn't like MultiIndexed series
    y = y.reset_index(drop=True)
    x = x.reset_index(drop=True)
    res = ols(y=y, x=x)
    add_straight_line(ax, m=res.beta['x'], c=res.beta['intercept'])
    annotate_regression_line(ax, res)

def annotate_regression_line(ax, ols_results):
    """
    Add slope, intercept and p-value to the plot in 
    the axes `ax`.
    """
    fractional_position = 0.95
    m = ols_results.beta['x']
    c = ols_results.beta['intercept']
    p = ols_results.p_value['x']
    x_min = ax.get_xlim()[0]
    x_max = ax.get_xlim()[1]
    y_min = ax.get_ylim()[0]
    y_max = ax.get_ylim()[1]
    x = (x_max - x_min) * fractional_position + x_min
    y = (y_max - y_min) * fractional_position + y_min    
    spacing = (y_max - y_min) / 30
    annotate(ax, "slope: %.2f" % m, [x, y])
    annotate(ax, "intercept: %.2f" % c, [x, y - spacing])
    annotate(ax, "p-value: %.3f" % p, [x, y - 2 * spacing])
    
def annotate(ax, text, xy):
    ax.annotate(text, xy,
                color='0.8', fontsize=15, family='monospace',
                horizontalalignment='right')

def _get_xaxis_range(ax):
    xlim = ax.get_xlim()
    return xlim[1] - xlim[0]

def _get_yaxis_range(ax):
    ylim = ax.get_ylim()
    return ylim[1] - ylim[0]

def add_straight_line(ax, m, c):
    """
    Adds a straight line with equation :math:`y=mx + c`
    to the axes `ax`
    """
    ax.set_autoscale_on(False)
    if _get_xaxis_range(ax) > _get_yaxis_range(ax):
        x = ax.get_xlim()
        y = [m * x_i + c for x_i in x]
    else:
        y = ax.get_ylim()
        x = [(y_i - c) / m for y_i in y]
    ax.plot(x, y, '0.8', linewidth=2, alpha=0.8, zorder=0)
    


