import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

__plot_dims__ = [12, 12] # inches

def _set_fig_size(fig):
    fig.set_figwidth(__plot_dims__[0])
    fig.set_figheight(__plot_dims__[1])   

def heatmap(df, ax, xrotation=45):
    """
    A heatmap of the DataFrame `df` with columns
    along the bottom and rows down the side
    """
    ax.pcolor(df, cmap=plt.cm.Greys, alpha=0.8)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(df.index, minor=False, rotation=xrotation, ha='left')
    ax.set_yticklabels(df.columns, minor=False)
    ax.set_xlim([0, df.shape[1] + .5])
    ax.set_ylim([0, df.shape[0] + .5])
    ax.invert_yaxis()

def density_plot(df, ax=None, **kwargs):
    """
    Convert df to a Series and draw a Gaussian kernel density plot
    """    
    try:
        df = df.stack()
    except AttributeError:
        pass
    density = gaussian_kde(df)
    xs = np.linspace(df.min(),df.max(),200)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
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

def scatter_plot(df, xcol=0, ycol=1, labelled=True,
                 ymin=0, ymax=None, xmin=0, xmax=None):
    """
    Create a new labelled scatter plot with data from `df`.
    
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
    """
    fig, ax = plt.subplots()
    _set_fig_size(fig)
    df.plot(xcol, ycol, kind='scatter',
        ax=ax, c=range(len(df)),
        colormap='Spectral', s=120, alpha=0.8,
        edgecolor='None', 
        ylim=[ymin, ymax], xlim=[xmin, xmax])
    if labelled:
        ax = annotate_axes(ax, df, xcol, ycol)
    return fig, ax

def annotate_axes(ax, df, xcol=0, ycol=1):
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
        ax.annotate(k, [v[xcol], v[ycol]],
                    xytext=(10,-5), textcoords='offset points',
                    family='sans-serif', fontsize=14, 
                    color='darkslategrey')
    return ax

def add_regression_line(ax, x, y):
    from pandas.stats.api import ols
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
    annotate(ax, "p-value: %.2g" % p, [x, y - 2 * spacing])
    
def annotate(ax, text, xy):
    ax.annotate(text, xy,
                color='0.8', fontsize=15, family='monospace',
                horizontalalignment='right')

def add_straight_line(ax, m, c):
    """
    Adds a straight line with equation :math:`y=mx + c`
    to the axes `ax`
    """
    ax.set_autoscale_on(False)
    x = ax.get_xlim()
    y = [m * x_i + c for x_i in x]
    ax.plot(x, y, '0.8', linewidth=2, alpha=0.8, zorder=0)
    


