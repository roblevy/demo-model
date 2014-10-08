# -*- coding: utf-8 -*-
"""
Collection of tools which relate to the treatment of various
model elements, such as final demand, as vectors

@author: rob
"""

def herfindahl(x):
    """
    Calculate the normalised Herfindahl index of the 
    vector `x`.
    
    The Herfindahl index is a measure of the diversity of the
    elements of `x`.
    1 represents complete domination of a single element,
    0 represents perfect diversity, with each element being
    equal to :math:`1/N`
    
    The basic index is calculated as
    :math:`H=\sum_{i=1}^N{s_i^2}`
    where :math:`s_i` is the share of element :math:`i`.
    It is then normalised by :math:`H^*=\\frac{H - 1/N}{1 - 1/N}`
    
    Parameters
    ----------
    x : pandas.Series
    The vector to be calculated

    Returns
    -------
    A number between 0 and 1
    """
    n = float(len(x))
    s = x.div(x.sum()) # Shares
    h = s.pow(2).sum() # Non-normalised Herfindahl index
    return (h - 1/n)/(1 - 1/n) # Normalise