# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 15:00:05 2014

Collection of tools which relate to the treatment of various
model elements, such as final demand, as vectors

@author: quintopia, rob

http://forums.xkcd.com/viewtopic.php?f=17&t=29603#p956761
https://gist.github.com/iizukak/1287876
"""
import numpy as np
import pandas as pd
from matplotlib import cm

eps = 1e-12
cmap = cm.get_cmap('Spectral')

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

# This section due to iizukak
# See: https://gist.github.com/iizukak/1287876
# ===========================
def gs_coefficient(v1, v2):
    """
    Gram-schmidt coefficient
    """
    return v1.dot(v2) / v1.dot(v1)
 
def proj(v1, v2):
    """
    Project v1 using the appropriate Gram-schmidt coefficient
    """
    return v1 * gs_coefficient(v1, v2)

def gs(x, y):
    """
    Gram-schmidt algorithm for producing an orthonormal
    basis including the plane spanned by x and y
    """
    A = pd.DataFrame(np.eye(len(x))) # Identity matrix
    x = normalise(x)
    A[0] = x.values # First col is x
    A[1] = y.values # Second col is y
    
    Y = pd.DataFrame()
    for i in range(len(A)):
        temp_vec = A[i]
        for col in Y.columns:
            proj_vec = proj(Y[col], A[i])
            temp_vec = temp_vec - proj_vec
        Y[i] = normalise(temp_vec)
    return Y
# ===========================

def normalise(x):
    """
    Divide by L2 norm
    """
    return x / norm(x)

def norm(x):
    """
    L2 norm, calculated as sqrt of sum of squares
    """
    return np.sqrt(x.pow(2).sum())

def equate_lengths(x, y):
    """
    Make y the same length (in L2 norm terms) as x
    """
    return y  * norm(x) / norm(y)

def equate_sum(x, y):
    """
    Make the elements of y sum to the sum of the elements of x
    """
    return y * x.sum() / y.sum()

def twod_rotation_matrix(n, theta):
    """
    A 2x2 rotation matrix for angle theta
    """
    A = pd.DataFrame(np.eye(n))
    A[0][0] = np.cos(theta)
    A[1][0] = -np.sin(theta)
    A[0][1] = np.sin(theta)
    A[1][1] = np.cos(theta)
    return A
        
def vector_rotate(x, y, theta):
    """
    An implementation of this
    http://forums.xkcd.com/viewtopic.php?f=17&t=29603#p956761
    
    First scale y such that it has the same length as x.
    Then, create a 2-d rotation matrix for theta.
    Then, use the Gram-Schmidt process to create a rotation matrix
    in the plane spanned by x and y.
    The final rotation matrix is :math:`brb^{-1}`.
    Rescale the final vector so the elements sum to the same as the
    elements of x.
    """
    y = equate_lengths(x, y) # L2 norm
    r = twod_rotation_matrix(len(x), theta) # r is 2x2
    b = gs(x, y) # Gram-schmidt
    rotation_matrix = b.dot(r).dot(np.linalg.inv(b))
    rotated = rotation_matrix.dot(x.values)
    rotated.index = x.index
    return equate_sum(x, rotated)

def angle_between(x, y):
    """
    The angle in radians between x and y
    """
    return np.arccos(normalise(x).dot(normalise(y)))

def is_orthonormal(basis):
    """
    Check the dot product between all dimensions is zero
    """
    size = basis.shape[0]
    zero_dot_product = [basis[a].dot(basis[b]) < eps 
        for a in range(size) for b in range(size) if a != b]
    return np.all(zero_dot_product)

def plot_rotation(x, y, steps=10):
    """
    Make a plot of the steps of rotation from x to y
    """
    max_angle = angle_between(x, y)
    theta_range = np.linspace(0, max_angle, steps)
    steps = [vector_rotate(x, y, i) for i in theta_range]
    index = [r"$\theta = %.0f$" % np.rad2deg(i) for i in theta_range]
    columns = ['index %i' % (i + 1) for i in range(len(x))]
    steps = pd.DataFrame(steps, index=index)
    steps.columns = columns
    steps = sort_cols_by_first_row(steps)
    steps.plot(linewidth=4, alpha=0.8, colormap=cmap)

def sort_cols_by_first_row(df):
    """
    Sort the columns of a dataframe by the values in the first row.
    
    Useful for plot_rotation to make the indices ordered by value
    for colouring.
    """
    return df.transpose().sort(columns=df.index[0]).transpose()
##%%    
_test_x = pd.Series([0.5, 3, 0.1, 5, 20, 3, 13])
_test_y = pd.Series([1, 8, 12, 6, 6, 9, 2])


