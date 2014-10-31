# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:26:10 2013

@author: rob
"""
import numpy as np

class Country(object):
    """
    
    A country is simply a collection of the vectors and matrices specific to
    a country in the global demo model.
    
    """
    def __init__(self, name, final_demand,
                 technical_coefficients, 
                 import_ratios):
        """
        
        Parameters
        ----------
        name : string
            A human-readable name for the country. Usually its ISO 3 code.
        final_demand : pandas.Series
            A vector of final demands, indexed on sector
        export_demand : pandas.Series
            A vector of export demands, indexed on sector
        import_demand : pandas.Series
            A vector of import requirements, indexed on sector
        technical_coefficients : pandas.DataFrame
            A matrix of technical coefficients describing the 'recipe' of sector inputs
            required to make one unit of sector output. The index (rows) is the 
            'from sector' and the columns are the 'to sector'
        import_ratios : pandas.Series
            A vector of import ratios, defining what proportion of the total demand
            for a given sector output is supplied by imports vs. produced
            domestically. Indexed on sector

        Attributes
        ----------
        f : pandas.Series
            A vector of final demands, indexed on sector
        e : pandas.Series
            A vector of export demands, indexed on sector
        m : pandas.Series
            A vector of import requirements, indexed on sector
        d : pandas.Series
            A vector of import ratios, defining what proportion of the total demand
            for a given sector output is supplied by imports vs. produced
            domestically. Indexed on sector.
        A : pandas.DataFrame
            A matrix of technical coefficients describing the 'recipe' of sector inputs
            required to make one unit of sector output. The index (rows) is the 
            'from sector' and the columns are the 'to sector'
        D : pandas.DataFrame
            A diagonal matrix, whose diagonal elements are the elements of d.
            Both the index (rows) and the columns are sector.
        """
        self.name = name
        self.f = final_demand
        self.n = final_demand * 0
        self.e = final_demand * 0
        self.flow_deltas = technical_coefficients * 0
        self.stock_deltas = final_demand * 0
        
        if name != 'RoW':
            self.A = technical_coefficients
            self.d = import_ratios
            self._I = self.A * 0 + np.eye(self.A.shape[0])
            self.D = self._I * self.d
            self.recalculate_economy(self.f, self.e)
        else:
            self._I = 0
  
    def __repr__(self):
        return ' '.join(["Country:", self.name])
                
    def __str__(self):
        return ' '.join(["Country:", self.name])
        
    def recalculate_economy(self, 
                            final_demand = None, 
                            exports = None, 
                            investments = None):
        """
        Calculate a new import vector from a set of demands.
        
        Set the country's final demand, export and investment vectors, and
        run the algorithm outlined in the section 'Model Algorithm' of the
        paper. This calculates import requirements based on demand,
        technical coefficients and import ratios.
        If the changes in input vectors are negligable (< __eps__), 
        nothing is recalculated.
        
        Parameters
        ----------
        final_demand : pandas.Series, optional
            A vector of final demands, indexed on sector
        exports : pandas.Series optional
            A vector of export demands, indexed on sector
        investments : pandas.Series, optional
            A vector of investment demands, indexed on sector. Defaults to
            zero.
        tolerance : float
            At least one element of either final_demand, exports or 
            investment must be different from their existing 
            values for this procedure to actually do anything.
            `tolerance` sets what counts as "different".
            
            
        Returns
        -------
        pandas.Series
            A vector of import demands, indexed on sector. Note that this
            function also sets the module-level variable `m`, so the return
            value is only returned for convenience and can safely be discarded.
        """
        f = final_demand
        e = exports
        n = investments
        
        if f is None:
            f = self.f
        if e is None:
            e = self.e
        if n is None:
            n = self.n
            
        if self.name == 'RoW':
            # See section: modelling the 'Rest of World' in the paper
            self.x = self._RoW_domestic_reqs(e)
            m = self._RoW_import_reqs(f)
        else:
            self.x = self._domestic_reqs(tech_coeffs=self.A,
                                         final_demand=f, investments=n,
                                         exports=e, import_ratios=self.d)
            m = self._import_reqs(tech_coeffs=self.A,
                                  total_production=self.x,
                                  final_demand=f,
                                  exports=e, investments=n,
                                  import_ratios=self.d)
            
            # Update Country-level variables:
            self.m = m
            self.n = n
            self.e = e
            self.f = f

        return m
    
    def gross_output(self):
        """
        The total value of output from every sector combined
        
        returns
        -------
        pandas.Series
        """
        return self.x.sum()
        
    def Z(self):
        """
        Sector-to-sector flows. 
        
        These flows are not used in the model, but are useful
        for analysis and visualisation. Since RoW has no intermediate flows
        this will return the output of `_restofworld_Z()`.
        """
        # xhat = diagonalise(self.x)
        if self.name == 'RoW':
            return self._restofworld_matrix()
        else:
            return self.A * self.x # self.A.dot(xhat)

    def _restofworld_matrix(self):
        """
        A matrix of zeros, the columns and rows of which are labelled with
        the sectors
        """
        import pandas as pd
        sectors = self.f.index.values
        return pd.DataFrame(0, index=sectors, columns=sectors)
        
    def Z_dagger(self):
        """
        Sector-to-sector flows (domestic only)
        
        Solves :math:`Z(I - d)`
        """
        return self.Z().mul(1 - self.d)

    def Z_star(self):
        """
        Sector-to-sector flows (imports only)
        
        Solves :math:`(dZ)
        """
        return self.Z() * self.d # 
        
    def f_dagger(self):
        """
        Final demand vector (domestic only)
        
        Calculated from the total final demand and the import ratios, D.
        """
        return (self._I - self.D).dot(self.f)

    def f_star(self):
        """
        Final demand vector (imports only)

        Calculated from the total final demand and the import ratios, D.
        """
        return self.D.dot(self.f)
                 
    def _import_reqs(self, tech_coeffs, 
                     total_production, final_demand, exports,
                     investments, import_ratios):
        """
        Given domestic requirements, total demand and the import ratios, 
        calculate import demand.
        
        Calculates :math:`m = (I - \hat{d})^{-1} \hat{d}(x-e)`
        where D is the diagonal matrix of import ratios.
        The exception to this is if any of the elements of :math:`\hat{d}`
        are 1. In these cases :math:`m_s=A_{s,:}x_s + f_s + n_s`.
        
        Notes
        -----
        The matrix algebra calculation has been replaced with a simple
        pandas element-wise binary operation.
        """
        d = import_ratios
        if 1 in d.values:
            ddash = d.replace(1,0.0)
        else:
            ddash = d
        A = tech_coeffs
        x = total_production
        f = final_demand
        e = exports
        n = investments
        # m = (d / (1 - d)) * (x - e)
        m = ddash.mul(x - e).div(1 - ddash)
        # set to Ax + f + n where where d = 1
        m[d == 1] = A.ix[d == 1].dot(x) + f[d == 1] + n[d == 1]
        return m

    def _domestic_reqs(self, tech_coeffs, final_demand, 
                       investments, exports, import_ratios):
        """calculate the production requirements, x, using the 
        technical coefficients matrix, A, and a given total
        demand F, fd + ex + investments.
        Solves :math:`(I - (I - \hat{d})A)x=(I - \hat{d})(f + n) + e`
        setting :math:`x_s=e_s` where :math:`d_s=0`.
        
        Notes
        -----
        Note that the diagonalisation of the import ratios, D, is no longer
        used, since multiplying rows of a matrix by elements of a vector
        is easy in pandas.
        """
        I = self._I
        A = tech_coeffs
        f = final_demand
        n = investments
        e = exports
        d = import_ratios
        x = e * 0 + np.linalg.solve(I - A.mul(1 - d, 'index'),
                            (f + n).mul(1 - d, 'index') + e)
    
        return x
                
    def _RoW_import_reqs(self,fd):
        return fd
        
    def _RoW_domestic_reqs(self, e):
        return e        
                    
  
def _change_is_significant(old, new, tolerance):
    """
    Test two comparable pandas.Series objects to see if 
    anything has changed 'much'
    
    True if any of the absolute differences in the values
    of the Series is greater than tolerance.
    """
    #return any(np.greater(np.abs(old - new), tolerance))
    return any(np.greater(np.abs(old - new), 0))


