# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:26:10 2013

@author: rob
"""
import numpy as np
from gdm_library import diagonalise

__eps__ = 1

class Country(object):
    """
    
    A country is simply a collection of the vectors and matrices specific to
    a country in the global demo model.
    
    """
    def __init__(self, name, final_demand, export_demand, import_demand,
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
        i : pandas.Series
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
        self.e = export_demand
        self.i = import_demand
        self.n = export_demand * 0
        
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
        final_demand : pandas.Series optional
            A vector of final demands, indexed on sector
        exports : pandas.Series optional
            A vector of export demands, indexed on sector
        investments : pandas.Series, optional
            A vector of investment demands, indexed on sector. Defaults to
            zero.
            
        Returns
        -------
        pandas.Series
            A vector of import demands, indexed on sector. Note that this
            function also sets the module-level variable `i`, so the return
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
            
        if (_change_is_significant(f, self.f) or 
            _change_is_significant(e, self.e) or
            _change_is_significant(n, self.n)):
            if self.name == 'RoW':
                # See section: Calibration of 'Rest of World' entity in the paper
                self.x = self._RoW_domestic_reqs(e)
                i = self._RoW_import_reqs(f)
            else:
                total_demand = f + e + n
                self.x = self._domestic_reqs(total_demand)
                i = self._import_reqs(self.x, total_demand)
            
            # Update Country-level variables:
            self.i = i
            self.n = n
            self.e = e
            self.f = f
        else:
            # The change in input vectors is not deemed big enough
            # to be worth recalculating anything
            i = self.i
        
        return i
    
    def gross_output(self):
        """
        The total value of output from every sector combined
        
        returns
        -------
        pandas.Series
        """
        return self.x.sum()
        
    def B(self):
        """
        Sector-to-sector flows. 
        
        These flows are not used in the model, but are useful
        for analysis and visualisation.
        """
        xhat = diagonalise(self.x)        
        return self.A.dot(xhat) # Total flows = A.xhat
        
    def B_dagger(self):
        """
        Sector-to-sector flows (domestic only)
        """
        return self.B.dot(self.D) 

    def B_star(self):
        """
        Sector-to-sector flows (imports only)
        """
        return self.B.dot(self._I - self.D)
        
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
                 
    def _import_reqs(self, domestic_requirements, total_demand):
        """
        Given domestic requirements, total demand and the import ratios, 
        calculate import demand.
        
        Calculates :math:`i = (I - D)^-1 Dx`
        where D is the diagonal matrix of import ratios.
        The inverse in the first part of the right-hand-side has no
        solution where D contains elements = 1. If these are replaced
        by 0.0 within the inverse only, the import for that product will
        simply be zero. They can then be retrospectively set to the equivalent
        total demand. This is safe because if import demand is genuinely zero,
        then total demand must also be zero.
        
        Notes
        -----
        The matrix algebra calculation has been replaced with a simple
        pandas element-wise binary operation.
        """
        if 1 in self.d.values:
            ddash = self.d.replace(1,0.0)
        else:
            ddash = self.d
        x = domestic_requirements
        i = ddash.mul(x).div(1 - ddash)
        i[i == 0] = total_demand # set to total_demand where 0
        return i

    def _domestic_reqs(self,total_demand):
        """calculate the production requirements, x, using the 
        technical coefficients matrix, A, and a given demand total
        demand F, fd + ex + investments.
        Solves x = [I - (I - D)A]^-1.(I - D)F
        The inverse on the right-hand-side has no solution if D has any zero
        elements.
        
        Notes
        -----
        Note that the diagonalisation of the import ratios, D, is no longer
        used, since multiplying rows of a matrix by elements of a vector
        is easy in pandas.
        """
        I = self._I
        A = self.A
        F = total_demand
        d = self.d
        x = np.linalg.solve(I - A.mul(1 - d, 'index'),F.mul(1 - d, 'index'))
    
        return x
                
    def _RoW_import_reqs(self,fd):
        return fd
        
    def _RoW_domestic_reqs(self, e):
        return e        
            
  
def _change_is_significant(old, new):
    """
    Test two comparable pandas.Series objects to see if 
    anything has changed 'much'
    
    True if any of the absolute differences in the values
    of the Series is greater than __eps__, a module-level
    variable.
    """
    return any(np.greater(np.abs(old - new), __eps__))

