# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:26:10 2013

@author: rob
"""
import numpy as np
from gdm_library import diagonalise

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
        
    def recalculate_economy(self, final_demand, exports, investments=None):
        """
        Calculate a new import vector from a set of demands.
        
        Set the country's final demand, export and investment vectors, and
        run the algorithm outlined in the section 'Model Algorithm' of the
        paper. This calculates import requirements based on demand,
        technical coefficients and import ratios.
        Additionally, recalculate :math:`f^*`, :math:`f^{dagger}`, :math:`B^*` and
        :math:`B^{dagger}`.
        
        Parameters
        ----------
        final_demand : pandas.Series
            A vector of final demands, indexed on sector
        exports : pandas.Series
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
        
        if investments is None:
            investments = (exports * 0)
            investments.name = "Investments"
        total_demand = final_demand + exports + investments
        if self.name == 'RoW':
            # See section: Calibration of 'Rest of World' entity in the paper
            self.x = self._RoW_domestic_reqs(exports)
            i = self._RoW_import_reqs(final_demand)
        else:
            self.x = self._domestic_reqs(total_demand)
            i = self._import_reqs(self.x, total_demand)
            
        # Update Country-level variables:
        self.i = i
        self.n = investments
        self.e = exports
        self.f = final_demand
        
        return i
    
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
        Calculates :math:`i = (I - D)^-1 Dx`
        where D is the diagonal matrix of import ratios.
        The inverse in the first part of the right-hand-side has no
        solution where D contains elements = 1. If these are replaced
        by 0.0 within the inverse only, the import for that product will
        simply be zero. They can then be retrospectively set to the equivalent
        total demand. This is safe because if import demand is genuinely zero,
        then total demand must also be zero.
        """
        I = self._I       
        D = self.D
        Ddash = D.replace(1,0.0)
        x = domestic_requirements
        i = np.linalg.solve(I - Ddash, Ddash.dot(x))
        i[i==0] = total_demand # set to total_demand where 0
        return i

    def _domestic_reqs(self,total_demand):
        """calculate the production requirements, x, using the 
        technical coefficients matrix, A, and a given demand total
        demand F, fd + ex + investments.
        Solves x = [I - (I - D)A]^-1.(I - D)F
        The inverse on the right-hand-side has no solution if D has any zero
        elements.
        """
        I = self._I
        A = self.A
        F = total_demand
        D = self.D
        x = np.linalg.solve(I - (I - D).dot(A), (I - D).dot(F))
        
        return x
                
    def _RoW_import_reqs(self,fd):
        return fd
        
    def _RoW_domestic_reqs(self, e):
        return e        
            
  

