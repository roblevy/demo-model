# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:26:10 2013

@author: rob
"""

import numpy as np
from gdm_library import diagonalise

class Country(object):
    """
    A country is simply a collection of the following vectors and matrices:
      1. f: a vector of final demands
      2. e: a vector of export demands
      3. i: a vector of import requirements
      4. A: a matrix of technical coefficients describing the 'recipe' of sector inputs
            required to make one unit of sector output
      5. P: a matrix of import propensities, with rows relating to countries and
            columns relating to sectors
      6. d: a vector of import ratios, defining what proportion of the total demand
            for a given sector output is supplied by imports vs. produced domestically
    """
    def __init__(self, name, f, e, i,
                 technical_coefficients, 
                 import_propensities, 
                 import_ratios):
        self.name = name
        self.f = f
        self.e = e
        self.i = i
        self.A = technical_coefficients
        self.P = import_propensities
        self.d = import_ratios
  
    def __repr__(self):
        return ' '.join(["Country:", self.name])
                
    def __str__(self):
        return ' '.join(["Country:", self.name])
        
    def recalculate_economy(self, final_demand, exports, investments=None):
        if investments is None:
            investments = (exports * 0)
            investments.name = "Investments"
            
        total_demand = final_demand + exports + investments
        x = self.domestic_reqs(total_demand)
        i = self.import_reqs(x)

        # Update Country-level variables:
        I = np.eye(self.A.shape[0])        
        self.x = x
        self.i = i
        self.n = investments
        self.e = exports
        self.f = final_demand
        self.f_star = self.d.dot(self.f)
        self.f_dagger = (I - self.d).dot(self.f)
        
        xhat = diagonalise(x)        
        self.B = self.A.dot(xhat) # Total flows = A.xhat
        self.B_dagger = self.B.dot(self.d) 
        self.B_star = self.B.dot(I - self.d)
                                
    def import_reqs(self, domestic_requirements):
        """Calculates i = (I + d)^-1 Dx 
        where d is the matrix of import ratios"""
        I = np.eye(self.A.shape[0])        
        d = self.d
        x = domestic_requirements
        i = np.linalg.solve(I - d, d.dot(x))
        return i

    def domestic_reqs(self,total_demand):
        """calculate the production requirements, x, using the 
        technical coefficients matrix, A, and a given demand total
        demand F, fd + ex + investments.
        Solves x = [I - (I - d)A]^-1.(I - d)F
        """
        I = np.eye(self.A.shape[0])
        A = self.A
        F = total_demand
        d = self.d
        x = np.linalg.solve(I - (I - d).dot(A), (I - d).dot(F))
       
        return x
                

    def _add_names(self, names_from, M):
        """ Takes a blank matrix M and adds the names of the
        rows/columns from the matrix names_from"""
        return names_from * 0 + M
        
   
    
  

