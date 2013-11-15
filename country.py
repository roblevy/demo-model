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
      5. d: a vector of import ratios, defining what proportion of the total demand
            for a given sector output is supplied by imports vs. produced domestically
    """
    def __init__(self, name, f, e, i,
                 technical_coefficients, 
                 import_ratios):
        self.name = name
        self.f = f
        self.e = e
        self.i = i
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
        Run the algorithm outlined in the section 'Model Algorithm' of the
        paper.        
        """
        
        if investments is None:
            investments = (exports * 0)
            investments.name = "Investments"
        total_demand = final_demand + exports + investments
        if self.name == 'RoW':
            # See section: Calibration of 'Rest of World' entity in the paper
            self.x = self._RoW_domestic_reqs(exports)
            self.i = self._RoW_import_reqs(final_demand)
        else:
            self.x = self._domestic_reqs(total_demand)
            self.i = self._import_reqs(self.x, total_demand)
            self.f_star = self.D.dot(self.f)
            self.f_dagger = (self._I - self.D).dot(self.f)
            
            # I've commented this stuff out, because I'm not sure we really
            # need B_dagger and B_star
#            xhat = diagonalise(x)        
#            self.B = self.A.dot(xhat) # Total flows = A.xhat
#            self.B_dagger = self.B.dot(self.D) 
#            self.B_star = self.B.dot(I - self.D)

        # Update Country-level variables:
        self.n = investments
        self.e = exports
        self.f = final_demand
        
        return self.e, self.i
                                
    def _import_reqs(self, domestic_requirements, total_demand):
        """
        Calculates i = (I - D)^-1 D.x 
        where d is the matrix of import ratios.
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
            
  

