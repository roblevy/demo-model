# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:26:10 2013

@author: rob
"""

import numpy as np
from gdm_library import diagonalise

class Country(object):
    """A country input-output model with attendant import and 
    import-propensity matrices for inclusion in a global model"""
    def __init__(self, name, data):
        self.name = name
        self.x = data["x"]
        self.i = data["i"]
        self.B_dagger = data["B_dagger"]
        self.B_star = data["B_star"]
        self.B = self.B_dagger + self.B_star
        self.A = data["A"]
        self.f = data["f"]
        self.f_dagger = data['f_dagger']
        self.f_star = data['f_star']
        self.e = data["e"]
        self.n = np.zeros(self.e.shape)
        self.D = data["D"] # Import ratios
        self.sectors = [s for s in self.D] # an array of sector names
        self.P = {}   # import_propensities must be
                      # explicitly set later
        self.num_sectors = self.A.shape[1]
  
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
        self.f_star = self.D.dot(self.f)
        self.f_dagger = (I - self.D).dot(self.f)
        
        xhat = diagonalise(x)        
        self.B = self.A.dot(xhat) # Total flows = A.xhat
        self.B_dagger = self.B.dot(self.D) 
        self.B_star = self.B.dot(I - self.D)
                                
    def import_reqs(self, domestic_requirements):
        """Calculates i = (I + D)^-1 Dx 
        where D is the matrix of import ratios"""
        I = np.eye(self.A.shape[0])        
        D = self.D
        x = domestic_requirements
        i = np.linalg.solve(I - D, D.dot(x))
        return i

    def domestic_reqs(self,total_demand):
        """calculate the production requirements, x, using the 
        technical coefficients matrix, A, and a given demand total
        demand F, fd + ex + investments.
        Solves x = [I - (I - D)A]^-1.(I - D)F
        """
        I = np.eye(self.A.shape[0])
        A = self.A
        F = total_demand
        D = self.D
        x = np.linalg.solve(I - (I - D).dot(A), (I - D).dot(F))
       
        return x
                

    def _add_names(self, names_from, M):
        """ Takes a blank matrix M and adds the names of the
        rows/columns from the matrix names_from"""
        return names_from * 0 + M
        
   
    
  

