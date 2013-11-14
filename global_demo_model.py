# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 11:57:49 2013

@author: Rob
"""

import pandas as pd
import country_setup
import import_propensities

reload(country_setup)

__MAX_ITERATIONS__ = 200
__DEFICIT_TOLERANCE__ = 0.0001

class GlobalDemoModel(object):
    """ A collection of Country objects connected via a 
    matrix of sectoral trade propensities"""
    
    def __init__(self, sector_flows, commodity_flows, services_flows):
        self.c = country_setup.create_countries_from_data(sector_flows, commodity_flows)
        self.countries = pd.unique(sector_flows['country_iso3']).tolist()
        trade_flows = pd.concat([services_flows,commodity_flows],join='inner')
        [stray_exports, stray_imports, relevant_flows] = self._relevant_flows(trade_flows,self.countries)
        self.countries.append('RoW')
        self.c['RoW'] = country_setup.create_RoW_country(stray_exports,stray_imports)
        self.sectors = self._get_sector_names(sector_flows)
        self.id_list = self._create_country_sector_ids(self.c, self.sectors)
        
        self.P = import_propensities.calculate_import_propensities(relevant_flows,
                                                                   self.countries,
                                                                   self.sectors)
        self._assemble_world_matrices()
                    
    def _world_matrix(self, attribute_name):
        x = {}
        
        for name, c in self.c.iteritems():
            x[name] = getattr(c, attribute_name)
       
        return pd.concat(x, 1)
    
    def recalculate_world(self):
        """ 
        Runs the iterative process to calculate each country's ipmort
        requirements, use the import propensities to calculate export requirements
        per sector-country, and apply these export requirements to each country's
        input-output model. This process is repeated at most __MAX_ITERATIONS__ times,
        or until the global trade deficit < __DEFICIT_TOLERANCE__
        """
        countries = self.c
        for i in range(__MAX_ITERATIONS__):            
            self._assemble_world_matrices()
            self._iterate_model(countries, self.M, self.P)                
            self._assemble_world_matrices()
            # Row sum of M - row sum of E, take the square
            # and sum the squares
            deficit = sum(pow(self.M.sum(1) - self.E.sum(1),2))
            if abs(deficit) < __DEFICIT_TOLERANCE__:
                print "World recalculated after %i iterations." % i                
                return 1
        return 0

    def _get_sector_names(self,flow_data):
        data = flow_data.copy()
        data = data[data['from_production_sector']]
        data = data[~pd.isnull(data['from_sector'])]
        return pd.unique(data['from_sector']).tolist()

        
    def _assemble_world_matrices(self):
        self.M = self._world_matrix('i')
        self.E = self._world_matrix('e')

    def _iterate_model(self, countries, M, P):
        """ Calculate export requirements. Apply these requirements
        to the export demand of each country. """
        E = self._world_export_requirements(M, P)
        self._apply_new_export_requirements(countries, E)
    
    def _world_export_requirements(self, M, P):
        """ Create the matrix of export requirements, E. This is
        done based on the import requirements of all sectors in
        all countries."""
        E = []  
        for sector, i_j in M.groupby(level=0):
            i_j = i_j.transpose()
            P_j = P[sector]
            E.append (P_j.dot(i_j).transpose())
        return pd.concat(E, 0)
    
    # Not sure this is even being used at the moment!
    # There's code in recalculate_world which does the job    
    def _get_world_export_deficit(self, countries, M):
        """Calculate the difference between imports, M,
        and the current level of export of each country
        in countries. The sum of these differences is the world
        export deficit."""
        e = self._world_matrix("e")
        # Create an all-zeros pandas.Series with the same fields
        # as the first country's export Series
        deficit = e.iloc[:,0].copy() * 0
        # Calculate the world's deficit one country at a time
        for country, e_needed in M.iteritems():
            current_e = e[country]
            deficit = deficit + (e_needed - current_e)
        return deficit
        
    def _apply_new_export_requirements(self, countries, E):
        """ For each country in countries, select the correct
        vector of exports from E and recalculate that countries
        economy on the basis of the new export demand"""
        for country_name, country in countries.items():
            # Pick the right column for the export matrix E
            # and apply it as an export vector to the current
            # country
            country.recalculate_economy(final_demand=country.f, 
                                        exports=E[country_name])

    def _relevant_flows(self, trade_flows,countries):
        """
        THIS NEEDS WRITING.. NEED TO CUT DOWN commodity_flows AND
        services_flows TO ONLY THOSE FLOWS WE CARE ABOUT.
        ALL THE HARD WORK IS DONE IN _create_RoW_country
        """
        fields = ['from_iso3','to_iso3','sector','trade_value']        
        data = trade_flows[~pd.isnull(trade_flows.sector)]
        
        # FROM HERE ON IN CAN BE IMPROVED WITH pd.Series.isin
        from_known = _in(data, countries, 'from_iso3')[fields]
        from_unknown = _in(data, countries, 'from_iso3',notin=True)[fields]
        
        known_to_known = _in(from_known, countries, 'to_iso3')
        known_to_unknown = _in(from_known, countries, 'to_iso3',notin=True)        
        unknown_to_known = _in(from_unknown, countries, 'to_iso3')
        
        # Set all unknown flows to go to RoW
        known_to_unknown['to_iso3'] = 'RoW'
        unknown_to_known['from_iso3'] = 'RoW'
        
        # Sum the flows which are identical
        fields.remove('trade_value')
        known_to_unknown = known_to_unknown.groupby(fields).sum()
        unknown_to_known = unknown_to_known.groupby(fields).sum()
        
        return known_to_unknown, unknown_to_known, pd.concat([known_to_known,known_to_unknown,unknown_to_known])
        
    def _create_country_sector_ids(self, countries, sectors):
        """ Create a unique id for each country/sector which
        will be used when visualising the model. """
        id_list = {}
        i = 1
        sectors_plus_fd = list(sectors)
        sectors_plus_fd.append('FD') # Add Final Demand for visualisation purposes           
        for c in countries:   
            for s in sectors_plus_fd: 
                id_list[i] = ({"country":c, "sector":s})
                i += 1
        return id_list
        
    def get_id(self, country, sector):
        the_id = [ k for k,v in self.id_list.iteritems() 
                     if v['country'] == country and v['sector'] == sector ]
        if len(the_id) > 1:
            print "Key error in model's id_list: [%s, %s] has multiple entries" % (country, sector)
        if len(the_id) < 1:
            print "(%s, %s) not found" % (country, sector)
            return None
        return the_id[0]
        
    def get_country_sector(self, lookup_id):
        if lookup_id in self.id_list:
            return self.id_list[lookup_id]
        else:
            return {"country":None, "sector":None}            
            
def _in(data, inlist, col_name, notin=False):
    if notin:
        how = 'left'
    else:
        how = 'inner'
    matchfld = 'x123abc'
    match = pd.DataFrame({matchfld:inlist,'matched':True})
    x = data.merge(match,how=how,left_on=col_name,right_on=matchfld)
    
    if notin:
        return x[pd.isnull(x['matched'])].drop(['matched',matchfld],1)
    else:
        return x.drop(['matched',matchfld],1)
