# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 11:57:49 2013

@author: Rob
"""

import pandas as pd
import country_setup
import cPickle
from import_propensities import calculate_import_propensities as get_P

reload(country_setup)

__MAX_ITERATIONS__ = 1000
__DEFICIT_TOLERANCE__ = 1

class GlobalDemoModel(object):
    """
    The Python implementation of an ENFOLDing global demonstration model.
    
    The ENFOLDing global demonstration model is a collection of country-level
    input-output models and a set of linear parameters for connecting them
    together into an international trade network.
    Typically the model is set up once from data, then recalculated once
    certain parts of the model have been tweaked by calling 
    `recalculate_world`. The initial creation of the
    object performs the first (re)calculation of the model.   
    """
    
    def __init__(self, 
                 countries, 
                 sectors, 
                 imports, 
                 exports, 
                 import_propensities,
                 calculate=False):
        """
        Parameters
        ----------
        countries: dict
            A dictionary of Country objects, keyed on country name
        sectors: list
            A list of sector names
        imports: pandas.Series
            A series of import values, indexed on sector/country
        exports: pandas.Series
            A series of export values, indexed on sector/country
        import_propensities: dict
            A dictionary of pandas.Series, keyed on sector. Each Series is
            a pandas.DataFrame with an index of country names and the columns
            being the same country names. Each entry is the propensity of the
            column country to import from the row (index) country.
            
        Attributes
        ----------
        c : dict
            A dictionary of Country objects keyed on the country's ISO 3 
            code, e.g. Great Britain's ISO 3 code is GBR.
        countries : list
            A list of all the country ISO 3 codes. This is created by getting
            unique values from the country_iso3 column of sector_flows.

        """
        self.countries = countries
        self.sectors = sectors
        self.country_names = countries.keys()
        self.imports = imports
        self.exports = exports
        self.import_propensities = import_propensities

        self.id_list = _create_country_sector_ids(self.country_names, sectors)

        if calculate:
            # Perform the first calculation of the model
            self.recalculate_world()
    
    @classmethod    
    def from_pickle(cls, picklefilename):
        model = cPickle.load(open(picklefilename,'r'))
        return cls(model['countries'], model['sectors'],
                   model['imports'], model['exports'],
                   model['import_propensities'])

    @classmethod
    def from_data(cls, sector_flows, commodity_flows, services_flows):
        """
        Create a demo model from data
        
        Parameters
        ----------
        sector_flows : pandas.DataFrame
            Input-output flow values per from/to sector, per country
            per year
        commodity_flows : pandas.DataFrame
            Commodity flow values per from/to country, per sector
            per year
        services_flows : pandas.DataFrame
            Services sector flow values per from/to country, per
            sector, per year. These will usually have been produced
            via a balancing procedure from import/export totals.
            
        Returns
        -------
        GlobalDemoModel
        """              
        countries = country_setup.create_countries_from_data(sector_flows, 
                                                             commodity_flows)
        country_names = pd.unique(sector_flows['country_iso3']).tolist()
        sectors = _get_sector_names(sector_flows)
        trade_flows = pd.concat([services_flows, commodity_flows], 
                                join='inner')
        [stray_exports, 
         stray_imports, 
         relevant_flows] = _relevant_flows(trade_flows,countries)
        country_names.append('RoW')
        countries['RoW'] = country_setup.create_RoW_country(stray_exports, 
                                                            stray_imports, 
                                                            sectors)
        # Initialise M, E and P
        (M, E, P) = _initialise(relevant_flows, countries, sectors)
        return cls(countries, sectors, M, E, P, calculate=True)

    def recalculate_world(self):
        """ 
        Iterate between setting import demands and export demands until
        trade in all sectors balances.
        
        Runs the iterative process to calculate each country's import
        requirements, use the import propensities to calculate export requirements
        per sector-country, and apply these export requirements to each country's
        input-output model. This process is repeated at most __MAX_ITERATIONS__ times,
        or until the global trade deficit < __DEFICIT_TOLERANCE__
        
        Returns
        -------
        bool
            True if the world converged after `__MAX_ITERATIONS__`
        """
        countries = self.countries
        M = self.imports
        E = self.exports
        P = self.import_propensities
        for i in range(__MAX_ITERATIONS__):
            [M, E] = _iterate_model(M, E, P, countries)                
            deficit = _export_deficit(M, E)
            if abs(deficit) < __DEFICIT_TOLERANCE__:
                self.imports = M
                self.exports = E
                print "World recalculated after %i iterations." % i                
                return True
        print "Warning: World didn't converge after %i iterations." % __MAX_ITERATIONS__
        return False
            
    def trade_flows(self, sector):
        """
        From-country to-country flows for the given sector.
        
        Calculated on the fly as :math:`PM`, where :math:`P` is the propensity
        matrix for the given sector, and :math:`M` is the import demand vector
        for the given sector.
        """
        M = self.imports.ix[sector]
        P = self.import_propensities[sector]
        return P * M
    
    def set_final_demand(self, country_name, sector, value):
        country = self.countries[country_name]        
        f = country.f.copy()
        f[sector] = value
        # Get new import demand for country        
        new_i = country.recalculate_economy(final_demand=f)
        # Put new import demand into self.M
        M = self.imports.swaplevel(0,1,copy=False).sortlevel()
        M.ix[country_name] = new_i
        self.imports = M.swaplevel(0,1).sortlevel()
        # Recalculate world
        self.recalculate_world()
    
    def to_file(self, filename):
        """
        Create a pickle of the currentdemo model
        """
        model = {'countries': self.countries,
                 'sectors': self.sectors,
                 'imports': self.imports,
                 'exports': self.exports,
                 'import_propensities': self.import_propensities}
        cPickle.dump(model, 
                     open(filename,'w'))
#                     protocol=cPickle.HIGHEST_PROTOCOL)
    
    def get_id(self, country, sector):
        the_id = [ k for k, v in self.id_list.iteritems() 
                     if v['country'] == country and v['sector'] == sector ]
        if len(the_id) > 1:
            print "Key error in model's id_list: [%s, %s] has multiple entries" % (country, sector)
        if len(the_id) < 1:
            print "(%s, %s) not found" % (country, sector)
            return None
        return the_id[0]
        
    def get_country_sector(self, lookup_id):
        """
        
        Do some magic
        
        """
        if lookup_id in self.id_list:
            return self.id_list[lookup_id]
        else:
            return {"country":None, "sector":None}            
            
def _initialise(data, countries, sectors):
    """ 
    Perform step 0, initialisation, of the algorithm in the
    paper
    """
    M = _get_M(data, countries, sectors)
    E = _get_E(M)
    P = get_P(data, M, countries, sectors)
    print "Initialisation complete"
    return M, E, P

def _in(data, inlist, col_name, notin=False):
    if notin:
        return data[~data[col_name].isin(inlist)]
    else:
        return data[data[col_name].isin(inlist)]

def _iterate_model(imports, exports, import_propensities, countries):
    """ Calculate export requirements. Apply these requirements
    to the export demand of each country. """
    M = imports
    E = exports
    P = import_propensities
    
    E = _world_export_requirements(M, E, P)
    M = _world_import_requirements(countries, M, E)
    return M, E
        
def _get_M(data, countries, sectors):
    # Construct a blank dataframe to house the import values        
    M = pd.DataFrame([[s, c, 0] for s in sectors for c in countries],
                     columns=['sector','to_iso3','trade_value'])
    M = M.set_index(['sector', 'to_iso3'])
    M = M.add(data.groupby(['sector', 'to_iso3']).aggregate(sum), fill_value=0)
    M = M.rename(columns={'trade_value':'i'})
    return M.squeeze() # Convert one-column pd.DataFrame to pd.Series
 
def _get_E(imports):
    M = imports    
    E = M * 0
    E.index.names[1] = 'from_iso3'
    return E


def _export_deficit(imports, exports):
    """
    Sum over sectors of M - sum of sectors of E, take the square
    and sum the squares
    """
    M = imports
    E = exports
    world_imports = M.sum(level='sector')
    world_exports = E.sum(level='sector')
    
    return sum(pow(world_imports - world_exports, 2))

def _get_sector_names(flow_data):
    data = flow_data
    data = data[data['from_production_sector']]
    data = data[~pd.isnull(data['from_sector'])]
    return pd.unique(data['from_sector']).tolist()

def _world_export_requirements(imports, exports, import_propensities):
    """ Create the matrix of export requirements, E. This is
    done based on the import requirements of all sectors in
    all countries."""
    M = imports
    E = exports
    P = import_propensities
    
    for s, P_s in P.iteritems():
        i_s = M.ix[s] # Get this sector's imports
        e_s = P_s.dot(i_s) # Equation: e = P.i
        e_s.index.names = ['from_iso3']
        E.ix[s] = e_s.squeeze()
    return E
    
def _world_import_requirements(countries, imports, exports):
    """ For each country in countries, select the correct
    vector of exports from E and recalculate that countries
    economy on the basis of the new export demand"""
    M = imports
    E = exports
    
    for country_name, country in countries.items():
        # Pick the right column for the export matrix E
        # and apply it as an export vector to the current
        # country
        e = E.ix[:, country_name]
        i = country.recalculate_economy(final_demand=country.f, 
                                        exports=e)
        # Now put the new import values back into M, the
        # import vector. This is currently a bit tedious. Is there a better way?
        M = M.swaplevel(0, 1) # Now indexed Country, Sector
        M.ix[country_name] = i # Set the relevant country part
        M = M.swaplevel(1, 0) # Swapped back
    return M
                            
def _relevant_flows(trade_flows, countries):
    """
    Keep only flows to and from countries in 'countries'.
    Set all other flows either to or from RoW. Discared 
    all flows both from and to countries not in 'countries'.
    known_to_unknown and unknown_to_known are kept for the creation
    of the RoW country later.
    """
    fields = ['from_iso3', 'to_iso3', 'sector', 'trade_value']        
    data = trade_flows[~pd.isnull(trade_flows.sector)]
    
    # Drop flows to-from same country
    data = data[~(data.from_iso3==data.to_iso3)]
    
    # FROM HERE ON IN CAN BE IMPROVED WITH pd.Series.isin
    from_known = data[data.from_iso3.isin(countries)][fields]
    from_unknown = data[~data.from_iso3.isin(countries)][fields]
    
    known_to_known = from_known[from_known.to_iso3.isin(countries)][fields]
    known_to_unknown = from_known[~from_known.to_iso3.isin(countries)][fields]
    unknown_to_known = from_unknown[from_unknown.to_iso3.isin(countries)][fields]
    
    # Set all unknown flows to go to RoW
    known_to_unknown['to_iso3'] = 'RoW'
    unknown_to_known['from_iso3'] = 'RoW'
    
    # Sum the flows which are identical
    fields.remove('trade_value')
    known_to_known = known_to_known.groupby(fields, as_index=False).sum()
    known_to_unknown = known_to_unknown.groupby(fields, as_index=False).sum()
    unknown_to_known = unknown_to_known.groupby(fields, as_index=False).sum()
    
    relevant_flows = pd.concat([known_to_known,
                                known_to_unknown,
                                unknown_to_known])
    
    return known_to_unknown, unknown_to_known, relevant_flows

def _create_country_sector_ids(country_names, sectors):
    """ Create a unique id for each country/sector which
    will be used when visualising the model. """
    id_list = {}
    i = 1
    sectors_plus_fd = list(sectors)
    sectors_plus_fd.append('FD') # Add Final Demand for visualisation purposes           
    for c in country_names:   
        for s in sectors_plus_fd: 
            id_list[i] = ({"country":c, "sector":s})
            i += 1
    return id_list
