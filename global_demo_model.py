# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 11:57:49 2013

@author: Rob
"""

import pandas as pd
import numpy as np
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
                 calculate=False,
                 silent=False,
                 tolerance=__DEFICIT_TOLERANCE__):
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
        self.tolerance = tolerance
        self._silent = silent
        self.countries = countries
        self.sectors = sectors
        self.country_names = countries.keys()
        self.imports = imports
        self.exports = exports
        self._import_propensities = import_propensities

        self.id_list = _create_country_sector_ids(self.country_names, sectors)
        self.deltas = pd.DataFrame()
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
    def from_data(cls, sector_flows, commodity_flows, 
                  services_flows=None, silent=False,
                  tolerance=__DEFICIT_TOLERANCE__):
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
        services_flows : pandas.DataFrame optional
            Services sector flow values per from/to country, per
            sector, per year. These will usually have been produced
            via a balancing procedure from import/export totals.
        silent : boolean optional
            The model shouldn't report any statuses to the console
        Returns
        -------
        GlobalDemoModel
        """              
        countries = country_setup.create_countries_from_data(sector_flows, 
                                                             commodity_flows)
        country_names = pd.unique(sector_flows['country_iso3']).tolist()
        sectors = _get_sector_names(sector_flows)
        if services_flows is None:
            services_flows = pd.DataFrame()
        trade_flows = pd.concat([services_flows, commodity_flows])
        [stray_exports, 
         stray_imports, 
         relevant_flows] = _relevant_flows(trade_flows,countries)
        country_names.append('RoW')
        countries['RoW'] = country_setup.create_RoW_country(stray_exports, 
                                                            stray_imports, 
                                                            sectors)
        # Initialise M, E and P
        (M, E, P) = _initialise(relevant_flows, countries, sectors, 
                                silent=silent)
        model = cls(countries, sectors, M, E, P, 
                    calculate=True, silent=silent, tolerance=tolerance)
        model._silent = silent
        return model

    def set_tolerance(self, tolerance):
        self.tolerance = tolerance        
    
    def recalculate_world(self, 
                          tolerance=None,
                          countries=None):
        """ 
        Iterate between setting import demands and export demands until
        trade in all sectors balances.
        
        Runs the iterative process to calculate each country's import
        requirements, use the import propensities to calculate export requirements
        per sector-country, and apply these export requirements to each country's
        input-output model. This process is repeated at most __MAX_ITERATIONS__ times,
        or until the global trade deficit < self.tolerance
        
        Parameters        
        ----------
        tolerance : float, optional
            the value of the export deficit below which iteration
            stops
        
        Returns
        -------
        bool
            True if the world converged after `__MAX_ITERATIONS__`
        """
        if tolerance is None:
            tolerance = self.tolerance
        countries = self.countries
        P = self._import_propensities

        E = self.exports * 0 # Set exports to zero

        for i in range(__MAX_ITERATIONS__):
            # Get imports from exports and an understanding of the
            # internal country structure. (E is zero first time round)
            M = _world_import_requirements(countries, E)
            # Calculate global import/export deficit
            deficit = _export_deficit(M, E)
            # Now get exports given import demands and import propensities
            E = _world_export_requirements(M, P)
            if abs(deficit) < tolerance:
                # stop iterating!
                self.imports = M
                self.exports = E
                if not self._silent:
                    print "World recalculated after %i iterations." % i
                return True    

        if not self._silent:
            print "Warning: World didn't converge " \
                  "after %i iterations." % __MAX_ITERATIONS__

        return False
            
    def trade_flows(self, sector, imports=None):
        """
        From-country to-country flows for the given sector.
        
        Calculated on the fly as :math:`PM`, where :math:`P` is the propensity
        matrix for the given sector, and :math:`M` is the import demand vector
        for the given sector.
        """
        if imports is None:
            imports = self.imports
        m = imports.ix[sector]
        P = self._import_propensities[sector]
        return P * m
    
    def total_production(self):
        """
        total production,x, for all countries (except RoW) in all sectors
        """
        country_names = sorted([c for c in self.country_names if c != 'RoW'])
        x = [self.countries[c].x for c in country_names]
        return pd.concat(x, keys=country_names, names=['country'])
    
    def set_final_demand(self, country, sector, value):
        """
        Set the `sector` element of the $f$ vector in `country`
        to `value`.
        
        This method also updates `self.imports`. It will additionally
        recalculate the world.
        
        Parameters
        ----------
        country_name : str
            The name of the country
        sector : str
            The name of the sector
        value : float
            The new value of the final demand
        tolerance : float optional
            Tolerance to use when recalculating the world
        """
        value = 0 if value < 0 else value
            
        if country.f[sector] != value:

            country.f[sector] = value
            self.recalculate_world()
    
    def set_technical_coefficient(self, country, 
                                  from_sector, to_sector, value):
        """
        Sets `a_rs` to `value` as long as the sum of column `s`
        would be less than one. Otherwise, does nothing.
        """
        value = 0 if value < 0 else value
        
        column_sum = country.A[to_sector].sum()
        technical_coefficient = country.A.ix[from_sector, to_sector]
        if technical_coefficient != value:
            if column_sum - technical_coefficient + value < 1:
                country.A.ix[from_sector, to_sector] = value
                self.recalculate_world()
   
    def set_import_ratio(self, country, sector, value):
        print "not implemented yet"
        
    def set_import_propensity(self, sector, from_country, to_country, value):
        print "not implemented yet"
   
    def final_demand(self):
        """
        A `pandas.Series` of final demands
        
        The Series is indexed on country_name/sector
        
        Returns:
        --------
         : pandas.Series
        """        
        df = pd.DataFrame()
        for name, country in self.countries.iteritems():
            fd = country.f.reset_index(name='final_demand')
            fd['country_iso3'] = name
            df = pd.concat([df, fd], 0)
        df = df.set_index(['country_iso3','sector']).squeeze()
        df.name = 'final_demand'
        return df
        
    def import_propensities(self):
        return self._import_propensities
        
    def technical_coefficients(self):
        A = {}
        for name, c in self.countries.iteritems():
            if name != 'RoW':
                A[name] = c.A
        return pd.Panel.from_dict(A)

    def import_ratios(self, as_matrix=False):
        """
        get import ratios from each country in the model
        
        If as_matrix = False, returns a pd.DataFrame of import ratios, 
        with countries as columns, and sectors as the index (rows)
        Otherwise returns a pd.Panel, keyed on country name, where each
        element is a DataFrame with a diagonalised imjport propensity matrix.
        
        Parameters
        ----------
        as_matrix : boolean
            Return diagonalised D matrices, or simple d vectors?
            
        Returns
        -------
        pd.Generic
        """
        D = {}
        for name, c in self.countries.iteritems():
            if name != 'RoW':
                if as_matrix:
                    D[name] = c.D
                else:
                    D[name] = c.d
        if as_matrix:
            return pd.Panel.from_dict(D)
        else:
            return pd.DataFrame.from_dict(D)

    def gross_output(self, with_RoW = False):
        """
        The gross output of each country in the model
        
        Returns
        -------
        pandas.Series
            A pandas.Series of gross_outputs, indexed on country.
        """
        country_names = list(self.country_names) # Make a copy
        if not with_RoW:
            country_names.remove('RoW')
        g_out = pd.Series(0, index=country_names)
        g_out.name = 'gross_output'
        for c_name in country_names:
            c = self.countries[c_name]
            g_out.ix[c_name] = c.gross_output()
        return g_out.sort_index()
    
    def to_file(self, filename):
        """
        Create a pickle of the currentdemo model
        """
        model = {'countries': self.countries,
                 'sectors': self.sectors,
                 'imports': self.imports,
                 'exports': self.exports,
                 'import_propensities': self._import_propensities}
        cPickle.dump(model, 
                     open(filename,'wb'))
#                     protocol=cPickle.HIGHEST_PROTOCOL)
    
    def adjacency_matrix(self):
        """
        A single, unified adjancency matrix of all flows
        
        This uses the fact that country/sector-country-sector
        flow :math:`y_{r,s}^{(i,j)}` can be calculated with 
        :math:`a_{r,s}^{(j)} x_{s}^{(j)} d_{r}^{(j)} p_{r}^{(i,j)}`
        """
        # Some preparation for building huge matrices indexed
        # by e.g. "GBR Wood"
        countries = self.country_names
        countries.sort()
        sectors = self.sectors
        sectors.sort()
        cs_labels = ['%s %s' % (c, s) for c in countries 
            for s in sectors]
        sc_labels = ['%s %s' % (c, s) for s in sectors
            for c in countries]
        c = len(self.countries)
        s = len(self.sectors)
        empty = np.zeros([c * s, c * s])
        #%%
        # Put together the csxcs flows matrix
        # First the internal flows:
        # Z = Axhat
        # Zdagger = Z(I - dhat)
        io_flows = empty.copy()
        for i, country in enumerate(countries):    
            if country != 'RoW':
                country = self.countries[country]
                start = i * s
                end = start + s
                io_flows[start:end, 
                         start:end] = country.Z_dagger()
        io_flows = pd.DataFrame(io_flows, index=cs_labels, columns=cs_labels)
        #%%
        # Now the international flows:
        # y = axdp
        P = empty.copy()
        for i, sector in enumerate(sectors):
            start = i * c
            end = start + c
            P[start:end, start:end] = self.import_propensities()[sector]
        P = pd.DataFrame(P, index=sc_labels, columns=sc_labels)
        
        A = empty.copy()
        D = empty.copy()
        X = empty.copy()
        for i, country in enumerate(countries):
            c = self.countries[country]
            start = i * s
            end = start + s
            if country != 'RoW':
                A[start:end, start:end] = c.A
                D[start:end, start:end] = np.diag(c.d)
                X[start:end, start:end] = np.diag(c.x)
        A = pd.DataFrame(A, index=cs_labels, columns=cs_labels)
        D = pd.DataFrame(D, index=cs_labels, columns=cs_labels)
        X = pd.DataFrame(X, index=cs_labels, columns=cs_labels)
        P_tilde = P.ix[cs_labels,cs_labels]
        # Used to do this: trade_flows = P_tilde.dot(D).dot(A).dot(X)
        trade_flows = P_tilde.dot(A.mul(np.diag(X), axis='columns') \
            .mul(np.diag(D), axis='index'))
        # now combine to produce all_flows
        all_flows = io_flows + trade_flows
        all_flows.index.name = 'from'
        all_flows.columns.name = 'to'
        return all_flows
    
    def _calculate_deltas(self, new_countries, old_countries, 
                          imports, tolerance):
    
        deltas = pd.DataFrame()
        for s in self.sectors:
            # Flow deltas
            new_flows = self.trade_flows(s, imports).stack()
            old_flows = self.trade_flows(s).stack()
            delta = _deltas(new_flows, old_flows, s, 'trade', tolerance)
            if len(delta) > 0:
                delta = delta.rename(columns={'level_1':'country2',
                                              'from_iso3':'country1',
                                              'to_iso3':'country2'})
                delta = delta[(delta.country1 != 'RoW') & 
                              (delta.country2 != 'RoW')]
                deltas = delta.append(deltas)
            # Total production deltas
            new_x = self.total_production(s, new_countries)
            old_x = self.total_production(s)
            delta = _deltas(new_x, old_x, s, 'x', tolerance)
            if len(delta) > 0:
                delta = delta.rename(columns={'level_1':'to_iso3',
                                              'country':'country1'})
                delta = delta[(delta.country1 != 'RoW')]
                deltas = deltas.append(delta)
        return deltas

    def to_pajek(self, filename):
        """
        create a .net file according to the Pajek specification
        
        See http://www.mapequation.org/apps/MapGenerator.html#fileformats
        for details.
        """
        if filename[-4:] != '.net':
            filename += '.net'
        print 'Writing to %s. This takes a while...' % filename
        adj = self.adjancency_matrix()
        flows = adj.stack().reset_index()
        nodes = zip(*pd.factorize(adj.columns))
        from_nodes = pd.factorize(flows['from'])
        to_nodes = pd.factorize(flows['to'])
        flows['from_number'] = from_nodes[0]
        flows['to_number'] = to_nodes[0]
        # Now build the output file
        out = '*Vertices %i\n' % len(nodes)
        out += '\n'.join(['%i "%s"' % (i, x) for i, x in nodes])
        out += '\nArcs %i\n' % len(flows)
        out += flows.to_string(columns=['from_number', 'to_number', 0],
                               header=False, index=False)
        with open(filename, 'w') as text_file:
            text_file.write(out)
            for i, x in flows.iterrows():
                row = '%s %s %s' % (x['from_number'],
                                    x['to_number'],
                                    x[0])
                text_file.write(row)
        print '%s written' % filename
                       

    def get_id(self, country, sector):
        the_id = [ k for k, v in self.id_list.iteritems() 
                     if v['country'] == country and v['sector'] == sector ]
        if len(the_id) > 1:
            print "Key error in model's id_list: [%s, %s] has " \
                "multiple entries" % (country, sector)
        if len(the_id) < 1:
            print "(%s, %s) not found" % (country, sector)
            return None
        return the_id[0]
        
    def get_country_sector(self, lookup_id):
        """
        
        Not documented yet.
        
        """
        if lookup_id in self.id_list:
            return self.id_list[lookup_id]
        else:
            return {"country":None, "sector":None}   
            
def _initialise(data, countries, sectors, silent=False):
    """ 
    Perform step 0, initialisation, of the algorithm in the
    paper
    """
    M = _create_M(data, countries, sectors)
    E = _create_E(M)
    P = get_P(data, M, countries, sectors)
    if not silent:
        print "Initialisation complete"
    return M, E, P

def _in(data, inlist, col_name, notin=False):
    if notin:
        return data[~data[col_name].isin(inlist)]
    else:
        return data[data[col_name].isin(inlist)]

        
def _create_M(data, countries, sectors):
    # Construct a blank dataframe to house the import values        
    M = pd.DataFrame([[s, c, 0] for s in sectors for c in countries],
                     columns=['sector','to_iso3','trade_value'])
    M = M.set_index(['sector', 'to_iso3'])
    M = M.add(data.groupby(['sector', 'to_iso3']).aggregate(sum), fill_value=0)
    M = M.rename(columns={'trade_value':'i'})
    return M.squeeze() # Convert one-column pd.DataFrame to pd.Series
 
def _insert_import_demand_into_M(all_imports,
                                 country_name,
                                 new_imports):
    """
    Insert a newly calculated import demand from a single country
    into the model's vector of all import demands
    """
    M = all_imports
    # Now put the new import values back into M, the
    # import vector. 
    # TODO This is currently a bit tedious. Is there a better way?
    M = M.swaplevel(0, 1) # Now indexed Country, Sector
    M.ix[country_name] = new_imports # Set the relevant country part
    return M.swaplevel(1, 0) # Swapped back

def _country_import_demand(country,
                           exports=None,
                           investments=None,
                           final_demand=None):
    return country.recalculate_economy(final_demand=final_demand,
                                       investments=investments, 
                                       exports=exports)
            
def _create_E(imports):
    M = imports    
    E = M * 0
    E.index.names[1] = 'from_iso3'
    return E

def _export_deficit(imports, exports):
    """
    Sum over sectors of M - sum of sectors of E, take the absolute value
    and sum.
    """
    M = imports
    E = exports
    world_imports = M.sum(level='sector')
    world_exports = E.sum(level='sector')
    
    return sum(np.abs(world_imports - world_exports))

def _get_sector_names(flow_data):
    data = flow_data
    data = data[data['from_production_sector']]
    data = data[~pd.isnull(data['from_sector'])]
    return pd.unique(data['from_sector']).tolist()

def _world_export_requirements(imports, import_propensities):
    """ Create the matrix of export requirements, E. This is
    done based on the import requirements of all sectors in
    all countries."""
    M = imports
    P = import_propensities
    sectors = sorted(P.keys())

    all_E = map(_sector_export_requirements,
                sectors, 
                [M.ix[s] for s in sectors],
                [P[s] for s in sectors])
    
    all_E = pd.concat(all_E, keys=sectors, names=['sector'])

    return all_E

def _sector_export_requirements(sector, sector_imports, 
                                sector_import_propensities):
    i_s = sector_imports
    P_s = sector_import_propensities
    e_s = P_s.dot(i_s)
    e_s.index.names = ['from_iso3']
    return e_s.squeeze()
    
def _world_import_requirements(countries, exports):
    """ For each country in countries, select the correct
    vector of exports from E and recalculate that countries
    economy on the basis of the new export demand"""
    E = exports
    cnames = sorted(countries.keys())
    all_M = map(_country_import_demand, 
                [countries[c] for c in cnames],
                [E.ix[:, c] for c in cnames])
    all_M = pd.concat(all_M, keys=cnames, names=['country'])
    all_M = all_M.swaplevel(0,1).sortlevel()

    return all_M


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
        
    
def _deltas(new, old, sector, type_name, tolerance):
    """
    Calculate `new' - `old' and return those values greater than `tolerance'
    
    Parameters
    ----------
    old : pd.Series
    
    new : pd.Series
    
    tolerance : float        
    """
    deltas = pd.DataFrame(new - old, columns=['delta'])
    deltas['type'] = type_name
    deltas['sector'] = sector
    return deltas[deltas.delta.abs() > tolerance].reset_index()
