# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 11:57:49 2013

Edited in virtualbox

@author: Rob
"""

import pandas as pd
import numpy as np
import country_setup
import cPickle
from country import Country
from import_propensities import calculate_import_propensities as get_P
from itertools import product
import json

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
                 calculate=True,
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
            unique values from the country column of sector_flows.

        """
        self.tolerance = tolerance
        self._silent = silent
        self._calculate = calculate
        self.countries = countries
        self.sectors = sectors
        self.country_names = self._countries_not_RoW(names=countries.keys())
        self.imports = imports
        self.exports = exports
        self._import_propensities = import_propensities
        self.country_ids = _create_ids(self.country_names)
        self.sector_ids = _create_ids(self.sectors)
        self._country_sector_ids = _create_ids(
            self.country_names, sectors)
        self._country_country_sector_ids = _create_ids(
            self.country_names, self.country_names, sectors)
        self.deltas = pd.DataFrame()
        # Perform the first calculation of the model
        self.recalculate_world()

    @classmethod
    def from_pickle(cls, picklefilename, silent=None):
        model = pd.read_pickle(picklefilename)
        silent = model['silent'] if silent is None else silent
        return cls(countries=model['countries'],
                   sectors=model['sectors'],
                   imports=model['imports'],
                   exports=model['exports'],
                   import_propensities=model['import_propensities'],
                   calculate=model['calculate'],
                   silent=silent,
                   tolerance=model['tolerance'])

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
        @rtype: global_demo_model.GlobalDemoModel
        """
        sector_flows = sector_flows.\
            rename(columns={'country_iso3':'country'})
        commodity_flows = commodity_flows.\
            rename(columns={'country_iso3':'country'})    

        countries = country_setup.create_countries_from_data(sector_flows,
                                                             commodity_flows)
        country_names = pd.unique(sector_flows['country']).tolist()
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

        if self._calculate:
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
                    _put_exports_into_countries(countries, E)
                    if not self._silent:
                        return "World recalculated after %i iterations." % i
                    else:
                        return None
            if not self._silent:
                return "Warning: World didn't converge " \
                      "after %i iterations." % __MAX_ITERATIONS__
            else:
                return None
        else:
            return None

    def sector_trade_flows(self, sector, imports=None):
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
        x = [self.countries[c].x for c in self._countries_not_RoW()]
        return pd.concat(x, keys=self._countries_not_RoW(), 
                         names=['country'])

    def technical_coefficients(self):
        """
        Technical coefficients for all countries except RoW
        """
        A = [self.countries[c].A.unstack().reorder_levels([1, 0]) \
                for c in self._countries_not_RoW()]
        return pd.concat(A, keys=self._countries_not_RoW(), 
                         names=['country'])

    def import_ratios(self):
        """
        get import ratios from each country in the model

        
        Parameters
        ----------
        as_matrix : boolean
            Return diagonalised D matrices, or simple d vectors?

        Returns
        -------
        pd.Series
        """
        d = [self.countries[c].d \
                for c in self._countries_not_RoW()]
        return pd.concat(d, keys=self._countries_not_RoW(), 
                         names=['country'])

    def _countries_not_RoW(self, names=None):
        if names is None:
            names = self.country_names
        return sorted([c for c in names \
            if c != 'RoW'])
    
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
            return self.recalculate_world()

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
                return self.recalculate_world()

    def set_import_ratio(self, country, sector, value):
        print "not implemented yet" # TODO:

    def set_import_propensity(self, sector, from_country, to_country, value):
        """
        Set the import propensity and rescale to ensure propensities from
        all countries sum to unity

        Parameters
        ----------
        sector : string
            Which sector's import propensities are being adjusted?
        from_country : string

        to_country : string

        value : float

        """
        to_country = self.get_country(to_country).name
        from_country = self.get_country(from_country).name
        sector = self.get_sector(sector)

        p_s = self._import_propensities.ix[sector]
        p_s.ix[from_country, to_country] = value
        # TODO: Rationalise to another function?
        by_to_country = p_s.swaplevel(0,1).sortlevel()
        rescaled = by_to_country[to_country] / sum(by_to_country[to_country]) 
        by_to_country[to_country].update(rescaled)
        p_s.update(by_to_country.swaplevel(0, 1).sortlevel())
        return self.recalculate_world()

    def final_demand(self):
        """
        A `pandas.Series` of final demands

        The Series is indexed on country_name/sector

        Returns:
        --------
         : pandas.Series
        """
        df = pd.DataFrame()
        for name in self._countries_not_RoW():
            country = self.countries[name]
            fd = country.f.reset_index(name='final_demand')
            fd['country'] = name
            df = pd.concat([df, fd], 0)
        df = df.set_index(['country','sector']).squeeze()
        df.name = 'final_demand'
        return df

    def import_propensities(self):
        return self._import_propensities

    def to_file(self, filename):
        """
        Create a pickle of the current demo model
        """
        model = {'countries': self.countries,
                 'sectors': self.sectors,
                 'imports': self.imports,
                 'exports': self.exports,
                 'import_propensities': self._import_propensities,
                 'silent': self._silent,
                 'calculate': self._calculate,
                 'tolerance': self.tolerance}
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
        c = len(self.country_names)
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
            P_s = self.import_propensities().ix[sector].unstack()
            P_s = P_s.ix[self.country_names, self.country_names] # Exclude RoW
            start = i * c
            end = start + c
            P[start:end, start:end] = P_s
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

    def set_from_json(self, json):
        """
        Set model parameters using a json file
        """
        self._calculate = False # Suspend automatic recalculation of the model
        for k, v in json.iteritems():
            value = v['value']
            if k == 'technicalCoefficient':
                to_sector = v['to_sector']
                from_sector = v['from_sector']
                country = v['country']
                self.set_technical_coefficient(country=country,
                    from_sector=from_sector, to_sector=to_sector, value=value)
            elif k == 'importRatio':
                print "not implemented yet" # TODO:
            elif k == 'finalDemand':
                country = v['country']
                sector = v['sector']
                self.set_final_demand(country=country, sector=sector,
                                      value=value)
            elif k == 'importPropensity':
                print "not implemented yet" # TODO: Implement me!
        self._calculate = True
        self.recalculate_world()

    def flows_to_json(self, flows):
        """
        Convert the given `pd.Series` of flows.
        
        Also rounds the flow's value to nearest integer.
        """
        value_column = flows.name if flows.name is not None else 0
        flows = flows.round().reset_index()
        output = {}
        countries = set()
        sectors = set()
        country_ids = _id_list_to_dictionaries(self.country_ids)
        sector_ids = _id_list_to_dictionaries(self.sector_ids)
        output['flows'] = []
        for i, (k, flow) in enumerate(flows.iterrows()):
            d = {}
            d['from'] = self.country_ids[flow['from_country']]
            countries.add(d['from'])
            d['to'] = self.country_ids[flow['to_country']]
            countries.add(d['to'])
            try:
                d['sector'] = self.sector_ids[flow['sector']]
                sectors.add(d['sector'])
            except KeyError:
                d['from_sector'] = self.sector_ids[flow['from_sector']]
                sectors.add(d['from_sector'])                
                d['to_sector'] = self.sector_ids[flow['to_sector']]
                sectors.add(d['to_sector'])                
            d['value'] = flow[value_column]
            d['id'] = i
            output['flows'].append(d)
        output['countries'] = [c for c in country_ids if c['id'] in countries]
        output['sectors']  = [s for s in sector_ids if s['id'] in sectors]
        return json.dumps(output)
        
    def _flow_to_ids(self, flow):
        """
        Convert from_country/to_country/sector into 
        _country_country_sector_ids and from_country and 
        to_country into self.country_ids
        """
        from_country = flow['from_country']
        to_country = flow['to_country']
        sector = flow['sector']
        ccs = (from_country, to_country, sector)
 
        index_from = self.country_ids[from_country]
        index_to = self.country_ids[to_country]
        flow_id = self._country_country_sector_ids[ccs]
        return {'flow_id':flow_id, 
                'index_from':index_from, 'index_to':index_to} 
        
    def all_flows(self, country_names=None, sectors=None):
        trade_flows = self._cs_cs_flows(country_names, sectors)
        domestic_flows = self._domestic_flows_to_foreign_format(
            self._io_flows(country_names, sectors))
        trade_flows.update(domestic_flows)
        fd_flows = self._flows_to_final_demand(country_names, sectors)
        return pd.concat([trade_flows, fd_flows]).sortlevel()

    def _flows_to_final_demand(self, country_names=None, sectors=None,
            with_RoW=False):
        """
        Calculate country-sector flows to country-sector
        final demand

        Uses self.final_demand(), self.import_ratios() and
        self.import_propensities()
        """
        fd_name = 'fd'
        if sectors is None:
            sectors = list(self.sectors) # Copy to avoid weirdness later
        sectors = _append_or_set(sectors, 'fd') # Needed for filtering later
        fd = self.final_demand()
        # Split into foreign and domestic flows using import ratios
        foreign = self._split_flows_by_import_ratios(fd, get_domestic=False)
        domestic = self._split_flows_by_import_ratios(fd, get_domestic=True)
        # Split the foreign flows by from_country using import propensities
        fd_flows = \
            self._split_flows_by_import_propensities(
                foreign, 'country', with_RoW=with_RoW)
        domestic_flows = self._domestic_flows_to_foreign_format(
            domestic)
        # Combine the two into a single Series
        fd_flows.update(domestic_flows.reorder_levels([2,1,0]))
        fd_flows = fd_flows.reset_index('sector').\
            rename(columns={'sector':'from_sector'})
        fd_flows['to_sector'] = fd_name
        fd_flows = fd_flows.set_index(['from_sector','to_sector'], 
                                      append=True).squeeze().sortlevel()
        # Now return only those flows asked for
        fd_flows = self._filter_flows(fd_flows, country_names, 
                                      sectors, with_RoW=with_RoW)
        return fd_flows

    def _domestic_flows_to_foreign_format(self, flows):
        """
        Turn domestic flows into foreign (from_country/to_country) flows
        
        Take sector/sector flows indexed on 'country'
        and return flows indexed on 'from_country' and 'to_country' where
        the two index levels are the same.
        """
        new_index = [x.replace('country', 'to_country') 
            for x in flows.index.names]
        new_index = ['from_country'] + new_index
        domestic_flows = flows.reset_index() \
            .rename(columns={'country':'to_country'})
        domestic_flows['from_country'] = domestic_flows['to_country']
        return domestic_flows.set_index(new_index).squeeze()
        

    def _cs_cs_flows(self, country_names=None, sectors=None):
        """
        Calculate country-sector to country-sector flows using
        country-country flows, the import ratios and the technical
        coefficients
        """
        cc_flows = self.trade_flows(country_names, sectors).reset_index()
        # TODO: This section is a real mess, due to the fact that pandas
        # 0.13.0 can not join two MultiIndexed series.
        import_ratios = self.import_ratios().reset_index()
        import_ratios = import_ratios.rename(
            columns={'country':'to_country', 0: 'd'})
        # TODO: This is wrong at the moment. Need to separate
        # off the part which satisfies final demand before doing this!
        tech_coeffs = self.technical_coefficients().reset_index()
        tech_coeffs = tech_coeffs.rename(
            columns={'country':'to_country', 0: 'a'})
        
        exports = self.exports.reset_index()
        exports = exports.rename(
            columns={'from_country':'to_country', 0: 'e'})
            
        x = pd.merge(cc_flows, import_ratios, on=['to_country', 'sector'])
        x = pd.merge(x, exports, on=['to_country', 'sector'])
        x = x.rename(columns={'sector':'from_sector'})
        x = pd.merge(x, tech_coeffs, on=['to_country', 'from_sector'])
        x = x.set_index(['from_country', 'to_country', 
                         'from_sector', 'to_sector']).sortlevel()
         
        x['flow_value'] = ((1 / (1 - x.d)) * x.cc_flow_value + x.e ) * x.a
        return x['flow_value'].squeeze().sortlevel()
            
    def _io_flows(self, country_names=None, sectors=None):
        if country_names is None:
            country_names = self._countries_not_RoW()
        d = [self._filter_flows(self.countries[c].Z_dagger().stack(), 
                                sectors=sectors) \
             for c in country_names]
        return pd.concat(d, keys=self._countries_not_RoW(), 
                         names=['country'])
        
    def trade_flows(self, country_names=None, sectors=None, with_RoW=False):
        """
        Per-sector country-country flows.
        
        Calculated from the global import vector and the 
        import propensities
        """
        imports = self._filter_flows(
            self.imports, country_names, sectors, with_RoW=with_RoW)
        return self._split_flows_by_import_propensities(
            imports, country_field='to_country', 
            country_names=country_names, with_RoW=with_RoW)

    def _filter_flows(self, flows, country_names=None, sectors=None,
                      with_RoW=False):
        """
        Filter a MultiIndexed pandas object by `country_names` and
        `sectors`.
        
        Both `country_names` and `sectors` can be None, a string or a
        list of strings.
        """
        if country_names is None and sectors is None and with_RoW:
            return flows
        else:
            # What should we filter the pandas object on? (index or columns?)
            if flows.index.names[0] is None:
                # Dataframe/Series is unindexed
                names = flows.columns.values.tolist()
            else:
                names = flows.index.names
            # Prepare the filtered pandas object
            filtered = flows.copy()
            if country_names is None:
                country_names = self.country_names
            # Remove RoW
            if with_RoW:
                country_names = country_names + ['RoW']
            if sectors is None:
                sectors = self.sectors
            # Level names on which to filter
            sector_levels = [l for l in names 
                if str(l).find('sector') >= 0]    
            country_levels = [l for l in names 
                if str(l).find('country') >= 0]
            
            for level in sector_levels:
                filtered = _filter_series(filtered, level, sectors)
            for level in country_levels:
                filtered = _filter_series(filtered, level, country_names)
                
            return filtered

    def _split_flows_by_import_propensities(self, flows,
                                            country_field='country',
                                            country_names=None,
                                            with_RoW=False):
        """
        Calculate country to country flows using
        the given flows vector and the import propensities
        """
        # TODO: This section is a real mess, due to the fact that pandas
        # 0.13.0 can not join two MultiIndexed series.
        r_name = '0_y' if flows.name is None else flows.name
        l_name = '0_x' if flows.name is None else 0
        result = self._filter_flows(self.import_propensities(), country_names,
                                    with_RoW=with_RoW)
        result = result.reset_index()
        x = pd.merge(result, 
            flows.reset_index().rename(columns={country_field:'to_country'}), 
            on=['sector', 'to_country'])
        x['cc_flow_value'] = x[l_name] * x[r_name]
        return x.set_index(['sector', 'from_country', 
                            'to_country'])['cc_flow_value'].sortlevel()
        
    def _split_flows_by_import_ratios(self, flows, get_domestic,
                                      country_names=None):
        """
        Use self.import_ratios() to get flows either domestically
        or foreign, depending on the value of `get_domestic`
        """
        # TODO: This section is a real mess, due to the fact that pandas
        # 0.13.0 can not join two MultiIndexed series.
        d = self._filter_flows(self.import_ratios(), country_names)
        d = 1 - d if get_domestic else d
        return flows.mul(d, fill_value=0)
        
    def get_id(self, country, sector):
        the_id = [ k for k, v in self._country_sector_ids.iteritems()
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
        if lookup_id in self._country_sector_ids:
            return self._country_sector_ids[lookup_id]
        else:
            return {"country":None, "sector":None}
                    
    def get_country_by_id(self, lookup_id):
        country_name = keys_to_items(self.country_ids)[int(lookup_id)]
        return self.countries[country_name]

    def get_sector_by_id(self, lookup_id):
        return keys_to_items(self.sector_ids)[lookup_id]

    def get_country(self, country):
        """
        get a Country object based on `country`

        If `country` is a Country, return `country`.
        If `country` is a string, assume it's the name
        of a country
        If `country` is a number, assume it's a country id
        """
        try:
            # Is country a 'Country'?
            if country.name == country.name:
                return country
        except AttributeError:
            try:
                # is country a country name?
                return self.countries[country]
            except KeyError:
                try:
                    return self.get_country_by_id(country)
                except KeyError:
                    return None
        return None

    def get_sector(self, sector):
        """
        Lookup by id if `sector` is a number, otherwise return `sector`
        """
        try:
            return self.get_sector_by_id(sector)
        except KeyError:
            return sector
            
    @classmethod
    def name_from_iso3(cls, iso3):
        names = {
            'AUS': 'Australia',
            'AUT': 'Austria',
            'BEL': 'Belgium',
            'BGR': 'Bulgaria',
            'BRA': 'Brazil',
            'CAN': 'Canada',
            'CHN': 'China',
            'CYP': 'Cyprus',
            'CZE': 'Czech Republic',
            'DEU': 'Germany',
            'DNK': 'Denmark',
            'ESP': 'Spain',
            'EST': 'Estonia',
            'FIN': 'Finland',
            'FRA': 'France',
            'GBR': 'UK',
            'GRC': 'Greece',
            'HUN': 'Hungary',
            'IDN': 'Indonesia',
            'IND': 'India',
            'IRL': 'Ireland',
            'ITA': 'Italy',
            'JPN': 'Japan',
            'KOR': 'S. Korea',
            'LTU': 'Lithuania',
            'LUX': 'Luxembourg',
            'LVA': 'Latvia',
            'MEX': 'Mexico',
            'MLT': 'Malta',
            'NLD': 'Netherlands',
            'POL': 'Poland',
            'PRT': 'Portugal',
            'ROU': 'Roumania',
            'RUS': 'Russia',
            'SVK': 'Slovakia',
            'SVN': 'Slovenia',
            'SWE': 'Sweden',
            'TUR': 'Turkey',
            'TWN': 'Taiwan',
            'USA': 'USA'       
        }
        try:
            return names[iso3]
        except KeyError:
            return iso3

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
    E.index = E.index.rename(['sector', 'from_country'])
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
    sectors = sorted(P.index.levels[0])

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
    e_s = P_s.unstack().dot(i_s)
    e_s.index.names = ['from_country']
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
    all_M = pd.concat(all_M, keys=cnames, names=['to_country'])
    all_M = all_M.swaplevel(0,1).sortlevel()

    return all_M

def _put_exports_into_countries(countries, exports):
    """
    Put the values from the global vector E into the
    country objects' e vectors
    """
    country_sector_exports = exports.swaplevel(0, 1).sortlevel()
    g = country_sector_exports.groupby(level='from_country')

    for country_name in g.groups.iterkeys():
        countries[country_name].e = country_sector_exports.ix[country_name]

def _relevant_flows(trade_flows, countries):
    """
    Get rid of flows we're not interested in
    
    Keep only flows to and from countries in 'countries'.
    Set all other flows either to or from RoW. Discard
    all flows both from and to countries not in 'countries'.
    known_to_unknown and unknown_to_known are kept for the creation
    of the RoW country later.
    Set all negative flows to zero.
    """
    trade_flows['trade_value'][trade_flows['trade_value'] < 0] = 0    
    
    fields = ['from_iso3', 'to_iso3', 'sector', 'trade_value']
    data = trade_flows[~pd.isnull(trade_flows.sector)]

    # Drop flows to-from same country
    data = data[~(data.from_iso3==data.to_iso3)]

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

def _create_ids(*args):
    sorted_args = []
    for arg in args:
        sorted_args.append(sorted(arg))
    if len(sorted_args) > 1:
        return {v:k for k, v in enumerate(product(*sorted_args))}
    else:
        return {v:k for k, v in enumerate(sorted_args[0])}
    
def _filter_series(x, filter_on, filter_by):
    """
    Filter a pd.Generic x by `filter_by` on the 
    MultiIndex level or column `filter_on`.
    
    Uses `pd.Index.get_level_values()` in the background
    """
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        if type(filter_by) is str:
            filter_by = [filter_by]
        try:
            index = x.index.get_level_values(filter_on).isin(filter_by)
        except:
            index = x[filter_on].isin(filter_by)
        return x[index]
    else:
        print "Not a pandas object"
    
def _append_or_set(x, to_append):
     try:
         x.append(to_append)
     except AttributeError:
         if x:
             x = [x, to_append]
         else:
             x = to_append
     finally:
         return x

def keys_to_items(unique_dictionary):
    """
    Swaps the keys and the items in a dictionary
    
    Only works if both unique_dictionary.items and unique_dictionary.keys
    are unique
    """
    try: 
        return {int(v):k for k, v in unique_dictionary.iteritems()}
    except ValueError:
        return {v:k for k, v in unique_dictionary.iteritems()}
        
def _id_list_to_dictionaries(id_list):
    return [{'name':k, 'id':v} for k, v in id_list.iteritems()]
