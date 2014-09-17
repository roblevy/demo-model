# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:15:29 2013

@author: Rob
"""
import numpy as np
import pandas as pd
import global_demo_model
import output_model
import unittest

reload(global_demo_model)
reload(output_model)

__REAL__ = True

class DemoModelInternals(unittest.TestCase):
    def setUp(self):
        # Get data (either dummy or 'real')
        if __REAL__:
            # REAL
            # ----
            self.tolerance = 1e-1
            self.test_country_name = 'GBR'
            self.second_test_country_name = 'USA'
            self.test_sector = 'Wood'
            self.second_test_sector = 'Business Services'
#            sector_flows = pd.read_csv('../Data/40 Countries/2008/sector_flows.csv',true_values='t',false_values='f')
#            trade_flows = pd.read_csv('../Data/200 Countries/2009/fn_trade_flows 2009.csv',true_values='t',false_values='f')
#            services_flows = pd.read_csv('../Data/200 Countries/2010/balanced_services_2010.csv')
#            self.model = global_demo_model. \
#                GlobalDemoModel.from_data(sector_flows,
#                                          trade_flows,
#                                          services_flows, 
#                                          silent=True, tolerance=self.tolerance)
            self.model = global_demo_model. \
                GlobalDemoModel.from_pickle('model.gdm', silent=True)
        else:
            # DUMMY
            # -----
            self.tolerance = 1e-8
            self.test_country_name = 'A'
            self.second_test_country_name = 'B'
            self.test_sector = 'T'
            self.second_test_sector = 'R'
            io_data = pd.read_csv('Dummy Data/dummy_io_flows.csv',
                                  true_values='t',false_values='f')
            goods_flows = pd.read_csv('Dummy Data/dummy_trade_flows.csv',
                                      true_values='t',false_values='f')
                                    
            # Create model
            self.model = global_demo_model. \
                GlobalDemoModel.from_data(sector_flows=io_data, 
                                          commodity_flows=goods_flows,
                                          silent=True, tolerance=self.tolerance)

        # Setup other useful details:
        self.test_country_1 = self.model.countries[self.test_country_name]
        self.test_country_2 = self.model. \
            countries[self.second_test_country_name]
        self.B = self.model.countries[self.second_test_country_name]
        self.f = self.test_country_1.f.copy()
        self.e = self.test_country_1.e.copy()
        self.m = self.test_country_1.m.copy()
        self.x = self.test_country_1.x.copy()
        
    def test_m_and_x_same_if_f_and_e_are_same(self):        
        # Make sure that feeding existing f and e into
        # recalculate_economy does nothing
        A = self.test_country_1
        f = self.f
        e = self.e
        m = self.m
        x = self.x
        
        A.recalculate_economy(final_demand=f,
                              exports=e) # nothing should happen
        # Check m doesn't change
        self.assertTrue(np.allclose(m, A.m, rtol=self.tolerance))
        # Check x doesn't change
        self.assertTrue(np.allclose(x, A.x, rtol=self.tolerance))

    def test_recalculate_world_does_nothing_on_its_own(self):
        E = self.model.exports.copy()
        M = self.model.imports.copy()        
        self.model.recalculate_world()
        # Exports should be unchanged
        self.assertTrue(np.allclose(E, self.model.exports))
        # Exports should be unchanged
        self.assertTrue(np.allclose(M, self.model.imports))

    def test_m_and_x_increase_when_f_increases(self):
        A = self.test_country_1
        m = self.m
        x = self.x

        f_R = A.f[self.test_sector]
        
        # Both m and x should change when final demand changes
        self.model.set_final_demand(A, self.test_sector, f_R + 1000)
        # Check m increases
        self.assertTrue(np.alltrue(np.greater(A.m, m)))
        # Check x increases
        self.assertTrue(np.alltrue(np.greater(A.x, x)))
        
    def test_country_trade_values_equal_global_trade_values(self):
        model = self.model        
        A = self.test_country_1
        m = self.m
        e = self.e
        s = self.test_sector
        
        # The value of m stored by the country should be the same
        # as the value stored by the model
        self.assertTrue(m[s] == model.imports.ix[s, A.name])
        # The value of e stored by the country should be the same
        # as the value stored by the model
        self.assertTrue(e[s] == model.exports.ix[s, A.name])
        
        # The above tests should still pass after something changes
        model.set_final_demand(A, s, A.f[s] * 2)
        self.assertTrue(A.m[s] == model.imports.ix[s, A.name])
        self.assertTrue(A.e[s] == model.exports.ix[s, A.name])
        
        
    def test_global_m_and_x_increase_when_f_increases(self):
        A = self.test_country_1
        B = self.B
        m = B.m.copy()
        x = B.x.copy()

        fA_R = A.f[self.test_sector]
        
        # Both m and x should change when final demand changes
        self.model.set_final_demand(A, self.test_sector, fA_R * 2)
        # Check m increases
        self.assertTrue(np.alltrue(np.greater(B.m, m)))
        # Check x increases
        self.assertTrue(np.alltrue(np.greater(B.x, x)))
    
    def test_global_m_and_x_decrease_when_f_decreases(self):
        A = self.test_country_1
        B = self.B
        m = B.m.copy()
        x = B.x.copy()

        fA_R = A.f[self.test_sector]
        
        # Both m and x should change when final demand changes
        self.model.set_final_demand(A, self.test_sector, fA_R / 2)
        # Check m increases
        self.assertTrue(np.alltrue(np.less(B.m, m)))
        # Check x increases
        self.assertTrue(np.alltrue(np.less(B.x, x)))            
    
    def test_supply_equals_demand(self):
        # When f and e are changed, check x + m = Ax + f + e
        A = self.test_country_1
        self.model.set_final_demand(A, self.test_sector, 
                                    A.f[self.test_sector] * 2)
        supply = A.x + A.m
        demand = np.dot(A.A, A.x) + A.f + A.e
        self.assertTrue(np.allclose(supply, demand))

    def test_imports_equal_exports_after_recalculating_world(self):
        model = self.model
        model.recalculate_world()
        self.assertTrue(np.allclose(model.exports.sum(level=0), 
                                    model.imports.sum(level=0)))

    def test_set_final_demand(self):
        model = self.model
        A = self.test_country_1
        model.set_final_demand(A, self.test_sector, 1234)
        self.assertTrue(A.f[self.test_sector] == 1234)

    def test_set_final_demand_changes_imports(self):
        model = self.model
        A = self.test_country_1
        M = model.imports.copy()
        model.set_final_demand(A, self.test_sector, 1234)
        self.assertTrue((model.imports != M).any)

    def test_set_tech_coeff_successfully(self):
        model = self.model
        country = self.test_country_1
        r = self.test_sector
        s = self.second_test_sector
        A = country.A
        a = A.ix[s, r]
        new_a = a / 2 if a > 0 else 0.001234
        model.set_technical_coefficient(country, s, r, new_a)
        self.assertTrue(country.A.ix[s, r] == new_a)

    def test_set_tech_coeff_too_much(self):
        model = self.model
        country = self.test_country_1
        r = self.test_sector
        s = self.second_test_sector
        A = country.A
        a = A.ix[s, r]
        model.set_technical_coefficient(country, s, r, 1.1)
        self.assertTrue(country.A.ix[s, r] == a)

    def test_set_tech_coeff_changes_import(self):
        model = self.model
        country = self.test_country_1
        r = self.test_sector
        s = self.second_test_sector
        A = country.A
        m = country.m.copy()
        a = A.ix[s, r]
        new_a = a / 2 if a > 0 else 0.001234
        model.set_technical_coefficient(country, s, r, new_a)
        self.assertTrue((country.m != m).any)
      
    def test_flows_to_fd_equals_fd(self):
        model = self.model
        fd = model.final_demand()
        flows_to_fd = model._flows_to_final_demand(with_RoW=True)
        flows_to_fd = flows_to_fd.sum(level=['from_sector', 'to_country'])
        flows_to_fd = flows_to_fd.swaplevel(0,1).sortlevel()
        self.assertTrue(np.allclose(fd, flows_to_fd))
        
    def test_trade_flows_sum_to_imports_and_exports(self):
        model = self.model
        imports = model.imports
        exports = model.exports
        trade_flows = model.trade_flows(with_RoW=True)
        import_trade_flows = trade_flows.sum(level=['sector', 'to_country'])
        export_trade_flows = trade_flows.sum(level=['sector', 'from_country'])
        idiff = import_trade_flows.sub(imports, fill_value=0)
        ediff = export_trade_flows.sub(exports, fill_value=0)
        # Trade flows summed over 'from_country' should equal imports        
        self.assertTrue(np.allclose(idiff, 0))
        # Trade flows summed over 'to_country' should equal exports
        self.assertTrue(np.allclose(ediff, 0))
        
    def test_all_parameters_positive(self):
        model = self.model
        p = model.import_propensities()
        self.assertTrue(len(p[p < 0]) == 0)
        
    def test_trade_flows_all_positive(self):
        model = self.model
        trade_flows = model.trade_flows()
        self.assertTrue(len(trade_flows[trade_flows < 0]) == 0)
    
if __name__ == '__main__':
    unittest.main(exit=False)

