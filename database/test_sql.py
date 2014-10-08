# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 15:48:26 2014

@author: rob
"""

import unittest
import pandas as pd
import sql

class DbWriting(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'a':[1,2,3],
            'b':[4,5,6],
            'c':['d','e','f'],
            'd':['g','h','i']
            })
        self.index_names = {'a':'hello', 'b':'goodbye'}
        self.db_names = {'c':'db_col1', 'd':'db_col2'}
        print self.df
    
    def test_build_where(self):
        for k, row in self.df.iterrows():
            where = sql.build_where(row, self.index_names)
            print where
        
    def test_build_set(self):
        for k, row in self.df.iterrows():
            set_ = sql.build_set(row, self.db_names)
            print set_

    def test_build_update(self):
        print sql.build_update(self.df, 'doobry', 
            indexcols=self.index_names,
            dbcols=self.db_names)

if __name__ == '__main__':
    unittest.main(exit=False)