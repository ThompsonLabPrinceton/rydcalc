#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:24:57 2020

@author: jt11
"""

import sqlite3
import numpy as np

class db_manager():
    
    def __init__(self,filename):
        """ Initialize database manager with database file = filename. Does not have to already exist """
        self.conn = sqlite3.connect(filename)
        self.c = self.conn.cursor()
        
        self.tables = {}
        
    def add_table(self,table_name,keys,results,npy_file=None):
        """ Add a table to the manager, and returns db_table object.
        
        If table does not exist in database, creates it, and loads from npy_file if it can.
        If it does exist, checks that it has the same keys/results as the requested table, otherwise returns None """
        
        self.tables[table_name] = db_table(table_name,keys,results,self,npy_file)
        
        return self.tables[table_name]
        

class db_table():
    
    table_name = 'dbman_table'
    
    def __init__(self,table_name,keys,results,mgr,npy_file=None):
        
        # self.conn = sqlite3.connect(filename)
        # self.c = self.conn.cursor()
        
        self.conn = mgr.conn
        self.c = mgr.c
        
        self.table_name = table_name
        self.npy_file = npy_file
        
        self.keys = keys
        
        if len(results) > 1:
            print("Error--more than one result variable not currently supported")
            return None
        
        self.results = results
        
        #print(self._create_str())
        
        # check to see if our table exists
        self.c.execute("""SELECT COUNT(*) FROM sqlite_master where type='table' AND name='%s'""" % self.table_name)
        
        if (self.c.fetchone()[0] == 0):
            # create it
            self.c.execute(self._create_str())
            
            self.load()
                
        else:
            # check that it is the right table by comparing the create string stored in sqlite to the one we would have created
            # this seems inelegant but not sure how else to do it
            self.c.execute("""SELECT sql FROM sqlite_master WHERE tbl_name = '%s' and type='table'""" % self.table_name)
            prev_str = self.c.fetchone()[0]
            
            if prev_str != self._create_str():
                print("Error in db_table() -- wanted create_str %s, got %s" % (self._create_str(),prev_str))
                return None
        
        self.conn.commit()
            
    def _create_str(self):
        # string for creating a table -- for now all of the keys are tinyint unsigned and results are doubles, and
        # all keys are part of the primary key
        
        keys_typed = [x + ' TINYINT UNSIGNED' for x in self.keys]
        results_typed = [x + ' DOUBLE' for x in self.results]

        create_str = "CREATE TABLE %s (" % self.table_name

        return create_str + ','.join(keys_typed + results_typed + ['PRIMARY KEY (%s)' % ','.join(self.keys)]) + ')'
    
    def load(self,altfile = None):
        """ Load values from npy file into database. This will give an error and undefined behavior
        if there are any key collisions, I think. Not obivously safe to run on existing, populated database. """
        
        file = altfile if altfile is not None else self.npy_file
        
        try:
            data = np.load(file)
            self.insert(data,many=True)
            print("Restored database table %s from %s (%d records)" % (self.table_name,file,self.db_size()))
        except:
            print("Error reloading database table %s from %s" % (self.table_name,file))
            
    def save(self,altfile = None):
        """ Save values from table to file, for later loading with load(). """
        
        file = altfile if altfile is not None else self.npy_file
        
        data = self.getall()
        np.save(file,data)
        
        print("Saved database table %s to %s (%d records)" % (self.table_name,file,len(data)))
        
    
    def insert(self,val,many=False):
        """ Add new items to table. Val should be tuple/array of length M = len(keys) + len(results).
        If many=True, Val can be NxM array to load N entries."""
        
        st = 'insert into %s values %s' % (self.table_name,'(' + ','.join(['?' for x in self.keys+self.results]) + ')')
        #print(st)
        
        if many:
            self.c.executemany(st, np.array(val,dtype=float))
        else:
            self.c.execute(st, np.array(val,dtype=float))
        
        self.conn.commit()

    def query(self,val):
        """ Find value in table. Val should be tuple/array of length len(keys) """
        
        st = 'select %s from %s where %s' % (self.results[0],self.table_name,' AND '.join([x + '=?' for x in self.keys]))
        #print(st)
        self.c.execute(st, np.array(val,dtype=float))
        
        ret = self.c.fetchone()
        
        if (ret):
            return ret[0]
        else:
            return None
        
    def getall(self):
        """ Get NxM array of all key,results in table """
        
        st = 'select %s from %s' % (','.join(self.keys+self.results),self.table_name)
        
        self.c.execute(st)
        
        return self.c.fetchall()
    
    def db_size(self):
        """ Get number of entries in table """
        
        self.c.execute('select count(*) from %s' % self.table_name)
        return self.c.fetchone()[0]