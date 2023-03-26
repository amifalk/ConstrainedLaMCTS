# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import json
import collections
import copy as cp
import math
from collections import OrderedDict
import os.path
import numpy as np
import pandas as pd
import time
import operator
import sys
import pickle
import os
import random
from datetime import datetime
from .Node import Node
from .utils import latin_hypercube, from_unit_cube
from torch.quasirandom import SobolEngine
import torch

from diversipy import polytope
import dill

class MCTS:
    #############################################

    def __init__(self, 
        lb, ub, 
        A_eq, b_eq, A_ineq, b_ineq, 
        ninits, func, dims, 
        num_threads, sim_workers, threads_per_sim,
        Cp = 1, leaf_size = 20, kernel_type = "rbf", gamma_type = "auto", solver_type = 'bo',
        turbo_max_samples=10000, turbo_batch=1, hopsy_thin=150):
        self.dims                    =  dims
        self.samples                 =  []
        self.nodes                   =  []
        self.Cp                      =  Cp
        self.lb                      =  lb
        self.ub                      =  ub
        # incorporate polytope constraints
        # A_eq, b_eq describe constraints of form A_eq @ x = b_eq
        # A_ineq, b_ineq describe constraints of form A_ineq @ x = b_ineq
        self.A_eq                    = A_eq
        self.b_eq                    = b_eq
        self.A_ineq                  = A_ineq
        self.b_ineq                  = b_ineq
        # -----------------------------------
        
        self.num_threads             =  num_threads      # number of total cores available
        self.sim_workers             =  sim_workers      # number of simulations to run simultaneously
        self.threads_per_sim         =  threads_per_sim  # number of cores/threads to use per simulation
        
        self.ninits                  =  ninits
        self.func                    =  func
        self.curt_best_value         =  float("-inf")
        self.curt_best_sample        =  None
        self.best_value_trace        =  []
        self.sample_counter          =  0
        self.visualization           =  False
        
        self.LEAF_SAMPLE_SIZE        =  leaf_size
        self.kernel_type             =  kernel_type
        self.gamma_type              =  gamma_type
        
        self.solver_type             =  solver_type #solver can be 'bo' or 'turbo'
        self.turbo_max_samples       =  turbo_max_samples
        self.turbo_batch             =  turbo_batch
        self.hopsy_thin              =  hopsy_thin 
        
        print("gamma_type:", gamma_type)
        
        #we start the most basic form of the tree, 3 nodes and height = 1
        root = Node( parent = None, dims = self.dims, reset_id = True, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
        self.nodes.append( root )
        
        # need to pass information about the global linear constraints
        # down to the Classifier object...
        # Represent as a dictionary of the matrices/vectors defining the linear constraints.
        self.GLOBAL_CONSTRAINT = {"A_eq": A_eq, "b_eq": b_eq, "A_ineq": A_ineq, "b_ineq": b_ineq}

        self.ROOT = root
        self.CURT = self.ROOT
        self.init_train()
        
    def populate_training_data(self):
        #only keep root
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()
        new_root  = Node(parent = None,   dims = self.dims, reset_id = True, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
        self.nodes.append( new_root )
        
        self.ROOT = new_root
        self.CURT = self.ROOT
        self.ROOT.update_bag( self.samples )
    
    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            if node.is_leaf() == True and len(node.bag) > self.LEAF_SAMPLE_SIZE and node.is_svm_splittable == True:
                status.append( True  )
            else:
                status.append( False )
        return np.array( status )
        
    def get_split_idx(self):
        split_by_samples = np.argwhere( self.get_leaf_status() == True ).reshape(-1)
        return split_by_samples
    
    def is_splitable(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False
        
    def dynamic_treeify(self):
        # we bifurcate a node once it contains over 20 samples
        # the node will bifurcate into a good and a bad kid
        self.populate_training_data()
        assert len(self.ROOT.bag) == len(self.samples)
        assert len(self.nodes)    == 1
                
        while self.is_splitable():
            to_split = self.get_split_idx()
            #print("==>to split:", to_split, " total:", len(self.nodes) )
            for nidx in to_split:
                parent = self.nodes[nidx] # parent check if the boundary is splittable by svm
                assert len(parent.bag) >= self.LEAF_SAMPLE_SIZE
                assert parent.is_svm_splittable == True
                # print("spliting node:", parent.get_name(), len(parent.bag))
                good_kid_data, bad_kid_data = parent.train_and_split()
                #creat two kids, assign the data, and push into lists
                # children's lb and ub will be decided by its parent
                assert len(good_kid_data) + len(bad_kid_data) == len(parent.bag)
                assert len(good_kid_data) > 0
                assert len(bad_kid_data)  > 0
                good_kid = Node(parent = parent, dims = self.dims, reset_id = False, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
                bad_kid  = Node(parent = parent, dims = self.dims, reset_id = False, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
                good_kid.update_bag( good_kid_data )
                bad_kid.update_bag(  bad_kid_data  )
            
                parent.update_kids( good_kid = good_kid, bad_kid = bad_kid )
            
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)
                
            #print("continue split:", self.is_splitable())
        
        self.print_tree()
        
    def collect_samples(self, sample, value = None):
        #TODO: to perform some checks here
        if value == None:
            value = self.func(sample)*-1
            
        if value > self.curt_best_value:
            self.curt_best_value  = value
            self.curt_best_sample = sample 
            if type(value) == np.ndarray:
                value = value.flat[0]
            self.best_value_trace.append( [value, self.sample_counter] )
        self.sample_counter += 1
        self.samples.append( (sample, value) )
        return value
    
    def constraint_sample(self,n):
        # no inequality constraints, do have equality constraint
        if self.A_ineq is None and self.b_ineq is None and self.A_eq is not None and self.b_eq is not None:
            samples = polytope.sample(
                    n_points=n,
                    lower=self.lb,
                    upper=self.ub,
                    A2=self.A_eq,
                    b2=self.b_eq
            )
        if self.A_ineq is not None and self.b_ineq is not None and self.A_eq is None and self.b_eq is None:
            samples = polytope.sample(
                    n_points=n,
                    lower=self.lb,
                    upper=self.ub,
                    A1=self.A_ineq,
                    b1=self.b_ineq
            )
        else: # just assume for now everything provided
            samples = polytope.sample(
                    n_points=n,
                    lower=self.lb,
                    upper=self.ub,
                    A1=self.A_ineq,
                    b1=self.b_ineq,
                    A2=self.A_eq,
                    b2=self.b_eq
            )
        return samples

    def init_train(self):
        # here we use latin hyper space to generate init samples in the search space
        # use only if no polytope constraints provided
        #if (self.A_eq):
        #    init_points = latin_hypercube(self.ninits, self.dims)
        #    init_points = from_unit_cube(init_points, self.lb, self.ub)

        # use diversipy polytope sampler
        # TODO: can add hopsy later?
        init_points = self.constraint_sample(self.ninits)
        print(init_points)

        for point in init_points:
            self.collect_samples(point)
        
        print("="*10 + 'collect '+ str(len(self.samples) ) +' points for initializing MCTS'+"="*10)
        print("lb:", self.lb)
        print("ub:", self.ub)
        print("Cp:", self.Cp)
        print("inits:", self.ninits)
        print("dims:", self.dims)
        print("="*58)
        
    def print_tree(self):
        print('-'*100)
        for node in self.nodes:
            print(node)
        print('-'*100)

    def reset_to_root(self):
        self.CURT = self.ROOT
    
    def load_agent(self, load_path='mcts_agent'):
        with open(load_path, 'rb') as data:
            #self = dill.load(data)
            print("loading state data",flush=True)
            state = dill.load(data)
            self.dims = state["dims"]
            self.samples = state["samples"]
            self.nodes = state["nodes"]
            self.Cp = state["Cp"]
            self.lb = state["lb"]
            self.ub = state["ub"]
            self.A_eq = state["A_eq"]
            self.b_eq = state["b_eq"]
            self.A_ineq = state["A_ineq"]
            self.b_ineq = state["b_ineq"]
            self.num_threads = state["num_threads"]
            self.ninits = state["ninits"]
            self.curt_best_value = state["curt_best_value"]
            self.curt_best_sample = state["curt_best_sample"]
            self.best_value_trace = state["best_value_trace"]
            self.sample_counter = state["sample_counter"]
            self.visualization = state["visualization"]
            self.LEAF_SAMPLE_SIZE = state["LEAF_SAMPLE_SIZE"]
            self.kernel_type = state["kernel_type"]
            self.gamma_type = state["gamma_type"]
            self.solver_type = state["solver_type"]
            print("=====>loads:", len(self.samples)," samples",flush=True)
            print("WARNING: Be sure to restore the state of the function!! It has not been loaded.",flush=True)


    def dump_agent(self, out_dir=None, name='mcts_agent'):
        if out_dir is None:
            node_path = name
        elif not os.path.exists(out_dir):
            node_path = name
        else:
            node_path = out_dir + name
        print("dumping the agent.....")
        with open(node_path,"wb") as outfile:
            # dill.dump(self, outfile)
            # because the func object will contain a multiprocess.Pool,
            # we cannot pickle it. So we just have to pickle everything 
            # else and restore the state of the objective function manually.
            state = {
                "dims" : self.dims,
                "samples" : self.samples,
                "nodes" : self.nodes,
                "Cp" : self.Cp,
                "lb" : self.lb,
                "ub" : self.ub,
                "A_eq" : self.A_eq,
                "b_eq" : self.b_eq,
                "A_ineq" : self.A_ineq,
                "b_ineq" : self.b_ineq,
                "num_threads" : self.num_threads,
                "ninits" : self.ninits,
                "curt_best_value" : self.curt_best_value,
                "curt_best_sample" : self.curt_best_sample,
                "best_value_trace" : self.best_value_trace,
                "sample_counter" : self.sample_counter,
                "visualization" : self.visualization,
                "LEAF_SAMPLE_SIZE" : self.LEAF_SAMPLE_SIZE,
                "kernel_type" : self.kernel_type,
                "gamma_type" : self.gamma_type,
                "solver_type" : self.solver_type
            }
            dill.dump(state, outfile)
            
    def dump_samples(self, out_dir=None, name='mcts_agent'):
        sample_path = '{}_samples_'+str(self.sample_counter)
        if out_dir is None:
            path = sample_path.format(name)
        elif not os.path.exists(out_dir):
            path = sample_path.format(name)
        else:
            path = out_dir + sample_path.format(name)

        with open(path, "wb") as outfile:
            dill.dump(self.samples, outfile)

    def load_samples(self, X, fX, best_X=None, best_fX=None):
        """
        X : list of numpy vectors
        fX : list of scalars?
        best_X, best_fX : optional 
        """
        self.samples = list(zip(X, fX))
        self.sample_counter

    def load_samples_from_file(self, samples, best_trace):
        """
        samples : path to csv containing samples
        if n columns: 
            columns 1 - (n-1) should contain the X,
            column n should contain the function value
        best_trace: 
            path to csv containing best trace
        """
        samples_df = pd.read_csv(samples, header=None)
        best_trace_df = pd.read_csv(best_trace, header=None)
        ncol_samples = samples_df.shape[1]
        nrow_samples = samples_df.shape[0]
        X_samples = np.array(samples_df.iloc[:,0:ncol_samples-1])
        fX_samples = np.array(samples_df.iloc[:,ncol_samples-1])
        self.samples = list(zip(X_samples,fX_samples))

        best_value_trace = np.array(best_trace_df.iloc[:,ncol_samples-1])
        self.best_value_trace = list(best_value_trace)
        self.curt_best_value = best_value_trace[-1]
        self.curt_best_sample = np.array(best_trace_df.iloc[nrow_samples-1,0:ncol_samples-1])
        self.sample_counter = nrow_samples


    
    def dump_trace(self):
        trace_path = 'best_values_trace'
        final_results_str = json.dumps(self.best_value_trace)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')

    def greedy_select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        if self.visualization == True:
            curt_node.plot_samples_and_boundary(self.func)
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_xbar() )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            if curt_node.is_leaf() == False and self.visualization == True:
                curt_node.plot_samples_and_boundary(self.func)
            print("=>", curt_node.get_name(), end=' ' )
        print("")
        return curt_node, path

    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [self.GLOBAL_CONSTRAINT] # Always include the global constraint information!
        
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_uct(self.Cp) )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            print("=>", curt_node.get_name(), end=' ' )
        print("")
        return curt_node, path
    
    def backpropogate(self, leaf, acc):
        curt_node = leaf
        while curt_node is not None:
            assert curt_node.n > 0
            curt_node.x_bar = (curt_node.x_bar*curt_node.n + acc) / (curt_node.n + 1)
            curt_node.n    += 1
            curt_node       = curt_node.parent

    def search(self, iterations, max_samples=np.inf):
        n_samples = max_samples

        for idx in range(self.sample_counter, iterations):
            if self.sample_counter > n_samples:
                break
            print("")
            print("="*10)
            print("iteration:", idx)
            print("="*10)
            self.dynamic_treeify()
            leaf, path = self.select()
            for i in range(0, 1):
                if self.solver_type == 'bo':
                    samples = leaf.propose_samples_bo( 1, path, self.lb, self.ub, self.samples )
                elif self.solver_type == 'turbo':
                    samples, values = leaf.propose_samples_turbo( 
                        self.turbo_max_samples, path, self.func, self.dims, 
                        num_threads=self.num_threads, 
                        sim_workers=self.sim_workers,
                        threads_per_sim=self.threads_per_sim,
                        hopsy_thin=self.hopsy_thin, batch_size=self.turbo_batch
                    )
                    #samples, values = leaf.propose_samples_turbo( 1000, path, self.func )
                #elif self.solver_type == 'scbo':
                #    samples, values = leaf.propose_samples_scbo( 10000, path, self.func , self.ROOT)
                #    #samples, values = leaf.propose_samples_scbo( 1000, path, self.func )
                else:
                    raise Exception("solver not implemented")
                for idx in range(0, len(samples)):
                    if self.solver_type == 'bo':
                        value = self.collect_samples( samples[idx])
                    elif self.solver_type == 'turbo':
                        value = self.collect_samples( samples[idx], values[idx] )
                    elif self.solver_type == 'scbo':
                        value = self.collect_samples( samples[idx], values[idx] )
                    else:
                        raise Exception("solver not implemented")
                    
                    self.backpropogate( leaf, value )
            print("total samples:", len(self.samples) )
            print("current best f(x):", np.absolute(self.curt_best_value) )
            # print("current best x:", np.around(self.curt_best_sample, decimals=1) )
            print("current best x:", self.curt_best_sample )
        
        self.dump_trace()



