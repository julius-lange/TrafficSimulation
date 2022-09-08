#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:02:05 2022

@author: juliuslange
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#%%

def Node_weight_distr(N_nodes,mean,std):
    dict_nodes={}
    n_weights=[]
    for i in range(N_nodes):
        n_weights.append(float(np.random.normal(mean, std)))
        dict_nodes[i] = n_weights[i]
    return dict_nodes

def Edge_dist_distr(N_edges,mean,std):
    e_weights=[]
    for i in range(N_edges):
        e_weights.append(float(np.random.normal(mean, std)))
    return e_weights

def ODP_demand_distr(G,exp):
    odp_demands=[]
    for i in range(len(G.nodes())):
        for j in range(len(G.nodes())):
            demand=(G.nodes[i]["node_weight"]*G.nodes[j]["node_weight"])**exp
            if i==j: demand=0
            odp_demands.append(demand)
    return odp_demands


    

class Player:
    def __init__(self, origin, destination, strategy):
        self.origin = origin
        self.destination = destination
        self.strategy = strategy

    origin=0
    destination=0
    strategy=0
    
#def choose_start_end(Player,G):
    


#%%
    
#Setting up network with weights
    
N_nodes=20
N_edges=100

node_weigth_mean=200
node_weight_std=80

edge_dist_mean=0.5
edge_dist_std=0.2

G=nx.gnm_random_graph(N_nodes,N_edges)

n_weights=Node_weight_distr(N_nodes,node_weigth_mean,node_weight_std)
e_dists=Edge_dist_distr(N_edges,edge_dist_mean,edge_dist_std)
#e_demands

nx.set_node_attributes(G,n_weights,"node_weight")


dict_edge=dict(zip(list(G.edges()),e_dists))

odp_demands=ODP_demand_distr(G,1)

for i in len(dict_edge):
    

nx.set_edge_attributes(G,dict_edges,"edge_weight")


nx.draw_networkx(G,pos=nx.spring_layout(G),node_size=list(nx.get_node_attributes(G,'node_weight').values()),width=np.array(e_dists))

#%%

#Setting up transport game

N_players=1000


