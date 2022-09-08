#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:14:54 2022

@author: juliuslange
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import random

#%%

def origin_generation(node_passengers):
    origins=[]
    for i in range(len(node_passengers)):
        for j in range(node_passengers[i]):
            origins.append(i)
    return origins

def destination_generation(N_passengers,node_weights,G):
    nodes=list(G.nodes)
    weight_sum=np.sum(node_weights)
    probabilities=node_weights/weight_sum
    destinations=random.choices(nodes, weights=probabilities, k=N_passengers)
    return destinations 

def shortest_path_distance(G,start,end):
    path=nx.shortest_path(G, source=start, target=end, weight="edge_distances")
    dist=0
    for i in range(len(path)-1):
        dist+=G[path[i]][path[i+1]]["edge_distances"]
    return dist,path

def longest_shortest_path(paths):
    length=0
    for i in paths:
        if len(i)>length:
            length=len(i)
    return length

def inverse_edges(G):
    edge_list=list(G.edges())
    inv_edge_list=[]
    for i in range(len(edge_list)):
       inv_edge_list.append(list(edge_list[i])) 
       inv_edge_list[i].reverse()
       inv_edge_list[i]=tuple(inv_edge_list[i])
    return inv_edge_list     

def route_demand(G,paths,step):
    edge_list=list(G.edges())
    inv_edge_list=inverse_edges(G)
    #print(edge_list)
    demands1=np.zeros(len(edge_list))
    demands2=np.zeros(len(edge_list))

    for i in paths:
        if len(i)>step+1:
            path_tuple=(i[step],i[step+1])
            #print(path_tuple)
            if path_tuple[0]<path_tuple[1]:
                idx1=edge_list.index(path_tuple)
                demands1[idx1]+=1
            else:
                idx2=inv_edge_list.index(path_tuple)
                demands2[idx2]+=1
    return demands1,demands2  

def step_times(G,demands1,demands2,step):
    times1=[]
    times2=[]
    capacities=list(nx.get_edge_attributes(G,'edge_capacities').values())
    distances=list(nx.get_edge_attributes(G,'edge_distances').values())
    for i in range(G.number_of_edges()):
        if demands1[i]>capacities[i]:
            factor1=demands1[i]/capacities[i]
        else: factor1=1
        if demands2[i]>capacities[i]:
            factor2=demands2[i]/capacities[i]
        else: factor2=1
        times1.append(distances[i]*factor1)
        times2.append(distances[i]*factor2)
    return times1,times2

def final_times(G,origins,destinations,passengers):
    odps=odps=list(zip(origins,destinations))
    dists=[]
    paths=[]
    for i in range(len(passengers)):
        dist,path=shortest_path_distance(G,odps[i][0],odps[i][1])
        dists.append(dist)
        paths.append(path)
    for i in range(len(dists)):
        if i==len(dists): break
        if dists[i]==0:
            del dists[i]
            del paths[i]
            
    N_steps=longest_shortest_path(paths)
    
    demands1=np.empty((N_steps,G.number_of_edges()))
    times1=np.empty((N_steps,G.number_of_edges()))
    demands2=np.empty((N_steps,G.number_of_edges()))
    times2=np.empty((N_steps,G.number_of_edges()))
    
    for i in range(N_steps):
        demand1,demand2=route_demand(G,paths,i)
        time1,time2=step_times(G,demand1,demand2,i)
        
        demands1[i]=demand1
        demands2[i]=demand2
        
        times1[i]=time1
        times2[i]=time2
        
    return times1,times2

def avg_congestion(G,times1,timess2):
    distances=list(nx.get_edge_attributes(G,'edge_distances').values())
    avg_t1=np.mean(times1,axis=0)
    avg_t2=np.mean(times1,axis=0)
    time_sum=avg_t1+avg_t2
    congestion=time_sum/(np.array(distances)*2)-1
    return congestion

def get_colors(inp, colormap, vmin=0, vmax=1):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

def draw_congested_network(G,congestion):
    node_weights=np.array(list(nx.get_node_attributes(G,'node_weight').values()))
    node_weights*=1
    edge_widths=list(nx.get_edge_attributes(G,'edge_capacities').values())
    edge_colors=get_colors(congestion, plt.cm.seismic)
    
    nx.draw_networkx(G,node_size=node_weights,width=edge_widths,edge_color=edge_colors,with_labels=True)
    return

 
        

#%%
'''
N_nodes=5
N_edges=6

node_weights=np.array([5,5,5,5,5])
node_passengers=np.array([5,5,5,5,5])
edge_capacities=np.array([2,2,2,2,2,2])
edge_distances=np.array([1,2,3,4,5,6])
'''
N_nodes=100
N_edges=180

node_weights=np.random.randint(1,10,size=N_nodes)
node_passengers=node_weights
edge_capacities=np.random.randint(1,5,size=N_edges)
edge_distances=np.random.randint(1,2,size=N_edges)

#%%

N_passengers=np.sum(node_passengers)

G=nx.gnm_random_graph(N_nodes,N_edges)
#G=nx.grid_2d_graph((10,10))
attempt=0
while nx.is_connected(G)!=True:
    print(attempt)
    attempt+=1
    G=nx.gnm_random_graph(N_nodes,N_edges)


nx.set_node_attributes(G,dict(zip(list(G.nodes()),node_weights)),"node_weight")
nx.set_node_attributes(G,dict(zip(list(G.nodes()),node_passengers)),"node_passengers")

nx.set_edge_attributes(G,dict(zip(list(G.edges()),edge_capacities)),"edge_capacities")
nx.set_edge_attributes(G,dict(zip(list(G.edges()),edge_distances)),"edge_distances")

passengers=list(np.arange(N_passengers))
origins=origin_generation(node_passengers) #require node_passengers==node_weights     
destinations=destination_generation(N_passengers,node_weights,G)

#odps=list(zip(origins,destinations))

#nx.draw_networkx(G,pos=nx.spring_layout(G),node_size=list(nx.get_node_attributes(G,'node_weight').values()))


'''
dists=[]
paths=[]
for i in range(len(passengers)):
    dist,path=shortest_path_distance(G,odps[i][0],odps[i][1])
    dists.append(dist)
    paths.append(path)
    

for i in range(len(dists)):
    if i==len(dists): break
    if dists[i]==0:
            del dists[i]
            del paths[i]
        
'''
           
times1,times2=final_times(G,origins,destinations,passengers)

congestion=avg_congestion(G,times1,times2)

draw_congested_network(G,congestion)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

#NEW YORK MODEL

N_nodes=30
N_edges=49

G=nx.grid_2d_graph(6,5)
G=nx.convert_node_labels_to_integers(G)

main_streets=[(2,7),(5,6),(6,7),(7,8),(8,9),(7,12),(12,17),(17,22),(20,21),(21,22),(22,23),(23,24),(22,27)]
edges=list(G.edges)
main_street_indices = [i for i, item in enumerate(edges) if item in main_streets]

node_weights=np.full(30,67)
node_passengers=node_weights
edge_distances=np.full(49,500)

edge_capacities=np.full(49,10)
j=0
for i in range(len(edge_capacities)):
    index=main_street_indices[j]
    if i==index:
        edge_capacities[i]=20
        j+=1
        if j==len(main_street_indices): break
    
N_passengers=np.sum(node_passengers)


#%%

nx.set_node_attributes(G,dict(zip(list(G.nodes()),node_weights)),"node_weight")
nx.set_node_attributes(G,dict(zip(list(G.nodes()),node_passengers)),"node_passengers")

nx.set_edge_attributes(G,dict(zip(list(G.edges()),edge_capacities)),"edge_capacities")
nx.set_edge_attributes(G,dict(zip(list(G.edges()),edge_distances)),"edge_distances")

passengers=list(np.arange(N_passengers))
origins=origin_generation(node_passengers) #require node_passengers==node_weights     
destinations=destination_generation(N_passengers,node_weights,G)

#odps=list(zip(origins,destinations))

#nx.draw_networkx(G,pos=nx.spring_layout(G),node_size=list(nx.get_node_attributes(G,'node_weight').values()))

#%%
           
times1,times2=final_times(G,origins,destinations,passengers)

congestion=avg_congestion(G,times1,times2)

#%%

draw_congested_network(G,congestion)

#%%

#Destination Histogram

plt.hist(destinations,np.arange(N_nodes+1))







            

        


        
    
    



        




