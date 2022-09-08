#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:05:56 2022

@author: juliuslange
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import random

#%%

def origin_generation(node_passengers):
    '''
    generate list of length of number of passengers with corresponding
    staring nodes
    ----------
    inputs:
    node_passengers : np.array - list of how many players start at each node
    ----------
    returns:
    origins : list - list of origin nodes
    '''
    origins=[]
    for i in range(len(node_passengers)):
        for j in range(node_passengers[i]):
            origins.append(i)
    return origins

def destination_generation(N_passengers,node_weights,G):
    '''
    attempt to generate destionation nodes by randomly choosing a node 
    according to weigthing provided by node_weights 
    SEEMS TO LEAD TO FAVOURING OF LOWER NUMBER NODE INDICES, DON'T USE
    ----------
    inputs:
    N_passengers : int - number of passengers
    node_weights : np.array - array of node weights, so far identical with 
                              node_passengers
    G : nx.network - network
    ----------
    returns:
    destinations : list - list of destination nodes
    '''
    nodes=list(G.nodes)
    weight_sum=np.sum(node_weights)
    probabilities=node_weights/weight_sum
    destinations=random.choices(nodes, weights=probabilities, k=N_passengers)
    return destinations 

def random_destination_generation(node_passengers):
    '''
    generates destination nodes by randomly shuffling origin list
    ----------
    inputs:
    node_passengers : np.array - list of how many players start at each node
    ----------
    returns:
    destinations : list - list of destination nodes
    '''
    destinations=origin_generation(node_passengers)
    random.shuffle(destinations)
    return destinations

def quickest_path(G,start,end):
    '''
    generates the quickst path between two nodes according to time take on the
    empty network, according to node distances and speedlimits
    ----------
    inputs:
    G : nx.network - network (with edge distances and speedlimits included)
    start/end : int - index of start/end node
    ----------
    returns:
    dist : float - generalised time for journey
    path : list - list of nodes on path
    '''
    path=nx.shortest_path(G,start,end,"edge_costs","bellman-ford")

    return path

def longest_shortest_path(paths):
    '''
    computes number of step in longest path for computation purposes
    ----------
    inputs:
    paths : list of path lists - list of path lists
    ----------
    returns:
    length : int - number of edges traversed in path
    '''
    length=0
    for i in paths:
        if len(i)>length:
            length=len(i)
    return length

def inverse_edges(G):
    '''
    reverses edge tuples of network to provide "both-way-navigation"
    ----------
    inputs:
    G : nx.network - network    
    ----------
    returns:
    inv_edge_list : list of tuples - list of node tuples of all edges starting
                    with lower index node
    '''
    edge_list=list(G.edges())
    inv_edge_list=[]
    for i in range(len(edge_list)):
       inv_edge_list.append(list(edge_list[i])) 
       inv_edge_list[i].reverse()
       inv_edge_list[i]=tuple(inv_edge_list[i])
    return inv_edge_list     

def route_demand(G,paths,step):
    '''
    determines how many players aim to take which edge during one step
    ----------
    inputs:
    G : nx.network - network    
    paths : list of path lists - list of path lists
    step : int - step number (starting from 0 up to longest_shortest_path)
    ----------
    returns:
    demands1 : np.array - number passengers using all edges (lower idex to 
                          upper idex during one step)
    demands2 : np.array - number passengers using all edges (upper idex to 
                          lower idex during one step)
    '''
    edge_list=list(G.edges())
    demands=np.zeros(len(edge_list))

    for i in paths:
        if len(i)>step+1:
            path_tuple=(i[step],i[step+1])
            #print(path_tuple)
            idx=edge_list.index(path_tuple)
            demands[idx]+=1
            
    return demands  

def step_times(G,demands,step):
    '''
    determines the times taken on each edge during one step according
    to demand and the edge characteristics (distance, capacity, and speedlimit)
    ----------
    inputs:
    G : nx.network - network  
    demands1/2 : np.array - demands per edge during one step (index convention)
    step : int - step number 
    ----------
    returns:
    times1/2 : np.array - times taken to traverse each edge according to
               formula (MAY WANT TO ADJUST TO DIFFERENT MODELS) using
               aforementioned index convention
    '''
    times=[]
    capacities=list(nx.get_edge_attributes(G,'edge_capacities').values())
    costs=list(nx.get_edge_attributes(G,'edge_costs').values())
    for i in range(G.number_of_edges()):
        if demands[i]>capacities[i]:
            factor=demands[i]/capacities[i]
        else: factor=1
        
        times.append(costs[i]*factor)
    return times

def final_times(G,origins,destinations,passengers):
    '''
    determines the times taken on each edge during all steps
    ----------
    inputs:
    G : nx.network - network  
    destinations : list - list of destination nodes
    origins : list - list of origin nodes
    passengers : list - list of passenger indices
    
    ----------
    returns:
    times1/2 : np.array of np.arrays - times taken to traverse each edge 
                according to formula using aforementioned index convention for
                all steps required
    paths : list of path lists - list of path lists for stats
    '''
    odps=odps=list(zip(origins,destinations))
    paths=[]
    for i in range(len(passengers)):
        path=quickest_path(G,odps[i][0],odps[i][1])
        paths.append(path)
            
    N_steps=longest_shortest_path(paths)
    
    demands=np.empty((N_steps,G.number_of_edges()))
    times=np.empty((N_steps,G.number_of_edges()))
    
    for i in range(N_steps):
        demand=route_demand(G,paths,i)
        time=step_times(G,demand,i)
        
        demands[i]=demand
        
        times[i]=time
        
    return times, paths

def avg_congestion(G,times):
    '''
    determined how congested each edge is on average over all steps according
    to kinda arbitrary formula
    ----------
    inputs:
    G : nx.network - network  
    times1/2 : np.array - times taken to traverse each edge
    ----------
    returns:
    congestion : np.array - list of congestion averages ranging from 0 (no
                             no congestion) to inf (but actually about 10ish)
    '''
    
    costs=list(nx.get_edge_attributes(G,'edge_costs').values())
    avg_times=np.mean(times,axis=0)
    congestion=avg_times/(np.array(costs))-1
    return congestion

def get_colors(inp, colormap, vmin=0, vmax=1):
    '''
    normalises imput values to create colormap for pretty plots
    '''
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

def draw_congested_network(G,congestion):
    '''
    draws network with node weights as node sizes, edge capacities as edge
    widths, and congestion values as edge color (blue - no congestion, red - 
    lots of congestion)
    ----------
    inputs:
    G : nx.network - network  
    congestion : np.array - list of congestion averages    
    '''
    node_weights=np.array(list(nx.get_node_attributes(G,'node_weight').values()))
    node_weights*=1
    edge_widths=list(nx.get_edge_attributes(G,'edge_capacities').values())
    edge_colors=get_colors(congestion, plt.cm.seismic)

    
    nx.draw_networkx(G,node_size=node_weights,width=edge_widths,edge_color=edge_colors,with_labels=True)
    return

def individual_travel_time(paths,edges,times,p_number):
    '''
    computes travel times in congested network from path/passenger number. 
    Potential for further analysis by returning list of times.
    ----------
    inputs:
    paths : list of path lists - list of path lists
    edges : list of tuples - list of edges
    inv_edges : list of tuples - list of investe edges (see convention)
    times1/2 : np.array - times taken to traverse each edge
    p_number : passenger/path index
    ----------
    returns:
    sum(individual_times) : float - time take for path
    '''
    
    path=paths[p_number]
    individual_times=[]
    for i in range(len(path)-1):
        subpath=(path[i],path[i+1])
        idx=edges.index(subpath)
        individual_times.append(times[i][idx])
        
    return sum(individual_times)

def travel_time_distribution(paths,edges,times):
    '''
    travel times for all passengers on congested network
    ----------
    inputs:
    paths : list of path lists - list of path lists
    edges : list of tuples - list of edges
    inv_edges : list of tuples - list of investe edges (see convention)
    times1/2 : np.array - times taken to traverse each edge
    ----------
    returns:
    ttime_distribution : list - list of time taken by all passengers
    '''
    ttime_distribution=[]
    for i in range(len(paths)):
        ttime_distribution.append(individual_travel_time(paths,edges,times,i))
    return ttime_distribution

def path_cost_distribution(paths,G):
    '''
    travel times for all passengers on completely empty network
    ----------
    inputs:
    paths : list of path lists - list of path lists
    G : nx.network - network  
    ----------
    returns:
    pathtime_distribution : list - list of time taken by all passengers without
                                    congestion
    '''
    pathtime_distribution=[]
    for i in range(len(paths)):
        path=quickest_path(G,paths[i][0],paths[i][-1])
        dist=ideal_time(path,G)
        pathtime_distribution.append(dist)
    return pathtime_distribution
        

#### From here functins for time optimisation
    
def index_shared_value(a, b):
  return [i for i, v in enumerate(a) if v == b[i]]

    
def path_to_edges(path):
    path_edges=[]
    for i in range(len(path)-1):
        path_edges.append((path[i],path[i+1]))
    return path_edges

def path_time(G,edges,times,path):
    path_time=0
    path_edges=path_to_edges(path)
    for i in range(len(path_edges)):
        index=edges.index(path_edges[i])
        path_time+=times[i][index]
    return path_time

def ideal_time(path,G):
    path_edges=path_to_edges(path)
    edges=list(G.edges())
    costs=list(nx.get_edge_attributes(G,'edge_costs').values())
    ideal_time=0
    for i in range(len(path_edges)):
        idx=edges.index(path_edges[i])
        ideal_time+=costs[idx]
    return ideal_time
        
        
        
    
        
#%%    
        
def path_sensitivity(origin,destination,G,edges,times):
    if origin==destination: return 100,[origin,destination]
    path_ideal=quickest_path(G,origin,destination)
    #time_ideal=ideal_time([origin,destination],G)
    time_actual=path_time(G,edges,times,path_ideal)
    alternative_paths=list(nx.all_simple_paths(G,origin,destination,len(path_ideal)))
    alternative_times=[]
    for i in range(len(alternative_paths)):
        alternative_times.append(path_time(G,edges,times,alternative_paths[i]))
    betterness=np.mean(alternative_times)/time_actual
    best_path_index=alternative_times.index(min(alternative_times))
    best_alternative=alternative_paths[best_path_index]
    return betterness,best_alternative

def path_sensitivites(G,times,odps):
    edges=list(G.edges())
    sensitivities=[]
    best_alternatives=[]
    for i in range(len(odps)):
        sensitivty,best_alternative=path_sensitivity(odps[i][0],odps[i][1],G,edges,times)
        sensitivities.append(sensitivty)
        best_alternatives.append(best_alternative)
    return sensitivities,best_alternatives

def one_step_iteration(G,times,paths,odps):
    sensitivities,best_alternatives=path_sensitivites(G,times,odps)
    for i in range(len(odps)):
        if sensitivities[i]<0.65:
            paths[i]=best_alternatives[i]
    return paths

def alternative_times(G,origins,destinations,passengers):
    
    odps=odps=list(zip(origins,destinations))
    paths=[]
    for i in range(len(passengers)):
        path=quickest_path(G,odps[i][0],odps[i][1])
        paths.append(path)
            
    N_steps=longest_shortest_path(paths)
    
    demands=np.empty((N_steps,G.number_of_edges()))
    times=np.empty((N_steps,G.number_of_edges()))

    
    for i in range(N_steps):
        demand=route_demand(G,paths,i)
        time=step_times(G,demand,i)
        
        demands[i]=demand
        times[i]=time
        
    better_paths=one_step_iteration(G,times,paths,odps)
    
    N_steps=longest_shortest_path(better_paths)
    
    for i in range(N_steps):
        demand=route_demand(G,better_paths,i)
        time=step_times(G,demand,i)
        
        demands[i]=demand
        times[i]=time
        
    return times, better_paths

        
    
                
    
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

G=nx.MultiDiGraph()

N_nodes=20
N_edges=50

for i in range(N_edges):
    A=random.randint(0,N_nodes)
    B=random.randint(0,N_nodes)
    G.add_edge(A,B)
    G.add_edge(B,A)

G.remove_edges_from(nx.selfloop_edges(G))

N_nodes=G.number_of_nodes()
N_edges=G.number_of_edges()
    
nx.draw_networkx(G)
    
#%%

edges_with_keys=list(G.edges)
edges=list(G.edges())
nodes=list(G.nodes())
inv_edges=inverse_edges(G)
#define edge and inv_edge lists

node_weights=np.full(N_nodes,40)
node_passengers=node_weights
#set initial passengers per node to 67

edge_distances=np.full(N_edges,500)
edge_distance_deviation=np.random.normal(0.0,1.0,N_edges)
edge_distances=edge_distances+edge_distance_deviation
#set edge distances to 500m, add random deviation to create definite best path

edge_speedlimits=np.full(N_edges,30) #30km/h
edge_capacities=np.full(N_edges,6) #6 cars/minute - not appropriate
#set edge distances and speedlimits to defaul values for small streets
    
N_passengers=np.sum(node_passengers)
passengers=list(np.arange(N_passengers))
#calculate number of passengers and create passenger list


nx.set_node_attributes(G,dict(zip(nodes,node_weights)),"node_weight")
nx.set_node_attributes(G,dict(zip(nodes,node_passengers)),"node_passengers")
#set node attributes node_weight (popularity) and number of initial passengers

nx.set_edge_attributes(G,dict(zip(edges_with_keys,edge_capacities)),"edge_capacities")
nx.set_edge_attributes(G,dict(zip(edges_with_keys,edge_distances/edge_speedlimits)),"edge_costs")
#set edge attributes capacity and cost (distance/speedlimit=time)

origins=origin_generation(node_passengers) #require node_passengers==node_weights    
#destinations2=destination_generation(N_passengers,node_weights,G) #this way produces weird results
destinations=random_destination_generation(node_passengers)
odps=list(zip(origins,destinations))
#generate origins, destinations, and zip for origin destination pairs

#nx.draw_networkx(G,pos=nx.spring_layout(G),node_size=list(nx.get_node_attributes(G,'node_weight').values()))
#draw network to check whether it seems right

#%%
           
times,paths=final_times(G,origins,destinations,passengers)
times2,paths2=alternative_times(G,origins,destinations,passengers)

congestion=avg_congestion(G,times)
congestion2=avg_congestion(G,times2)

#Simulation

#%%

#draw_congested_network(G,congestion)
draw_congested_network(G,congestion)

#plot congested network might have to do multiple times to get "flat" network

#%%

#Destination Histogram (should be flat or somewhat flat)

plt.hist(destinations,np.arange(N_nodes+1),histtype=u'step') 
#plt.hist(destinations2,np.arange(N_nodes+1),histtype=u'step') 
#%%

plt.hist(congestion,histtype=u'step',color="r") 
#plt.hist(congestion2,histtype=u'step',color="b") 



#%%

#OD Pair Histogram (should also be uniform for large networks and bad resolution)

plt.hist2d(origins, destinations, bins=(2,2), cmap=plt.cm.jet)
plt.colorbar()
plt.show()

#%%

#Travel Time Histogram (probably will need to use them to investigate differnet
#   navigation strategies)

tt_dist=travel_time_distribution(paths,edges,times)

ptt_dist=path_cost_distribution(paths,G)
ntt_dist=travel_time_distribution(paths2,edges,times2)



plt.hist(tt_dist,bins=np.linspace(0, 400, 41),histtype=u'step',label="congestion") 
plt.hist(ptt_dist,bins=np.linspace(0, 400, 41),histtype=u'step',label="no congestion") 
plt.hist(ntt_dist,bins=np.linspace(0, 400, 41),histtype=u'step',label="our solution") 

plt.legend()
plt.xlabel("time [s]")
plt.show()

print(np.sum(ptt_dist))
print(np.sum(tt_dist))
print(np.sum(ntt_dist))

#%%

plt.plot([0.4,0.5,0.6,0.7,0.75,0.8],[0,9315,42601,43283,16688,-23445])
plt.xlabel("threshold sensitivity to path change")
plt.ylabel("total time saving [s]")
plt.show()
plt.tight_layout()



#%%

#Longest Path (calculate which path took the most time, how long it took,
#   and how long it was supposed to take without congestion)

max_index = tt_dist.index(max(tt_dist))
print(paths[max_index])
print(tt_dist[max_index])
print(ptt_dist[max_index])


#%%

path_lengths=[]
for i in paths:
    path_lengths.append(len(i))

#%%

plt.hist(path_lengths,histtype=u'step',label="congestion") 























