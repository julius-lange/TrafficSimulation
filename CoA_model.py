#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:49:01 2022

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
    path=nx.shortest_path(G,source=start,target=end,weight="edge_costs",method="bellman-ford")
    dist=0
    for i in range(len(path)-1):
        dist+=G[path[i]][path[i+1]]["edge_costs"]
    return dist,path

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
    times1=[]
    times2=[]
    capacities=list(nx.get_edge_attributes(G,'edge_capacities').values())
    costs=list(nx.get_edge_attributes(G,'edge_costs').values())
    for i in range(G.number_of_edges()):
        if demands1[i]>capacities[i]:
            factor1=demands1[i]/capacities[i]
        else: factor1=1
        if demands2[i]>capacities[i]:
            factor2=demands2[i]/capacities[i]
        else: factor2=1
        times1.append(costs[i]*factor1)
        times2.append(costs[i]*factor2)
    return times1,times2

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
    dists=[]
    paths=[]
    for i in range(len(passengers)):
        dist,path=quickest_path(G,odps[i][0],odps[i][1])
        dists.append(dist)
        paths.append(path)
    for i in range(len(dists)):
        if i==len(dists): break
#        if dists[i]==0:
#            del dists[i]
#            del paths[i]
            
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
        
    return times1,times2, paths

def avg_congestion(G,times1,timess2):
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
    avg_t1=np.mean(times1,axis=0)
    avg_t2=np.mean(times1,axis=0)
    time_sum=avg_t1+avg_t2
    congestion=time_sum/(np.array(costs)*2)-1
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

def individual_travel_time(paths,edges,inv_edges,times1,times2,p_number):
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
        if path[i]<path[i+1]:
            idx=edges.index(subpath)
            individual_times.append(times1[i][idx])
        elif path[i]>path[i+1]:
            idx=inv_edges.index(subpath)
            individual_times.append(times2[i][idx])
        else: individual_times.append(0)
    return sum(individual_times)

def travel_time_distribution(paths,edges,inv_edges,times1,times2):
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
        ttime_distribution.append(individual_travel_time(paths,edges,inv_edges,times1,times2,i))
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
        dist,path=quickest_path(G,paths[i][0],paths[i][-1])
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

def path_time(G,edges,inv_edges,times1,times2,path):
    path_time=0
    path_edges=path_to_edges(path)
    for i in range(len(path_edges)):
        if path_edges[i] in edges:
            index=edges.index(path_edges[i])
            path_time+=times1[i][index]
        else:
            index=inv_edges.index(path_edges[i])
            path_time+=times2[i][index]
    return path_time
        
#%%    
        
def path_sensitivity(origin,destination,G,edges,inv_edges,times1,times2):
    if origin==destination: return 100,[origin,destination]
    time_ideal,path_ideal=quickest_path(G,origin,destination)
    time_actual=path_time(G,edges,inv_edges,times1,times2,path_ideal)
    alternative_paths=list(nx.all_simple_paths(G,origin,destination,len(path_ideal)))
    alternative_times=[]
    for i in range(len(alternative_paths)):
        alternative_times.append(path_time(G,edges,inv_edges,times1,times2,alternative_paths[i]))
    betterness=np.mean(alternative_times)/time_actual
    best_path_index=alternative_times.index(min(alternative_times))
    best_alternative=alternative_paths[best_path_index]
    return betterness,best_alternative

def path_sensitivites(G,times1,times2,odps):
    edges=list(G.edges)
    inv_edges=inverse_edges(G)
    sensitivities=[]
    best_alternatives=[]
    for i in range(len(odps)):
        sensitivty,best_alternative=path_sensitivity(odps[i][0],odps[i][1],G,edges,inv_edges,times1,times2)
        sensitivities.append(sensitivty)
        best_alternatives.append(best_alternative)
    return sensitivities,best_alternatives

def one_step_iteration(G,times1,times2,paths,odps):
    sensitivities,best_alternatives=path_sensitivites(G,times1,times2,odps)
    for i in range(len(odps)):
        if sensitivities[i]<0.65:
            paths[i]=best_alternatives[i]
    return paths

def alternative_times2(G,origins,destinations,passengers,n_reps):
    
    odps=odps=list(zip(origins,destinations))
    dists=[]
    paths=[]
    for i in range(len(passengers)):
        dist,path=quickest_path(G,odps[i][0],odps[i][1])
        dists.append(dist)
        paths.append(path)
    for i in range(len(dists)):
        if i==len(dists): break
#        if dists[i]==0:
#           del dists[i]
#            del paths[i]
            
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
        
    for i in range(n_reps):
        
        better_paths=one_step_iteration(G,times1,times2,paths,odps)
    
        N_steps=longest_shortest_path(better_paths)
        
        for i in range(N_steps):
            demand1,demand2=route_demand(G,better_paths,i)
            time1,time2=step_times(G,demand1,demand2,i)
        
            demands1[i]=demand1
            demands2[i]=demand2
        
            times1[i]=time1
            times2[i]=time2
            
        paths=better_paths
        
    return times1,times2, better_paths

def alternative_times(G,origins,destinations,passengers):
    
    odps=odps=list(zip(origins,destinations))
    dists=[]
    paths=[]
    for i in range(len(passengers)):
        dist,path=quickest_path(G,odps[i][0],odps[i][1])
        dists.append(dist)
        paths.append(path)
    for i in range(len(dists)):
        if i==len(dists): break
#        if dists[i]==0:
#           del dists[i]
#            del paths[i]
            
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
        
    better_paths=one_step_iteration(G,times1,times2,paths,odps)
    
    N_steps=longest_shortest_path(better_paths)
    
    for i in range(N_steps):
        demand1,demand2=route_demand(G,better_paths,i)
        time1,time2=step_times(G,demand1,demand2,i)
        
        demands1[i]=demand1
        demands2[i]=demand2
        
        times1[i]=time1
        times2[i]=time2
        
    return times1,times2, better_paths

        
    
                
    
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

#NEW YORK MODEL - 5x6 grid with three main roads

N_nodes=30
N_edges=49

G=nx.grid_2d_graph(6,5)
G=nx.convert_node_labels_to_integers(G)
#setting up network and relableing all nodes to int

main_streets=[(2,7),(5,6),(6,7),(7,8),(8,9),(7,12),(12,17),(17,22),(20,21),(21,22),(22,23),(23,24),(22,27)]
#defining edges which are main streets - needs to change for different maps

edges=list(G.edges)
inv_edges=inverse_edges(G)
#define edge and inv_edge lists

main_street_indices = [i for i, item in enumerate(edges) if item in main_streets]
#get indices of main_street elements in edges to change their characteristics

node_weights=np.full(30,67)
node_passengers=node_weights
#set initial passengers per node to 67

edge_distances=np.full(49,500)
edge_distance_deviation=np.random.normal(0.0,1.0,49)
edge_distances=edge_distances+edge_distance_deviation
#set edge distances to 500m, add random deviation to create definite best path

edge_speedlimits=np.full(49,30) #30km/h
edge_capacities=np.full(49,6) #6 cars/minute - not appropriate
#set edge distances and speedlimits to defaul values for small streets

j=0
for i in range(len(edge_capacities)):
    index=main_street_indices[j]
    if i==index:
        edge_capacities[i]=12 #12 cars/minute
        edge_speedlimits[i]=50 #50 km/h
        j+=1
        if j==len(main_street_indices): break
#set capacity and speedlimits to higher values for main streets
    
N_passengers=np.sum(node_passengers)
passengers=list(np.arange(N_passengers))
#calculate number of passengers and create passenger list


nx.set_node_attributes(G,dict(zip(list(G.nodes()),node_weights)),"node_weight")
nx.set_node_attributes(G,dict(zip(list(G.nodes()),node_passengers)),"node_passengers")
#set node attributes node_weight (popularity) and number of initial passengers

nx.set_edge_attributes(G,dict(zip(list(G.edges()),edge_capacities)),"edge_capacities")
nx.set_edge_attributes(G,dict(zip(list(G.edges()),edge_distances/edge_speedlimits)),"edge_costs")
#set edge attributes capacity and cost (distance/speedlimit=time)

origins=origin_generation(node_passengers) #require node_passengers==node_weights    
#destinations2=destination_generation(N_passengers,node_weights,G) #this way produces weird results
destinations=random_destination_generation(node_passengers)
odps=list(zip(origins,destinations))
#generate origins, destinations, and zip for origin destination pairs

#nx.draw_networkx(G,pos=nx.spring_layout(G),node_size=list(nx.get_node_attributes(G,'node_weight').values()))
#draw network to check whether it seems right

#%%
           
times1,times2,paths=final_times(G,origins,destinations,passengers)
times1_2,times2_2,paths_2=alternative_times(G,origins,destinations,passengers)
#times1_2,times2_2,paths_2=alternative_times2(G,origins,destinations,passengers,5)

congestion=avg_congestion(G,times1,times2)
congestion2=avg_congestion(G,times1_2,times2_2)

#Simulation

#%%

#draw_congested_network(G,congestion)
draw_congested_network(G,congestion2)

#plot congested network might have to do multiple times to get "flat" network

#%%

#Destination Histogram (should be flat or somewhat flat)

plt.hist(destinations,np.arange(N_nodes+1),histtype=u'step') 
#plt.hist(destinations2,np.arange(N_nodes+1),histtype=u'step') 
#%%

plt.hist(congestion,histtype=u'step',color="r") 
plt.hist(congestion2,histtype=u'step',color="b") 



#%%

#OD Pair Histogram (should also be uniform for large networks and bad resolution)

plt.hist2d(origins, destinations, bins=(2,2), cmap=plt.cm.jet)
plt.colorbar()
plt.show()

#%%

#Travel Time Histogram (probably will need to use them to investigate differnet
#   navigation strategies)

tt_dist=travel_time_distribution(paths,edges,inv_edges,times1,times2)

ptt_dist=path_cost_distribution(paths,G)
ntt_dist=travel_time_distribution(paths_2,edges,inv_edges,times1_2,times2_2)



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

 






            

        


        
    
    



        




