#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:30:45 2022

@author: juliuslange
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import random


#%%

def origin_generation(node_passengers,nodes):
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
            origins.append(nodes[i])
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

def random_destination_generation(node_passengers,nodes):
    '''
    generates destination nodes by randomly shuffling origin list
    ----------
    inputs:
    node_passengers : np.array - list of how many players start at each node
    ----------
    returns:
    destinations : list - list of destination nodes
    '''
    destinations=origin_generation(node_passengers,nodes)
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
    if nx.has_path(G,start,end)==False:
        end=start
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
            if capacities[i]<1:
                factor=demands[i]
            else: factor=demands[i]/capacities[i]
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
        print(i)
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
        if i>=len(times):
            path_time+=times[-1][index]
        else:
            path_time+=times[i][index]
    return path_time

def path_congestion(G,edges,times,path):
    path_times=[]
    ideal_times=[]
    path_cong=[]
    path_edges=path_to_edges(path)
    costs=list(nx.get_edge_attributes(G,'edge_costs').values())
    for i in range(len(path_edges)):
        index=edges.index(path_edges[i])
        if i>=len(times):
            path_times.append(times[-1][index])
            ideal_times.append(costs[index])
            path_cong.append(times[-1][index]/costs[index])
        else:
            path_times.append(times[i][index])
            ideal_times.append(costs[index])
            path_cong.append(times[i][index]/costs[index])
    return path_cong

def most_congested_subpath(G,edges,times,path):
    #path_edges=path_to_edges(path)
    if len(path)==1:
        return path
    path_cong=path_congestion(G,edges,times,path)
    max_cong=np.max(path_cong)
    max_cong_index=path_cong.index(max_cong)
    #subpath=[]
    if max_cong>1:
        subpath_length=int(max_cong*4)
    elif max_cong>10:
        subpath_length=40
    else: subpath_length=0
        
    if int(max_cong_index+subpath_length/2)<0:
        subpath=path
    elif int(max_cong_index+subpath_length/2)>len(path):
        subpath=path
    else:
        subpath=path[int(max_cong_index-subpath_length/2):int(max_cong_index+subpath_length/2)]
    return subpath
            
    

def ideal_time(path,G):
    path_edges=path_to_edges(path)
    edges=list(G.edges())
    costs=list(nx.get_edge_attributes(G,'edge_costs').values())
    ideal_time=0
    for i in range(len(path_edges)):
        idx=edges.index(path_edges[i])
        ideal_time+=costs[idx]
    return ideal_time

def normalise(something,min_val,max_val):
    list_min=min(something)
    list_max=max(something)
    list_range=list_max-list_min
    
    output=[]
    for i in range(len(something)):
        value=something[i]/list_range+list_min
        output.append(value)
    return output
        
        
def path_sensitivity(origin,destination,G,edges,times):
    if origin==destination: return 100,[origin,destination]
    path_ideal=quickest_path(G,origin,destination)
    #time_ideal=ideal_time([origin,destination],G)
    time_actual=path_time(G,edges,times,path_ideal)
    #alternative_paths=list(nx.all_simple_paths(G,origin,destination,len(path_ideal)))
    alternative_paths=[]
    alternative_path_generator=nx.all_simple_paths(G,origin,destination,len(path_ideal))
    #k = int(40/len(path_ideal))+1
    k=2
    if len(path_ideal)<20:
        for counter, path in enumerate(alternative_path_generator):
            alternative_paths.append(path)
            if counter == k-1:
                print(k-1)
                break
    else:
        print(0)
        alternative_paths.append([destination,destination])

        
    if len(alternative_paths)==1:
        return 100,[origin,destination]
    alternative_times=[]
    for i in range(len(alternative_paths)):
        alternative_times.append(path_time(G,edges,times,alternative_paths[i]))
    if len(alternative_times)<=1:
        return 100,[origin,destination]
    betterness=np.mean(alternative_times)/time_actual
    best_path_index=alternative_times.index(min(alternative_times))
    best_alternative=alternative_paths[best_path_index]
    return betterness,best_alternative

def path_sensitivity2(origin,destination,G,edges,times):
    if origin==destination: return 100,[origin,destination]
    path_ideal=quickest_path(G,origin,destination)
    if len(path_ideal)>39:
        return 100, path_ideal
    congested_subpath=most_congested_subpath(G,edges,times,path_ideal)
    if len(congested_subpath)<=1.2:
        return 100,path_ideal
    
    #time_ideal=ideal_time([origin,destination],G)
    time_actual=path_time(G,edges,times,congested_subpath)
    #alternative_paths=list(nx.all_simple_paths(G,origin,destination,len(path_ideal)))
    alternative_paths=[]
    alternative_path_generator=nx.all_simple_paths(G,congested_subpath[0],congested_subpath[-1],len(congested_subpath))
    #k = int(40/len(path_ideal))+1
    k=5
    start_index=path_ideal.index(congested_subpath[0])
    end_index=path_ideal.index(congested_subpath[-1])
    
    for counter, path in enumerate(alternative_path_generator):
        alternative_paths.append(path)
        if counter == k-1:
            print(k-1)
            break
        
    alternative_times=[]
    for i in range(len(alternative_paths)):
        alternative_times.append(path_time(G,edges,times[start_index:],alternative_paths[i]))

    betterness=np.mean(alternative_times)/time_actual
    best_path_index=alternative_times.index(np.min(alternative_times))
    best_alternative=alternative_paths[best_path_index]

    better_path=path_ideal[:start_index]+best_alternative+path_ideal[end_index+1:]
    return betterness,better_path

def alt_path_sensitivity(origin,destination,G,edges,times):
    if origin==destination: return 100,[origin,destination]
    path_ideal=quickest_path(G,origin,destination)
    time_actual=path_time(G,edges,times,path_ideal)
    path_edges=path_to_edges(path_ideal)
    ideal_path_costs=[]
    augmented_path_costs=[]
    path_edges_with_keys=[]
    for i in path_edges:
        ideal_path_costs.append(G.get_edge_data(i[0], i[1], 0)["edge_costs"])
        augmented_path_costs.append(100)
        path_edges_with_keys.append(tuple([i[0],i[1],0]))
    nx.set_edge_attributes(G,dict(zip(path_edges_with_keys,augmented_path_costs)),"edge_costs")
    alternative_path=quickest_path(G,origin,destination)
    altermative_path_time=path_time(G,edges,times,alternative_path)
    nx.set_edge_attributes(G,dict(zip(path_edges_with_keys,ideal_path_costs)),"edge_costs")
    if time_actual==0:
        betterness=100
    else:
        betterness=altermative_path_time/time_actual
    return betterness,alternative_path

def path_sensitivites(G,origins,destinations,passengers,paths,times):
    #odps=list(zip(origins,destinations))
    edges=list(G.edges())
    sensitivities=[]
    best_alternatives=[]
    for i in range(len(origins)):
        print(i)
        sensitivty,best_alternative=path_sensitivity2(origins[i],destinations[i],G,edges,times)
        sensitivities.append(sensitivty)
        best_alternatives.append(best_alternative)
        if sensitivities[i]<1:
            paths[i]=best_alternatives[i]    
    
    return sensitivities,best_alternatives,paths

def alt_path_sensitivites(G,origins,destinations,passengers,paths,times):
    #odps=list(zip(origins,destinations))
    edges=list(G.edges())
    sensitivities=[]
    best_alternatives=[]
    for i in range(len(origins)):
        print(i)
        sensitivty,best_alternative=alt_path_sensitivity(origins[i],destinations[i],G,edges,times)
        sensitivities.append(sensitivty)
        best_alternatives.append(best_alternative)
        if sensitivities[i]<1:
            paths[i]=best_alternatives[i]    
    
    return sensitivities,best_alternatives,paths

def one_step_iteration(G,origins,destinations,passengers,paths,times):
    #odps=list(zip(origins,destinations))
    sensitivities,best_alternatives,paths=path_sensitivites(G,origins,destinations,passengers,paths,times)
    #for i in range(len(origins)):
    #    if sensitivities[i]<0.7:
     #       paths[i]=best_alternatives[i]
    return paths

def alternative_times(G,origins,destinations,passengers,times,paths):
    
    #odps=list(zip(origins,destinations))
    
    #times,paths=final_times(G,origins,destinations,passengers)    
    #better_paths=one_step_iteration(G,origins,destinations,passengers,paths,times)
    sensitivities,best_alternatives,better_paths=path_sensitivites(G,origins,destinations,passengers,paths,times)
    #sensitivities,best_alternatives,better_paths=alt_path_sensitivites(G,origins,destinations,passengers,paths,times)
    
    N_steps=longest_shortest_path(better_paths)
    
    for i in range(N_steps):
        demand=route_demand(G,better_paths,i)
        time=step_times(G,demand,i)
        
        #demands[i]=demand
        times[i]=time
        
    return times, better_paths

def looking_at_map_times(G,origins,destinations,passengers,congestion):
    
    edges_with_keys=list(G.edges)
    #odps=odps=list(zip(origins,destinations))
    paths=[]
    edge_costs=list(nx.get_edge_attributes(G,'edge_costs').values())
    nx.set_edge_attributes(G,dict(zip(edges_with_keys,edge_costs/(congestion+1))),"better_costs")

    for i in range(len(passengers)):
        if nx.has_path(G,origins[i],destinations[i])==False:
            origins[i]=destinations[i]
        path=nx.shortest_path(G,origins[i],destinations[i],"better_costs","bellman-ford")
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

def unique_paths(paths):
    unique_paths = []
    for x in paths:
        if x not in unique_paths:
            unique_paths.append(x)
    return unique_paths

#%%
i=0

def weight_func(start,stop,attr,step_no=0):
    step_no+=1
    if type(attr)=="NoneType":
        print("none")
        return 1
    ptt_list=attr[0]["projected_travel_time"]
    if type(ptt_list)==float:
        return ptt_list
    
    return ptt_list[step_no]
    
    

def optimum_model(G,origins,destinations,passengers):
    #random_passengers=random.shuffle(passengers)
    #random_origins=random.shuffle(origins)
    #random_destinations=random.shuffle(destinations)
    edges=list(G.edges())
    
    combined=list(zip(origins,destinations,passengers))
    random.shuffle(combined)
    random_origins,random_destinations,random_passengers=zip(*combined)
    demands=[[0]*len(edges) for _ in range(len(edges))] 
    projected_travel_times=[[0]*len(edges) for _ in range(len(edges))] 
    capacities=list(nx.get_edge_attributes(G,'edge_capacities').values())
    edge_costs=list(nx.get_edge_attributes(G,'edge_costs').values())
    paths=[]
    path_lengths=[]
    
    for i in range(len(passengers)):
        print(i,random_origins[i],random_destinations[i])
        if nx.has_path(G,random_origins[i],random_destinations[i])==False:
            #random_destinations[i]=random_origins[i]
            path=[random_origins[i],random_origins[i]]
            #continue
        else:
            path=nx.shortest_path(G,random_origins[i],random_destinations[i],"projected_travel_times","bellman-ford")
        paths.append(path)
        path_lengths.append(len(path))
        path_edges=path_to_edges(path)
        path_edges_with_keys=[]
        for j in range(len(path_edges)):
            path_edges_with_keys.append((path_edges[j][0],path_edges[j][1],0))
            if G.has_edge(path_edges[j][0],path_edges[j][1])==False: 
                continue
            edge_index=edges.index(path_edges[j])
            demands[j][edge_index]+=1
            if capacities[edge_index]==0:
                capacities[edge_index]=1
            congestion=demands[j][edge_index]/capacities[edge_index]
            if congestion<1:
                projected_travel_times[j][edge_index]=edge_costs[edge_index]
            else:
                projected_travel_times[j][edge_index]=congestion*edge_costs[edge_index]
            
        
            nx.set_edge_attributes(G,dict(zip(path_edges_with_keys,projected_travel_times[j])),str(i))
        
    
        N_steps=np.max(path_lengths)
        times=np.empty((N_steps,G.number_of_edges()))
        
        for i in range(N_steps):
            actual_demand=demands[i]
            time=step_times(G,actual_demand,i)
            
            #demands[i]=demand
            
            times[i]=time
            
    return times,paths









        
    
                