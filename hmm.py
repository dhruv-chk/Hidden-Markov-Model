# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:28:07 2017

@author: Dhruv
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# create state space and initial state probabilities

states = ['Walking same', 'Walking opposite', 'Standing','Sitting','Empty']
pi = [0.20,0.20,0.20,0.20,0.20]
state_space = pd.Series(pi, index=states, name='states')
print(state_space)
print(state_space.sum())

# create transition matrix
# equals transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

q_df = pd.DataFrame(columns=states, index=states)
q_df.loc[states[0]] = [0.20,0.20,0.20,0.20,0.20]
q_df.loc[states[1]] = [0.20,0.20,0.20,0.20,0.20]
q_df.loc[states[2]] = [0.238,0.238,0.237,0.237,0.05]
q_df.loc[states[3]] = [0.238,0.238,0.237,0.237,0.05]
q_df.loc[states[4]] = [0.267,0.267,0.05,0.05,0.366]

print(q_df)

q = q_df.values
print('\n', q, q.shape, '\n')
print(q_df.sum(axis=1))

from pprint import pprint 

# create a function that maps transition probability dataframe 
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges

edges_wts = _get_markov_edges(q_df)
pprint(edges_wts)

"""
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(states)
print('Nodes:\n{G.nodes()}\n')

# edges represent transition probabilities
for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
print('Edges:')
pprint(G.edges(data=True))    

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos)

# create edge labels for jupyter plot but is not necessary
edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'group_activity_markov.dot')
"""
# create state space and initial state probabilities

hidden_states = ['Group Walking', 'Ignore','Standing talking','Sitting Discussion','Gather','Split','Empty Space']
pi = [0.143,0.143,0.143,0.143,0.143,0.143,0.142]
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
print('\n', state_space.sum())

# create hidden transition matrix
# a or alpha 
#   = transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.5,0.1,0,0,0.3,0,0.1]
a_df.loc[hidden_states[1]] = [0,0.7,0,0,0,0,0.3]
a_df.loc[hidden_states[2]] = [0.2,0,0.4,0.2,0,0.2,0]
a_df.loc[hidden_states[3]] = [0,0,0.3,0.7,0,0,0]
a_df.loc[hidden_states[4]] = [0,0.3,0.7,0,0,0,0]
a_df.loc[hidden_states[5]] = [0.5,0.5,0,0,0,0,0]
a_df.loc[hidden_states[6]] = [0.4,0.4,0,0,0,0,0.2]
print(a_df)

a = a_df.values
print('\n', a, a.shape, '\n')
print(a_df.sum(axis=1))

# create matrix of observation (emission) probabilities
# b or beta = observation probabilities given state
# matrix is size (M x O) where M is number of states 
# and O is number of different possible observations

observable_states = states

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [1,0,0,0,0]
b_df.loc[hidden_states[1]] = [0,1,0,0,0]
b_df.loc[hidden_states[2]] = [0,0,1,0,0]
b_df.loc[hidden_states[3]] = [0,0,0,1,0]
b_df.loc[hidden_states[4]] = [0.2,0.2,0.6,0,0]
b_df.loc[hidden_states[5]] = [0.5,0.5,0,0,0]
b_df.loc[hidden_states[6]] = [0,0,0,0,1]

print(b_df)
b = b_df.values
print('\n', b, b.shape, '\n')
print(b_df.sum(axis=1))

# create graph edges and weights

hide_edges_wts = _get_markov_edges(a_df)
pprint(hide_edges_wts)

emit_edges_wts = _get_markov_edges(b_df)
pprint(emit_edges_wts)
'''
# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(hidden_states)
print('Nodes:\n{G.nodes()}\n')

# edges represent hidden probabilities
for k, v in hide_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

# edges represent emission probabilities
for k, v in emit_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    
print('Edges:')
pprint(G.edges(data=True))    

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='neato')
nx.draw_networkx(G, pos)

# create edge labels for jupyter plot but is not necessary
emit_edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=emit_edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'Group_activity_hidden_markov.dot')
'''
# observation sequence of human activity
# observations are encoded numerically
#obs_map = {'Walking same':0, 'Standing':1, 'Sitting':2, 'Walking opposite':3, '}
#obs = np.array([1,1,2,1,0,1,2,1,0,2,2,0,1,0,1])
obs_map = {'Walking same':0, 'Walking opposite':1, 'Standing':2, 'Sitting':3, 'Empty':4}
obs = np.array([3,3,3,3,3,3,3,3])

inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print( pd.DataFrame(np.column_stack([obs, obs_seq]), 
                columns=['Obs_code', 'Obs_seq']) )
                
# define Viterbi algorithm for shortest path
# code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py

def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    #print(nStates)
    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        #p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi

path, delta, phi = viterbi(pi, a, b, obs)
print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)

#hidden_states = ['Group Walking', 'Ignore','Standing talking','Sitting Discussion','Gather','Split']

state_map = {6:'Group Walking', 1:'Ignore', 2:'Standing talking', 3:'Sitting Discussion', 4:'Gather', 5:'Split', 0:'Empty Space'}
state_path = [state_map[v] for v in path]

(pd.DataFrame()
 .assign(Observation=obs_seq)
 .assign(Best_Path=state_path))
