import networkx as nx
import gym
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from gym import spaces
from time import time
from pprint import pprint
import random


class Environment(gym.Env):
  '''
  Define the environment

  Observation space: the capacity and the allocated bandwidth of all the edges in the graph (2 x num_edges)
  Action space: the k possible paths that the agent has to choose when receiving a demand (k-dimensional space)
  '''
  def __init__(self, graph, k, max_demand):
    
    self.graph = graph   # the input graph
    self.num_nodes = len(graph.nodes())
    self.num_edges = len(graph.edges())
    self.num_features = None
    self.features = None
    self.state = None
    self.initial_state = None
    self.k = k
    self.first_k_shortest_paths = dict()  # for any given pair of nodes save the first k shortest paths
    self.is_episode_over = True
    self.reward = 0
    self.edges_topology = None
    self.max_demand = max_demand
    self.max_capacity = sorted(self.graph.edges(data=True),key= lambda x: x[2]['capacity'],reverse=True)[0][2]['capacity']

    self.action_space = spaces.Discrete(k)
    self.observation_space = spaces.Box(low=0, high=self.max_capacity, shape=(2, self.num_edges))
    

  def get_edges(self):
    return tf.reshape(tf.convert_to_tensor(list(self.graph.edges())), [2, -1])
  
  def set_seed(self, seed):
    random.seed(seed)
    np.random.seed(seed)
    
  def get_edges_topology(self):
    '''
    Get the topology of the edges, i.e. for every edge what are the neighboring edges
    This is used in the neural network for the neighboring aggregation of the edges 
    '''
    first = list()
    second = list()

    # For each edge we iterate over all neighbour edges
    edges = [tuple(sorted(edge)) for edge in self.graph.edges()]
    ordered_edges = sorted(edges)

    for i, j in ordered_edges:
          
      neighbour_edges = self.graph.edges(i)

      for m, n in neighbour_edges:
        if ((i != m or j != n) and (i != n or j != m)):
          first.append(self.graph.get_edge_data(i, j)['ID'])
          second.append(self.graph.get_edge_data(m, n)['ID'])

      neighbour_edges = self.graph.edges(j)
      for m, n in neighbour_edges:
        if ((i != m or j != n) and (i != n or j != m)):
          first.append(self.graph.get_edge_data(i, j)['ID'])
          second.append(self.graph.get_edge_data(m, n)['ID'])
    
    self.edges_topology = tf.constant([first, second], dtype='int32')

  def generate_environment(self): #--------------------------------------------------------------------------------------------------------------------------------

    # index the edges
    id = 0
    for i, j in self.graph.edges():
      self.graph.get_edge_data(i, j)['ID'] = id
      id += 1


    self.get_edges_topology()



    # save first k shortest paths for every pair of nodes
    num_shortest_path = np.zeros((self.num_edges)) # for every edge we store the number of (first k) shortest paths that pass through it
    for n1 in self.graph:
      for n2 in self.graph:
        # if (n1 != n2) and (str(i) + ':' + str(j) not in self.first_k_shortest_paths):
          if (n1 != n2):

            all_shortest_paths = list(nx.all_simple_paths(self.graph, n1, n2, cutoff=nx.diameter(self.graph)*2))  # prima usavo shortest

            all_shortest_paths = sorted(all_shortest_paths, key=lambda item: len(item))
            # switch to edges
            for i in range(len(all_shortest_paths)):
              element = all_shortest_paths[i]
              element = [(i,j) for i,j in zip(element[:-1], element[1:])]
              element = list(map(lambda x: self.graph.get_edge_data(x[0], x[1])['ID'], element))
              all_shortest_paths[i] = element
            
            if len(all_shortest_paths) > self.k:
              self.first_k_shortest_paths[str(n1) + ':' + str(n2)] = all_shortest_paths[:self.k]
            else:
              self.first_k_shortest_paths[str(n1) + ':' + str(n2)] = all_shortest_paths
            # voglio iterare sui first k shortest path e incrementare di 1 il num_shortest_paths degli edge che li compongono
            for path in self.first_k_shortest_paths[str(n1) + ':' + str(n2)]:
              for edge in path:
                num_shortest_path[edge] += 1

    for i,j in self.graph.edges():
      id = self.graph.get_edge_data(i, j)['ID']
      betweenness = num_shortest_path[id] / ((2.0 * self.num_nodes * (self.num_nodes - 1) * self.k) + 0.00000001)
      self.graph.get_edge_data(i, j)['betweenness'] = betweenness
      #print(betweenness)

  
    # -1 beacuse we don't consider ID
    self.num_features = len(self.graph.get_edge_data(0, 1)) - 1

    '''
    Here we define the state as the total capacity remained and the current bandwidth allocated in the current snapshot
    note that it is NOT the cumulative bandwidth up to the point we are in the episode, 
    it is just the bandwdth allocated by the current path chosen by the agent
    '''
    self.state = np.zeros((self.num_edges, 2)) # 
    self.features = np.zeros((self.num_edges, self.num_features))

    # update first column with the edge capacity
    for i, j in self.graph.edges():
      index = self.graph.get_edge_data(i, j)['ID']
      capacity = self.graph.get_edge_data(i, j)['capacity']
      self.state[index][0] = capacity
      features_ = self.graph.get_edge_data(i,j).copy()
      features_.pop('ID')
      #print(features_)
      self.features[index,:] = list(features_.values())

    # save initial state
    self.initial_state = np.copy(self.state)
    


  def make_step(self, state, action, required_capacity, source, destination): #------------------------------------------------------------------------------------------------
    self.state = np.copy(state)

    # take the path corresponding to the action
    chosen_path = self.first_k_shortest_paths[str(source) + ':' + str(destination)][action]

    # iterate over the edges of the chosen path
    for edge in chosen_path:
      # update the state decreasing the total capacity by the required capacity
      self.state[edge][0] -= required_capacity

      # if there is not enough capacity we stop 
      # this represents the end of an episode
      if self.state[edge][0] < 0:    
        self.is_episode_over = True
        return self.state, self.reward, self.is_episode_over, self.demand, self.source, self.destination # self.demand, self.source, self.destination will be discarded
    
    # by definition the reward is the capacity succesfully allocated
    self.reward = required_capacity/self.max_demand

    # if we succefully sastisfy the request we go on with the current episode
    self.is_episode_over = False
    # and we generate another demand
    self.demand = np.random.randint(1, self.max_demand)
    self.source = np.random.randint(self.num_nodes) 
    while True:
      self.destination = np.random.randint(self.num_nodes)
      if self.destination != self.source:
        break


    return self.state, self.reward, self.is_episode_over, self.demand, self.source, self.destination

  

  def reset(self, demand = None, source = None, destination = None): #-----------------------------------------------------------------------------------------------------
    
    self.state = np.copy(self.initial_state)

    if demand == None:
      self.demand = np.random.randint(1, self.max_demand)
      self.source = np.random.randint(self.num_nodes) 
      while True:
        self.destination = np.random.randint(self.num_nodes)
        if self.destination != self.source:
          break

    else:
      self.demand = demand
      self.source = source
      self.destination = destination

    return self.state, self.demand, self.source, self.destination
