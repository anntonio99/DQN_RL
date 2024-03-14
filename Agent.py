import numpy as np
from Neural_Network import MessagePassingNN as mpnn
import tensorflow as tf
from collections import deque
import random
import os
from pprint import pprint

directory_path = os.path.dirname(os.path.realpath(__file__))
logs = os.path.join(directory_path, 'Logs')

# constants
MAX_QUEUE_SIZE = 4000
MULTI_FACTOR_BATCH = 6 # Number of batches used in training

copy_weights_interval = 50




class Agent():
    def __init__(self, environment, epsilon = 1):
      self.hyperparamters = {
         'l2': 0.1,
         'dropout_rate': 0.01,
         'hidden_dim': 20,
         'readout_units': 35,
         'learning_rate': 0.0001,
         'batch_size': 32,
         'T': 4, 
         'num_demands': len(environment.list_of_possible_demands)
}
      self.memory = deque(maxlen=MAX_QUEUE_SIZE)
      self.epsilon = epsilon  # probability of taking random action
      self.epsilon_min = 0.01
      self.epsilon_decay = 0.995
      self.gamma = 0.95
      self.k = environment.k
      self.q_network = mpnn(self.hyperparamters)  # primary network
      self.q_network.build()
      self.target_network = mpnn(self.hyperparamters)  # secondary network
      self.target_network.build()
      self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.hyperparamters['learning_rate'],
                                               momentum=0.9,
                                               nesterov=True)
      self.environment = environment

    def _get_graph_ids(self, num_edges, num_graphs):  #  mi sa non ti serve 
       l = []
       for i in range(num_graphs):
          l.append(tf.fill((num_edges,), i))
       ids = tf.concat(l, axis=0)
       return ids
       

    def act(self, environment, state, demand, source, destination, evaluate):
        
        take_random_action = False

        # first k shortest paths between source and destination
        first_k_shortest_paths = environment.first_k_shortest_paths[str(source) + ':' + str(destination)]
        path = 0
  
        if evaluate == False:
        # if evaluate == False we are training so with probability epsilon we want to take a random action
          a = np.random.random()
          if a < self.epsilon:
            take_random_action = True
            # take random action
            path = np.random.randint(0, len(first_k_shortest_paths))
        
  
        edges_topology = environment.edges_topology 
        shift = np.max(edges_topology, axis=1)[:, np.newaxis] + 1 
        

        # compute q-value for all k paths and take the maximum
        graphs_features = []
        edges_topologies = []

        # iterate over all k paths, allocate the demand, extract the corresponding features
  
        while path < len(first_k_shortest_paths):  # there may be less than k paths!
          state_copy = np.copy(state)
          # allocate the demand on all path edges
          path_edges = first_k_shortest_paths[path]
          state_copy[path_edges,1] = demand
          features = self.get_graph_features(environment, state_copy)
          graphs_features.append(features)
          edges_topologies.append(edges_topology + path*shift) 
          if take_random_action == True:
             return path, features
          path += 1

        # prepare other inputs for the neural network
        graphs_ids = tf.concat([[i]*environment.num_edges for i in range(len(first_k_shortest_paths))], axis=0) # shape: 1 x (num_edges x k)
        features = tf.concat(graphs_features, axis=0)  # shape: (num_edges x k) x num_features
        edges_topologies = tf.concat(edges_topologies, axis=1)  # shape  2 x (k x num_edge_topology)

        list_q_values = self.q_network(features = features,
                                       graph_ids = graphs_ids,
                                       edges_topology = edges_topologies)

      
        action = np.argmax(list_q_values)
        features_correspoding_to_action = graphs_features[action]
        # return 
        # action: a number between 0 and k, corresponding to the chosen path
        # the corresponing features of the resulting graph
        return action, features_correspoding_to_action
    
    def get_graph_features(self, environment, state_copy):
       # get the graph features and process them to feed them into the neural network
       # do not use the state from the environment, use state_copy
        
        capacity_feature = np.array([data['capacity'] for _, _, data in environment.graph.edges(data=True)])
        betweeneess_feature = np.array([data['betweenness'] for _, _, data in environment.graph.edges(data=True)])

        # normalizzazione che fa lui, capisci come generalizzarla
        capacity_feature = (capacity_feature - 100.)/200.

        demands = environment.list_of_possible_demands
        bw_allocated_feature = np.zeros((environment.num_edges, len(demands)))
        iter = 0
        for i in state_copy[:, 1]: 
            if i == demands[0]:
                bw_allocated_feature[iter][0] = 1
            elif i == demands[1]:
                bw_allocated_feature[iter][1] = 1
            elif i == demands[2]:
                bw_allocated_feature[iter][2] = 1
            iter = iter + 1
        
        # convert everything to tensor
        bw_allocated_feature = tf.convert_to_tensor(bw_allocated_feature, dtype=tf.float32)
        capacity_feature = tf.convert_to_tensor(capacity_feature, dtype=tf.float32)
        betweeneess_feature = tf.convert_to_tensor(betweeneess_feature, dtype=tf.float32)

        padding = self.hyperparamters['hidden_dim'] - len(demands) - environment.num_features + 1 # in num_features there is also bw_allocated, so we add 1
        padding = np.zeros((environment.num_edges, padding))

        features = tf.concat([tf.reshape(capacity_feature, (environment.num_edges, 1)),
                              tf.reshape(betweeneess_feature, (environment.num_edges, 1)),   # reshape to go from (num_edges,) to (num_edges, 1) for the concatenation
                              bw_allocated_feature, 
                              padding
                              ], 
                              axis=1)
      
        return features
    
    def _forward_pass(self, features_s, graph_id_s, edges_topology_s, features_s_prime, graph_id_s_prime, edges_topology_s_prime):
      # s is the previous state, s' is the one we end up when taking the action
      # primary network
      prediction_state = self.q_network(features_s,
                                        graph_id_s,
                                        edges_topology_s,
                                        training=True)
      # secondary network (we don't compute gradient for it)a
      preds_next_target = tf.stop_gradient(self.target_network(features_s_prime,
                                                               graph_id_s_prime,
                                                               edges_topology_s_prime,
                                                               training=True))
      return prediction_state, preds_next_target
    
    def _train_step(self, batch):

      # every x in batch is:
      # state S
      # x[0] features
      # x[1] graph_id (it's just all 0s)
      # x[2] edges_topology
      #
      # x[3] reward
      # x[4] done
      #
      # state S'
      # x[5] features
      # x[6] graph_id
      # x[7] edges_topology

      with tf.GradientTape() as tape:
        preds_state = []
        target = []
        for x in batch:

          prediction_state, preds_next_target = self._forward_pass(x[0], x[1], x[2], x[5], x[6], x[7])

          # Take q-value of the action performed
          preds_state.append(prediction_state[0]) # we take component 0 because it's the only one since there is only one graph
          # We multiple by 0 if done==TRUE to cancel the second term
          reward = x[3] 
          done = x[4] 

          target.append(tf.stop_gradient([reward + self.gamma*tf.math.reduce_max(preds_next_target)*(1-done)])) # perche appendo una lista??



        loss = tf.keras.losses.MSE(tf.stack(target, axis=1), tf.stack(preds_state, axis=1))
        # Loss function using L2 Regularization
        regularization_loss = sum(self.q_network.losses)
        loss = loss + regularization_loss


      # Computes the gradient using operations recorded in context of this tape
      grad = tape.gradient(loss, self.q_network.trainable_variables)

      #gradients, _ = tf.clip_by_global_norm(grad, 5.0)

      #pprint(self.q_network.trainable_variables)
      #pprint(self.q_network.variables)

      grad = [tf.clip_by_value(gradient, -1., 1.) for gradient in grad]
      self.optimizer.apply_gradients(zip(grad, self.q_network.trainable_variables))
      del tape
      return grad, loss
    
    


    def replay(self, iteration):
        for i in range(MULTI_FACTOR_BATCH):
            batch = random.sample(self.memory, self.hyperparamters['batch_size']) # è senza replacement, potrebbe essere meglio con?
            
            grad, loss = self._train_step(batch)

            # save
            with open(os.path.join(logs, 'loss.txt'), "a") as file:
              file.write('%.9f' % loss.numpy() + ",")

        # Hard weights update
        if iteration % copy_weights_interval == 0:
            self.target_network.set_weights(self.q_network.get_weights()) 



    def add_sample_to_memory(self, environment, features_s, action, reward, done, new_state, new_demand, new_source, new_destination):
       # state S
       graph_id_s = tf.fill([tf.shape(features_s)[0]], 0)
       edges_topology_s = environment.edges_topology

       # state S'
       first_k_shortest_paths = self.environment.first_k_shortest_paths[str(new_source) + ':' + str(new_destination)]
       edges_topology = environment.edges_topology
       shift = np.max(edges_topology, axis=1)[:, np.newaxis] + 1 

       graphs_features_s_prime = []
       edges_topologies_s_prime = []
       # iterate over all k paths, allocate the demand, extract the corresponding features
       for i, path_edges in enumerate(first_k_shortest_paths):
          state_copy = np.copy(new_state)
          # allocate the demand on all path edges
          state_copy[path_edges,1] = new_demand 
          features = self.get_graph_features(self.environment, state_copy)
          graphs_features_s_prime.append(features)
          edges_topologies_s_prime.append(edges_topology + i*shift) # se non va qualcosa controlla questo

        # prepare other inputs for the neural network
       features_s_prime = tf.concat(graphs_features_s_prime, axis=0)  # shape: (num_edges x k) x num_features
       graphs_ids_s_prime = tf.concat([[i]*self.environment.num_edges for i in range(len(first_k_shortest_paths))], axis=0) # shape: 1 x (num_edges x k)
       edges_topologies_s_prime = tf.concat(edges_topologies_s_prime, axis=0)  # shape  (2 x k) x num_edge_topology


       self.memory.append((
                            # S
                            features_s, 
                            graph_id_s, 
                            edges_topology_s,
                            # reward
                            tf.convert_to_tensor(reward, dtype=tf.float32), 
                            # done
                            tf.convert_to_tensor(int(done==True), dtype=tf.float32),
                            # S'
                            features_s_prime, 
                            graphs_ids_s_prime,
                            edges_topologies_s_prime
        ))



