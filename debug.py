import os
import glob
import pickle
from pprint import pprint
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
from Environment import Environment
from Agent import Agent
from utils import create_simple_graph
from utils import create_complex_graph
from utils import create_simple_graph_2


g = create_simple_graph_2()
# g = create_complex_graph()

env = Environment(graph = g, k = 4, list_of_possible_demands = [8, 32, 64])
env.generate_environment()

agent = Agent(env, epsilon=0)


state, demand, source, destination = env.reset()
i = 0
while 1:
    i += 1
    action, corresponding_features = agent.act(env, state, demand, source, destination, False)

    new_state, reward, done, new_demand, new_source, new_destination = env.make_step(state, action, demand, source, destination)

    agent.add_sample_to_memory(env, corresponding_features, action, reward, done, new_state, new_demand, new_source, new_destination)
    state = new_state
    demand = new_demand
    source = new_source
    destination = new_destination
    if done:
        break


#print(i)

#pprint(env.edges_topology)
#pprint(env.first_k_shortest_paths)



