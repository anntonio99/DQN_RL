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
import matplotlib.pyplot as plt


#g = create_simple_graph_2()
g = create_complex_graph()

env = Environment(graph = g, k = 4, list_of_possible_demands = [8, 32, 64])
env.generate_environment()

#agent = Agent(env, epsilon=0)



NUMBER_OF_EPISODES = 40


# load model-------------------------------------------------------------------------------------------------------------------------------------------------------------

agent = Agent(env)
#checkpoint_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'external_logs', '2500')
checkpoint_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Logs', 'Models')
checkpoint = tf.train.Checkpoint(model=agent.q_network, secondary_network=agent.target_network)
model_file = glob.glob(os.path.join(checkpoint_directory, 'ckpt-*.index'))[0][:-6]
checkpoint.restore(model_file)


# test ------------------------------------------------------------------------------------------------------------------------------------------------------------------

cumulative_rewards = np.zeros(NUMBER_OF_EPISODES)

for i in range(NUMBER_OF_EPISODES):
    cumulative_reward = 0
    state, demand, source, destination = env.reset()

    while 1:
        action, corresponding_features = agent.act(env, state, demand, source, destination, True)

        new_state, reward, done, new_demand, new_source, new_destination = env.make_step(state, action, demand, source, destination)

        state = new_state
        demand = new_demand
        source = new_source
        destination = new_destination
        cumulative_reward += reward
        if done:
            break
    cumulative_rewards[i] = cumulative_reward


print('RL')
print(np.mean(cumulative_rewards))
print(np.std(cumulative_rewards))
#plt.plot(cumulative_rewards)
#plt.show()


# random agent ---------------------------------------------------------------------------------------------------------------------------------------------------------------------


cumulative_rewards = np.zeros(NUMBER_OF_EPISODES)

for i in range(NUMBER_OF_EPISODES):
    cumulative_reward = 0
    state, demand, source, destination = env.reset()

    while 1:
        action = np.random.randint(len(env.first_k_shortest_paths[str(source) + ':' + str(destination)]))

        new_state, reward, done, new_demand, new_source, new_destination = env.make_step(state, action, demand, source, destination)

        state = new_state
        demand = new_demand
        source = new_source
        destination = new_destination
        cumulative_reward += reward
        if done:
            break
    cumulative_rewards[i] = cumulative_reward

print('Random')
print(np.mean(cumulative_rewards))
print(np.std(cumulative_rewards))
#plt.plot(cumulative_rewards)
#plt.show()