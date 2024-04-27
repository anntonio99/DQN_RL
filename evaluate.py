import numpy as np
from other_agents.random_agent import random_Agent
from other_agents.shortest_path_agent import shortest_path_Agent
from Agent import Agent
from Environment import Environment
from utils import create_complex_graph
from utils import generate_random_graph
import os
import glob
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import random
import networkx as nx
import matplotlib.pyplot as plt


g = create_complex_graph()



NUMBER_OF_EPISODES = 30
NUMBER_OF_DEMANDS_PER_EPISODE = 200
SEED = 9



random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(1)
tf.keras.utils.set_random_seed(1)
os.environ['PYTHONHASHSEED']=str(SEED)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


#g = generate_random_graph(32)
#nx.draw(g)
#plt.show()


max_demand = 64

def evaluate_agent(agent, environment, episodes):

    cumulative_rewards = np.zeros(NUMBER_OF_EPISODES)


    for episode_index, episode in enumerate(episodes):
        for i in range(len(episode)):

            demand = episode[i][0]
            source = episode[i][1]
            destination = episode[i][2]

            # if it is a new episode reset the state
            if i == 0:
                state, _, _, _ = environment.reset(demand, source, destination)
            
            # allocate the demand
            action, _ = agent.act(environment, state, demand, source, destination, True)  
            new_state, reward, done, _, _, _ = environment.make_step(state, action, demand, source, destination)
            cumulative_rewards[episode_index] += reward
            state = new_state

            if done:
                break


    return cumulative_rewards


# define the environment

environment = Environment(graph = g, k = 4, max_demand=max_demand)
#shortest_path_environment = Environment(graph = g, k = 4, list_of_possible_demands = list_of_demands)
#dqn_environment = Environment(graph = g, k = 4, list_of_possible_demands = list_of_demands)

# generate the environments

environment.generate_environment()
#shortest_path_environment.generate_environment()
#dqn_environment.generate_environment()


# define the agents
            
random_agent = random_Agent(environment)
random_agent.set_seed(SEED)
shortest_path_agent = shortest_path_Agent(environment)

# load the model
dqn_agent = Agent(environment)

checkpoint_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'external_logs', 'Logs_2', 'Models')
#checkpoint_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Logs', 'Models')
checkpoint = tf.train.Checkpoint(model=dqn_agent.q_network)
model_file = glob.glob(os.path.join(checkpoint_directory, 'ckpt-*'))[1][:-6]

checkpoint.restore(model_file)

#checkpoint.restore(r'C:\Users\ant.rocca\Desktop\Tesi\Codice\Logs\Models\ckpt-1.index')


# create episodes

test = []
for episode in range(NUMBER_OF_EPISODES):
    requests = []
    for i in range(NUMBER_OF_DEMANDS_PER_EPISODE):
        # generate demand
        demand = np.random.randint(1, max_demand)
        source = np.random.choice(environment.num_nodes)

        while True:
            destination = np.random.choice(environment.num_nodes)
            if destination != source:
                requests.append((demand, source, destination))
                break
    test.append(requests)


random_rewards = evaluate_agent(random_agent, environment, test)

shortest_path_rewards = evaluate_agent(shortest_path_agent, environment, test)

dqn_rewards = evaluate_agent(dqn_agent, environment, test)



# save

directory_path = os.path.dirname(os.path.realpath(__file__))
logs = os.path.join(directory_path, 'Logs')


np.savetxt(os.path.join(logs, 'random_agent_rewards.csv'), random_rewards, delimiter=',')
np.savetxt(os.path.join(logs, 'shortest_path_agent_rewards.csv'), shortest_path_rewards, delimiter=',')
np.savetxt(os.path.join(logs, 'dqn_rewards.csv'), dqn_rewards, delimiter=',')





'''

# print(tf.keras.backend.eval(dqn_agent.q_network.layers[0].weights[0]))



for layer in dqn_agent.target_network.layers:
  print(layer.name)
  print(layer.weights)
'''