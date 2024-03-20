RESUME = False

import os
import glob
import pickle
import random
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
from Environment import Environment
from Agent import Agent
from utils import create_simple_graph
from utils import create_complex_graph
import shutil
import time
from pprint import pprint


ITERATIONS = 10000
TRAINING_EPISODES = 20
EVALUATION_EPISODES = 40
FIRST_WORK_TRAIN_EPISODE = 60
SEED = 37


# set global determinism
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(1)
tf.keras.utils.set_random_seed(1)
os.environ['PYTHONHASHSEED']=str(SEED)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

print_train_every = ITERATIONS//10

evaluate_every = 10 
epsilon_start_decay = 70

directory_path = os.path.dirname(os.path.realpath(__file__))
logs = os.path.join(directory_path, 'Logs')


os.makedirs(logs, exist_ok=True)



list_of_possible_demands = [8, 32, 64]

g = create_complex_graph()

k = 4

# define environments
env_training = Environment(graph = g, k = k, list_of_possible_demands = list_of_possible_demands)
env_eval = Environment(graph = g, k = k, list_of_possible_demands = list_of_possible_demands)

# set seeds
env_training.set_seed(SEED)
env_eval.set_seed(SEED)

# generate environments
env_training.generate_environment()
env_eval.generate_environment()


# define agent
agent = Agent(env_training)
# set seed
agent.set_seed(SEED)

'''
occhio perchè inizializza solo i pesi del layer recursive update
capire sta cosa

for layer in agent.target_network.layers:
  print(layer.name)
  print(layer.weights)

'''




checkpoint = tf.train.Checkpoint(model=agent.q_network) # se vuoi fare il resume ti devi salvare anche la target network
manager = tf.train.CheckpointManager(checkpoint,
                                     directory=os.path.join(logs, 'Models'), 
                                     max_to_keep=1)



if RESUME == True: # resume training
    #model_file = glob.glob(os.path.join(logs, 'Models', 'ckpt-*.index'))[0][:-6]
    checkpoint_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'external_logs', '2500')
    model_file = glob.glob(os.path.join(checkpoint_directory, 'ckpt-*.index'))[0][:-6]
    checkpoint.restore(model_file)
    '''
    with open(os.path.join(logs, 'train_info.txt'), 'r') as file:
    # Read the content of the file
        content = file.read()
        values = content.split(',')
        iteration_resume = int(values[0])
        agent.epsilon = float(values[1])

    with open(os.path.join(logs, 'memory.pkl'), 'rb') as file:
        agent.memory = pickle.load(file)
    '''
    iteration_resume = 10000
    agent.epsilon = agent.epsilon_min
    shutil.rmtree(logs)
    os.makedirs(logs)

else: # start training from scratch
     
     shutil.rmtree(logs)
     os.makedirs(logs)
     iteration_resume = 0



# TRAIN ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

print('\nStart of training...\n')

start_time = time.time()

max_reward = 0

for training_iteration in range(iteration_resume, ITERATIONS):
        if training_iteration % print_train_every == 0:
            print("Training iteration: ", training_iteration, '/', ITERATIONS)


        if training_iteration==0:
            train_episodes = FIRST_WORK_TRAIN_EPISODE
        else:
            train_episodes = TRAINING_EPISODES

        # EPISODES ------------------------------------------------------------------
        for episode in range(TRAINING_EPISODES):
            tf.random.set_seed(1)
            
            state, demand, source, destination = env_training.reset()            
            

            while 1:
                # The agent chooses an action and returns it along with the features coresponding to the resulting state 
                action, state_action = agent.act(env_training, state, demand, source, destination, False)
                new_state, reward, done, new_demand, new_source, new_destination = env_training.make_step(state, action, demand, source, destination)

                agent.add_sample_to_memory(env_training, state_action, action, reward, done, new_state, new_demand, new_source, new_destination)
                state = new_state
                demand = new_demand
                source = new_source
                destination = new_destination
                if done:
                    break
        '''
        print(demand)
        print(source)
        print(destination)
        '''
        # LEARNING ------------------------------------------------------------------
        agent.replay(training_iteration)

        # EPSILON DECAY ------------------------------------------------------------------
        if training_iteration > epsilon_start_decay and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay 
            agent.epsilon *= agent.epsilon_decay

        # EVALUATE MODEL ------------------------------------------------------------------
        if training_iteration % evaluate_every == 0:
           
            cumulative_rewards = np.zeros(EVALUATION_EPISODES)
            for eps in range(EVALUATION_EPISODES):
                state, demand, source, destination = env_eval.reset()
                cumulative_reward = 0
                while 1:

                    action, _ = agent.act(env_eval, state, demand, source, destination, True)  
                    state, reward, done, demand, source, destination = env_eval.make_step(state, action, demand, source, destination)  
                    cumulative_reward = cumulative_reward + reward
                    if done:
                        break
                cumulative_rewards[eps] = cumulative_reward
            mean_reward = np.mean(cumulative_rewards)
            '''
            print(demand)
            print(source)
            print(destination)
            '''
            if mean_reward > max_reward:
                max_reward = mean_reward
                # save model
                manager.save()

            # save memory
            '''
            with open(os.path.join(logs, 'memory.pkl'), 'wb') as file:
                pickle.dump(agent.memory, file)
            '''

            # save mean reward
            with open(os.path.join(logs, 'mean_reward.txt') , "a") as file:
                file.write('%.3f' % mean_reward + ",")

            # save training iteration and epsilon
            with open(os.path.join(logs, 'train_info.txt'), "w") as file:
                file.write(str(training_iteration) + ',' + str(agent.epsilon))


print('\nEnd of training\n')


end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes = remainder//60

print(f"Model trained in: {int(hours)} hours and {int(minutes)} minutes")



'''
for layer in agent.q_network.layers:
  print(layer.name)
  print(layer.weights)
'''

'''
randomicità
- generazione source, destination, demand
- probabilità epsilon 
- scelta cammino random
- inizializzazione pesi reti neurali
'''