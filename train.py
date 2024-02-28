RESUME = False

import os
import glob
import pickle
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

ITERATIONS = 10
TRAINING_EPISODES = 20
EVALUATION_EPISODES = 40
FIRST_WORK_TRAIN_EPISODE = 60

print_train_every = ITERATIONS//10

evaluation_interval = 1 # evaluate model every ... interations
epsilon_start_decay = 70

directory_path = os.path.dirname(os.path.realpath(__file__))
logs = os.path.join(directory_path, 'Logs')

if not os.path.exists(logs):
    os.makedirs(logs)



list_of_possible_demands = [8, 32, 64]

g = create_complex_graph()

env_training = Environment(graph = g, k = 4, list_of_possible_demands = list_of_possible_demands)
env_eval = Environment(graph = g, k = 4, list_of_possible_demands = list_of_possible_demands)

env_training.generate_environment()

env_eval.generate_environment()



agent = Agent(env_training)


checkpoint = tf.train.Checkpoint(model=agent.q_network, optimizer=agent.optimizer)
manager = tf.train.CheckpointManager(checkpoint,
                                     directory=os.path.join(logs, 'Models'), 
                                     max_to_keep=1)



if RESUME == True: # resume training
    model_file = glob.glob(os.path.join(logs, 'Models', 'ckpt-*.index'))[0][:-6]
    checkpoint.restore(model_file)
    with open(os.path.join(logs, 'train_info.txt'), 'r') as file:
    # Read the content of the file
        content = file.read()
        values = content.split(',')
        iteration_resume = int(values[0])
        agent.epsilon = float(values[1])

    with open(os.path.join(logs, 'memory.pkl'), 'rb') as file:
        agent.memory = pickle.load(file)

else: # start training from scratch

    # # empty the Logs foler
     files = glob.glob(os.path.join(logs, '*'))
    #for f in files:
        #os.remove(f)
     shutil.rmtree(logs)
     os.makedirs(logs)
     iteration_resume = 0


# TRAIN ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

print('\nStart of training...\n')

start_time = time.time()

for ep_it in range(iteration_resume, ITERATIONS):
        if ep_it % print_train_every == 0:
            print("Training iteration: ", ep_it, '/', ITERATIONS)


        if ep_it==0:
            train_episodes = FIRST_WORK_TRAIN_EPISODE
        else:
            train_episodes = TRAINING_EPISODES

        # EPISODES ------------------------------------------------------------------
        for training_episode in range(train_episodes):
            
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

        # LEARNING ------------------------------------------------------------------
        agent.replay(ep_it)

        # EPSILON DECAY ------------------------------------------------------------------
        # Decrease epsilon (from epsion-greedy exploration strategy)
        if ep_it > epsilon_start_decay and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon *= agent.epsilon_decay

        # EVALUATE MODEL ------------------------------------------------------------------
        # We only evaluate the model every evaluation_interval steps
        if ep_it % evaluation_interval == 0:
            cumulative_rewards = np.zeros(EVALUATION_EPISODES)
            for eps in range(EVALUATION_EPISODES):
                state, demand, source, destination = env_eval.reset()
                cumulative_reward = 0
                while 1:
                    # We execute evaluation over current state

                    action, _ = agent.act(env_eval, state, demand, source, destination, True)  # forse dovresti passargli l'environment in input

                    
                    state, reward, done, demand, source, destination = env_eval.make_step(state, action, demand, source, destination)  # forse dovresti passargli anche lo state
                    cumulative_reward = cumulative_reward + reward
                    if done:
                        break
                cumulative_rewards[eps] = cumulative_reward
            mean_reward = np.mean(cumulative_rewards)
            
            # save model 
            manager.save()
            # save mean reward
            with open(os.path.join(logs, 'mean_reward.txt') , "a") as file:
              file.write('%.3f' % mean_reward + ",")

            # save also the training iteration and epsilon
            with open(os.path.join(logs, 'train_info.txt'), "w") as file:
              file.write(str(ep_it) + ',' + str(agent.epsilon))

            # save memory
            pickle.dump(agent.memory, open(os.path.join(logs, 'memory.pkl'), 'wb'))



print('\nEnd of training\n')


end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes = remainder//60

print(f"Model trained in: {int(hours)} hours and {int(minutes)} minutes")