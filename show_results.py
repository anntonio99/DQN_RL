import numpy as np
import matplotlib.pyplot as plt
import os

directory_path = os.path.dirname(os.path.realpath(__file__))
logs = os.path.join(directory_path, 'Logs')

#logs = r'C:\Users\ant.rocca\Desktop\Tesi\mio\external_logs\2\Logs'
# loss -----------------------------------------------------------------------------------------------------------------------------------------------
with open(os.path.join(logs, 'loss.txt'), "r") as file:
    content = file.read()
# remove last comma
if content[-1] == ',':
    content = content[:-1]
values = content.split(',')
values = [float(value) for value in values]
loss = np.array(values)

# calculate mean

reshaped_data = loss.reshape(-1, 6)  # 6 è il numero di batch
means = np.mean(reshaped_data, axis=1)
loss = means.reshape(-1)

# Plot the array with customization
plt.plot(loss)
plt.title('Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# save as image
plt.savefig(os.path.join(directory_path, 'Images', 'loss.png'))

# Show the plot
plt.show()




# log loss ---------------------------------------------------------------------------------------------------------------------------------------------

plt.plot(loss)
plt.title('Log-Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.yscale('log')

# save as image
plt.savefig(os.path.join(directory_path, 'Images', 'log_loss.png'))

# Show the plot
plt.show()

# evaluation rewards -----------------------------------------------------------------------------------------------------------------------------------

random_rewards = np.loadtxt(os.path.join(logs, 'random_agent_rewards.csv'), delimiter = ',')
shortest_path_rewards = np.loadtxt(os.path.join(logs,  'shortest_path_agent_rewards.csv'), delimiter = ',')
dqn_rewards = np.loadtxt(os.path.join(logs, 'dqn_rewards.csv'), delimiter = ',')



plt.plot(shortest_path_rewards, label='Shortest Path', color = 'r', linestyle=':')
plt.axhline(y=np.mean(shortest_path_rewards), color = 'r', linestyle=':')

plt.plot(random_rewards, label='Random', color = 'y', linestyle='--')
plt.axhline(y=np.mean(random_rewards), color = 'y', linestyle='--')

plt.plot(dqn_rewards, label='RL', color = 'b', linestyle='-')
plt.axhline(y=np.mean(dqn_rewards), color = 'b', linestyle='-')

# Adding labels and title
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Evaluation')
plt.legend()  





plt.savefig(os.path.join(directory_path, 'Images', 'rewards_other_agents.png'))

plt.show()


# box plot -----------------------------------------------------------------------------------------------------------------------------------------------------

if False:
    random_mean = np.mean(random_rewards)
    shortest_path_mean = np.mean(shortest_path_rewards)
    dqn_mean = np.mean(dqn_rewards)

    # Create a box plot
    data_to_plot = [random_rewards, shortest_path_rewards, dqn_rewards]
    mean_values = [random_mean, shortest_path_mean, dqn_mean]

    plt.boxplot(data_to_plot, labels=['Random Agent', 'Shortest Path Agent', 'DQN Agent'])
    plt.scatter(np.arange(1, 4), mean_values, color='red', marker='o', label='Mean')

    # Adding labels and title
    plt.xlabel('Agents')
    plt.ylabel('Rewards')
    plt.title('Box Plot of Mean Rewards for Three Agents')
    plt.legend()

    # Show the plot
    plt.show()


# Train rewards ----------------------------------------------------------------------------------------------------------------------------------------------


with open(os.path.join(logs, 'mean_reward.txt'), "r") as file:
    content = file.read()
# remove last comma
if content[-1] == ',':
    content = content[:-1]
values = content.split(',')
values = [float(value) for value in values]
rewards = np.array(values)

with open(os.path.join(logs, 'train_info.txt'), 'r') as file:
    # Read the content of the file
        content = file.read()
        values = content.split(',')
        number_iterations = int(values[0]) + 1


plt.plot(np.linspace(0, number_iterations, len(rewards)), rewards)
plt.title('Mean Reward')
plt.xlabel('Iterations')
plt.ylabel('Reward')


# save
plt.savefig(os.path.join(directory_path, 'Images', 'reward.png'))
# Show the plot
plt.show()