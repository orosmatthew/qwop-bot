import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id="QWOP",
    entry_point="dql.gym_qwop.envs:CustomEnv",
    max_episode_steps=2000
)


# from simple_dqn_torch_2020 import Agent
# from utils import plotLearning


class DeepQNetwork(nn.Module):
    # fc - stands for fully connected layer
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # passing the input to the first layer
        x = F.relu(self.fc1(state))

        # passing the output from first layer to the second layer
        x = F.relu(self.fc2(x))

        # passing the output from the second layer into the action layer
        actions = F.relu(self.fc3(x))

        return actions


class Agent():
    # gamma - determines the weighting of future rewards
    # epsilon - determines how often does the agent spend exploring its environment vs taking the best known action
    # batch_size - learning from of batch of memory
    def __init__(self, gamma, epsilon, learning_rate, input_dims, batch_size, num_actions,
                 max_mem_size=100_000, epsilon_end=0.01, epsilon_decrement=5e-5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_dec = epsilon_decrement
        self.learning_rate = learning_rate
        # saves the actions that the agent can take
        self.action_space = [i for i in range(num_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        # to keep track of the position of the first available memory for string the agent's memory
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.learning_rate, n_actions=num_actions, input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        # memory of the states that resulted from agent's actions
        # we gonna get the value of each action given the current state based on the earlier estimates
        # we use one estimate to update another
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        # used to store the last few observations that the agent experiences before the end of an episode
        # so we can update the estimates of the Q value
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state[0]
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        # if random is greater take a best known action
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        # else take a random action from action space
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        # here we can either choose if the agent will start learning when the whole memory is filled up
        # or when a memory batch is filled up
        # we choose the batch because it would more efficient

        # if the batch is not filled up don't learn
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        # False so we dont select the same memory twice
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # converting a numpy array subset of out agent's memory into a pytorch sensor
        # tensor is multidimensional array, which is a fundamental data structure used
        # for building and training neural networks.
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)

        # we do the same thing for the new states
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)

        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)

        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        # performs feed forwards through our deep neural network to get
        # relevant parameters for out loss function

        # we want to be moving the agents estimate for the value of the current state
        # towards the maximal value for the next state
        # in simple words, nudging it towards selecting maximal actions

        # the reason for [batch_index, action_batch] is that we want to get the values of the actions we took
        # we can't update the values of the actions that we didn't take

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)

        q_next_clone = q_next.clone()
        q_next_clone[terminal_batch] = 0.0


        # T.max returns a tuple (maximum value for the next state) and we want the first value that's why we use 0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        T.autograd.set_detect_anomaly(True)
        # TODO: fix this piece of pie
        # measures how much each connections contributes to the overall solution using back propagation
        loss.backward()

        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min \
            else self.epsilon_min


if __name__ == "__main__":
    env = gym.make("QWOP")
    agent = Agent(gamma=0.97, epsilon=1.0, batch_size=64, num_actions=4, epsilon_end=0.01, input_dims=[24],
                  learning_rate=0.003)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()[0] #for qwop it will be the positions of its limbs

        while not done:

            action = agent.choose_action(observation)
            # what we get for taking this action
            env_step = env.step(action)
            # the next state of the environment after taking the action, represented as an array of numbers
            next_observation = env_step[0]
            # the reward obtained by the agent for taking the action in the current state
            reward = env_step[1]
            # boolean variable that indicates whether the episode has terminated or not
            done = env_step[2]

            info = env_step[3]

            score += reward
            agent.store_transition(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation

        scores.append(score)
        eps_history.append(agent.epsilon)

        # scores of last 100 games to see if our agent is learning
        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    env.close()

