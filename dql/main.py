import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import numpy as np
import gym
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
    def __init__(self, learning_rate: float, input_dims: list[int], fc1_dims: int, fc2_dims: int, n_actions: int):
        super(DeepQNetwork, self).__init__()
        self.input_dims: list[int] = input_dims
        self.fc1_dims: int = fc1_dims
        self.fc2_dims: int = fc2_dims
        self.n_actions: int = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # passing the input to the first layer
        x: torch.Tensor = functional.relu(self.fc1(state))
        # passing the output from first layer to the second layer
        x: torch.Tensor = functional.relu(self.fc2(x))

        # passing the output from the second layer into the action layer
        actions: torch.Tensor = functional.relu(self.fc3(x))

        return actions







class Agent:
    # gamma - determines the weighting of future rewards
    # epsilon - determines how often does the agent spend exploring its environment vs taking the best known action
    # batch_size - learning from of batch of memory
    def __init__(self, gamma: float, epsilon: float, learning_rate: float, input_dims: list[int], batch_size: int,
                 num_actions: int, max_mem_size: int = 100_000, epsilon_end: float = 0.01,
                 epsilon_decrement: float = 5e-5):
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_end
        self.epsilon_dec: float = epsilon_decrement
        self.learning_rate: float = learning_rate
        # saves the actions that the agent can take
        self.action_space: list[int] = [i for i in range(num_actions)]
        self.mem_size: int = max_mem_size
        self.batch_size: int = batch_size
        # to keep track of the position of the first available memory for string the agent's memory
        self.mem_cntr: int = 0

        self.Q_eval = DeepQNetwork(self.learning_rate, n_actions=num_actions, input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)

        self.state_memory: np.ndarray = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        # memory of the states that resulted from agent's actions
        # we are going to get the value of each action given the current state based on the earlier estimates
        # we use one estimate to update another
        self.new_state_memory: np.ndarray = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory: np.ndarray = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory: np.ndarray = np.zeros(self.mem_size, dtype=np.float32)

        # used to store the last few observations that the agent experiences before the end of an episode
        # then we can update the estimates of the Q value
        self.terminal_memory: np.ndarray = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state: gym.core.ObsType, action: int, reward, new_state: gym.core.ObsType,
                         done: bool) -> None:
        index: int = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation: gym.core.ObsType) -> int:
        # if random is greater, then take best known action
        if np.random.random() > self.epsilon:
            state: torch.Tensor = torch.tensor([observation]).to(self.Q_eval.device)
            actions: torch.Tensor = self.Q_eval.forward(state)
            action: int = torch.argmax(actions).item()
        # else take a random action from action space
        else:
            action: int = np.random.choice(self.action_space)

        return action

    def learn(self) -> None:
        # here we can either choose if the agent will start learning when the whole memory is filled up
        # or when a memory batch is filled up
        # we choose the batch because it would more efficient

        # if the batch is not filled up don't learn
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem: int = min(self.mem_cntr, self.mem_size)
        # False so we don't select the same memory twice
        batch: np.ndarray = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index: np.ndarray = np.arange(self.batch_size, dtype=np.int32)

        # converting a numpy array subset of out agent's memory into a pytorch sensor
        # tensor is multidimensional array, which is a fundamental data structure used
        # for building and training neural networks.
        state_batch: torch.Tensor = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)

        # we do the same thing for the new states
        new_state_batch: torch.Tensor = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)

        reward_batch: torch.Tensor = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)

        terminal_batch: torch.Tensor = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch: np.ndarray = self.action_memory[batch]

        # performs feed forwards through our deep neural network to get
        # relevant parameters for out loss function

        # we want to be moving the agents estimate for the value of the current state
        # towards the maximal value for the next state
        # in simple words, nudging it towards selecting maximal actions

        # the reason for [batch_index, action_batch] is that we want to get the values of the actions we took
        # we can't update the values of the actions that we didn't take

        q_eval: torch.Tensor = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next: torch.Tensor = self.Q_eval.forward(new_state_batch)

        q_next_clone: torch.Tensor = q_next.clone()
        q_next_clone[terminal_batch] = 0.0

        # torch.max returns a tuple (maximum value for the next state) and we want the first value that's why we use 0
        q_target: float = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss: torch.Tensor = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        torch.autograd.set_detect_anomaly(True)
        # measures how much each connections contributes to the overall solution using back propagation
        loss.backward()

        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min


if __name__ == "__main__":
    env: gym.Env = gym.make("QWOP")
    # env.reset()
    # env.render()

    agent: Agent = Agent(gamma=0.97, epsilon=1.0, batch_size=64, num_actions=4, epsilon_end=0.01, input_dims=[24],
                         learning_rate=0.003)
    scores: list[int] = []
    eps_history: list[float] = []
    n_games: int = 500

    for i in range(n_games):
        score: int = 0
        done: bool = False
        observation: gym.core.ObsType = env.reset()[0]  # for qwop it will be the positions of its limbs

        while not done:
            action: int = agent.choose_action(observation)
            # what we get for taking this action
            env_step: tuple[gym.core.ObsType, float, bool, bool, dict] = env.step(action)
            # the next state of the environment after taking the action, represented as an array of numbers
            next_observation: gym.core.ObsType = env_step[0]
            # the reward obtained by the agent for taking the action in the current state
            reward: float = env_step[1]
            # boolean variable that indicates whether the episode has terminated or not
            done: bool = env_step[2]

            info: dict = env_step[4]

            score += reward
            agent.store_transition(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation

        scores.append(score)
        eps_history.append(agent.epsilon)

        # scores of last 100 games to see if our agent is learning
        avg_score: np.ndarray = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    env.close()
