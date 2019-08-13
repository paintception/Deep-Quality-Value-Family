from utils import Utils
from collections import deque
import numpy as np
import random


class ddqnAgent(object):
    def __init__(self, nb_actions):
        self.target_model = True
        self.nb_actions = nb_actions
        self.render = True
        self.state_size = (84, 84, 4)
        self.action_size = 3
        self.epsilon = 1
        self.epsilon_start, self.epsilon_end = 1, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (
            self.epsilon_start - self.epsilon_end) / self.exploration_steps

        self.clip = (-1, 1)

        self.discount_factor = 0.99
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        self.batch_size = 32
        self.train_start = 50000

        self.q_model = utils.get_state_action_value_network(self.nb_actions, self.state_size)
        self.target_q_model = utils.get_state_action_value_network(self.nb_actions, self.state_size)

        self.q_optimizer = utils.get_q_optimizer(self.q_model, self.nb_actions)

    def update_target_model(self):
        """
        Like standard DQN DDQN uses a target network which periodically gets updated

        :return: None
        """
        self.target_q_model.set_weights(self.q_model.get_weights())

    def get_max_q_estimates(self, history):
        """
        We want to keep track of the maximum Q values which are predicted
        in order to measure whether the algorithms suffer from the over-estimation
        bias of the Q-function

        :param history: a state coming from the Atari game
        :return: the maximum Q-value associated to that state
        """

        history = np.float32(history / 255.0)
        q_values = self.q_model.predict(history)

        max_q = abs(max(q_values[0]))

        return(max_q)

    def get_model(self):
        """
        We return the Q network

        :return: the state-action-value model
        """
        return(self.q_model)

    def get_action(self, history):
        """
        We return an action given by the state-action value network

        :param history: a state coming from the Atari game
        :return: an action based on the e-greedy exloration policy
        """

        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.q_model.predict(history)
            return np.argmax(q_value[0])

    def store_replay_memory(self, history, action, reward, next_history, dead):
        """
        The experience replay memory buffer which stores RL trajectories

        :param history: a state coming from the Atari game
        :param action: the action taken by the agent
        :param reward: the reward coming from the environment
        :param next_history: the next state coming from the Atari game
        :param dead: a flag telling us whether the agent has died or not
        :return: None
        """

        self.memory.append((history, action, reward, next_history, dead))

    def train_replay(self):
        """
        Function used for training from the experience replay memory buffer based on DDQN
        update rules.

        :return: None
        """

        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size, ))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        value = self.q_model.predict(next_history)
        target_value = self.target_q_model.predict(next_history)

        # like Q Learning, but we get the maximum Q value at s_{t+1} from the target model
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                # the key point of Double-DQN
                # the selection of the action comes from the main q_model
                # but for the update we use the target model which is the closest to an unbiased
                # estimator of the Q function we can get

                target[i] = reward[i] + self.discount_factor * \
                                        target_value[i][np.argmax(value[i])]

        self.q_optimizer([history, action, target])

utils = Utils()
