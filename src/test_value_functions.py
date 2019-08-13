import gym
import random
import numpy as np

from skimage.color import rgb2gray
from skimage.transform import resize
from utils import Utils

from scipy.ndimage.filters import gaussian_filter1d

from matplotlib import pyplot as plt

plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 50})

EPISODES = 1

GAME = "EnduroDeterministic-v4"

# For DQV
#Q_WEIGHTS = "../models/EnduroDeterministic-v4/online/dqv/state_action_value_model.h5"
#V_WEIGHTS = "../models/EnduroDeterministic-v4/online/dqv/state_value_model.h5"

# For DQV-Max
Q_WEIGHTS = "../models/EnduroDeterministic-v4/offline/dqv-max/state_action_value_model.h5"
V_WEIGHTS = "../models/EnduroDeterministic-v4/offline/dqv-max/state_value_model.h5"

utils = Utils()

class DQVAgent:
    def __init__(self, action_size):
        self.render = False
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.q_model = utils.get_state_action_value_network(self.action_size, self.state_size)
        self.v_model = utils.get_state_value_network(self.state_size)
        self.q_model.load_weights(Q_WEIGHTS)
        self.v_model.load_weights(V_WEIGHTS)

        self.higher_v = 0
        self.higher_q = 0
        self.gamma = 0.99

        self.value_estimates = []
        self.max_q_estimates = []

    def get_expected_value(self, history):
        """
        We get the predictions coming from the state-value network needed to check whether the algorithms
        systematically overestimate the V function instead of the Q function.

        :param history: a state
        :return: None
        """
        history = np.float32(history / 255.0)

        print("Expected Value for the current state {}:".format(self.v_model.predict(history)[0][0]))

    def get_action(self, history):
        """
        Given a state and an already trained model we get the predictions of the state-action value
        network in order to unroll the learned policies.

        :param history: a state
        :return: the best possible action
        """
        history = np.float32(history / 255.0)
        q_value = self.q_model.predict(history)
        #print("Maximum Q value for best state-action pair {}:".format(max(q_value[0])))

        return np.argmax(q_value[0])

    def calculate_stats(self, history):
        """
        We keep track of how many times the estimates of the V network are higher than the ones coming from the
        Q network.

        :param history: a state
        :return: None
        """

        history = np.float32(history / 255.0)
        visited = False

        if visited is False:
            state_value = self.v_model.predict(history)[0][0]
            q_action_values = self.q_model.predict(history)

            max_q = max(q_action_values[0])

            self.value_estimates.append(state_value)
            self.max_q_estimates.append(max_q)

            if max_q > state_value:
                self.higher_q += 1
            elif max_q < state_value:
                self.higher_v += 1

    @staticmethod
    def pre_processing(observe):
        """

        :param observe: a state coming from the Atari ALE environment which gets pre-processed before using it as input for
                        the different models
        :return: a pre-processed state
        """
        processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
        return processed_observe


if __name__ == "__main__":
    """
    We unroll the policies of the trained models, check the cumulative reward which is obtained and compute the 
    the average discounted return from each visited state. Such baseline value is then used in order to check if 
    at training time the max Q(s',a') estimates diverge from it, therefore making the algorithms overoptimistic.
    """

    env = gym.make(GAME)
    agent = DQVAgent(action_size=3)

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, 30)):
            observe, _, _, _ = env.step(1)

        state = agent.pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        episode_reward = 0
        discounted_return = 0
        k = 0
        returns = []

        while not done:
            if agent.render:
                env.render()

            agent.calculate_stats(history)
            #print('-------------------------------------')
            #agent.get_expected_value(history)
            action = agent.get_action(history)
            #print('-------------------------------------')

            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            observe, reward, done, info = env.step(real_action)

            discounted_return += (agent.gamma**k)*reward
            returns.append(discounted_return)

            next_state = agent.pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            action = agent.get_action(next_history)

            next_action = agent.get_action(next_history)

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            if dead:
                dead = False
            else:
                history = next_history

            episode_reward += reward

            k += 1

        print("Reward obtained by the trained model is {}: ".format(episode_reward))

    print("The amount of times the max Q-value is higher than the state-value estimate is {}".format(agent.higher_q))
    print("The amount of times the state-value estimate is higher than the max Q-value is {}".format(agent.higher_v))
    print("The averaged discounted return of all visited states is {}".format(np.mean(returns)))

    N = len(agent.value_estimates)
    x = range(N)

    plt.xlabel("States")
    plt.ylabel('Value estimates')
    plt.plot(gaussian_filter1d(agent.value_estimates, sigma=2), c="skyblue", label=r'$V(s)$')
    plt.fill_between(range(len(agent.value_estimates)), gaussian_filter1d(agent.value_estimates, sigma=2),
                     color="skyblue", alpha=0.4)
    plt.plot(gaussian_filter1d(agent.max_q_estimates, sigma=2), c="orange", label=r'$max\:Q(s,a)$')
    plt.fill_between(range(len(agent.value_estimates)), gaussian_filter1d(agent.max_q_estimates, sigma=2),
                     color="orange", alpha=0.4)
    plt.legend(loc='upper left', prop={'size': 10})
    plt.show()
