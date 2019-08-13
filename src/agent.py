import gym
import random

import numpy as np

from utils import Utils
from dqv import dqvAgent
from dqn import dqnAgent
from ddqn import ddqnAgent
from dqv_max import dqvMaxAgent

from duelling_dqv import duellingDQV


class RlAgent(object):
    def __init__(self, source_game, policy_mode, algorithm, episodes):
        self.game = source_game
        self.env = gym.make(source_game)
        self.no_op_steps = 30
        self.nb_actions = 3
        self.policy_mode = policy_mode
        self.algorithm = algorithm
        self.global_steps = 0
        self.update_target_rate = 10000

        self.target_model = True
        self.episodes = episodes
        self.episode_reward = 0

        self.max_q_estimates = []
        self.episode_rewards = []

        self.choose_agent()

    def choose_agent(self):
        """
        Function which allows choosing between the differently tested agents
        :return: None
        """
        if self.policy_mode == "offline":
            if self.algorithm == "dqn":
                agent = dqnAgent(self.nb_actions)
            elif self.algorithm == "ddqn":
                agent = ddqnAgent(self.nb_actions)
            elif self.algorithm == "dqv-max":
                agent = dqvMaxAgent(self.nb_actions)

        elif self.policy_mode == "online":
            if self.algorithm == "dqv":
                agent = dqvAgent(self.nb_actions)
            elif self.algorithm == "duelling-dqv":
                agent = duellingDQV(self.nb_actions)

        self.start_training(agent)

    def start_training(self, agent):
        """
        The main training loop used by each agent
        :param agent: one among the differently tested algorithms
        :return: None
        """

        for e in range(self.episodes):
            done = False
            dead = False

            step, score, start_life = 0, 0, 5
            observe = self.env.reset()

            for _ in range(random.randint(1, agent.no_op_steps)):
                observe, _, _, _ = self.env.step(1)

            state = utils.pre_processing(observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                if agent.render:
                    self.env.render()

                self.global_steps += 1
                step += 1

                self.max_q_estimates.append(agent.get_max_q_estimates(history))
                action = agent.get_action(history)

                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3

                observe, reward, done, info = self.env.step(real_action)
                next_state = utils.pre_processing(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(
                    next_state, history[:, :, :, :3], axis=3)

                clipped_reward = np.clip(reward, -1., 1.)

                agent.store_replay_memory(
                    history, action, clipped_reward, next_history, dead)

                agent.train_replay()

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                if agent.target_model is True:
                    if self.global_steps % self.update_target_rate == 0:
                        agent.update_target_model()

                self.episode_reward += reward

                if dead:
                    dead = False
                else:
                    history = next_history

            print('Episode Reward: {}'.format(self.episode_reward))

            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0

            utils.make_storing_paths(
                self.game, self.policy_mode, self.algorithm)
            utils.save_results(
                self.episode_rewards,
                self.max_q_estimates,
                self.algorithm)

            if self.algorithm == "dqv" or self.algorithm == "dqv-max":
                print('Storing the models.')
                models = agent.get_models()
                utils.store_double_weights(models[0], models[1], self.policy_mode)
                print('Weights are stored')

            elif self.algorithm == "dqn" or "ddqn":
                print('Storing the model.')
                model = agent.get_model()
                utils.store_single_weights(model)
                print('Weights are stored')

utils = Utils()