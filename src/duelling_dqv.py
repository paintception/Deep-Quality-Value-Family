import os

import gym
import random
import time
import copy

import numpy as np
import tensorflow as tf

from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K

from keras.utils import plot_model

EPISODES = 1000


class duellingDQV:
    def __init__(self, nb_actions):
        self.render = True
        self.target_model = True
        self.nb_actions = nb_actions
        self.load_model = False
        self.state_size = (84, 84, 4)
        self.epsilon = .5
        self.epsilon_start, self.epsilon_end = 1, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (
            self.epsilon_start - self.epsilon_end) / self.exploration_steps

        self.learning_rate = 0.001
        self.discount_factor = 0.99

        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30

        self.batch_size = 32
        self.train_start = 50

        self.duelling_model = self.build_duelling_models()

        if self.target_model is True:
            self.target_duelling_model = self.build_target_duelling_models()

        self.optimizer = self.optimizer()

    def update_target_model(self):
        print("Updating the Target model")
        self.target_v_duelling_part.set_weights(self.v_duelling_part.get_weights())

    def build_target_duelling_models(self):
        input_state = Input(shape=(84, 84, 4), name='game_state')

        first_conv = Conv2D(
            32, (8, 8), strides=(
                4, 4), activation='relu')(input_state)
        second_conv = Conv2D(
            64, (4, 4), strides=(
                2, 2), activation='relu')(first_conv)
        third_conv = Conv2D(
            64, (3, 3), strides=(
                1, 1), activation='relu')(second_conv)
        flattened = Flatten()(third_conv)

        dense_layer = Dense(512, activation='relu')(flattened)
        state_action_layer = Dense(self.nb_actions)(dense_layer)
        value_advantage_layer = Dense(1)(dense_layer)

        self.target_q_duelling_part = Model(
            input=[input_state],
            outputs=[state_action_layer],
            name='state_action_duelling_dqv')
        self.target_v_duelling_part = Model(
            input=[input_state],
            outputs=[value_advantage_layer],
            name='value_duelling_dqv')

        self.target_v_duelling_part.compile(
            loss='mse', optimizer=RMSprop(
                lr=0.001, rho=0.95, epsilon=0.01))

    def build_duelling_models(self):
        input_state = Input(shape=(84, 84, 4), name='game_state')

        first_conv = Conv2D(
            32, (8, 8), strides=(
                4, 4), activation='relu')(input_state)
        second_conv = Conv2D(
            64, (4, 4), strides=(
                2, 2), activation='relu')(first_conv)
        third_conv = Conv2D(
            64, (3, 3), strides=(
                1, 1), activation='relu')(second_conv)
        flattened = Flatten()(third_conv)

        dense_layer = Dense(512, activation='relu')(flattened)
        state_action_layer = Dense(self.nb_actions)(dense_layer)
        value_advantage_layer = Dense(1)(dense_layer)

        self.q_duelling_part = Model(
            input=[input_state],
            outputs=[state_action_layer],
            name='state_action_duelling_dqv')
        self.v_duelling_part = Model(
            input=[input_state],
            outputs=[value_advantage_layer],
            name='value_duelling_dqv')

        self.v_duelling_part.compile(
            loss='mse', optimizer=RMSprop(
                lr=0.001, rho=0.95, epsilon=0.01))

    def optimizer(self):

        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        # the output tensor for the state-action pairs
        py_x = self.q_duelling_part.output

        a_one_hot = K.one_hot(a, 3)
        q_value = K.sum(py_x * a_one_hot, axis=1)

        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part

        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)

        updates = optimizer.get_updates(
            self.q_duelling_part.trainable_weights, [], loss)

        train = K.function([self.q_duelling_part.input, a, y], [
                           loss], updates=updates)

        return train

    def get_max_q_estimates(self, history):
        history = np.float32(history / 255.0)
        q_values = self.q_duelling_part.predict(history)

        max_q = abs(max(q_values[0]))

        return(max_q)

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(3)

        else:
            q_values = self.q_duelling_part.predict(history)

            return np.argmax(q_values[0])

    def store_replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    def train_replay(self):

        if len(self.memory) < self.train_start:
            return

        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))

        # Initialize the Value targets to optimize
        v_target = np.zeros((self.batch_size,))

        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        # current state-action values Q(st, at)
        q_outputs = self.q_duelling_part.predict(history)

        # TD-values for updating the networks coming from a target model
        # or the original dqv-dueling network
        if self.target_model is True:
            v_target_value = self.target_v_duelling_part.predict(next_history)
        elif self.target_model is False:
            v_target_value = self.v_duelling_part.predict(next_history)

        q_targets = list()  # List for updating coming state-action values

        for i in range(self.batch_size):
            if dead[i]:
                v_target[i] = reward[i]
                q_outputs[i][action[i]] = reward[i]

            else:
                v_target[i] = reward[i] + \
                    self.discount_factor * v_target_value[i]
                q_outputs[i][action[i]] = reward[i] + \
                    self.discount_factor * v_target_value[i]

            q_targets.append(q_outputs[i][action[i]])

        loss = self.optimizer([history, action, q_targets])

        self.v_duelling_part.fit(history, v_target, epochs=1, verbose=0)

        # End of updating the DCNNs in experience replay batch
