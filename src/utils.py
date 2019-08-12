import numpy as np
from keras import backend as K
from skimage.transform import resize
from skimage.color import rgb2gray
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
import os

RESULTS_PATH = "./final_results/"


class Utils(object):
    def make_storing_paths(self, source_game, policy_mode, algorithm):
        self.rewards_path = RESULTS_PATH + "rewards/" + \
                            source_game + "/" + policy_mode + "/" + algorithm + "/"
        self.max_q_estimates_path = RESULTS_PATH + "q_estimates/" + \
                                    source_game + "/" + policy_mode + "/" + algorithm + "/"
        self.weights_path = RESULTS_PATH + "model_weights/" + \
                            source_game + "/" + policy_mode + "/" + algorithm + "/"

        if not os.path.exists(self.rewards_path):
            os.makedirs(self.rewards_path)
        if not os.path.exists(self.max_q_estimates_path):
            os.makedirs(self.max_q_estimates_path)
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)

    def save_results(self, episode_rewards, max_q_estimates, agent):
        np.save(self.rewards_path + agent + "_rewards.npy", episode_rewards)
        np.save(
            self.max_q_estimates_path +
            agent +
            "_max_q_estimates.npy",
            max_q_estimates)

    def store_double_weights(self, model_1, model_2, policy_mode):
        """

        :rtype: object
        """
        if policy_mode == "offline":
            model_1.save_weights(self.weights_path + "state_value_model.h5")
            model_2.save_weights(self.weights_path + "state_action_value_model.h5")
        elif policy_mode == "online":
            model_1.save_weights(self.weights_path + "state_value_model.h5")
            model_2.save_weights(self.weights_path + "state_action_value_model.h5")

    def store_single_weights(self, model):
        model.save_weights(self.weights_path + "state_action_value_model.h5")

    def get_q_optimizer(self, q_model, nb_actions):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = q_model.output

        a_one_hot = K.one_hot(a, nb_actions)
        q_value = K.sum(py_x * a_one_hot, axis=1)

        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part

        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(q_model.trainable_weights, [], loss)
        trainable = K.function([q_model.input, a, y], [loss], updates=updates)

        return trainable

    def get_state_action_value_network(self, nb_actions, state_size):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),
                         activation='relu', input_shape=state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(nb_actions))

        return model

    def get_state_value_network(self, state_size):
        """

        :type state_size: object
        """
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),
                         activation='relu', input_shape=state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=RMSprop(
            lr=0.001, rho=0.95, epsilon=0.01))

        return model

    def pre_processing(self, state):
        processed_state = np.uint8(
            resize(rgb2gray(state), (84, 84), mode='constant') * 255)
        return processed_state