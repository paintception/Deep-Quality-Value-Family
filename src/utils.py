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
        """
        We ensure to create proper paths where to store the results of the experiments
        :param source_game: the Atari game which is being tested
        :param policy_mode: whether the algorithm is on-policy or off-policy
        :param algorithm: which agent it is
        :return: None
        """

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
        """
        We store the results of the experiments
        :param episode_rewards: the reward obtained by the agent at each testing episode
        :param max_q_estimates: the max-Q estimate necessary for checking whether overestimation occurs
        :param agent: the agent which is being tested
        :return: None
        """
        np.save(self.rewards_path + agent + "_rewards.npy", episode_rewards)
        np.save(
            self.max_q_estimates_path +
            agent +
            "_max_q_estimates.npy",
            max_q_estimates)

    def store_double_weights(self, model_1, model_2, policy_mode):
        """
        We store the weights of the models of the DQV-family of algorithms
        The first one corresponds to the V-network while the second to the Q-network
        """
        if policy_mode == "offline":
            model_1.save_weights(self.weights_path + "state_value_model.h5")
            model_2.save_weights(self.weights_path + "state_action_value_model.h5")
        elif policy_mode == "online":
            model_1.save_weights(self.weights_path + "state_value_model.h5")
            model_2.save_weights(self.weights_path + "state_action_value_model.h5")

    def store_single_weights(self, model):
        """
        We store the model of the DQN and DDQN algorithms
        :param model: the model itself
        :return: None
        """
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
        """
        We create the Q-network which is used by all tested algorithms
        :param nb_actions: amount of possible actions in the environment
        :param state_size: the size of the preprocessed frames which is 84x84x4
        :return: the state-action-value network
        """

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
        We create the network which estimates the state-value function which is used by DQV and DQV-Max
        :param state_size: the size of the preprocessed frames which is 84x84x4
        :return: the state-value network
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
        """
        A pre-processing function which reshapes each state of the Atari learning environment
        :param state: a state
        :return: a reshaped gray-scaled version of the same state
        """
        processed_state = np.uint8(
            resize(rgb2gray(state), (84, 84), mode='constant') * 255)
        return processed_state