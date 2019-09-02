import random
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K


class duellingDQV(object):
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
        """
        We update the target network with the weights of the online state-value network: we have to be sure
        we only get the weights that are required for learning the state-value function as given by
        self.v_duelling_part.get_weights()
        :return: None
        """

        self.target_v_duelling_part.set_weights(self.v_duelling_part.get_weights())

    def build_duelling_models(self):
        """
        We create a multi-head neural network which jointly learns an approximation of the state-value function and the
        state-action value function. Its convolutional layers are fully shared whereas once the feature maps get flattened
        we add a specific hidden layer before the outputs which estimate V(s) and Q(s,a). For the different Dueling architectures
        reported in the paper the position of the flattened and value_layers has to be changed according to the depth of the
        network. HARD-DQV follows the exact same structure of this architecture with the difference that the state_action_output
        and the state_value_output are directly connected to flattened, without passing through a specific fully connected layer.

        :return: a Dueling-Architecture to be optimized with DQV's learning update rules.
        """

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

        state_value_layer = Dense(512, activation='relu')(flattened)
        state_action_value_layer = Dense(512, activation='relu')(flattened)

        state_action_output = Dense(self.nb_actions)(state_action_value_layer)
        state_value_output = Dense(1)(state_value_layer)

        self.q_duelling_part = Model(
            input=[input_state],
            outputs=[state_action_output],
            name='state_action_duelling_dqv')
        self.v_duelling_part = Model(
            input=[input_state],
            outputs=[state_value_output],
            name='value_duelling_dqv')

        self.v_duelling_part.compile(
            loss='mse', optimizer=RMSprop(
                lr=0.001, rho=0.95, epsilon=0.01))

    def build_target_duelling_models(self):
        """
        We create a copy of the previous network which is necessary for estimating the targets Dueling-DQV will
        bootstrap on.

        :return: a copy of the main Dueling network
        """

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

        state_value_layer = Dense(512, activation='relu')(flattened)
        state_action_value_layer = Dense(512, activation='relu')(flattened)

        state_action_output = Dense(self.nb_actions)(state_action_value_layer)
        state_value_output = Dense(1)(state_value_layer)

        self.q_duelling_part = Model(
            input=[input_state],
            outputs=[state_action_output],
            name='state_action_duelling_dqv')
        self.target_v_duelling_part = Model(
            input=[input_state],
            outputs=[state_value_output],
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
        """
        We want to keep track of the maximum Q values which are predicted
        in order to measure whether the algorithms suffer from the over-estimation
        bias of the Q-function

        :param history: a state coming from the Atari game
        :return: the maximum Q-value associated to that state
        """

        history = np.float32(history / 255.0)
        q_values = self.q_duelling_part.predict(history)

        max_q = abs(max(q_values[0]))

        return(max_q)

    def get_action(self, history):
        """
        We return an action given by the state-action value network

        :param history: a state coming from the Atari game
        :return: an action based on the e-greedy exloration policy
        """
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(3)

        else:
            q_values = self.q_duelling_part.predict(history)

            return np.argmax(q_values[0])

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
        Function used for training from the experience replay memory buffer based on DQV
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

        # TD-values for updating the networks coming from the target model
        if self.target_model is True:
            v_target_value = self.target_v_duelling_part.predict(next_history)
        elif self.target_model is False:
            v_target_value = self.v_duelling_part.predict(next_history)

        q_targets = []

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

        self.optimizer([history, action, q_targets]) # optimize the state-action-value head
        self.v_duelling_part.fit(history, v_target, epochs=1, verbose=0) # optimize the state-value head

