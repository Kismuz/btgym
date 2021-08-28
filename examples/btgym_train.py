import os
from btgym import BTgymEnv
import IPython.display as Display
import PIL.Image as Image
from gym import spaces

import gym
import numpy as np
import random

'''
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
'''

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
from tensorflow.keras.optimizers import RMSprop, Adam

from collections import deque

# OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=20000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        # state_shape  = list(self.env.observation_space.shape.items())[0][1]
        # Reshaping for LSTM
        # state_shape=np.array(state_shape)
        # state_shape= np.reshape(state_shape, (30,4,1))
        '''
        model.add(Dense(24, input_dim=state_shape[1], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        '''
        model.add(LSTM(64,
                       input_shape=(4, 1),
                       # return_sequences=True,
                       stateful=False
                       ))
        model.add(Dropout(0.5))

        # model.add(LSTM(64,
        # input_shape=(1,4),
        # return_sequences=False,
        #               stateful=False
        #               ))
        model.add(Dropout(0.5))

        # model.add(Dense(self.env.action_space.n, init='lecun_uniform'))
        model.add(Dense(len(self.env.action_space.base_actions), activation='relu'))
        model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        rms = RMSprop()
        adam = Adam()
        model.compile(loss='mse', optimizer=adam)
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

    def show_rendered_image(self, rgb_array):
        """
        Convert numpy array to RGB image using PILLOW and
        show it inline using IPykernel.
        """
        dirPath = os.path.dirname(os.path.abspath(__file__))
        filePath = os.path.abspath(os.path.join(dirPath, './img', 'dqn.png'))
        Display.display(Image.fromarray(rgb_array))
        # img = Image.open(filePath)
        Image.Image.save(Image.fromarray(rgb_array), fp=filePath)

    def render_all_modes(self, env):
        """
        Retrieve and show environment renderings
        for all supported modes.
        """
        for mode in self.env.metadata['render.modes']:
            print('[{}] mode:'.format(mode))
            self.show_rendered_image(self.env.render(mode))


def main(filename):
    env = BTgymEnv(filename=filename,
                   state_shape={'raw': spaces.Box(low=-100, high=100, shape=(30, 4))},
                   skip_frame=5,
                   start_cash=100000,
                   broker_commission=0.02,
                   fixed_stake=100,
                   connect_timeout=180,
                   drawdown_call=90,
                   render_state_as_image=True,
                   render_ylabel='Price Lines',
                   render_size_episode=(12, 8),
                   render_size_human=(8, 3.5),
                   render_size_state=(10, 3.5),
                   render_dpi=75,
                   multiprocessing=1,
                   port=5000,
                   data_port=4999,
                   verbose=0, )

    env.reset()  # <=== CORRECTED HERE: fake reset() tells data_master to start data_server_process

    gamma = 0.9
    epsilon = .95

    trials = 10
    trial_len = 1000

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        # dqn_agent.model= load_model("./model.model")
        cur_state = np.array(list(env.reset().items())[0][1])
        cur_state = np.reshape(cur_state, (30, 4, 1))
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            reward = reward * 10 if not done else -10
            new_state = list(new_state.items())[0][1]
            new_state = np.reshape(new_state, (30, 4, 1))
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break

        print("Completed trial #{} ".format(trial))
        dqn_agent.render_all_modes(env)
        dqn_agent.save_model("model.model".format(trial))


if __name__ == "__main__":
    dirPath = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.abspath(os.path.join(dirPath, './data/DAT_ASCII_EURUSD_M1_2016.csv'))
    main(filePath)
