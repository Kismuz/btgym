Package Description
-------------------
**Btgym** is an OpenAI Gym-compatible environment for Backtrader backtesting/trading library,
designed to provide gym-integrated framework for
running reinforcement learning experiments
in [close to] real world algorithmic trading environments.

**[experimental]:**
Besides core environment package includes implementations of several deep RL algorithms,
tuned [to attempt] to solve this particular type of tasks.


**Backtrader** is open-source algorithmic trading library:
GitHub: http://github.com/mementum/backtrader
Documentation and community:
http://www.backtrader.com/


**OpenAI Gym** is...,
well, everyone knows Gym:
GitHub: http://github.com/openai/gym
Documentation and community:
https://gym.openai.com/

**DISCLAIMER:**
This package is neither out-of-the-box-moneymaker, nor it provides ready-to-converge RL solutions.
Rather, it is framework for setting experiments with complex, non stationary, time-series based environments.
I have no idea what kind of algorithm and setup will solve it [if any]. Explore on your own!

Installation
------------

Clone or copy btgym repository to local disk, cd to it and run: `pip install -e .` to install package and all dependencies::

    git clone https://github.com/Kismuz/btgym.git

    cd btgym

    pip install -e .

To update to latest version::

    cd btgym

    git pull

    pip install --upgrade -e .

Quickstart
----------

Making gym environment with all parmeters set to defaults is as simple as::

    from btgym import BTgymEnv

    MyEnvironment = BTgymEnv(filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv',)

Adding more controls may look like::

    from btgym import BTgymEnv

    MyEnvironment = BTgymEnv(filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
                             episode_len_days=2,
                             episode_len_hours=23,
                             episode_len_minutes=55,
                             drawdown_call=50,
                             state_shape=(4,20),
                             port=5555,
                             verbose=1,
                             )


Same one but registering environment in Gym preferred way::

    import gym
    from btgym import BTgymEnv

    env_params = dict(filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
                      episode_len_days=2,
                      episode_len_hours=23,
                      episode_len_minutes=55,
                      drawdown_call=50,
                      state_shape=(20,4),
                      port=5555,
                      verbose=1,
                      )

    gym.envs.register(id='backtrader-v5555', entry_point='btgym:BTgymEnv', kwargs=env_params,)

    MyEnvironment = gym.make('backtrader-v5555')


Maximum environment flexibility is achieved by explicitly defining and passing `Dataset` and `Cerebro` instances::

    from gym import spaces
    import backtrader as bt
    from btgym import BTgymDataset, BTgymStrategy, BTgymEnv

    MyCerebro = bt.Cerebro()
    MyCerebro.addstrategy(BTgymStrategy,
                          state_shape={'raw_state': spaces.Box(low=0,high=1,shape=(20,4))},
                          skip_frame=5,
                          state_low=None,
                          state_high=None,
                          drawdown_call=50,
                          )

    MyCerebro.broker.setcash(100.0)
    MyCerebro.broker.setcommission(commission=0.001)
    MyCerebro.addsizer(bt.sizers.SizerFix, stake=10)
    MyCerebro.addanalyzer(bt.analyzers.DrawDown)

    MyDataset = BTgymDataset(filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
                             start_weekdays=[0, 1, 2, 4],
                             start_00=True,
                             episode_len_days=0,
                             episode_len_hours=23,
                             episode_len_minutes=55,
                             time_gap_days=0,
                             time_gap_hours=5,
                             )

    MyEnvironment = BTgymEnv(dataset=MyDataset,
                             engine=MyCerebro,
                             port=5555,
                             verbose=1,
                             )

