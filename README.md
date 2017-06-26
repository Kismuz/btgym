## <a name="title"></a>Backtrader gym Environment
**Implementation of OpenAI Gym environment for Backtrader backtesting/trading library.**
****
Backtrader is open-source algorithmic trading library:  
GitHub: http://github.com/mementum/backtrader  
Documentation and community:
http://www.backtrader.com/

OpenAI Gym is...,
well, everyone knows Gym:  
GitHub: http://github.com/openai/gym  
Documentation and community:
https://gym.openai.com/
****
### <a name="outline"></a>Outline:
General purpose of this wrapper is to provide gym-integrated framework for
running reinforcement learning experiments 
in [close to] real world algorithmic trading environments.

#### [See news and update notes below](#news)

```
DISCLAIMER:
This package is neither out-of-the-box-moneymaker, nor it provides ready-to-converge RL solutions.
Rather, it is framework for setting experiments with complex, non stationary, time-series based environments.
I have no idea what kind of algorithm and setup will solve it [if any]. Explore on your own!
```
###### This work is in early development stage. Any suggestions, feedback and contributions are welcome.
****
### <a name="contents"></a>Contents
- [Installation](#install)
- [Quickstart](#start)
- [Description](#description)
    - [Problem setting](#problem)
    - [Data sampling approaches](#data)
    - [Environment engine description](#engine)
    - [General notes](#notes)
- [Reference](#reference) 
- [Current issues and limitations](#issues)
- [Roadmap](#roadmap)
- [Update news](#news)

    

****
### <a name="install"></a>[Installation](#contents)
- Btgym requires:  `gym`, `backtrader`, `pandas`, `numpy`, `pyzmq`; 
  `matplotlib` required for `env.render()` method.
- Examples requires: `scipy`, .
- Clone or copy btgym repository to local disk, cd to it and run: `pip install -e . `
to install package and dependencies e.g.:
``` 
got clone https://github.com/Kismuz/btgym.git
cd btgym
pip install -e .
```
- To update to latest version:
```
cd btgym
git pull
pip install --upgrade -e .
```

****
### <a name="start"></a>[Quickstart](#contents)
Making gym environment with all parmeters set to defaults is as simple as:

```python
from btgym import BTgymEnv
 
MyEnvironment = BTgymEnv(filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv',)
```
Adding more controls may look like:
```python
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
                 
```
Same one but registering environment in Gym preferred way:
```python
import gym
from btgym import BTgymEnv
  
env_params = dict(filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
                  episode_len_days=2,
                  episode_len_hours=23,
                  episode_len_minutes=55,
                  drawdown_call=50,
                  state_shape=(4,20),
                  port=5555,
                  verbose=1,
                  )
                  
gym.envs.register(id='backtrader-v5555', entry_point='btgym:BTgymEnv', kwargs=env_params,)
                  
MyEnvironment = gym.make('backtrader-v5555')
```

Maximum environment flexibility is achieved by explicitly defining and passing `Dataset` and `Cerebro` instances:
```python
import backtrader as bt
from btgym import BTgymDataset, BTgymStrategy, BTgymEnv
 
MyCerebro = bt.Cerebro()
MyCerebro.addstrategy(BTgymStrategy,
                      state_shape=(4,20),
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
```
###### See notebooks in `examples` directory.
****
### <a name="description"></a> [General description](#contents)
#### <a name="problem"></a> Problem setting
Consider reinforcement learning setup for equity/currency trading:
- agent action space is discrete (`buy`, `sell`, `close` [position], `hold` [do nothing]);
- environment is episodic: maximum  episode duration and episode termination conditions
  are set;
- for every timestep of the episode agent is given environment state observation as tensor of last
  m price open/high/low/close values for every equity considered and based on that information is making
  trading decisions.
- agent's goal is to maximize cumulative capital;
- classic 'market liquidity' and 'capital impact' assumptions are met.
- environment setup is set close to real trading conditions, including commissions, order execution delays,
  trading calendar etc.

#### <a name="data"></a> Data selection options for backtest agent training:
- random sampling:
  historic price change dataset is divided to training, cross-validation and testing subsets.
  Since agent actions do not influence market, it is possible to randomly sample continuous subset
  of training data for every episode. [Seems to be] most data-efficient method.
  Cross-validation and testing performed later as usual on most "recent" data;
- sequential sampling:
  full dataset is feeded sequentially as if agent is performing real-time trading,
  episode by episode. Most reality-like, least data-efficient, natural non-stationarity remedy.
- sliding time-window sampling:
  mixture of above, episde is sampled randomly from comparatively short time period, sliding from
  furthest to most recent training data. Should be less prone to overfitting than random sampling.
- NOTE: only random sampling is currently implemented.

### <a name="engine"></a> [Environment engine](#contents)
  BTgym uses Backtrader framework for actual environment computations, for extensive documentation see:
https://www.backtrader.com/docu/index.html.
In brief:
- User defines backtrading engine parameters by composing `Backtrader.Cerebro()` subclass,
  provides historic prices dataset as `BTgymDataset()` instance and passes it as arguments when making BTgym environment.
  See https://www.backtrader.com/docu/concepts.html for general Backtrader concepts descriptions.
- Environment starts separate server process responsible for rendering gym environment
  queries like `env.reset()` and `env.step()` by repeatedly sampling episodes form given dataset and running
  backtesting `Cerebro` engine on it. See OpenAI Gym documentation for details: https://gym.openai.com/docs

#### Data flow
```
            BTgym Environment                                 RL Framework
                                           +-+
   (episode mode)  +<-----<action>--- -----| |<--------------------------------------+
          |        |                       |e|                                       |
          +<------>+------<       state >->|n|--->[feature *]---><state>--+->[agent]-+
          |        |      < observation >  |v|    [estimator]             |     |
          |        |                       |.|                            |     |
    [Backtrader]   +-----<  portfolio >--->|s|--->[reward  *]---><reward>-+     |
    [Server    ]   |     < statistics >    |t|    [estimator]                   |
       |           |                       |e|                                  |
       |           +------<is_done>------->|p|--+>[runner **]<----------------->+
  (control mode)   |                       | |  |    |
       |           +------<aux.info>--- -->| |--+    |
       |                                   +-+       |
       +--<'_stop'><------------------->|env._stop|--+
       |                                             |
       +--<'_reset'><------------------>|env.reset|--+
 
* - can be done on server side;
** - RL framework specific module;
```
#### Sample workflow:
1. Define backtesting `BTgymStrategy(bt.Strategy)`, which will
   control Environment inner dynamics and backtesting logic.
    - As for RL-specific part, any `STATE`,
   `REWARD`, `DONE` and `INFO` computation logic can be implemented by overriding `get_state()`, `get_reward()`,
   `get_info()`, `is_done()` and `set_datalines()` methods.
    - As for Broker/Trading specific part, custom order execution logic, stake sizing,
      analytics tracking can be implemented as for regular `bt.Strategy()`.
2. Instantiate `Cerbro()`, add `BTgymStrategy()`, backtrader `Sizers`, `Analyzers` and `Observers` (if needed).
3. Define dataset by passing CSV datafile and parameters to BTgymDataset instance.
    - `BTgymDataset()` is simply `Backtrader.feeds` class wrapper, which pipes
    `CSV`[source]-->`pandas`[for efficient sampling]-->`bt.feeds` routine
    and implements random episode data sampling.
4. Initialize (or register and `make()`) gym environment with `Cerebro()` and `BTgymDataset()` along with other kwargs.
5. Run your favorite RL algorithm:
    - start episode by calling `env.reset()`;
    - advance one step of episode by calling `env.step()`, perform agent training or testing;
    - after single episode is finished, retrieve agent performance statistic by `env.get_stat()`.
****
#### Server operation details:
Backtrader server starts when `env.reset()` method is called for first time , runs as separate process, follows
simple Request/Reply pattern (every request should be paired with reply message) and operates one of two modes:
- Control mode: initial mode, accepts only `_reset`, `_stop` and `_getstat` messages. Any other message is ignored
  and replied with simple info messge. Shuts down upon recieving `_stop` via `env._stop_server()` method,
  goes to episode mode upon `_reset` (via `env.reset()`) and send last run episode statistic (if any) upon `_getstat`
  via `env.get_stat()`.
- Episode mode: runs episode according `BtGymStrategy()` logic. Accepts `action` messages,
  returns `tuple`: `([state observation], [reward], [is_done], [aux.info])`.
  Finishes episode upon recieving `action`==`_done` or according to strategy termination rules, than falls
  back to control mode.
    - Before every episode start, BTserver samples episode data and adds it to `bt.Cerebro()` instance
   along with specific `_BTgymAnalyzer`. The service of this hidden Analyzer is twofold:
        - enables strategy-environment communication by calling RL-related `BTgymStrategy` methods:
       `get_state()`, `get_reward()`, `get_info()` and `is_done()` [see below];
        - controls episode termination conditions.
    - Episode runtime: after preparing environment initial state by running `BTgymStrategy` `start()`, `prenext()`
      methods, server halts and waits for incoming agent `action`. Upon receiving `action`, server performs all
necessary `next()` computations (e.g. issues orders, computes broker values etc.),
composes environment response and sends it back to agent ( via `_BTgymAnalyzer`). Actually, since 'no market impact' is assumed, all state
computations are performed one step ahead:

#### Server loop:
```pseudocode
Initialize by receiving engine [bt.Cerebro()] and dataset [BTgymDataset()]
Repeat until received message '_stop':
    Wait for incoming message
    If message is '_getstat':
        send episode statistics
    If message is '_reset':
        Randomly sample episode data from BTgymDataset
        Add episode data to bt.Cerebro()
        Add service _BTgymAnalyzer() to bt.Cerebro()
        Add DrawDown observer to bt.Cerebro(), if not already present
        Prepare BTgymStrategy initial state
        Set agent <action> to 'hold'
        Repeat until episode termination conditions are met:
            Issue and process orders according to recieved agent action
            Perform all backtesting engine computations
            Estimate state observation 
            Eestimate env. reward
            Compose aux. information
            Check episode termination conditions
            Wait for incoming <action> message
            Send (state, reward, done, info) response
```
****
 
### <a name="notes"></a> [Notes](#contents)

 1. There is a choice: where to place most of state observation/reward estimation and prepossessing such as
    featurization, normalization, frame skipping and all other -zation: either to hide it inside environment or to do it
    inside RL algorytm?
    - E.g. while state feature estimators are commonly parts of RL algorithms, reward estimation is often taken
    directly from environment.
    In case of portfolio optimisation reward function can be tricky (not to mention state preprocessing),
    so it is reasonable to make it easyly accessable inside single module for ease of experimenting
    and hyperparameter tuning.
     - BTgym allows to do it both ways: either pass "raw" state observation and do all heavy work inside RL loop
      or put it inside get_state() and get_reward() methods.
    - To mention, it seems reasonable to pass all preprocessing work to server, since it can be done asynchronously
    with agent own computations and thus somehow speed up training.

 2. [state matrix], returned by Environment by default is 2d [m,n] numpy array of floats,
    where m - number of Backtrader Datafeed values: v[-n], v[-n+1], v[-n+2],...,v[0],
    i.e. from n steps back to present step, and every v[i] is itself a vector of m features
    (open, close,...,volume,..., mov.avg., etc.).
    - in case of n=1 process is obviously POMDP. Ensure Markov property by 'frame stacking' or/and
    employing stateful function approximators.
    - When n>1 process [somehow] approaches MDP (by means of Takens' delay embedding theorem).

 3. Why Gym, not Universe VNC environment?
    - At a glance, vnc-type environment should fit algorithmic trading extremely well.
    But to best of my knowledge, OpenAI is yet to publish its "DIY VNC environment" kit. Let's wait.

 4. Why Backtrader library, not Zipline/PyAlgotrader etc.?
    - Those are excellent platforms, but what I really like about Backtrader is clear [to me], flexible  programming logic
    and ease of customisation. You dont't need to do tricks, say, to disable automatic calendar fetching, etc.
    I mean, it's nice feature and making it easy-to-run for trading people but prevents from
    correctly running intraday trading strategies. Since RL-algo-trading is in active research stage, it's impossible to tell
    in advance which setup and logic could do the job. IMO Backtrader is just well suited for this kinds of experiments.
    Besides this framework is being actively maintained.

 5. Why Currency data by default?
    - Obviously environment is data/market agnostic. Backtesting dataset size is what matters.
    Deep Q-value algorithm, most sample efficient among deep RL, take about 1M steps just to lift off.
    1 year 1 minute FX data contains about 300K samples. Feeding dataset consisting of several years of data and
    performing random sampling [arguably]
    makes it realistic to expect algorithm to converge for intra-day or intra-week trading setting (~1500-5000 steps per episode).
    Besides, currency trading holds market liquidity and impact assumptions.
    - That's just preliminary assumption, not proved at all!

 6. Note for backtrader users:
    - There is a shift on meaning 'Backtrader Strategy' in case of reinforcement learning: BtgymStrategy is mostly used for
    technical and service tasks, like data preparation and order executions, while all trading decisions are taken
    by RL agent.

 7. On current implementation: 
    - my commit was to treat backtrader engine as black box and create wrapper using explicitly
    defined and documented methods only. While it is not efficiency-optimised approach, I think
    it is still decent alpha-solution.
****
   
    
### <a name="reference"></a> [Reference*](#contents)
###### *- very incomplete, refer to source files!

### class BTgymEnv(gym.Env, args):
   OpenAI Gym environment wrapper for Backtrader framework.
   See source code comments for parameters definitions.
#### Methods:

#### reset():
Implementation of OpenAI Gym `env.reset()` method.
'Rewinds' backtrader server and starts new episode
within randomly selected time period. Returns initial environment observation.

#### step(action):
Implementation of OpenAI Gym `env.step()` method.
Relies on remote backtrader server for actual environment dynamics computing.
Accepts:
`'buy', 'sell', 'hold', 'close'` - actions;
Returns:
- response - `tuple (O, R, D, I)`:
    - `OBSERVATION` - observation of the current environment state, could be any tensor;
        default is [4,m] array of < fl32 >, where:
        - m - num. of last datafeed values,
        - 4 - num. of data features (O, H, L, C  price values).
    - `REWARD` - current portfolio statistics for environment reward estimation;
    - `DONE` - episode termination flag;
    - `INFO` - auxiliary information.

#### close():
Implementation of OpenAI Gym `env.close()` method.
Stops BTgym server process. Environment instance can be 're-opened' by simply calling `env.reset()`

#### get_stat():
Returns last episode statistics.
Currently, returns `dict` of results, obtained from calling all
attached to `Cerebro()` analyzers by their `get_analysis()` methods.
See backtrader docs for analyzers reference: https://www.backtrader.com/docu/analyzers/analyzers.html
- Note: when invoked, this method forces running episode to terminate.

### class BTgymStrategy():
Controls Environment inner dynamics and backtesting logic.
Any `State`, `Reward` and `Info` computation logic can be implemented by
subclassing `BTgymStrategy()` and overriding at least `get_state()`, `get_reward()`,
`get_info()`, `is_done()` and `set_datalines()` methods.
- One can always 'go deeper' and override `init()` and `next()` methods for desired
server cerebro engine behaviour, including order execution management logic etc.
- Since it is `bt.Strategy()` subclass, see:
https://www.backtrader.com/docu/strategy.html
for more information.
- Note: `bt.observers.DrawDown` observer will be automatically added [by server process]
to `BTgymStrategy` instance at runtime.

#### Methods*:
*- specific to BTgym, for general reference see:
   https://www.backtrader.com/docu/strategy.html 

#### set_datalines():
Default datalines are: `Open`, `Low`, `High`, `Close` [no `Volume`**] (see Backtrader docs).
Any other custom data lines, indicators, etc.
should be explicitly defined by overriding this method.
Invoked once by Strategy `init()`.
- This is just a convention method.
- ** - FX data contains no `Volume` information.

#### get_state():
Default state observation composer.
- Returns time-embedded environment state observation as [n,m] numpy matrix, where
    - n - number of signal features [ == `env.state_dim_0`, default is 4 ],
    - m - time-embedding length.
- One can override this method,
defining necessary calculations and returning arbitrary shaped tensor.
It's possible either to compute entire featurized environment state
or just pass raw price `data` to RL algorithm featurizer module.
- Note: `data` referes to `bt.startegy datafeeds` and should be treated as such.
Data Lines that are not default to `BTgymStrategy` should be explicitly defined by
`define_datalines()`.

#### get_reward():
Default reward estimator.
- Default implementation: returns amplified one-step portfolio value difference.
- Same as for state composer applies. Can return raw portfolio
performance statictics or enclose entire reward estimation module.

#### get_info():
Composes information part of environment response, by default returns `dict`, but can be any string/object, 
- Override to own taste.

#### get_done():
Episode termination estimator,
defines any trading logic conditions episode stop is called upon.
- It is just a structural a convention method.
- Expected to return tuple `(<is_done, type=bool>, <message, type=str>)`,
  e.g.: `(True, 'OMG! Stop it, we became too rich!')`
- Default method is empty.


#### _get_done():
Default episode termination method,
checks base conditions episode stop is called upon:
1. Reached maximum episode duration. Need to check it explicitly, because `is_done` flag
   is sent as part of environment response.
2. Got `_done` signal from outside. E.g. via `env.reset()` method invoked by outer RL algorithm.
3. Hit drawdown threshold.
 
This method shouldn't be overridden or called explicitly.
```
Runtime execution logic is:
    terminate episode if:
        get_done() returned (True, 'something')
        OR
        ANY _get_done() default condition is met.
```

#### next():
Default implementation for `BTgymStrategy` exists.
- Defines one step environment routine for server 'Episode mode'.
- At least, it should handle order execution logic according to action received.

### class BTgymDataset():
`Backtrader.CSVfeeds()` class wrapper.
- Pipes `CSV`[source]-->`pandas`[for efficient sampling]-->`bt.feeds` routine.
- Supports random episode data sampling.
- Default parameters are set to correctly parse 1 minute Forex generic ASCII
data files from www.HistData.com:
- See source code comments for parameters definitions.
- Suggested usage:
```
---user defined ---
Dataset = BTgymDataset(<filename>,<params>)
---inner BTgymServer routine---
Dataset.read_csv(<filename>)
Repeat until bored:
    EpisodeDataset = Dataset.get_sample()
    DataFeed = EpisodeDataset.to_btfeed()
    Engine = bt.Cerebro()
    Engine.adddata(DataFeed)
    Engine.run()
```
#### Methods:
#### read_csv(filename):
Populates instance by loading data from CSV file.

#### sample_random():
Randomly samples continuous subset of data.
- Returns `BTgymDataset` instance, holding single episode data with
number of records ~ max_episode_len.

#### to_btfeed():
Performs `BTgymDataset`-->`bt.feed` conversion.
- Returns Cerebro-ready `bt.datafeed` instance.

#### describe():
Returns summary dataset statisitc [for every column] as pandas dataframe. Useful for preprocessing.
- records count,
- data mean,
- data std dev,
- min value,
- 25% percentile,
- 50% percentile,
- 75% percentile,
- max value.
****

****
### <a name="issues"></a> [Current issues and limitations:](#title)

- by default, is configured to accept Forex 1 min. data from www.HistData.com;
- only random data sampling is implemented;
- no built-in dataset splitting to training/cv/testing subsets;
- only one equity/currency pair can be traded;
- ~~no 'skip-frames' implementation within environment;~~ done
- env.get_stat() method is returning strategy analyzers results only. No observers yet.
- no plotting features, except if using pycharm integration observer. Not sure if it is suited for intraday strategies.
- ~~making new environment kills all processes using specified network port. Watch out your jupyter kernels.~~ fixed 

****
### <a name="roadmap"></a> [TODO's and Road Map:](#title)
 - [x] refine logic for parameters applying priority (engine vs strategy vs kwargs vs defaults);
 - [ ] full reference docs;
 - [ ] examples;
 - [x] frame-skipping feature;
 - [ ] dataset tr/cv/t splitting feature;
 - [x] state rendering;
 - [ ] retrieving results for observers and plotting features - aka 'episode rendering';
 - [ ] tensorboard integration;
 - [ ] multiply agents asynchronous operation feature (e.g for A3C):
    -  [possibly] via dedicated data server;
 - [ ] sequential and sliding time-window sampling;
 - [ ] multiply instruments trading;
 
 
### <a name="news"></a>[News and update notes](#title)
- 25.06.17:
  Basic rendering implemented. 

- 23.06.17:
  alpha 0.0.4:
  added skip-frame feature,
  redefined parameters inheritance logic,
  refined overall stability;
  
- 17.06.17:
  first working alpha v0.0.2.

