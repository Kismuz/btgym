### Backtrader gym Environment
**Implementation of OpenAI Gym environment for Backtrader backtesting/trading library.**

Backtrader is open-source algorithmic trading library:

http://github.com/mementum/backtrader  
http://www.backtrader.com/

OpenAI Gym is...,
well, everyone knows Gym:

http://github.com/openai/gym

#### Outline:
General purpose of this wrapper is to provide gym-integrated framework for
running realistic experiments on algorithmic trading tasks, enabling simple and convinient
exploration of decision-making algorithms.

##### This work is in early development stage, any reports, feedback and suggestions are welcome.

#### Current issues and limitations:
- working alpha as of 14.06.17;
- by default, is configured to accept Forex 1 min. data from www.HistData.com;
- only random data sampling is implemented;
- no built-in dataset splitting to training/cv/testing subsets;
- only one equity/currency pair can be traded;
- no 'skip-frames' implementation within environment;
- env.get_stat() method is returning strategy analyzers results only. No observers yet.
- no plotting features, except if using pycharm integration observer. Not sure if it suited for intraday strategies.


#### Installation
Clone or copy btgym repository to local disk, cd to it and run: `pip install e .`
to instal package and dependencies.
- Btgym requires:  `gym`, `backtrader`, `pandas`, `numpy`, `pyzmq`.
- Examples requires: `scipy`, `matplotlib`.

#### Quickstart

Making environment with all possible defaults is as simple as:

```
from btgym import BTgymEnv

MyEnvironment = BTgymEnv(filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv')
 
```

Creating more working

```
import backtrader as bt
from btgym import BTgymDataset, BTgymStrategy, BTgymEnv

MyCerebro = bt.Cerebro()
MyCerebro.addstrategy(MyStrategy,
                      state_dim_time=30,
                      state_dim_0=30,
                      drawdown_call=0.5)

MyCerebro.broker.setcash(10.0)
MyCerebro.broker.setcommission(commission=0.001)
MyCerebro.addsizer(bt.sizers.SizerFix, stake=10)
MyCerebro.addobserver(bt.observers.DrawDown)


# Provide data source file:
CSVfilename = '../examples/data/DAT_ASCII_EURUSD_M1_2016.csv'

MyDataset = BTgymDataset(filename=CSVfilename,)

env =BTgymEnv(dataset=MyDataset,
              engine=MyCerebro,
              verbose=1)


```

#### General problem setting:
Consider reinforcement learning setup for equity/currency trading:
- agent action space is discrete ('buy', 'sell', 'close'[position], 'hold'[do nothing]);
- environment is episodic: maximum  episode duration and episode termination conditions
  are set;
- for every timestep of the episode agent is given environment state observation as tensor of last
  m price open/high/low/close values for every equity considered and based on that information is making
  trading decisions.
- agent's goal is to maximize cumulative capital;
- classic 'market liquidity' and 'capital impact' assumptions are met.
- environment setup is set close to real trading conditions, including commissions, order execution delays,
  trading calendar etc.

#### Data selection options for backtest agent training:
- random sampling:
  historic price change dataset is divided to training, cross-validation and testing subsets.
  Since agent actions do not influence market, it is possible to randomly sample continuous subset
  of training data for every episode. This is most data-efficient method.
  Cross-validation and testing performed later as usual on most "recent" data;
- sequential sampling:
  full dataset is feeded sequentially as if agent is performing real-time trading,
  episode by episode. Most reality-like, least data-efficient;
- sliding time-window sampling:
  mixture of above, episde is sampled randomly from comparatively short time period, sliding from
  furthest to most recent training data.
- NOTE: only random sampling is currently implemented.

#### Environment engine:
  BTgym uses Backtrader framework for actual environment computations, for extensive documentation see:
https://www.backtrader.com/docu/index.html.
In brief:
- User defines backtrading engine parameters by composing `Backtrader.Cerebro()` subclass,
  provides historic prices dataset as `BTgymDataset()` instance passes it as arguments when making BTgym environment.
  See Backtrader documentation for details.
- Environment starts separate server process responsible for rendering gym environment
  queries like `env.reset()` and `env.step()` by repeatedly sampling episodes form given dataset and running
  backtesting `Cerebro` engine on it. See OpenAI Gym documentation for details.

See notebooks in examples directory.
#### Data flow:
```
            BTgym Environment                                 RL alorithm
                                           +-+
   (episode mode)  +<-----<action>--- -----| |<--------------------------------------+
          |        |                       |e|                                       |
          +<------>+------<state observ.>->|n|--->[feature  ]---><state>--+->[agent]-+
          |        |      <      matrix >  |v|    [estimator]             |     |
          |        |                       |.|                            |     |
    [Backtrader]   +------<portfolio  >--->|s|--->[reward   ]---><reward>-+     |
    [Server    ]   |      <statistics>     |t|    [estimator]                   |
       |           |                       |e|                                  |
       |           +------<is_done>------->|p|--+>[runner]<-------------------->+
  (control mode)   |                       | |  |    |
       |           +------<aux.info>--- -->| |--+    |
       |                                   +-+       |
       +--<'_stop'><------------------->|env._stop|--+
       |                                             |
       +--<'_reset'><------------------>|env.reset|--+

```
#### Simple workflow:
1. Define backtesting `BTgymStrategy(bt.Strategy)`, which will
   control Environment inner dynamics and backtesting logic.
    - As for RL-specific part,any State,
   Reward and Info computation logic can be implemented by overriding `get_state()`, `get_reward()`,
   `get_info()`, `is_done()` and `set_datalines()` methods.
    - As for Broker/Trading specific part, custom order execution logic can be implemented, stake sizing,
      analytics tracking can be implemented as for regular `bt.Strategy()`.
2. Instantiate `Cerbro()`, add `BTgymStrategy()`, backtrader `Sizers`, `Analyzers` and `Observers` (if needed).
3. Define dataset by passing CSV datafile and parameters to BTgymDataset instance.
    - `BTgymDataset()` is simply `Backtrader.feeds` class wrapper, which pipes
    `CSV`[source]-->`pandas`[for efficient sampling]-->`bt.feeds` routine
    and implements random episode data sampling.
4. Initialize (or use `make()`, if need to register) gym environment with `Cerebro()` and `BTgymDataset()` along with other kwargs.
5. Run your favorite RL algorithm:
    - start episode by calling `env.reset()`;
    - advance one timframe of episode by calling `env.step()`, perform agent training or testing;
    - after single episode is finished, retrieve agent performance statistic by `env.get_stat()`.

**See notebooks in examples directory.**
#### Server operation details:
Backtrader server starts when `env.reset()` method is called for first time , runs as separate process, follows
simple Request/Reply pattern (every request should be paired with reply message) and operates one of two modes:
- Control mode: initial mode, accepts only `_reset`, `_stop` and `_getstat` messages. Any other message is ignored
  and replied with simple info messge. Shuts down upon recieving `_stop` via `env._stop_server()` method,
  goes to episode mode upon `_reset` (via `env.reset()`) and send last run episode statistic (if any) upon `_getstat`
  via `env.get_stat()`.
- Episode mode: runs episode according `BtGymStrategy()` logic. Accepts `action` messages,
  returns `dict.`: {`[state observation], [reward], [is_done], [aux.info]`}.
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
composes environment response and sends it back to agent. Actually, since 'no market impact' is assumed, all state
computations are performed one step ahead:

**Server loop:**
```
Initialize by receiving engine [bt.Cerebro()] and dataset [BTgymDataset()]
Repeat until received messge '_stop':
    Wait for incoming message
    If message is '_getstat':
        send episode statistics
    If message is '_reset':
        Randomly sample episode data from BTgymDataset
        Add episode data to bt.Cerebro()
        Add service BTgymAnalyzer() to bt.Cerebro()
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
            Send {state, reward, done, info} response
```
#### Reference [ incomplete, refer to source files! ]:
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
- response - `dict`:
    - `Observation` - observation of the current environment state, could be any tensor;
        default is [4,m] array of < fl32 >, where:
        - m - num. of last datafeed values,
        - 4 - num. of data features (O,H,L,C  price values).
    - `Reward` - current portfolio statistics for environment reward estimation;
    - `Done` - episode termination flag;
    - `Info` - auxiliary information.

#### close():
[kind of] Implementation of OpenAI Gym `env.close()` method.
Forces BTgymServer to go in 'Control Mode'.

#### get_stat():
Returns last episode statistics.
Currently, returns `dict` of results, obtained from calling all
attached to `Cerebro()` analyzers by their `get_analysis()` methods.
See backtrader docs for analyzers reference.
- Note:
    - Drawdown Analyzer is get attached by default.
    - When invoked, this method forces running episode to terminate.

#### _stop_server():
Stops BT server process, releases network resources.

### class BTgymStrategy():
Controls Environment inner dynamics and backtesting logic.
Any `State`, `Reward` and `Info` computation logic can be implemented by
subclassing BTgymStrategy and overriding at least `get_state()`, `get_reward()`,
`get_info()`, `is_done()` and `set_datalines()` methods.
- One can always 'go deeper' and override `init()` and `next()` methods for desired
server cerebro engine behaviour, including order execution etc.
- Since it is bt.Strategy subclass, see:
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
- Default implementation: returns one-step portfolio value difference.
- Same as for state composer applies. Can return raw portfolio
performance statictics or enclose entire reward estimation module.

#### get_info():
Composes information part of environment response, default is `dict`, can be any string/object, 
- Override as needed.

#### get_done():
Episode termination estimator,
defines any trading logic conditions episode stop is called upon,
e.g. <OMG! Stop it, we became too rich!> .
- If any desired condition is met, it should set BTgymStrategy `is_done` variable to True,
and [optionaly] set `broker_message` to some info string.
- Episode runtime termination logic is:
`ANY <get_done() condition is met> OR ANY <_get_done() default condition is met>`
- It is just a structural convention method.

#### _get_done():
Default episode termination method,
checks base conditions episode stop is called upon:
1. Reached maximum episode duration. Need to check it explicitly, because `is_done` flag
   is sent as part of environment response.
2. Got `_done` signal from outside. E.g. via `env.reset()` method invoked by outer RL algorithm.
3. Hit drawdown threshold.

#### next():
Default implementation for `BTgymStrategy` exists.
- Defines one step environment routine for server 'Episode mode'.
- At least, it should handle order execution logic according to action received.

### class BTgymDataset():
`Backtrader.CSVfeeds()` class wrapper.
- Currently pipes `CSV`[source]-->`pandas`[for efficient sampling]-->`bt.feeds` routine.
- Implements random episode data sampling.
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

### Notes:
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
    - in case of n=1 process is obviously POMDP. Ensure MDP property by 'frame stacking' or/and
    employing recurrent function approximators.
    - When n>1 process [somehow] approaches MDP (by means of Takens' delay embedding theorem).

 3. Why Gym, not Universe VNC environment?
    - For algorithmic trading, vnc-type environment should fit better.
    But to best of my knowledge, OpenAI is yet to publish its "DIY VNC environment" kit. We'll wait.

 4. Why Backtrader library, not Zipline/PyAlgotrader etc.?
    - Those are excellent platforms, but what I really like about Backtrader is clear [to me], flexible  programming logic
    and ease of customisation. You dont't need to do tricks, say, to disable automatic calendar fetching
    as with Zipline. I mean, it's nice feature and making it easy-to-run for trading people but prevents from
    correctly running intraday trading strategies. Since RL-algo-trading is in active research stage, it's impossible to tell
    in advance which setup and logic could do the job. IMO Backtrader is just well suited for this kinds of experiments.

 5. Why Currency data by default?
    - Obviously environment is data/market agnostic. Backtesting dataset size is what matters.
    Deep Q-value algorithm, most sample efficient among deep RL, take 1M steps just to lift off.
    1 year 1 minute FX data contains about 300K samples. Feeding dataset consisting of several years of data and
    performing random sampling [hopefully]
    makes it realistic to expect algorithm to converge for intra-day or intra-week trading setting (~1500-5000 steps per episode).
    Besides, currency trading holds market liquidity and impact assumptions.
    - That's just preliminary assumption, not proved at all!

 6. Mostly for backtrader users:
    - There is a shift on meaning 'Backtrader Strategy' in case of reinforcement learning: BtgymStrategy is mostly used for
    technical and service tasks, like data preparation and order executions, while all trading decisions are taken
    by RL agent.

 7. On implementattion: 
    - my commit was to treat backtrader engine as black box and create wrapper using explicitly
    defined and documented methods only. While it is obviously not efficiency-optimised approach, I think
    it is still decent alpha-solution.





