```
...Minimizing the mean square error on future experience.  - Richard S. Sutton
```

## <a name="title"></a>BTGym
Scalable event-driven RL-friendly backtesting library. Build on top of Backtrader with OpenAI Gym environment API.


_Backtrader_ is open-source algorithmic trading library:  
GitHub: http://github.com/mementum/backtrader   
Documentation and community:  
http://www.backtrader.com/  

_OpenAI Gym_ is...,
well, everyone knows Gym:   
GitHub: http://github.com/openai/gym   
Documentation and community:  
https://gym.openai.com/  

****

### <a name="outline"></a>Outline

General purpose of this project is to provide gym-integrated framework for
running reinforcement learning experiments
in [close to] real world algorithmic trading environments.

```
DISCLAIMER:
Code presented here is research/development grade.
Can be unstable, buggy, poor performing and is subject to change.

Note that this package is neither out-of-the-box-moneymaker, nor it provides ready-to-converge RL solutions.
Think of it as framework for setting experiments with complex non-stationary stochastic environments.

As a research project BTGym in its current stage can hardly deliver easy end-user experience in as sense that
setting meaninfull  experiments will require some practical programming experience as well as general knowledge
of reinforcement learning theory.
```
****

### [News and update notes](#news)

****
### <a name="contents"></a>Contents
- [Installation](#install)
- [Quickstart](#start)
- [Description](#description)
    - [Problem setting](#problem)
    - [Data sampling approaches](#data)
- [Documentation and community](#reference)
- [Known bugs and limitations](#issues)
- [Roadmap](#roadmap)
- [Update news](#news)


****
### <a name="install"></a>[Installation](#contents)
It is highly recommended to run BTGym in designated virtual environment.

Clone or copy btgym repository to local disk, cd to it and run: `pip install -e .` to install package and all dependencies:

    git clone https://github.com/Kismuz/btgym.git

    cd btgym

    pip install -e .

To update to latest version::

    cd btgym

    git pull

    pip install --upgrade -e .

##### Notes:
1. BTGym requres **Matplotlib version 2.0.2**, downgrade your installation if you have version 2.1:

    pip install matplotlib==2.0.2

2. **LSOF utility** should be installed to your OS, which can not be the default case for some Linux distributives,
see: https://en.wikipedia.org/wiki/Lsof

****
### <a name="start"></a>[Quickstart](#contents)
Making gym environment with all parmeters set to defaults is as simple as:

```python
from btgym import BTgymEnv

MyEnvironment = BTgymEnv(filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv',)
```
Adding more controls may look like:
```python
from gym import spaces
from btgym import BTgymEnv

MyEnvironment = BTgymEnv(filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
                         episode_duration={'days': 2, 'hours': 23, 'minutes': 55},
                         drawdown_call=50,
                         state_shape=dict(raw=spaces.Box(low=0,high=1,shape=(30,4))),
                         port=5555,
                         verbose=1,
                         )
```

##### See more options at [Documentation: Quickstart >>](https://kismuz.github.io/btgym/intro.html#quickstart)

##### and how-to's in [Examples directory >>](./examples).
****
### <a name="description"></a> [General description](#contents)
#### <a name="problem"></a> Problem setting

- **Discrete actions setup:** consider setup with one riskless asset acting as broker account cash and K (by default - one) risky assets.
For every risky asset there exists track of historic price records referred as `data-line`.
Apart from assets data lines there [optionally] exists number of exogenous data lines holding some
information and statistics, e.g. economic indexes, encoded news, macroeconomic indicators, weather forecasts
etc. which are considered relevant to decision-making.
It is supposed for this setup that:
    1. there is no interest rates for any asset;
    2. broker actions are fixed-size market orders (`buy`, `sell`, `close`); short selling is permitted;
    3. transaction costs are modelled via broker commission;
    4. 'market liquidity' and 'capital impact' assumptions are met;
    6. time indexes match for all data lines provided;
- The problem is modelled as discrete-time finite-horizon partially observable Markov decision process for equity/currency trading:
    - *for every asset* traded agent action space is discrete `(0: `hold` [do nothing], 1:`buy`, 2: `sell`, 3:`close` [position])`;
    - environment is episodic: maximum  episode duration and episode termination conditions
      are set;
    - for every timestep of the episode agent is given environment state observation as tensor of last
      `m` time-embedded preprocessed values for every data-line included and emits actions according some stochastic policy.
    - agent's goal is to maximize expected cumulative capital by learning optimal policy;

- **Continuous actions setup[BETA]:** this setup closely relates to continuous portfolio optimisation problem definition;
it differs from setup above in:
    1. base broker actions are real numbers: `a[i] in [0,1], 0<=i<=K, SUM{a[i]} = 1`  for `K` risky assets added;
       each action is a market target order to adjust portfolio to get share `a[i]*100%` for `i`-th  asset;
    2. entire single-step broker action is dictionary of form:
       `{cash_name: a[0], asset_name_1: a[1], ..., asset_name_K: a[K]}`;
    3. short selling is not permitted;
- For RL it implies having continuous action space as `K+1` dim vector.


#### <a name="data"></a> Data selection options for backtest agent training:
_Notice: data shaping approach is under development, expect some changes. [7.01.18]_
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

****
### <a name="reference"></a>[Documentation and Community](#title)

- Read **[Docs and API Reference](https://kismuz.github.io/btgym/)**.
- Browse **[Development Wiki](https://github.com/Kismuz/btgym/wiki)**.
- Review opened and closed **[Issues](https://github.com/Kismuz/btgym/issues?utf8=%E2%9C%93&q=)**.
- Go to **[BTGym Slack channel](https://btgym.slack.com)**. If you are new -
[use this invite link](https://join.slack.com/t/btgym/shared_invite/zt-750fx9ky-hT0o6diVw1f4Oa1FGZLf4A)
 to join.


****
### <a name="issues"></a> [Known bugs and limitations:](#title)
- requres Matplotlib version 2.0.2;
- matplotlib backend warning: appears when importing pyplot and using `%matplotlib inline` magic
  before btgym import. It's recommended to import btacktrader and btgym first to ensure proper backend
  choice;
- not tested with Python < 3.5;
- doesn't seem to work correctly under Windows; partially done
- by default, is configured to accept Forex 1 min. data from www.HistData.com;
- ~~only random data sampling is implemented;~~
- ~~no built-in dataset splitting to training/cv/testing subsets;~~ done
- ~~only one equity/currency pair can be traded~~ done
- ~~no 'skip-frames' implementation within environment;~~ done
- ~~no plotting features, except if using pycharm integration observer.~~
    ~~Not sure if it is suited for intraday strategies.~~ [partially] done
- ~~making new environment kills all processes using specified network port. Watch out your jupyter kernels.~~ fixed

****
### <a name="roadmap"></a> [TODO's and Road Map:](#title)
 - [x] refine logic for parameters applying priority (engine vs strategy vs kwargs vs defaults);
 - [X] API reference;
 - [x] examples;
 - [x] frame-skipping feature;
 - [x] dataset tr/cv/t approach;
 - [x] state rendering;
 - [x] proper rendering for entire episode;
 - [x] tensorboard integration;
 - [x] multiply agents asynchronous operation feature (e.g for A3C):
 - [x] dedicated data server;
 - [x] multi-modal observation space shape;
 - [x] A3C implementation for BTgym;
 - [x] UNREAL implementation for BTgym;
 - [x] PPO implementation for BTgym;
 - [ ] RL^2 / MAML / DARLA adaptations - IN PROGRESS;
 - [x] learning from demonstrations; -  partially done
 - [ ] risk-sensitive agents implementation;
 - [x] sequential and sliding time-window sampling;
 - [x] multiply instruments trading;
 - [x] docker image; - CPU version, `Signalprime` contribution, 
 - [ ] TF serving model serialisation functionality;


### <a name="news"></a>[News and updates:](#title)
- 10.01.2019:
    - **docker CPU version** is now available, contributed by `Signalprime`, 
    (https://github.com/signalprime), see `btgym/docker/README.md` for details;

- 9.02.2019:
    - **Introduction to analytic data model** notebook added to [model_based_stat_arb](./examples/model_based_stat_arb/) examples folder.

- 25.01.2019: updates:
    - **lstm_policy** class now requires both `internal` and `external` observation sub-spaces to be present and allows both be one-level nested
        sub-spaces itself (was only true for `external`); all declared sub-spaces got encoded by separate convolution encoders;
    - **policy deterministic action** option is implemented for discrete action spaces and can be utilised by `syncro_runner`;
        by default it is enabled for test episodes;
    - **data_feed** classes now accept `pd.dataframes` as historic data dource via `dataframe` kwarg (was: `.csv` files only);

- 18.01.2019: updates:
    - **data model** classes are under active development to power model-based framework:
        - common statistics incremental estimator classes has been added (mean, variance, covariance, linear regression etc.);
        - incremental Singular Spectrum Analysis class implemented;
        - for a pair of asset prices, two-factor state-space model is proposed
    - new **data_feed** iterator classes has been added to provide training framework with synthetic data generated by model mentioned above;
    - **strategy_gen_6** data handling and pre-processing has been redesigned:
        - market data SSA decomposition;
        - data model state as additional input to policy
        - variance-based normalisation for broker statistics

- 11.12.2018: updates and fixes:
    - **training Launcher class** got convenience features to save and reload model parameters,
        see https://github.com/Kismuz/btgym/blob/master/examples/unreal_stacked_lstm_strat_4_11.ipynb for details
    - **combined model-based/model-free** aproach package in early development stage is added to `btgym.reserach`

- 17.11.2018: updates and fixes:
    - **minor fixes to base data provider** class episode sampling
    - **update to btgym.datafeed.synthetic** subpackage: new stochastic processes generators added etc.
    - **new btgym.research.startegy_gen_5 subpackage:**
        efficient parameter-free signal preprocessing implemented, other minor improvements

- 30.10.2018: updates and fixes:
    - **fixed numpy random state issue** causing replicating of seeds among workers on POSIX os
    - **new synthetic datafeed generators** - added simple Ornshtein-Uhlenbeck process data generating classes;
        see `btgym/datafeed/synthetic/ou.py` and `btgym/research/ou_params_space_eval` for details;

- 14.10.2018: update:
    - **base reward function redesign** -> noticeable algorithms performance gain;

- 20.07.2018: major update to package:
    - **enchancements to agent architecture**:
        - casual convolution state encoder with attention for LSTM agent;
        - dropout regularization added for conv. and LSTM layers;
    - **base strategy update**: new convention for naming `get_state` methods,  see `BaseStrategy` class for details;

    - **multiply datafeeds and assets trading** implemented in two flavors:
        - **discrete actions** space via MultiDiscreteEnv class;
        - **continious actions** space via PortfolioEnv which is closely related to
          contionious portfolio optimisation problem setup;
            - description and docs:
                - **MultiDataFeed:** https://kismuz.github.io/btgym/btgym.datafeed.html#btgym.datafeed.multi.BTgymMultiData
                - **ActionSpace:** https://kismuz.github.io/btgym/btgym.html#btgym.spaces.ActionDictSpace
                - **MultiDiscreteEnv:** https://kismuz.github.io/btgym/btgym.envs.html#btgym.envs.multidiscrete.MultiDiscreteEnv
                - **PortfolioEnv:** https://kismuz.github.io/btgym/btgym.envs.html#btgym.envs.portfolio.PortfolioEnv

            - examples:
                - **MultiDiscreteEnv:** https://github.com/Kismuz/btgym/blob/master/examples/multi_discrete_setup_intro.ipynb
                - **PortfolioEnv:** https://github.com/Kismuz/btgym/blob/master/examples/portfolio_setup_BETA.ipynb
        - **Notes on multi-asset setup**:
            - adding these features forced substantial package redesign;
              expect bugs, some backward incompatibility, broken examples etc - please report;
            - current algorithms and agents architectures are ok with multiply data lines but seem not to cope well with multi-asset setup.
              It is especially evident in case of continuous actions, where agents completely fail to converge on train data;
            - current reward function design seems inappropriate; need to reshape;
            - continuous space in `beta` and still needs some improvement, esp. for broker order execution logic as well as
              action sampling routine for continuous A3C (which is Dirichlet process by now);
            - multi-discrete space is more consistent but severely limited in number of portfolio assets (but not data-lines)
              due to exponential rise of action space cardinality;
              the option is to as use many datalines as desired while limiting portfolio to 1 - 4 assets;
            - no Guided Policy available for multi-asset setup yet - in progress;
            - all but `episode` rendering modes are temporally disabled;
            - whole thing is shamelessly resource-hungry;

- 17.02.18: First results on applying guided policy search ideas (GPS) to btgym setup can be seen
           [here](./examples/guided_a3c.ipynb).  
    - tensorboard summaries are updated with additional renderings:
      actions distribution, value function and LSTM_state; presented in the same notebook.

- 6.02.18: Common update to all a3c agents architectures:
    - all dense layers are now Noisy-Net ones,
      see: [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) paper by Fortunato at al.;
    - note that entropy regularization is still here, kept in ~0.01 to ensure proper exploration;
    - policy output distribution is 'centered' using layer normalisation technique;

        - all of the above results in about 2x training speedup in terms of train iterations;

- 20.01.18: Project [Wiki pages](https://github.com/Kismuz/btgym/wiki) added;

- 12.01.18: Minor fixes to logging, enabled BTgymDataset train/test data split. AAC framework train/test cycle enabled
            via
            [`episode_train_test_cycle`](https://kismuz.github.io/btgym/btgym.algorithms.html#module-btgym.algorithms.aac)
            kwarg.

- 7.01.18: Update:
    - Major data pipe redesign. `Domain -> Trial -> Episode` sampling routine implemented. For motivation and
      formal definitions refer to
      [Section 1.Data of this DRAFT](https://github.com/Kismuz/btgym/blob/master/docs/papers/btgym_formalism_draft.pdf),
      API [Documentation](https://kismuz.github.io/btgym/btgym.datafeed.html#btgym-datafeed-package)
      and [Intro example](./examples/data_domain_api_intro.ipynb). Changes should be backward compatible.
      In brief, it is necessry framework for upcoming meta-learning algorithms.
    - logging changes: now relying in python `logbook` module. Should eliminate errors under Windows.
    - Stacked_LSTM_Policy agent implemented. Based on NAV_A3C from
      [DeepMind paper](https://arxiv.org/pdf/1611.03673.pdf) with some minor mods. Basic usage
      [Example is here](./examples/unreal_stacked_lstm_strat_4_11.ipynb).
      Still in research code area and need further tuning; yet faster than simple LSTM agent,
      able to converge on 6-month 1m dataset.

- 5.12.17: Inner btgym comm. fixes >> speedup ~5%.

- 02.12.17: Basic `sliding time-window train/test` framework implemented via
            [BTgymSequentialTrial()](https://kismuz.github.io/btgym/btgym.html#btgym.datafeed.BTgymSequentialTrial)
            class. UPD: replaced by `BTgymSequentialDataDomain` class.

- 29.11.17: Basic meta-learning RL^2 functionality implemented.
    - See [Trial_Iterator Class](https://kismuz.github.io/btgym/btgym.html#btgym.datafeed.BTgymRandomTrial) and
    [RL^2 policy](https://kismuz.github.io/btgym/btgym.research.html#btgym.research.policy_rl2.AacRL2Policy)
    for description.
    - Effectiveness is not tested yet, examples are to follow.

- 24.11.17: A3C/UNREAL finally adapted to work with BTGym environments.
    - Examples with synthetic simple data(sine wawe) and historic financial data added,
      see [examples directory](./examples/);
    - Results on potential-based functions reward shaping in `/research/DevStartegy_4_6`;
    - Work on Sequential/random Trials Data iterators (kind of sliding time-window) in progress,
      start approaching the toughest part: non-stationarity battle is ahead.

- 14.11.17: BaseAAC framework refraction; added per worker batch-training option and LSTM time_flatten option; Atari
            examples updated; see [Documentation](https://kismuz.github.io/btgym/) for details.

- 30.10.17: Major update, some backward incompatibility:
    - BTGym now can be thougt as two-part package: one is environment itself and the other one is
      RL algoritms tuned for solving algo-trading tasks. Some basic work on shaping of later is done. Three advantage
      actor-critic style algorithms are implemented: A3C itself, it's UNREAL extension and PPO. Core logic of these seems
      to be implemented correctly but further extensive BTGym-tuning is ahead.
      For now one can check [atari tests](./examples/atari_tests).
    - Finally, basic [documentation and API reference](https://kismuz.github.io/btgym/) is now available.

- 27.09.17: A3C [test_4.2](./examples/a3c/a3c_test_4_2_no_feature_signal_conv1d.ipynb) added:
    - some progress on estimator architecture search, state and reward shaping;

- 22.09.17: A3C [test_4](./examples/a3c/a3c_test_4_sma_bank_features.ipynb) added:
    - passing train convergence test on small (1 month) dataset of EURUSD 1-minute bar data;

- 20.09.17: A3C optimised sine-wave test added [here.](./examples/a3c/a3c_reject_test_3_sine_conv1d_sma_log_grad.ipynb)
    - This notebook presents some basic ideas on state presentation, reward shaping,
      model architecture and hyperparameters choice.
      With those tweaks sine-wave sanity test is converging faster and with greater stability.

- 31.08.17: Basic implementation of A3C algorithm is done and moved inside BTgym package.
    - algorithm logic consistency tests are passed;
    - still work in early stage, experiments with obs. state features and policy estimator architecture ahead;
    - check out [`examples/a3c`](./examples/a3c) directory.

- 23.08.17: `filename` arg in environment/dataset specification now can be list of csv files.
    - handy for bigger dataset creation;
    - data from all files are concatenated and sampled uniformly;
    - no record duplication and format consistency checks preformed.

- 21.08.17: UPDATE: BTgym is now using multi-modal observation space.
     - space used is simple extension of gym: `DictSpace(gym.Space)` - dictionary (not nested yet) of core gym spaces.
     - defined in `btgym/spaces.py`.
     - `raw_state` is default Box space of OHLC prices. Subclass BTgymStrategy and override `get_state()` method to
            compute alll parts of env. observation.
     - rendering can now be performed for avery entry in observation dictionary as long as it is Box ranked <=3
            and same key is passed in reneder_modes kwarg of environment.
            'Agent' mode renamed to 'state'. See updated examples.


- 07.08.17: BTgym is now optimized for asynchronous operation with multiply environment instances.
     - dedicated data_server is used for dataset management;
     - improved overall internal network connection stability and error handling;
     - see example `async_btgym_workers.ipynb` in [`examples`](./examples) directory.

- 15.07.17: UPDATE, BACKWARD INCOMPATIBILITY: now state observation can be tensor of any rank.
     - Consequently, dim. ordering convention has changed to ensure compatibility with
            existing tf models: time embedding is first dimension from now on, e.g. state
            with shape (30, 20, 4) is 30x steps time embedded with 20 features and 4 'channels'.
            For the sake of 2d visualisation only one 'cannel' can be rendered, can be
            chosen by setting env. kwarg `render_agent_channel=0`;
     - examples are updated;
     - better now than later.

- 11.07.17: Rendering battle continues: improved stability while low in memory,
            added environment kwarg `render_enabled=True`; when set to `False`
             - all renderings are disabled. Can help with performance.

- 5.07.17:  Tensorboard monitoring wrapper added; pyplot memory leak fixed.

- 30.06.17: EXAMPLES updated with 'Setting up: full throttle' how-to.

- 29.06.17: UPGRADE: be sure to run `pip install --upgrade -e .`
    - major rendering rebuild: updated with modes: `human`, `agent`, `episode`;
      render process now performed by server and returned to environment as `rgb numpy array`.
      Pictures can be shown either via matplolib or as pillow.Image(preferred).
    - 'Rendering HowTo' added, 'Basic Settings' example updated.
    - internal changes: env. state divided on `raw_state`  - price data,
      and `state` - featurized representation. `get_raw_state()` method added to strategy.
    - new packages requirements: `matplotlib` and `pillow`.

- 25.06.17:
  Basic rendering implemented.

- 23.06.17:
  alpha 0.0.4:
  added skip-frame feature,
  redefined parameters inheritance logic,
  refined overall stability;

- 17.06.17:
  first working alpha v0.0.2.


<a href="https://stackexchange.com/users/10204071/andrew-muzikin"><img src="https://stackexchange.com/users/flair/10204071.png" width="208" height="58" alt="profile for Andrew Muzikin on Stack Exchange, a network of free, community-driven Q&amp;A sites" title="profile for Andrew Muzikin on Stack Exchange, a network of free, community-driven Q&amp;A sites" /></a>
