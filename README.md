## Backtrader gym environment
The idea is to implement  OpenAI Gym environment for Backtrader backtesting/trading library to test some
reinforcement learning algorithms in algo-trading domain.

Backtrader is open-source algorithmic trading library, well structured and maintained at:
http://github.com/mementum/backtrader
http://www.backtrader.com/

OpenAI Gym is.., well, everyone knows OpenAI.
http://github.com/openai/gym

### Update 9.06.17: Basic work done. Few days to first working alpha.
```
OUTLINE:

Proposed data flow:

            BacktraderEnv                                  RL alorithm
                                           +-+
   (episode mode)  +<------<action>------->| |<--------------------------------------+
          |        |                       |e|                                       |
          +<------>+-------<state >------->|n|--->[feature  ]---><state>--+->[agent]-+
          |        |       <matrix>        |v|    [estimator]             |     |
          |        |                       |.|                            |     |
    [Backtrader]   +-------<portfolio >--->|s|--->[reward   ]---><reward>-+     |
    [Server    ]   |       <statistics>    |t|    [estimator]                   |
       |           |                       |e|                                  |
       |           +-------<is_done>------>|p|--+>[runner]<-------------------->+
  (control mode)   |                       | |  |    |
       |           +-------<aux.info>----->| |--+    |
       |                                   +-+       |
       +--<'stop'><-------------------->|env.close|--+
       |                                             |
       +--<'reset'><------------------->|env.reset|--+


 Notes:
 1. While feature estimator and 'MDP state composer' are traditionally parts of RL algorithms,
    reward estimation is often performed inside environment. In case of portfolio optimisation
    reward function can be tricky, so it is reasonable to make it easyly accessable inside RL algorithm,
    computed by [reward estimator] module and based on some set of portfolio statisics.
 2. [state matrix], returned by Environment is 2d [m,n] array of floats, where m - number of Backtrader
    Datafeed values: v[-n], v[-n+1], v[-n+2],...,v[0] i.e. from present step to n steps back, and
    every v[i] is itself a vector of m features (open, close,...,volume,..., mov.avg., etc.).
    - in case of n=1 process is obviously POMDP. Ensure MDP property by 'frame stacking' or/and
      employing reccurent function approximators. When n>>1 process [somehow] approaches MDP (by means of
      Takens' delay embedding theorem).
    - features are defined by WorkHorseStrategy.next() method,
      wich itself lives inside bt_server_process() function, and can be customised as needed.
    - same holds for portfolio statistics.
    <<TODO: pass features and stats as parameters of environment>>
 3. Action space is discrete with basic actions: 'buy', 'sell', 'hold', 'close',
    and control action 'done' for early epidsode termination.
    <<!:very incomplete: order amounts? ordering logic not defined>>
 4. This environment is meant to be [not nessesserily] paired with Tensorforce RL library,
    that's where [runner] module came from.

 5. Why Gym, not Universe VNC environment?
    For algorithmic trading, clearly, vnc-type environment fits much better.
    But to the best of my knowledge, OpenAI yet to publish docs on custom Universe VNC environment creation.

 6. Why Backtrader library, not Zipline/PyAlgotrader etc.?
    Those are excellent platforms, but what I really like about Backtrader is open programming logic
    and ease of customisation. You dont't need to do tricks, say, to disable automatic calendar fetching
    as with Zipline. I mean, it's nice feture and very convinient for trading people but prevents from
    correctly feeding forex data. IMO Backtrader is simply better suited for this kind of experiments.

 7. Why Forex data?
    Obviously environment is data/market agnostic. Backtesting dataset size is what matters.
    Deep Q-value algorithms, sample efficient among deep RL, take 1M steps just to lift off.
    1 year 1 minute FX data contains about 300K samples.Feeding several years of data makes it realistic
    to expect algorithm to converge for intraday trading (~1000-1500 steps per episode).
    That's just preliminary experiment setup, not proved!

SERVER OPERATION:
Backtrader server starts when BacktraderEnv is instantiated, runs as separate process, follows
Request/Reply pattern (every request should be paired with reply message) and operates one of two modes:
1. Control mode: initial mode, accepts only 'reset' and 'stop' messages. Any other message is ignored
   and replied with simple info messge. Shuts down upon recieving 'stop' via environment close() method,
   goes to episode mode upon 'reset' (via env.reset()).
2. Episode mode: runs episode following WorkHorseStrategy logic and parameters. Accepts <action> messages,
   returns tuple <[state matr.], [portf.stats], [is_done], [aux.info]>.
   Finishes episode upon recieving <action>='done' or according to WorkHorseStrategy logic, falls
   back to control mode.


```


