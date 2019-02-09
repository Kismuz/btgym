##### Model-based RL statistical arbitrage.

`This module contains examples related to ongoing research work and should be treated accordingly`

------------------------------------------------------------
###### Problem setup

The task is constrained to well known setup of pair statistical arbitrage trading.
An agent is operating on two asset prices which are supposed to be cointegrated
(i.e. it exists a linear combination of those time series that is stationary and
exhibit mean-reverting properties).
Such combination is further referred to as “spread”.
Agent only allowed to open a balanced “spread position”: short one asset
and long other and vice versa.
“Balanced” here means that relative amount of exposure opened on each asset
is determined by cointegration relation.

In econometrics 'cointegration' is well established and widely used concept
providing strong theoretical background for 'mean-reverting' trading paradigm.
Our key point is that this task can be cast as [at least locally] stationary
markov decision process, justifying application of RL framework.

###### Approach

Current state-of-the-art deep reinforcement learning algorithms are generally
characterized by low sampling efficiency which makes it either prohibitively
expensive or simply impossible to train on experience collected from
genuine environment.
Proposed way to satisfy these objectives is to combine model-based control
and model-free reinforcement learning methods.
The idea is to use finite set of empirically collected data to approximate
true environment dynamics by some “world” model.
The learned model is then used to generate new experience samples.
A standard model-free RL algorithm can use these samples to find a close to
optimal behavioral policy and exploit it in original environment.
Approach is appealing because it provides infinite simulated experience resulting in
cheap training and reduced overfitting.
Principal drawback is that policy learnt shows suboptimal performance due to intrinsic model
inaccuracy referred to as “model bias”.
Fighting “model bias” induced gap in performance is one modern key challenge
for reinforcement learning community.
Two complementary approaches exist: either improve model to minimise bias or
correct behavioral policy to compensate existing bias.

###### Example notebooks


1. **[Introduction to analytic data model ](./analytic_data_model_an_introduction.ipynb)**
