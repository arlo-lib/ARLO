**********
ARLO
**********

**ARLO: Automated Reinforcement Learning Optimizer.**

This is the repository containing the code for running the experiments of the paper ``ARLO: A Framework for Automated Reinforcement Learning``.

What is ARLO
============
ARLO is a Python library for Automated Reinforcement Learning.

The full documentation can be downloaded `here <https://github.com/arlo-lib/ARLO/blob/main/ARLO_documentation.pdf>`_.

Installation
============
You can install ``ARLO`` via: 

.. code:: shell

    pip3 install -e /path/to/ARLO

If you don't have MuJoCo installed you need to `install <https://mujoco.org/download>`_ it. 
Moreover Python >= 3.7 is needed.

Running Experiments
===================
You can find the code needed to run the experiments of the paper in the folder ``experiments``. In order to be able to run the
experiments you need to install ``ARLO``. 

The only thing you need to configure in order to run experiments is the value of the variable ``dir_chkpath``, present in the first line
after the main guard in each script, which is the path to the folder used to save the outputs of the experiments. 

Moreover in the folder ``experiments`` there is a sub-folder named ``plotting scripts`` that contains the scripts used to generate the
plots and the tables present in the paper.

Examples
========
Before diving into the ``experiments`` you may want to checkout the folder ``examples`` where simple examples of usage of ``ARLO``
are present.

Supported Blocks
================
* Data Generation: Random Uniform Policy, MEPOL.

* Data Preparation: Identity Block, 1-KNN Imputation, Mean Imputation.

* Feature Engineering: Identity Block, Recursive Feature Selection, Forward Feature Selection, Nystroem Map Feature Generation.

* Model Generation: FQI, DoubleFQI, LSPI, DQN, PPO, DDPG, SAC, GPOMDP.

* Metric: TD Error, Discounted Reward, Time Series Rolling Discounted Reward.

* Tuner: TunerGenetic, TunerOptuna.

* Input Loader: Load same environment, Load same dataset, Load bootstrapped dataset, Load bootstrapped dataset of different lenghts
  and combinations of the above.

* Environment: Grid World, Car On Hill, CartPole, Inverted Pendulum, LQG, HalfCheetah, Ant, Hopper, Humanoid, Swimmer, Walker2d.

Other than the blocks there are also other implemented capabilities in the library: saving and loading of all objects, creations
of plots with the performance obtained throughout the learning procedure of Online Model Generation blocks, creations of heatmaps
showcasing the impact of pairs of hyper-parameters on the peformance of the optimal configuration obtained in a Tunable Unit of 
an Automatic Unit.

Why you should use ARLO
=======================
* It is well written and documented
 
* Given that AutoML (and thus AutoRL) are very computationally expensive ARLO tries to optimize as much as it can all the operations. 
  For example you can extract a dataset with a Data Generation block in parallel, you can learn RL algorithms in parallel, you can 
  evaluate blocks in parallel and so on and so forth.
 
* It is ``fully`` extendable: anything (a block, a RL algorithm, a tuner, a metric, an environment, and so on and so forth) can be 
  made up into a Block compatible with the framework and the library.
  
Practically, you are not bound to a specific set of RL algorithms, or to a specific tuner, as it happens with many AutoML libraries.

Links to resources used in the paper and in the library
=======================================================
`MushroomRL <https://github.com/MushroomRL/mushroom-rl>`_

`Optuna <https://github.com/optuna/optuna>`_

`Dam Enironment <https://github.com/AndreaTirinzoni/iw-transfer-rl>`_

`LQG environment <https://github.com/T3p/potion/blob/master/potion/envs/lq.py>`_

`Data Generation with MEPOL <https://github.com/muttimirco/mepol/tree/303fb69d90e03cbb45a4619c1ed3843735f640ba>`_