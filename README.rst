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

If you don't have mujoco installed you need to `install <https://mujoco.org/download>`_ it. 
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
Data Generation: Random Uniform Policy, MEPOL

Data Preparation: 1-KNN Imputation, Mean Imputation

Feature Engineering: Recursive Feature Selection, Forward Feature Selection, Nystroem Map Feature Generation

Model Generation: FQI, DoubleFQI, LSPI, DQN, PPO, DDPG, SAC, GPOMDP

Metric: TD Error, Discounted Reward

Tuner: TunerGenetic, TunerOptuna

Links to resources used for the paper and the library
=====================================================
MushroomRL
Optuna
Dam environment
LQG environment
MEPOL