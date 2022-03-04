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
You  can find the code needed to run the experiments of the paper in the folder ``experiments``.

Supported Blocks
================
Data Generation: Random Uniform Policy, MEPOL

Data Preparation: 1-KNN Imputation, Mean Imputation

Feature Engineering: Recursive Feature Selection, Forward Feature Selection, Nystroem Map Feature Generation

Model Generation: FQI, DoubleFQI, LSPI, DQN, PPO, DDPG, SAC, GPOMDP

Metric: TD Error, Discounted Reward

Tuner: TunerGenetic, TunerOptuna