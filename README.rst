**********
ARLO: Automated Reinforcement Learning Optimizer
**********

This is the repository containing the implementation of the framework, and the code for running the experiments, of the 
paper ``ARLO: A Framework for Automated Reinforcement Learning``.

What is ARLO
============
ARLO is a Python library for Automated Reinforcement Learning.

The full documentation can be downloaded `here <https://github.com/arlo-lib/ARLO/blob/main/resources/ARLO_documentation.pdf>`_, 
while the site can be found `here <https://arlo-lib.github.io/arlo-lib/>`_.

In ARLO the most general offline and online RL pipelines are the ones represented below:

.. image:: resources/pipelines.png
   :scale: 55 %   
   
A given stage of one of the above pipelines can be run with a fixed set of hyper-paramters or it can be an automatic stage in 
which the hyper-paramters are tuned. 

Moreover it is also possible to have an automatic pipeline in which all the hyper-paramters of all the stages making up the 
pipeline are tuned.

Custom algorithms can be used in any stage for any purpose (algorithm, metric, tuner, environment, and so on, and so forth).
      
Installation
============
You can install ``ARLO`` via: 

.. code:: shell

    pip3 install -e /path/to/ARLO

If you don't have ``MuJoCo`` installed you need to `install <https://mujoco.org/download>`_ it. Moreover Python >= 3.7 is needed.

Notice that sometimes there can be problems with the installation of ``mujoco_py``. This is not related to ``ARLO`` but it is
solely related to the installation of ``mujoco_py``. 

One common issue that arises is that ``MuJoCo`` and the Python environment cannot use the same ``GLFW`` library. 
As exaplained `here <https://github.com/openai/mujoco-py/issues/495>`_, a simple fix is to remove ``libglfw.3.dylib`` from 
``/path/to/.mujoco/mujoco210/bin`` and then in that folder create a symlink by calling: 

.. code:: shell

    ln -s /usr/local/lib/python3.8/site-packages/glfw/libglfw.3.dylib libglfw.3.dylib

For more troubleshooting regarding the installation of ``mujoco_py``, see their GitHub page
`here <https://github.com/openai/mujoco-py>`_, or open an issue on the GitHub page of ``ARLO``.

The library is tested over macOS and Linux.

Running Experiments
===================
You can find the code needed to run the experiments of the paper in the folder ``experiments``. In order to be able to run the
experiments you need to install ``ARLO``. 

The only thing you need to configure in order to run experiments is the value of the variable ``dir_chkpath``, present in the first line
after the main guard in each script, which is the path to the folder used to save the outputs of the experiments. 

Moreover in the folder ``experiments`` there is a sub-folder named ``Scripts for creating plots`` that contains the scripts used to 
generate the plots and the tables present in the paper.

Examples
========
Before diving into the ``experiments`` you may want to checkout the folder ``examples`` where simple examples of usage of ``ARLO``
are present.

Supported Units
================
* Data Generation: Random Uniform Policy, MEPOL `[Mutti et al., 2021] <https://github.com/muttimirco/mepol/tree/303fb69d90e03cbb45a4619c1ed3843735f640ba>`_

* Data Preparation: Identity Block, 1-KNN Imputation, Mean Imputation.

* Feature Engineering: Identity Block, Recursive Feature Selection `[Castelletti et al., 2011] <https://re.public.polimi.it/retrieve/handle/11311/635835/161137/Castelletti%20et%20al._Unknown_Tree-based%20Variable%20Selection%20for%20Dimensionality%20Reduction%20of%20Large-scale%20Control%20Systems.pdf>`_, 
  Forward Feature Selection via Mutual Information `[Beraha et al., 2019] <https://arxiv.org/abs/1907.07384>`_, 
  Nystroem Map Feature Generation.

* Model Generation: FQI, DoubleFQI, LSPI, DQN, PPO, DDPG, SAC, GPOMDP. These are wrappers of the algorithms implemented in
  `MushroomRL <https://github.com/MushroomRL/mushroom-rl>`_.

* Metric: TD Error, Discounted Reward, Time Series Rolling Discounted Reward.

* Tuner: Genetic Algorithm, `Optuna <https://github.com/optuna/optuna>`_.

* Input Loader: Load same environment, Load same dataset, Load bootstrapped dataset, Load bootstrapped dataset of different lenghts
  and combinations of the above.

* Environment: Grid World, Car On Hill, Cart Pole, Inverted Pendulum, 
  `LQG <https://github.com/T3p/potion/blob/master/potion/envs/lq.py>`_, HalfCheetah, Ant, Hopper, Humanoid, Swimmer, Walker2d.

There are also other implemented capabilities in the library: 

* Saving and loading of all objects

* Creation of plots with the performance obtained throughout the learning procedure of Online Model Generation blocks

* Creation of heatmaps showcasing the impact of pairs of hyper-parameters on the peformance of the optimal configuration obtained
  in a Tunable Unit of an Automatic Unit. These heatmaps can be create automatically, if specified, at the end of every Tunable 
  Unit, saved in an ``html`` file, with `Plotly <https://plotly.com>`_, and are also interactive (you can play with one 
  `here <https://arlo-lib.github.io/arlo-lib/plotly_heatmap_example.html>`_). A screenshot is shown below:

.. image:: resources/plotly_example.png
   :width: 700 
 
Why you should use ARLO
=======================
* It is well written and documented
 
* Given that AutoML (and thus AutoRL) are computationally intensive ARLO tries to optimize, as much as possible, all the operations. 
  For example you can extract a dataset with a Data Generation block in parallel, you can learn RL algorithms in parallel, you can 
  evaluate blocks in parallel, and so on, and so forth.
 
* It is ``fully`` extendable: anything (a unit, a RL algorithm, a tuner, a metric, an environment, and so on, and so forth) can be 
  made up into a Block compatible with the framework and the library.
  Basically, differently from what happens with many AutoML libraries, you are ``not`` bound to a specific set of RL algorithms, 
  or to a specific tuner, and so on, and so forth.
 
Cite ARLO
=========
If you are using ARLO for your scientific publications, please cite:

.. code:: bibtex

    @article{DBLP:journals/corr/abs-2205-10416,
      author    = {Marco Mussi and
                   Davide Lombarda and
                   Alberto Maria Metelli and
                   Francesco Trov{\`{o}} and
                   Marcello Restelli},
      title     = {ARLO: A Framework for Automated Reinforcement Learning},
      journal   = {CoRR},
      volume    = {abs/2205.10416},
      year      = {2022},
      url       = {https://doi.org/10.48550/arXiv.2205.10416},
      doi       = {10.48550/arXiv.2205.10416},
      eprinttype = {arXiv},
      eprint    = {2205.10416}
   }

How to contact us
=================
For any question, drop an e-mail at marco.mussi@polimi.it
