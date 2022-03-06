"""
This module contains the implementation of the Class DataGeneration, of the Class DataGenerationRandomUniformPolicy and of the
Class DataGenerationMEPOL. 

The Class DataGeneration inherits from the Class Block, while the Classes DataGenerationRandomUniformPolicy and 
DataGenerationMEPOL inherit from the Class DataGeneration.

The Class DataGeneration is a Class used to group all Classes that do DataGeneration.

The Class DataGenerationRandomUniformPolicy implements a specific data generation algorithm: it uses a random uniform 
distribution (i.e: a random uniform policy) to pick an action with which we probe the environment.

The Class DataGenerationMEPOL implements a specific data generation algorithm: it implements Task-Agnostic Exploration via 
Policy Gradient of a Non-Parametric State Entropy Estimate as described in https://arxiv.org/abs/2007.04640.
"""

import numpy as np
import copy
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import scipy
import scipy.special
from sklearn.neighbors import NearestNeighbors
from mushroom_rl.utils.spaces import Discrete

from ARLO.block.block_output import BlockOutput
from ARLO.block.block import Block
from ARLO.dataset.dataset import TabularDataSet
from ARLO.environment.environment import BaseWrapper
from ARLO.hyperparameter.hyperparameter import Integer, Real, Categorical


class DataGeneration(Block):
    """
    This is an abstract class. It is used as generic base class for all data generation blocks.
    """
    
    def __repr__(self):
        return 'DataGeneration'+'('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
               +', seeder='+ str(self.seeder)+', local_prng='+ str(self.local_prng)\
               +', log_mode='+str(self.log_mode)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
               +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
               +', works_on_online_rl='+str(self.works_on_online_rl)+', works_on_offline_rl='+str(self.works_on_offline_rl)\
               +', works_on_box_action_space='+str(self.works_on_box_action_space)\
               +', works_on_discrete_action_space='+str(self.works_on_discrete_action_space)\
               +', works_on_box_observation_space='+str(self.works_on_box_observation_space)\
               +', works_on_discrete_observation_space='+str(self.works_on_discrete_observation_space)\
               +', pipeline_type='+str(self.pipeline_type)+', is_learn_successful='+str(self.is_learn_successful)\
               +', is_parametrised='+str(self.is_parametrised)+', block_eval='+str(self.block_eval)\
               +', logger='+str(self.logger)+')' 
    
    def _get_sample(self, env, state, episode_steps, discrete_actions, policy=None):
        """
        Parameters
        ----------
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.
        
        state: This must be a numpy array containing the current state.

        episode_steps: This is an integer that contains the current number of steps in the current episode. This is needed for 
                       computing the last (episode terminal) flag.
            
        discrete_actions: This is a boolean and it is True if the environment has a Discrete action space, and it is False 
                          otherwise.
        
        policy: The policy to use for generating the actions to be used for interacting with the environment. If None it means
                that a random uniform policy is going to be used.
                
                This must be an object of a Class implementing the method 'def draw_action(state)'.
                
                The default is None.
            
        Returns
        -------
        A sample from the environment made of: the current state, the current taken action, the current obtained reward, the 
        next state (reached by taking the current action in the current state), the done (absorbing) flag and the last 
        (episode terminal) flag.
        """
        
        if(policy is None):
            #If the action_space is Discrete I sample from a discrete uniform distribution, else if the action_space is Box I 
            #sample from a continuous uniform distribution.
            if(discrete_actions):
                action = np.array([self.local_prng.integers(env.action_space.n)])
            else:
                action = env.sample_from_box_action_space()
        else:
            action = policy.draw_action(state=state)
            
        next_state, reward, done, _ = env.step(action)
        
        last = not(episode_steps < env.info.horizon and not done)
        
        return state, action, reward, next_state, done, last
        
    def _generate_a_dataset(self, env, n_samples, discrete_actions, policy=None):
        """
        Parameters
        ----------
        env: This is the environment, it must be an object of a Class inheriting from the Class BaseEnvironment.
     
        n_samples: This is an integer greater than or equal to 1 and it represents the number of samples to extract from the 
                   environment.
       
        discrete_actions: This is True if the environment action space is Discrete, and False otherwise.

        policy: The policy to use for generating the actions to be used for interacting with the environment. If None it means
                that a random uniform policy is going to be used.
                
                This must be an object of a Class implementing the method 'def draw_action(state)'.
                
                The default is None.
            
        Returns
        -------
        new_dataset: This is one dataset with a number of samples equal to n_samples. 
        """
        
        new_dataset = []
        state = env.reset()
        last = False
        episode_steps = 0        
        
        while len(new_dataset) < n_samples:            
            if(last):
                state = env.reset()
                episode_steps = 0

            sample = self._get_sample(env=env, state=state, episode_steps=episode_steps, discrete_actions=discrete_actions,
                                      policy=policy)
            new_dataset.append(sample)

            state = sample[3]
            last = sample[5]  
            episode_steps += 1
                        
        return new_dataset
    
    def pre_learn_check(self, train_data=None, env=None):
        """
        Parameters
        ----------
        train_data: This can be a dataset that will be used for training. It must be an object of a Class inheriting from Class
                    BaseDataSet.
                    
                    The default is None.
                                              
        env: This must be a simulator/environment. It must be an object of a Class inheriting from Class BaseEnvironment.
        
             The default is None.

        Returns
        -------
        pre_learn_check_outcome: This is either True or False. It is True if the call to the method pre_learn_check() implemented
                                 in the Class Block was successful.
        """
        
        #i need to reload the params to the value it was originally
        self.algo_params = self.algo_params_upon_instantiation
        
        pre_learn_check_outcome = super().pre_learn_check(train_data=train_data, env=env)
                
        return pre_learn_check_outcome 
    
    def learn(self, train_data=None, env=None):  
        """
        Parameters
        ----------
        train_data: This can be a dataset that will be used for training. It must be an object of a Class inheriting from Class
                    BaseDataSet.
                    
                    The default is None.
                                              
        env: This must be a simulator/environment. It must be an object of a Class inheriting from Class BaseEnvironment.
        
             The default is None.
        
        Returns
        -------
        If the call to the method learn() implemented in the Class Block was successful it returns the env. Else it returns an
        empty object of Class BlockOutput (to signal that something up the chain went wrong).
        """
             
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None:       
        tmp_out = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(tmp_out, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        #i need an env to perform data generation:
        if(env is None):
            self.is_learn_successful = False
            self.logger.error(msg='The \'env\' must not be \'None\'!')
            return BlockOutput(obj_name=self.obj_name)
        
        #below i select only the inputs relevant to the current type of block: DataGeneration blocks work on the environment:       
        return env
    
    def get_params(self):
        """
        Returns
        -------
        A deep copy of the parameters in the dictionary self.algo_params.
        """
        
        return copy.deepcopy(self.algo_params)
    
    def set_params(self, new_params):
        """
        Parameters
        ----------
        new_params: The new parameters to be used in the specific data generation algorithm. It must be a dictionary that does 
                    not contain any dictionaries(i.e: all parameters must be at the same level).
                    
        Returns
        -------
        bool: This method returns True if new_params is set correctly, and False otherwise.
        """
        
        if(new_params is not None):   
            self.algo_params = new_params
            
            current_params = self.get_params()
            
            if(current_params is not None):
                self.algo_params_upon_instantiation = copy.deepcopy(current_params)
                return True
            else:
                self.logger.error(msg='There was an error getting the parameters!')
                return False
        else:
            self.logger.error(msg='Cannot set parameters: \'new_params\' is \'None\'!')
            return False            
        
    def analyse(self):
        """
        This method is yet to be implemented.
        """
        
        raise NotImplementedError
        
        
class DataGenerationRandomUniformPolicy(DataGeneration):
    """
    This Class implements a specific data generation algorithm: it uses a random uniform policy to pick an action with which we
    probe the environment.
    """
    
    def __init__(self, eval_metric, obj_name, seeder=2, algo_params=None, log_mode='console', checkpoint_log_path=None, 
                 verbosity=3, n_jobs=1, job_type='process'):
        """        
        Parameters
        ----------  
        algo_params: This contains the parameters of the algorithm implemented in this class.
                    
                     The default is None.                                                         
                     
                     If None then the following parameters will be used:
                     'n_samples': 100000 
                     This is the number of samples to extract.
                     
        Non-Parameters Members
        ----------------------                     
        algo_params_upon_instantiation: This a copy of the original value of algo_params, namely the value of algo_params that 
                                        the object got upon creation. This is needed for re-loading objects.
        
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.works_on_online_rl = False
        self.works_on_offline_rl = True
        self.works_on_box_action_space = True
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = True
                
        #this block has parameters but there is no meaning in tuning them, I can tune the number of samples directly in a 
        #ModelGeneration block via the parameter n_train_samples
        self.is_parametrised = False
        
        self.algo_params = algo_params
        
        if(self.algo_params is None):
            self.algo_params = {'n_samples': Integer(hp_name='n_samples', obj_name='n_samples_data_gen', 
                                                     current_actual_value=100000)} 
            
        self.algo_params_upon_instantiation = copy.deepcopy(self.algo_params)
        
    def __repr__(self):
        return str(self.__class__.__name__)+'('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
               +', seeder='+ str(self.seeder)+', local_prng='+ str(self.local_prng)+', algo_params='+str(self.algo_params)\
               +', log_mode='+str(self.log_mode)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
               +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
               +', works_on_online_rl='+str(self.works_on_online_rl)+', works_on_offline_rl='+str(self.works_on_offline_rl)\
               +', works_on_box_action_space='+str(self.works_on_box_action_space)\
               +', works_on_discrete_action_space='+str(self.works_on_discrete_action_space)\
               +', works_on_box_observation_space='+str(self.works_on_box_observation_space)\
               +', works_on_discrete_observation_space='+str(self.works_on_discrete_observation_space)\
               +', pipeline_type='+str(self.pipeline_type)+', is_learn_successful='+str(self.is_learn_successful)\
               +', is_parametrised='+str(self.is_parametrised)+', block_eval='+str(self.block_eval)\
               +', algo_params_upon_instantiation='+str(self.algo_params_upon_instantiation)+', logger='+str(self.logger)+')'  
            
    def learn(self, train_data=None, env=None):
        """
        Parameters
        ----------
        train_data: This can be a dataset that will be used for training. It must be an object of a Class inheriting from Class
                    BaseDataSet.
                    
                    The default is None.
                                              
        env: This must be a simulator/environment. It must be an object of a Class inheriting from Class BaseEnvironment.
        
             The default is None.

        Returns
        -------
        This method returns an object of Class BlockOutput in which in the member train_data there is an object of Class 
        TabularDataSet where the dataset member is a list of: state, action, reward, next state, absorbing flag, epsiode 
        terminal flag. 
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_env = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_env, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        #seed the env
        starting_env.seed(self.seeder)
                
        discrete_actions = False
        if(isinstance(starting_env.info.action_space, Discrete)):
            discrete_actions = True
        
        discrete_observations = False
        if(isinstance(starting_env.info.observation_space, Discrete)):
            discrete_observations = True
            
        generated_dataset = TabularDataSet(dataset=None, observation_space=starting_env.info.observation_space,
                                           action_space=starting_env.info.action_space, discrete_actions=discrete_actions, 
                                           discrete_observations=discrete_observations, gamma=starting_env.info.gamma, 
                                           horizon=starting_env.info.horizon, obj_name=self.obj_name+str('_generated_dataset'),
                                           seeder=self.seeder, log_mode=self.log_mode, 
                                           checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
        
        if(self.n_jobs > self.algo_params['n_samples'].current_actual_value):
            self.logger.warning(msg='\'n_jobs\' cannot be higher than \'n_samples\', setting \'n_jobs\' equal to \'n_samples\'!')
            self.n_jobs = self.algo_params['n_samples'].current_actual_value
            
        if(self.n_jobs == 1):
            #in the call of the parallel function i am setting: samples[agent_index], thus even if i use a single process i need
            #to have a list otherwise i cannot index an integer:
            samples = [self.algo_params['n_samples'].current_actual_value]
            envs = [starting_env]
        else: 
            samples = []
            envs = []
            for i in range(self.n_jobs):
                samples.append(int(self.algo_params['n_samples'].current_actual_value/self.n_jobs))
                envs.append(copy.deepcopy(starting_env))
                envs[i].set_local_prng(new_seeder=starting_env.seeder+i)
                
            samples[-1] = self.algo_params['n_samples'].current_actual_value - sum(samples[:-1])

        parallel_generated_datasets = Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)
        parallel_generated_datasets = parallel_generated_datasets(delayed(self._generate_a_dataset)(envs[agent_index], 
                                                                                                    samples[agent_index],
                                                                                                    discrete_actions,
                                                                                                    None) 
                                                                  for agent_index in range(self.n_jobs))
        stacked_dataset = []
        for n in range(len(parallel_generated_datasets)):
            #concatenates the two lists:
            stacked_dataset += parallel_generated_datasets[n]
        
        generated_dataset.dataset = stacked_dataset
        
        #transforms tuples to lists. This is needed since tuples are immutable:
        generated_dataset.tuples_to_lists()
        
        res = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                          checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity, train_data=generated_dataset)
        
        self.is_learn_successful = True
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object learnt successfully!')
        return res
        

class DataGenerationMEPOL(DataGeneration):
    """
    The Class DataGenerationMEPOL implements a specific data generation algorithm: it implements Task-Agnostic Exploration via 
    Policy Gradient of a Non-Parametric State Entropy Estimate as described in cf. https://arxiv.org/abs/2007.04640.
    
    The following code is a readaptation of the code present in the GitHub repository associated with the above paper, namely 
    cf. https://github.com/muttimirco/mepol
    """
    
    def __init__(self, eval_metric, obj_name, seeder=2, algo_params=None, log_mode='console', checkpoint_log_path=None, 
                 verbosity=3, n_jobs=1, job_type='process'):
        """        
        Parameters
        ----------  
        algo_params: This contains the parameters of the algorithm implemented in this class.
                    
                     The default is None.                                                         
                     
                     If None then the following parameters will be used:
                     'n_samples': 100000,
                     'train_steps': 100,
                     'batch_size': 5000,
                     'create_policy_learning_rate': 0.00025,
                     'state_filter': None,
                     'zero_mean_start': 1,
                     'hidden_sizes': [300, 300],
                     'activation': ReLU,
                     'log_std_init': -0.5,
                     'eps': 1e-15,
                     'k': 30,
                     'kl_threshold': 0.1,
                     'max_off_iters': 20,
                     'use_backtracking': 1,
                     'backtrack_coeff': 2,
                     'max_backtrack_try': 10,
                     'learning_rate': 1e-3,
                     'num_traj': 20,
                     'traj_len': 500,
                     'num_epochs': 100,
                     'optimizer': Adam,
                     'full_entropy_traj_scale': 2, 
                     'full_entropy_k': 4
                     
                     Where:
                     -n_samples: number of samples to extract.      
                     -train_steps: used in the function create_policy when zero_mean_start is True.
                     -batch size: used in the function create_policy when zero_mean_start is True.
                     -create_policy_learning_rate: used in the function create_policy when zero_mean_start is True.
                     -zero_mean_start: whether to make the policy start from a zero mean output. Either 1 or 0.
                     -hidden_sizes: hidden layer sizes.
                     -activation: activation function used in the hidden layers.
                     -log_std_init: log_std initialization for GaussianPolicy.
                     -state_filter: list of indices representing the set of features over which entropy is maximized.                                 
                                    state_filter equal to None means that no filtering is done.
                     -eps: epsilon factor to deal with KNN aliasing.
                     -k: the number of neighbors.
                     -kl_threshold: the threshold after which the behavioral policy is updated.
                     -max_off_iters: maximum number of off policy optimization steps.
                     -use_backtracking: whether to use backtracking or not. Either 1 or 0.
                     -max_backtrack_try: maximum number of backtracking try.
                     -num_traj: the batch of trajectories used in off policy optimization.
                     -traj_len: the maximum length of each trajectory in the batch of trajectories used in off policy 
                                optimization.
                     -optimizer: torch.optim.Adam or torch.optim.RMSprop.
                     -full_entropy_traj_scale: the scale factor to be applied to the number of trajectories to compute the full 
                                               entropy.
                     -full_entropy_k: the number of neighbors used to compute the full entropy.
                     
        Non-Parameters Members
        ----------------------                     
        algo_params_upon_instantiation: This a copy of the original value of algo_params, namely the value of algo_params that 
                                        the object got upon creation. This is needed for re-loading objects.
        
        policy: This is the policy learnt by this block which is used to extract a dataset from the environment.
        
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.works_on_online_rl = False
        self.works_on_offline_rl = True
        self.works_on_box_action_space = True
        self.works_on_discrete_action_space = False
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = False
                
        #this block has parameters and I may want to tune them:
        self.is_parametrised = True
        
        self.algo_params = algo_params
        
        if(self.algo_params is None):    
            self.algo_params = {'n_samples': Integer(hp_name='n_samples', obj_name='n_samples_mepol', 
                                                     current_actual_value=100000),
                                'train_steps': Integer(hp_name='train_steps', obj_name='train_steps_mepol', 
                                                     current_actual_value=100),
                                'batch_size': Integer(hp_name='batch_size', obj_name='batch_size_mepol', 
                                                      current_actual_value=5000),
                                'create_policy_learning_rate': Real(hp_name='create_policy_learning_rate',
                                                                    obj_name='create_policy_learning_rate_mepol', 
                                                                    current_actual_value=0.00025),                                    
                                'state_filter': Categorical(hp_name='state_filter', obj_name='state_filter_mepol',
                                                            current_actual_value=None),
                                'zero_mean_start': Categorical(hp_name='zero_mean_start', obj_name='zero_mean_start_mepol',
                                                               current_actual_value=1), 
                                'hidden_sizes': Categorical(hp_name='hidden_sizes', obj_name='hidden_sizes_mepol',
                                                            current_actual_value=[300, 300]),                                    
                                'activation': Categorical(hp_name='activation', obj_name='activation_mepol',
                                                               current_actual_value=nn.ReLU),
                                'log_std_init': Real(hp_name='log_std_init', obj_name='log_std_init_mepol', 
                                                     current_actual_value=-0.5),
                                'eps': Real(hp_name='eps', obj_name='eps_mepol', current_actual_value=1e-15),
                                'k': Integer(hp_name='k', obj_name='k_mepol', current_actual_value=30), 
                                'kl_threshold': Real(hp_name='kl_threshold', obj_name='kl_threshold_mepol',
                                                     current_actual_value=0.1),
                                'max_off_iters': Integer(hp_name='max_off_iters', obj_name='max_off_iters_mepol',
                                                         current_actual_value=20), 
                                'use_backtracking': Categorical(hp_name='use_backtracking', obj_name='use_backtracking_mepol',
                                                               current_actual_value=1), 
                                'backtrack_coeff': Integer(hp_name='backtrack_coeff', obj_name='backtrack_coeff_mepol',
                                                           current_actual_value=2), 
                                'max_backtrack_try': Integer(hp_name='max_backtrack_try', obj_name='max_backtrack_try_mepol',
                                                             current_actual_value=10), 
                                'learning_rate': Real(hp_name='learning_rate', obj_name='learning_rate_mepol',
                                                      current_actual_value=1e-3),
                                'num_traj': Integer(hp_name='num_traj', obj_name='num_traj_mepol', current_actual_value=20), 
                                'traj_len': Integer(hp_name='traj_len', obj_name='traj_len_mepol', current_actual_value=500), 
                                'num_epochs': Integer(hp_name='num_epochs', obj_name='num_epochs_mepol',
                                                      current_actual_value=100), 
                                'optimizer':  Categorical(hp_name='optimizer', obj_name='optimizer_mepol',
                                                          current_actual_value=torch.optim.Adam), 
                                'full_entropy_traj_scale': Integer(hp_name='full_entropy_traj_scale', 
                                                                   obj_name='full_entropy_traj_scale_mepol',
                                                                   current_actual_value=2), 
                                'full_entropy_k': Integer(hp_name='full_entropy_k', obj_name='full_entropy_k_mepol',
                                                          current_actual_value=4)
                               } 
        
        if(self.num_traj.current_actual_value % self.n_jobs != 0): 
            exc_msg = '\'num_traj\' is not divisible by \'n_jobs\', please provide a number of trajectories that can be'\
                      +' equally split among the workers!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
            
        self.algo_params_upon_instantiation = copy.deepcopy(self.algo_params)
        
        self.policy = None
        
        class GaussianPolicy(nn.Module):
            """
            Gaussian Policy with state-independent diagonal covariance matrix.
            """
        
            def __init__(self, hidden_sizes, num_features, action_dim, log_std_init=-0.5, activation=nn.ReLU):
                super().__init__()
        
                self.activation = activation
        
                layers = []
                layers.extend((nn.Linear(num_features, hidden_sizes[0]), self.activation()))
                for i in range(len(hidden_sizes) - 1):
                    layers.extend((nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), self.activation()))
                                            
                self.net = nn.Sequential(*layers)
        
                self.mean = nn.Linear(hidden_sizes[-1], action_dim)
                self.log_std = nn.Parameter(log_std_init * torch.ones(action_dim, dtype=torch.float64))
        
                #Constants
                self.log_of_two_pi = torch.tensor(np.log(2*np.pi), dtype=torch.float64)
        
                self.initialize_weights()
        
            def initialize_weights(self):
                nn.init.xavier_uniform_(self.mean.weight)
        
                for l in self.net:
                    if isinstance(l, nn.Linear):
                        nn.init.xavier_uniform_(l.weight)
        
            def get_log_p(self, states, actions):
                mean, _ = self(states)
                long_term = self.log_of_two_pi + 2*self.log_std + ((actions - mean)**2 / (torch.exp(self.log_std)+1e-7)**2)
                
                return torch.sum(-0.5*(long_term), dim=1)
        
            def forward(self, x, deterministic=False):
                mean = self.mean(self.net(x.float()))
        
                if deterministic:
                    output = mean
                else:
                    output = mean + torch.randn(mean.size(), dtype=torch.float64) * torch.exp(self.log_std)
        
                return mean, output
        
            def predict(self, s, deterministic=False):
                with torch.no_grad():
                    s = torch.tensor(s, dtype=torch.float64).unsqueeze(0)
                    return self(s, deterministic=deterministic)[1][0]
                
        self.gaussian_policy = GaussianPolicy
        
        #Seed everything
        np.random.seed(self.seeder)
        torch.manual_seed(self.seeder)
        torch.cuda.manual_seed(self.seeder)
           
    def __repr__(self):
        return str(self.__class__.__name__)+'('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
               +', seeder='+ str(self.seeder)+', local_prng='+ str(self.local_prng)+', algo_params='+str(self.algo_params)\
               +', log_mode='+str(self.log_mode)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
               +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
               +', works_on_online_rl='+str(self.works_on_online_rl)+', works_on_offline_rl='+str(self.works_on_offline_rl)\
               +', works_on_box_action_space='+str(self.works_on_box_action_space)\
               +', works_on_discrete_action_space='+str(self.works_on_discrete_action_space)\
               +', works_on_box_observation_space='+str(self.works_on_box_observation_space)\
               +', works_on_discrete_observation_space='+str(self.works_on_discrete_observation_space)\
               +', pipeline_type='+str(self.pipeline_type)+', is_learn_successful='+str(self.is_learn_successful)\
               +', is_parametrised='+str(self.is_parametrised)+', block_eval='+str(self.block_eval)\
               +', algo_params_upon_instantiation='+str(self.algo_params_upon_instantiation)+', logger='+str(self.logger)\
               +', policy='+str(self.policy)+')'  
            
    def _collect_particles(self, env, policy, num_traj, traj_len):
        """
        This method collects num_traj * traj_len samples by running policy in the env.
        """
        
        states = np.zeros((num_traj, traj_len+1, len(self.algo_params['state_filter'].current_actual_value)), dtype=np.float32)
        actions = np.zeros((num_traj, traj_len, env.action_space.shape[0]), dtype=np.float32)
        real_traj_lengths = np.zeros((num_traj, 1), dtype=np.int32)
    
        for trajectory in range(num_traj):
            s = env.reset()
    
            for t in range(traj_len):
                states[trajectory, t] = s
                a = policy.predict(s).numpy()
    
                actions[trajectory, t] = a
    
                ns, _, done, _ = env.step(a)
                s = ns
    
                if done:
                    break
    
            states[trajectory, t+1] = s
            real_traj_lengths[trajectory] = t+1
    
        next_states = None
        len_state_filter = len(self.algo_params['state_filter'].current_actual_value)
        
        for n_traj in range(num_traj):
            traj_real_len = real_traj_lengths[n_traj].item()
            traj_next_states = states[n_traj, 1:traj_real_len+1, :].reshape(-1, len_state_filter)
            if next_states is None:
                next_states = traj_next_states
            else:
                next_states = np.concatenate([next_states, traj_next_states], axis=0)
        
        next_states = next_states[:, self.algo_params['state_filter'].current_actual_value]
    
        return states, actions, real_traj_lengths, next_states
    
    def _compute_importance_weights(self, behavioral_policy, target_policy, states, actions, num_traj, real_traj_lengths):
        """
        This method computes the importance weights used in the calculation of the entropy.
        """
        
        #Initialize to None for the first concat
        importance_weights = None
    
        #Compute the importance weights
        #build iw vector incrementally from trajectory particles
        for n_traj in range(num_traj):
            traj_length = real_traj_lengths[n_traj][0].item()
    
            traj_states = states[n_traj, :traj_length]
            traj_actions = actions[n_traj, :traj_length]
    
            traj_target_log_p = target_policy.get_log_p(traj_states, traj_actions)
            traj_behavior_log_p = behavioral_policy.get_log_p(traj_states, traj_actions)
    
            traj_particle_iw = torch.exp(torch.cumsum(traj_target_log_p - traj_behavior_log_p, dim=0))
    
            if importance_weights is None:
                importance_weights = traj_particle_iw
            else:
                importance_weights = torch.cat([importance_weights, traj_particle_iw], dim=0)
    
        #Normalize the weights
        importance_weights /= torch.sum(importance_weights)
      
        return importance_weights
    
    def _compute_entropy(self, behavioral_policy, target_policy, states, actions, num_traj, real_traj_lengths, distances, indices,
                         G, B, ns):
        """
        This method computes the entropy.
        """
        
        importance_weights = self._compute_importance_weights(behavioral_policy=behavioral_policy, target_policy=target_policy,
                                                              states=states, actions=actions, num_traj=num_traj,
                                                              real_traj_lengths=real_traj_lengths)
        #Compute objective function
        #compute weights sum for each particle
        weights_sum = torch.sum(importance_weights[indices[:, :-1]], dim=1)
        #compute volume for each particle
        volumes = (torch.pow(distances[:, self.algo_params['k'].current_actual_value], ns) 
                   * torch.pow(torch.tensor(np.pi), ns/2)) / G
        #compute entropy
        to_log = (weights_sum/(volumes + self.algo_params['eps'].current_actual_value))\
                 + self.algo_params['eps'].current_actual_value
        entropy = - torch.sum((weights_sum/self.algo_params['k'].current_actual_value) * torch.log(to_log)) + B
    
        return entropy
    
    def _compute_kl(self, behavioral_policy, target_policy, states, actions, num_traj, real_traj_lengths, distances, indices):
        """
        This method computes the KL divergence between the behavioral and the target policy.
        """
        
        importance_weights = self._compute_importance_weights(behavioral_policy=behavioral_policy, target_policy=target_policy, 
                                                              states=states, actions=actions, num_traj=num_traj, 
                                                              real_traj_lengths=real_traj_lengths)
    
        weights_sum = torch.sum(importance_weights[indices[:, :-1]], dim=1)
    
        #Compute KL divergence between behavioral and target policy
        N = importance_weights.shape[0]
        kl = (1/N)*torch.sum(torch.log(self.algo_params['k'].current_actual_value/(N*weights_sum) + 
                                       self.algo_params['eps'].current_actual_value))
    
        numeric_error = torch.isinf(kl) or torch.isnan(kl)
    
        #Minimum KL is zero
        #NOTE: do not remove epsilon factor
        kl = torch.max(torch.tensor(0.0), kl)
    
        return kl, numeric_error
    
    def _collect_particles_and_compute_knn(self, env, behavioral_policy, num_traj, traj_len):
        """
        This methods collects the particles and computes KNN needed for particle-based approximation.
        """
            
        #Collect particles using behavioral policy
        res = Parallel(n_jobs=self.n_jobs)(delayed(self._collect_particles)(env, behavioral_policy, 
                                                                            int(num_traj/self.n_jobs), traj_len)
                                           for _ in range(self.n_jobs))
        
        states, actions, real_traj_lengths, next_states = [np.vstack(x) for x in zip(*res)]
    
        #Fit knn for the batch of collected particles
        nbrs = NearestNeighbors(n_neighbors=self.algo_params['k'].current_actual_value+1, 
                                metric='euclidean', algorithm='auto', n_jobs=self.n_jobs)
        nbrs.fit(next_states)
        distances, indices = nbrs.kneighbors(next_states)
    
        #Return tensors so that downstream computations can be moved to any target device (#todo)
        states = torch.tensor(states, dtype=torch.float64)
        actions = torch.tensor(actions, dtype=torch.float64)
        next_states = torch.tensor(next_states, dtype=torch.float64)
        real_traj_lengths = torch.tensor(real_traj_lengths, dtype=torch.int64)
        distances = torch.tensor(distances, dtype=torch.float64)
        indices = torch.tensor(indices, dtype=torch.int64)
    
        return states, actions, real_traj_lengths, next_states, distances, indices
    
    def _policy_update(self, behavioral_policy, target_policy, states, actions, num_traj, traj_len, distances, indices, 
                       G, B, ns):
        """
        This method updates the policy.
        """
        
        self.algo_params['optimizer'].current_actual_value.zero_grad()
    
        #Maximize entropy
        loss = -self._compute_entropy(behavioral_policy=behavioral_policy, target_policy=target_policy, states=states, 
                                      actions=actions, num_traj=num_traj, real_traj_lengths=traj_len, distances=distances, 
                                      indices=indices, G=G, B=B, ns=ns)
    
        numeric_error = torch.isinf(loss) or torch.isnan(loss)
    
        loss.backward()
        
        self.algo_params['optimizer'].current_actual_value.step()
    
        return loss, numeric_error
    
    def _mepol(self, env, create_policy):
        """
        Parameters
        ----------
        env: This is the environment of our problem at hand. It must be an object of a Class inheriting from the Class 
             BaseEnvironment.
        
        create_policy: This is the function create_policy() defined in the learn() method of this Class.
        
        This method applies the algorithm defined in the paper cf. https://arxiv.org/abs/2007.04640
        """
        
        #Create a behavioral, a target policy and a tmp policy used to save valid target policies
        #(those with kl <= kl_threshold) during off policy opt
        behavioral_policy = create_policy(is_behavioral=True)
        
        target_policy = create_policy()
        
        last_valid_target_policy = create_policy()
        
        target_policy.load_state_dict(behavioral_policy.state_dict())
        
        last_valid_target_policy.load_state_dict(behavioral_policy.state_dict())
        
        #Set optimizer
        new_optim = self.algo_params['optimizer'].current_actual_value(target_policy.parameters(),
                                                                       lr=self.algo_params['learning_rate'].current_actual_value)
        self.algo_params['optimizer'].current_actual_value = new_optim

        #Fixed constants
        ns = len(self.algo_params['state_filter'].current_actual_value) 
        B = np.log(self.algo_params['k'].current_actual_value) - scipy.special.digamma(self.algo_params['k'].current_actual_value)
        full_B = np.log(self.algo_params['full_entropy_k'].current_actual_value)\
                 - scipy.special.digamma(self.algo_params['full_entropy_k'].current_actual_value)
        G = scipy.special.gamma(ns/2 + 1)
    
        #At epoch 0 do not optimize
        epoch = 0
        self.logger.info(msg='Epoch: 0')
    
        #Full entropy
        params = dict(env=env, behavioral_policy=behavioral_policy, 
                      num_traj=self.algo_params['num_traj'].current_actual_value\
                              *self.algo_params['full_entropy_traj_scale'].current_actual_value, 
                      traj_len=self.algo_params['traj_len'].current_actual_value)
        
        states, actions, real_traj_lengths, next_states, distances, indices = self._collect_particles_and_compute_knn(**params)
        
        with torch.no_grad():
            full_entropy = self._compute_entropy(behavioral_policy=behavioral_policy, target_policy=behavioral_policy, 
                                                 states=states, actions=actions,
                                                 num_traj=self.algo_params['num_traj'].current_actual_value\
                                                         *self.algo_params['full_entropy_traj_scale'].current_actual_value, 
                                                 real_traj_lengths=real_traj_lengths, distances=distances, indices=indices, G=G, 
                                                 B=full_B, ns=ns)
    
        # Entropy
        params = dict(env=env, behavioral_policy=behavioral_policy, num_traj=self.algo_params['num_traj'].current_actual_value, 
                      traj_len=self.algo_params['traj_len'].current_actual_value)
        states, actions, real_traj_lengths, next_states, distances, indices = self._collect_particles_and_compute_knn(**params)
    
        with torch.no_grad():
            entropy = self._compute_entropy(behavioral_policy=behavioral_policy, target_policy=behavioral_policy, 
                                            states=states, actions=actions, 
                                            num_traj=self.algo_params['num_traj'].current_actual_value, 
                                            real_traj_lengths=real_traj_lengths, distances=distances, indices=indices, G=G, B=B, 
                                            ns=ns)
    
        full_entropy = full_entropy.numpy()
        entropy = entropy.numpy()
        loss = - entropy
       
        #Main Loop
        global_num_off_iters = 0
    
        if self.algo_params['use_backtracking'].current_actual_value:
            original_lr = self.algo_params['learning_rate'].current_actual_value
    
        while epoch < self.algo_params['num_epochs'].current_actual_value:
            self.logger.info(msg='Epoch: '+str(epoch+1))
            #Off policy optimization
            kl_threshold_reached = False
            last_valid_target_policy.load_state_dict(behavioral_policy.state_dict())
            num_off_iters = 0
    
            #Collect particles to optimize off policy
            params = dict(env=env, behavioral_policy=behavioral_policy, 
                          num_traj=self.algo_params['num_traj'].current_actual_value, 
                          traj_len=self.algo_params['traj_len'].current_actual_value)
            states, actions, real_traj_lengths, next_states, distances, indices = self._collect_particles_and_compute_knn(**params)
    
            if self.algo_params['use_backtracking'].current_actual_value:
                self.algo_params['learning_rate'].current_actual_value = original_lr
    
                for param_group in self.algo_params['optimizer'].current_actual_value.param_groups:
                    param_group['lr'] = self.algo_params['learning_rate'].current_actual_value
    
                backtrack_iter = 1
            else:
                backtrack_iter = None
    
            while not kl_threshold_reached:
                #Optimize policy
                loss, numeric_error = self._policy_update(behavioral_policy=behavioral_policy, target_policy=target_policy, 
                                                          states=states, actions=actions, 
                                                          num_traj=self.algo_params['num_traj'].current_actual_value, 
                                                          traj_len=real_traj_lengths, distances=distances, indices=indices, G=G,
                                                          B=B, ns=ns)
                entropy = -loss.detach().numpy()
    
                with torch.no_grad():
                    kl, kl_numeric_error = self._compute_kl(behavioral_policy=behavioral_policy, target_policy=target_policy, 
                                                            states=states, actions=actions, 
                                                            num_traj=self.algo_params['num_traj'].current_actual_value, 
                                                            real_traj_lengths=real_traj_lengths, distances=distances, 
                                                            indices=indices)
    
                kl = kl.numpy()
    
                if(not numeric_error and not kl_numeric_error and kl <= self.algo_params['kl_threshold'].current_actual_value):
                    #Valid update
                    last_valid_target_policy.load_state_dict(target_policy.state_dict())
                    num_off_iters += 1
                    global_num_off_iters += 1
    
                else:
                    if(self.algo_params['use_backtracking'].current_actual_value):
                        #We are here either because we could not perform any update for this epoch
                        #or because we need to perform one last update
                        if(not backtrack_iter == self.algo_params['max_backtrack_try'].current_actual_value):
                            target_policy.load_state_dict(last_valid_target_policy.state_dict())
                            new_lr = original_lr/(self.algo_params['backtrack_coeff'].current_actual_value**backtrack_iter)
                            self.algo_params['learning_rate'].current_actual_value = new_lr
    
                            for param_group in self.algo_params['optimizer'].current_actual_value.param_groups:
                                param_group['lr'] = self.algo_params['learning_rate'].current_actual_value
    
                            backtrack_iter += 1
                            continue
    
                    #Do not accept the update, set exit condition to end the epoch
                    kl_threshold_reached = True
    
                if(self.algo_params['use_backtracking'].current_actual_value and (backtrack_iter > 1)):
                    # Just perform at most 1 step using backtracking
                    kl_threshold_reached = True
    
                if(num_off_iters == self.algo_params['max_off_iters'].current_actual_value):
                    # Set exit condition also if the maximum number
                    # of off policy opt iterations has been reached
                    kl_threshold_reached = True
    
                if kl_threshold_reached:
                    # Compute entropy of new policy
                    with torch.no_grad():
                        entropy = self._compute_entropy(behavioral_policy=last_valid_target_policy, 
                                                        target_policy=last_valid_target_policy, states=states, actions=actions, 
                                                        num_traj=self.algo_params['num_traj'].current_actual_value,
                                                        real_traj_lengths=real_traj_lengths, distances=distances, indices=indices, 
                                                        G=G, B=B, ns=ns)
    
                    if(torch.isnan(entropy) or torch.isinf(entropy)):
                        err_msg = 'Aborting because final entropy is either NaN or inf. There might be a problem in KNN'\
                                  +' aliasing. Try using a higher value for \'k\'!'
                        self.logger.error(msg=err_msg)
                        return None
                    else:
                        #End of epoch
                        epoch += 1
    
                        #Update behavioral policy
                        behavioral_policy.load_state_dict(last_valid_target_policy.state_dict())
                        target_policy.load_state_dict(last_valid_target_policy.state_dict())
    
                        loss = -entropy.numpy()
                        entropy = entropy.numpy()

        return behavioral_policy

    def _extract_dataset_using_policy(self, env):
        """
        Parameters
        ----------
        env: This is the environment from which we can extract a dataset using the learnt policy. It must be an object of a Class
             inheriting from the Class BaseEnvironment.
             
        Returns
        -------
        stacked_dataset: This is a list and it represents the extracted dataset: each element of the list is a sample from the
                         environment containing the current state, the current action, the reward, the next state, the absorbing 
                         state flag and the episode terminal flag.
        """
        
        self.logger.info(msg='Now extracting the dataset using the learnt policy.')
        
        if(self.n_jobs == 1):
            #in the call of the parallel function i am setting: samples[agent_index], thus even if i use a single process i need
            #to have a list otherwise i cannot index an integer:
            samples = [self.algo_params['n_samples'].current_actual_value]
            envs = [env]
        else: 
            samples = []
            envs = []
            for i in range(self.n_jobs):
                samples.append(int(self.algo_params['n_samples'].current_actual_value/self.n_jobs))
                envs.append(copy.deepcopy(env))
                envs[i].set_local_prng(new_seeder=env.seeder+i)
                
            samples[-1] = self.algo_params['n_samples'].current_actual_value - sum(samples[:-1])

        parallel_generated_datasets = Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)
        parallel_generated_datasets = parallel_generated_datasets(delayed(self._generate_a_dataset)(envs[agent_index], 
                                                                                                    samples[agent_index],
                                                                                                    False,
                                                                                                    self.policy) 
                                                                  for agent_index in range(self.n_jobs))
        stacked_dataset = []
        for n in range(len(parallel_generated_datasets)):
            #concatenates the two lists:
            stacked_dataset += parallel_generated_datasets[n]
            
        return stacked_dataset 
        
    def learn(self, train_data=None, env=None):
        """
        Parameters
        ----------
        train_data: This can be a dataset that will be used for training. It must be an object of a Class inheriting from Class
                    BaseDataSet.
                    
                    The default is None.
                                              
        env: This must be a simulator/environment. It must be an object of a Class inheriting from Class BaseEnvironment.
        
             The default is None.

        Returns
        -------
        This method returns an object of Class BlockOutput in which in the member train_data there is an object of Class 
        TabularDataSet where the dataset member is a list of: state, action, reward, next state, absorbing flag, epsiode 
        terminal flag. 
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_env = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_env, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        if(self.n_jobs > self.algo_params['n_samples'].current_actual_value):
            self.logger.warning(msg='\'n_jobs\' cannot be higher than \'n_samples\', setting \'n_jobs\' equal to \'n_samples\'!')
            self.n_jobs = self.algo_params['n_samples'].current_actual_value
        
        starting_env.set_local_prng(self.seeder)
         
        class ErgodicEnv(BaseWrapper):
          """
          Environment wrapper to ignore the done signal assuming the MDP is ergodic.
          """
          
          def __init__(self, env, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                       job_type='process'):
              
              super().__init__(env=env, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                               checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
      
          def step(self, a):
              s, r, d, i = super().step(a)
              
              return s, r, False, i
         
        env = ErgodicEnv(env=starting_env, obj_name=str(starting_env.obj_name)+'_ergodic_version')
        
        #If state_filter is None then I consider all the features
        if(self.algo_params['state_filter'].current_actual_value is None):
            self.algo_params['state_filter'].current_actual_value = np.arange(env.observation_space.shape[0]).tolist()
                    
        def create_policy(is_behavioral=False):         
            policy = self.gaussian_policy(num_features=len(self.algo_params['state_filter'].current_actual_value), 
                                          hidden_sizes=self.algo_params['hidden_sizes'].current_actual_value, 
                                          action_dim=env.action_space.shape[0],
                                          activation=self.algo_params['activation'].current_actual_value, 
                                          log_std_init=self.algo_params['log_std_init'].current_actual_value)
        
            if(is_behavioral and self.algo_params['zero_mean_start'].current_actual_value):
                optimizer = torch.optim.Adam(policy.parameters(), 
                                             lr=self.algo_params['create_policy_learning_rate'].current_actual_value)

                for _ in range(self.algo_params['train_steps'].current_actual_value):
                    optimizer.zero_grad()
                    
                    states = [env.sample_from_box_observation_space()[:len(self.algo_params['state_filter'].current_actual_value)] 
                              for _ in range(self.algo_params['batch_size'].current_actual_value)]
                    states = torch.tensor(np.array(states), dtype=torch.float64)
                    actions = policy(states)[0]
                    
                    loss = torch.mean((actions - torch.zeros_like(actions, dtype=torch.float64))**2)
                    loss.backward()
                    optimizer.step()
                    
            return policy
        
        policy = self._mepol(env=env, create_policy=create_policy)
        
        if(policy is None):
            err_msg = 'Cannot extract a dataset: the learnt policy is \'None\'!'
            self.logger.error(msg=err_msg)
            return BlockOutput(obj_name=self.obj_name)
        else:
            self.logger.info(msg='The policy was learnt.')
        
        class deterministic_policy_wrapper:
            def __init__(self, policy):
                self.policy = policy
            
            def draw_action(self, state):
                return self.policy.predict(state).numpy()
        
        self.policy = deterministic_policy_wrapper(policy=policy)

        #now extract dataset with learnt policy:
        new_dataset = self._extract_dataset_using_policy(env=starting_env)    
        
        generated_dataset = TabularDataSet(dataset=new_dataset, observation_space=env.info.observation_space,
                                           action_space=env.info.action_space, discrete_actions=False, 
                                           discrete_observations=False, gamma=env.info.gamma, 
                                           horizon=env.info.horizon, obj_name=self.obj_name+str('_generated_dataset'),
                                           seeder=self.seeder, log_mode=self.log_mode, 
                                           checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
        
        #transforms tuples to lists. This is needed since tuples are immutable:
        generated_dataset.tuples_to_lists()
        
        res = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                          checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity, train_data=generated_dataset)
        
        self.is_learn_successful = True
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object learnt successfully!')
        return res        