"""
This module contains the implementation of the Classes: Metric, TDError, DiscountedReward, 
TimeSeriesRollingAverageDiscountedReward, SomeSpecificMetric.

The Class Metric inherits from the Class AbstractUnit and from ABC.

The Class Metric is an abstract Class used as base class for all types of metrics.

A metric can be used in a generic block to evaluate the good-ness of what was learnt in such block, but also in the Tuners to 
evalaute a single block. 
"""

from abc import ABC, abstractmethod
import numpy as np
import copy
from joblib import Parallel, delayed

from ARLO.abstract_unit.abstract_unit import AbstractUnit
from ARLO.dataset.dataset import TabularDataSet
from ARLO.environment.environment import BaseEnvironment


class Metric(AbstractUnit, ABC):
    """
    This is an abstract Class. It is used as generic base class for all metrics. These are used to evaluate the good-ness of what 
    was learnt in any block of a pipeline. These are also used in the Tuner for guiding the tuning procedure.
    
    The metric Classes check whether what they get as input is of the correct type: the TDError metric should check that
    it is getting a TabularDataSet.
    
    This Class inherits from the Class AbstractUnit and from ABC.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """       
        Non-Parameters Members
        ----------------------               
        requires_dataset: This is true if the metric requires a dataset to work.
        
        requires_env: This is true if the metric requires an environment to work.
            
        The other parameters and non-parameters members are described in the Class AbstractUnit.
        """
       
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)

        #these two are needed for checking the consistency of the metric with an input_loader. In this abstract class i set both
        #to None, but in the specific Classes I will need to set these either to True or False.
        self.requires_dataset = None
        self.requires_env = None
        
    def __repr__(self):
        return 'Metric('+'obj_name='+str(self.obj_name)+', log_mode='+str(self.log_mode)\
               +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
               +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', seeder='+str(self.seeder)\
               +', local_prng='+str(self.local_prng)+', requires_dataset='+str(self.requires_dataset)\
               +', requires_env='+str(self.requires_env)+', logger='+str(self.logger)+')'
        
    @abstractmethod
    def evaluate(self, block_res, block=None, train_data=None, env=None):
        raise NotImplementedError
    
    @abstractmethod
    def which_one_is_better(self, block_1_eval, block_2_eval):
        raise NotImplementedError
       
        
class TDError(Metric):    
    """
    This Class implements a specific metric: given a dataset on which the model generation algorithm was learnt on it computes
    the TD error.
    
    For each episode we compute the TD error, then from the TD error we compute its square. In the end we average over all the 
    episodes.
    
    Here smaller is better.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """            
        Non-Parameters Members
        ----------------------
        eval_mean: The mean of the evaluation across the episodes.
        
        eval_var: The variance of the evaluation across the episodes.
        
        single_episode_evaluations: This is the evaluation across the episodes: this may be useful to check if an algorithm is 
                                    overfitting: it reaches the optimum but then the performance degrades.
                                    
        The other parameters and non-parameters members are described in the Class Metric.
        """
       
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
                
        #these two are needed for checking the consistency of the metric with an input_loader:
        self.requires_dataset = True
        self.requires_env = False
        
        self.eval_mean = None
        self.eval_var = None
        
        self.single_episode_evaluations = None
            
    def __repr__(self):
         return 'TDError('+'obj_name='+str(self.obj_name)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', seeder='+str(self.seeder)\
                +', local_prng='+str(self.local_prng)+', requires_dataset='+str(self.requires_dataset)\
                +', requires_env='+str(self.requires_env)+', logger='+str(self.logger)\
                +', eval_mean='+str(self.eval_mean)+', eval_var='+str(self.eval_var)+')'
        
    def evaluate(self, block_res, block=None, train_data=None, env=None):
        """
        Parameters
        ----------
        block_res: This must be an object of Class BlockOutput.
            
        block: This must be an object of a Class inheriting from the Class Block.
            
        train_data: This must be an object of a Class inheriting from the Class BaseDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.

        Returns
        -------
        ag_eval: This is the block_res evaluation. It is a float.
        """
        
        ag_eval = None
                
        if(train_data is not None):
            if(not isinstance(train_data, TabularDataSet)):
                exc_msg = '\'train_data\' must be an object of Class \'TabularDataSet\'!'
                self.logger.exception(msg=exc_msg)
                raise TypeError(exc_msg)
            
            if(block_res.policy.approximator is None):
                exc_msg = 'The \'approximator\' of the \'policy\' in \'block_res\' is \'None\'!'
                self.logger.exception(msg=exc_msg)
                raise ValueError(exc_msg)
                
            if(block_res.policy.regressor_type == 'generic_regressor'):
                exc_msg = 'The \'regressor_type\' of the \'policy\' in \'block_res\' cannot be equal to \'generic_regressor\'!'\
                          +' To compute the TDError an approximator of the Q-function is needed!'
                self.logger.exception(msg=exc_msg)
                raise ValueError(exc_msg)   
            
            self.logger.info(msg='Evaluating: '+str(block_res.obj_name))
            states, actions, rewards, next_states, absorbing_flags, last_flags = train_data.parse_data()
            
            absorbing_flags = absorbing_flags.astype('int')
            last_flags = last_flags.astype('int')

            #last_flags is made of ones and zeros
            last_masks = last_flags
        
            predicted_values_of_next_states = block_res.policy.approximator.predict(next_states)
            
            maxQ = []
            for i in range(len(train_data.dataset)):
                maxQ.append(np.max(predicted_values_of_next_states[i]))
                
            agent_rewards = block_res.policy.approximator.predict(states, actions)
            
            td_target = rewards + last_masks*train_data.gamma*maxQ

            total_err = td_target - agent_rewards
            
            #computes squared error:
            total_err = total_err**2
            
            by_eps = []
            tmp_err = 0
            for i in range(len(total_err)):
                if(last_flags[i] == 0):
                    tmp_err += total_err[i]
                else:
                    by_eps.append(tmp_err)
                    tmp_err = 0
            
            #if in the last iteration of the for-loop above I did not reset tmp_err then it means that absorbing_flag[i] was 
            #False meaning that the episode did not complete. If I did not reset tmp_err i also did not append it to by_eps hence 
            #i need to do it here:
            if(tmp_err != 0):
                by_eps.append(tmp_err)
            
            #store single episodes evaluations: this way I can see the performance over any episode
            self.single_episode_evaluations = by_eps

            ag_eval = np.mean(by_eps)
            ag_var = np.var(by_eps)
            
            self.eval_mean = ag_eval
            self.eval_var = ag_var
                        
            self.logger.info(msg='Done evaluating: '+str(block_res.obj_name))
        else:
            self.logger.error(msg='In \'TDError\' the \'train_data\' is \'None\'!')
        
        #check that ag_eval is not None
        if(ag_eval is None):
            exc_msg = 'The \'evaluate\' method of \'TDError\' returned \'None\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
        else:
            return ag_eval
        
    def which_one_is_better(self, block_1_eval, block_2_eval):
        """
        This method is needed to decide which of the two blocks is best. Since this metric (TDError) is better if minimised
        then the best block is the one with the lowest evaluation.
        
        This method is needed to decide among different blocks keeping in mind that a metric may have to be minimised 
        (TDError) or it may have to be maximised (DiscountedReward). 
        
        This method returns 0 if the first block is the best one, else it returns 1.
        """
        
        if(block_1_eval < block_2_eval):
            return 0
        else:
            return 1

            
class DiscountedReward(Metric):    
    """
    This Class implements a specific metric: given an environment and a policy (i.e: the output of a model generation block) it 
    computes the discounted reward.
    
    Here bigger is better.
    """
    
    def __init__(self, obj_name, n_episodes, env_dict_of_params=None, batch=False, seeder=2, log_mode='console', 
                 checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):
        """
        Parameters
        ----------
        n_episodes: This is the number of episodes for which to evaluate the agent.
        
        env_dict_of_params: This is a dictionary containing some members of the environment that you may want to modify only for
                            the evaluation phase. An example of application could be setting the controller_noise in the LQG 
                            environment.
                            
        batch: This is a boolean: 
                -If True then the discounted reward is computed in batch mode, namely we copy the environment for a number of
                times equal to n_episodes, and we make one step of advance at the time for all episodes. In this mode we cannot 
                parallelise since the overhead would be too much.
               
               -If False then the discounted reward is computed either in serial or in parallel: if serial then we go through one
               single episode at the time, if parallel we divide the episodes over the selected number of processes.
               
               The default is False.
                     
        Non-Parameters Members
        ----------------------
        eval_mean: The mean of the evaluation across the episodes.
        
        eval_var: The variance of the evaluation across the episodes.
        
        single_episode_evaluations: This is the evaluation across the episodes: this may be useful to check if an algorithm is 
                                    overfitting: it reaches the optimum but then the performance degrades.
        
        The other parameters and non-parameters members are described in the Class Metric.
        """
       
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)

        self.n_episodes = n_episodes 
        
        self.env_dict_of_params = env_dict_of_params
        
        self.batch = batch
        
        #these two are needed for checking the consistency of the metric with an input_loader:
        self.requires_dataset = False
        self.requires_env = True
    
        self.eval_mean = None
        self.eval_var = None
        
        self.single_episode_evaluations = None
            
    def __repr__(self):
         return 'DiscountedReward('+'obj_name='+str(self.obj_name)+', n_episodes='+str(self.n_episodes)\
                +', env_dict_of_params='+str(self.env_dict_of_params)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', seeder='+str(self.seeder)\
                +', local_prng='+str(self.local_prng)+', requires_dataset='+str(self.requires_dataset)\
                +', requires_env='+str(self.requires_env)+', logger='+str(self.logger)\
                +', eval_mean='+str(self.eval_mean)+', eval_var='+str(self.eval_var)+')'
                
    def _evaluate_some_episodes(self, block_res, local_n_episodes, env=None):  
        """
        Parameters
        ----------
        block_res: This must be an object of Class BlockOutput.
    
        local_n_episodes: This is an integer representing the number of episodes for which we need to evaluate the block_res.                        
                    
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.

        Returns
        -------
        total_rew: This is a list of floats containing the evaluation of block_res for every episode out of the local_n_episodes.
        """
        
        #saves the discounted reward of each episode in a list:
        total_rew = []
        tmp_rew = 0
        eps_counter = 0
        n_steps_in_eps = 0
                
        #reset env:
        obs = env.reset()
        while eps_counter < local_n_episodes: 
            tmp_act = block_res.policy.policy.draw_action(state=obs)
            
            obs, rew, done, _ = env.step(tmp_act)  
                            
            tmp_rew += (env.gamma**n_steps_in_eps)*rew
                
            last = not(n_steps_in_eps < env.info.horizon and not done)
            
            #i can update n_steps_in_eps here: if last == True, then it will anyway be reset to 0
            n_steps_in_eps += 1

            #if the state is terminal: absorbing or n_steps_in_eps > env.info.horizon
            if(last):
                total_rew.append(tmp_rew)
                
                tmp_rew = 0
                eps_counter += 1
                n_steps_in_eps = 0
                
                obs = env.reset()
        
        if(tmp_rew != 0):
            #if in the last iteration of the for-loop above I did not reset tmp_rew then it means that last == False meaning that 
            #the episode did not complete. If I did not reset tmp_rew i also did not append it to total_rew hence i need to do it
            #now:
            total_rew.append(tmp_rew)
                          
        return total_rew        
        
    def _non_batch_eval(self, block_res, train_data=None, env=None):
        """
        Parameters
        ----------
        block_res: This must be an object of Class BlockOutput.
                        
        train_data: This must be an object of a Class inheriting from the Class BaseDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.

        Returns
        -------
        tmp_rew: This is a numpy.array of floats containing the evaluation of block_res for every episode.
        """
        
        if(self.n_jobs == 1):
            #in the call of the parallel function i am setting: samples[agent_index], thus even if i use a single process 
            #i need to have a list otherwise i cannot index an integer:
            episodes =  [self.n_episodes]
            envs = [env]
            envs[0].set_local_prng(new_seeder=env.seeder)
        else: 
            episodes = []
            envs = []
            for i in range(self.n_jobs):
                episodes.append(int(self.n_episodes/self.n_jobs))
                envs.append(copy.deepcopy(env))
                envs[i].set_local_prng(new_seeder=env.seeder+i)
                
            episodes[-1] = self.n_episodes - sum(episodes[:-1])
            
        delayed_func = delayed(self._evaluate_some_episodes)
        parallel_generated_evals = Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)
        parallel_generated_evals = parallel_generated_evals(delayed_func(block_res, episodes[agent_index], envs[agent_index]) 
                                                            for agent_index in range(self.n_jobs))
            
        unlisted_evals = []
        for elem in parallel_generated_evals:
            for local_elem in elem:
                unlisted_evals.append(local_elem)                                                                                 
                                     
        return unlisted_evals
        
    def _batch_eval_parallel_step(self, block_res, train_data=None, env=None):
        """
        Parameters
        ----------
        block_res: This must be an object of Class BlockOutput.
                        
        train_data: This must be an object of a Class inheriting from the Class BaseDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.

        Returns
        -------
        tmp_rew: This is a numpy.array of floats containing the evaluation of block_res for every episode.
        """
        
        envs = []
        for i in range(self.n_episodes):                       
            envs.append(copy.deepcopy(env))
            envs[i].set_local_prng(new_seeder=env.seeder+i)
        
        tmp_rew = np.zeros(self.n_episodes)
        n_steps_in_eps = np.zeros(self.n_episodes)
        
        last = np.zeros(len(envs))
        
        #reset envs:
        obs = []
        for j in range(len(envs)):
            obs.append(envs[j].reset())
        
        obs = np.array(obs)
        
        while not last.all():   
            preds = block_res.policy.approximator.predict(obs)
            actions = []
            
            for j in range(len(envs)):
                if(block_res.policy.regressor_type == 'generic_regressor'):
                     #a generic_regressor directly models the policy:
                    tmp_act = preds[j]
                else:        
                    #if regressor_type is not a generic_regressor then we have a q-function approximator:
                    #here i can use ravel since the output of np.argmax is a numpy.int64 object. On an basic integer i would not 
                    #be able to call it:
                    tmp_act = np.argmax(preds[j]).ravel()
                    if(len(tmp_act) > 1):
                        tmp_act = np.array([self.local_prng.choice(tmp_act)])
                     
                actions.append(tmp_act)
                        
            parallel_generated_actions = Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)
            parallel_res = parallel_generated_actions(delayed(envs[j].step)(actions[j]) for j in range(len(envs)))

            j = 0
            for obs_j, tmp_rew_j, tmp_done, _ in parallel_res:
                if(not last[j]):                 
                    last[j] = not(n_steps_in_eps[j] < envs[j].info.horizon and not tmp_done)
                    obs[j] = obs_j
                    tmp_rew[j] += (envs[j].gamma**n_steps_in_eps[j])*tmp_rew_j
                    n_steps_in_eps[j] += 1
                    
                j += 1
                    
        return tmp_rew
            
    def _batch_eval_no_parallel(self, block_res, train_data=None, env=None):
        """
        Parameters
        ----------
        block_res: This must be an object of Class BlockOutput.
                        
        train_data: This must be an object of a Class inheriting from the Class BaseDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.

        Returns
        -------
        tmp_rew: This is a numpy.array of floats containing the evaluation of block_res for every episode.
        """
                
        envs = []
        for i in range(self.n_episodes):                       
            envs.append(copy.deepcopy(env))
            envs[i].set_local_prng(new_seeder=env.seeder+i)
        
        tmp_rew = np.zeros(self.n_episodes)
        n_steps_in_eps = np.zeros(self.n_episodes)
        
        last = np.zeros(len(envs))
        
        #reset envs:
        obs = []
        for j in range(len(envs)):
            obs.append(envs[j].reset())
        
        obs = np.array(obs)

        while not last.all():   
            preds = block_res.policy.approximator.predict(obs)
            
            for j in range(len(envs)):
                if(not last[j]): 
                    if(block_res.policy.regressor_type == 'generic_regressor'):
                        #a generic_regressor directly models the policy:
                        tmp_act = preds[j]
                    else:
                        #if regressor_type is not a generic_regressor then we have a q-function approximator:
                        #here i can use ravel since the output of np.argmax is a numpy.int64 object. On an basic integer i would 
                        #not be able to call it:
                        tmp_act = np.argmax(preds[j]).ravel()
                        if(len(tmp_act) > 1):
                            tmp_act = np.array([self.local_prng.choice(tmp_act)])
                         
                    obs_j, tmp_rew_j, tmp_done, _ = envs[j].step(tmp_act)
    
                    last[j] = not(n_steps_in_eps[j] < envs[j].info.horizon and not tmp_done)

                    obs[j] = obs_j
                    tmp_rew[j] += (envs[j].gamma**n_steps_in_eps[j])*tmp_rew_j
                    n_steps_in_eps[j] += 1
                            
        return tmp_rew
    
    def evaluate(self, block_res, block=None, train_data=None, env=None): 
        """
        Parameters
        ----------
        block_res: This must be an object of Class BlockOutput.
            
        block: This must be an object of a Class inheriting from the Class Block.
            
        train_data: This must be an object of a Class inheriting from the Class BaseDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.

        Returns
        -------
        ag_eval: This is the block_res evaluation. It is a float.
        """
        
        if(block_res.policy.policy is None):
            exc_msg = 'The \'policy\' cannot be \'None\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
        
        if(not isinstance(env, BaseEnvironment)):
            exc_msg = '\'env\' must be an object of a Class inheriting from the Class \'BaseEnvironment\'!'
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg)
                
        env_dict_of_old_values = None
        if(self.env_dict_of_params  is not None):
            #i need to get the original value of the parameters as at the end of this method i need to revert back the values:
            env_dict_of_old_values = env.get_params(params_names=list(self.env_dict_of_params.keys()))
            env.set_params(params_dict=self.env_dict_of_params)
        
        ag_eval = None
        
        if(self.n_jobs > self.n_episodes):
            log_msg = '\'n_jobs\' cannot be higher than \'n_episodes\', setting \'n_jobs\' equal to \'n_episodes\'!'
            self.logger.warning(msg=log_msg)
            self.n_jobs = self.n_episodes
                    
        if(env is not None):
            self.logger.info(msg='Evaluating: '+str(block_res.obj_name))
            
            if(self.batch):  
                if(block_res.policy.approximator is None):
                    wrn_msg = 'The policy approximator is \'None\': it probably means that the policy is not deterministic'\
                              +' and thus you cannot use the \'batch\' option! If you want to use the \'batch\' option call the'\
                              +' method \'make_policy_deterministic\' of the object of Class \'BlockOutput\' or set'\
                              +' \'deterministic_output_policy\' equal to \'True\' in the \'ModelGeneration\' block!'
                    self.logger.warning(msg=wrn_msg)
                    eps_eval = self._non_batch_eval(block_res=block_res, train_data=train_data, env=env)
                else:
                    if(self.n_jobs > 1):
                        eps_eval = self._batch_eval_parallel_step(block_res=block_res, train_data=train_data, env=env)
                    else:
                        eps_eval = self._batch_eval_no_parallel(block_res=block_res, train_data=train_data, env=env)
            else:
                eps_eval = self._non_batch_eval(block_res=block_res, train_data=train_data, env=env)
                            
            #store single episodes evaluations: this way I can see the performance over any episode
            self.single_episode_evaluations = eps_eval
                
            ag_eval = np.mean(eps_eval)
            ag_var = np.var(eps_eval)
            self.eval_mean = ag_eval
            self.eval_var = ag_var
            
            self.logger.info(msg='Done evaluating: '+str(block_res.obj_name))
        else:
            self.logger.error(msg='In \'DiscountedReward\' the \'env\' is \'None\'!')

        #check that ag_eval is not None
        if(ag_eval is None):
            exc_msg = 'The \'evaluate\' method of \'DiscountedReward\' returned \'None\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)       
        else:
            #reset non-evaluation values of the parameters
            if(self.env_dict_of_params  is not None):
                env.set_params(params_dict=env_dict_of_old_values)
        
            return ag_eval
        
    def which_one_is_better(self, block_1_eval, block_2_eval):
        """
        This method is needed to decide which of the two blocks is best. Since this metric (DiscountedReward) is better if 
        maximised then the best block is the one with the highest evaluation.
        
        This method is needed to decide among different blocks keeping in mind that a metric may have to be minimised 
        (TDError) or it may have to be maximised (DiscountedReward). 
        
        This method returns 0 if the first block is the best one, else it returns 1.
        """
        
        if(block_1_eval > block_2_eval):
            return 0
        else:
            return 1


class TimeSeriesRollingAverageDiscountedReward(Metric):    
    """
    This Class implements a specific metric: given an environment and a policy (i.e: the output of a model generation block) it 
    computes the rolling average discounted reward.
    
    Suppose we have an environment in which each horizon is made up of one day, then:
        -1. We learn the block on the first N days and evaluate it from the (N+1)-th day to the (N+M)-th day.
        -2. We learn the block on the first N+M days and evaluate it from the (N+M)-th day to the (N+2M)-th day.
        -3. We learn the block on the first N+2M days and evaluate it from the (N+2M)-th day to the (N+3M)-th day.
        -4. And so on and so forth...
    The final result is the average of the above steps.
    
    In order to be able to use this metric the provided environment must have an attribute for selecting the time step at which
    to start the episode (e.g: If the environment uses data from a single year, the time step can be a day of the year).
    
    The environment must have two members: 
        -min_time_step_for_time_series_evaluation
        -max_time_step_for_time_series_evaluation
    These two members must limit the values the time step can take when calling the method reset() of the environment.
        
    Here bigger is better.
    """
    
    def __init__(self, obj_name, n_episodes_train, n_evaluations, n_episodes_eval, n_episodes_per_fit=None, data_gen_block=None, 
                 env_dict_of_params=None, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """
        Parameters
        ----------        
        n_episodes_train: This is a number and it is the starting number of episodes to use. We train the block on n_episodes and 
                          we evaluate it on n_episodes_eval, then we train the block on (n_episodes_train+n_episodes_eval) and we 
                          evaluate it on n_episodes_eval, and so on and so forth.
             
        n_evaluations: This is a number and it is the total number of evaluations to perform. 
    
        n_episodes_eval: This is a number and it is the number of episodes for evaluation to use.
        
        n_episodes_per_fit: This is a parameter needed for learning the online model generation block, and it must be less than 
                            n_episodes. It is not used if the block to be evaluated is an offline ModelGeneration block.
                            
                            The default is None.
        
        data_gen_block: This must be an object of a Class inheriting from the Class DataGeneration. This is used if the block to
                        evaluate is an offline ModelGeneration block in which case we need a way to extract data from the given 
                        environment.
                        
                        The default is None.
    
        env_dict_of_params: This is a dictionary containing some members of the environment that you may want to modify only for
                            the evaluation phase. An example of application could be setting the controller_noise in the LQG 
                            environment.
                        
        Non-Parameters Members
        ----------------------
        eval_mean: The mean of the evaluation across the episodes.
        
        eval_var: The variance of the evaluation across the episodes.
        
        The other parameters and non-parameters members are described in the Class Metric.
        """
       
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
                
        self.n_episodes_train = n_episodes_train
        self.n_evaluations = n_evaluations
        self.n_episodes_eval = n_episodes_eval
        
        self.n_episodes_per_fit = n_episodes_per_fit
        self.data_gen_block = data_gen_block
        
        self.env_dict_of_params = env_dict_of_params
        
        #these two are needed for checking the consistency of the metric with an input_loader:
        self.requires_dataset = False
        self.requires_env = True
    
        self.eval_mean = None
        self.eval_var = None
            
    def __repr__(self):
         return 'TimeSeriesRollingAverageDiscountedReward('+'obj_name='+str(self.obj_name)\
                +', n_episodes_train='+str(self.n_episodes_train)+', n_episodes_per_fit='+str(self.n_episodes_per_fit)\
                +', n_evaluations='+str(self.n_evaluations)+', n_episodes_eval='+str(self.n_episodes_eval)\
                +', env_dict_of_params='+str(self.env_dict_of_params)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', seeder='+str(self.seeder)\
                +', local_prng='+str(self.local_prng)+', requires_dataset='+str(self.requires_dataset)\
                +', requires_env='+str(self.requires_env)+', logger='+str(self.logger)+', eval_mean='+str(self.eval_mean)\
                +', eval_var='+str(self.eval_var)+')'
                        
    def _online_blocks_time_series_rolling_eval(self, block, rolling_window_idx, original_lower_bound, train_data=None, 
                                                env=None):
        """
        Parameters
        ----------
        block: This must be an object of a Class inheriting from the Class Block.
        
        rolling_window_idx: This is an integer representing the current number of rolling window.
               
        original_lower_bound: This is an integer or float representing the original lower bound for the time step of the env.
      
        train_data: This must be an object of a Class inheriting from the Class BaseDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.
        
        Returns
        -------
        np.mean(eps_rew), np.var(eps_rew): the mean and the variance of the discounted reward over the test episodes.
        
        This method only works on online model generation blocks and as such I might be calling methods that are only present in
        those blocks.
        """
        
        local_block = copy.deepcopy(block)
        
        n_episodes_to_train_for = self.n_episodes_train + self.n_episodes_eval*rolling_window_idx
            
        #training phase:
        local_block_params = local_block.get_params()
            
        local_block_params['n_episodes'].current_actual_value = n_episodes_to_train_for 
        local_block_params['n_episodes_per_fit'].current_actual_value = self.n_episodes_per_fit
        local_block_params['n_steps'].current_actual_value = None
        local_block_params['n_steps_per_fit'].current_actual_value = None

        is_set_param_success = local_block.set_params(local_block_params)
        if(not is_set_param_success):
            err_msg = 'There was an error setting the parameters of the block!'
            self.logger.error(msg=err_msg)
            return [], []

        upper_bound = env.min_time_step_for_time_series_evaluation + env.horizon*n_episodes_to_train_for
        env.set_params({'min_time_step_for_time_series_evaluation': env.min_time_step_for_time_series_evaluation,
                        'max_time_step_for_time_series_evaluation': upper_bound})
                
        is_ok_to_learn = local_block.pre_learn_check(env=env)
        if(not is_ok_to_learn):
            exc_msg = 'There was an error in the \'pre_learn_check\' method of the online \'ModelGeneration\' block that'\
                       +' needs to be evaluated!'
            self.logger.exception(msg=exc_msg)
            raise RuntimeError(exc_msg)            
            
        fully_instantiated = local_block.full_block_instantiation(info_MDP=env.info)
        if(not fully_instantiated):
            exc_msg = 'There was an error in the \'full_block_instantiation\' method of the online \'ModelGeneration\' block'\
                       +' that needs to be evaluated!'
            self.logger.exception(msg=exc_msg)
            raise RuntimeError(exc_msg)          
        
        current_res = local_block.learn(env=env)
                 
        lower_bound = 1 + env.min_time_step_for_time_series_evaluation + env.horizon*n_episodes_to_train_for
        
        total_episodes = n_episodes_to_train_for + self.n_episodes_eval
        upper_bound = 1 + env.min_time_step_for_time_series_evaluation + env.horizon*total_episodes

        env.set_params({'min_time_step_for_time_series_evaluation': lower_bound,
                        'max_time_step_for_time_series_evaluation': upper_bound})
      
        #evaluation phase:
            #I set n_jobs to 1 because I do not want a parallel metric inside an already parallelised metric:
        local_metric = DiscountedReward(obj_name='local_metric_for_time_series', n_episodes=self.n_episodes_eval, 
                                        seeder=self.seeder, verbosity=0, n_jobs=1, job_type=self.job_type)
        if(current_res.policy.approximator is not None):
            #no parallel metric inside an already parallelised metric:
            eps_rew = local_metric._batch_eval_no_parallel(block_res=current_res, env=env)
        else:
            eps_rew = local_metric._non_batch_eval(block_res=current_res, env=env)
                        
        return np.mean(eps_rew), np.var(eps_rew)
    
    def _offline_blocks_time_series_rolling_eval(self, block, rolling_window_idx, original_lower_bound, train_data=None, 
                                                 env=None):
        """
        Parameters
        ----------
        block: This must be an object of a Class inheriting from the Class Block.
        
        rolling_window_idx: This is an integer representing the current number of rolling window.
               
        original_lower_bound: This is an integer or float representing the original lower bound for the time step of the env.
      
        train_data: This must be an object of a Class inheriting from the Class BaseDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.
                        
        Returns
        -------
        np.mean(eps_rew), np.var(eps_rew): the mean and the variance of the discounted reward over the test episodes.
        
        This method only works on offline model generation blocks and as such I might be calling methods that are only present in
        those blocks.
        """
        
        local_block = copy.deepcopy(block)
        
        n_episodes_to_train_for = self.n_episodes_train + self.n_episodes_eval*rolling_window_idx
            
        #training phase:        
        upper_bound = original_lower_bound + env.horizon*n_episodes_to_train_for
    
        env.set_params({'min_time_step_for_time_series_evaluation': original_lower_bound,
                        'max_time_step_for_time_series_evaluation': upper_bound})
        
        self.data_gen_block.pipeline_type = 'offline'
        
        is_ok_to_learn = self.data_gen_block.pre_learn_check(env=env)   
            
        if(not is_ok_to_learn):
            exc_msg = 'There was an error in the \'pre_learn_check\' method of the data generation object needed to extract'\
                       +' the data, which is used for evaluating an offline \'ModelGeneration\' block!'
            self.logger.exception(msg=exc_msg)
            raise RuntimeError(exc_msg)
            
        out = self.data_gen_block.learn(env=env) 
        extracted_train_data = out.train_data   
        
        if(not self.data_gen_block.is_learn_successful):
            exc_msg = 'There was an error in the \'learn\' method of the data generation object needed to extract'\
                       +' the data, which is used for evaluating an offline \'ModelGeneration\' block!'
            self.logger.exception(msg=exc_msg)
            raise RuntimeError(exc_msg)
        
        is_ok_to_learn = local_block.pre_learn_check(train_data=extracted_train_data)
        if(not is_ok_to_learn):
            exc_msg = 'There was an error in the \'pre_learn_check\' method of the offline \'ModelGeneration\' block that'\
                       +' needs to be evaluated!'
            self.logger.exception(msg=exc_msg)
            raise RuntimeError(exc_msg)            
            
        fully_instantiated = local_block.full_block_instantiation(info_MDP=env.info)
        if(not fully_instantiated):
            exc_msg = 'There was an error in the \'full_block_instantiation\' method of the offline \'ModelGeneration\' block'\
                       +' that needs to be evaluated!'
            self.logger.exception(msg=exc_msg)
            raise RuntimeError(exc_msg)            
            
        current_res = local_block.learn(train_data=extracted_train_data)
        
        lower_bound = 1 + original_lower_bound + env.horizon*n_episodes_to_train_for
        upper_bound = 1 + original_lower_bound + env.horizon*(n_episodes_to_train_for+self.n_episodes_eval)

        env.set_params({'min_time_step_for_time_series_evaluation': lower_bound,
                        'max_time_step_for_time_series_evaluation': upper_bound})
      
        #evaluation phase:
        #I set n_jobs to 1 because I do not want a parallel metric inside an already parallelised metric:
        local_metric = DiscountedReward(obj_name='local_metric_for_time_series', n_episodes=self.n_episodes_eval, 
                                        seeder=self.seeder, verbosity=0, n_jobs=1, job_type=self.job_type)
        
        if(current_res.policy.approximator is not None):
            #no parallel metric inside an already parallelised metric:
            eps_rew = local_metric._batch_eval_no_parallel(block_res=current_res, env=env)
        else:
            eps_rew = local_metric._non_batch_eval(block_res=current_res, env=env)
                                        
        return np.mean(eps_rew), np.var(eps_rew)
             
    def evaluate(self, block_res, block=None, train_data=None, env=None):
        """
        Parameters
        ----------
        block_res: This must be an object of Class BlockOutput.
            
        block: This must be an object of a Class inheriting from the Class Block.
            
        train_data: This must be an object of a Class inheriting from the Class BaseDataSet.
            
        env: This must be an object of a Class inheriting from the Class BaseEnvironment.

        Returns
        -------
        ag_eval: This is the block_res evaluation. It is a float.
        """
        
        if(block_res.policy.policy is None):
            exc_msg = 'The \'policy\' cannot be \'None\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
            
        if(block is None):
            exc_msg = '\'block\' cannot be \'None\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
            
        if(not isinstance(env, BaseEnvironment)):
            exc_msg = '\'env\' must be an object of a Class inheriting from the Class \'BaseEnvironment\'!'
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg)
            
        if(not hasattr(env, 'min_time_step_for_time_series_evaluation')):
            exc_msg = '\'env\' does not have the member \'min_time_step_for_time_series_evaluation\'!'
            self.logger.exception(msg=exc_msg)
            raise AttributeError(exc_msg)
                        
        original_min_max = env.get_params(params_names=['min_time_step_for_time_series_evaluation', 
                                                        'max_time_step_for_time_series_evaluation'])
        
        original_min = original_min_max['min_time_step_for_time_series_evaluation']
        original_max = original_min_max['max_time_step_for_time_series_evaluation']
        
        env_dict_of_old_values = None
        if(self.env_dict_of_params  is not None):
            #i need to get the original value of the parameters as at the end of this method i need to revert back the values:
            env_dict_of_old_values = env.get_params(params_names=list(self.env_dict_of_params.keys()))
            env.set_params(params_dict=self.env_dict_of_params)
        
        ag_eval = None
        
        if(env is not None):
            #If the verbosity is greater than 4 (in which case it is debug and we want to print everything) I do not silence the 
            #call to the inner metric. Otherwise i silence it:
            if(self.verbosity < 4):
                block.update_verbosity(new_verbosity=0)
                block.eval_metric.update_verbosity(new_verbosity=0)
        
            self.logger.info(msg='Evaluating: '+str(block_res.obj_name))
            
            env.random_reset = False
        
            eps_eval_mean = []
            eps_eval_var = []
            
            method_to_call = None
                        
            if(block.pipeline_type == 'offline'):
                if(self.data_gen_block is None):
                    exc_msg = '\'data_gen_block\' is \'None\'!'
                    self.logger.exception(msg=exc_msg)
                    raise ValueError(exc_msg)
                
                method_to_call = self._offline_blocks_time_series_rolling_eval
            else:
                if(self.n_episodes_per_fit is None):
                    exc_msg = '\'n_episodes_per_fit\' is \'None\'!'
                    self.logger.exception(msg=exc_msg)
                    raise ValueError(exc_msg)   
                    
                method_to_call = self._online_blocks_time_series_rolling_eval
                
            envs = []
            for i in range(self.n_evaluations):
                envs.append(copy.deepcopy(env))
                envs[-1].reset()
            
            parallel_generated_evals = Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)
            parallel_res_eval = parallel_generated_evals(delayed(method_to_call)(block, j, original_min, train_data, envs[j]) 
                                                                                 for j in range(self.n_evaluations))
            
            for eval_mean, eval_var in parallel_res_eval:
                eps_eval_mean.append(eval_mean)
                eps_eval_var.append(eval_var)
            
            #average mean
            ag_eval = np.mean(eps_eval_mean)
            #average variance
            ag_var = np.mean(eps_eval_var)
            
            self.eval_mean = ag_eval
            self.eval_var = ag_var
            
            self.logger.info(msg='Done evaluating: '+str(block_res.obj_name))
        else:
            self.logger.error(msg='In \'TimeSeriesRollingAverageDiscountedReward\' the \'env\' is \'None\'!')

        #check that ag_eval is not None
        if(ag_eval is None):
            exc_msg = 'The \'evaluate\' method of \'TimeSeriesRollingAverageDiscountedReward\' returned \'None\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)       
        else:
            #reset non-evaluation values of the parameters
            if(self.env_dict_of_params  is not None):
                env.set_params(params_dict=env_dict_of_old_values)
            
            #reset min_time_step_for_time_series_evaluation and max_time_step_for_time_series_evaluation
            env.set_params({'min_time_step_for_time_series_evaluation': original_min,
                            'max_time_step_for_time_series_evaluation': original_max})
  
            return ag_eval
        
    def which_one_is_better(self, block_1_eval, block_2_eval):
        """
        This method is needed to decide which of the two blocks is best. Since this metric (that is still a DiscountedReward) is 
        better if maximised then the best block is the one with the highest evaluation.
        
        This method is needed to decide among different blocks keeping in mind that a metric may have to be minimised 
        (TDError) or it may have to be maximised (DiscountedReward). 
        
        This method returns 0 if the first block is the best one, else it returns 1.
        """
        
        if(block_1_eval > block_2_eval):
            return 0
        else:
            return 1
    
    
class SomeSpecificMetric(Metric):
    """
    This Class implements a specific metric: This is a placeholder to use in blocks for which there are no metrics,
    or for which you do not need a metric.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """            
        Non-Parameters Members
        ----------------------
        eval_mean: The mean of the evaluation across the episodes.
        
        eval_var: The variance of the evaluation across the episodes.
        
        The other parameters and non-parameters members are described in the Class Metric.
        """
       
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
                
        #these two are needed for checking the consistency of the metric with an input_loader:
        self.requires_dataset = False
        self.requires_env = False
        
        self.eval_mean = None
        self.eval_var = None
    
    def __repr__(self):
         return 'SomeSpecificMetric('+'obj_name='+str(self.obj_name)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', seeder='+str(self.seeder)\
                +', local_prng='+str(self.local_prng)+', requires_dataset='+str(self.requires_dataset)\
                +', requires_env='+str(self.requires_env)+', logger='+str(self.logger)\
                +', eval_mean='+str(self.eval_mean)+', eval_var='+str(self.eval_var)+')'
                
    def evaluate(self, block_res, block=None, train_data=None, env=None):
        raise NotImplementedError
        
    def which_one_is_better(self, block_1_eval, block_2_eval):
        raise NotImplementedError
