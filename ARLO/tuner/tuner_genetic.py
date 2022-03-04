"""
This module contains the implementation of the Class: TunerGenetic.

The Class TunerGenetic inherits from the Class Tuner.
"""

import numpy as np
import copy
from joblib import Parallel, delayed

from ARLO.tuner.tuner import Tuner
from ARLO.rl_pipeline.rl_pipeline import RLPipeline
from ARLO.block.model_generation import ModelGeneration

        
class TunerGenetic(Tuner):
    """
    This Class implements a population based tuner, namely it implements a Genetic Algorithm.
    
    This Class inherits from the Class Tuner.
    """
    
    def __init__(self, block_to_opt, eval_metric, input_loader, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, 
                 verbosity=3, n_agents=10, n_generations=100, prob_point_mutation=0.5, tuning_mode='best_performant_elitism', 
                 pool_size=None, n_jobs=1, job_type='process', output_save_periodicity=25):
        """
        Parameters
        ----------
        n_agents: This is the number of agents in each generation. 
                  
                  The default is 10.
        
        n_generations: This is the number of generations of the genetic algorithm.
                       
                       The default is 100.
                     
        prob_point_mutation: This is the probability of mutating a certain hyper-parameter of block_to_opt.
                             
                             The default is 0.5.
                             
        tuning_mode: This is a string and it represents the tuning mode: it can be 'no_elitism', 'best_performant_elitism', or
                     'pool_elitism'. 
                     
                     If 'no_elitism' is selected then the best agent of each generation is not kept as is across generations but 
                     it undergoes mutation.
                     
                     If 'best_performant_elitism' is selected then the best agent so far is added to each generation both as is 
                     and also mutated.
                     
                     If 'pool_elitism' is selected then the most performant pool_size agents are kept across generations and are
                     used to generate the new offspring by mutating them.
                     
                     In the first two cases we perform tournament selection (note that we always reserve a spot for the best 
                     agent of the previous generation (which will be mutated)).
        
                     The default is 'best_performant_elitism'.
                             
        pool_size: This is an integer and it represents the number of best agents to use in each generation as starting point for
                   obtaining the next generation.
                     
                   The default is None.
                     
        Non-Parameters Members
        ----------------------
        trial_number: This is an integer used for keeping track of how many trials (agents) are being done. 

        The other parameters and non-parameters members are described in the Class Tuner.
        """
        
        super().__init__(block_to_opt=block_to_opt, eval_metric=eval_metric, input_loader=input_loader, obj_name=obj_name, 
                         seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, verbosity=verbosity,
                         n_jobs=n_jobs, job_type=job_type, output_save_periodicity=output_save_periodicity)
            
        self.n_agents = n_agents
        self.n_generations = n_generations
        self.prob_point_mutation = prob_point_mutation
        self.tuning_mode = tuning_mode
        
        self.pool_size = pool_size
        if(self.pool_size is not None):
            if((self.n_agents % self.pool_size) != 0):
                exc_msg = '\'n_agents\' must be an exact multiple of \'pool_size\'!'
                self.logger.exception(msg=exc_msg)
                raise ValueError(exc_msg)
        
        self.trial_number = 0
                
    def __repr__(self):
         return 'TunerGenetic('+'block_to_opt='+str(self.block_to_opt)+', eval_metric='+str(self.eval_metric)\
                +', input_loader='+str(self.input_loader)+', obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)\
                +', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', n_agents='+str(self.n_agents)\
                +', n_generations='+str(self.n_generations)+', prob_point_mutation='+str(self.prob_point_mutation)\
                +', tuning_mode='+str(self.tuning_mode)+', pool_size='+str(self.pool_size)\
                +', output_save_periodicity='+str(self.output_save_periodicity)+', trial_number='+str(self.trial_number)\
                +', logger='+str(self.logger)+')'
        
    def _get_agent_data(self, current_agent, train_data=None, env=None): 
        """
        Parameters
        ----------            
        current_agent: This is an object of a Class inheriting from the Class Block. It represents the current agent for which
                       we want to extract the train_data and env.
            
        train_data: This is the train_data that entered the tuner. It must be an object of a Class inheriting from the Class 
                    BaseDataSet.
                                        
                    The default is None.
            
        env: This is the env that entered the tuner. It must be an object of a Class inheriting from the Class BaseEnvironment.
                                
             The default is None.

        Returns
        -------
        tmp_agent_train_data: This is the selected train_data for the current_agent. It is an object of a Class inheriting from 
                              the Class BaseDataSet.
    
        tmp_agent_env: This is the selected env for the current_agent. It is an object of a Class inheriting from the Class 
                       BaseEnvironment.
        """
        
        if((train_data is None) and (env is None)):
            exc_msg = 'In \'_get_agent_data\' of \'TunerGenetic\' \'train_data\' and \'env\' are both \'None\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
                
        current_agent_train_data, current_agent_env = self.input_loader.get_input(blocks=[current_agent], n_inputs_to_load=1, 
                                                                                  train_data=train_data, env=env)
            
        #input loaders methods get_input return two lists: one for the train_data and one for the env.        
        tmp_agent_train_data = None
        if(current_agent_train_data is not None):
            tmp_agent_train_data = current_agent_train_data[0]
       
        tmp_agent_env = None
        if(current_agent_env is not None):
            tmp_agent_env = current_agent_env[0]
            
        return tmp_agent_train_data, tmp_agent_env
    
    def _mutate_gather_data_and_env(self, current_gen_n, current_gen_length, tmp_agent_to_tune, train_data=None, env=None, 
                                    first_mutation=False):
        """
        Parameters
        ----------
        current_gen_n: This is an integer representing the number of the current generation.
            
        current_gen_length: This is an integer representing the length of the current generation.
            
        tmp_agent_to_tune: This is an object of a Class inheriting from the Class Block. It represents the current agent.
        
        train_data: This is the train_data that entered the tuner. It must be an object of a Class inheriting from the Class 
                    BaseDataSet.
                         
                    The default is None.

        env: This is the env that entered the tuner. It must be an object of a Class inheriting from the Class BaseEnvironment.
             
             The default is None.
        
        first_mutation: This is True if this is the first time generation else it is False.
                    
                        The default is False.

        Returns
        -------
        tmp_agent: This is the mutated agent. It is an object of a Class inheriting from the Class Block.
        
        tmp_agent_train_data: This is the selected train_data for the mutated agent. It is an object of a Class inheriting from 
                              the Class BaseDataSet.
        
        tmp_agent_env: This is the selected env for the mutated agent. It is an object of a Class inheriting from the Class 
                       BaseEnvironment.
        """
        
        #Note that the mutation is used also as initialisation of the different agents. Moreover we call deepcopy because we need 
        #separate agents
        agent_to_tune = copy.deepcopy(tmp_agent_to_tune)
        
        #silence the agent: I use output_save_periodicity to print some informations every now and then. This is done unless the
        #verbosity is greater than 4 (in which case it is debug and we want to print everything)
        if(self.verbosity < 4):
            agent_to_tune.update_verbosity(new_verbosity=0)
            self.eval_metric.update_verbosity(new_verbosity=0)

        #i need to change the seed of the agents:
        agent_to_tune.set_local_prng(new_seeder=agent_to_tune.seeder+current_gen_length)
        
        tmp_agent = self._mutate(agent=agent_to_tune, first_mutation=first_mutation)
                       
        tmp_agent.obj_name = self.block_to_opt.obj_name+'_Gen_'+str(current_gen_n)+'_Agent_'+str(current_gen_length)
        tmp_agent.logger.name_obj_logging = self.block_to_opt.logger.name_obj_logging+'_Gen_'+str(current_gen_n)\
                                            +'_Agent_'+str(current_gen_length)
        
        #check that the call to self.mutate was successful
        if(tmp_agent is None):
            self.is_tune_successful = False
            self.logger.error(msg='There was an error mutating an agent!')
            return None, None, None
            
        tmp_agent_train_data, tmp_agent_env = self._get_agent_data(current_agent=tmp_agent,
                                                                   train_data=train_data, 
                                                                   env=env)
        
        return tmp_agent, tmp_agent_train_data, tmp_agent_env
    
    def _learn_and_evaluate(self, tmp_agent, tmp_agent_train_data, tmp_agent_env):
        """
        Parameters
        ----------
        tmp_agent: This is the mutated agent. It is an object of a Class inheriting from the Class Block.
        
        tmp_agent_train_data: This is the selected train_data for the mutated agent. It is an object of a Class inheriting from 
                              the Class BaseDataSet.
        
        tmp_agent_env: This is the selected env for the mutated agent. It is an object of a Class inheriting from the Class 
                       BaseEnvironment.

        Returns
        -------
        tmp_agent: This is the agent on which we called the method learn(). It has its evaluation inside the member block_eval.
        """
                
        #it may happen that the hyper-parameters selected are ill so that the block returns NaN. This may happen for RL model 
        #generation blocks: this breaks the interaction with the environment. This is not an issue of implementation, it is just
        #that by searching for hyper-parameters configurations it may happen to find a configuration that either through PyTorch
        #or through something else makes the policy return NaN actions.
        #cf. https://github.com/hill-a/stable-baselines/issues/340
        #cf. https://stable-baselines.readthedocs.io/en/master/guide/checking_nan.html
        try:
            tmp_res = tmp_agent.learn(train_data=tmp_agent_train_data, env=tmp_agent_env)        
        except Exception as exc:
            #i need to extract the params since the pre_learn_check will reset it otherwise.
            prev_params = tmp_agent.get_params()
            
            is_pre_learn_check_successful = tmp_agent.pre_learn_check(train_data=tmp_agent_train_data, env=tmp_agent_env)
            
            #revert effect of the call of pre_learn_check:
            self.fully_instantiated = True
            is_set_params_successful = tmp_agent.set_params(prev_params)
                
            if(self.trial_number == 0 or (not is_pre_learn_check_successful) or (not is_set_params_successful)):
                exc_msg = 'Exception Type: '+str(type(exc).__name__)+'. Exception Message: '+str(exc)+'. Information about'\
                          +' failed trial: trial_number='+str(self.trial_number)+', is_pre_learn_check_successful='\
                          +str(is_pre_learn_check_successful)+', is_set_params_successful='+str(is_set_params_successful)
                self.logger.exception(msg=exc_msg)
                tmp_agent = None
            else:
                #if the metric should be maximised i set this agent evaluation equal to -inf, else if the metric should be
                #minimised i set this agent evaluation equal to +inf.
                sign_of_eval = -1
                if(self.eval_metric.which_one_is_better(block_1_eval=0, block_2_eval=1) == 0):
                    sign_of_eval = 1
                    
                tmp_agent.block_eval = sign_of_eval*np.inf
                
                tmp_agent.is_learn_successful = True
                
                #mutate a lot the hyper-parameters to move away from bad configuration:
                tmp_agent = self._mutate(agent=tmp_agent, first_mutation=True)
                
                if(tmp_agent is None):
                    self.logger.exception(msg='The \'tmp_agent\' is \'None\'!')
            
            #return the agent now: all the other steps do not take place 
            return tmp_agent
            
        if(not tmp_agent.is_learn_successful):
            self.is_tune_successful = False
            self.logger.error(msg='There was an error in the \'learn\' method of an agent!')
            tmp_agent.block_eval = None
            return tmp_agent
        else:
            tmp_agent_eval = self._evaluate(agent_res=tmp_res, agent=tmp_agent, train_data=tmp_agent_train_data, 
                                            env=tmp_agent_env)    
            tmp_agent.block_eval = tmp_agent_eval
            
            if(((self.trial_number % self.output_save_periodicity) == 0) and (self.trial_number != 0)):
                self.logger.debug(msg='Agent: '+str(tmp_agent.obj_name)+' Evaluation: '+str(tmp_agent_eval))
                tmp_agent.save()
                tmp_res.save()
                
            #save all new best agents:
            if((self.trial_number == 0) or (self.eval_metric.which_one_is_better(block_1_eval=tmp_agent.block_eval, 
                                                                                 block_2_eval=self.best_agent.block_eval) == 0)):
                self.best_agent = tmp_agent
                
                self.logger.info(msg='New best agent: '+str(tmp_agent.obj_name)+' Evaluation: '+str(tmp_agent.block_eval))
                
                tmp_agent.obj_name += '_new_best'
                tmp_res.obj_name += '_new_best'
        
                tmp_agent.save()
                tmp_res.save()
            
            self.trial_number += 1
            
            return tmp_agent
        
    def _no_elitism_or_best_performant_elitism_common(self, agents_population, preserve_best_agent, train_data=None, env=None):   
        """
        Parameters
        ----------
        agents_population: This is a list containing the first generation of learnt agents.
            
        preserve_best_agent: This is True if tuning_mode is equal to 'best_performant_elitism', else it is False.
            
        train_data: This is the train_data that entered the tuner. It must be an object of a Class inheriting from the Class 
                    BaseDataSet.
                                        
                    The default is None.
            
        env: This is the env that entered the tuner. It must be an object of a Class inheriting from the Class BaseEnvironment.
                                
             The default is None.
            
        Returns
        -------
        agents_population: This is a list containing the last generation of learnt agents.
        """
                        
        for gen_index in range(self.n_generations-1):
            self.logger.info(msg='Generation: '+str(gen_index+1))
            
            new_agents_population = []
            
            self.input_loader.set_local_prng(new_seeder=self.seeder+gen_index)
                        
            tmp_new_agents_population = []
            
            while len(tmp_new_agents_population) < self.n_agents:
                #if it is the start of the generation i select the best agent of the previous generation to be passed on:
                if(len(tmp_new_agents_population) == 0):
                    selected_agent = self._evaluate_a_generation(gen=agents_population)[0]
                    log_msg = 'Previous generation best agent evaluation: '+str(selected_agent.block_eval)
                    self.logger.info(msg=log_msg)
                else:
                    #if the new generation is not emepty i already copied the best agent and so i pick a new one with 
                    #self.select(). I deepcopy the agents_population since I do not want to destroy it with self.select()
                    selected_agent = self._select(agents_pop=copy.deepcopy(agents_population))
                 
                tmp_new_agents_population.append(selected_agent)   
            
            agents=[]
            datas=[]
            envs=[]
            for n_agent_current_gen in range(self.n_agents): 
                dict_mutate_gather_data = dict(current_gen_n=gen_index+1, current_gen_length=n_agent_current_gen,
                                               tmp_agent_to_tune=tmp_new_agents_population[n_agent_current_gen],
                                               train_data=train_data, env=env, first_mutation=False)
                
                tmp_agent, tmp_agent_train_data, tmp_agent_env = self._mutate_gather_data_and_env(**dict_mutate_gather_data)
                agents.append(tmp_agent)
                datas.append(tmp_agent_train_data)
                envs.append(tmp_agent_env)
                 
            parallel_agents_res = Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)
            
            parallel_agents_res = parallel_agents_res(delayed(self._learn_and_evaluate)(agents[agent_index],
                                                                                        datas[agent_index],
                                                                                        envs[agent_index]) 
                                                      for agent_index in range(self.n_agents))          
            
            #If preserve_best_agent is True I need to pass on the best agent overall
            if(preserve_best_agent):
                new_agents_population.append(self.best_agent)
                
            for tmp_agent in parallel_agents_res:
                 if(tmp_agent is not None):
                     new_agents_population.append(tmp_agent)
                 else:
                     self.is_tune_successful = False
                     self.logger.error(msg='There was an error in the \'_common_tuning_step\' method for an agent!')
                     return None
             
            agents_population = new_agents_population
        
        return agents_population
        
    def _no_elitism(self, agents_population, train_data=None, env=None):   
        """
        Parameters
        ----------
        agents_population: This is a list containing the first generation of learnt agents.
                        
        train_data: This is the train_data that entered the tuner. It must be an object of a Class inheriting from the Class 
                    BaseDataSet.
                                        
                    The default is None.
            
        env: This is the env that entered the tuner. It must be an object of a Class inheriting from the Class BaseEnvironment.
                                
             The default is None.
            
        Returns
        -------
        agents_population: This is a list containing the last generation of learnt agents.
        """
        
        agents_population = self._no_elitism_or_best_performant_elitism_common(agents_population=agents_population, 
                                                                               preserve_best_agent=False, train_data=train_data, 
                                                                               env=env)
        
        return agents_population
            
    def _best_performant_elitism(self, agents_population, train_data=None, env=None):
        """
        Parameters
        ----------
        agents_population: This is a list containing the first generation of learnt agents.
                        
        train_data: This is the train_data that entered the tuner. It must be an object of a Class inheriting from the Class 
                    BaseDataSet.
                                        
                    The default is None.
            
        env: This is the env that entered the tuner. It must be an object of a Class inheriting from the Class BaseEnvironment.
                                
             The default is None.
            
        Returns
        -------
        agents_population: This is a list containing the last generation of learnt agents.
        """
        
        agents_population = self._no_elitism_or_best_performant_elitism_common(agents_population=agents_population,  
                                                                               preserve_best_agent=True, train_data=train_data, 
                                                                               env=env)
                                            
        return agents_population
        
    def _pool_elitism(self, agents_population, train_data=None, env=None):
        """
        Parameters
        ----------
        agents_population: This is a list containing the first generation of learnt agents.
                        
        train_data: This is the train_data that entered the tuner. It must be an object of a Class inheriting from the Class 
                    BaseDataSet.
                                        
                    The default is None.
            
        env: This is the env that entered the tuner. It must be an object of a Class inheriting from the Class BaseEnvironment.
                                
             The default is None.
            
        Returns
        -------
        population_pool_of_best: This is a list containing the last generation of learnt agents.
        """
        
        ag_rews  = []
        for tmp_agent in agents_population:
            ag_rews.append([tmp_agent.block_eval, len(ag_rews)])
        
        ag_rews = np.sort(ag_rews, axis=0)
        
        idxs = []
        
        if(self.eval_metric.which_one_is_better(0, 1) == 0):
            for i in range(self.pool_size):
                idxs.append(ag_rews[i][1])
        else:
            for i in range(len(ag_rews)-self.pool_size, len(ag_rews)):
                idxs.append(ag_rews[i][1])
        
        #numpy array indices cannot be floats:
        idxs = [int(elem_idxs) for elem_idxs in idxs]
        population_pool_of_best = list(np.array(agents_population)[idxs])
        
        for gen_index in range(self.n_generations-1):
            self.logger.info(msg='Generation: '+str(gen_index+1))
            
            new_agents_population = []
            
            self.input_loader.set_local_prng(new_seeder=self.seeder+gen_index)
                        
            tmp_new_agents_population = []
            
            for i in range(len(population_pool_of_best)):
                for j in range(int(self.n_agents/self.pool_size)):
                    tmp_new_agents_population.append(population_pool_of_best[i])
            
            agents=[]
            datas=[]
            envs=[]
            for n_agent_current_gen in range(self.n_agents): 
                dict_mutate_gather_data = dict(current_gen_n=gen_index+1, current_gen_length=n_agent_current_gen,
                                               tmp_agent_to_tune=tmp_new_agents_population[n_agent_current_gen],
                                               train_data=train_data, env=env, first_mutation=False)
                
                tmp_agent, tmp_agent_train_data, tmp_agent_env = self._mutate_gather_data_and_env(**dict_mutate_gather_data)
                agents.append(tmp_agent)
                datas.append(tmp_agent_train_data)
                envs.append(tmp_agent_env)
                 
                 
            parallel_agents_res = Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)
            
            parallel_agents_res = parallel_agents_res(delayed(self._learn_and_evaluate)(agents[agent_index],
                                                                                        datas[agent_index],
                                                                                        envs[agent_index]) 
                                                      for agent_index in range(self.n_agents))          
                 
            for tmp_agent in parallel_agents_res:
                 if(tmp_agent is not None):
                     new_agents_population.append(tmp_agent)
                 else:
                     self.is_tune_successful = False
                     self.logger.error(msg='There was an error in the \'_common_tuning_step\' method for an agent!')
                     return None
             
            agents_population = new_agents_population
            
            population_pool_of_best = []
            ag_rews = []
        
            for tmp_agent in agents_population:
                ag_rews.append([tmp_agent.block_eval, len(ag_rews)])
        
            ag_rews = np.sort(ag_rews, axis=0)
        
            idxs = []
            
            if(self.eval_metric.which_one_is_better(0, 1) == 0):
                for i in range(self.pool_size):
                    idxs.append(ag_rews[i][1])
            else:
                for i in range(len(ag_rews)-self.pool_size, len(ag_rews)):
                    idxs.append(ag_rews[i][1])
               
            #numpy array indices cannot be floats:
            idxs = [int(elem_idxs) for elem_idxs in idxs]

            population_pool_of_best = list(np.array(agents_population)[idxs])
        
        return population_pool_of_best
    
    def tune(self, train_data=None, env=None):
        """
        Parameters
        ----------        
        train_data: This is the dataset that can be used by the Tuner. 
                            
                    It must be an object of a Class inheriting from BaseDataSet.
        
        env: This is the environment that can be used by the Tuner.
        
             It must be an object of a Class inheriting from BaseEnvironment.

        Returns
        -------
        best_agent: This is the tuned agent: the original agent but with the tuned hyper-parameters.
        
        best_agent_eval: This is the evaluation of the best tuned agent, according to the eval_metric of the Tuner.
        """
 
        #checks if self.block_to_opt.is_parametrised == True and re-sets is_tune_successful to False. Also checks for the
        #consistency between the metric and the input_loader.
        tmp_out = super().tune(train_data=train_data, env=env)

        #super().tune() may return best_agent, best_agent_eval or None: If nothing went wrong and we can continue super().tune()
        #returns None, and hence in tmp_out we will have None:
        if(tmp_out is not None):
            return tmp_out[0], tmp_out[1]
        
        #if we are here it means tmp_out is None, and hence we need to proceed with the tuning procedure:
        
        #learn the starting agent to understand how good the tuning is:
        starting_res = self.block_to_opt.learn(train_data=train_data, env=env) 
       
        starting_eval = self._evaluate(agent_res=starting_res, agent=self.block_to_opt, train_data=train_data, env=env)
        
        self.logger.info(msg='The provided \'block_to_opt\' has a starting evaluation equal to: '+str(starting_eval))
        
        #i want to save the best agent that i have ever created across all generations:
        self.best_agent = None    
            
        self.logger.info(msg='Generation: '+str(0))
        
        #create and initialise base population of agnets: first generation   
        agents_population = [] 

        agents=[]
        datas=[]
        envs=[]
        
        for agent_index in range(self.n_agents):
            dict_mutate_gather_data = dict(current_gen_n=0, current_gen_length=agent_index, 
                                           tmp_agent_to_tune=self.block_to_opt, train_data=train_data, env=env,
                                           first_mutation=True)
            
            tmp_agent, tmp_agent_train_data, tmp_agent_env  = self._mutate_gather_data_and_env(**dict_mutate_gather_data)
            if(tmp_agent is None):
                self.is_tune_successful = False
                return None, None
            else:
                agents.append(tmp_agent)
                datas.append(tmp_agent_train_data)
                envs.append(tmp_agent_env)
                
        parallel_agents_res = Parallel(n_jobs=self.n_jobs, backend=self.backend, verbose=0, prefer=self.prefer)
        
        parallel_agents_res = parallel_agents_res(delayed(self._learn_and_evaluate)(agents[agent_index], datas[agent_index], 
                                                                                    envs[agent_index])
                                                  for agent_index in range(self.n_agents))
        
        for tmp_agent in parallel_agents_res:
            if(tmp_agent.block_eval is not None):
                agents_population.append(tmp_agent)
            else:
                self.is_tune_successful = False
                self.logger.error(msg='There was an error evaluating an agent!')
                return None, None
        
        if(self.tuning_mode == 'no_elitism'):
            tuner_final_pop = self._no_elitism(agents_population=agents_population, train_data=train_data, env=env)
        elif(self.tuning_mode == 'best_performant_elitism'):
            tuner_final_pop = self._best_performant_elitism(agents_population=agents_population, train_data=train_data, env=env)
        elif(self.tuning_mode == 'pool_elitism'):
            if(self.pool_size is not None):
                tuner_final_pop = self._pool_elitism(agents_population=agents_population, train_data=train_data, env=env)
            else:
                self.is_tune_successful = False
                err_msg = '\'pool_size\' cannot be \'None\' when \'tuning_mode\' is only be equal to \'pool_elitism\'!'
                self.logger.error(msg=err_msg)
                return None, None
        else:
            self.is_tune_successful = False
            err_msg = 'The parameter \'tuning_mode\' can only be equal to: \'no_elitism\', \'best_performant_elitism\' or'\
                      +' \'pool_elitism\'!'
            self.logger.error(msg=err_msg)
            return None, None
            
        last_gen_best_agent, last_gen_best_agent_eval = self._evaluate_a_generation(gen=tuner_final_pop)
        log_msg = 'Last generation best agent evaluation: '+str(last_gen_best_agent.block_eval)
        self.logger.info(msg=log_msg)
                     
        self.logger.info(msg='Best agent evaluation: '+str(self.best_agent.block_eval))
        
        self.best_agent.obj_name = 'best_agent_' + self.best_agent.obj_name
        self.best_agent.save()
        
        best_agent = self.best_agent
        best_agent_eval = self.best_agent.block_eval
        
        #create heatmap
        self.create_explanatory_heatmap_hyperparameters()   
        
        self.is_tune_successful = True
        
        return best_agent, best_agent_eval
                        
    def _mutate(self, agent, first_mutation=False):
        """
        Parameters
        ----------
        agent: This is the agent that we need to mutate. It is an object of a Class inheriting from the Class Block.

        first_mutation: This is True if this is the first generation, else it is False.
        
                        The default is False.

        Returns
        -------
        agent: This is the mutated agent. It is an object of a Class inheriting from the Class Block.
        """
        
        #the method get_params should return a flat dictionary (no dictionaries inside it).
        agent_params = agent.get_params()
        
        if(agent_params is None):   
            self.is_tune_successful = False
            self.logger.error(msg='The method \'get_params\' of an agent returned \'None\'!')
            return None
            
        rolling_seed = 0
        for key_hyper_params in list(agent_params.keys()):
            #update hyperparameter seeds (otherwise they are the same for all agents and we would get the same result):
            rolling_seed += 1
            agent_params[key_hyper_params].set_local_prng(new_seeder=agent.seeder+rolling_seed)
                
            if(agent_params[key_hyper_params].to_mutate):        
                #we perform a mutation only with probability: prob_point_mutation
                if(agent.local_prng.uniform() < self.prob_point_mutation):
                    agent_params[key_hyper_params].mutate(first_mutation=first_mutation)

        #now call agent.set_params(). It sets new params and returns True if everything was alright, False otherwise:
        is_set_param_successful = agent.set_params(agent_params) 
                
        if(not is_set_param_successful):
            self.is_tune_successful = False
            self.logger.error(msg='There was an error setting the parameters of an agent!')
            return None
        else:
            return agent
    
    def _evaluate(self, agent_res, agent=None, train_data=None, env=None):
        """
        Parameters
        ----------
        agent_res: This is the output of the method learn() of the agent that we need to evaluate. It is an object of Class 
                   BlockOutput.
            
        agent: This is the agent that we need to evaluate. It is an object of a Class inheriting from the Class Block.
                
               The default is None.
        
        train_data: This is the selected train_data for the agent_res. It is an object of a Class inheriting from the Class
                    BaseDataSet.
                    
                    The default is None.
                    
        env: This is the selected env for the agent_res. It is an object of a Class inheriting from the Class BaseEnvironment.
        
             The default is None.
            
        Returns
        -------
        tmp_single_agent_eval: This is a float and it represents the evaluation of the agent according to the eval_metric.
          
        This method evaluates a single agent by calling the evaluate method of the evaluation metric of the Tuner onto the agent.
        """

        if((train_data is None) and (env is None)):
            exc_msg = 'In \'self.evaluate\' of \'TunerGenetic\' \'train_data\' and \'env\' are both  \'None\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
            
        #if we need to evaluate a RLPipeline block we need to use as input to the evaluation the train_data and the env created
        #by the various blocks making up the pipeline. For example if we have a FeaturEngineering block and then a 
        #ModelGeneration block we need to use the environment modified by the FeatureEngineering block in evaluating the 
        #ModelGeneration block.
        #This only needs to be done if the last block in the pipeline is a ModelGeneration block.
        #Evaluating a RLPipeline means evaluating its last block.
        if(isinstance(agent, RLPipeline)):
            if(isinstance(agent.list_of_block_objects[-1], ModelGeneration)):
                train_data = agent_res.train_data
                env = agent_res.env
        
        tmp_single_agent_eval = self.eval_metric.evaluate(block_res=agent_res, block=agent, train_data=train_data, env=env)
                
        return tmp_single_agent_eval
                
    def _select(self, agents_pop):
        """
        Parameters
        ----------
        agents_pop: This is a list of agents, namely of objects inheriting from the Class Block.
            
        Returns
        -------
        selected_ag: This is the selected agent. It is an object of a Class inheriting from the Class Block.
            
        This method selects an agent from the current population: this agent will be passed onto the next generation.
        """
        
        #we perform tournament selection: select randomly 3 agents and pick the most performing among them:
        size_agents = 3
        
        tmp_gen = list(self.local_prng.choice(agents_pop, size=size_agents))
        
        #pick the best among the selected one:
        agent_to_pass_on = self._evaluate_a_generation(gen=tmp_gen)
        
        #returns agent_to_pass_on[0] since _evaluate_a_generation returns best_agent, best_agent_eval!
        selected_ag = agent_to_pass_on[0]
        
        return selected_ag

    def _evaluate_a_generation(self, gen):
        """
        Parameters
        ----------
        gen: This is a list of agents, namely of objects inheriting from the Class Block.

        Returns
        -------
        best_agent: This is the best agent in the gen. It is an object of a Class inheriting from the Class Block.
            
        best_agent_eval: This is the evaluation of the best agent in the gen. It is a float.
        """

        if(gen is None):
            self.is_tune_successful = False
            exc_msg = '\'gen\' cannot be \'None\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
        
        best_agent = None
        best_agent_eval = None
        
        agents_list_of_evaluations = []
        for tmp_agent in gen:
            agents_list_of_evaluations.append(tmp_agent.block_eval)
            
        sign_for_sorting = -1
        #if the metric needs to be minimised then i want the agent with the smallest evaluation:
        if(self.eval_metric.which_one_is_better(0, 1) == 0):
            sign_for_sorting = 1
                    
        best_agent_idx = np.argsort(sign_for_sorting*np.array(agents_list_of_evaluations))[0]
        
        best_agent = gen[best_agent_idx]
        best_agent_eval = best_agent.block_eval
                        
        return best_agent, best_agent_eval