"""
This module contains the implementation of the Class: TunerOptuna.

The Class TunerOptuna inherits from the Class Tuner.
"""

import copy
import numpy as np

import optuna

from ARLO.tuner.tuner import Tuner
from ARLO.hyperparameter.hyperparameter import Real, Integer, Categorical
from ARLO.block.model_generation import ModelGeneration
from ARLO.rl_pipeline.rl_pipeline import RLPipeline


class TunerOptuna(Tuner):
    """
    This Class implements an hyperparameter optimisation algorithm, namely it provides access to the Optuna library.
    cf. https://optuna.readthedocs.io/en/stable/index.html
    cf. https://arxiv.org/abs/1907.10902

    This Class inherits from the Class Tuner.    
    """
    
    def __init__(self, block_to_opt, eval_metric, input_loader, obj_name, create_explanatory_heatmap=False, sampler='TPE',
                 n_trials=100, max_time_seconds=3600, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, 
                 n_jobs=1, job_type='process', output_save_periodicity=25):
        """
        Parameters
        ----------
        sampler: This is a string representing the sampler to use out of the ones provided by Optuna. It can be: 'CMA-ES', 
                 'TPE', 'RANDOM', 'GRID'.
                 
                 Note that some samplers have some requirements: for example the 'CMA-ES' sampler does not work on categorical
                 parameters.
                 
                 cf. https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
                     #sampling-algorithms
                 
                 The default is 'TPE'.
        
        n_trials: This is a positive integer representing the number of trials to do.
                  
                  The default is 100.
        
        max_time_seconds: This is a positive integer representing the maximum allowed time in seconds.
                
                          The default is 3600.
        
        Non-Parameters Members
        ----------------------    
        optuna_object_sampler: This is a sampler object from the Optuna library. It is obtained from the sampler parameter.
            
        opt_direction: This is a string, either 'maximize' or 'minimize'. It is 'maximize' if the metric of the block needs to be
                       maximised, else it is 'minimize'.
        
        The other parameters and non-parameters members are described in the Class Tuner.
        """
        
        super().__init__(block_to_opt=block_to_opt, eval_metric=eval_metric, input_loader=input_loader, obj_name=obj_name, 
                         create_explanatory_heatmap=create_explanatory_heatmap, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type,
                         output_save_periodicity=output_save_periodicity)
        
        self.sampler = sampler
        if(self.sampler not in ['CMA-ES', 'TPE', 'GRID', 'RANDOM']):
            exc_msg = '\'sampler\' can only be: \'CMA-ES\', \'TPE\', \'GRID\' or \'RANDOM\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
            
        self.search_space = {}
      
        dict_of_samplers = {'CMA-ES': optuna.samplers.CmaEsSampler, 
                            'TPE': optuna.samplers.TPESampler,  
                            'GRID': optuna.samplers.GridSampler, 
                            'RANDOM': optuna.samplers.RandomSampler}    
        
        self.optuna_object_sampler = dict_of_samplers[self.sampler]
        
        self.opt_direction = 'maximize'
        #test the metric to see if it should be maximised or minimised: i assume a block eval was 1 and another block eval was 2:
        #if the better one is the first one, then it means 1 is better than 2, hence the metric should be minimised
        if(self.eval_metric.which_one_is_better(block_1_eval=1, block_2_eval=2) == 0):
            self.opt_direction = 'minimize'

        self.n_trials = n_trials
        self.max_time_seconds = max_time_seconds

    def __repr__(self):
         return 'TunerOptuna('+'block_to_opt='+str(self.block_to_opt)+', eval_metric='+str(self.eval_metric)\
                +', input_loader='+str(self.input_loader)+', obj_name='+str(self.obj_name)\
                +', create_explanatory_heatmap='+str(self.create_explanatory_heatmap)+', sampler='+str(self.sampler)\
                +', n_trials='+str(self.n_trials)+', max_time_seconds='+str(self.max_time_seconds)\
                +', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
                +', output_save_periodicity='+str(self.output_save_periodicity)+', logger='+str(self.logger)+')'

    def _objective(self, trial):
        """
        Parameters
        ----------
        trial: This is an object of Class optuna.trial.Trial. This object provides interfaces to get suggestions for the new
               values of the hyper-parameters.
               
               cf. https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
            
        Returns
        -------
        tmp_agent_eval: This is the evaluation, according to eval_metric, of the new agent. This is a float.
        """
        
        agent_params = self.block_to_opt.get_params()
        if(agent_params is None):
            self.is_tune_successful = False
            self.logger.error(msg='There was an error getting the parameters of an agent!')
            return None
        
        my_agent = copy.deepcopy(self.block_to_opt)
        
        my_agent.obj_name = str(self.block_to_opt.obj_name)+'_trial_number_'+str(trial.number)
        my_agent.logger.name_obj_logging = str(self.block_to_opt.logger.name_obj_logging)+'_trial_number_'+str(trial.number)
        my_agent.checkpoint_log_path = self.checkpoint_log_path
        
        #silence the agent: I use output_save_periodicity to print some informations every now and then. This is done unless the
        #verbosity is greater than 4 (in which case it is debug and we want to print everything)
        if(self.verbosity < 4):
            my_agent.update_verbosity(new_verbosity=0)
            self.eval_metric.update_verbosity(new_verbosity=0)

        params_optuna = copy.deepcopy(agent_params)
        for key_hyper_params in list(agent_params.keys()):
            if(agent_params[key_hyper_params].to_mutate):                
                tmp = agent_params[key_hyper_params]
                
                if(isinstance(tmp, Real)):
                    params_optuna[key_hyper_params].current_actual_value = trial.suggest_float(tmp.hp_name, 
                                                                                               tmp.range_of_values[0],
                                                                                               tmp.range_of_values[1])
                elif(isinstance(tmp, Integer)):                    
                    params_optuna[key_hyper_params].current_actual_value = trial.suggest_int(tmp.hp_name, 
                                                                                             tmp.range_of_values[0],
                                                                                             tmp.range_of_values[1])
                else:
                    params_optuna[key_hyper_params].current_actual_value = trial.suggest_categorical(tmp.hp_name, 
                                                                                                     tmp.possible_values)
        set_params_res = my_agent.set_params(params_optuna)
        
        if(not set_params_res):
            self.is_tune_successful = False
            self.logger.error(msg='There was an error setting the parameters of an agent!')
            return None
        
        tmp_res = my_agent.learn(train_data=self.data, env=self.env)
        
        if(not my_agent.is_learn_successful):
            self.logger.info(msg='There was an error in the \'learn\' method of an agent!')
            return None
        
        #if we need to evaluate a RLPipeline block we need to use as input to the evaluation the train_data and the env created
        #by the various blocks making up the pipeline. For example if we have a FeaturEngineering block and then a 
        #ModelGeneration block we need to use the environment modified by the FeatureEngineering block in evaluating the 
        #ModelGeneration block.
        #This only needs to be done if the last block in the pipeline is a ModelGeneration block.
        #Evaluating a RLPipeline means evaluating its last block.
        if(isinstance(my_agent, RLPipeline)):
            if(isinstance(my_agent.list_of_block_objects[-1], ModelGeneration)):
                self.data = tmp_res.train_data
                self.env = tmp_res.env
        
        tmp_agent_eval = self.eval_metric.evaluate(block_res=tmp_res, block=my_agent, train_data=self.data, env=self.env)
        
        if((trial.number % self.output_save_periodicity) == 0):
            self.logger.debug(msg='Agent: '+str(my_agent.obj_name)+' Evaluation: '+str(tmp_agent_eval))
            my_agent.block_eval = tmp_agent_eval
            my_agent.save()
            tmp_res.save()
            
        #save each best agent:
        if((trial.number == 0) or (self.eval_metric.which_one_is_better(block_1_eval=tmp_agent_eval, 
                                                                        block_2_eval=self.best_agent_eval) == 0)):
            self.best_agent_eval = tmp_agent_eval
            self.logger.info(msg='New best agent: '+str(my_agent.obj_name)+' Evaluation: '+str(tmp_agent_eval))
            
            my_agent.obj_name += '_new_best'
            tmp_res.obj_name += '_new_best'
            
            my_agent.save()
            tmp_res.save()
            
        #i set these since i cannot pass them as parameters to the method '_objective'
        new_data, new_env = self.input_loader.get_input(blocks=[my_agent], n_inputs_to_load=1, train_data=self.original_data, 
                                                        env=self.original_env)
        
        #the input loader returns a list, and since i specified n_inputs_to_load=1, then it returns a list with only one element,
        #here with [0] i am selecting that element from the list:
        if(new_data is not None):
            self.data = new_data[0]
        if(new_env is not None):
            self.env = new_env[0]
          
        return tmp_agent_eval
                
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
        best_agent = copy.deepcopy(self.block_to_opt)
        best_agent_params = copy.deepcopy(best_agent.get_params())
        
        if(best_agent_params is None):
            self.is_tune_successful = False
            self.logger.error(msg='There was an error getting the parameters of an agent!')
            return None, None
        
        #copy these: the input loader will do some operations on these, but i need to keep track of the original data so that i 
        #can repeat the call to the input loader for each agent:
        self.original_data = train_data
        self.original_env = env
        
        #i set these since i cannot pass them as parameters to the method '_objective'
        new_data, new_env = self.input_loader.get_input(blocks=[best_agent], n_inputs_to_load=1, train_data=self.original_data, 
                                                        env=self.original_env)
        
        #the input loader returns a list, and since i specified n_inputs_to_load=1, then it returns a list with only one element,
        #here with [0] i am selecting that element from the list:
        self.data = None
        self.env = None
        if(new_data is not None):
            self.data = new_data[0]
        if(new_env is not None):
            self.env = new_env[0]
        
        #i need this in order to detect and save every new best agent:
        self.best_agent_eval = None
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        if(self.sampler == 'GRID'):
            algo_params = self.block_to_opt.get_params()
          
            for tmp_key in list(algo_params.keys()):
                if(algo_params[tmp_key].to_mutate):
                    if(isinstance(algo_params[tmp_key], Real)):
                        l_low = algo_params[tmp_key].range_of_values[0]
                        l_high = algo_params[tmp_key].range_of_values[1]
                        self.search_space.update({tmp_key: list(np.linspace(l_low, l_high, 10))})
                    if(isinstance(algo_params[tmp_key], Integer)):
                        l_low = algo_params[tmp_key].range_of_values[0]
                        l_high = algo_params[tmp_key].range_of_values[1]+1
                        self.search_space.update({tmp_key: list(range(l_low, l_high))})
                    if(isinstance(algo_params[tmp_key], Categorical)):
                        self.search_space.update({tmp_key: algo_params[tmp_key].possible_values})

            self.optuna_object_sampler = self.optuna_object_sampler(search_space=self.search_space)
        elif(self.sampler == 'TPE'):
            algo_params = self.block_to_opt.get_params()
            n_hp_to_opt = 0
            
            for tmp_key in list(algo_params.keys()):
                if(algo_params[tmp_key].to_mutate):         
                    n_hp_to_opt += 1
            
            is_multivariate = False
            if(n_hp_to_opt > 1):
                is_multivariate = True

            self.optuna_object_sampler = self.optuna_object_sampler(seed=self.seeder, multivariate=is_multivariate)
        elif(self.sampler == 'CMA-ES'):
            self.optuna_object_sampler = self.optuna_object_sampler(seed=self.seeder, restart_strategy='ipop')
        else:
            self.optuna_object_sampler = self.optuna_object_sampler(seed=self.seeder)
            
        study = optuna.create_study(sampler=self.optuna_object_sampler, #pruner=optuna.pruners.HyperbandPruner(), 
                                    study_name=self.obj_name+'_optuna_study', direction=self.opt_direction)  
  
        study.optimize(self._objective, n_trials=self.n_trials, timeout=self.max_time_seconds, n_jobs=self.n_jobs, 
                       show_progress_bar=False)
                     
        trial = study.best_trial
        best_agent_eval = trial.value
        dict_of_params = {}
        
        for key, value in trial.params.items():
            if(isinstance(best_agent_params[key], Integer)):
                new_param = Integer(hp_name=key, current_actual_value=value, 
                                    range_of_values=best_agent_params[key].range_of_values, 
                                    seeder=best_agent_params[key].seeder, obj_name=key)
            elif(isinstance(best_agent_params[key], Real)):
                new_param = Real(hp_name=key, current_actual_value=value, 
                                 range_of_values=best_agent_params[key].range_of_values, 
                                 seeder=best_agent_params[key].seeder, obj_name=key)
            else:
                new_param = Categorical(hp_name=key, current_actual_value=value, 
                                        possible_values=best_agent_params[key].possible_values, 
                                        seeder=best_agent_params[key].seeder, obj_name=key)
                
            dict_of_params.update({key: new_param})
            
        for tmp_key in list(best_agent_params.keys()):
            if(not best_agent_params[tmp_key].to_mutate):
                dict_of_params.update({tmp_key: best_agent_params[tmp_key]})
            
        set_params_res = best_agent.set_params(dict_of_params)
        
        if(not set_params_res):
            self.is_tune_successful = False
            self.logger.error(msg='There was an error setting the parameters of the best agent!')
            return None, None
        
        best_agent.obj_name = 'best_agent_' + best_agent.obj_name
        best_agent.block_eval = best_agent_eval
        best_agent.save()
        
        if(self.create_explanatory_heatmap):
            #create heatmap
            self.create_explanatory_heatmap_hyperparameters()       
        
        self.is_tune_successful = True
        
        return best_agent, best_agent_eval        
     
    def _evaluate_a_generation(self, gen): 
        exc_msg = 'The method \'_evaluate_a_generation\' is not implemented, because it is not needed!'
        self.logger.exception(msg=exc_msg)
        raise AttributeError(exc_msg)