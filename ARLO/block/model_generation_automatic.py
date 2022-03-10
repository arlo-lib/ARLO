"""
This module contains the implementation of the Class AutoModelGeneration.

The Class AutoModelGeneration inherits from the Class ModelGeneration.

An AutoModelGeneration Block optimises over the proposed algorithms: it calls a Tuner on each proposed algorithm and picks the
most performing one according to some metric.
"""

import copy

from ARLO.block.block_output import BlockOutput
from ARLO.block.model_generation import ModelGeneration
from ARLO.block.model_generation_default import automatic_model_generation_default

                
class AutoModelGeneration(ModelGeneration):
    """
    This Class implements automatic model generation. Given the metric and the tuner_blocks_dict this block picks the best 
    model generation algorithm among the possible ones.
    """

    def __init__(self, eval_metric, obj_name, seeder=2, tuner_blocks_dict=None, log_mode='console', 
                 checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):
        """
        Parameters
        ----------
        tuner_blocks_dict: This is a dictionary where the key is a string name while the value is an object of a Class inheriting 
                           from the Class Tuner.
        
        Non-Parameters Members
        ----------------------
        tuner_blocks_dict_upon_instantiation: This a copy of the original value of tuner_blocks_dict, namely the value of 
                                              tuner_blocks_dict that the object got upon creation. This is needed for re-loading 
                                              objects.
                                              
        fully_instantiated: This is True if the block is fully instantiated, False otherwise. It is mainly used to make sure that 
                            when we call the learn method the model generation blocks have been fully instantiated as they 
                            undergo two stage initialisation being info_MDP unknown at the beginning of the pipeline.
                            
        info_MDP: This is a dictionary compliant with the parameters needed in input to all mushroom_rl model generation 
                  algorithms. It containts the observation space, the action space, the MDP horizon and the MDP gamma.
        
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
          
        self.works_on_online_rl = True
        self.works_on_offline_rl = True
        self.works_on_box_action_space = True
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = True
        
        #For the moment i do not want to tune automatic blocks:
        self.is_parametrised = False
        
        if(tuner_blocks_dict is None): 
            #the default is generic. Each block in the default is then learnt only if of the appropriate type. for example if
            #QLearning is in the default it will not be learnt in an Offline RL Pipeline!
            self.tuner_blocks_dict = automatic_model_generation_default
        else:
            self.tuner_blocks_dict = tuner_blocks_dict
            
        if(len(self.tuner_blocks_dict) == 0):
            exc_msg = 'The \'tuner_blocks_dict\' is empty!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
        
        self.fully_instantiated = False
        self.info_MDP = None 
        self.tuner_blocks_dict_upon_instantiation = copy.deepcopy(self.tuner_blocks_dict)
        
    def __repr__(self):
        return 'AutoModelGeneration('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
               +', seeder='+ str(self.seeder)+', local_prng='+ str(self.local_prng)\
               +', tuner_blocks_dict='+str(self.tuner_blocks_dict)+', log_mode='+str(self.log_mode)\
               +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
               +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
               +', works_on_online_rl='+str(self.works_on_online_rl)+', works_on_offline_rl='+str(self.works_on_offline_rl)\
               +', works_on_box_action_space='+str(self.works_on_box_action_space)\
               +', works_on_discrete_action_space='+str(self.works_on_discrete_action_space)\
               +', works_on_box_observation_space='+str(self.works_on_box_observation_space)\
               +', works_on_discrete_observation_space='+str(self.works_on_discrete_observation_space)\
               +', pipeline_type='+str(self.pipeline_type)+', is_learn_successful='+str(self.is_learn_successful)\
               +', is_parametrised='+str(self.is_parametrised)+', block_eval='+str(self.block_eval)\
               +', tuner_blocks_dict_upon_instantiation='+str(self.tuner_blocks_dict_upon_instantiation)\
               +', logger='+str(self.logger)+', fully_instantiated='+str(self.fully_instantiated)\
               +', info_MDP='+str(self.info_MDP)+')'  
             
    def full_block_instantiation(self, info_MDP):     
        """
        Parameters
        ----------
        info_MDP: This is an object of Class mushroom_rl.environment.MDPInfo. It contains the action and observation spaces, 
                  gamma and the horizon of the MDP.
        
        Returns
        -------        
        Since there is only one default dictionary for tuner_blocks_dict, it may contain many blocks that do not work 
        on all environment spaces and (or) pipelines and thus for such blocks the full_block_instantiation would fail.
        
        Thus here I return True.
        """        
        
        self.fully_instantiated = True
        self.info_MDP = info_MDP

        return True

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
        This method overrides the one of the base Class Block and always returns True. Why is this needed? 
        Because there is only one default dictionary for tuner_blocks_dict therefore it may contain many blocks that do not work 
        on all environment spaces and (or) pipelines. 
        
        We do not want to stop learning an automatic block only because it contains a block that is not consistent: we just 
        want to skip over it. 
        
        In the method learn of this Class we make sure that at least one block was tuned (and hence that at least one block was 
        compatible).
        """
        
        #if this is a model generation block i need to reset self.fully_instantiated to False:
        self.fully_instantiated = False
        
        #i need to reload the params to the value it was originally
        self.tuner_blocks_dict = self.tuner_blocks_dict_upon_instantiation
        
        return True
    
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
        best_agent: Each agent is tuned and then we return the fitted Agent corresponding to the most performant agent. Since 
                    this is the output of a model generation block then it means that it is an object of Class BlockOutput.
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_train_data_and_env = super().learn(train_data=train_data, env=env)
        starting_train_data = None
        starting_env = None
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and 
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_train_data_and_env, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
                
        #count how many blocks i actually tuned:
        count_tuned_blocks = 0
        
        best_agent = None
        best_agent_eval = None
        for tmp_key in list(self.tuner_blocks_dict.keys()):
            #pick a Tuner
            tmp_tuner = self.tuner_blocks_dict[tmp_key]
            
            self.logger.info(msg='Now tuning: '+str(tmp_tuner.obj_name)+'_'+str(tmp_tuner.block_to_opt.obj_name))

            #initialise self.pipeline_type of each model generation block contained in each Tuner: pipeline_type can be both
            #online and offline:
            if(self.pipeline_type == 'offline'):
                tmp_tuner.block_to_opt.pipeline_type = 'offline'
                #starting_train_data_and_env has length 2 and it is made of train_data, env. Here since it is offline RL env may
                #be None. 
                starting_train_data = starting_train_data_and_env[0]
                #i may want to use the environment for evaluatiing the agents:
                if(starting_train_data_and_env[1] is not None):
                    starting_env = starting_train_data_and_env[1]
            else:
                tmp_tuner.block_to_opt.pipeline_type = 'online'
                #starting_train_data_and_env has length 2 and it is made of train_data, env. Here since it is offline RL env may
                #be None. 
                starting_env = starting_train_data_and_env[1]
                
            #if the block can work on the provided environment spaces and pipeline I fully instantiate it, else i skip over it:
            is_consistent = tmp_tuner.block_to_opt.pre_learn_check(train_data=starting_train_data, env=starting_env)
            if(is_consistent):
                out_full_block_instantiation = tmp_tuner.block_to_opt.full_block_instantiation(info_MDP=self.info_MDP)
            
                #if the full instantiation of the specific model generation block we skip over it:
                if(not out_full_block_instantiation):
                    tmp_tuner.block_to_opt.fully_instantiated = False
                    err_msg = 'There was an error fully instantiating a specific \'ModelGeneration\' block'\
                              +' even though it was compatible with the pipeline and with the environment spaces!'
                    self.logger.error(msg=err_msg)
                    #we need to skip an iteration of the loop
                    continue
            else:
                log_msg = 'A block of \'AutoModelGeneration\' will not be considered in the tuning procedure since it is not'\
                          +' consistent with the observation and (or) action space and (or) the pipeline type!'
                self.logger.info(msg=log_msg)
                #we need to skip an iteration of the loop
                continue
            
            #call tune method on Tuner. Returns the tuned agent and its evaluation:
            #its evaluation, according to the provided evaluation metric
            tmp_tuned_agent, tmp_tuned_agent_eval = tmp_tuner.tune(train_data=starting_train_data, env=starting_env)
            
            #skip over the tuners that have input loader and metric not consistent:
            if(not tmp_tuner.is_metric_consistent_with_input_loader()):
                log_msg = 'A block of \'AutoModelGeneration\' will not be considered in the tuning procedure since its'\
                          +' input loader and metric are not consistent with each other!'
                self.logger.info(msg=log_msg)
                #we need to skip an iteration of the loop
                continue 

            if(not tmp_tuner.is_tune_successful):
                self.is_learn_successful = False
                err_msg = 'There was an error in one of the tuning procedures inside the \'AutoModelGeneration\' block!'
                self.logger.error(msg=err_msg)
                return BlockOutput(obj_name=self.obj_name)
                     
            if((tmp_tuned_agent is not None) and (tmp_tuner.is_tune_successful)):
                #if it is the first Tuner I need to assign the value to best_agent and best_agent_eval. This is also what i do in 
                #case it is not the first Tuner and the new evaluation is  'better' than the current best evaluation.
                #For more on what 'better' means look at the method which_one_is_better of the eval_metric.
                if((count_tuned_blocks == 0) or (self.eval_metric.which_one_is_better(block_1_eval=tmp_tuned_agent_eval, 
                                                                                      block_2_eval=best_agent_eval) == 0)):
                    best_agent = tmp_tuned_agent
                    best_agent_eval = tmp_tuned_agent_eval
                    
                #count how many blocks i actually tuned:
                count_tuned_blocks += 1
            else:
                self.is_learn_successful = False
                self.logger.error(msg='Something went wrong tuning the current block inside the \'AutoModelGeneration\' block!')
                return BlockOutput(obj_name=self.obj_name)
                
        #if i tuned at least a block:
        if(count_tuned_blocks > 0):
            self.logger.info(msg='Now learning the best tuned agent on the entire starting input...')
            #call the best tuned agent on the entire starting input to the automatic block:
            best_agent_generated = best_agent.learn(train_data=starting_train_data, env=starting_env)
            
            if(not best_agent.is_learn_successful):
                self.is_learn_successful = False
                err_msg = 'There was an error in the \'learn\' method of the best agent in the \'AutoModelGeneration\' block!'
                self.logger.error(msg=err_msg)
                return BlockOutput(obj_name=self.obj_name)
            else:
                self.is_learn_successful = True 
                self.logger.info(msg='The \'AutoModelGeneration\' block was successful!')
                return best_agent_generated
        else:
            err_msg = 'No blocks were tuned since they were not compatible with the observation and (or) action space'\
                      +' and (or) the pipeline type, or no tuner had consistent input loader and metric! The'\
                      +' \'AutoModelGeneration\' block was not successful!'
            self.is_learn_successful = False
            self.logger.error(msg=err_msg)
            return BlockOutput(obj_name=self.obj_name)
    
    def get_params(self):
        """
        Returns
        -------
        copy.deepcopy(self.tuner_blocks_dict)
        """
        
        return copy.deepcopy(self.tuner_blocks_dict)
    
    def set_params(self, new_params_dict):
        """
        Parameters
        ----------
        new_params_dict: This is the dictionary for the parameters of all model generation algorithms present in this automatic
                         block.
                        
        Returns
        -------
        bool: This method returns True if new_params_dict is set correctly, and False otherwise.
        """
        
        if(new_params_dict is not None):
            self.tuner_blocks_dict = new_params_dict
            
            self.tuner_blocks_dict_upon_instantiation = copy.deepcopy(self.tuner_blocks_dict)

            return True
        else:
            self.logger.error(msg='Cannot set parameters: \'new_params_dict\' is \'None\'!')
            return False  
            
    def analyse(self):
        """
        This method is not yet implemented.
        """
        
        raise NotImplementedError
        
    def update_verbosity(self, new_verbosity):
        """
        Parameters
        ----------
        new_verbosity: This is an integer and it represents the new verbosity level.
        
        This method calls the method update_verbosity implemented in the Class Block and then it calls such method of every 
        ModelGeneration block present in this automatic block.
        """
        
        super().update_verbosity(new_verbosity=new_verbosity)
        
        for tmp_key in list(self.tuner_blocks_dict.keys()):
            self.tuner_blocks_dict[tmp_key].update_verbosity(new_verbosity=new_verbosity)