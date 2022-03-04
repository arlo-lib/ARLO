"""
This module contains the implementation of the Class AutoRLPipeline. 

The Class AutoRLPipeline inherits from the Class Block. Why does it not inherit from the Class RLPipeline? Because we need the 
member tuner_blocks_dict which is made up of objects of Class inheriting from Class RLPipeline.

An AutoRLPipeline is used to solve a reinforcement learning problem by jointly optimising the pipeline. 
"""

import copy

from ARLO.block.block_output import BlockOutput
from ARLO.block.block import Block
from ARLO.block.data_generation import DataGeneration
from ARLO.block.data_preparation import DataPreparation
from ARLO.block.feature_engineering import FeatureEngineering
from ARLO.block.model_generation import ModelGeneration

                
class AutoRLPipeline(Block):
    """
    This Class implements automatic reinforcement learning: Given the metric and the tuner_blocks_dict this block picks the best 
    reinforcement learning pipeline among the possible ones.
    """
    
    def __init__(self, eval_metric, obj_name, tuner_blocks_dict, online_task, seeder=2, log_mode='console',
                 checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):
        """
        Parameters
        ----------
        tuner_blocks_dict: This is a dictionary where the key is a string name while the value is an object inheriting from the
                           Class Tuner.
                           
        online_task: This is True if the task is an online RL problem, and it is False otherwise.
                           
        Non-Parameters Members
        ----------------------
        tuner_blocks_dict_upon_instantiation: This a copy of the original value of tuner_blocks_dict, namely the value of 
                                              tuner_blocks_dict that the object got upon creation. This is needed for re-loading 
                                              objects.
        
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.online_task = online_task
        
        self.works_on_online_rl = self.online_task
        self.works_on_offline_rl = not self.online_task
        self.works_on_box_action_space = True
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = True
          
        if(self.works_on_online_rl):
            self.pipeline_type = 'online'
        else:
            self.pipeline_type = 'offline'  
                                          
        #There is no need to have an AutoRLPipeline block inside a Tuner!
        self.is_parametrised = False      

        self.tuner_blocks_dict = tuner_blocks_dict
            
        self.tuner_blocks_dict_upon_instantiation = copy.deepcopy(self.tuner_blocks_dict)
            
    def __repr__(self):
        return 'AutoRLPipeline('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
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
               +', logger='+str(self.logger)+')'    
    
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
        If the selected pipeline block can work with environment spaces it returns True, else it returns False.

        This method overrides the one of the base Class Block. Why is this needed? 
        Because there is only one default dictionary for tuner_blocks_dict therefore it may contain both online and offline
        pipelines.
        
        We do not want to stop learning an automatic block only because it contains an online pipeline while the problem is 
        an offline problem: we just want to skip over it. 
        
        In the method learn of this Class we make sure that at least one pipeline was tuned (and hence that at least one 
        pipeline was compatible).
        """
        
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
             
        Non-Parameters Members
        ----------------------
        self.is_learn_successful: This is a bool and it is True if the learning procedure was successful, False otherwise.
                                  The learning procedure is successful when all the blocks were learnt successfully without 
                                  any errors.
        
        Returns
        -------
        This is an object of Class BlockOutput and it corresponds to the train_data, env, policy, policy_eval of the pipeline
        that performed the best out of all the tuned ones. 
        
        If the pipeline was learnt successfully this object contains: train_data, env, policy, policy_eval. Else this object will 
        have None values for the memebers: train_data, env, policy, policy_eval.
        
        Note that if the AutoRLPipeline Block is working on an online problem then train_data will be None.
        
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None:
        tmp_learn_res = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and 
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(tmp_learn_res, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        #Check that all pipelines pass their consistency_check and also check that all the pipelines end with a block of the same
        #class so that their evaluation can be meaningfully compared. Indeed it makes no sense to confront 2 pipelines, 
        #one which returns a policy and one which returns a policy evaluation:
        are_pipelines_consistent = self.consistency_check(train_data=train_data, env=env)
        
        if(not are_pipelines_consistent):
            self.is_learn_successful = False
            err_msg = 'Either not all pipelines passed their \'consistency_check\' or the pipelines in \'tuner_blocks_dict\''\
                      +' are not consistent: all the pipelines should end with a block of the same Class! Indeed it makes no'\
                      +' sense to confront 2 pipelines one which returns a policy and one which returns a policy evaluation!'
            self.logger.error(msg=err_msg)
            return BlockOutput(obj_name=self.obj_name)
        
        #count how many pipelines i actually tuned:
        count_tuned_pipelines = 0
        
        best_pipeline = None
        best_pipeline_eval = None
        for tmp_key in list(self.tuner_blocks_dict.keys()):
            #pick a Tuner
            tmp_tuner = self.tuner_blocks_dict[tmp_key]
            
            self.logger.info(msg='Now tuning: '+str(tmp_tuner.obj_name)+'_'+str(tmp_tuner.block_to_opt.obj_name))
            
            #initialise self.pipeline_type of each data preparation block contained in each Tuner: pipeline_type can be both 
            #online and offline:
            if(self.pipeline_type == 'offline'):
                tmp_tuner.block_to_opt.pipeline_type = 'offline'
            else:
                tmp_tuner.block_to_opt.pipeline_type = 'online'
                
            #check that the pipeline satisfies pre_learn_check: if it does not i skip over it
            if(not tmp_tuner.block_to_opt.pre_learn_check(train_data=train_data, env=env)):
                log_msg = 'A pipeline of \'AutoRLPipeline\' will not be considered in the tuning procedure since it is not'\
                          +' consistent with the observation and (or) action space and (or) the pipeline type!'
                self.logger.info(msg=log_msg)
                continue                
                
            #call tune method of the Tuner. Returns the tuned pipeline and its evaluation: its evaluation, according to the 
            #provided evaluation metric
            tmp_tuned_pipeline, tmp_tuned_pipeline_eval = tmp_tuner.tune(train_data=train_data, env=env)
            
            #skip over the tuners that have input loader and metric not consistent:
            if(not tmp_tuner.is_metric_consistent_with_input_loader()):
                log_msg = 'A pipeline of \'AutoRLPipeline\' will not be considered in the tuning procedure since its'\
                          +' input loader and metric are not consistent with each other!'
                self.logger.info(msg=log_msg)
                #we need to skip an iteration of the loop
                continue 
            
            if(not tmp_tuner.is_tune_successful):
                self.is_learn_successful = False 
                self.logger.error(msg='There was an error in one of the tuning procedures inside the \'AutoRLPipeline\' block!')
                return BlockOutput(obj_name=self.obj_name)
            
            if((tmp_tuned_pipeline is not None) and (tmp_tuner.is_tune_successful)):
                #if it is the first Tuner I need to assign the value to best_pipeline and best_pipeline_eval. This is also what
                #i do in case it is not the first Tuner and the new evaluation is 'better' than the current best evaluation.
                #For more on what 'better' means look at the method which_one_is_better of the eval_metric.
                if((count_tuned_pipelines == 0) or (self.eval_metric.which_one_is_better(block_1_eval=tmp_tuned_pipeline_eval, 
                                                                                         block_2_eval=best_pipeline_eval) == 0)):
                    best_pipeline = tmp_tuned_pipeline
                    best_pipeline_eval = tmp_tuned_pipeline_eval
                    
                #count how many pipelines i actually tuned:
                count_tuned_pipelines += 1   
            else:
                self.is_learn_successful = False
                self.logger.error(msg='Something went wrong tuning the current block inside the \'AutoRLPipeline\' block!')
                return BlockOutput(obj_name=self.obj_name)

        #if i tuned at least a pipeline:
        if(count_tuned_pipelines > 0):
            self.logger.info(msg='Now learning the best tuned pipeline on the entire starting input...')                
            #call the best tuned pipeline on the entire starting input to the automatic block:
            best_pipeline_res = best_pipeline.learn(train_data=train_data, env=env)
            
            if(not best_pipeline.is_learn_successful):
                self.is_learn_successful = False
                err_msg = 'There was an error in the \'learn\' method of the best pipeline in the \'AutoRLPipeline\' block!'      
                self.logger.error(msg=err_msg)
                return BlockOutput(obj_name=self.obj_name)
            else:
                self.is_learn_successful = True 
                self.logger.info(msg='The \'AutoRLPipeline\' block was successful!')
                return best_pipeline_res
        else:
            self.is_learn_successful = False
            err_msg = 'No pipelines were tuned since they were not compatible with the observation and (or) action space'\
                      +' and (or) the pipeline type! The \'AutoRLPipeline\' block was not successful!'
            self.logger.error(msg=err_msg)
            return BlockOutput(obj_name=self.obj_name)
    
    def consistency_check(self, train_data=None, env=None):
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
        It returns True if the all the pipelines terminate with a block of the same type and if all pipelines are consistent. This
        is important since we cannot compare pipelines (and it makes no sense) that end with a block of different type.
        
        Otherwise it returns False.
        """
      
        first_pipeline_last_block = list(self.tuner_blocks_dict.values())[0].block_to_opt.list_of_block_objects[-1]
        last_block_class = None
        
        #There are 4 types of blocks with which a pipeline can end with: DataGeneration, DataPrepartion, FeatureEngineering,
        #ModelGeneration:
        for x_type in list(DataGeneration, DataPreparation, FeatureEngineering, ModelGeneration):
            if(isinstance(first_pipeline_last_block, x_type)):
                last_block_class = x_type    
              
        for tmp_key in list(self.tuner_blocks_dict.keys()):
              #pick a Tuner
              tmp_tuner = self.tuner_blocks_dict[tmp_key]
              tmp_pipeline = tmp_tuner.block_to_opt
              
              if(not tmp_pipeline.consistency_check(train_data=train_data, env=env)):
                  return False
              
              n = list(self.tuner_blocks_dict.values()).index(tmp_tuner)
              n_pipeline_last_block = list(self.tuner_blocks_dict.values())[n].block_to_opt.list_of_block_objects[-1]
              
              if(not isinstance(n_pipeline_last_block, last_block_class)):
                  return False
              
        #if i reach this point every pipeline is consistent and all pipelines have as last block a block of the same type     
        return True
  
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
        new_params_dict: This is the dictionary for the parameters of all pipelines present in this automatic block.
                        
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
        This method is yet to be implemented.
        """
        
        raise NotImplementedError
        
    def update_verbosity(self, new_verbosity):
        """
        Parameters
        ----------
        new_verbosity: This is an integer and it represents the new verbosity level.
        
        This method calls the method update_verbosity implemented in the Class Block and then it calls such method of every 
        pipeline present in this automatic block.
        """

        super().update_verbosity(new_verbosity=new_verbosity)
        
        for tmp_key in list(self.tuner_blocks_dict.keys()):
            self.tuner_blocks_dict[tmp_key].update_verbosity(new_verbosity=new_verbosity)