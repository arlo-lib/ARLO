"""
This module contains the implementation of the Class OnlineRLPipeline. 

The Class OnlineRLPipeline inherits from the Class RLPipeline.

An OnlineRLPipeline is used to solve an online reinforcement learning problem. 
"""

from ARLO.environment.environment import BaseEnvironment
from ARLO.rl_pipeline.rl_pipeline import RLPipeline
from ARLO.block.block_output import BlockOutput
from ARLO.block.feature_engineering import FeatureEngineering
from ARLO.block.model_generation import ModelGeneration


class OnlineRLPipeline(RLPipeline):
    """
    This Class implements the Online Reinforcement Learning Pipeline. Being Online RL it works on an environment provided by
    the user.
    """
    
    def __init__(self, list_of_block_objects, eval_metric, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, 
                 verbosity=3, n_jobs=1, job_type='process'):
        """  
        The other parameters and non-parameters members are described in the Class RLPipeline.
        """
        
        super().__init__(list_of_block_objects=list_of_block_objects, eval_metric=eval_metric, obj_name=obj_name, 
                         seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, 
                         n_jobs=n_jobs, job_type=job_type)
                                         
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
        This is an object of Class BlockOutput. If the pipeline was learnt successfully this object contains: env, policy, 
        policy_eval. Else this object will have None values for the members: env, policy, policy_eval.
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None:
        tmp_learn_res = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and 
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(tmp_learn_res, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
               
        #note: the consistency check is called in the learn method and not in the __init__ because you could create a
        #consistent pipeline but then modify afterwards the list_of_block_objects in a way such that is is no longer consistent
        if(self.consistency_check(train_data=train_data, env=env)):
            self.logger.info(msg='Learning the online RL pipeline...')
                    
            policy = None
            policy_eval = None

            for tmp_block in self.list_of_block_objects:
                tmp_res = None
                
                self.logger.info(msg='Now learning the following block: '+tmp_block.obj_name)   
                                    
                #Before learning i need to check that the selected block works on the chosen problem. I need to check that 
                #the block works in an offline/online pipeline, that it works in continuous/discrete action/observation spaces
                #note that train_data = None since we never have it in online RL!
                can_proceed = tmp_block.pre_learn_check(train_data=None, env=env)
                 
                if(not can_proceed):
                    self.is_learn_successful = False
                    err_msg = 'The current block failed the check on consistency with the pipeline and the environment spaces!'
                    self.logger.error(msg=err_msg)
                    return BlockOutput(obj_name=self.obj_name)
                
                #fully instantiate model generation tmp block with the info of the MDP of the environment produced by the 
                #previous block. The other blocks are already fully instantiated. I simply pass env since this is kept updated
                #with the results of all blocks:
                if(isinstance(tmp_block, ModelGeneration) and (not tmp_block.fully_instantiated)):
                    block_fully_instantiated = tmp_block.full_block_instantiation(info_MDP = self.get_info_MDP(latest_env=env))
                        
                #there is never train_data in onlineRL
                tmp_res = tmp_block.learn(train_data=None, env=env)      
                    
                #If the block was not learned successfully the pipeline learning process needs to finish here
                if(not tmp_block.is_learn_successful):
                    self.is_learn_successful = False
                    self.logger.error(msg='There was an error learning the current block!')
                    return BlockOutput(obj_name=self.obj_name)
                
                #tmp_res must be an object of Class BlockOutput
                if(not isinstance(tmp_res, BlockOutput)):
                    self.is_learn_successful = False
                    self.logger.error(msg='\'tmp_res\' must be an object of Class \'BlockOutput\'!')
                    return BlockOutput(obj_name=self.obj_name)
                    
                #in online RL every block must return only one output:
                if(tmp_res.n_outputs != 1):
                    self.is_learn_successful = False
                    self.logger.error(msg='In onlineRL each block must return exactly one output! This block did not!')   
                    return BlockOutput(obj_name=self.obj_name)

                #if i reach this point the block was learnt successfully. Each block in onlineRL can either return a
                #BaseEnvironment or a BasePolicy:  
                if(isinstance(tmp_block, ModelGeneration)):
                    #ModelGeneration blocks return a BasePolicy:                                        
                    policy = tmp_res.policy         
                    #computes policy evaluation:
                    self.logger.info(msg='Now evaluating learnt policy...')
                    res_eval = tmp_block.eval_metric.evaluate(block_res=tmp_res, block=tmp_block, env=env)                    
                    policy_eval = {'eval_mean': res_eval, 'eval_var': tmp_block.eval_metric.eval_var}
                elif(isinstance(tmp_block, FeatureEngineering)):
                    #FeatureEngineering blocks return an env:
                    env = tmp_res.env
                
            res =  BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                               checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity, env=env, policy=policy,
                               policy_eval=policy_eval)
            self.is_learn_successful = True
            self.logger.info('Learning is complete!')
            return res
        else:
            self.is_learn_successful = False
            self.logger.error(msg='No learning will occur since the \'list_of_block_objects\' is not consistent!')  
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
        It returns True if the pipeline is consistent, and False otherwise.
             
        First it calls the consistency_check method of the Class RLPipeline. Then it makes sure that there are no DataGeneration,
        nor DataPreparation blocks, as it would not make sense since we are in an online pipeline.
        """
        
        is_base_pipeline_consistent, lst = super().consistency_check(train_data=train_data, env=env)
        
        if(is_base_pipeline_consistent):
            #check that there are no data generation blocks in onlineRLPipeline. Indeed it makes no sense to have such blocks 
            #in an online pipeline.
            if(self.const_DataGeneration in lst):
                self.is_learn_successful = False
                err_msg = 'The \'list_of_block_objects\' contains a \'DataGeneration\' block which does not make sense in'\
                          +' onlineRL!'
                self.logger.error(msg=err_msg)
                return False
            
            #check that there are no data preparation blocks in onlineRLPipeline. Indeed it makes no sense to have such blocks in 
            #an online pipeline because we have a simulator so we don't have missing data and we don't need data augmentation
            if(self.const_DataPreparation in lst):
                self.is_learn_successful = False
                err_msg = 'The \'list_of_block_objects\' contains a \'DataPreparation\' block which does not make sense in'\
                          +' onlineRL!'
                self.logger.error(msg=err_msg)
                return False
            
            #In onlineRL the env cannot be None
            if(env is None):
                self.is_learn_successful = False 
                self.logger.error(msg='In onlineRL the \'env\' cannot be \'None\'!')
                return False      
            
            #if i reach this point then the online pipeline is consistent
            return True
        else:
            self.is_learn_successful = False
            return False
        
    def analyse(self):
        """
        This method analyses the pipeline given the evaluation metrics: it calls the analyse method of each block composing the
        pipeline.
        """
        
        if((self.list_of_block_objects is not None) and self.is_learn_successful):
            for tmp_block in self.list_of_block_objects:
                tmp_block.analyse()        
        else:
            self.logger.error(msg='Either the \'list_of_block_objects\' is empty or the learning procedure was not successful!')    
          
    def get_info_MDP(self, latest_env):
        """
        Parameters
        ----------
        latest_env: This is an object of a Class inheirting from the Class BaseEnvironment.
        
        Returns
        -------
        latest_env.info: This is an object of Class MDPInfo from the library MushroomRL. It contains the observation space, the 
        action space, gamma and the horizon of the environment.
        
        Given latest_env object of a Class inheriting from the Class BaseEnvironment it extracts the member info of such object, 
        which contains the observation space, the action space, gamma and the horizon of the environment.
        """

        #latest_env is contains the latest environment: it is the output of the last block of FeatureEngineering before the
        #ModelGeneration block.
        
        #Note that the block coming before ModelGeneration cannot be of type DataGeneration, DataPreparation since it does not 
        #make sense to have these blocks in online RL!
        if(not isinstance(latest_env, BaseEnvironment)):
            exc_msg = '\'latest_env\' is not an object of a Class inheriting from the Class \'BaseEnvironment\'!'
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg)

        return latest_env.info