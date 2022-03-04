"""
This module contains the implementation of the Class OfflineRLPipeline. 

The Class OfflineRLPipeline inherits from the Class RLPipeline.

An OfflineRLPipeline is used to solve an offline reinforcement learning problem. 
"""

from ARLO.dataset.dataset import BaseDataSet
from ARLO.rl_pipeline.rl_pipeline import RLPipeline
from ARLO.block.block_output import BlockOutput
from ARLO.block.data_generation import DataGeneration
from ARLO.block.data_preparation import DataPreparation
from ARLO.block.feature_engineering import FeatureEngineering
from ARLO.block.model_generation import ModelGeneration


class OfflineRLPipeline(RLPipeline):
    """
    This Class implements the Offline Reinforcement Learning Pipeline. Being Offline RL it primarly works on a dataset. It can
    use an environment to extract a dataset where the policy will be learnt, or a dataset can be provided.
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
        This is an object of Class BlockOutput. If the pipeline was learnt successfully this object contains: train_data, env, 
        policy, policy_eval. Else this object will have None values for the members: train_data, env, policy, policy_eval.
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
            self.logger.info('Learning the offline RL pipeline...')
            
            policy = None
            policy_eval = None
                        
            for tmp_block in self.list_of_block_objects:              
                tmp_res = None

                self.logger.info(msg='Now learning the following block: '+tmp_block.obj_name)       
                
                #Before learning i need to check that the selected block works on the chosen problem. I need to check that 
                #the block works in an offline/online pipeline, that it works in continuous/discrete action/observation spaces
                can_proceed = tmp_block.pre_learn_check(train_data=train_data, env=env)

                if(not can_proceed):
                    self.is_learn_successful = False 
                    err_msg = 'The current block failed the check on consistency with the pipeline and the environment spaces!'
                    self.logger.error(msg=err_msg)
                    return BlockOutput(obj_name=self.obj_name)
                
                #fully instantiate model generation tmp block with the info of the MDP of the dataset produced by the 
                #previous block. The other blocks are already fully instantiated. I simply pass train_data since this is kept 
                #updated with the results of all blocks:    
                if(isinstance(tmp_block, ModelGeneration) and (not tmp_block.fully_instantiated)):
                    block_fully_instantiated = tmp_block.full_block_instantiation(info_MDP = self.get_info_MDP(latest_dataset=
                                                                                                               train_data))
                
                tmp_res = tmp_block.learn(train_data=train_data, env=env)                                              

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
                
                #in offline RL DataGeneration, DataPreparation, ModelGeneration blocks must return exactly one output:
                if((tmp_res.n_outputs != 1) and (not isinstance(tmp_block, FeatureEngineering))):
                    self.is_learn_successful = False
                    #By design in offlineRL blocks of type different from FeatureEngineering must return exactly one output:
                    err_msg = 'In offlineRL, blocks of type different from \'FeatureEngineering\' must return exactly one'\
                              +' output! This block did not!' 
                    self.logger.error(msg=err_msg)
                    return BlockOutput(obj_name=self.obj_name)
                    
                #in offline RL FeatureEngineering blocks must return exactly two outputs:
                if((tmp_res.n_outputs != 2) and isinstance(tmp_block, FeatureEngineering)):
                    self.is_learn_successful = False
                    err_msg = 'In offlineRL \'FeatureEngineering\' blocks must return exactly two outputs! This block did not!'
                    self.logger.error(msg=err_msg) 
                    return BlockOutput(obj_name=self.obj_name)
                
                #if i reach this point the block was learnt successfully. Each block in offlineRL can either return a Policy, 
                #the evaluation of the policy, a BaseEnvironment or a BaseDataSet:                    
                if(isinstance(tmp_block, ModelGeneration)):
                    #ModelGeneration blocks return a policy:                                        
                    policy = tmp_res.policy                       
                    #computes policy evaluation:
                    self.logger.info(msg='Now evaluating learnt policy...')
                    res_eval = tmp_block.eval_metric.evaluate(block_res=tmp_res, block=tmp_block, train_data=train_data, env=env)
                    policy_eval = {'eval_mean': res_eval, 'eval_var': tmp_block.eval_metric.eval_var}
                elif(isinstance(tmp_block, DataGeneration) or isinstance(tmp_block, DataPreparation)):
                    #DataGeneration and DataPreparation blocks return BaseDataSet:    
                    train_data = tmp_res.train_data
                elif(isinstance(tmp_block, FeatureEngineering)):
                    #FeatureEngineering blocks return a BaseDataSet and a BaseEnvironment:
                    train_data = tmp_res.train_data
                    env = tmp_res.env
                                                               
            res =  BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                               checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity, 
                               train_data=train_data, env=env, policy=policy, policy_eval=policy_eval)
            
            self.is_learn_successful = True
            self.logger.info(msg='Learning is complete!')
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
             
        First it calls the consistency_check method of the Class RLPipeline. Then it makes sure that: 
            -If the env is provided but the train_data is not then there is a DataGeneration block
            -If both the env and the train_data are provided then there is no DataGeneration block
            -If the train_data is provided but the env is not then there is no DataGeneration block
        """
        
        is_base_pipeline_consistent, lst = super().consistency_check(train_data=train_data, env=env)
        
        if(is_base_pipeline_consistent):          
            #In offlineRL train_data and env cannot both be None
            if((train_data is None) and (env is None)):
                 self.is_learn_successful = False 
                 self.logger.error(msg='In offlineRL \'train_data\' and \'env\' cannot both be \'None\'!')
                 return False
            
            #If both train_data and env are provided: Then it means that for training we have a dataset and for testing we will 
            #need to use this simulator/environment.
            #Moreover: 
            #-no DataGeneration block must be present 
            #-a ModelGeneration block must be present
            if((train_data is not None) and (env is not None)):
                if(self.const_DataGeneration in lst): 
                    self.is_learn_successful = False
                    self.logger.error(msg='We do not need a \'DataGeneration\' block: the \'train_data\' was provided!')
                    return False
                
            #If train_data is provided and env is None: Then it means we are only doing training and no testing can be done.
            #Moreover:
            #-no DataGeneration block must be present 
            if((train_data is not None) and (env is None)):
                if(self.const_DataGeneration in lst): 
                    self.is_learn_successful = False
                    self.logger.error(msg='We do not need a \'DataGeneration\' block: the \'train_data\' was provided!')
                    return False
                                
            #If train_data is None and env is provided: If we want to do DataPreparation, FeatureEngineering or ModelGeneration
            #we need a DataGeneration block.
            if((train_data is None) and (env is not None)):
                if(self.const_DataGeneration not in lst):      
                    self.is_learn_successful = False
                    self.logger.error(msg='We need a \'DataGeneration\' block: the \'train_data\' was not provided!')
                    return False
                
            #if i reach this point then the offline pipeline is consistent
            return True
        else:
            self.is_learn_successful = False
            return False
        
    def analyse(self):
        """
        This method analyses the pipeline given the evaluation metrics.
        """
        
        if((self.list_of_block_objects is not None) and (self.is_learn_successful)):
            for tmp_block in self.list_of_block_objects:
                tmp_block.analyse()        
        else:
            self.logger.error(msg='Either the \'list_of_block_objects\' is empty or the learning procedure was not successful!')
                
    def get_info_MDP(self, latest_dataset):
        """
        Parameters
        ----------
        latest_dataset: This is an object of a Class inheirting from the Class BaseDataSet.
        
        Returns
        -------
        latest_dataset.info: This is an object of Class MDPInfo from the library MushroomRL. It contains the observation space, 
        the action space, gamma and the horizon of the dataset.
        
        Given latest_dataset object of a Class inheriting from the Class BaseDataSet it extracts the member info of such object, 
        which contains the observation space, the action space, gamma and the horizon of the dataset.
        """
        
        #latest_dataset is contains the latest dataset: it is the output of the last block of DataGeneration, or of 
        #DataPreparation, or of FeatureEngineering before the ModelGeneration block.
        
        if(not isinstance(latest_dataset, BaseDataSet)):
            exc_msg = '\'latest_dataset\' is not an object of a Class inheriting from the Class \'BaseDataSet\'!'
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg)
            
        return latest_dataset.info