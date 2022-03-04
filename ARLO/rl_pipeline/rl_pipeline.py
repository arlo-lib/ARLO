"""
This module contains the implementation of the Class RLPipeline. This serves as base class for the Classes OnlineRLPipeline and 
OfflineRLPipeline. 

The Class RLPipeline inherits from the Class Block.

A pipeline is a collection of blocks: it can contain user defined blocks, specific blocks or automatic blocks.
"""

from abc import abstractmethod
import copy

from ARLO.block.block import Block
from ARLO.block.data_generation import DataGeneration
from ARLO.block.data_preparation import DataPreparation
from ARLO.block.feature_engineering import FeatureEngineering
from ARLO.block.model_generation import ModelGeneration


class RLPipeline(Block):
    """
    This is an abstract Class and it defines a RLPipeline. A RLPipeline is a reinforcement learning pipeline and it is composed
    of an arbitrary number of blocks. The blocks can be of type: DataGeneration, DataPreparation, FeatureEningeering, 
    ModelGeneration.
    
    This Class inheirts from the Class Block.
    """
    
    def __init__(self, list_of_block_objects, eval_metric, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, 
                 verbosity=3, n_jobs=1, job_type='process'):
        """
        Parameters
        ----------
        list_of_block_objects: This is the list of the block objects that make up the pipeline. It must be specified by the
                               user.       
                            
        Non-Parameters Members
        ----------------------     
        list_of_block_objects_upon_instantiation: This a copy of the original value of list_of_block_objects, namely the value of
                                                  list_of_block_objects that the object got upon creation. This is needed for
                                                  re-loading objects.
        
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
               
        self.works_on_online_rl = True
        if(str(self.__class__.__name__) == 'OfflineRLPipeline'):
            self.works_on_online_rl = False

        self.works_on_offline_rl = not self.works_on_online_rl
        self.works_on_box_action_space = True
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = True
        
        self.list_of_block_objects = list_of_block_objects
        self.list_of_block_objects_upon_instantiation = copy.deepcopy(self.list_of_block_objects)
        
        if(self.works_on_online_rl):
            self.pipeline_type = 'online'
        else:
            self.pipeline_type = 'offline'
                                            
        #set pipeline_type for every block in the pipeline:
        for tmp_block in self.list_of_block_objects:
            tmp_block.pipeline_type = self.pipeline_type
               
        #constants used in self._consistency_check(). This is to avoid usage of magic numbers
        #cf. https://stackoverflow.com/questions/47882/what-is-a-magic-number-and-why-is-it-bad
        self.const_DataGeneration = 1
        self.const_DataPreparation = 2
        self.const_FeatureEngineering = 3
        self.const_ModelGeneration = 4
        
    def __repr__(self):
         return str(self.__class__.__name__)+'('+'list_of_block_objects='+str(self.list_of_block_objects)\
                +', eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)+', seeder='+ str(self.seeder)\
                +', local_prng='+ str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
                +', works_on_online_rl='+str(self.works_on_online_rl)+', works_on_offline_rl='+str(self.works_on_offline_rl)\
                +', works_on_box_action_space='+str(self.works_on_box_action_space)\
                +', works_on_discrete_action_space='+str(self.works_on_discrete_action_space)\
                +', works_on_box_observation_space='+str(self.works_on_box_observation_space)\
                +', works_on_discrete_observation_space='+str(self.works_on_discrete_observation_space)\
                +', pipeline_type='+str(self.pipeline_type)+', is_learn_successful='+str(self.is_learn_successful)\
                +', is_parametrised='+str(self.is_parametrised)+', block_eval='+str(self.block_eval)\
                +', list_of_block_objects_upon_instantiation='+str(self.list_of_block_objects_upon_instantiation)\
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
        pre_learn_check_outcome: This is either True or False. It is True if the pre_learn_check was successful, and False 
                                 otherwise.
        
        If the learning was interrupted we need to reset the value of list_of_block_objects to the value it was when the object
        was created to avoid any problems. Then we call the method pre_learn_check of the Class Block.
        """
        
        #i need to reload the params to the value it was originally
        self.list_of_block_objects = self.list_of_block_objects_upon_instantiation
        
        pre_learn_check_outcome = super().pre_learn_check(train_data=train_data, env=env)
        
        return pre_learn_check_outcome 
    
    def get_params(self):
        """
        Returns
        -------
        entire_dict: This is a flat dictionary containing all the parameters of the parametrised blocks inside the pipeline 
                     (i.e: those blocks that have is_parametrised equal to True).
        
        Calls the get_params method of each block contained in the list_of_block_objects. Note that since the get_params method
        return a deep copy, then this method will too return a deep copy.
        """
        
        entire_dict = {}
        
        block_flag = 0
        
        for tmp_block in self.list_of_block_objects:
            #if the member is_parameterised of the block is True, then it has parameters, else it does not:
            if(tmp_block.is_parametrised):    
                tmp_dict = tmp_block.get_params()
                if(tmp_dict is None):
                    exc_msg = 'The \'get_params\' method of a block in the \'list_of_block_objects\' returned \'None\' even'\
                              +' though that block has \'is_parametrised\' equal to \'True\'!'
                    self.logger.exception(msg=exc_msg)
                    raise ValueError(exc_msg)
                    
                #add block_owner_flag
                for tmp_key in list(tmp_dict.keys()):
                    tmp_dict[tmp_key].block_owner_flag = block_flag 
                
                entire_dict = {**entire_dict, **tmp_dict} 
                
            block_flag += 1 

        return entire_dict

    def set_params(self, new_params):
        """
        Parameters
        ----------
        new_params: This is a dictionary containing all the parameters to be used in all the blocks that have is_parametrised 
                    equal to True (i.e: those that have parameters).
                    
                    It must be a dictionary that does not contain any dictionaries(i.e: all parameters must be at the same level)
                        
        Returns
        -------
        bool: This method returns True if new_params is set correctly, and False otherwise.
        """
        
        if(new_params is not None):
            block_flag = 0
            
            for tmp_block in self.list_of_block_objects:
                if(tmp_block.is_parametrised):    
                    tmp_dict = {}
                    for tmp_key in list(new_params.keys()):
                        if(new_params[tmp_key].block_owner_flag == block_flag):
                            tmp_dict.update({tmp_key: new_params[tmp_key]})
                    
                    set_params_res = tmp_block.set_params(tmp_dict)
                    
                    if(not set_params_res):
                        err_msg = 'There was an error setting the parameters of a block of a pipeline!'
                        self.logger.error(msg=err_msg)
                        return False
                    
                block_flag += 1
                
            return True
        else:
            self.logger.error(msg='Cannot set parameters: \'new_params\' is \'None\'!')
            return False  
  
        
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
        bool, []: This method returns a boolean and a list.
                 
                  This method returns True if the list_of_block_objects is consistent. Else it returns False.  
                  A pipeline is consistent if the DataGeneration blocks come beofre the DataPreparation blocks, which comes 
                  before the FeatureEngineering blocks that come before the ModelGeneration blocks.
              
                  Moreover this method returns also lst which is a list containing one number for each block. Blocks of 
                  DataGeneration are associated to 1, Blocks of DataPreparation are associated to 2, Blocks of FeatureEngineering 
                  are associated to 3, Blocks of ModelGeneration are associated to 4. This is represented by: 
                  const_DataGeneration, const_DataPreparation, const_FeatureEngineering, const_ModelGeneration.
        """
        
        lst = []
        
        for tmp_block in self.list_of_block_objects:
            if(isinstance(tmp_block, DataGeneration)):
                lst.append(self.const_DataGeneration)
            elif(isinstance(tmp_block, DataPreparation)):
                lst.append(self.const_DataPreparation)
            elif(isinstance(tmp_block, FeatureEngineering)):
                lst.append(self.const_FeatureEngineering)
            elif(isinstance(tmp_block, ModelGeneration)):
                lst.append(self.const_ModelGeneration)
        
        #there must be at least one block: this check makes sure that the list_of_block_objects is in the right form since in the 
        #for loop above at least one of the if statements was satisfied:
        if(len(lst) == 0):
            self.logger.error(msg='The \'list_of_block_objects\' is not in the right form: no block was recognised!')
            return False, lst
        
        #the pipeline must only contain blocks whose base type is one of the 4 generic blocks: DataGeneration, DataPreparation,
        #FeatureEngineering, ModelGeneration
        if(len(self.list_of_block_objects) != len(lst)):
            err_msg = 'The \'list_of_block_objects\' must be made only of blocks that inherit from one of the 4 base generic'\
                      +' blocks!'
            self.logger.error(msg=err_msg)
            return False, lst
        
        #there can be only one data generation block:
        if(lst.count(self.const_DataGeneration) > 1):
            self.logger.error(msg='There can be only one \'DataGeneration\' block!')
            return False, lst
        
        #there can be only one model generation block:
        if(lst.count(self.const_ModelGeneration) > 1):
            self.logger.error(msg='There can be only one \'ModelGeneration\' block!')
            return False, lst
            
        #the pipeline must be ordered: DataGeneration->DataPreparation->FeatureEngineering->ModelGeneration
        if(sorted(lst) != lst):
            self.logger.error(msg='The \'list_of_block_objects\' is not consistent!')
            return False, lst
        
        #If i reach this point there were no errors and so the list_of_block_objects is consistent
        self.logger.info(msg='The \'list_of_block_objects\' is consistent!')
        return True, lst
    
    @abstractmethod
    def analyse(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_info_MDP(self):
       raise NotImplementedError
       
    def update_verbosity(self, new_verbosity):
        """
        Parameters
        ----------
        new_verbosity: This is an integer and it represents the new verbosity level.
        
        This method calls the method update_verbosity implemented in the Class Block and then it calls such method of every 
        block present in the pipeline.
        """
        
        super().update_verbosity(new_verbosity=new_verbosity)
        
        for tmp_block in self.list_of_block_objects:
            tmp_block.update_verbosity(new_verbosity=new_verbosity)
        