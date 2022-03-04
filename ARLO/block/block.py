"""
This module contains the implementation of the Class Block. This is used to represent a generic block: these blocks can make up
the pipeline, it can be an automatic block or it can be a pipeline itself.

The Class Block is an abstract Class and so it inherits both from the Class AbstractUnit and from ABC. 
"""

from abc import ABC, abstractmethod

from mushroom_rl.utils.spaces import Discrete, Box

from ARLO.block.block_output import BlockOutput
from ARLO.abstract_unit.abstract_unit import AbstractUnit
from ARLO.metric.metric import Metric
from ARLO.dataset.dataset import BaseDataSet
from ARLO.environment.environment import BaseEnvironment


class Block(AbstractUnit, ABC):
    """
    This is an abstract Class used to expose the same generic interface.  All blocks that can make up the pipeline, and the 
    pipeline itself, inherit from this Class.
    
    This Class inherits from the Class AbstractUnit.
    """
    
    def __init__(self, eval_metric, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, 
                 n_jobs=1, job_type='process'):
        """      
        Parameters
        ----------            
        eval_metric: This is the metric by which the specific block will be ranked. It must be an object of a Class inheriting
                     from the Class Metric.
    
        Non-Parameters Members
        ----------------------        
        works_on_online_rl: This is either True or False. It is True if the block works on online reinforcement learning, while 
                            it is False otherwise.
                                  
                            The default is True.
                                  
        works_on_offline_rl: This is either True or False. It is True if the block works on offline reinforcement learning, while
                             it is False otherwise.
                                   
                             The default is True.
                                   
        works_on_box_action_space: This is either True or False. It is True if the block works on box action spaces, while it is
                                   False otherwise.
                                    
                                   The default is True.
                                      
        works_on_discrete_action_space: This is either True or False. It is True if the block works on discrete action spaces, 
                                        while it is False otherwise.
                                        
                                        The default is True.
                                        
        works_on_box_observation_space: This is either True or False. It is True if the block works on box observation spaces,
                                        while it is False otherwise.
                                        
                                        The default is True.
                                        
        works_on_discrete_observation_space: This is either True or False. It is True if the block works on discrete observation
                                             spaces, while it is False otherwise.
                                             
                                             The default is True.
                                             
        pipeline_type: This is either 'online' or 'offline' and it represents the type of pipeline in which the current block is 
                       present. This is needed to know in each block the type of pipeline it is being inserted in.
                           
        is_learn_successful: This is True if the block was learnt properly, False otherwise.
        
        is_parametrised: This is either True or False. It is True if the block is parameterised, meaning that it has parameters
                         that can be optimised. One can also create a block with is_parametrised == False. This tells the Class
                         Tuner not to do anything with such a block. 
        
        block_eval: This is used by the Class Tuner to save the evaluation of the block corresponding to a specific set of hyper
                    parameters with respect to the evaluation metric that is being used in the Class Tuner. This is useful for
                    having the information about the block evaluation even after the Class Tuner method tune is called 
                    (i.e: we can access this information in automatic blocks).
                    
                    This parameter is also filled by the analyse method of the block.
                    
                    The default is None.  
            
        The other parameters and non-parameters members are described in the Class AbstractUnit.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)

        self.eval_metric = eval_metric    
        
        if(not isinstance(self.eval_metric, Metric)):
            exc_msg = '\'eval_metric\' must be an object of a Class inheriting from Class \'Metric\'!'
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg)
                
        self.works_on_online_rl = True
        self.works_on_offline_rl = True
        self.works_on_box_action_space = True
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = True
        
        self.pipeline_type = None
        self.is_learn_successful = False               
        self.is_parametrised = True
        self.block_eval = None
    
    def __repr__(self):
        return 'Block'+'('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
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
               
    def pre_learn_check(self, train_data=None, env=None):
        """
        Parameters
        ----------        
        train_data: This can be a dataset that will be used for training. It must be an object of a Class inheriting from the 
                    Class BaseDataSet.
                    
                    The default is None.
                                              
        env: This must be a simulator/environment. It must be an object of a Class inheriting from  the Class BaseEnvironment.
        
             The default is None.
             
        Returns
        -------
        This method returns either True or False.
        
        Before learning a block I need to check that the selected block works on the chosen problem. I need to check that the
        block works in an offline/online pipeline, that it works in continuous/discrete action/observation spaces.
        
        If the selected block can work with the pipeline and environment spaces it returns True, else it returns False.
        """
         
        #I check the pipeline type consistency:
        if(self.pipeline_type == 'online'):
            #only blocks that can be in an online pipeline can be learnt:
            if(not self.works_on_online_rl):
                self.is_learn_successful = False 
                self.logger.error(msg='The pipeline_type is \'online\' but this block cannot work \'online\'!')
                return False
        elif(self.pipeline_type == 'offline'):
            #only blocks that can be in an offline pipeline can be learnt:
            if(not self.works_on_offline_rl):
                self.is_learn_successful = False 
                self.logger.error(msg='The pipeline_type is \'offline\' but this block cannot work \'offline\'!')
                return False
            
        tmp_obs_space = None
        tmp_act_space = None
        #if I have both train_data and env then their observation and action space must be the same so i only need to check one 
        #of them:
        if(train_data is not None):
            tmp_obs_space = train_data.observation_space
            tmp_act_space = train_data.action_space
        elif(env is not None):
            tmp_obs_space = env.observation_space
            tmp_act_space = env.action_space
        else:
             self.is_learn_successful = False 
             self.logger.error(msg='Both \'train_data\' and \'env\' are \'None\': this is not possible!')
             return False
     
        if(isinstance(tmp_obs_space, Box)):
            #since the observation space is a Box i can only learn blocks that work on continuous observation spaces:
            if(not self.works_on_box_observation_space):
                self.is_learn_successful = False
                err_msg = 'The observation space is \'Box\' but this block does not work on \'Box\' observation spaces!'
                self.logger.error(msg=err_msg)
                return False
        elif(isinstance(tmp_obs_space, Discrete)):
            #since the observation space is Discrete i can only learn blocks that work on discrete observation spaces:
            if(not self.works_on_discrete_observation_space):
                self.is_learn_successful = False 
                err_msg = 'The observation space is \'Discrete\' this block does not work on \'Discrete\' observation spaces!'
                self.logger.error(msg=err_msg)
                return False
        else:
            self.is_learn_successful = False 
            self.logger.error(msg='At the moment the only supported spaces are: \'Box\' and \'Discrete\'!')
            return False
        
        if(isinstance(tmp_act_space, Box)):
            #since the action space is a Box i can only learn blocks that work on continuous action spaces:
            if(not self.works_on_box_action_space):
                self.is_learn_successful = False 
                err_msg = 'The action space is \'Box\' but this block does not work on \'Box\' action spaces!'
                self.logger.error(msg=err_msg) 
                return False
        elif(isinstance(tmp_act_space, Discrete)):
            #since the action space is Discrete i can only learn blocks that work on discrete action spaces:
            if(not self.works_on_discrete_action_space):
                self.is_learn_successful = False 
                err_msg = 'The action space is \'Discrete\' but this block does not work on \'Discrete\' action spaces!'
                self.logger.error(msg=err_msg)
                return False
        else:
            self.is_learn_successful = False 
            self.logger.error(msg='At the moment the only supported spaces are: \'Box\' and \'Discrete\'!')
            return False

        #if i reach this point then the pre_learn_check is successful:
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
        This method returns an object of Class BlockOutput.
        
        First i check that the pipeline type of every block is either online or offline and that at least one of: 
        train_data and env are not None.
        
        If the block was not learned successfully the object of Class BlockOutput contains all None, else the object of Class
        BlockOutput has some non-None members which correspond to the output of the block.
        """
        
        #each time i re-learn the block i need to set is_learn_successful to False
        self.is_learn_successful = False 
        
        #pipeline_type can only be online or offline
        if((self.pipeline_type != 'online') and (self.pipeline_type != 'offline')):
            self.is_learn_successful = False 
            self.logger.error(msg='There was an error setting the \'pipeline_type\' of one of the blocks!')
            return BlockOutput(obj_name=self.obj_name)
        
        #train_data and env cannot both be None
        if((train_data is None) and (env is None)):
            self.is_learn_successful = False 
            self.logger.error(msg='\'train_data\' and \'env\' cannot both be \'None\'!')
            return BlockOutput(obj_name=self.obj_name)
                
        if((train_data is not None) and (not isinstance(train_data, BaseDataSet))):
            self.is_learn_successful = False 
            self.logger.error(msg='The \'train_data\' must be an object of a Class inheriting from Class \'BaseDataSet\'!')
            return BlockOutput(obj_name=self.obj_name)
        
        if((env is not None) and (not isinstance(env, BaseEnvironment))):
            self.is_learn_successful = False 
            self.logger.error(msg='The \'env\' must be an object of a Class inheriting from Class \'BaseEnvironment\'!')
            return BlockOutput(obj_name=self.obj_name)
        
    @abstractmethod
    def get_params(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_params(self):
        raise NotImplementedError
    
    def get_metric(self):
        """
        Returns
        -------
        self.eval_metric: The evaluation metric (or KPI) used by the block to assess the quality of the learnt algorithm present 
                          in the block.
        """
        
        return self.eval_metric
    
    def set_metric(self, new_metric):
        """
        Parameters
        ----------
        new_metric: The new metric (or KPI) that the block will use to assess the quality o the learnt algorithm present in the 
                    block.
                    
        Returns
        -------
        This method returns True if new_metric was set successfully, else it returns False.
        """
    
        #can only use metrics from the Class Metric. 
        if(isinstance(new_metric, Metric)):    
            self.eval_metric = new_metric
            return True
        else:
            err_msg = '\'new_metric\' must be an object of a Class inheriting from Class \'Metric\'!'
            self.logger.error(msg=err_msg)
            return False
        
    @abstractmethod
    def analyse(self):
        raise NotImplementedError
    
    def update_verbosity(self, new_verbosity):
        """
        Parameters
        ----------
        new_verbosity: This is an integer and it represents the new verbosity level.
        
        This method calls the method update_verbosity implemented in the Class AbstractUnit and then it calls such method of the
        eval_metric.
        """
        
        super().update_verbosity(new_verbosity=new_verbosity)
        self.eval_metric.update_verbosity(new_verbosity=new_verbosity)