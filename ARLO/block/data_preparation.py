"""
This module contains the implementation of the Class DataPreparation, DataPreparationIdentity, DataPreparationImputation, 
DataPreparation1NNImputation and DataPreparationMeanImputation. 

The Class DataPreparation inherits from the Class Block, while the Classes DataPreparationIdentity and DataPreparationImputation 
inherit from the Class DataPreparation.

The Classes DataPreparation1NNImputation and DataPreparationMeanImputation inherit from the Class DataPreparationImputation.

The Class DataPreparation is a Class used to group all Classes that do DataPreparation.

The Class DataPreparationIdentity implements a specific data preparation algorithm: it is an identity block meaning that it will 
not do anything on the input train_data.

The Classes inheriting from the Class DataPreparationImputation all do different kind of data imputation.
"""

from abc import abstractmethod
import copy
import numpy as np
from sklearn.impute import KNNImputer

from mushroom_rl.utils.spaces import Box

from ARLO.block.block_output import BlockOutput
from ARLO.block.block import Block


class DataPreparation(Block):
    """
    This is an abstract class. It is used as generic base class for all data preparation blocks.
    """
    
    def __repr__(self):
        return str(self.__class__.__name__)+'('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
               +', seeder='+ str(self.seeder)+', local_prng='+ str(self.local_prng)+', log_mode='+str(self.log_mode)\
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
        If the call to the method learn() implemented in the Class Block was successful it returns the train_data. Else it
        returns an empty object of Class BlockOutput (to signal that something up the chain went wrong).
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None:          
        tmp_out = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(tmp_out, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        #i need an train_data to perform data generation:
        if(train_data is None):
            self.is_learn_successful = False
            self.logger.error(msg='The \'train_data\' must not be \'None\'!')
            return BlockOutput(obj_name=self.obj_name)
            
        #below i select only the inputs relevant to the current type of block: DataPreparation blocks work on the train_data:
        return train_data
             
    @abstractmethod
    def get_params(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_params(self):
        raise NotImplementedError
   
    @abstractmethod
    def analyse(self):
        raise NotImplementedError
    
        
class DataPreparationIdentity(DataPreparation):
    """
    This Class implements a specific data preparation algorithm: it is an identity block meaning that it will not do anything
    on the input train_data. This can be useful when jointly optimising the pipeline.    
    
    This Class inherits from the Class DataPreparation.
    """
    
    def __init__(self, eval_metric, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """                                                
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
          
        #this block has no parameters
        self.is_parametrised = False
        
        #Set algo_params and algo_params_upon_instantiation for consitency. This is also needed else we would need to modify the
        #method pre_learn_check()
        self.algo_params = None
        self.algo_params_upon_instantiation = None
        
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
        TabularDataSet where the dataset member is exactly the same as the one provided in input to the block. Indeed this is
        an identity block.
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_dataset = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_dataset, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        #identity block
        new_dataset = starting_dataset

        res = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                          checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity, train_data=new_dataset)   
        self.is_learn_successful = True
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object learnt successfully!')
        return res
    
    def get_params(self):
        """
        Returns
        -------
        This method returns None since this block does not have any parameters.
        """
        
        err_msg = 'The block \''+str(self.__class__.__name__)+'\' does not have any parameters!'
        self.logger.error(msg=err_msg)
        return None
    
    def set_params(self, new_params):
        """
        Parameters
        ----------
        This method always returns False since this block does not have any parameters.
        """
        
        err_msg = 'The block \''+str(self.__class__.__name__)+'\' does not have any parameters!'
        self.logger.error(msg=err_msg)
        return False
        
    def analyse(self):
        """
        This method is yet to be implemented.
        """

        raise NotImplementedError
        
        
class DataPreparationImputation(DataPreparation):
    """
    This Class is used to group different operations common to all Data Imputation blocks.
    
    This Class is an Abstract Class and it inherits from the Class DataPreparation.
    """
    
    def _transform_data_for_imputation(self, train_data):
        """
        Parameters
        ----------
        train_data: This can be a dataset that will be used for training. It must be an object of a Class inheriting from Class
                    TabularDataSet.            
        Returns
        -------
        stacked_dataset: This is a numpy.ndarray containing all the samples of the dataset where the features are the 
                         current state, the action, the reward and the next state.    
        """
        
        stacked_dataset = []
        
        #I assume to have a vector of data that is of the proper length, and some of its members are numpy.nan
        for i in range(len(train_data.dataset)):
            current_sample = train_data.dataset[i]
            
            new_sample = np.hstack((current_sample[0].ravel(), 
                                    current_sample[1].ravel(),
                                    current_sample[2],
                                    current_sample[3].ravel())).tolist()
            
            stacked_dataset.append(new_sample)
        
        return stacked_dataset
    
    def get_params(self):
        """
        Returns
        -------
        This method returns None since this block does not have any parameters.
        """
        
        err_msg = 'The block \''+str(self.__class__.__name__)+'\' does not have any parameters!'
        self.logger.error(msg=err_msg)
        return None
    
    def set_params(self, new_params):
        """
        Parameters
        ----------
        This method always returns False since this block does not have any parameters.
        """
        
        err_msg = 'The block \''+str(self.__class__.__name__)+'\' does not have any parameters!'
        self.logger.error(msg=err_msg)
        return False
        
    def analyse(self):
        """
        This method is yet to be implemented.
        """

        raise NotImplementedError


class DataPreparation1NNImputation(DataPreparationImputation):
    """
    This Class implements a specific data preparation algorithm: it performs imputation of the missing data using 1-NN imputation.
    Note that it can impute only non-boolean data: the terminal episodes flags and the absorbing state flags are not imputed.
    Moreover the dataset is assumed to have missing values.
    
    This Class inherits from the Class DataPreparationImputation.
    """
    
    def __init__(self, eval_metric, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """                                                
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
          
        #this block has no parameters
        self.is_parametrised = False
        
        #Set algo_params and algo_params_upon_instantiation for consitency. This is also needed else we would need to modify the
        #method pre_learn_check()
        self.algo_params = None
        self.algo_params_upon_instantiation = None
        
    def _impute_data(self, train_data):
        """
        Parameters
        ----------
        train_data: This can be a dataset that will be used for training. It must be an object of a Class inheriting from Class
                    TabularDataSet.
        Returns
        -------
        new_tabular_dataset: This is a new object of Class TabularDataSet containing the imputed dataset.
        """
    
        knn_imputer = KNNImputer(n_neighbors=1)
        
        formatted_data = self._transform_data_for_imputation(train_data=train_data)
        
        imputed_res = knn_imputer.fit_transform(formatted_data)
        
        if(isinstance(train_data.observation_space, Box)):
            size_obs_space = len(train_data.observation_space.low)
        else:
            size_obs_space = 1
            
        if(isinstance(train_data.action_space, Box)):
            size_act_space = len(train_data.action_space.low)
        else:
            size_act_space = 1
        
        parsed_data = train_data.parse_data() 
        imputed_dataset = train_data.arrays_as_data(imputed_res[:,:size_obs_space], 
                                                    imputed_res[:,size_obs_space:size_obs_space+size_act_space],
                                                    imputed_res[:,size_obs_space+size_act_space],
                                                    imputed_res[:,size_obs_space+size_act_space+1:
                                                                  2*size_obs_space+size_act_space+1],
                                                    parsed_data[4],
                                                    parsed_data[5]
                                                    )
        
        new_tabular_dataset = copy.deepcopy(train_data)
        new_tabular_dataset.dataset = imputed_dataset
        
        return new_tabular_dataset
    
    def learn(self, train_data=None, env=None):
        """
        Parameters
        ----------
        train_data: This can be a dataset that will be used for training. It must be an object of a Class inheriting from Class
                    TabularDataSet.
                    
                    The default is None.
                                              
        env: This must be a simulator/environment. It must be an object of a Class inheriting from Class BaseEnvironment.
        
             The default is None.

        Returns
        -------
        This method returns an object of Class BlockOutput in which in the member train_data there is an object of Class 
        TabularDataSet where the dataset member is imputed using 1-NN imputation.
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_dataset = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_dataset, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        new_dataset = self._impute_data(train_data=starting_dataset)

        res = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                          checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity, train_data=new_dataset)   
        self.is_learn_successful = True
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object learnt successfully!')
        return res
    

class DataPreparationMeanImputation(DataPreparationImputation):
    """
    This Class implements a specific data preparation algorithm: it performs imputation of the missing data using mean imputation.
    Note that it can impute only non-boolean data: the terminal episodes flags and the absorbing state flags are not imputed.
    Moreover the dataset is assumed to have missing values.
    
    This Class inherits from the Class DataPreparationImputation.    
    """
    
    def __init__(self, eval_metric, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """                                                
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
          
        #this block has no parameters
        self.is_parametrised = False
    
        #Set algo_params and algo_params_upon_instantiation for consitency. This is also needed else we would need to modify the
        #method pre_learn_check()
        self.algo_params = None
        self.algo_params_upon_instantiation = None
                
    def _impute_data(self, train_data):
        """
        Parameters
        ----------
        train_data: This can be a dataset that will be used for training. It must be an object of a Class inheriting from Class
                    TabularDataSet.
        Returns
        -------
        new_tabular_dataset: This is a new object of Class TabularDataSet containing the imputed dataset.
        """
        
        size_obs_space = len(train_data.observation_space.low)
  
        size_act_space = len(train_data.action_space.low)

        formatted_data = np.array(self._transform_data_for_imputation(train_data=train_data))
        
        mean_value = np.nanmean(formatted_data, axis=0)
        for i in range(2*size_obs_space+size_act_space+1):
            formatted_data[np.isnan(formatted_data[:,i]),i] = mean_value[i]
        
        parsed_data = train_data.parse_data() 
        imputed_dataset = train_data.arrays_as_data(formatted_data[:,:size_obs_space], 
                                                    formatted_data[:,size_obs_space:size_obs_space+size_act_space],
                                                    formatted_data[:,size_obs_space+size_act_space],
                                                    formatted_data[:,size_obs_space+size_act_space+1:
                                                                     2*size_obs_space+size_act_space+1],
                                                    parsed_data[4],
                                                    parsed_data[5]
                                                    )
        
        new_tabular_dataset = copy.deepcopy(train_data)
        new_tabular_dataset.dataset = imputed_dataset
        
        return new_tabular_dataset
    
    def learn(self, train_data=None, env=None):
        """
        Parameters
        ----------
        train_data: This can be a dataset that will be used for training. It must be an object of a Class inheriting from Class
                    TabularDataSet.
                    
                    The default is None.
                                              
        env: This must be a simulator/environment. It must be an object of a Class inheriting from Class BaseEnvironment.
        
             The default is None.

        Returns
        -------
        This method returns an object of Class BlockOutput in which in the member train_data there is an object of Class 
        TabularDataSet where the dataset member is imputed using mean imputation.
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_dataset = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_dataset, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        new_dataset = self._impute_data(train_data=starting_dataset)

        res = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                          checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity, train_data=new_dataset)   
        self.is_learn_successful = True
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object learnt successfully!')
        return res