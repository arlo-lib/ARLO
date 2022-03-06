"""
This module contains the implementation of the Class FeatureEngineering, of the Class FeatureEngineeringIdentity, of the Class
FeatureEngineeringRFS, of the Class FeatureEngineeringFSCMI and of the Class FeatureEngineeringNystroemMap. 

The Class FeatureEngineering inherits from the Class Block, while the Classes FeatureEngineeringIdentity, FeatureEngineeringRFS 
FeatureEngineeringFSCMI, and FeatureEngineeringNystroemMap all inherit from the Class FeatureEngineering.

The Class FeatureEngineering is a Class used to group all Classes that do FeatureEngineering.
"""

from abc import abstractmethod
import copy
import numpy as np
from joblib import Parallel, delayed
from scipy.special import digamma
from scipy.spatial import cKDTree

from mushroom_rl.utils.spaces import Box
from sklearn.kernel_approximation import Nystroem
from xgboost import XGBRegressor

from ARLO.block.block_output import BlockOutput
from ARLO.block.block import Block
from ARLO.block.data_generation import DataGenerationRandomUniformPolicy
from ARLO.environment.environment import BaseObservationWrapper, BaseActionWrapper
from ARLO.dataset.dataset import TabularDataSet
from ARLO.metric.metric import SomeSpecificMetric
from ARLO.hyperparameter.hyperparameter import Real, Integer, Categorical


class FeatureEngineering(Block):
    """
    This is an abstract Class. It is used as generic base Class for all feature engineering blocks. 
    """
    
    def __repr__(self):
        return 'FeatureEngineering'+'('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
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
    
    def _select_current_actual_value_from_hp_classes(self, params_objects_dict):  
        """
        Parameters
        ----------
        params_objects_dict: This is a flat dictionary containing object of a Class inheriting from the Class HyperParameter.
            
        Returns
        -------
        algo_params_values: This is a flat dictionary containing the current_actual_value of the corresponding HyperParameter 
                            object.
        """
        
        
        algo_params_values = copy.deepcopy(params_objects_dict)
        
        for tmp_key in list(algo_params_values.keys()):
            algo_params_values.update({tmp_key: algo_params_values[tmp_key].current_actual_value})
             
        return algo_params_values
                  
    @abstractmethod
    def _feature_engineer_env(self, old_env):
        #this is impemented in the actual FeatureEngineering blocks
        raise NotImplementedError
        
    @abstractmethod 
    def _feature_engineer_data(self, old_data):
        #this is impemented in the actual FeatureEngineering blocks
        raise NotImplementedError
            
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
        This method calls the method learn() implemented in the Class Block. If the pipeline_type is compliant with what is 
        being passed to this method the train_data and env are passed on, else an empty object of Class BlockOutput is returned.
        
        If the pipeline_type is 'online' the env must be not None. If the pipeline_type is 'offline' the train_data must not be
        None.
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None:           
        tmp_out = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(tmp_out, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
             
        #below i select only the inputs relevant to the current type of block: FeatureEngineering blocks work on the train_data 
        #and on the env:
        if(self.pipeline_type == 'online'):
            if(env is None):
                self.is_learn_successful = False
                self.logger.error(msg='A \'FeatureEngineering\' block of an OnlineRLPipeline got an \'env\' that is \'None\'!')
                return BlockOutput(obj_name=self.obj_name)    
        elif(self.pipeline_type == 'offline'):
            if(train_data is None):
                self.is_learn_successful = False
                err_msg = 'A \'FeatureEngineering\' block of an OfflineRLPipeline got a \'train_data\' that is \'None\'!'
                self.logger.error(msg=err_msg)
                return BlockOutput(obj_name=self.obj_name)    
            
            #if env is None: i raise a warning so that the user knows it
            if(env is None):
                warn_msg = 'In an offline pipeline a specific \'FeatureEngineering\' Block got an \'env\' that is \'None\'!'
                self.logger.warning(msg=warn_msg)
            
        return train_data, env
    
    def get_params(self):
        """
        Returns
        -------
        A deep copy of the parameters in the dictionary self.algo_params.
        """
        
        return copy.deepcopy(self.algo_params)
    
    def set_params(self, new_params):
        """
        Parameters
        ----------
        new_params: The new parameters to be used in the specific feature engineering algorithm. It must be a dictionary that 
                    does not contain any dictionaries(i.e: all parameters must be at the same level).
                    
        Returns
        -------
        bool: This method returns True if new_params is set correctly, and False otherwise.
        """
        
        if(new_params is not None):   
            self.algo_params = new_params
            
            current_params = self.get_params()
            
            if(current_params is not None):
                self.algo_params_upon_instantiation = copy.deepcopy(current_params)
                return True
            else:
                self.logger.error(msg='There was an error getting the parameters!')
                return False
        else:
            self.logger.error(msg='Cannot set parameters: \'new_params\' is \'None\'!')
            return False            
          
    @abstractmethod
    def analyse(self):
        raise NotImplementedError
        
        
class FeatureEngineeringIdentity(FeatureEngineering):
    """
    This Class implements a specific feature engineering algorithm: it provides the identity block. This simply returns the env 
    and the train_data as is without changing anything.
    
    This might be useful when jointly optimising the entire pipeline.
    """
    
    def __init__(self, eval_metric, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1,
                 job_type='process'):
        """        
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
        
        #this block has no parameters
        self.is_parametrised = False
           
        #Set algo_params and algo_params_upon_instantiation for consitency. This is also needed else we would need to modify the
        #method pre_learn_check()
        self.algo_params = None
        self.algo_params_upon_instantiation = None
        
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
        
    def _feature_engineer_env(self, old_env):
        """
        Parameters
        ----------
        old_env: This is the env as is before entering this block. It must be an object of a Class inheriting from the Class    
                 BaseEnvironment.

        Returns
        -------
        new_env: This is the env as is after going through this block. It must be an object of a Class inheriting from the Class    
                 BaseEnvironment.
                 
                 Since this is an identity block this is the same as old_env.
        """
        
        #identity block
        new_env = old_env

        return new_env
        
    def _feature_engineer_data(self, old_data):
        """
        Parameters
        ----------
        old_data: This is the train_data as is before entering this block. It must be an object of a Class inheriting from the 
                  Class BaseDataSet.

        Returns
        -------
        new_data: This is the train_data as is after going through this block. It must be an object of a Class inheriting from 
                  the Class BaseDataSet.
                 
                  Since this is an identity block this is the same as old_data.
        """
        
        #identity block
        new_data = old_data
        
        return new_data
    
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
        outputFE: This is an object of Class BlockOutput containing as train_data the same train_data that entered this block, 
                  and as env the same env that entered this block.
                  
                  If the call to the method learn() implemented in the Class FeatureEngineering was not successful the object of
                  Class BlockOutput is empty.
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_data_env = super().learn(train_data=train_data, env=env)      
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_data_env, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        new_train_data = None
        new_env = None
        
        #starting_data_env is a list of two: train_data and env. In starting_data_env[0] we have the train_data and in 
        #starting_data_env[1] we have the env. On these we call the methods feature_engineer_data and feature_engineer_env
        #only in case they are not None:
        if(starting_data_env[0] is not None):
            new_train_data = self._feature_engineer_data(old_data=starting_data_env[0])            
        if(starting_data_env[1] is not None):
            new_env = self._feature_engineer_env(old_env=starting_data_env[1])
        
        outputFE = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                               checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity, train_data=new_train_data, 
                               env=new_env)
            
        self.is_learn_successful = True
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object learnt successfully!')
        return outputFE

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


class FeatureEngineeringRFS(FeatureEngineering):
    """
    This Class implements a specific feature engineering algorithm: it provides a recursive feature selection algorithm for RL.
    
    First an XGBRegressor is fitted on a dataset where the target is the reward and the features are the current state and 
    actions. Then a threshold value is picked and only the meaningful features are kept: this is done via the member 
    feature_importances_ of XGBRegressor.
    
    We continue applying the above procedure recursively taking as target all the next-states. We stop when all the states have 
    been explained, that is when the only variable deemed important for a next state variable is the previous state variable 
    itself.
    
    At the end the list of states that are selected are the ones that appears as important variables at least once in the 
    procedure. 
    
    This block can only be used on continuous (Box) observation spaces. If the action space is continuous (Box) the feature 
    selection will also be applied to the actions, else if the action space is discrete no feature selection will be applied to 
    the actions.
    """
    
    def __init__(self, eval_metric, obj_name, seeder=2, algo_params=None, data_gen_block_for_env_wrap=None, log_mode='console', 
                 checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):
        """        
        Parameters
        ----------
        algo_params: This contains the parameters of the algorithm implemented in this class. 
                     
                     The default is None.
                     
                     If None the following will be used:
                     'threshold': 0.1
                     'n_recursions': 100
                     'feature_selector': XGBRegressor()
                     
        data_gen_block_for_env_wrap: This must be an object of a Class inheriting from the Class DataGeneration. This is used to 
                                     extract a dataset from the environment and this dataset will be used for fitting the feature
                                     engineering algorithm. This is needed because otherwise we would fit and transform a single
                                     sample from the environment at the time and thus we would have different features across
                                     different samples from the environment.
                                     
                                     This must extract an object of a Class inheriting from the Class TabularDataSet.
                                     
                                     This is only used if the method learn() does receives an env but does not receive train_data.
                                     
                                     The default is None.
                                     
                                     If None a DataGenerationRandomUniformPolicy with 'n_samples': 100000 will be used.
                                         
        Non-Parameters Members
        ----------------------                                     
        algo_params_upon_instantiation: This a copy of the original value of algo_params, namely the value of
                                        algo_params that the object got upon creation. This is needed for re-loading
                                        objects.
                                        
        count: This is needed to make sure to stop the recursive feature elimination after the prescribed number.
       
               This is a non-negative integer: after the method _recursive_call has been called more than n_recursion times we 
               stop the feature selection procedure.
       
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.works_on_online_rl = True
        self.works_on_offline_rl = True
        self.works_on_box_action_space = True
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = False
        
        #this block has parameters and I may want to tune them:
        self.is_parametrised = True
        
        self.algo_params = algo_params
       
        self.data_gen_block_for_env_wrap = data_gen_block_for_env_wrap
  
        if(self.data_gen_block_for_env_wrap is None):
            #extract data from the env using a DataGeneration block. This is needed because I need to have the same 
            #transformation for the entire environment, and the only way to do so is to fit the transformation on the same 
            #dataset. Otherwise for each call of the step method i would fit a different transformation, and so the operation 
            #would not be consistent among different calls of the step method of the same environment.
            data_gen_params = dict(eval_metric=SomeSpecificMetric(obj_name='place_holder_metric'), 
                                   obj_name='data_gen_for_wrappin_env', seeder=self.seeder, 
                                   algo_params={'n_samples': Integer(hp_name='n_samples', obj_name='n_samples_data_gen', 
                                                current_actual_value=100000)}, 
                                   log_mode=self.log_mode, 
                                   checkpoint_log_path=self.checkpoint_log_path, verbosity=0, n_jobs=self.n_jobs, 
                                   job_type=self.job_type)
            
            self.data_gen_block_for_env_wrap = DataGenerationRandomUniformPolicy(**data_gen_params)
            
        if(self.algo_params is None):
            threshold = Real(hp_name='threshold', obj_name='threshold', current_actual_value=0.1, range_of_values=[0.1,0.8], 
                             to_mutate=True) 
            
            n_recursions = Integer(hp_name='n_recursions', obj_name='n_recursions', current_actual_value=100, to_mutate=False) 
            
            feature_selector = Categorical(hp_name='feature_selector', obj_name='feature_selector', to_mutate=False,
                                           current_actual_value=XGBRegressor(random_state=self.seeder, n_jobs=self.n_jobs))
            
            self.algo_params = {'threshold': threshold, 'n_recursions': n_recursions,'feature_selector': feature_selector}
            
        self.algo_params_upon_instantiation = copy.deepcopy(self.algo_params)
        
        #used to stop after n_recursions
        self.count = 0
        
    def __repr__(self):
       return str(self.__class__.__name__)+'('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
              +', seeder='+ str(self.seeder)+', local_prng='+ str(self.local_prng)+', algo_params='+str(self.algo_params)\
              +', data_gen_block_for_env_wrap='+str(self.data_gen_block_for_env_wrap)+', log_mode='+str(self.log_mode)\
              +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
              +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
              +', works_on_online_rl='+str(self.works_on_online_rl)+', works_on_offline_rl='+str(self.works_on_offline_rl)\
              +', works_on_box_action_space='+str(self.works_on_box_action_space)\
              +', works_on_discrete_action_space='+str(self.works_on_discrete_action_space)\
              +', works_on_box_observation_space='+str(self.works_on_box_observation_space)\
              +', works_on_discrete_observation_space='+str(self.works_on_discrete_observation_space)\
              +', pipeline_type='+str(self.pipeline_type)+', is_learn_successful='+str(self.is_learn_successful)\
              +', is_parametrised='+str(self.is_parametrised)+', block_eval='+str(self.block_eval)\
              +', algo_params_upon_instantiation='+str(self.algo_params_upon_instantiation)+', logger='+str(self.logger)+')'      
        
    def _from_tabular_dataset_extract_y(self, original_train_data, idx_target):
        """
        Parameters
        ----------
        original_train_data: This is the original train_data and it must be an object of a Class inheriting from the Class 
                             TabularDataSet.
                             
        idx_target: This is a list containing the index of the target variable.

        Returns
        -------
        y: This is a numpy.array containing the values of the target variable. This is the y parameter used in the method fit() 
           of XGBRegressor.
        """
        
        parsed_data = original_train_data.parse_data()
        
        data_as_arrays = np.hstack((parsed_data[0], parsed_data[1], parsed_data[2].reshape(-1,1), parsed_data[3]))

        y = data_as_arrays[:,np.array(idx_target)].tolist()
                      
        y = np.array(y).ravel() 
                
        return y 
            
    def _recursive_call(self, original_train_data, list_of_selected_state_features, list_of_selected_action_features, 
                        list_of_states_not_explained):
        """
        Parameters
        ----------
        original_train_data: This is the original train_data and it must be an object of a Class inheriting from the Class 
                             TabularDataSet.

        list_of_selected_state_features: This is the list containing all the state features that appeared at least once 
                                         throughout all the recursive calls of this method.
            
        list_of_selected_action_features: This is the list containing all the action features that appeared at least once  
                                          throughout all the recursive calls of this method.
        
        list_of_states_not_explained: This is the list used to determine when to stop: it contains the states variables that are 
                                      deemed not explained: those for which the target next state variable is not explained just
                                      by its previous value but by also other state variables.
                                      
        This method is recursive.
        """   
        
        self.count += 1
        
        if(len(list_of_states_not_explained) != 0 and self.count < self.algo_params['n_recursions'].current_actual_value):
            for tmp_state in list_of_states_not_explained:
                if(tmp_state not in list_of_selected_state_features):
                    list_of_selected_state_features.append(tmp_state)

                y = self._from_tabular_dataset_extract_y(original_train_data=original_train_data, 
                                                         idx_target=[len(self.list_feats)+1+tmp_state])
                
                self.algo_params['feature_selector'].current_actual_value.fit(self.X, y)
                
                features_importances = self.algo_params['feature_selector'].current_actual_value.feature_importances_
                
                meaningful_features = features_importances[features_importances > 
                                                           self.algo_params['threshold'].current_actual_value]
                
                if(len(meaningful_features) == 0):
                    return
                
                meaningful_state_features = []
                meaningful_action_features = []
                for j in range(len(meaningful_features)):
                    idx_current = list(features_importances).index(meaningful_features[j])
        
                    if(idx_current < self.dim_observation_space):
                        meaningful_state_features.append(idx_current)
                    else:
                        meaningful_action_features.append(idx_current-self.dim_observation_space-1)

                actions_to_add = []
                if((meaningful_action_features != []) and (list_of_selected_action_features != [])):
                    for elem in meaningful_action_features:
                        if(elem not in list_of_selected_action_features):
                            actions_to_add.append(elem)
                        
                if(actions_to_add != []):
                    list_of_selected_action_features = list_of_selected_action_features + copy.deepcopy(actions_to_add)
                
                if(len(meaningful_state_features) != 0):
                    if(len(meaningful_state_features) != 1):
                        self._recursive_call(original_train_data=original_train_data,
                                             list_of_selected_state_features=list_of_selected_state_features, 
                                             list_of_selected_action_features=list_of_selected_action_features, 
                                             list_of_states_not_explained=meaningful_state_features)
                    elif((len(meaningful_state_features) == 1) and (meaningful_state_features != [tmp_state])):
                        self._recursive_call(original_train_data=original_train_data,
                                             list_of_selected_state_features=list_of_selected_state_features, 
                                             list_of_selected_action_features=list_of_selected_action_features, 
                                             list_of_states_not_explained=meaningful_state_features)
    
    def _recursive_feature_selection(self, original_train_data):
        """
        Parameters
        ----------
        original_train_data: This is the original train_data and it must be an object of a Class inheriting from the Class 
                             TabularDataSet.

        Returns
        -------
        list_of_selected_state_features: This is the list of state features to keep. This can be None.
        
        list_of_selected_action_features: This is the list of the action features to keep. This can be None. 
        """
                
        dim_action_space = None
        dim_observation_space = None
        
        original_observation_space = original_train_data.observation_space
        original_action_space = original_train_data.action_space
        
        #this block only works on Box observation spaces: this is checked in the pre_learn_check method so we can safely assume
        #that at this point the observation space is a Box:
        dim_observation_space = original_observation_space.low.shape[0]
        
        if(isinstance(original_action_space, Box)):
            dim_action_space = original_action_space.low.shape[0]
        else:
            dim_action_space = 1
        
        self.dim_observation_space = dim_observation_space
        
        list_of_selected_state_features = []
        list_of_selected_action_features = []
        
        len_obs_space_act_space = dim_observation_space + dim_action_space
        
        self.list_feats = list(np.arange(len_obs_space_act_space))
           
        parsed_data = original_train_data.parse_data()
        
        data_as_arrays = np.hstack((parsed_data[0], parsed_data[1], parsed_data[2].reshape(-1,1), parsed_data[3]))
        
        self.X = data_as_arrays[:,np.array(self.list_feats)].tolist()
                    
        y = self._from_tabular_dataset_extract_y(original_train_data=original_train_data, idx_target=[len_obs_space_act_space])
        
        self.algo_params['feature_selector'].current_actual_value.fit(self.X, y)
        
        features_importances = self.algo_params['feature_selector'].current_actual_value.feature_importances_
        meaningful_features = features_importances[features_importances > self.algo_params['threshold'].current_actual_value]
        
        meaningful_state_features = []
        meaningful_action_features = []
        
        if(len(meaningful_features) == 0):
            return None, None
        
        for j in range(len(meaningful_features)):
            idx_current = list(features_importances).index(meaningful_features[j])
        
            if(idx_current < dim_observation_space):
                meaningful_state_features.append(idx_current)
            else:
                meaningful_action_features.append(idx_current-self.dim_observation_space-1)
            
        list_of_states_not_explained = meaningful_state_features

        list_of_selected_action_features = meaningful_action_features
        
        if(len(list_of_states_not_explained) == 0):
            #i count this as a fail and i do not remove any state features nor action features
            return None, None
        
        self._recursive_call(original_train_data=original_train_data,
                             list_of_selected_state_features=list_of_selected_state_features, 
                             list_of_selected_action_features=list_of_selected_action_features, 
                             list_of_states_not_explained=list_of_states_not_explained)
        
        if((len(list_of_selected_state_features) == 0) or 
           (isinstance(original_action_space, Box) and len(list_of_selected_action_features) == 0)):
            #i count this as a fail and i do not dremove any features
            return None, None 
        
        list_of_selected_state_features = np.sort(list_of_selected_state_features).tolist()
        list_of_selected_action_features = np.sort(list_of_selected_action_features).tolist()
        
        return list_of_selected_state_features, list_of_selected_action_features
        
    def _feature_engineer_env(self, old_env):
        """
        Parameters
        ----------
        old_env: This is the env as is before entering this block. It must be an object of a Class inheriting from the Class    
                 BaseEnvironment.

        Returns
        -------
        new_env: This is the env as is after going through this block. It must be an object of a Class inheriting from the Class    
                 BaseEnvironment.
                 
        This method wraps the environment by selecting at each call of the method step() only the meaningful state and action 
        features.
        
        A dataset is collected from the environment and the meaningful features are obtained.
        """
        
        if(self.original_data_input_to_the_block is None):
            self.data_gen_block_for_env_wrap.pipeline_type = 'offline'
            
            #check the pre_learn:
            is_ok_to_learn = self.data_gen_block_for_env_wrap.pre_learn_check(env=old_env)   
                
            if(not is_ok_to_learn):
                exc_msg = 'There was an error in the \'pre_learn_check\' method of the data generation object needed to extract'\
                          +' the data, which is used for feature selection!'
                self.logger.exception(msg=exc_msg)
                raise RuntimeError(exc_msg)
                
            generated_data = self.data_gen_block_for_env_wrap.learn(env=old_env) 
            generated_data = generated_data.train_data
            
            if(not self.data_gen_block_for_env_wrap.is_learn_successful):
                exc_msg = 'There was an error in the \'learn\' method of the data generation object needed to extract'\
                          +' the data, which is used for feature selection!'
                self.logger.exception(msg=exc_msg)
                raise RuntimeError(exc_msg)
                                                
            if(not isinstance(generated_data, TabularDataSet)):            
                exc_msg = 'The \'data_gen_block_for_env_wrap\' must extract an object of a Class inheriting from'\
                          +' the Class \'TabularDataSet\'!'
                self.logger.exception(msg=exc_msg)
                raise RuntimeError(exc_msg)
        else:
            generated_data = self.original_data_input_to_the_block 
        
        out = self._recursive_feature_selection(original_train_data=generated_data)
        list_of_selected_state_features, list_of_selected_action_features = out[0], out[1]
        
        if(list_of_selected_state_features is not None):
            class Wrapper(BaseObservationWrapper):
                def __init__(self, env, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                             job_type='process'):
                    super().__init__(env=env, obj_name=obj_name, seeder=seeder, log_mode=log_mode,
                                     checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, 
                                     job_type=job_type)
                    
                    old_observation_space = self.observation_space
                    
                    #this block can only work on Box observation spaces:
                    new_low = copy.deepcopy(old_observation_space.low)[np.array(list_of_selected_state_features)] 
                    new_high = copy.deepcopy(old_observation_space.high)[np.array(list_of_selected_state_features)] 
                    
                    #set the new observation space:
                    self.observation_space = Box(low=new_low, high=new_high)
                    
                def observation(self, observation):
                    new_obs = observation[np.array(list_of_selected_state_features)] 
                    
                    return new_obs
            
            new_env = Wrapper(env=old_env, obj_name='wrapped_env_feature_selection', seeder=old_env.seeder, 
                              log_mode=old_env.log_mode, checkpoint_log_path=old_env.checkpoint_log_path, 
                              verbosity=old_env.verbosity, n_jobs=old_env.n_jobs, job_type=old_env.job_type)  
            
            if(isinstance(old_env.action_space, Box) and list_of_selected_action_features is not None):
                class Wrapper(BaseActionWrapper):
                    def __init__(self, env, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, 
                                 n_jobs=1, job_type='process'):
                        super().__init__(env=env, obj_name=obj_name, seeder=seeder, log_mode=log_mode,
                                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, 
                                         job_type=job_type)
                    
                        old_action_space = self.action_space
                        
                        #this block can only work on Box action spaces:
                        new_low = copy.deepcopy(old_action_space.low)[np.array(list_of_selected_action_features)] 
                        new_high = copy.deepcopy(old_action_space.high)[np.array(list_of_selected_action_features)] 
                        
                        #set the new action space:
                        self.action_space = Box(low=new_low, high=new_high)
                    
                    def action(self, action):
                        new_act = action[np.array(list_of_selected_action_features)] 
                        
                        return new_act
            
                new_env = Wrapper(env=copy.deepcopy(new_env), obj_name='wrapped_env_feature_selection', seeder=old_env.seeder, 
                                  log_mode=old_env.log_mode, checkpoint_log_path=old_env.checkpoint_log_path, 
                                  verbosity=old_env.verbosity, n_jobs=old_env.n_jobs, job_type=old_env.job_type)  
                    
            return new_env
        else:
            wrn_msg = 'The recursive feature selection failed: no states were selected. Returning input \'env\' as is.' 
            self.logger.warning(msg=wrn_msg)
            
            return old_env
        
    def _feature_engineer_data(self, old_data):
        """
        Parameters
        ----------
        old_data: This is the train_data as is before entering this block. It must be an object of a Class inheriting from the 
                  Class TabularDataSet.

        Returns
        -------
        new_data: This is the train_data as is after going through this block. It must be an object of a Class inheriting from 
                  the Class TabularDataSet.
                  
        This method creates a new dataset by selecting only the meaningful state and action features.
        """
        
        new_data = copy.deepcopy(old_data)
                
        out = self._recursive_feature_selection(original_train_data=old_data)
        list_of_selected_state_features, list_of_selected_action_features = out[0], out[1]
        
        if(list_of_selected_state_features is not None):
            #this block can only work on Box observation spaces:
            new_low = copy.deepcopy(old_data.observation_space.low)[np.array(list_of_selected_state_features)] 
            new_high = copy.deepcopy(old_data.observation_space.high)[np.array(list_of_selected_state_features)] 
            new_obs_space = Box(new_low, new_high)
            
            new_current_states = []
            old_states = old_data.get_states()
            for i in range(len(old_states)):
                new_current_states.append(old_states[i][np.array(list_of_selected_state_features)])
            new_current_states = np.array(new_current_states)
                
            new_next_states = []
            old_next_states = old_data.get_next_states()
            for i in range(len(old_next_states)):
                new_next_states.append(old_next_states[i][np.array(list_of_selected_state_features)])
            new_next_states = np.array(new_next_states)
                 
            if(isinstance(old_data.action_space, Box) and list_of_selected_action_features is not None):
                new_low = copy.deepcopy(old_data.action_space.low)[np.array(list_of_selected_action_features)] 
                new_high = copy.deepcopy(old_data.action_space.high)[np.array(list_of_selected_action_features)] 
                new_act_space = Box(new_low, new_high)       
                
                new_actions = []
                old_actions = old_data.get_actions()
                for i in range(len(old_actions)):
                    new_actions.append(old_actions[i][np.array(list_of_selected_action_features)])
                new_actions = np.array(new_actions)
            else:
                #if the action_space is Discrete it only has a feature and so i do not perform feature selection on the actions:
                new_act_space = copy.deepcopy(old_data.action_space)
                new_actions = copy.deepcopy(old_data.get_actions())
                          
            new_data.observation_space = new_obs_space
            new_data.action_space = new_act_space
            
            new_data.dataset = new_data.arrays_as_data(states=new_current_states, actions=new_actions, 
                                                       rewards=old_data.get_rewards(), next_states=new_next_states, 
                                                       absorbings=old_data.get_absorbing(), 
                                                       lasts=old_data.get_episode_terminals())
            new_data.tuples_to_lists()
            
            return new_data
        else:
            wrn_msg = 'The recursive feature selection failed: no states were selected. Returning input \'train_data\' as is.' 
            self.logger.warning(msg=wrn_msg)
            
            return old_data
    
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
        outputFE: This is an object of Class BlockOutput containing as train_data the new train_data and as env the new env: by 
                  new we mean that it utilise the new state-action representation containing only the selected meaningful 
                  features.
                  
                  If the call to the method learn() implemented in the Class FeatureEngineering was not successful the object of
                  Class BlockOutput is empty.
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_data_env = super().learn(train_data=train_data, env=env)      
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_data_env, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        new_train_data = None
        new_env = None
        
        #this block only works on objects of Class TabularDataSet
        if((starting_data_env[0] is not None) and (not isinstance(starting_data_env[0], TabularDataSet))):
            self.is_learn_successful = False 
            self.logger.error(msg='The \'train_data\' must be an object of a Class inheriting from Class \'TabularDataSet\'!')
            return BlockOutput(obj_name=self.obj_name)
        
        #starting_data_env is a list of two: train_data and env. In starting_data_env[0] we have the train_data and in 
        #starting_data_env[1] we have the env. On these we call the methods feature_engineer_data and feature_engineer_env
        #only in case they are not None:
        if(starting_data_env[0] is not None):
            if(starting_data_env[0].observation_space.shape[0] == 1):
                wrn_msg = 'The observation space has only one dimension: no feature selection is possible. Returning the'\
                          +' original train_data!'
                self.logger.warning(msg=wrn_msg)
                new_train_data = starting_data_env[0]
            else:
                new_train_data = self._feature_engineer_data(old_data=starting_data_env[0])            
        if(starting_data_env[1] is not None):
            if(starting_data_env[1].observation_space.shape[0] == 1):
                wrn_msg = 'The observation space has only one dimension: no feature selection is possible. Returning the'\
                          +' original env!'
                self.logger.warning(msg=wrn_msg)
                new_env = starting_data_env[1]
            else:
                if(starting_data_env[0] is None):
                    self.original_data_input_to_the_block = None
                else:
                    wrn_msg = 'The \'train_data\' is not \'None\': using the provided \'train_data\' instead of extracting'\
                              +' a new dataset with \'data_gen_block_for_env_wrap\'!'
                    self.logger.warning(msg=wrn_msg)
                    self.original_data_input_to_the_block = starting_data_env[0]
                
                new_env = self._feature_engineer_env(old_env=starting_data_env[1])
        
        outputFE = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                               checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity, train_data=new_train_data, 
                               env=new_env)
            
        self.is_learn_successful = True
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object learnt successfully!')
        return outputFE

    def analyse(self):
        """
        This method is yet to be implemented.
        """

        raise NotImplementedError
    
    
class FeatureEngineeringFSCMI(FeatureEngineering):
    """
    This Class implements a specific feature engineering algorithm: it performs forward feature selection of the observation 
    space using as metric the mutual information as proposed in Feature Selection via Mutual Information: New Theoretical 
    Insights. 
    
    cf. https://arxiv.org/abs/1907.07384
    
    The implementation of this block is based on the implementation associated with the cited paper.
    
    This can be applied only to continuous spaces: it has no effect on discrete spaces.
    """
          
    def __init__(self, eval_metric, obj_name, seeder=2, algo_params=None, data_gen_block_for_env_wrap=None, log_mode='console', 
                  checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):
        """        
        Parameters
        ----------
        algo_params: This contains the parameters of the algorithm implemented in this class. 
                     
                     The default is None.
                     
                     If None the following will be used:
                     'threshold': 0
                     'k': 5
                     
        data_gen_block_for_env_wrap: This must be an object of a Class inheriting from the Class DataGeneration. This is used to 
                                     extract a dataset from the environment and this dataset will be used for fitting the feature
                                     engineering algorithm. This is needed because otherwise we would fit and transform a single
                                     sample from the environment at the time and thus we would have different features across
                                     different samples from the environment.
                                     
                                     This must extract an object of a Class inheriting from the Class TabularDataSet.
                                     
                                     This is only used if the method learn() does receives an env but does not receive train_data.
                                     
                                     The default is None.
                                     
                                     If None a DataGenerationRandomUniformPolicy with 'n_samples': 100000 will be used.
                                         
        Non-Parameters Members
        ----------------------                                     
        algo_params_upon_instantiation: This a copy of the original value of algo_params, namely the value of
                                        algo_params that the object got upon creation. This is needed for re-loading
                                        objects.
       
        The other parameters and non-parameters members are described in the Class Block.
        """
                       
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                          checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.works_on_online_rl = True
        self.works_on_offline_rl = True
        self.works_on_box_action_space = True
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = False
        
        #this block has parameters and I may want to tune them:
        self.is_parametrised = True
        
        self.algo_params = algo_params
       
        self.data_gen_block_for_env_wrap = data_gen_block_for_env_wrap

        if(self.data_gen_block_for_env_wrap is None):
            #extract data from the env using a DataGeneration block. This is needed because I need to have the same 
            #transformation for the entire environment, and the only way to do so is to fit the transformation on the same 
            #dataset. Otherwise for each call of the step method i would fit a different transformation, and so the operation 
            #would not be consistent among different calls of the step method of the same environment.
            data_gen_params = dict(eval_metric=SomeSpecificMetric(obj_name='place_holder_metric'), 
                                   obj_name='data_gen_for_wrappin_env', seeder=self.seeder, 
                                   algo_params={'n_samples': Integer(hp_name='n_samples', obj_name='n_samples_data_gen', 
                                                current_actual_value=100000)}, 
                                   log_mode=self.log_mode, 
                                   checkpoint_log_path=self.checkpoint_log_path, verbosity=0, n_jobs=self.n_jobs, 
                                   job_type=self.job_type)
            
            self.data_gen_block_for_env_wrap = DataGenerationRandomUniformPolicy(**data_gen_params)
            
        if(self.algo_params is None):
            threshold = Real(hp_name='threshold', obj_name='threshold', current_actual_value=0, range_of_values=[0,1], 
                              to_mutate=True) 
            k = Integer(hp_name='k', obj_name='k', current_actual_value=5, range_of_values=[1,50], to_mutate=True) 
            
            self.algo_params = {'threshold': threshold, 'k': k}
                    
        self.algo_params_upon_instantiation = copy.deepcopy(self.algo_params)

    def _MIEstimateMixed(self, X, Y):
        """ 
        MI Estimator based on Mixed Random Variable Mutual Information Estimator - Gao et al.
        """
        
        k = self.algo_params['k'].current_actual_value
        
        nSamples = len(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        dataset = np.concatenate((X, Y), axis=1) #concatenate Y to X as a column
        
        #kdtree to quickly find the K-NN
        tree_xy = cKDTree(dataset)
        tree_x = cKDTree(X)
        tree_y = cKDTree(Y)
        
        # rho
        knn_dist = [tree_xy.query(sample, k + 1, p=float('inf'))[0][k] for sample in dataset]
        
        res = 0
        for i in range(nSamples):
            k_hat, n_xi, n_yi = k, k, k
            if knn_dist[i] <= 1e-15:
                #points at distance less than or equal to zero (almost)
                k_hat = len(tree_xy.query_ball_point(dataset[i], 1e-15, p=float('inf')))
                n_xi = len(tree_x.query_ball_point(X[i], 1e-15, p=float('inf')))
                n_yi = len(tree_y.query_ball_point(Y[i], 1e-15, p=float('inf')))
            else:
                k_hat = k
                
                #points at distance less than or equal to rho
                n_xi = len(tree_x.query_ball_point(X[i], knn_dist[i] - 1e-15, p=float('inf')))
                n_yi = len(tree_y.query_ball_point(Y[i], knn_dist[i] - 1e-15, p=float('inf')))
            
            res += (digamma(k_hat) + np.log(nSamples) - digamma(n_xi) - digamma(n_yi)) / nSamples
            
        return res
    
    def _CMIEstimateMixed(self, X, Y, Z):
        """ 
        I(X;Y|Z) = I(X,Z; Y) - I(Z; Y) 
        """
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        XZ = np.hstack((X, Z))  
        
        i1 = self._MIEstimateMixed(X=XZ, Y=Y)
        i2 = self._MIEstimateMixed(X=Z, Y=Y)
      
        return i1 - i2
        
    def _mixed_mutual_info_forward_fs(self, features, target):
        """ 
        The function order feature importance in forward way features is an m*n matrix (m-samples; n-features), target can 
        either be a one dim array or a matrix.
        """
    
        remaining_features, target = np.array(features), np.array(target)
        selected_features = np.zeros(remaining_features.shape)
    
        sorted_ids, sorted_scores = [], []
        
        mi_values = np.array(Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)(
                                      delayed(self._MIEstimateMixed)(remaining_features[:, ii], target) 
                                              for ii in range(remaining_features.shape[1])))
                
        selected_features[:, 0] = remaining_features[:,np.argmax(mi_values)]

        ids_aux = list(range(features.shape[1]))

        remaining_features = np.delete(remaining_features, np.argmax(mi_values), axis=1)
        
        sorted_ids.append(ids_aux.pop(np.argmax(mi_values)))
        sorted_scores.append(np.max(mi_values))
    
        for ii in range(1, selected_features.shape[1]):
            cmi_values = np.array(Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)(
                                           delayed(self._CMIEstimateMixed)(remaining_features[:,jj], target, 
                                                                           selected_features[:,:ii]) 
                                                   for jj in range(remaining_features.shape[1])))
           
            if(max(cmi_values) < self.algo_params['threshold'].current_actual_value):
                break
            
            selected_features[:,ii] = remaining_features[:,np.argmax(cmi_values)]
            remaining_features = np.delete(remaining_features, np.argmax(cmi_values), axis=1)  
      
            sorted_ids.append(ids_aux.pop(np.argmax(cmi_values)))            
            sorted_scores.append(np.max(cmi_values))
            
        self.logger.info(msg='Sorted features ids: '+str(sorted_ids))
        self.logger.info(msg='Sorted features scores: '+str(sorted_scores))
        
        self.feature_importance_scores = sorted_scores 
        self.ordered_features = sorted_ids
        
        return sorted_ids
    
    def _feature_engineer_env(self, old_env):
        """
        Parameters
        ----------
        old_env: This is the env as is before entering this block. It must be an object of a Class inheriting from the Class    
                 BaseEnvironment.

        Returns
        -------
        new_env: This is the env as is after going through this block. It must be an object of a Class inheriting from the Class    
                 BaseEnvironment.
                 
        This method wraps the environment by selecting at each call of the method step() only the meaningful state features.
        
        A dataset is collected from the environment and the meaningful features are obtained.
        """
        
        if(self.original_data_input_to_the_block is None):
            self.data_gen_block_for_env_wrap.pipeline_type = 'offline'
            
            #check the pre_learn:
            is_ok_to_learn = self.data_gen_block_for_env_wrap.pre_learn_check(env=old_env)   
                
            if(not is_ok_to_learn):
                exc_msg = 'There was an error in the \'pre_learn_check\' method of the data generation object needed to extract'\
                          +' the data, which is used for feature selection!'
                self.logger.exception(msg=exc_msg)
                raise RuntimeError(exc_msg)
                
            generated_data = self.data_gen_block_for_env_wrap.learn(env=old_env) 
            generated_data = generated_data.train_data
            
            if(not self.data_gen_block_for_env_wrap.is_learn_successful):
                exc_msg = 'There was an error in the \'learn\' method of the data generation object needed to extract'\
                          +' the data, which is used for feature selection!'
                self.logger.exception(msg=exc_msg)
                raise RuntimeError(exc_msg)
                                                
            if(not isinstance(generated_data, TabularDataSet)):            
                exc_msg = 'The \'data_gen_block_for_env_wrap\' must extract an object of a Class inheriting from'\
                          +' the Class \'TabularDataSet\'!'
                self.logger.exception(msg=exc_msg)
                raise RuntimeError(exc_msg)
        else:
            generated_data = self.original_data_input_to_the_block 
        
        parsed = generated_data.parse_data()
        
        target = np.hstack((parsed[2].reshape(-1,1),parsed[3]))

        list_of_selected_state_features = self._mixed_mutual_info_forward_fs(features=parsed[0], target=target)
                
        if(list_of_selected_state_features is not None):
            class Wrapper(BaseObservationWrapper):
                def __init__(self, env, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                             job_type='process'):
                    super().__init__(env=env, obj_name=obj_name, seeder=seeder, log_mode=log_mode,
                                     checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, 
                                     job_type=job_type)
                    
                    old_observation_space = self.observation_space
                    
                    #this block can only work on Box observation spaces:
                    new_low = copy.deepcopy(old_observation_space.low)[np.array(list_of_selected_state_features)] 
                    new_high = copy.deepcopy(old_observation_space.high)[np.array(list_of_selected_state_features)] 
                    
                    #set the new observation space:
                    self.observation_space = Box(low=new_low, high=new_high)
                    
                def observation(self, observation):
                    new_obs = observation[np.array(list_of_selected_state_features)] 
                    
                    return new_obs
            
            new_env = Wrapper(env=old_env, obj_name='wrapped_env_feature_selection', seeder=old_env.seeder, 
                              log_mode=old_env.log_mode, checkpoint_log_path=old_env.checkpoint_log_path, 
                              verbosity=old_env.verbosity, n_jobs=old_env.n_jobs, job_type=old_env.job_type)  
                    
            return new_env
        else:
            wrn_msg = 'The forward feature selection failed: no states were selected. Returning input \'env\' as is.' 
            self.logger.warning(msg=wrn_msg)
            
            return old_env
        
    def _feature_engineer_data(self, old_data):      
        new_data = copy.deepcopy(old_data)
        
        parsed = old_data.parse_data()
        
        target = np.hstack((parsed[2].reshape(-1,1),parsed[3]))

        list_of_selected_state_features = self._mixed_mutual_info_forward_fs(features=parsed[0], target=target)

        if(list_of_selected_state_features is not None):
            #this block can only work on Box observation spaces:
            new_low = copy.deepcopy(old_data.observation_space.low)[np.array(list_of_selected_state_features)] 
            new_high = copy.deepcopy(old_data.observation_space.high)[np.array(list_of_selected_state_features)] 
            new_obs_space = Box(new_low, new_high)
            
            new_current_states = []
            old_states = old_data.get_states()
            # for i in range(len(old_states)):
            #     new_current_states.append(old_states[i][np.array(list_of_selected_state_features)])
            # new_current_states = np.array(new_current_states)
            new_current_states = old_states[:,np.array(list_of_selected_state_features)]    
            
            new_next_states = []
            old_next_states = old_data.get_next_states()
            for i in range(len(old_next_states)):
                new_next_states.append(old_next_states[i][np.array(list_of_selected_state_features)])
            new_next_states = np.array(new_next_states)
             
            new_act_space = copy.deepcopy(old_data.action_space)
            new_actions = copy.deepcopy(old_data.get_actions())
                          
            new_data.observation_space = new_obs_space
            new_data.action_space = new_act_space
            
            new_data.dataset = new_data.arrays_as_data(states=new_current_states, actions=new_actions, 
                                                        rewards=old_data.get_rewards(), next_states=new_next_states, 
                                                        absorbings=old_data.get_absorbing(), 
                                                        lasts=old_data.get_episode_terminals())
            new_data.tuples_to_lists()
            
            return new_data
        else:
            wrn_msg = 'The forward feature selection failed: no states were selected. Returning input \'train_data\' as is.' 
            self.logger.warning(msg=wrn_msg)
            
            return old_data
    
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
        outputFE: This is an object of Class BlockOutput containing as train_data the new train_data and as env the new env: by 
                  new we mean that it utilise the new state-action representation containing only the selected meaningful 
                  features.
                  
                  If the call to the method learn() implemented in the Class FeatureEngineering was not successful the object of
                  Class BlockOutput is empty.
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_data_env = super().learn(train_data=train_data, env=env)      
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_data_env, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        new_train_data = None
        new_env = None
        
        #this block only works on objects of Class TabularDataSet
        if((starting_data_env[0] is not None) and (not isinstance(starting_data_env[0], TabularDataSet))):
            self.is_learn_successful = False 
            self.logger.error(msg='The \'train_data\' must be an object of a Class inheriting from Class \'TabularDataSet\'!')
            return BlockOutput(obj_name=self.obj_name)
        
        #starting_data_env is a list of two: train_data and env. In starting_data_env[0] we have the train_data and in 
        #starting_data_env[1] we have the env. On these we call the methods feature_engineer_data and feature_engineer_env
        #only in case they are not None:
        if(starting_data_env[0] is not None):
            if(starting_data_env[0].observation_space.shape[0] == 1):
                wrn_msg = 'The observation space has only one dimension: no feature selection is possible. Returning the'\
                          +' original train_data!'
                self.logger.warning(msg=wrn_msg)
                new_train_data = starting_data_env[0]
            else:
                new_train_data = self._feature_engineer_data(old_data=starting_data_env[0])    
                
        if(starting_data_env[1] is not None):
            if(starting_data_env[1].observation_space.shape[0] == 1):
                wrn_msg = 'The observation space has only one dimension: no feature selection is possible. Returning the'\
                          +' original env!'
                self.logger.warning(msg=wrn_msg)
                new_env = starting_data_env[1]
            else:
                if(starting_data_env[0] is None):
                    self.original_data_input_to_the_block = None
                else:
                    wrn_msg = 'The \'train_data\' is not \'None\': using the provided \'train_data\' instead of extracting'\
                              +' a new dataset with \'data_gen_block_for_env_wrap\'!'
                    self.logger.warning(msg=wrn_msg)
                    self.original_data_input_to_the_block = starting_data_env[0]
                
                new_env = self._feature_engineer_env(old_env=starting_data_env[1])
        
        outputFE = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                               checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity, train_data=new_train_data, 
                               env=new_env)
            
        self.is_learn_successful = True
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object learnt successfully!')
        return outputFE

    def analyse(self):
        """
        This method is yet to be implemented.
        """

        raise NotImplementedError
    
    
class FeatureEngineeringNystroemMap(FeatureEngineering):
    """
    This Class implements a specific feature engineering algorithm: it constructs an approximate feature map for an arbitrary 
    kernel using a subset of the data as basis. 
    
    cf. https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html
    
    This can be applied only to continuous observation spaces.
    """
    
    def __init__(self, eval_metric, obj_name, seeder=2, algo_params=None, data_gen_block_for_env_wrap=None, log_mode='console', 
                 checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):
        """        
        Parameters
        ----------
        algo_params: This contains the parameters of the algorithm implemented in this class. 
                      
                     The default is None.
                     
                     If None the following will be used:
                     'kernel': 'rbf'
                     'gamma': 0.1
                     'n_components': 100
        
        data_gen_block_for_env_wrap: This must be an object of a Class inheriting from the Class DataGeneration. This is used to 
                                     extract a dataset from the environment and this dataset will be used for fitting the feature
                                     engineering algorithm. This is needed because otherwise we would fit and transform a single
                                     sample from the environment at the time and thus we would have different features across
                                     different samples from the environment.
                                     
                                     This must extract an object of a Class inheriting from the Class TabularDataSet.

                                     This is only used if the method learn() does receives an env but does not receive train_data.

                                     The default is None.
                                     
                                     If None a DataGenerationRandomUniformPolicy with 'n_samples': 100000 will be used.
        
        Non-Parameters Members
        ----------------------        
        algo_object: This is the object containing the actual feature engineering algorithm.
                             
        algo_params_upon_instantiation: This a copy of the original value of algo_params, namely the value of
                                        algo_params that the object got upon creation. This is needed for re-loading
                                        objects.
       
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.works_on_online_rl = True
        self.works_on_offline_rl = True
        self.works_on_box_action_space = True
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = False
        
        #this block has parameters and I may want to tune them:
        self.is_parametrised = True
                
        self.algo_params = algo_params
        
        self.data_gen_block_for_env_wrap = data_gen_block_for_env_wrap
        
        if(self.data_gen_block_for_env_wrap is None):
            #extract data from the env using a DataGeneration block. This is needed because I need to have the same 
            #transformation for the entire environment, and the only way to do so is to fit the transformation on the same 
            #dataset. Otherwise for each call of the step method i would fit a different transformation, and so the operation 
            #would not be consistent among different calls of the step method of the same environment.
            data_gen_params = dict(eval_metric=SomeSpecificMetric(obj_name='place_holder_metric'), 
                                   obj_name='data_gen_for_wrappin_env', seeder=self.seeder, 
                                   algo_params={'n_samples': Integer(hp_name='n_samples', obj_name='n_samples_data_gen', 
                                                current_actual_value=100000)},
                                   log_mode=self.log_mode, 
                                   checkpoint_log_path=self.checkpoint_log_path, verbosity=0, n_jobs=self.n_jobs, 
                                   job_type=self.job_type)
            
            self.data_gen_block_for_env_wrap = DataGenerationRandomUniformPolicy(**data_gen_params)
        
        if(self.algo_params is None):
            kernel = Categorical(hp_name='kernel', current_actual_value='rbf', possible_values=['additive_chi2', 'chi2',
                                                                                                'linear', 'poly', 'polynomial', 
                                                                                                'rbf', 'laplacian', 'sigmoid', 
                                                                                                'cosine'], 
                                 to_mutate=True, seeder=self.seeder, obj_name='nystroem_kernel', log_mode=self.log_mode, 
                                 checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            gamma = Real(hp_name='gamma', current_actual_value=0.1, range_of_values=[0.001,1], to_mutate=True, 
                         seeder=self.seeder, obj_name='nystroem_gamma', log_mode=self.log_mode, 
                         checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            n_components = Integer(hp_name='n_components', current_actual_value=100, range_of_values=[5,150], to_mutate=True, 
                                   seeder=self.seeder, obj_name='nystroem_n_components', log_mode=self.log_mode, 
                                   checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            random_state = Integer(hp_name='random_state', current_actual_value=self.seeder, to_mutate=False, 
                                   seeder=self.seeder, obj_name='nystroem_random_state', log_mode=self.log_mode, 
                                   checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)

            n_of_jobs = Integer(hp_name='n_jobs', current_actual_value=self.n_jobs, to_mutate=False, 
                                seeder=self.seeder, obj_name='nystroem_n_jobs', log_mode=self.log_mode, 
                                 checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            self.algo_params = {'kernel': kernel, 'gamma': gamma, 'random_state': random_state, 'n_components': n_components,
                                'n_jobs': n_of_jobs}
            
        #create a dictionary with values and not objects of Class HyperParameter:
        dict_of_values = self._select_current_actual_value_from_hp_classes(params_objects_dict=self.algo_params)
            
        self.algo_object = Nystroem(**dict_of_values)
                                    
        self.algo_params_upon_instantiation = copy.deepcopy(self.algo_params)
        
    def __repr__(self):
       return str(self.__class__.__name__)+'('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
              +', seeder='+ str(self.seeder)+', local_prng='+ str(self.local_prng)+', algo_params='+str(self.algo_params)\
              +', data_gen_block_for_env_wrap='+ str(self.data_gen_block_for_env_wrap)+', log_mode='+str(self.log_mode)\
              +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
              +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
              +', works_on_online_rl='+str(self.works_on_online_rl)+', works_on_offline_rl='+str(self.works_on_offline_rl)\
              +', works_on_box_action_space='+str(self.works_on_box_action_space)\
              +', works_on_discrete_action_space='+str(self.works_on_discrete_action_space)\
              +', works_on_box_observation_space='+str(self.works_on_box_observation_space)\
              +', works_on_discrete_observation_space='+str(self.works_on_discrete_observation_space)\
              +', pipeline_type='+str(self.pipeline_type)+', is_learn_successful='+str(self.is_learn_successful)\
              +', is_parametrised='+str(self.is_parametrised)+', block_eval='+str(self.block_eval)\
              +', algo_params_upon_instantiation='+str(self.algo_params_upon_instantiation)+', logger='+str(self.logger)+')'
                                     
    def _feature_engineer_env(self, old_env):
        """
        Parameters
        ----------
        old_env: This is the env as is before entering this block. It must be an object of a Class inheriting from the Class    
                 BaseEnvironment.

        Returns
        -------
        new_env: This is the env as is after going through this block. It must be an object of a Class inheriting from the Class    
                 BaseEnvironment.
                 
        This method wraps the old_env and it creates a new_env. This is achieved by sub-classing the Class BaseObservationWrapper
        of the module environment.py of this library.
        
        The wrapped environment has modified observations and observation_space: a NystroemMap is applied.
        """
        
        new_env = None
        
        #these are needed, else in the Class Wrapper the self will be masked by the self of the Class Wrapper
        transformer = self.algo_object
        algo_params = self.algo_params 
        
        class Wrapper(BaseObservationWrapper):
            def __init__(self, env, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                         job_type='process'):
                super().__init__(env=env, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                                 checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
                
                old_observation_space = self.observation_space

                #if the observation space does not contain +np.inf nor -np.inf I transform it:
                if((np.inf not in old_observation_space.low) and (-np.inf not in old_observation_space.low) and 
                   (np.inf not in old_observation_space.high) and (-np.inf not in old_observation_space.high)):
                    new_low = transformer.transform(old_observation_space.low.reshape(1,-1)).ravel()
                    new_high = transformer.transform(old_observation_space.high.reshape(1,-1)).ravel()
                else:
                    new_low = -np.inf*np.ones(algo_params['n_components'].current_actual_value)
                    new_high = np.inf*np.ones(algo_params['n_components'].current_actual_value)
                
                #set the new observation space:
                self.observation_space = Box(low=new_low, high=new_high)
                
            def observation(self, observation):
                new_obs = None
                new_obs = transformer.transform(observation.reshape(1,-1)).ravel()
                
                if(new_obs is None):
                    exc_msg = 'There was an error applying the block \''+str(self.__class__.__name__)+'\'!'
                    self.logger.exception(msg=exc_msg)
                    raise RuntimeError(exc_msg)
                
                return new_obs
            
        new_env = Wrapper(env=old_env, obj_name='wrapped_env_nystroem_map', seeder=old_env.seeder, log_mode=old_env.log_mode,
                          checkpoint_log_path=old_env.checkpoint_log_path, verbosity=old_env.verbosity, n_jobs=old_env.n_jobs,
                          job_type=old_env.job_type)
        
        return new_env
        
    def _feature_engineer_data(self, old_data):
        """
        Parameters
        ----------
        old_data: This is the train_data as is before entering this block. It must be an object of a Class inheriting from the 
                  Class TabularDataSet.

        Returns
        -------
        new_data: This is the train_data as is after going through this block. It must be an object of a Class inheriting from 
                  the Class TabularDataSet.
                     
        This method wraps the old_data and it creates a new_data. This is achieved by applying to the observation_space contained
        in the TabularDataSet object and to the current states and to the next states contained in the dataset the NystroemMap.
        """
        
        new_data = copy.deepcopy(old_data)
        
        #if the observation space does not contain +np.inf nor -np.inf I transform it:
        if((np.inf not in new_data.observation_space.low) and (-np.inf not in new_data.observation_space.low) and 
           (np.inf not in new_data.observation_space.high) and (-np.inf not in new_data.observation_space.high)):
            #compute new low and high:
            new_low = self.algo_object.transform(old_data.observation_space.low.reshape(1,-1)).ravel()
            new_high = self.algo_object.transform(old_data.observation_space.high.reshape(1,-1)).ravel()            
        else:
            new_low = -np.inf*np.ones(self.algo_params['n_components'].current_actual_value)
            new_high = np.inf*np.ones(self.algo_params['n_components'].current_actual_value)
            
        #set the new observation_space:
        new_data.observation_space = Box(low=new_low, high=new_high)

        old_states = old_data.get_states()
        new_states = self.algo_object.transform(old_states)
        
        old_next_states = old_data.get_next_states()
        new_next_states = self.algo_object.transform(old_next_states)
       
        new_data.dataset = new_data.arrays_as_data(states=new_states, actions=old_data.get_actions(), 
                                                   rewards=old_data.get_rewards(), 
                                                   next_states=new_next_states, absorbings=old_data.get_absorbing(), 
                                                   lasts=old_data.get_episode_terminals())
        new_data.tuples_to_lists()

        return new_data
    
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
        outputFE: This is an object of Class BlockOutput containing as train_data the new train_data and env, obtained after the
                  NystroemMap was applied.
                
                  If the call to the method learn() implemented in the Class FeatureEngineering was not successful the object of
                  Class BlockOutput is empty.
                  
                  In order to use the same transformation for the train_data and for the different steps from the environment a
                  dataset is extracted from the environment and the transformation is learnt on such dataset. The same learnt
                  transformation then is used for wrapping the environment. 
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_data_env = super().learn(train_data=train_data, env=env)      
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_data_env, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        new_train_data = None
        new_env = None
        
        #this block only works on objects of Class TabularDataSet
        if((starting_data_env[0] is not None) and (not isinstance(starting_data_env[0], TabularDataSet))):
            self.is_learn_successful = False 
            self.logger.error(msg='The \'train_data\' must be an object of a Class inheriting from Class \'TabularDataSet\'!')
            return BlockOutput(obj_name=self.obj_name)
        
        #starting_data_env is a list of two: train_data and env. In starting_data_env[0] we have the train_data and in 
        #starting_data_env[1] we have the env. On these we call the methods feature_engineer_data and feature_engineer_env
        #only in case they are not None:
        if(starting_data_env[0] is not None):
            #fit the Nystroem object on the dataset states:
            self.algo_object.fit(starting_data_env[0].get_states())
            
            new_train_data = self._feature_engineer_data(old_data=starting_data_env[0])         

        if(starting_data_env[1] is not None):            
            if(starting_data_env[0] is None):
                #this is needed since data generation blocks can only be inserted in an offline pipeline.
                self.data_gen_block_for_env_wrap.pipeline_type = 'offline'
                
                #check the pre_learn:
                is_ok_to_learn = self.data_gen_block_for_env_wrap.pre_learn_check(env=starting_data_env[1])   
                    
                if(not is_ok_to_learn):
                    exc_msg = 'There was an error in the \'pre_learn_check\' method of the data generation object needed to'\
                              +' extract the data, which is used for fitting a \'FeatureEngineering\' block!'
                    self.logger.exception(msg=exc_msg)
                    raise RuntimeError(exc_msg)
                    
                generated_data = self.data_gen_block_for_env_wrap.learn(env=starting_data_env[1])
                generated_data = generated_data.train_data
                    
                if(not self.data_gen_block_for_env_wrap.is_learn_successful):
                    exc_msg = 'There was an error in the \'learn\' method of the data generation object needed to extract'\
                               +' the data, which is used for fitting a \'FeatureEngineering\' block!'
                    self.logger.exception(msg=exc_msg)
                    raise RuntimeError(exc_msg)
                    
                if(not isinstance(generated_data, TabularDataSet)):            
                    exc_msg = 'The \'data_gen_block_for_env_wrap\' must extract an object of a Class inheriting from'\
                              +' the Class \'TabularDataSet\'!'
                    self.logger.exception(msg=exc_msg)
                    raise RuntimeError(exc_msg)
            else:
                wrn_msg = 'The \'train_data\' is not \'None\': using the provided \'train_data\' instead of extracting a new'\
                          +' dataset with \'data_gen_block_for_env_wrap\'!'
                self.logger.warning(msg=wrn_msg)
                generated_data = starting_data_env[0]
            
            #fit the Nystroem object on the collected dataset states:
            self.algo_object.fit(generated_data.get_states())

            new_env = self._feature_engineer_env(old_env=starting_data_env[1])
        
        outputFE = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                               checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity, train_data=new_train_data, 
                               env=new_env)
            
        self.is_learn_successful = True
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object learnt successfully!')
        return outputFE
    
    def analyse(self):
        """
        This method is yet to be implemented.
        """
        
        raise NotImplementedError    