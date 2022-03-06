"""
This module contains the implementation of the Class ModelGeneration.

The Class ModelGeneration inherits from the Class Block.

The Class ModelGeneration is a Class used to group all Classes that do ModelGeneration: this includes also AutoModelGeneration.
"""

from abc import abstractmethod
import copy

from ARLO.block.block_output import BlockOutput
from ARLO.block.block import Block
from ARLO.policy.policy import BasePolicy


class ModelGeneration(Block):
    """
    This is an abstract Class. It is used as generic base Class for all model generation blocks. Both single blocks and automatic 
    blocks inherit from this Class.
    
    A ModelGeneration Block must return an object of Class BlockOutput with a non-None policy member which itself must be an 
    object of a Class inheirting from the Class BasePolicy.
    """
    
    def __repr__(self):
        return 'ModelGeneration'+'('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
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
    
    def construct_policy(self, policy, regressor_type, approximator=None):
        """
        Parameters
        ----------
        policy: Generally this should be an object of a Class inheriting from the Class mushroom_rl.policy.Policy. 
        
                Alternatively it can also be an object of any Class that exposes the method draw_action() taking as parameter 
                just a single state.
                
        regressor_type: This is the regressor_type used in the policy. It is either: 'action_regressor', 'q_regressor' or 
                        'generic_regressor'
            
        approximator: This is the approximator used in the policy: this may be used when doing batch_evaluation in the 
                      DiscountedReward Class for example. 
                       
                      Generally this should be an object of a Class inheriting from the Class 
                      mushroom_rl.approximators.regressor.Regressor.
                      
                      Alternatively it can also be an object of any Class that exposes the method predict() taking as parameter
                      either a single sample, or multiple samples.
            
                      The default is None.

        Returns
        -------
        An object of Class BasePolicy containing the specified policy, and, if specified, the approximator.
        """
        
        if(policy is None):
            exc_msg = 'In the \'construct_policy\' method the \'policy\' is \'None\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
       
        if(regressor_type is None):
            exc_msg = 'In the \'construct_policy\' method the \'regressor_type\' is \'None\'!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
            
        return BasePolicy(policy=copy.deepcopy(policy), regressor_type=regressor_type, approximator=copy.deepcopy(approximator), 
                          obj_name=str(self.obj_name)+'_policy', seeder=self.seeder, log_mode=self.log_mode, 
                          checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
        
    def _walk_dict_to_select_current_actual_value(self, dict_of_hyperparams):
        """
        Parameters
        ----------
        dict_of_hyperparams: This is a dictionary containing objects of a Class inheriting from the Class HyperParameter. This 
                             can be a structured dictionary.
        
        Returns
        -------
        dict_of_hyperparams: This is a dictionary containing the corresponding current_actual_value of the objects of a Class 
                             inheriting from the Class HyperParameter, that were in the original dictionary.
        """
        
        for key in dict_of_hyperparams:
            if isinstance(dict_of_hyperparams[key], dict):
                self._walk_dict_to_select_current_actual_value(dict_of_hyperparams=dict_of_hyperparams[key])
            else:
                dict_of_hyperparams.update({key: dict_of_hyperparams[key].current_actual_value})
        
        return dict_of_hyperparams
                
    def _select_current_actual_value_from_hp_classes(self, params_structured_dict):
        """
        Parameters
        ----------
        params_structured_dict: This is a dictionary containing objects of a Class inheriting from the Class HyperParameter. 
                                This can be a structured dictionary.
            
        Returns
        -------
        algo_params_values: This is a dictionary containing the corresponding current_actual_value of the objects of a Class 
                            inheriting from the Class HyperParameter, that were in the original dictionary.
        """
        
        #this method is called before creating the actual agent object. This method makes sure to pass to the actual agent object 
        #numerical values and not objects of the Class Hyperparameters
        
        #deep copy the parameters: I need the dictionary with the objects of Class Hyperparameters for the Tuner algorithms.  
        copy_params_structured_dict = copy.deepcopy(params_structured_dict)
        algo_params_values = self._walk_dict_to_select_current_actual_value(dict_of_hyperparams=copy_params_structured_dict)
        
        return algo_params_values
                                                      
    @abstractmethod
    def full_block_instantiation(self, info_MDP):
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
                                 in the Class Block was successful. It also checks that the regressor_type has a meaningful value.
        """
        
        #if this is a model generation block i need to reset self.fully_instantiated to False:
        self.fully_instantiated = False
                
        #i need to reload the params to the value it was originally
        self.algo_params = self.algo_params_upon_instantiation
        
        pre_learn_check_outcome = super().pre_learn_check(train_data=train_data, env=env)
        
        if(self.regressor_type not in ['action_regressor', 'q_regressor', 'generic_regressor']):
            err_msg = 'The \'regressor_type\' must be one of: \'action_regressor\', \'q_regressor\' or \'generic_regressor\'!'
            self.logger.error(msg=err_msg)
            
            return False
        
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

        #each time i re-learn the block i need to set is_learn_successful to False
        tmp_out = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and 
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(tmp_out, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        #i can proceed in calling the learn method of the model generation block only if the block was fully instantiated:
        if(not self.fully_instantiated):
            self.is_learn_successful = False
            self.logger.error(msg='A model generation block can call its \'learn\' method only when fully instantiated!')
            return BlockOutput(obj_name=self.obj_name)
        
        #ModelGeneration blocks work on train_data for offlineRL and on env for onlineRL:
        if(self.pipeline_type == 'offline'):
            if(train_data is None):
                self.is_learn_successful = False
                self.logger.error(msg='The \'train_data\' must not be \'None\'!')
                return BlockOutput(obj_name=self.obj_name)
        elif(self.pipeline_type == 'online'):
            if(env is None): 
                self.is_learn_successful = False
                self.logger.error(msg='The \'env\' must not be \'None\'!')
                return BlockOutput(obj_name=self.obj_name)
            
        #below i select only the inputs relevant to the current type of block: ModelGeneration blocks work on the train_data and
        #on the env:
        return train_data, env 

    def _walk_dict_to_flatten_it(self, structured_dict, dict_to_fill):   
        """
        Parameters
        ----------
        structured_dict: This is the input structured dictonary: it can contain several nested sub-dictionaries.
        
        dict_to_fill: This is the dictionary passed to every recursive call to this method.
        
        Returns
        -------
        dict_to_fill: This is the final flat dictionary: it does not contain sub-dictionaries.
        """
        
        for key in structured_dict:
            if isinstance(structured_dict[key], dict):
                self._walk_dict_to_flatten_it(structured_dict=structured_dict[key], dict_to_fill=dict_to_fill)
            else:
                dict_to_fill.update({key: structured_dict[key]})
        
        return dict_to_fill
    
    def get_params(self):
        """
        Returns
        -------
        flat_dict: This is a deep copy of the parameters in the dictionary self.algo_params but they are inserted in a dictionary 
                   that is of only one level (unlike the self.algo_params which is a dictionary nested into a dictionary).
                   This is needed for the Tuner Classes.
        """
        
        #I need a copy of the dict since i need to mutate it afterwards and create a new aget with these parameters.
        #I need to do this because since I want to save the best agent then without deep copying i might mutate the parameters of 
        #the best agent. 
                
        #i need to deep copy since self.algo_params contains objects of class HyperParameter and i need a new copy of these 
        #objects:        
        flat_dict = self._walk_dict_to_flatten_it(structured_dict=copy.deepcopy(self.algo_params), dict_to_fill={})
                
        return flat_dict     
    
    @abstractmethod
    def set_params(self):
        raise NotImplementedError
    
    @abstractmethod
    def analyse(self):
        raise NotImplementedError