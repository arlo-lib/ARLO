"""
This module contains the implementation of the Classes: ModelGenerationMushroomOffline, ModelGenerationMushroomOfflineFQI, 
ModelGenerationMushroomOfflineDoubleFQI and ModelGenerationMushroomOfflineLSPI. The Class ModelGenerationMushroomOffline inherits 
from the Class ModelGeneration. The Classes ModelGenerationMushroomOfflineFQI, ModelGenerationMushroomOfflineDoubleFQI and 
ModelGenerationMushroomOfflineLSPI inherit from the Class ModelGenerationMushroomOffline.
""" 

from abc import abstractmethod

import copy
import numpy as np

from xgboost import XGBRegressor
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.algorithms.value import FQI, DoubleFQI, LSPI

from ARLO.block.block_output import BlockOutput
from ARLO.block.model_generation import ModelGeneration
from ARLO.hyperparameter.hyperparameter import Real, Integer, Categorical


class ModelGenerationMushroomOffline(ModelGeneration):
    """
    This Class is used to contain all the common methods for the offline model generation algorithms that are implemented in
    MushroomRL.
    """
                
    def __repr__(self):
         return str(self.__class__.__name__)+'('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
                +', seeder='+ str(self.seeder)+', local_prng='+ str(self.local_prng)+', model='+str(self.model)\
                +', algo_params='+str(self.algo_params)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
                +', works_on_online_rl='+str(self.works_on_online_rl)+', works_on_offline_rl='+str(self.works_on_offline_rl)\
                +', works_on_box_action_space='+str(self.works_on_box_action_space)\
                +', works_on_discrete_action_space='+str(self.works_on_discrete_action_space)\
                +', works_on_box_observation_space='+str(self.works_on_box_observation_space)\
                +', works_on_discrete_observation_space='+str(self.works_on_discrete_observation_space)\
                +', pipeline_type='+str(self.pipeline_type)+', is_learn_successful='+str(self.is_learn_successful)\
                +', is_parametrised='+str(self.is_parametrised)+', block_eval='+str(self.block_eval)\
                +', algo_params_upon_instantiation='+str(self.algo_params_upon_instantiation)\
                +', logger='+str(self.logger)+', fully_instantiated='+str(self.fully_instantiated)\
                +', info_MDP='+str(self.info_MDP)+', regressor_type='+str(self.regressor_type)+')'  
             
    @abstractmethod
    def full_block_instantiation(self, info_MDP):
        raise NotImplementedError
    
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
        This method calls the method learn() implemented in the Class Block: if such call was successful the method fit() of the 
        algo_object is called onto the dataset contained in the starting_train_data, else an empty object of Class BlockOutput is 
        returned.        
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_train_data_and_env = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_train_data_and_env, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        #since this is an offline block we only have a train_data, which is the first element of the list 
        #starting_train_data_and_env
        starting_train_data = starting_train_data_and_env[0]
        
        #starting_train_data is an object of Class TabularDataSet: so we pick the member dataset to feed to the RL agent fit 
        #method         
        self.algo_object.fit(starting_train_data.dataset)
        
        res = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode,
                          checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity,                        
                          policy=self.construct_policy(policy=self.algo_object.policy, regressor_type=self.regressor_type,
                                                       approximator=self.algo_object.approximator))
                
        self.is_learn_successful = True 
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object learnt successfully!')
        return res
    
    def set_params(self, new_params):
        """
        Parameters
        ----------
        new_params: The new parameters to be used in the specific model generation algorithm. It must be a dictionary that does 
                    not contain any dictionaries(i.e: all parameters must be at the same level).
                                        
                    We need to create the dictionary in the right form for MushroomRL. Then it needs to update self.algo_params. 
                    Then it needs to update the object self.algo_object: to this we need to pass the actual values and not 
                    the Hyperparameter objects. 
                    
                    We call _select_current_actual_value_from_hp_classes: to this method we need to pass the dictionary already 
                    in its final form. 
        Returns
        -------
        bool: This method returns True if new_params is set correctly, and False otherwise.
        """
           
        if(new_params is not None):                
            mdp_info = Categorical(hp_name='mdp_info', obj_name='mdp_info_'+str(self.model.__name__), 
                                   current_actual_value=self.info_MDP)
            
            input_shape = Categorical(hp_name='input_shape', obj_name='input_shape_'+str(self.model.__name__), 
                                      current_actual_value=self.info_MDP.observation_space.shape)
            
            if(self.regressor_type == 'action_regressor'):
                output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                           current_actual_value=(1,))
                n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                         current_actual_value=self.info_MDP.action_space.n)
            elif(self.regressor_type == 'q_regressor'):
                output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                           current_actual_value=(self.info_MDP.action_space.n,))
                n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                         current_actual_value=self.info_MDP.action_space.n)
            elif(self.regressor_type == 'generic_regressor'):
                output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                           current_actual_value=self.info_MDP.action_space.shape)
                #to have a generic regressor I must not specify n_actions
                n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                         current_actual_value=None)
                
            #can set epsilon to [0-0.05] so that we mainly exploit the policy (little random actions). I leave a few random 
            #actions to avoid overfitting the testing procedure:
            policy = Categorical(hp_name='policy', obj_name='policy_'+str(self.model.__name__), 
                                 current_actual_value=EpsGreedy(epsilon=Parameter(value=0)))
                
            tmp_structured_algo_params = {'mdp_info': mdp_info,
                                          'policy': policy,
                                          'approximator_params': {'input_shape': input_shape,
                                                                  'n_actions': n_actions,
                                                                  'output_shape': output_shape
                                                                 }
                                         }

            for tmp_key in list(new_params.keys()):
                #i do not want to change mdp_info or policy
                if(tmp_key in ['approximator', 'n_iterations']):
                    tmp_structured_algo_params.update({tmp_key: new_params[tmp_key]})
                 
                #mdp_info, policy, approximator, n_iterations, input_shape, n_actions, output_shape are set above. 
                #I check also for n_train_samples: n_train_samples must not go to MushroomRL
                if(tmp_key not in ['mdp_info', 'policy', 'approximator', 'n_iterations', 'input_shape', 'n_actions', 
                                   'output_shape', 'n_train_samples']):
                    tmp_structured_algo_params['approximator_params'].update({tmp_key: new_params[tmp_key]})
                        
            structured_dict_of_values = self._select_current_actual_value_from_hp_classes(params_structured_dict=
                                                                                          tmp_structured_algo_params)
                                    
            #i need to un-pack structured_dict_of_values for model (which can be FQI, DoubleFQI) 
            self.algo_object = self.model(**structured_dict_of_values, quiet=True)   

            final_dict_of_params = tmp_structured_algo_params
            final_dict_of_params['n_train_samples'] = new_params['n_train_samples']
            
            self.algo_params = final_dict_of_params
            
            tmp_new_params = self.get_params()
            
            if(tmp_new_params is not None):
                self.algo_params_upon_instantiation = copy.deepcopy(tmp_new_params)
            else:
                self.logger.error(msg='There was an error getting the parameters!')
                return False

            return True
        else:
            self.logger.error(msg='Cannot set parameters: \'new_params\' is \'None\'!')
            return False
    
    def analyse(self):
        """
        This method is not yet implemented.
        """
        
        raise NotImplementedError
        
    def save(self):
        """
        This method saves to a pickle file the object. Before saving it the algo_object is cleared since it can weigh quite a bit.
        """
        
        #clean up the algo_object: this member can possibly make the output file, created when calling the method save, be very 
        #heavy. 
        
        #I need to clean this in a deep copy: otherwise erasing algo_object I cannot call twice in a row the learn method 
        #because the algo_object is set in the method set params
        
        copy_to_save = copy.deepcopy(self)
        
        copy_to_save.algo_object = None
                
        #calls method save() implemented in base Class ModelGeneration of the instance copy_to_save
        super(ModelGenerationMushroomOffline, copy_to_save).save()
    
    
class ModelGenerationMushroomOfflineFQI(ModelGenerationMushroomOffline):
    """
    This Class implements a specific offline model generation algorithm: FQI. This class wraps the Fitted Q-Iteration method 
    implemented in MushroomRL.
    
    cf. https://github.com/MushroomRL/mushroom-rl/blob/dev/mushroom_rl/algorithms/value/batch_td/fqi.py
    
    This Class inherits from the Class ModelGenerationMushroomOffline.
    """
    
    def __init__(self, eval_metric, obj_name, regressor_type='action_regressor', n_train_samples=None, seeder=2, 
                 algo_params=None, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):
        """        
        Parameters
        ----------
        algo_params: This is either None or a dictionary containing all the needed parameters. 
        
                     The default is None.                                                         
                     
                     If None then the following parameters will be used:
                     'policy': EpsGreedy(epsilon=Parameter(value=0)),
                     'approximator': XGBRegressor, 
                     'n_iterations': 10,
                     'input_shape': self.info_MDP.observation_space.shape,
                     'n_actions': self.info_MDP.action_space.n, 
                     'output_shape': (1,),
                     'n_estimators': 300,
                     'subsample': 0.8,
                     'colsample_bytree': 0.3,
                     'colsample_bylevel': 0.7,
                     'learning_rate': 0.08,
                     'verbosity': 0,
                     'random_state': 3,
                     'n_jobs': 1   
               
        regressor_type: This is a string and it can either be: 'action_regressor',  'q_regressor' or 'generic_regressor'. This is
                        used to pick one of the 3 possible kind of regressor made available by MushroomRL.
                        
                        Note that if you want to use a 'q_regressor' then the picked regressor must be able to perform 
                        multi-target regression, as a single regressor is used for all actions.
                        
                        The default is 'action_regressor'.
                    
        n_train_samples: This must be an object of Class Integer (sub-Class of HyperParameter) and it represents a tunable 
                         parameter. This has effect only when using the input loader:
                         LoadDifferentSizeForEachBlock and LoadDifferentSizeForEachBlockAndEnv.
                         
                         For this to work the block must have is_parametrised equal to True, otherwise it does not reach the 
                         tuen method of the Tuner!
                         
                         The default is None.
                         
                         If None then the following will be used:
                         Integer(hp_name='n_train_samples', current_actual_value=10000, range_of_values=[100,1000000], 
                                 to_mutate=True, type_of_mutation='perturbation')
            
        Non-Parameters Members
        ----------------------
        fully_instantiated: This is True if the block is fully instantiated, False otherwise. It is mainly used to make sure that 
                            when we call the learn method the model generation blocks have been fully instantiated as they 
                            undergo two stage initialisation being info_MDP unknown at the beginning of the pipeline.
                            
        info_MDP: This is a dictionary compliant with the parameters needed in input to all MushroomRL model generation 
                  algorithms. It containts the observation space, the action space, the MDP horizon and the MDP gamma.
        
        
        algo_object: This is the object containing the actual model generation algorithm.
                     
        algo_params_upon_instantiation: This a copy of the original value of algo_params, namely the value of algo_params that 
                                        the object got upon creation. This is needed for re-loading objects.
        
        model: This is used in set_params in the generic Class ModelGenerationMushroomOffline. With this member we avoid 
               re-writing for each Class inheriting from the Class ModelGenerationMushroomOffline the set_params method. 
               In this Class this member equals to FQI, which is the Class of MushroomRL implementing FQI.
               
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.works_on_online_rl = False
        self.works_on_offline_rl = True
        self.works_on_box_action_space = False
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = True
        
        self.regressor_type = regressor_type
        
        #this block has parameters and I may want to tune them:
        self.is_parametrised = True
        
        self.algo_params = algo_params
       
        self.fully_instantiated = False
        self.info_MDP = None 
        self.algo_object = None 
        self.algo_params_upon_instantiation = copy.deepcopy(self.algo_params)
        
        #used in the generic Class ModelGenerationMushroomOffline:
        self.model = FQI
        
        #this seeding is needed for the policy of MushroomRL. Indeed the evaluation at the start of the learn method is done 
        #using the policy and in the method draw_action, np.random is called! 
        np.random.seed(self.seeder)
        
        #this is a hyperparameter needed to mutate the number of samples for which to train the agent on:
        self.n_train_samples = n_train_samples
        
        if(self.n_train_samples is None):
            self.n_train_samples = Integer(hp_name='n_train_samples', current_actual_value=10000, range_of_values=[100,1000000], 
                                           to_mutate=True, type_of_mutation='perturbation', seeder=self.seeder, 
                                           obj_name='fqi_n_train_samples', log_mode=self.log_mode,
                                           checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
        
    def full_block_instantiation(self, info_MDP):
        """
        Parameters
        ----------
        info_MDP: This is an object of Class mushroom_rl.environment.MDPInfo. It contains the action and observation spaces, 
                  gamma and the horizon of the MDP.
        
        Returns
        -------
        This method returns True if the algo_params were set successfully, and False otherwise.
        """
        
        self.info_MDP = info_MDP
                                 
        if(self.algo_params is None):
            approximator = Categorical(hp_name='approximator', obj_name='approximator_fqi', 
                                       current_actual_value=XGBRegressor)
            
            n_iter = Integer(hp_name='n_iterations', current_actual_value=10, range_of_values=[10,100], to_mutate=True, 
                             seeder=self.seeder, obj_name='fqi_n_iterations', log_mode=self.log_mode, 
                             checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            n_estim = Integer(hp_name='n_estimators', current_actual_value=300, range_of_values=[100,500], to_mutate=True, 
                              seeder=self.seeder, obj_name='fqi_xgb_n_estimators', log_mode=self.log_mode,
                              checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            sub_samp = Real(hp_name='subsample', current_actual_value=0.8, range_of_values=[0.6,1], to_mutate=True, 
                            seeder=self.seeder, obj_name='fqi_xgb_subsample', log_mode=self.log_mode,
                            checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            cols_by_tree = Real(hp_name='colsample_bytree', current_actual_value=0.3, range_of_values=[0.2,1], to_mutate=True, 
                                seeder=self.seeder, obj_name='fqi_xgb_ccolsample_bytree', log_mode=self.log_mode,
                                checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            cols_by_level = Real(hp_name='colsample_bylevel', current_actual_value=0.7, range_of_values=[0.6,1], to_mutate=True, 
                                 seeder=self.seeder, obj_name='fqi_xgb_colsample_bylevel', log_mode=self.log_mode,
                                 checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            lr_rate = Real(hp_name='learning_rate', current_actual_value=0.08, range_of_values=[0.005,0.2], to_mutate=True, 
                           seeder=self.seeder, obj_name='fqi_xgb_learning_rate', log_mode=self.log_mode,
                           checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            verbosity = Integer(hp_name='verbosity', obj_name='verbosity_xgb_fqi', current_actual_value=0)
            
            random_state = Integer(hp_name='random_state', obj_name='random_state_xgb_fqi', current_actual_value=self.seeder)
            
            n_of_jobs = Integer(hp_name='n_jobs', obj_name='n_jobs_xgb_fqi', current_actual_value=1)
            
            self.algo_params = {'approximator': approximator, 
                                'n_iterations': n_iter,
                                'n_estimators': n_estim,
                                'subsample': sub_samp,
                                'colsample_bytree': cols_by_tree,
                                'colsample_bylevel': cols_by_level,
                                'learning_rate': lr_rate,
                                'verbosity': verbosity,
                                'random_state': random_state,
                                'n_jobs': n_of_jobs                                      
                               }
           
        dict_of_n_train_samples = {'n_train_samples': self.n_train_samples}
        self.algo_params = {**self.algo_params, **dict_of_n_train_samples}
        
        is_set_param_success = self.set_params(new_params=self.algo_params)
            
        if(not is_set_param_success):
            err_msg = 'There was an error setting the parameters of a'+'\''+str(self.__class__.__name__)+'\' object!'
            self.logger.error(msg=err_msg)
            self.fully_instantiated = False
            self.is_learn_successful = False
            return False
           
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object fully instantiated!')
        self.fully_instantiated = True
        return True 
       
        
class ModelGenerationMushroomOfflineDoubleFQI(ModelGenerationMushroomOffline):
    """
    This Class implements a specific offline model generation algorithm: DoubleFQI. This class wraps the 
    Double Fitted Q-Iteration method implemented in MushroomRL.
    
    cf. https://github.com/MushroomRL/mushroom-rl/blob/dev/mushroom_rl/algorithms/value/batch_td/double_fqi.py
    
    This Class inherits from the Class ModelGenerationMushroomOffline.
    """
    
    def __init__(self, eval_metric, obj_name, regressor_type='action_regressor', n_train_samples=None, seeder=2, 
                 algo_params=None, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):
        """        
        Parameters
        ----------
        algo_params: This is either None or a dictionary containing all the needed parameters.
                            
                     The default is None.                                                         
                     
                     If None then the following parameters will be used:
                     'policy': EpsGreedy(epsilon=Parameter(value=0)),
                     'approximator': XGBRegressor,                     
                     'n_iterations': 10,
                     'input_shape': self.info_MDP.observation_space.shape,
                     'n_actions': self.info_MDP.action_space.n, 
                     'output_shape': (1,),
                     'n_estimators': 300,
                     'subsample': 0.8,
                     'colsample_bytree': 0.3,
                     'colsample_bylevel': 0.7,
                     'learning_rate': 0.08,
                     'verbosity': 0,
                     'random_state': 3,
                     'n_jobs': 1   
                     
        regressor_type: This is a string and it can either be: 'action_regressor',  'q_regressor' or 'generic_regressor'. This is
                        used to pick one of the 3 possible kind of regressor made available by MushroomRL.
                        
                        Note that if you want to use a 'q_regressor' then the picked regressor must be able to perform 
                        multi-target regression, as a single regressor is used for all actions. 
                        
                        The default is 'action_regressor'.
                    
        n_train_samples: This must be an object of Class Integer (sub-Class of HyperParameter) and it represents a tunable 
                         parameter. This has effect only when using the input loader:
                         LoadDifferentSizeForEachBlock and LoadDifferentSizeForEachBlockAndEnv.
                         
                         For this to work the block must have is_parametrised equal to True, otherwise it does not reach the 
                         tuen method of the Tuner!
                         
                         The default is None.
                         
                         If None then the following will be used:
                         Integer(hp_name='n_train_samples', current_actual_value=10000, range_of_values=[100,1000000], 
                                 to_mutate=True, type_of_mutation='perturbation')
                         
        Non-Parameters Members
        ----------------------
        fully_instantiated: This is True if the block is fully instantiated, False otherwise. It is mainly used to make sure that 
                            when we call the learn method the model generation blocks have been fully instantiated as they 
                            undergo two stage initialisation being info_MDP unknown at the beginning of the pipeline.
                            
        info_MDP: This is a dictionary compliant with the parameters needed in input to all MushroomRL model generation 
                  algorithms. It containts the observation space, the action space, the MDP horizon and the MDP gamma.
        
        
        algo_object: This is the object containing the actual model generation algorithm.
                     
        algo_params_upon_instantiation: This a copy of the original value of algo_params, namely the value of
                                        algo_params that the object got upon creation. This is needed for re-loading
                                        objects.
                                        
        model: This is used in set_params in the generic Class ModelGenerationMushroomOffline. With this member we avoid 
               re-writing for each Class inheriting from the Class ModelGenerationMushroomOffline the set_params method. 
               In this Class this member equals to DoubleFQI, which is the Class of MushroomRL implementing DoubleFQI.
        
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.works_on_online_rl = False
        self.works_on_offline_rl = True
        self.works_on_box_action_space = False
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = True
        
        self.regressor_type = regressor_type
        
        #this block has parameters and I may want to tune them:
        self.is_parametrised = True
        
        self.algo_params = algo_params
       
        self.fully_instantiated = False
        self.info_MDP = None 
        self.algo_object = None 
        self.algo_params_upon_instantiation = copy.deepcopy(self.algo_params)

        #used in the generic Class ModelGenerationMushroomOffline:
        self.model = DoubleFQI
    
        #this seeding is needed for the policy of MushroomRL. Indeed the evaluation at the start of the learn method is done 
        #using the policy and in the method draw_action, np.random is called! 
        np.random.seed(self.seeder)
        
        #this is a hyperparameter needed to mutate the number of samples for which to train the agent on:
        self.n_train_samples = n_train_samples
        
        if(self.n_train_samples is None):
            self.n_train_samples = Integer(hp_name='n_train_samples', current_actual_value=10000, range_of_values=[100,1000000], 
                                           to_mutate=True, type_of_mutation='perturbation', seeder=self.seeder, 
                                           obj_name='double_fqi_n_train_samples', log_mode=self.log_mode,
                                           checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
        
    def full_block_instantiation(self, info_MDP):
        """
        Parameters
        ----------
        info_MDP: This is an object of Class mushroom_rl.environment.MDPInfo. It contains the action and observation spaces, 
                  gamma and the horizon of the MDP.
        
        Returns
        -------
        This method returns True if the algo_params were set successfully, and False otherwise.
        """
        
        self.info_MDP = info_MDP
                 
        if(self.algo_params is None):
            approximator = Categorical(hp_name='approximator', obj_name='approximator_double_fqi', 
                                       current_actual_value=XGBRegressor)
            
            n_iter = Integer(hp_name='n_iterations', current_actual_value=10, range_of_values=[10,100], to_mutate=True, 
                             seeder=self.seeder, obj_name='double_fqi_n_iterations', log_mode=self.log_mode, 
                             checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            n_estim = Integer(hp_name='n_estimators', current_actual_value=300, range_of_values=[100,500], to_mutate=True, 
                              seeder=self.seeder, obj_name='double_fqi_xgb_n_estimators', log_mode=self.log_mode,
                              checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            sub_samp = Real(hp_name='subsample', current_actual_value=0.8, range_of_values=[0.6,1], to_mutate=True, 
                            seeder=self.seeder, obj_name='double_fqi_xgb_subsample', log_mode=self.log_mode,
                            checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            cols_by_tree = Real(hp_name='colsample_bytree', current_actual_value=0.3, range_of_values=[0.2,1], to_mutate=True, 
                                seeder=self.seeder, obj_name='double_fqi_xgb_ccolsample_bytree', log_mode=self.log_mode,
                                checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            cols_by_level = Real(hp_name='colsample_bylevel', current_actual_value=0.7, range_of_values=[0.6,1], to_mutate=True, 
                                 seeder=self.seeder, obj_name='double_fqi_xgb_colsample_bylevel', log_mode=self.log_mode,
                                 checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            lr_rate = Real(hp_name='learning_rate', current_actual_value=0.08, range_of_values=[0.005,0.2], to_mutate=True, 
                           seeder=self.seeder, obj_name='double_fqi_xgb_learning_rate', log_mode=self.log_mode,
                           checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            verbosity = Integer(hp_name='verbosity', obj_name='verbosity_xgb_double_fqi', current_actual_value=0)
            
            random_state = Integer(hp_name='random_state', obj_name='random_state_xgb_double_fqi', 
                                   current_actual_value=self.seeder)
            
            n_of_jobs = Integer(hp_name='n_jobs', obj_name='n_jobs_xgb_double_fqi', current_actual_value=1)
            
            self.algo_params = {'approximator': approximator, 
                                'n_iterations': n_iter,
                                'n_estimators': n_estim,
                                'subsample': sub_samp,
                                'colsample_bytree': cols_by_tree,
                                'colsample_bylevel': cols_by_level,
                                'learning_rate': lr_rate,
                                'verbosity': verbosity,
                                'random_state': random_state,
                                'n_jobs': n_of_jobs
                               }
            
        dict_of_n_train_samples = {'n_train_samples': self.n_train_samples}
        self.algo_params = {**self.algo_params, **dict_of_n_train_samples}
        
        is_set_param_success = self.set_params(new_params=self.algo_params)
            
        if(not is_set_param_success):
            err_msg = 'There was an error setting the parameters of a'+'\''+str(self.__class__.__name__)+'\' object!'
            self.logger.error(msg=err_msg)
            self.fully_instantiated = False
            self.is_learn_successful = False
            return False
           
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object fully instantiated!')
        self.fully_instantiated = True
        return True 
    

class ModelGenerationMushroomOfflineLSPI(ModelGenerationMushroomOffline):
    """
    This Class implements a specific offline model generation algorithm: LSPI. This class wraps the Least-Squares Policy 
    Iteration method implemented in MushroomRL.
    
    cf. https://github.com/MushroomRL/mushroom-rl/blob/dev/mushroom_rl/algorithms/value/batch_td/lspi.py
    
    This Class inherits from the Class ModelGenerationMushroomOffline.
    """
    
    def __init__(self, eval_metric, obj_name, regressor_type='action_regressor', n_train_samples=None, seeder=2, 
                 algo_params=None, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):
        """        
        Parameters
        ----------
        algo_params: This is either None or a dictionary containing all the needed parameters.
                            
                     The default is None.                                                         
                     
                     If None then the following parameters will be used:
                     'policy': EpsGreedy(epsilon=Parameter(value=0.)),
                     'input_shape': self.info_MDP.observation_space.shape,
                     'n_actions': self.info_MDP.action_space.n, 
                     'output_shape': (1,),
                     'epsilon': 1e-2   
                     
        regressor_type: This is a string and it can either be: 'action_regressor',  'q_regressor' or 'generic_regressor'. This is
                        used to pick one of the 3 possible kind of regressor made available by MushroomRL.
                        
                        Note that if you want to use a 'q_regressor' then the picked regressor must be able to perform 
                        multi-target regression, as a single regressor is used for all actions.
                        
                        The default is 'action_regressor'.
                        
        n_train_samples: This must be an object of Class Integer (sub-Class of HyperParameter) and it represents a tunable 
                         parameter. This has effect only when using the input loader:
                         LoadDifferentSizeForEachBlock and LoadDifferentSizeForEachBlockAndEnv.
                         
                         For this to work the block must have is_parametrised equal to True, otherwise it does not reach the 
                         tuen method of the Tuner!
                         
                         The default is None.
                         
                         If None then the following will be used:
                         Integer(hp_name='n_train_samples', current_actual_value=10000, range_of_values=[100,1000000], 
                                 to_mutate=True, type_of_mutation='perturbation')
                    
        Non-Parameters Members
        ----------------------
        fully_instantiated: This is True if the block is fully instantiated, False otherwise. It is mainly used to make sure that 
                            when we call the learn method the model generation blocks have been fully instantiated as they 
                            undergo two stage initialisation being info_MDP unknown at the beginning of the pipeline.
                            
        info_MDP: This is a dictionary compliant with the parameters needed in input to all MushroomRL model generation 
                  algorithms. It containts the observation space, the action space, the MDP horizon and the MDP gamma.
        
        
        algo_object: This is the object containing the actual model generation algorithm.
                     
        algo_params_upon_instantiation: This a copy of the original value of algo_params, namely the value of
                                        algo_params that the object got upon creation. This is needed for re-loading
                                        objects.
                                        
        model: This is used in set_params in the generic Class ModelGenerationMushroomOffline. With this member we avoid 
               re-writing for each Class inheriting from the Class ModelGenerationMushroomOffline the set_params method. 
               In this Class this member equals to LSPI, which is the Class of MushroomRL implementing LSPI.
                                                
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.works_on_online_rl = False
        self.works_on_offline_rl = True
        self.works_on_box_action_space = False
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = True
        
        self.regressor_type = regressor_type
        
        #this block has parameters and I may want to tune them:
        self.is_parametrised = True
        
        self.algo_params = algo_params
       
        self.fully_instantiated = False
        self.info_MDP = None 
        self.algo_object = None 
        self.algo_params_upon_instantiation = copy.deepcopy(self.algo_params)    
        
        #used in the generic Class ModelGenerationMushroomOffline:
        self.model = LSPI
        
        #this seeding is needed for the policy of MushroomRL. Indeed the evaluation at the start of the learn method is done 
        #using the policy and in the method draw_action, np.random is called! 
        np.random.seed(self.seeder)
        
        #this is a hyperparameter needed to mutate the number of samples for which to train the agent on:
        self.n_train_samples = n_train_samples
        
        if(self.n_train_samples is None):
            self.n_train_samples = Integer(hp_name='n_train_samples', current_actual_value=10000, range_of_values=[100,1000000], 
                                           to_mutate=True, type_of_mutation='perturbation', seeder=self.seeder, 
                                           obj_name='lspi_n_train_samples', log_mode=self.log_mode,
                                           checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
        #if self.n_train_samples should not be mutated then i set is_parametrised to False:
        if(not self.n_train_samples.to_mutate):
            self.is_parametrised = False
            wrn_msg = 'Since \'n_train_samples\' has the member \'to_mutate\' equal to \'False\', I am setting'\
                      +' \'is_parametrised\' equal to \'False\' as there are no hyperparameters left to tune!'
            self.logger.warning(msg=wrn_msg)
             
    def full_block_instantiation(self, info_MDP):
        """
        Parameters
        ----------
        info_MDP: This is an object of Class mushroom_rl.environment.MDPInfo. It contains the action and observation spaces, 
                  gamma and the horizon of the MDP.
        
        Returns
        -------
        This method returns True if the algo_params were set successfully, and False otherwise.
        """
        
        self.info_MDP = info_MDP
                 
        if(self.algo_params is None):
            eps = Integer(hp_name='epsilon', obj_name='epsilon_lspi', current_actual_value=1e-2)
            
            self.algo_params = {'epsilon': eps} 
           
        dict_of_n_train_samples = {'n_train_samples': self.n_train_samples}
        self.algo_params = {**self.algo_params, **dict_of_n_train_samples}
        
        is_set_param_success = self.set_params(new_params=self.algo_params)
        
        if(not is_set_param_success):
            err_msg = 'There was an error setting the parameters of a'+'\''+str(self.__class__.__name__)+'\' object!'
            self.logger.error(msg=err_msg)
            self.fully_instantiated = False
            self.is_learn_successful = False
            return False
           
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object fully instantiated!')
        self.fully_instantiated = True
        return True 
         
    def set_params(self, new_params):
        """
        Parameters
        ----------
        new_params: The new parameters to be used in the specific model generation algorithm. 
                    It must be a dictionary that does not contain any dictionaries 
                    (i.e: all parameters must be at the same level).
                    
                    The previous parameters are lost (except for mdpInfo, policy, input_shape)
                    
                    We need to create the dictionary in the right form for MushroomRL. Then it needs to update self.algo_params. 
                    Then it needs to update the object self.algo_object: to this we need to pass the actual values and not 
                    the Hyperparameter objects. 
                    
        Returns
        -------
        bool: This method returns True if new_params is set correctly, and False otherwise.
        
        I am overriding the method of the Class ModelGenerationMushroomOffline since in LSPI I do not have n_iterations, nor the
        approximator.
        """
           
        if(new_params is not None):     
            mdp_info = Categorical(hp_name='mdp_info', obj_name='mdp_info_'+str(self.model.__name__), 
                                   current_actual_value=self.info_MDP)
            
            input_shape = Categorical(hp_name='input_shape', obj_name='input_shape_'+str(self.model.__name__), 
                                      current_actual_value=self.info_MDP.observation_space.shape)
        
            if(self.regressor_type == 'action_regressor'):
                output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                           current_actual_value=(1,))
                n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                         current_actual_value=self.info_MDP.action_space.n)
            elif(self.regressor_type == 'q_regressor'):
                output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                           current_actual_value=(self.info_MDP.action_space.n,))
                n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                         current_actual_value=self.info_MDP.action_space.n)
            elif(self.regressor_type == 'generic_regressor'):
                output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                           current_actual_value=self.info_MDP.action_space.shape)
                #to have a generic regressor I must not specify n_actions
                n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                         current_actual_value=None)
                
            #can set epsilon to [0-0.05] so that we mainly exploit the policy (little random actions). I leave a few random 
            #actions to avoid overfitting the testing procedure:                    
            policy = Categorical(hp_name='policy', obj_name='policy_'+str(self.model.__name__), 
                                 current_actual_value=EpsGreedy(epsilon=Parameter(value=0)))
                
            tmp_structured_algo_params = {'mdp_info': mdp_info,
                                          'policy': policy,
                                          'approximator_params': {'input_shape': input_shape,
                                                                  'n_actions': n_actions,
                                                                  'output_shape': output_shape
                                                                 }
                                         }
            
            #in LSPI i now have only epsilon left to pick:
            if('epsilon' in list(new_params.keys())):
                tmp_structured_algo_params.update({'epsilon': new_params['epsilon']})
            
            structured_dict_of_values = self._select_current_actual_value_from_hp_classes(params_structured_dict=
                                                                                          tmp_structured_algo_params)

            #i need to un-pack structured_dict_of_values for LSPI
            self.algo_object = LSPI(**structured_dict_of_values)   
            
            final_dict_of_params = tmp_structured_algo_params
            final_dict_of_params['n_train_samples'] = new_params['n_train_samples']
            
            self.algo_params = final_dict_of_params

            tmp_new_params = self.get_params()
            
            if(tmp_new_params is not None):
                self.algo_params_upon_instantiation = copy.deepcopy(tmp_new_params)
            else:
                self.logger.error(msg='There was an error getting the parameters!')
                return False

            return True
        else:
            self.logger.error(msg='Cannot set parameters: \'new_params\' is \'None\'!')
            return False
