import gym
import numpy as np
np.random.seed(2)

from mushroom_rl.utils.spaces import Box, Discrete
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from ARLO.environment import BaseEnvironment
from ARLO.block import DataGenerationRandomUniformPolicy, ModelGenerationMushroomOfflineFQI, AutoModelGeneration
from ARLO.metric import SomeSpecificMetric, DiscountedReward
from ARLO.hyperparameter import Integer, Categorical, Real
from ARLO.rl_pipeline import OfflineRLPipeline
from ARLO.tuner import TunerGenetic
from ARLO.input_loader import LoadUniformSubSampleWithReplacementAndEnv

if __name__ == '__main__':
    dir_chkpath = '/path/for/saving/output' 
    
    log_mode = 'file'
    
    """
    The code below is taken from https://github.com/AndreaTirinzoni/iw-transfer-rl
    
    Cyclostationary Dam Control
    
    Info
    ----
      - State space: 2D Box (storage,day)
      - Action space: 1D Box (release decision)
      - Parameters: capacity, demand, flooding threshold, inflow mean per day, inflow std, demand weight, flooding weigt=ht
    
    References
    ----------
      - Simone Parisi, Matteo Pirotta, Nicola Smacchia,
        Luca Bascetta, Marcello Restelli,
        Policy gradient approaches for multi-objective sequential decision making
        2014 International Joint Conference on Neural Networks (IJCNN)
        
      - A. Castelletti, S. Galelli, M. Restelli, R. Soncini-Sessa
        Tree-based reinforcement learning for optimal water reservoir operation
        Water Resources Research 46.9 (2010)
        
      - Andrea Tirinzoni, Andrea Sessa, Matteo Pirotta, Marcello Restelli.
        Importance Weighted Transfer of Samples in Reinforcement Learning.
        International Conference on Machine Learning. 2018.
    """
    
    class Dam(gym.Env):
        # metadata = {
        #     'render.modes': ['human', 'rgb_array'],
        #     'video.frames_per_second': 30
        # }
        
        def set_local_prng(self, new_seeder):
            self.local_prng = np.random.default_rng(new_seeder)
            self.seeder = new_seeder
     
        def __init__(self, inflow_profile = 1, alpha = 0.5, beta = 0.5, penalty_on = False, experiment=False):
            
            self.local_prng = np.random.default_rng(2)
            
            self.experiment = experiment
            
            self.horizon = 360
            self.gamma = 0.999
            self.state_dim = 2
            self.action_dim = 1
    
            self.DEMAND = 10.0  # Water demand -> At least DEMAND/day must be supplied or a cost is incurred
            self.FLOODING = 300.0  # Flooding threshold -> No more than FLOODING can be stored or a cost is incurred
            self.MIN_STORAGE = 50.0 # Minimum storage capacity -> At most max{S - MIN_STORAGE, 0} must be released
            self.MAX_STORAGE = 500.0  # Maximum storage capacity -> At least max{S - MAX_STORAGE, 0} must be released
            
            # Random inflow (e.g. rain) mean for each day (360-dimensional vector)
            self.INFLOW_MEAN = self._get_inflow_profile(inflow_profile)  
            self.INFLOW_STD = 2.0 # Random inflow std
            
            assert alpha + beta == 1.0 # Check correctness
            self.ALPHA = alpha # Weight for the flooding cost
            self.BETA = beta # Weight for the demand cost
            
            self.penalty_on = penalty_on # Whether to penalize illegal actions or not
            
            # Gym attributes
            self.viewer = None
            
            self.action_space = Discrete(8)
            
            if(self.experiment):
                self.observation_space = Box(low=np.zeros(31),
                                             high=np.inf*np.ones(31))
            else:
                self.observation_space = Box(low=np.array([0,1]),
                                             high=np.array([np.inf,360]))
    
        def _get_inflow_profile(self,n):
            assert n >= 1 and n <= 7
            
            if n == 1:
                return self._get_inflow_1()
            elif n == 2:
                return self._get_inflow_2()
            elif n == 3:
                return self._get_inflow_3()
            elif n == 4:
                return self._get_inflow_4()
            elif n == 5:
                return self._get_inflow_5()
            elif n == 6:
                return self._get_inflow_6()
            elif n == 7:
                return self._get_inflow_7()
        
        def _get_inflow_1(self):
            y = np.zeros(360)  
            x = np.arange(360)
            y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) + 0.5
            y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 2 + 0.5
            y[240:] = np.sin(x[240:] * 3 * np.pi / 359) + 0.5
            return y * 8 + 4
        
        def _get_inflow_2(self):
            y = np.zeros(360)  
            x = np.arange(360)
            y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) / 2 + 0.25
            y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359 + np.pi) * 3 + 0.25
            y[240:] = np.sin(x[240:] * 3 * np.pi / 359 + np.pi) / 4 + 0.25
            return y * 8 + 4
        
        def _get_inflow_3(self):
            y = np.zeros(360)  
            x = np.arange(360)
            y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) * 3 + 0.25
            y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 4 + 0.25
            y[240:] = np.sin(x[240:] * 3 * np.pi / 359) / 2 + 0.25
            return y * 8 + 4
        
        def _get_inflow_4(self):
            y = np.zeros(360)  
            x = np.arange(360)
            y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) + 0.5
            y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 2.5 + 0.5
            y[240:] = np.sin(x[240:] * 3 * np.pi / 359) + 0.5
            return y * 7 + 4
        
        def _get_inflow_5(self):
            y = np.zeros(360)  
            x = np.arange(360)
            y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359 - np.pi / 12) / 2 + 0.5
            y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359 - np.pi / 12) / 2 + 0.5
            y[240:] = np.sin(x[240:] * 3 * np.pi / 359 - np.pi / 12) / 2 + 0.5
            return y * 8 + 5
        
        def _get_inflow_6(self):
            y = np.zeros(360)  
            x = np.arange(360)
            y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359 + np.pi / 8) / 3 + 0.5
            y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359 + np.pi / 8) / 3 + 0.5
            y[240:] = np.sin(x[240:] * 3 * np.pi / 359 + np.pi / 8) / 3 + 0.5
            return y * 8 + 4
        
        def _get_inflow_7(self):
            y = np.zeros(360)  
            x = np.arange(360)
            y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) + 0.5
            y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 3 + 0.5
            y[240:] = np.sin(x[240:] * 3 * np.pi / 359) * 2 + 0.5
            return y * 8 + 5
            
        def step(self, action):
            action = action[0]
            
            actions_pool = [0, 3, 5, 7, 10, 15, 20, 30]
            
            action = actions_pool[action]
            
            action = float(action)
            
            # Get current state
            state = self.get_state()
            storage = state[0]
            day = state[1]
            
            # Bound the action
            actionLB = max(storage - self.MAX_STORAGE, 0.0)
            actionUB = max(storage - self.MIN_STORAGE, 0.0)
    
            # Penalty proportional to the violation
            bounded_action = min(max(action, actionLB), actionUB)
            penalty = -abs(bounded_action - action) * self.penalty_on
    
            # Transition dynamics
            action = bounded_action
            inflow = self.INFLOW_MEAN[int(day-1)] + self.local_prng.normal() * self.INFLOW_STD
          
            nextstorage = max(storage + inflow - action, 0.0)
    
            # Cost due to the excess level wrt the flooding threshold
            reward_flooding = -max(storage - self.FLOODING, 0.0) / 4
    
            # Deficit in the water supply wrt the water demand
            reward_demand = -max(self.DEMAND - action, 0.0) ** 2
            
            # The final reward is a weighted average of the two costs
            reward = self.ALPHA * reward_flooding + self.BETA * reward_demand + penalty
    
            # Get next day
            nextday = day + 1 if day < 360 else 1
    
            self.state = [nextstorage, nextday]
            if(self.experiment):
                inflow = self.INFLOW_MEAN + self.local_prng.normal() * self.INFLOW_STD
                if(day >= 31):
                    lagged_inflows = inflow[int(day-31):int(day-1)].tolist()
                else:
                    lagged_inflows = inflow[360-int(31-day):].tolist()  + inflow[:int(day-1)].tolist()
    
                next_state = np.array([[nextstorage] + lagged_inflows])
            else:
                next_state = self.get_state()
                
            return next_state, reward, False, {}
    
        def reset(self, state=None):
            
            if state is None:
                init_days = np.array([1, 120, 240])
                self.state = [self.local_prng.uniform(self.MIN_STORAGE, self.MAX_STORAGE), 
                              init_days[self.local_prng.integers(low=0,high=3)]]
            else:
                self.state = np.array(state)
    
            if(self.experiment):
                current_state = []
                for i in range(31):
                    current_state.append(self.local_prng.uniform(self.MIN_STORAGE, self.MAX_STORAGE))
                
                current_state = np.array(current_state)
            else:
                current_state = self.get_state()
            return current_state
    
        def get_state(self):
            return np.array(self.state)
    
    # Create Env Class for ARLO
    class myDam(BaseEnvironment):
        def __init__(self, obj_name, experiment, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                      job_type='process'):
            super().__init__(obj_name, seeder, log_mode, checkpoint_log_path, verbosity, n_jobs, job_type)
            
            self.dam_env = Dam(inflow_profile = 1, alpha = 0.3, beta = 0.7, experiment=experiment)
            
            self.observation_space =  self.dam_env.observation_space
            self.action_space = self.dam_env.action_space
            self.horizon = self.dam_env.horizon
            self.gamma =self.dam_env.gamma
                
        def set_local_prng(self, new_seeder):
            self.dam_env.set_local_prng(new_seeder)
            
        def seed(self, seeder):
            self.set_local_prng(new_seeder=seeder)
    
        def step(self, action):
            out = self.dam_env.step(action=action)    
    
            new_state = np.array([out[0][0]])
            
            return new_state, out[1], out[2], out[3]
       
        def reset(self, state=None):
            return np.array([self.dam_env.reset()[0]])
    
        def render(self, mode='human'):
            raise NotImplementedError
            
    my_dam = myDam(obj_name='my_dam', experiment=False)
                                
    #############################################################################################################################
    extra_trees_params = {'approximator': Categorical(hp_name='approximator', obj_name='approximator', 
                                                      current_actual_value=ExtraTreesRegressor),
                          'n_iterations': Integer(hp_name='n_iterations', current_actual_value=60, to_mutate=True, 
                                                  range_of_values=[2,60], obj_name='fqi_n_iterations'),
                          'criterion': Categorical(hp_name='criterion', current_actual_value='squared_error', obj_name='criterion'),
                          'min_samples_split': Integer(hp_name='min_samples_split', current_actual_value=10, range_of_values=[1,50], 
                                                       to_mutate=True, obj_name='min_samples_split'),
                          'n_estimators': Integer(hp_name='n_estimators', current_actual_value=100, range_of_values=[5,250],
                                                  to_mutate=True, obj_name='n_estimators'),
                          'n_jobs': Integer(obj_name='n_jobs', hp_name='n_jobs', current_actual_value=16)
                         }
    
    model_gen_extra_trees = ModelGenerationMushroomOfflineFQI(eval_metric=SomeSpecificMetric('model_gen'), 
                                                              obj_name='fqi_extra_trees', regressor_type='action_regressor',
                                                              algo_params=extra_trees_params, log_mode=log_mode, 
                                                              checkpoint_log_path=dir_chkpath)
    
    pbt_dict = dict(block_to_opt=model_gen_extra_trees, n_agents=20, n_generations=50, n_jobs=1, job_type='thread', seeder=2,
                    eval_metric=DiscountedReward(obj_name='discounted_rew_genetic_algo', n_episodes=10, batch=True, 
                                                 n_jobs=1, job_type='process', log_mode=log_mode, checkpoint_log_path=dir_chkpath),
                    input_loader=LoadUniformSubSampleWithReplacementAndEnv(obj_name='input_loader_env', single_split_length=10800,
                                                                           log_mode=log_mode, checkpoint_log_path=dir_chkpath), 
                    obj_name='genetic_algo_extra_trees', prob_point_mutation=0.5, output_save_periodicity=20,
                    tuning_mode='no_elitism', pool_size=None, log_mode=log_mode, checkpoint_log_path=dir_chkpath)
    #############################################################################################################################
    xgb_params = {'approximator': Categorical(hp_name='approximator', obj_name='approximator', 
                                              current_actual_value=XGBRegressor),
                  'n_iterations': Integer(hp_name='n_iterations', current_actual_value=60, to_mutate=True, 
                                          range_of_values=[2,60], obj_name='fqi_n_iterations'),
                  'subsample': Real(hp_name='subsample', current_actual_value=0.8, obj_name='fqi_xgb_subsample',
                                    range_of_values=[0.5,1], to_mutate=True),
                  'max_depth': Integer(hp_name='max_depth', current_actual_value=10, range_of_values=[4,15], to_mutate=True,
                                       obj_name='fqi_xgb_max_depth'),
                  'learning_rate': Real(hp_name='learning_rate', current_actual_value=0.3, range_of_values=[1e-3, 0.4], 
                                        to_mutate=True, obj_name='learning_rate_fqi_xgb'),
                  'verbosity': Integer(hp_name='verbosity', current_actual_value=0, obj_name='xgb_verb'),
                  'min_child_weight': Integer(hp_name='min_child_weight', current_actual_value=10, range_of_values=[1,100], 
                                              to_mutate=True, obj_name='min_child_weight'),
                  'n_estimators': Integer(hp_name='n_estimators', current_actual_value=100, range_of_values=[5,250],
                                                  to_mutate=True, obj_name='n_estimators'),
                  'n_jobs': Integer(obj_name='n_jobs', hp_name='n_jobs', current_actual_value=16)
                 }
    
    model_gen_xgb = ModelGenerationMushroomOfflineFQI(eval_metric=SomeSpecificMetric('model_gen'), obj_name='fqi_xgb',
                                                      regressor_type='action_regressor',
                                                      algo_params=xgb_params, log_mode=log_mode, 
                                                      checkpoint_log_path=dir_chkpath)
    
    pbt_dict_2 = dict(block_to_opt=model_gen_xgb, n_agents=20, n_generations=50, n_jobs=16, job_type='thread', seeder=2,
                      eval_metric=DiscountedReward(obj_name='discounted_rew_genetic_algo', n_episodes=10, batch=True, 
                                                   n_jobs=1, job_type='process', log_mode=log_mode, checkpoint_log_path=dir_chkpath),
                      input_loader=LoadUniformSubSampleWithReplacementAndEnv(obj_name='input_loader_env', single_split_length=10800,
                                                                             log_mode=log_mode, checkpoint_log_path=dir_chkpath), 
                      obj_name='genetic_algo_xgb', prob_point_mutation=0.5, output_save_periodicity=20,
                      tuning_mode='no_elitism', pool_size=None, log_mode=log_mode, checkpoint_log_path=dir_chkpath)
    #############################################################################################################################

    pbt_1 = TunerGenetic(**pbt_dict)
    pbt_2 = TunerGenetic(**pbt_dict_2)
    
    auto_model_gen = AutoModelGeneration(eval_metric=DiscountedReward(obj_name='discounted_rew_auto_model_gen', n_episodes=10,
                                                                      batch=True, n_jobs=1, job_type='process', 
                                                                      log_mode=log_mode, checkpoint_log_path=dir_chkpath),
                                         obj_name='auto_model_gen', tuner_blocks_dict={'genetic_tuner_extra_trees': pbt_1,
                                                                                       'genetic_tuner_xgb': pbt_2}, 
                                         log_mode=log_mode, checkpoint_log_path=dir_chkpath)
    
    data_gen = DataGenerationRandomUniformPolicy(eval_metric=SomeSpecificMetric('data_gen'), obj_name='data_gen', 
                                                  algo_params={'n_samples': Integer(obj_name='n_samples', 
                                                                                    current_actual_value=10800,
                                                                                    hp_name='n_samples_fqi')})
    
    my_pipeline = OfflineRLPipeline(list_of_block_objects=[data_gen, auto_model_gen],
                                   eval_metric=SomeSpecificMetric(obj_name='some_specific_metric_pipeline'), 
                                   obj_name='OnlinePipeline',  log_mode=log_mode, checkpoint_log_path=dir_chkpath) 
    
    out = my_pipeline.learn(env=my_dam)