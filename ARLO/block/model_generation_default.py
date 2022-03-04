"""
This module contains the dictionary which is used as default for the member tuner_blocks_dict of the Class AutoModelGeneration. 
"""

from ARLO.metric.metric import DiscountedReward, TDError
from ARLO.input_loader.input_loader import LoadUniformSubSampleWithReplacementAndEnv, LoadUniformSubSampleWithReplacement
from ARLO.input_loader.input_loader import LoadSameEnv
from ARLO.block.model_generation_offline import ModelGenerationMushroomOfflineFQI
from ARLO.block.model_generation_online import ModelGenerationMushroomOnlinePPO
from ARLO.tuner.tuner_genetic import TunerGenetic


model_gen_offline_fqi_with_env = ModelGenerationMushroomOfflineFQI(eval_metric=DiscountedReward(obj_name='discounted_rew', 
                                                                                                n_episodes=10, 
                                                                                                batch=True), 
                                                                   obj_name='FQI_default_xgb')

model_gen_offline_fqi_without_env = ModelGenerationMushroomOfflineFQI(eval_metric=TDError(obj_name='td_error'), 
                                                                      obj_name='FQI_default_xgb')

model_gen_online_ppo = ModelGenerationMushroomOnlinePPO(eval_metric=DiscountedReward(obj_name='discounted_rew', n_episodes=10, 
                                                                                     batch=False), 
                                                        obj_name='FQI_default_ppo')

input_loader_offline_with_env = LoadUniformSubSampleWithReplacementAndEnv(obj_name='input_loader', single_split_length=10000)

input_loader_offline_without_env = LoadUniformSubSampleWithReplacement(obj_name='input_loader', single_split_length=10000)

input_loader_online = LoadSameEnv(obj_name='input_loader_online_default')

default_discounted_rew_batch = DiscountedReward(obj_name='discounted_rew_default', batch=True, n_episodes=10)

default_discounted_rew_non_batch = DiscountedReward(obj_name='discounted_rew_default', batch=False, n_episodes=10)

default_td_error = TDError(obj_name='td_error')

#dictionary with tuners that is used as default in case the user doesn't want to specify anything. This dictionary can be called 
#and modified by the user in the main to add/remove methods.
automatic_model_generation_default = {'MushroomOfflineFQI_XGB_GA_with_env': 
                                      TunerGenetic(block_to_opt=model_gen_offline_fqi_with_env, n_agents=10, 
                                                   n_generations=5, eval_metric=default_discounted_rew_batch,
                                                   input_loader=input_loader_offline_with_env, 
                                                   obj_name='genetic_algo_fqi_default'),
                                     'MushroomOfflineFQI_XGB_GA_without_env': 
                                      TunerGenetic(block_to_opt=model_gen_offline_fqi_without_env, n_agents=10, 
                                                   n_generations=5, eval_metric=default_td_error,
                                                   input_loader=input_loader_offline_without_env, 
                                                   obj_name='genetic_algo_fqi_default'),
                                      'MushroomOnlinePPO_GA': 
                                       TunerGenetic(block_to_opt=model_gen_online_ppo, n_agents=10, n_generations=5,
                                                    eval_metric=default_discounted_rew_non_batch,
                                                    input_loader=input_loader_online, obj_name='genetic_algo_ppo_default')

                                     }