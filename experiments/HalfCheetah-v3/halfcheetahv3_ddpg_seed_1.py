import numpy as np
np.random.seed(2)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.policy import OrnsteinUhlenbeckPolicy            

from ARLO.hyperparameter import Categorical, Real, Integer
from ARLO.block import ModelGenerationMushroomOnlineDDPG
from ARLO.rl_pipeline import OnlineRLPipeline
from ARLO.tuner import TunerGenetic
from ARLO.block import AutoModelGeneration
from ARLO.input_loader import LoadSameEnv
from ARLO.metric import DiscountedReward
from ARLO.environment import BaseHalfCheetah

if __name__ == '__main__':
    dir_chkpath = '/path/for/saving/output' 
        
    my_env = BaseHalfCheetah(obj_name='my_env', gamma=1, horizon=1000)
    my_env.seed(2)
    
    my_log_mode = 'file'
    current_seed = 2
    
    class CriticNetwork(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
    
            n_input = input_shape[-1]
            n_output = output_shape[0]
            
            self.hlin = nn.Linear(n_input, 128)
            self.hl1 = nn.Linear(128, 128)
            self.hl2 = nn.Linear(128, 128)
            self.hlout = nn.Linear(128, n_output)
            
            nn.init.xavier_uniform_(self.hlin.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.hl1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.hl2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.hlout.weight, gain=nn.init.calculate_gain('relu'))
    
        def forward(self, state, action, **kwargs):
            state_action = torch.cat((state.float(), action.float()), dim=1)
            h = F.relu(self.hlin(state_action))
            h = F.relu(self.hl1(h))
            h = F.relu(self.hl2(h))
    
            q = self.hlout(h)
    
            return torch.squeeze(q)
            
    class ActorNetwork(nn.Module):
        def __init__(self, input_shape, output_shape,  **kwargs):
            super(ActorNetwork, self).__init__()
    
            n_input = input_shape[-1]
            n_output = output_shape[0]
    
            self.hlin = nn.Linear(n_input, 128)
            self.hl1 = nn.Linear(128, 128)
            self.hl2 = nn.Linear(128, 128)
            self.hlout = nn.Linear(128, n_output)
            
            nn.init.xavier_uniform_(self.hlin.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.hl1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.hl2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.hlout.weight, gain=nn.init.calculate_gain('relu'))
         
        def forward(self, state, **kwargs):
            h = F.relu(self.hlin(torch.squeeze(state, 1).float()))
            h = F.relu(self.hl1(h))
            h = F.relu(self.hl2(h))
    
            return self.hlout(h)
    
    policy_class = Categorical(hp_name='policy_class', obj_name='policy_class_ddpg', 
                               current_actual_value=OrnsteinUhlenbeckPolicy, seeder=2)
    
    sigma = Real(hp_name='sigma', current_actual_value=0.2, obj_name='sigma_ddpg', seeder=2)
    theta = Real(hp_name='theta', current_actual_value=0.15, obj_name='theta_ddpg', seeder=2)
    dt =  Real(hp_name='dt', current_actual_value=1e-2, obj_name='dt_ddpg', seeder=2)
    
    critic = CriticNetwork 
    actor = ActorNetwork
    
    #actor:
    actor_network = Categorical(hp_name='actor_network', obj_name='actor_network_ddpg', current_actual_value=actor)
     
    actor_class = Categorical(hp_name='actor_class', obj_name='actor_class_ddpg', current_actual_value=optim.Adam) 
    
    actor_lr = Real(hp_name='actor_lr', obj_name='actor_lr_ddpg', current_actual_value=1e-4, range_of_values=[1e-5, 1e-2],
                    to_mutate=True)
    
    #critic:
    critic_network = Categorical(hp_name='critic_network', obj_name='critic_network_ddpg', current_actual_value=critic)
    
    critic_class = Categorical(hp_name='critic_class', obj_name='critic_class_ddpg', current_actual_value=optim.Adam) 
    
    critic_lr = Real(hp_name='critic_lr', obj_name='critic_lr_ddpg', current_actual_value=1e-3, range_of_values=[1e-5, 1e-2], 
                     to_mutate=True)
    
    critic_loss = Categorical(hp_name='loss', obj_name='loss_ddpg', current_actual_value=F.mse_loss)
                
    batch_size = Integer(hp_name='batch_size', obj_name='batch_size_ddpg', current_actual_value=128, range_of_values=[8, 256], 
                         to_mutate=True)
    
    initial_replay_size = Integer(hp_name='initial_replay_size', current_actual_value=5000, range_of_values=[1000, 20000], 
                                  to_mutate=True, obj_name='initial_replay_size_ddpg')
    
    max_replay_size = Integer(hp_name='max_replay_size', current_actual_value=1000000, range_of_values=[10000, 1500000],
                              to_mutate=True, obj_name='max_replay_size_ddpg')
    
    tau = Real(hp_name='tau', current_actual_value=0.001, obj_name='tau_ddpg')
    
    policy_delay = Integer(hp_name='policy_delay', current_actual_value=1, obj_name='policy_delay_ddpg')
    
    n_epochs = Integer(hp_name='n_epochs', current_actual_value=50, range_of_values=[1,50], to_mutate=True, obj_name='n_epochs_ddpg')
    
    n_steps = Integer(hp_name='n_steps', current_actual_value=7500, range_of_values=[1000,15000], to_mutate=True, 
                      obj_name='n_steps_ddpg')
    
    n_steps_per_fit = Integer(hp_name='n_steps_per_fit', current_actual_value=10, range_of_values=[1,10000], to_mutate=True, 
                              obj_name='n_steps_per_fit_ddpg')
    
    n_episodes = Integer(hp_name='n_episodes', current_actual_value=None, obj_name='n_episodes_ddpg')
    
    n_episodes_per_fit = Integer(hp_name='n_episodes_per_fit', current_actual_value=None, obj_name='n_episodes_per_fit_ddpg')
        
    dict_of_params = {'policy_class': policy_class,
                      'sigma': sigma,
                      'theta': theta,
                      'dt': dt,
                      'actor_network': actor_network, 
                      'actor_class': actor_class, 
                      'actor_lr': actor_lr,
                      'critic_network': critic_network, 
                      'critic_class': critic_class, 
                      'critic_lr': critic_lr,           
                      'loss': critic_loss,
                      'batch_size': batch_size,
                      'initial_replay_size': initial_replay_size,
                      'max_replay_size': max_replay_size,
                      'tau': tau,
                      'policy_delay': policy_delay,
                      'n_epochs': n_epochs,
                      'n_steps': n_steps,
                      'n_steps_per_fit': n_steps_per_fit,
                      'n_episodes': n_episodes,
                      'n_episodes_per_fit': n_episodes_per_fit
                     }
    
    my_ddpg = ModelGenerationMushroomOnlineDDPG(eval_metric=DiscountedReward(obj_name='ddpg_metric', n_episodes=10, batch=False, 
                                                                             log_mode=my_log_mode, 
                                                                             checkpoint_log_path=dir_chkpath), 
                                                obj_name='my_ddpg', regressor_type='generic_regressor', log_mode=my_log_mode, 
                                                checkpoint_log_path=dir_chkpath, n_jobs=16, seeder=current_seed,
                                                algo_params=dict_of_params, deterministic_output_policy=True)
    
    tuner_dict = dict(block_to_opt=my_ddpg, n_agents=20, n_generations=50, n_jobs=1, job_type='thread', seeder=current_seed,
                      eval_metric=DiscountedReward(obj_name='discounted_rew_genetic_algo', n_episodes=10, batch=False, 
                                                   n_jobs=1, job_type='process', log_mode=my_log_mode, checkpoint_log_path=dir_chkpath),
                      input_loader=LoadSameEnv(obj_name='input_loader_env'), 
                      obj_name='genetic_algo', prob_point_mutation=0.5, output_save_periodicity=1, 
                      tuning_mode='no_elitism', pool_size=None, log_mode=my_log_mode, checkpoint_log_path=dir_chkpath)
    
    tuner = TunerGenetic(**tuner_dict)
    
    auto_model_gen = AutoModelGeneration(eval_metric=DiscountedReward(obj_name='discounted_rew_auto_model_gen', n_episodes=10,
                                                                      batch=False, n_jobs=1, job_type='process', 
                                                                      log_mode=my_log_mode, checkpoint_log_path=dir_chkpath),
                                         obj_name='auto_model_gen', tuner_blocks_dict={'genetic_tuner': tuner}, 
                                         log_mode=my_log_mode, checkpoint_log_path=dir_chkpath)
    
    my_pipeline = OnlineRLPipeline(list_of_block_objects=[auto_model_gen], 
                                   eval_metric=DiscountedReward(obj_name='pipeline_metric', n_episodes=10, batch=True, 
                                                                log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                   obj_name='OnlinePipeline',  log_mode=my_log_mode, checkpoint_log_path=dir_chkpath) 
    
    out = my_pipeline.learn(env=my_env)