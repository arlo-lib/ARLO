import copy

import numpy as np
np.random.seed(2)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from AutoRL.environment import LQG
from AutoRL.block import ModelGenerationMushroomOnlineSAC
from AutoRL.rl_pipeline import OnlineRLPipeline
from AutoRL.tuner import PBTGeneticAlgorithm
from AutoRL.block import AutoModelGeneration
from AutoRL.input_loader import LoadSameEnv
from AutoRL.metric import DiscountedReward, SomeSpecificMetric
from AutoRL.hyperparameter import Categorical, Real, Integer

if __name__ == '__main__':
    dir_chkpath = '/path/for/saving/output' 
    
    A = np.array([[1,0],[0,1]])
    B = np.array([[1,0,0],[0,0,1]])
    Q = 0.7*np.array([[1,0],[0,1]])
    R = 0.3*np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    my_lqg = LQG(obj_name='lqg', A=A, B=B, Q=Q, R=R, max_pos=3.5, max_action=3.5, env_noise=0.1*np.eye(2), 
                 controller_noise=0*1e-4*np.eye(3), seeder=2, horizon=15, gamma=0.9)
    
    my_log_mode = 'file'
    current_seed = 2022
    
    class CriticNetwork(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
    
            n_input = input_shape[0]
            n_output = output_shape[0]
            
            self.hl0 = nn.Linear(n_input, 16)
            self.hl1 = nn.Linear(16, 16)
            self.hl2 = nn.Linear(16, n_output)
            
            nn.init.xavier_uniform_(self.hl0.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.hl1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.hl2.weight, gain=nn.init.calculate_gain('relu'))
    
        def forward(self, state, action, **kwargs):
            state_action = torch.cat((state.float(), action.float()), dim=1)
            h = F.relu(self.hl0(state_action))
            h = F.relu(self.hl1(h))
            q = self.hl2(h)
    
            return torch.squeeze(q)
            
    class ActorNetwork(nn.Module):
        def __init__(self, input_shape, output_shape,  **kwargs):
            super(ActorNetwork, self).__init__()
        
            n_input = input_shape[0]
            n_output = output_shape[0]
        
            self.hl0 = nn.Linear(n_input, 16)
            self.hl1 = nn.Linear(16, 16)
            self.hl2 = nn.Linear(16, n_output)
            
            nn.init.xavier_uniform_(self.hl0.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.hl1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.hl2.weight, gain=nn.init.calculate_gain('relu'))
         
        def forward(self, state, **kwargs):
            h = F.relu(self.hl0(torch.squeeze(state, 1).float()))
            h = F.relu(self.hl1(h))
        
            return self.hl2(h)
    
    #actor:
    actor_network_mu = Categorical(hp_name='actor_network_mu', obj_name='actor_network_mu_sac', 
                                   current_actual_value=ActorNetwork)
    
    actor_network_sigma = Categorical(hp_name='actor_network_sigma', obj_name='actor_network_sigma_sac', 
                                      current_actual_value=copy.deepcopy(ActorNetwork))
     
    actor_class = Categorical(hp_name='actor_class', obj_name='actor_class_sac', 
                              current_actual_value=optim.Adam) 
    
    actor_lr = Categorical(hp_name='actor_lr', obj_name='actor_lr_sac', current_actual_value=1e-2, 
                           possible_values=[1e-5, 1e-4, 1e-3, 1e-2], to_mutate=True)
    
    #critic:
    critic_network = Categorical(hp_name='critic_network', obj_name='critic_network_sac', current_actual_value=CriticNetwork)
    
    critic_class = Categorical(hp_name='critic_class', obj_name='critic_class_sac', current_actual_value=optim.Adam) 
    
    critic_lr = Categorical(hp_name='critic_lr', obj_name='critic_lr_sac', current_actual_value=1e-2, 
                            possible_values=[1e-5, 1e-4, 1e-3, 1e-2], to_mutate=True)
    
    critic_loss = Categorical(hp_name='loss', obj_name='loss_sac', current_actual_value=F.mse_loss)
                
    batch_size = Categorical(hp_name='batch_size', obj_name='batch_size_sac', 
                             current_actual_value=64, possible_values=[8, 16, 32, 64, 128], to_mutate=True)
    
    initial_replay_size = Categorical(hp_name='initial_replay_size', current_actual_value=100, 
                                      possible_values=[10, 100, 300, 500, 1000, 5000],
                                      to_mutate=True, obj_name='initial_replay_size_sac')
    
    max_replay_size = Categorical(hp_name='max_replay_size', current_actual_value=10000, possible_values=[3000, 10000, 30000, 100000],
                                  to_mutate=True, obj_name='max_replay_size_sac')
    
    warmup_transitions = Categorical(hp_name='warmup_transitions', current_actual_value=100, possible_values=[50, 100, 500], 
                                     to_mutate=True, obj_name='warmup_transitions_sac')
    
    tau = Categorical(hp_name='tau', current_actual_value=0.005, obj_name='tau_sac')
    
    lr_alpha = Categorical(hp_name='lr_alpha', current_actual_value=1e-3, obj_name='lr_alpha_sac', possible_values=[1e-5, 1e-4, 1e-3], 
                           to_mutate=True)
    
    log_std_min = Real(hp_name='log_std_min', current_actual_value=-20, obj_name='log_std_min_sac')
    
    log_std_max = Real(hp_name='log_std_max', current_actual_value=3, obj_name='log_std_max_sac')
    
    target_entropy = Real(hp_name='target_entropy', current_actual_value=None, obj_name='target_entropy_sac')
    
    n_epochs = Integer(hp_name='n_epochs', current_actual_value=15, range_of_values=[1,30], to_mutate=True, obj_name='n_epochs')
    
    n_steps = Integer(hp_name='n_steps', current_actual_value=None,  obj_name='n_steps')
    
    n_steps_per_fit = Integer(hp_name='n_steps_per_fit', current_actual_value=None, obj_name='n_steps_per_fit')
    
    n_episodes = Integer(hp_name='n_episodes', current_actual_value=500, range_of_values=[1,1600], to_mutate=True, 
                         obj_name='n_episodes')
    
    n_episodes_per_fit = Integer(hp_name='n_episodes_per_fit', current_actual_value=100, range_of_values=[1,500], to_mutate=True, 
                                 obj_name='n_episodes_per_fit')
    
    dict_of_params_sac = {'actor_network_mu': actor_network_mu, 
                          'actor_network_sigma': actor_network_sigma,
                          'actor_class': actor_class, 
                          'actor_lr': actor_lr,
                          'critic_network': critic_network, 
                          'critic_class': critic_class, 
                          'critic_lr': critic_lr,           
                          'loss': critic_loss,
                          'batch_size': batch_size,
                          'initial_replay_size': initial_replay_size,
                          'max_replay_size': max_replay_size,
                          'warmup_transitions': warmup_transitions,
                          'tau': tau,
                          'lr_alpha': lr_alpha,
                          'log_std_min': log_std_min,
                          'log_std_max': log_std_max,
                          'target_entropy': target_entropy,
                          'n_epochs': n_epochs,
                          'n_steps': n_steps,
                          'n_steps_per_fit': n_steps_per_fit,
                          'n_episodes': n_episodes,
                          'n_episodes_per_fit': n_episodes_per_fit
                         }
    
    my_sac = ModelGenerationMushroomOnlineSAC(eval_metric=DiscountedReward(obj_name='sac_metric', n_episodes=100, batch=False,
                                                                           log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                              obj_name='my_sac', regressor_type='generic_regressor', log_mode=my_log_mode, 
                                              checkpoint_log_path=dir_chkpath, n_jobs=16, seeder=current_seed, 
                                              algo_params=dict_of_params_sac,
                                              deterministic_output_policy=False)
                                                                                         
    pbt_dict = dict(block_to_opt=my_sac, n_agents=20, n_generations=100, n_jobs=16, job_type='thread', seeder=current_seed,
                    eval_metric=DiscountedReward(obj_name='discounted_rew_genetic_algo', n_episodes=100, batch=False, 
                                                 n_jobs=1, job_type='process', log_mode=my_log_mode, checkpoint_log_path=dir_chkpath),
                    input_loader=LoadSameEnv(obj_name='input_loader_env'), obj_name='genetic_algo', prob_point_mutation=0.5, 
                    output_save_periodicity=1, log_mode=my_log_mode, checkpoint_log_path=dir_chkpath, tuning_mode='no_elitism')
    
    pbt = PBTGeneticAlgorithm(**pbt_dict)
    
    auto_model_gen = AutoModelGeneration(eval_metric=DiscountedReward(obj_name='discounted_rew_auto_model_gen', n_episodes=100,
                                                                      batch=False, n_jobs=1, job_type='process', 
                                                                      log_mode=my_log_mode, checkpoint_log_path=dir_chkpath),
                                         obj_name='auto_model_gen', tuner_blocks_dict={'ppo_tuner': pbt}, 
                                         log_mode=my_log_mode, checkpoint_log_path=dir_chkpath)
    
    my_pipeline = OnlineRLPipeline(list_of_block_objects=[auto_model_gen], 
                                   eval_metric=SomeSpecificMetric(obj_name='some_specific_metric_pipeline'), 
                                   obj_name='OnlinePipeline', log_mode=my_log_mode, checkpoint_log_path=dir_chkpath) 
    
    out = my_pipeline.learn(env=my_lqg)