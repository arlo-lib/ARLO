import os
import numpy as np
import matplotlib.pyplot as plt

from ARLO.hyperparameter import Categorical, Real
from ARLO.abstract_unit import load

if __name__ == '__main__':
    dir_path = '/path/folder/containing/pickled/agents'
    
    block_to_opt_obj_name = 'my_ddpg' 
    
    #read filenames in folder results:
    filenames = next(os.walk(dir_path), (None, None, []))[2]
    
    #population of tuned agents:
    agents_pop = []
    gens_order = []
    for tmp_file in filenames:
        if((block_to_opt_obj_name in tmp_file) and ('.pkl' in tmp_file)):
            if(('_result_' not in tmp_file) and ('_new_best_' not in tmp_file) and ('best_agent' not in tmp_file)):
                new_agent = load(os.path.join(dir_path, str(tmp_file))) 
                agents_pop.append(new_agent)
                for i in range(50):
                    if('Gen_'+str(i)+'_' in tmp_file):
                        gens_order.append(i)
    
    best_agent = None
    for tmp_file in filenames:
        if(tmp_file == 'my_best_agent.pkl'):
            best_agent = load(os.path.join(dir_path, tmp_file))
            break
        
    tmp_params = best_agent.get_params()
    
    params = {}
    #remove categorical hyperparameters and only keep mutated hyperparameters:
    for tmp_key in list(tmp_params.keys()):
        if((not isinstance(tmp_params[tmp_key], Categorical)) and (tmp_params[tmp_key].to_mutate)):
            params.update({tmp_key: tmp_params[tmp_key]})
            
        if(isinstance(tmp_params[tmp_key], Categorical) and (tmp_params[tmp_key].to_mutate)):
            if(isinstance(tmp_params[tmp_key].current_actual_value, int) or 
               isinstance(tmp_params[tmp_key].current_actual_value, float)):
                new_parameter = Real(hp_name=tmp_key, current_actual_value=tmp_params[tmp_key].current_actual_value, 
                                     obj_name=tmp_key+'_artificial_param', 
                                     range_of_values=[tmp_params[tmp_key].possible_values[0], 
                                                      tmp_params[tmp_key].possible_values[-1]], 
                                     to_mutate=True)
                params.update({tmp_key: new_parameter})
            elif(hasattr(tmp_params[tmp_key].current_actual_value,'dtype') and 
                 tmp_params[tmp_key].current_actual_value.dtype in ['float64','int64']):
                new_parameter = Real(hp_name=tmp_key, current_actual_value=tmp_params[tmp_key].current_actual_value, 
                                     obj_name=tmp_key+'_artificial_param', 
                                     range_of_values=[tmp_params[tmp_key].possible_values[0], 
                                                      tmp_params[tmp_key].possible_values[-1]], 
                                     to_mutate=True)
                params.update({tmp_key: new_parameter})
    
    #create dataset containing the values of the hyperparameters of the tuned agents:
    agents_dataset_params = []
    for agent in agents_pop:
        tmp_params_agent = agent.get_params()
        tmp_params_values_agent = {}
        
        tmp_effective_params = []
    
        for tmp_key in list(tmp_params_agent.keys()):
            if((not isinstance(tmp_params_agent[tmp_key], Categorical)) and (tmp_params_agent[tmp_key].to_mutate)):
                tmp_params_values_agent.update({tmp_key: tmp_params_agent[tmp_key].current_actual_value})
                tmp_effective_params.append({tmp_key: tmp_params_agent[tmp_key].current_actual_value})
            if(isinstance(tmp_params[tmp_key], Categorical) and (tmp_params[tmp_key].to_mutate)):
                if(isinstance(tmp_params[tmp_key].current_actual_value,int) or 
                   isinstance(tmp_params[tmp_key].current_actual_value,float)):
                    tmp_params_values_agent.update({tmp_key: tmp_params_agent[tmp_key].current_actual_value})
                    tmp_effective_params.append({tmp_key: tmp_params_agent[tmp_key].current_actual_value})
                elif(hasattr(tmp_params[tmp_key].current_actual_value,'dtype') and 
                     tmp_params[tmp_key].current_actual_value.dtype in ['float64','int64']):
                     tmp_params_values_agent.update({tmp_key: tmp_params_agent[tmp_key].current_actual_value})
                     tmp_effective_params.append({tmp_key: tmp_params_agent[tmp_key].current_actual_value})
                
        agents_dataset_params.append(tmp_effective_params)    
    
    ordered_dataset = None
    for i in range(50):
        new_gen = []
        
        lst = [gens_order[k]==i for k in range(len(gens_order))]
        new_gen = np.array(agents_dataset_params)[np.array(lst)].tolist()
            
        if(ordered_dataset is None):
            ordered_dataset = new_gen 
        else:
            ordered_dataset = ordered_dataset + new_gen
    
    #DDPG+HalfCheetah
    hp_names = {0:'actor_lr', 1:'critic_lr', 2:'batch_size', 
                3:'initial_replay_size', 4:'max_replay_size', 5:'n_epochs', 
                6:'n_steps', 7:'n_steps_per_fit'}  
       
    print(best_agent.obj_name) 
    best_ag_gen = 18     
    
    for hp_val in range(len(hp_names.keys())):
       
        one_hp = []
        for i in range(len(ordered_dataset)):
            one_hp.append(list(ordered_dataset[i][hp_val].values())[0])
        
        plt.plot(dpi=600)
        diff = 0
        for i in range(50):    
            unique, counts = np.unique(one_hp[20*i:20*(i+1)], return_counts=True)
            y = unique
            x = np.ones(len(y))
            
            fct = (1+np.log(counts))
            fct = counts
            
            plt.scatter(x+diff-1,y,s=3*fct, color='#ff7f0e')
            diff += 1
               
        plt.axhline(y=tmp_params[hp_names[hp_val]].current_actual_value, color='red', linestyle='-', linewidth=1)
            
        if(hp_names[hp_val] in ['actor_lr', 'critic_lr', 'lr_alpha', 'n_steps_per_fit', 'n_episodes_per_fit']):
            plt.yscale('log')
            
        plt.show()        